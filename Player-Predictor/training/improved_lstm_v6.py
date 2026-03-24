#!/usr/bin/env python3
"""
IMPROVED LSTM v6 — PTS-Focused Asymmetric Architecture

Key changes over v4 (avg MAE 4.709):
1. Asymmetric loss: PTS errors weighted 2x vs TRB/AST
2. PTS head gets more capacity (wider hidden layer)
3. Sample weighting: games with large PTS deviations get higher weight (continuous, not binary)
4. Gradient clipping tightened for stability
5. Slightly longer patience for EarlyStopping (35 vs 30)

What we learned from v5 (failed):
- Dual-path spike blending destabilized training (val_loss oscillated 0.10-0.29)
- Binary spike classification worked but the gating mechanism hurt delta predictions
- The simpler approach: just weight the loss asymmetrically, don't try to predict spikes

Architecture: Same as v4 (proven stable)
- Single Bidirectional LSTM + MLP Temporal Attention
- Residual dense block + LayerNormalization
- Per-target heads (PTS head gets extra capacity)
- 5-model ensemble with architectural diversity
- Mixup augmentation + cosine annealing + SWA
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, Concatenate,
    GaussianNoise, SpatialDropout1D, Lambda, Add, LayerNormalization,
    Multiply, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from unified_moe_trainer import UnifiedMoETrainer


# ============================================================
# TEMPORAL ATTENTION (same as v4)
# ============================================================

class TemporalAttention(Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W = self.add_weight("attn_W", (feat_dim, self.units), initializer="glorot_uniform")
        self.b = self.add_weight("attn_b", (self.units,), initializer="zeros")
        self.v = self.add_weight("attn_v", (self.units, 1), initializer="glorot_uniform")

    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        alpha = tf.nn.softmax(tf.matmul(score, self.v), axis=1)
        return tf.reduce_sum(x * alpha, axis=1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# ============================================================
# PTS-WEIGHTED ASYMMETRIC LOSS
# ============================================================

def pts_weighted_loss(y_true, y_pred):
    """
    Asymmetric composite loss that weights PTS errors more heavily.
    
    v4 loss: 0.6*Huber + 0.3*MSE + 0.1*R²_penalty (equal weight per target)
    v6 loss: Same structure but PTS column gets 2x weight in MSE component,
             plus an additional penalty for large PTS errors.
    
    This directly addresses PTS MAE being the weakest link (8.94 vs TRB 2.5, AST 2.7).
    """
    # Standard Huber (robust to outliers)
    huber = tf.keras.losses.huber(y_true, y_pred, delta=1.5)
    
    # Per-target MSE with PTS weighting
    sq_err = tf.square(y_true - y_pred)  # [batch, 3]
    # Weight: PTS=2.0, TRB=1.0, AST=1.0 (normalized so mean is ~1.33)
    target_weights = tf.constant([2.0, 1.0, 1.0], dtype=tf.float32)
    weighted_sq = sq_err * target_weights
    mse = tf.reduce_mean(weighted_sq, axis=-1)
    
    # R² penalty (same as v4)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    r2_penalty = tf.reduce_mean(tf.maximum(0.0, -r2)) * 0.1
    
    # PTS large-error penalty: extra quadratic cost when PTS error > 1 std
    pts_err = tf.abs(y_true[:, 0] - y_pred[:, 0])
    pts_large_penalty = tf.reduce_mean(tf.maximum(0.0, pts_err - 1.0) ** 2) * 0.05
    
    return 0.55 * huber + 0.30 * mse + 0.10 * r2_penalty + 0.05 * pts_large_penalty


# ============================================================
# CALLBACKS (same as v4 with minor tuning)
# ============================================================

class CosineAnnealingWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=5, max_lr=7e-4, min_lr=1e-6, total_epochs=150):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)


class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, swa_start=35, swa_freq=5):
        super().__init__()
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.swa_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            current_weights = self.model.get_weights()
            if self.swa_weights is None:
                self.swa_weights = [w.copy() for w in current_weights]
            else:
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] = (self.swa_weights[i] * self.swa_count + current_weights[i]) / (self.swa_count + 1)
            self.swa_count += 1

    def apply_swa(self):
        if self.swa_weights is not None:
            self.model.set_weights(self.swa_weights)
            return True
        return False


class R2MonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, b_val, scaler_y, y_val_raw, target_names):
        super().__init__()
        self.X_val = X_val
        self.b_val = b_val
        self.scaler_y = scaler_y
        self.y_val_raw = y_val_raw
        self.target_names = target_names
        self.best_avg_mae = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            dp = self.model.predict([self.X_val, self.b_val], verbose=0, batch_size=512)
            ps = self.b_val + dp
            po = self.scaler_y.inverse_transform(ps)
            parts = []
            total_mae = 0
            for i, name in enumerate(self.target_names):
                mae = mean_absolute_error(self.y_val_raw[:, i], po[:, i])
                r2 = r2_score(self.y_val_raw[:, i], po[:, i])
                parts.append(f"{name}: MAE={mae:.3f} R²={r2:.3f}")
                total_mae += mae
            avg_mae = total_mae / len(self.target_names)
            if avg_mae < self.best_avg_mae:
                self.best_avg_mae = avg_mae
                self.best_epoch = epoch + 1
            print(f"\n  [Epoch {epoch+1}] {' | '.join(parts)} | Avg={avg_mae:.3f} (best={self.best_avg_mae:.3f}@{self.best_epoch})")


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def build_lstm_v6(seq_len, n_features, n_targets, seed=42, config=None):
    """
    v6 architecture — v4 backbone with PTS-enhanced head.
    
    Changes from v4:
    - PTS head (index 0) gets wider hidden layer (head_units + 12)
    - PTS head gets less dropout (allows more expressiveness for PTS)
    - Everything else identical to v4 for stability
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    cfg = config or {}
    lstm_units = cfg.get("lstm_units", 96)
    attn_units = cfg.get("attn_units", 32)
    dense_units = cfg.get("dense_units", 80)
    head_units = cfg.get("head_units", 28)
    drop = cfg.get("dropout", 0.30)
    rec_drop = cfg.get("recurrent_dropout", 0.15)
    noise_std = cfg.get("noise_std", 0.03)
    l2_reg = cfg.get("l2_reg", 5e-5)

    seq_input = Input(shape=(seq_len, n_features), name="seq_input")
    base_input = Input(shape=(n_targets,), name="base_input")

    # Input regularization
    x = GaussianNoise(noise_std, name="input_noise")(seq_input)
    x = SpatialDropout1D(0.1, name="spatial_drop")(x)

    # Bidirectional LSTM backbone
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True,
             dropout=drop, recurrent_dropout=rec_drop,
             kernel_regularizer=regularizers.l2(l2_reg)),
        name="bilstm"
    )(x)

    # Temporal attention
    context = TemporalAttention(units=attn_units, name="temporal_attn")(x)
    last_hidden = Lambda(lambda z: z[:, -1, :], name="last_hidden")(x)

    # Merge
    merged = Concatenate(name="merge")([context, last_hidden, base_input])
    merged = LayerNormalization(name="ln_merge")(merged)

    # Residual dense block
    h = Dense(dense_units, activation="relu",
              kernel_regularizer=regularizers.l2(l2_reg), name="dense1")(merged)
    h = Dropout(drop, name="drop1")(h)
    h2 = Dense(dense_units, activation="relu",
               kernel_regularizer=regularizers.l2(l2_reg), name="dense2")(h)
    h2 = Dropout(drop * 0.5, name="drop2")(h2)
    h = Add(name="residual")([h, h2])
    h = LayerNormalization(name="ln_res")(h)

    # Per-target heads — PTS gets extra capacity
    target_preds = []
    for i in range(n_targets):
        if i == 0:  # PTS head — wider + less dropout
            ti = Dense(head_units + 12, activation="relu",
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name=f"head_{i}_fc")(h)
            ti = Dropout(drop * 0.2, name=f"head_{i}_drop")(ti)
            ti = Dense(1, activation=None, name=f"head_{i}_out")(ti)
        else:  # TRB, AST heads — same as v4
            ti = Dense(head_units, activation="relu",
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name=f"head_{i}_fc")(h)
            ti = Dropout(drop * 0.3, name=f"head_{i}_drop")(ti)
            ti = Dense(1, activation=None, name=f"head_{i}_out")(ti)
        target_preds.append(ti)

    delta_pred = Concatenate(name="delta_pred")(target_preds)

    model = Model(inputs=[seq_input, base_input], outputs=delta_pred,
                  name=f"lstm_v6_s{seed}")
    return model


# ============================================================
# MIXUP GENERATOR WITH SAMPLE WEIGHTING
# ============================================================

class WeightedMixupGenerator(tf.keras.utils.Sequence):
    """
    Mixup generator that also applies sample weights based on PTS deviation.
    
    Games with large PTS deviations from baseline get higher weight,
    encouraging the model to learn from high-variance games.
    """
    def __init__(self, X, b, delta, sample_weights, batch_size=96,
                 mixup_alpha=0.15, target_noise=0.015):
        self.X = X
        self.b = b
        self.delta = delta
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.mixup_alpha = mixup_alpha
        self.target_noise = target_noise
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_b = self.X[batch_idx].copy()
        b_b = self.b[batch_idx].copy()
        d_b = self.delta[batch_idx].copy()
        w_b = self.sample_weights[batch_idx].copy()

        # Mixup
        bs = len(X_b)
        mix_idx = np.random.permutation(bs)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=(bs, 1))
        lam_seq = lam[:, :, np.newaxis]

        X_b = lam_seq * X_b + (1 - lam_seq) * X_b[mix_idx]
        b_b = lam * b_b + (1 - lam) * b_b[mix_idx]
        d_b = lam * d_b + (1 - lam) * d_b[mix_idx]
        w_b = np.maximum(w_b, w_b[mix_idx])  # keep higher weight

        # Target noise (less noise on high-weight samples)
        noise_scale = self.target_noise / np.maximum(w_b, 1.0)
        d_b += np.random.normal(0, 1.0, d_b.shape) * noise_scale[:, np.newaxis]

        return ([X_b, b_b], d_b, w_b.flatten())

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def compute_sample_weights(delta_scaled, scaler_y, pts_threshold=6.0):
    """
    Continuous sample weighting based on PTS deviation magnitude.
    
    Unlike v5's binary spike labels, this gives a smooth weight:
    - Normal games: weight = 1.0
    - Games with PTS deviation > threshold: weight scales up smoothly
    - Max weight capped at 3.0 (less aggressive than v5's 5.0)
    
    This avoids the instability of binary spike classification while still
    giving the model more signal from high-variance games.
    """
    pts_scale = scaler_y.scale_[0]  # PTS standard deviation
    pts_thresh_scaled = pts_threshold / pts_scale
    
    pts_abs_delta = np.abs(delta_scaled[:, 0])
    
    # Smooth ramp: weight = 1 + clamp((|delta| - threshold) / threshold, 0, 2)
    excess = np.maximum(0, pts_abs_delta - pts_thresh_scaled) / pts_thresh_scaled
    weights = 1.0 + np.minimum(excess, 2.0)  # range [1.0, 3.0]
    
    n_upweighted = np.sum(weights > 1.01)
    print(f"  Sample weighting: {n_upweighted}/{len(weights)} games upweighted ({100*n_upweighted/len(weights):.1f}%)")
    print(f"    Weight range: [{weights.min():.2f}, {weights.max():.2f}], mean={weights.mean():.2f}")
    print(f"    PTS threshold: ±{pts_threshold} raw points (±{pts_thresh_scaled:.3f} scaled)")
    
    return weights.astype(np.float32)


# ============================================================
# TRAINING
# ============================================================

def train_single(X_train, b_train, delta_train, weights_train,
                 X_val, b_val, delta_val,
                 seq_len, n_features, n_targets, seed=42, config=None,
                 scaler_y=None, y_val_raw=None, target_names=None):
    """Train a single v6 model."""
    cfg = config or {}
    epochs = cfg.get("epochs", 150)
    batch_size = cfg.get("batch_size", 96)
    max_lr = cfg.get("max_lr", 7e-4)

    model = build_lstm_v6(seq_len, n_features, n_targets, seed=seed, config=cfg)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=max_lr, clipnorm=0.7),
        loss=pts_weighted_loss,
        metrics=['mae']
    )

    train_gen = WeightedMixupGenerator(
        X_train, b_train, delta_train, weights_train,
        batch_size=batch_size,
        mixup_alpha=0.15,
        target_noise=0.015
    )

    swa_cb = SWACallback(swa_start=35, swa_freq=5)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1),
        CosineAnnealingWarmup(warmup_epochs=6, max_lr=max_lr, min_lr=1e-6, total_epochs=epochs),
        swa_cb,
    ]

    if scaler_y is not None and y_val_raw is not None:
        callbacks.append(R2MonitorCallback(
            X_val, b_val, scaler_y, y_val_raw,
            target_names or ["PTS", "TRB", "AST"]
        ))

    history = model.fit(
        train_gen,
        validation_data=([X_val, b_val], delta_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    best_val = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val) + 1
    print(f"  Best val_loss: {best_val:.5f} at epoch {best_epoch}")

    # Try SWA weights
    es_val = best_val
    es_weights = model.get_weights()

    if swa_cb.apply_swa():
        swa_val = model.evaluate([X_val, b_val], delta_val, verbose=0)[0]
        print(f"  SWA val_loss: {swa_val:.5f} vs ES val_loss: {es_val:.5f}")
        if swa_val < es_val:
            print(f"  -> Using SWA weights (better by {es_val - swa_val:.5f})")
            best_val = swa_val
        else:
            print(f"  -> Keeping EarlyStopping weights")
            model.set_weights(es_weights)

    return model, history, best_val


def train_ensemble(X_train, b_train, delta_train, weights_train,
                   X_val, b_val, delta_val,
                   seq_len, n_features, n_targets, n_models=5, config=None,
                   scaler_y=None, y_val_raw=None, target_names=None):
    """Train N models with architectural diversity."""
    models = []
    val_losses = []

    # Same variants as v4 for fair comparison
    variants = [
        {"lstm_units": 96,  "dense_units": 80,  "attn_units": 32, "dropout": 0.30, "noise_std": 0.03},
        {"lstm_units": 112, "dense_units": 72,  "attn_units": 28, "dropout": 0.25, "noise_std": 0.04},
        {"lstm_units": 80,  "dense_units": 96,  "attn_units": 36, "dropout": 0.32, "noise_std": 0.025},
        {"lstm_units": 104, "dense_units": 80,  "attn_units": 32, "dropout": 0.28, "noise_std": 0.035},
        {"lstm_units": 88,  "dense_units": 88,  "attn_units": 30, "dropout": 0.30, "noise_std": 0.03},
    ]

    for i in range(n_models):
        seed = 42 + i * 19
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE MEMBER {i+1}/{n_models}  (seed={seed})")
        print(f"{'='*60}")

        cfg = dict(config or {})
        cfg.update(variants[i % len(variants)])

        model, history, best_val = train_single(
            X_train, b_train, delta_train, weights_train,
            X_val, b_val, delta_val,
            seq_len, n_features, n_targets,
            seed=seed, config=cfg,
            scaler_y=scaler_y, y_val_raw=y_val_raw,
            target_names=target_names
        )
        models.append(model)
        val_losses.append(best_val)

    return models, val_losses


def weighted_ensemble_predict(models, val_losses, X, baselines):
    """Inverse-loss weighted ensemble prediction."""
    weights = 1.0 / (np.array(val_losses) + 1e-8)
    weights /= weights.sum()
    preds = [m.predict([X, baselines], verbose=0, batch_size=512) for m in models]
    return sum(p * w for p, w in zip(preds, weights))


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("IMPROVED LSTM v6 — PTS-Focused Asymmetric Architecture")
    print("=" * 80)

    trainer = UnifiedMoETrainer()
    X, baselines, y, df = trainer.prepare_data()

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    b_train, b_val = baselines[:split_idx], baselines[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    delta_train = y_train - b_train
    delta_val = y_val - b_val

    y_val_orig = trainer.scaler_y.inverse_transform(y_val)
    b_val_orig = trainer.scaler_y.inverse_transform(b_val)

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    print(f"Delta train std: {np.std(delta_train, axis=0)}")

    # Compute continuous sample weights based on PTS deviation
    print("\n--- Computing sample weights ---")
    weights_train = compute_sample_weights(delta_train, trainer.scaler_y, pts_threshold=6.0)

    n_targets = y_train.shape[1]
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    config = {
        "epochs": 150,
        "batch_size": 96,
        "max_lr": 7e-4,
        "lstm_units": 96,
        "attn_units": 32,
        "dense_units": 80,
        "head_units": 28,
        "dropout": 0.30,
        "recurrent_dropout": 0.15,
        "noise_std": 0.03,
        "l2_reg": 5e-5,
    }

    N_MODELS = 5
    models, val_losses = train_ensemble(
        X_train, b_train, delta_train, weights_train,
        X_val, b_val, delta_val,
        seq_len, n_features, n_targets,
        n_models=N_MODELS, config=config,
        scaler_y=trainer.scaler_y,
        y_val_raw=y_val_orig,
        target_names=trainer.target_columns
    )

    # ---- Evaluate ----
    print("\n" + "=" * 80)
    print("INDIVIDUAL MEMBERS")
    print("=" * 80)
    for i, m in enumerate(models):
        dp = m.predict([X_val, b_val], verbose=0, batch_size=512)
        ps = b_val + dp
        po = trainer.scaler_y.inverse_transform(ps)
        maes = [mean_absolute_error(y_val_orig[:, j], po[:, j]) for j in range(n_targets)]
        r2s = [r2_score(y_val_orig[:, j], po[:, j]) for j in range(n_targets)]
        print(f"  Member {i+1}: PTS={maes[0]:.3f}(R²={r2s[0]:.3f})  "
              f"TRB={maes[1]:.3f}(R²={r2s[1]:.3f})  "
              f"AST={maes[2]:.3f}(R²={r2s[2]:.3f})  avg={np.mean(maes):.3f}")

    # Weighted ensemble
    print("\n" + "=" * 80)
    print("WEIGHTED ENSEMBLE")
    print("=" * 80)
    delta_pred = weighted_ensemble_predict(models, val_losses, X_val, b_val)
    preds_scaled = b_val + delta_pred
    pred_orig = trainer.scaler_y.inverse_transform(preds_scaled)

    total_mae = 0
    results = {}
    for i, target in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], pred_orig[:, i])
        r2 = r2_score(y_val_orig[:, i], pred_orig[:, i])
        baseline_mae = mean_absolute_error(y_val_orig[:, i], b_val_orig[:, i])
        improvement = baseline_mae - mae
        pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
        total_mae += mae
        results[target] = {"mae": float(mae), "r2": float(r2), "baseline_mae": float(baseline_mae)}
        print(f"\n{target}:")
        print(f"  Ensemble MAE: {mae:.4f}")
        print(f"  Ensemble R²:  {r2:.4f}")
        print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%)")

    avg_mae = total_mae / len(trainer.target_columns)

    print(f"\n{'='*60}")
    print(f"  OVERALL AVG MAE: {avg_mae:.4f}")
    print(f"  v4 weighted:     4.709  (PTS=8.938 TRB=2.509 AST=2.680)")
    print(f"  v2 weighted:     4.811")
    print(f"{'='*60}")

    # Top-3 ensemble
    print("\n--- Top-3 ensemble ---")
    sorted_idx = np.argsort(val_losses)[:3]
    top_preds = [models[i].predict([X_val, b_val], verbose=0, batch_size=512) for i in sorted_idx]
    top_losses = [val_losses[i] for i in sorted_idx]
    top_weights = 1.0 / (np.array(top_losses) + 1e-8)
    top_weights /= top_weights.sum()
    delta_top = sum(p * w for p, w in zip(top_preds, top_weights))
    po_top = trainer.scaler_y.inverse_transform(b_val + delta_top)
    top_total = 0
    for i, target in enumerate(trainer.target_columns):
        mae = mean_absolute_error(y_val_orig[:, i], po_top[:, i])
        r2 = r2_score(y_val_orig[:, i], po_top[:, i])
        top_total += mae
        print(f"  {target}: MAE={mae:.4f}  R²={r2:.4f}")
    print(f"  Top-3 avg MAE: {top_total / len(trainer.target_columns):.4f}")

    # Save
    os.makedirs("model", exist_ok=True)
    for i, m in enumerate(models):
        m.save_weights(f"model/lstm_v6_member_{i}.weights.h5")
    joblib.dump(trainer.scaler_x, 'model/lstm_v6_scaler_x.pkl')
    joblib.dump(trainer.scaler_y, 'model/lstm_v6_scaler_y.pkl')

    meta = {
        'model_type': 'lstm_v6_pts_weighted_ensemble',
        'n_models': N_MODELS,
        'seq_len': seq_len,
        'n_features': n_features,
        'n_targets': n_targets,
        'target_columns': trainer.target_columns,
        'baseline_features': trainer.baseline_features,
        'feature_columns': trainer.feature_columns,
        'seeds': [42 + i * 19 for i in range(N_MODELS)],
        'val_losses': [float(v) for v in val_losses],
        'config': config,
        'results': results,
        'avg_mae': float(avg_mae),
    }
    with open('model/lstm_v6_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to model/")


if __name__ == "__main__":
    main()
