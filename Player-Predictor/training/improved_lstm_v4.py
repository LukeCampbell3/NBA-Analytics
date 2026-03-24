#!/usr/bin/env python3
"""
IMPROVED LSTM v4 — Hybrid: v2 architecture + v3 regularization + new tricks

Key changes from v2/v3:
1. Bidirectional LSTM (from v2) — captures forward+backward context
2. GaussianNoise + SpatialDropout1D (from v3 + new) — input regularization
3. L2 regularization on LSTM kernels (from v3)
4. Moderate dropout (0.30) — between v2's 0.25 and v3's 0.35
5. Label smoothing via target noise during training
6. Stochastic Weight Averaging (SWA) — average weights from last N epochs
7. Mixup augmentation on sequences
8. Gradient accumulation via larger effective batch (batch=96)
9. Single bidirectional LSTM layer (not 2) — less capacity than v2, more than v3
10. Learned attention (from v2) — more expressive than v3's dot-product
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Concatenate,
    Lambda, Add, LayerNormalization, GaussianNoise, SpatialDropout1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from unified_moe_trainer import UnifiedMoETrainer


class TemporalAttention(tf.keras.layers.Layer):
    """Learned MLP attention — which timesteps matter most."""
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W = self.add_weight("attn_W", shape=(feat_dim, self.units),
                                 initializer="glorot_uniform")
        self.b = self.add_weight("attn_b", shape=(self.units,),
                                 initializer="zeros")
        self.v = self.add_weight("attn_v", shape=(self.units, 1),
                                 initializer="glorot_uniform")

    def call(self, x):
        score = tf.tanh(tf.matmul(x, self.W) + self.b)
        score = tf.matmul(score, self.v)
        alpha = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(x * alpha, axis=1)
        return context

    def get_config(self):
        config = super().get_config()
        config["units"] = self.units
        return config


def composite_loss(y_true, y_pred):
    """Huber + MSE + R² penalty — same proven loss from v2."""
    huber = tf.keras.losses.huber(y_true, y_pred, delta=1.5)
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    # R² penalty
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    r2_penalty = tf.reduce_mean(tf.maximum(0.0, -r2)) * 0.1
    return 0.6 * huber + 0.3 * mse + 0.1 * r2_penalty


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
    """Stochastic Weight Averaging — average weights from last N epochs for better generalization."""
    def __init__(self, swa_start=30, swa_freq=5):
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


def build_lstm_v4(seq_len, n_features, n_targets, seed=42, config=None):
    """
    Hybrid architecture: v2's expressiveness + v3's regularization.
    - Single Bidirectional LSTM (not 2 layers — sweet spot)
    - Learned MLP attention
    - GaussianNoise + SpatialDropout1D on input
    - L2 on LSTM + dense kernels
    - Per-target heads with dropout
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

    # Input regularization: noise + spatial dropout
    x = GaussianNoise(noise_std, name="input_noise")(seq_input)
    x = SpatialDropout1D(0.1, name="spatial_drop")(x)

    # Single Bidirectional LSTM — sweet spot between v2 (2 layers) and v3 (unidirectional)
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True,
             dropout=drop, recurrent_dropout=rec_drop,
             kernel_regularizer=regularizers.l2(l2_reg)),
        name="bilstm"
    )(x)

    # Learned temporal attention
    context = TemporalAttention(units=attn_units, name="temporal_attn")(x)

    # Last hidden state
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

    # Per-target heads
    target_preds = []
    for i in range(n_targets):
        ti = Dense(head_units, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_reg),
                   name=f"head_{i}_fc")(h)
        ti = Dropout(drop * 0.3, name=f"head_{i}_drop")(ti)
        ti = Dense(1, activation=None, name=f"head_{i}_out")(ti)
        target_preds.append(ti)

    delta_pred = Concatenate(name="delta_pred")(target_preds)

    model = Model(inputs=[seq_input, base_input], outputs=delta_pred,
                  name=f"lstm_v4_s{seed}")
    return model


def mixup_batch(X, b, delta, alpha=0.2):
    """Mixup augmentation — interpolate between random pairs of samples."""
    batch_size = len(X)
    indices = np.random.permutation(batch_size)
    lam = np.random.beta(alpha, alpha, size=(batch_size, 1))
    lam_seq = lam[:, :, np.newaxis]  # for sequence data [B, 1, 1]

    X_mix = lam_seq * X + (1 - lam_seq) * X[indices]
    b_mix = lam * b + (1 - lam) * b[indices]
    d_mix = lam * delta + (1 - lam) * delta[indices]
    return X_mix, b_mix, d_mix


class MixupGenerator(tf.keras.utils.Sequence):
    """Data generator with mixup augmentation."""
    def __init__(self, X, b, delta, batch_size=96, mixup_alpha=0.2, target_noise=0.02):
        self.X = X
        self.b = b
        self.delta = delta
        self.batch_size = batch_size
        self.mixup_alpha = mixup_alpha
        self.target_noise = target_noise
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        b_batch = self.b[batch_idx].copy()
        d_batch = self.delta[batch_idx].copy()

        # Apply mixup with 50% probability
        if np.random.random() < 0.5 and len(X_batch) > 1:
            X_batch, b_batch, d_batch = mixup_batch(X_batch, b_batch, d_batch, self.mixup_alpha)

        # Label smoothing — small noise on targets
        if self.target_noise > 0:
            d_batch = d_batch + np.random.normal(0, self.target_noise, d_batch.shape)

        return [X_batch, b_batch], d_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def train_single(X_train, b_train, delta_train, X_val, b_val, delta_val,
                 seq_len, n_features, n_targets, seed=42, config=None,
                 scaler_y=None, y_val_raw=None, target_names=None):
    """Train a single v4 model with mixup + SWA."""
    cfg = config or {}
    epochs = cfg.get("epochs", 150)
    batch_size = cfg.get("batch_size", 96)
    max_lr = cfg.get("max_lr", 7e-4)

    model = build_lstm_v4(seq_len, n_features, n_targets, seed=seed, config=cfg)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=max_lr, clipnorm=0.8),
        loss=composite_loss,
        metrics=['mae']
    )

    # Mixup data generator
    train_gen = MixupGenerator(
        X_train, b_train, delta_train,
        batch_size=batch_size,
        mixup_alpha=0.15,
        target_noise=0.015
    )

    swa_cb = SWACallback(swa_start=35, swa_freq=5)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
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

    # Try SWA weights — use if they improve val_loss
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


def train_ensemble(X_train, b_train, delta_train, X_val, b_val, delta_val,
                   seq_len, n_features, n_targets, n_models=5, config=None,
                   scaler_y=None, y_val_raw=None, target_names=None):
    """Train N models with architectural diversity."""
    models = []
    val_losses = []

    # Architectural variants — moderate diversity
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
            X_train, b_train, delta_train,
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


def main():
    print("=" * 80)
    print("IMPROVED LSTM v4 — Hybrid: v2 Architecture + v3 Regularization + Mixup + SWA")
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
        X_train, b_train, delta_train,
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
        dp = m.predict([X_val, b_val], verbose=0)
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
    print(f"  v2 weighted:     4.811  (R² ~0.21)")
    print(f"  v2 top-3:        4.780")
    print(f"  v3 best member:  4.857  (R² ~0.13)")
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
        m.save_weights(f"model/lstm_v4_member_{i}.weights.h5")
    joblib.dump(trainer.scaler_x, 'model/lstm_v4_scaler_x.pkl')
    joblib.dump(trainer.scaler_y, 'model/lstm_v4_scaler_y.pkl')

    meta = {
        'model_type': 'lstm_v4_ensemble',
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
    with open('model/lstm_v4_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to model/")


if __name__ == "__main__":
    main()
