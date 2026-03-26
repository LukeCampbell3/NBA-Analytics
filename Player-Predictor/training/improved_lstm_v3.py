#!/usr/bin/env python3
"""
Improved LSTM v3 — Anti-overfitting focus.

v2 results: avg MAE 4.81, R²~0.21 (positive!), but heavy overfitting after epoch 15.
v3 strategy: trade some training fit for better generalization.

Key changes from v2:
1. Single-layer LSTM (not bidirectional) — less capacity, less overfitting
2. Gaussian noise on inputs — data augmentation effect
3. Higher dropout (0.35) + L2 regularization
4. Larger batch size (128) for smoother gradients
5. Lower max LR (5e-4) with longer warmup
6. SWA-like weight averaging via longer patience + restore_best
7. Simpler attention (dot-product, not learned MLP)
8. Skip connection from baseline directly to output
"""

import sys
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Concatenate,
    LayerNormalization, Lambda, Add, GaussianNoise
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

from unified_moe_trainer import UnifiedMoETrainer


# ============================================================================
# SIMPLE DOT-PRODUCT ATTENTION
# ============================================================================

class SimpleAttention(tf.keras.layers.Layer):
    """Lightweight dot-product attention — fewer params than MLP attention."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.query = self.add_weight("query", shape=(feat_dim,),
                                     initializer="glorot_uniform")

    def call(self, x):
        # x: [B, T, D], query: [D]
        scores = tf.reduce_sum(x * self.query, axis=-1, keepdims=True)  # [B, T, 1]
        alpha = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(x * alpha, axis=1)  # [B, D]
        return context


# ============================================================================
# LOSS: Huber + R² penalty (simpler than v2)
# ============================================================================

def robust_delta_loss(y_true, y_pred):
    """Huber loss + mild R² penalty. Simpler = more stable."""
    huber = tf.keras.losses.huber(y_true, y_pred, delta=1.5)  # wider delta = less aggressive

    # R² penalty only when negative (don't over-optimize R² at expense of MAE)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    r2_penalty = tf.reduce_mean(tf.nn.relu(-r2))

    return tf.reduce_mean(huber) + 0.3 * r2_penalty


# ============================================================================
# COSINE ANNEALING WITH WARMUP
# ============================================================================

class CosineAnnealingWarmup(Callback):
    def __init__(self, warmup_epochs=8, max_lr=5e-4, min_lr=1e-6, total_epochs=200):
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


# ============================================================================
# R² MONITOR
# ============================================================================

class R2MonitorCallback(Callback):
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            delta_pred = self.model.predict([self.X_val, self.b_val], verbose=0)
            preds_scaled = self.b_val + delta_pred
            preds_orig = self.scaler_y.inverse_transform(preds_scaled)

            total_mae = 0
            parts = []
            for i, t in enumerate(self.target_names):
                mae = mean_absolute_error(self.y_val_raw[:, i], preds_orig[:, i])
                r2 = r2_score(self.y_val_raw[:, i], preds_orig[:, i])
                total_mae += mae
                parts.append(f"{t}: MAE={mae:.3f} R²={r2:.3f}")

            avg_mae = total_mae / len(self.target_names)
            if avg_mae < self.best_avg_mae:
                self.best_avg_mae = avg_mae
                self.best_epoch = epoch + 1

            print(f"\n  [Epoch {epoch+1}] {' | '.join(parts)} | Avg={avg_mae:.3f} (best={self.best_avg_mae:.3f}@{self.best_epoch})")


# ============================================================================
# MODEL BUILDER
# ============================================================================

def build_lstm_v3(seq_len, n_features, n_targets, seed=42, config=None):
    """
    Simpler LSTM with strong regularization.
    Key: single LSTM layer, gaussian noise, L2, high dropout.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    cfg = config or {}
    lstm_units = cfg.get("lstm_units", 80)
    dense_units = cfg.get("dense_units", 64)
    head_units = cfg.get("head_units", 24)
    drop = cfg.get("dropout", 0.35)
    rec_drop = cfg.get("recurrent_dropout", 0.2)
    noise_std = cfg.get("noise_std", 0.05)
    l2_reg = cfg.get("l2_reg", 1e-4)

    seq_input = Input(shape=(seq_len, n_features), name="seq_input")
    base_input = Input(shape=(n_targets,), name="base_input")

    # Gaussian noise on input — acts as data augmentation
    x = GaussianNoise(noise_std, name="input_noise")(seq_input)

    # Single LSTM layer (not bidirectional — less capacity, less overfitting)
    x = LSTM(
        lstm_units,
        return_sequences=True,
        dropout=drop,
        recurrent_dropout=rec_drop,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="lstm"
    )(x)

    # Simple attention
    context = SimpleAttention(name="attention")(x)

    # Last hidden state
    last_hidden = Lambda(lambda z: z[:, -1, :], name="last_hidden")(x)

    # Merge
    merged = Concatenate(name="merge")([context, last_hidden, base_input])
    merged = LayerNormalization(name="ln_merge")(merged)

    # Dense block with residual
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
                  name=f"lstm_v3_s{seed}")
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_single(X_train, b_train, delta_train, X_val, b_val, delta_val,
                 seq_len, n_features, n_targets, seed=42, config=None,
                 scaler_y=None, y_val_raw=None, target_names=None):
    cfg = config or {}
    epochs = cfg.get("epochs", 200)
    batch_size = cfg.get("batch_size", 128)
    max_lr = cfg.get("max_lr", 5e-4)

    model = build_lstm_v3(seq_len, n_features, n_targets, seed=seed, config=cfg)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=max_lr, clipnorm=0.5),
        loss=robust_delta_loss,
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        CosineAnnealingWarmup(warmup_epochs=8, max_lr=max_lr, min_lr=1e-6, total_epochs=epochs),
    ]

    if scaler_y is not None and y_val_raw is not None:
        callbacks.append(R2MonitorCallback(
            X_val, b_val, scaler_y, y_val_raw,
            target_names or ["PTS", "TRB", "AST"]
        ))

    history = model.fit(
        [X_train, b_train], delta_train,
        validation_data=([X_val, b_val], delta_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    best_val = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val) + 1
    print(f"  Best val_loss: {best_val:.5f} at epoch {best_epoch}")
    return model, history, best_val


def train_ensemble(X_train, b_train, delta_train, X_val, b_val, delta_val,
                   seq_len, n_features, n_targets, n_models=5, config=None,
                   scaler_y=None, y_val_raw=None, target_names=None):
    models = []
    val_losses = []

    for i in range(n_models):
        seed = 42 + i * 17
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE MEMBER {i+1}/{n_models}  (seed={seed})")
        print(f"{'='*60}")

        # Less architectural diversity than v2 — keep it simple
        cfg = dict(config or {})
        variants = [
            {"lstm_units": 80, "dense_units": 64, "dropout": 0.35},
            {"lstm_units": 96, "dense_units": 56, "dropout": 0.30},
            {"lstm_units": 72, "dense_units": 72, "dropout": 0.38},
            {"lstm_units": 88, "dense_units": 64, "dropout": 0.33},
            {"lstm_units": 80, "dense_units": 80, "dropout": 0.36},
        ]
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
    losses = np.array(val_losses)
    weights = 1.0 / (losses + 1e-8)
    weights /= weights.sum()
    print(f"  Ensemble weights: {[f'{w:.3f}' for w in weights]}")

    preds = [m.predict([X, baselines], verbose=0, batch_size=256) for m in models]
    return sum(p * w for p, w in zip(preds, weights))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("IMPROVED LSTM v3 — Anti-Overfitting + Stronger Regularization")
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
        "epochs": 200,
        "batch_size": 128,
        "max_lr": 5e-4,
        "lstm_units": 80,
        "dense_units": 64,
        "head_units": 24,
        "dropout": 0.35,
        "recurrent_dropout": 0.2,
        "noise_std": 0.05,
        "l2_reg": 1e-4,
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
        results[target] = {"mae": mae, "r2": r2, "baseline_mae": baseline_mae}
        print(f"\n{target}:")
        print(f"  Ensemble MAE: {mae:.4f}")
        print(f"  Ensemble R²:  {r2:.4f}")
        print(f"  Baseline MAE: {baseline_mae:.4f}")
        print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%)")

    avg_mae = total_mae / len(trainer.target_columns)
    print(f"\n{'='*60}")
    print(f"  OVERALL AVG MAE: {avg_mae:.4f}")
    print(f"  v2 result:       4.811")
    print(f"  v2 top-3:        4.780")
    print(f"  Old baselines:   4.78 (simple LSTM), 4.795 (MoE)")
    print(f"{'='*60}")

    # Top-3 ensemble
    print("\n--- Top-3 ensemble ---")
    sorted_idx = np.argsort(val_losses)[:3]
    top_preds = [models[i].predict([X_val, b_val], verbose=0, batch_size=256) for i in sorted_idx]
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
        m.save_weights(f"model/lstm_v3_member_{i}.weights.h5")
    joblib.dump(trainer.scaler_x, 'model/lstm_v3_scaler_x.pkl')
    joblib.dump(trainer.scaler_y, 'model/lstm_v3_scaler_y.pkl')

    meta = {
        'model_type': 'lstm_v3_ensemble',
        'n_models': N_MODELS,
        'seq_len': seq_len,
        'n_features': n_features,
        'n_targets': n_targets,
        'target_columns': trainer.target_columns,
        'baseline_features': trainer.baseline_features,
        'feature_columns': trainer.feature_columns,
        'seeds': [42 + i * 17 for i in range(N_MODELS)],
        'val_losses': [float(v) for v in val_losses],
        'config': config,
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'avg_mae': float(avg_mae),
    }
    with open('model/lstm_v3_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to model/")


if __name__ == "__main__":
    main()
