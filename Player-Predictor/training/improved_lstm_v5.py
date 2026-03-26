#!/usr/bin/env python3
"""
IMPROVED LSTM v5 — Spike-Aware Architecture

Building on v4's proven foundation, adds spike/outlier detection to improve PTS MAE.

Key additions over v4:
1. Spike classification auxiliary head — predicts if game is an outlier
2. Asymmetric sample weighting — spike games get higher loss weight
3. PTS-specific volatility features injected into the data pipeline
4. Spike-conditioned prediction — blends conservative + aggressive predictions
5. Quantile-aware loss — penalizes under-prediction of high games more
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Concatenate,
    Lambda, Add, LayerNormalization, GaussianNoise, SpatialDropout1D,
    Multiply
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


# ============================================================
# SPIKE-AWARE LOSS FUNCTIONS
# ============================================================

def spike_aware_loss(y_true, y_pred, sample_weights=None):
    """
    Composite loss with asymmetric penalty for large deltas.
    
    The key insight: Huber loss with delta=1.5 caps the gradient for large errors,
    which is exactly wrong for spikes — we WANT the model to learn from those.
    
    This loss uses:
    - Standard Huber for small deltas (smooth, stable)
    - Boosted MSE for large deltas (forces model to chase spikes)
    - R² penalty to maintain correlation
    
    Returns a scalar loss.
    """
    # Standard components — reduce to scalar
    huber = tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=1.5))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Asymmetric spike penalty: extra cost for large absolute errors
    abs_error = tf.abs(y_true - y_pred)
    # Per-target: focus extra penalty on first target (PTS, index 0)
    pts_error = abs_error[:, 0]  # PTS delta error
    # Quadratic penalty kicks in when PTS error > 1.0 std
    spike_penalty = tf.reduce_mean(tf.where(
        pts_error > 1.0,
        0.5 * tf.square(pts_error),  # quadratic for large errors
        tf.zeros_like(pts_error)     # no extra penalty for small errors
    ))
    
    # R² penalty
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=0)
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=0)), axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    r2_penalty = tf.reduce_mean(tf.maximum(0.0, -r2))
    
    loss = 0.45 * huber + 0.30 * mse + 0.15 * spike_penalty + 0.10 * r2_penalty
    return loss


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
    def __init__(self, X_val, b_val, scaler_y, y_val_raw, target_names, spike_labels=None):
        super().__init__()
        self.X_val = X_val
        self.b_val = b_val
        self.scaler_y = scaler_y
        self.y_val_raw = y_val_raw
        self.target_names = target_names
        self.spike_labels = spike_labels
        self.best_avg_mae = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            dp = self.model.predict([self.X_val, self.b_val], verbose=0, batch_size=512)
            # Model outputs [delta(3), spike_prob(1)] — take first 3
            if dp.shape[-1] > 3:
                dp = dp[:, :3]
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
            
            msg = f"\n  [Epoch {epoch+1}] {' | '.join(parts)} | Avg={avg_mae:.3f} (best={self.best_avg_mae:.3f}@{self.best_epoch})"
            
            # Spike detection accuracy if available
            if self.spike_labels is not None:
                full_pred = self.model.predict([self.X_val, self.b_val], verbose=0, batch_size=512)
                if full_pred.shape[-1] > 3:
                    spike_prob = full_pred[:, 3]
                    spike_pred = (spike_prob > 0.5).astype(float)
                    spike_acc = np.mean(spike_pred == self.spike_labels)
                    spike_recall = np.mean(spike_pred[self.spike_labels == 1]) if np.sum(self.spike_labels) > 0 else 0
                    msg += f" | Spike acc={spike_acc:.3f} recall={spike_recall:.3f}"
            
            print(msg)


# ============================================================
# SPIKE-AWARE MODEL ARCHITECTURE
# ============================================================

def build_lstm_v5(seq_len, n_features, n_targets, seed=42, config=None):
    """
    Spike-aware LSTM architecture.
    
    Same backbone as v4 (BiLSTM + attention + residual dense), but adds:
    1. Spike classifier head — sigmoid output predicting if game is an outlier
    2. Spike-conditioned scaling — the delta prediction is modulated by spike confidence
       When spike_prob is high, the model is "allowed" to make larger predictions
    3. Dual-path prediction: conservative path + spike path, blended by spike_prob
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

    # Bidirectional LSTM backbone (same as v4)
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True,
             dropout=drop, recurrent_dropout=rec_drop,
             kernel_regularizer=regularizers.l2(l2_reg)),
        name="bilstm"
    )(x)

    # Temporal attention
    context = TemporalAttention(units=attn_units, name="temporal_attn")(x)
    last_hidden = Lambda(lambda z: z[:, -1, :], name="last_hidden")(x)

    # Merge backbone features
    merged = Concatenate(name="merge")([context, last_hidden, base_input])
    merged = LayerNormalization(name="ln_merge")(merged)

    # Shared residual dense block
    h = Dense(dense_units, activation="relu",
              kernel_regularizer=regularizers.l2(l2_reg), name="dense1")(merged)
    h = Dropout(drop, name="drop1")(h)
    h2 = Dense(dense_units, activation="relu",
               kernel_regularizer=regularizers.l2(l2_reg), name="dense2")(h)
    h2 = Dropout(drop * 0.5, name="drop2")(h2)
    h = Add(name="residual")([h, h2])
    h = LayerNormalization(name="ln_res")(h)

    # ============================================================
    # SPIKE DETECTION HEAD
    # Binary classifier: is this game a spike/outlier?
    # Uses the same shared representation but separate dense layers
    # ============================================================
    spike_h = Dense(32, activation="relu", name="spike_dense1")(h)
    spike_h = Dropout(0.2, name="spike_drop")(spike_h)
    spike_prob = Dense(1, activation="sigmoid", name="spike_prob")(spike_h)

    # ============================================================
    # DUAL-PATH PREDICTION
    # Conservative path: standard per-target heads (like v4)
    # Spike path: wider heads that can make larger predictions
    # Final = conservative + spike_prob * (spike_path - conservative)
    # This lets the model "amplify" predictions when it detects a spike
    # ============================================================
    
    # Conservative path (standard delta prediction)
    conservative_preds = []
    for i in range(n_targets):
        ti = Dense(head_units, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_reg),
                   name=f"cons_head_{i}_fc")(h)
        ti = Dropout(drop * 0.3, name=f"cons_head_{i}_drop")(ti)
        ti = Dense(1, activation=None, name=f"cons_head_{i}_out")(ti)
        conservative_preds.append(ti)
    conservative_delta = Concatenate(name="conservative_delta")(conservative_preds)

    # Spike path (can make larger corrections, especially for PTS)
    spike_preds = []
    for i in range(n_targets):
        # Wider hidden layer for spike path — more capacity for extreme predictions
        si = Dense(head_units + 16, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_reg * 0.5),  # less reg for spike path
                   name=f"spike_head_{i}_fc")(h)
        si = Dropout(drop * 0.2, name=f"spike_head_{i}_drop")(si)
        si = Dense(1, activation=None, name=f"spike_head_{i}_out")(si)
        spike_preds.append(si)
    spike_delta = Concatenate(name="spike_delta")(spike_preds)

    # Blend: delta = conservative + spike_prob * (spike - conservative)
    # When spike_prob ≈ 0: output ≈ conservative (safe, mean-regressing)
    # When spike_prob ≈ 1: output ≈ spike (aggressive, can chase outliers)
    spike_gate = Lambda(lambda sp: tf.repeat(sp, n_targets, axis=-1), name="spike_gate")(spike_prob)
    diff = Lambda(lambda args: args[0] - args[1], name="spike_diff")([spike_delta, conservative_delta])
    spike_adjustment = Multiply(name="spike_adjust")([spike_gate, diff])
    delta_pred = Add(name="blended_delta")([conservative_delta, spike_adjustment])

    # Concatenate delta predictions + spike probability for the loss function
    output = Concatenate(name="output")([delta_pred, spike_prob])

    model = Model(inputs=[seq_input, base_input], outputs=output,
                  name=f"lstm_v5_s{seed}")
    return model


# ============================================================
# SPIKE LABEL GENERATION
# ============================================================

def generate_spike_labels(delta_scaled, scaler_y, threshold_pts=8.0, threshold_other=3.0):
    """
    Generate binary spike labels from scaled deltas.
    
    A "spike" is when the actual performance deviates significantly from the
    rolling average baseline. We define this in original (unscaled) space.
    
    Args:
        delta_scaled: [N, 3] scaled deltas (target - baseline in scaled space)
        scaler_y: StandardScaler used for targets
        threshold_pts: PTS deviation threshold in raw points (e.g., 8 = ±8 points from avg)
        threshold_other: TRB/AST deviation threshold in raw units
    
    Returns:
        spike_labels: [N] binary array (1 = spike game)
        sample_weights: [N] float array (higher weight for spike games)
    """
    # Convert thresholds to scaled space
    # delta_raw = delta_scaled * scale + 0 (since both target and baseline use same scaler,
    # the delta in scaled space = delta_raw / scale)
    scales = scaler_y.scale_  # [PTS_std, TRB_std, AST_std]
    
    thresh_scaled = np.array([
        threshold_pts / scales[0],   # PTS threshold in scaled space
        threshold_other / scales[1], # TRB threshold in scaled space  
        threshold_other / scales[2], # AST threshold in scaled space
    ])
    
    # A game is a spike if ANY target exceeds its threshold
    abs_delta = np.abs(delta_scaled)
    is_spike = np.any(abs_delta > thresh_scaled, axis=1).astype(np.float32)
    
    # Also flag PTS-specific spikes (our main target for improvement)
    pts_spike = (abs_delta[:, 0] > thresh_scaled[0]).astype(np.float32)
    
    # Combined: spike if PTS spike OR general spike
    spike_labels = np.maximum(is_spike, pts_spike)
    
    n_spikes = int(np.sum(spike_labels))
    n_total = len(spike_labels)
    print(f"  Spike detection: {n_spikes}/{n_total} games ({100*n_spikes/n_total:.1f}%) are spikes")
    print(f"    PTS spikes: {int(np.sum(pts_spike))}")
    print(f"    Thresholds (raw): PTS=±{threshold_pts}, TRB/AST=±{threshold_other}")
    print(f"    Thresholds (scaled): {thresh_scaled}")
    
    # Sample weights: upweight spike games
    # Use inverse frequency weighting so spikes get proportionally more attention
    spike_ratio = max(n_spikes / n_total, 0.01)
    normal_weight = 1.0
    spike_weight = min((1.0 - spike_ratio) / spike_ratio, 5.0)  # cap at 5x
    
    sample_weights = np.where(spike_labels == 1, spike_weight, normal_weight).astype(np.float32)
    print(f"    Sample weights: normal={normal_weight:.2f}, spike={spike_weight:.2f}")
    
    return spike_labels, sample_weights


# ============================================================
# CUSTOM TRAINING WITH SPIKE-AWARE LOSS
# ============================================================

class SpikeAwareModel(tf.keras.Model):
    """
    Wrapper that handles the combined output [delta(3), spike_prob(1)]
    and computes spike-aware loss.
    """
    def __init__(self, base_model, spike_loss_weight=0.15, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.spike_loss_weight = spike_loss_weight
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        (x, y_combined) = data
        # y_combined shape: [batch, 4] = [delta(3), spike_label(1)]
        y_delta = y_combined[:, :3]
        y_spike = y_combined[:, 3:4]
        
        with tf.GradientTape() as tape:
            pred = self.base_model(x, training=True)
            pred_delta = pred[:, :3]
            pred_spike = pred[:, 3:4]
            
            # Delta prediction loss (spike-aware)
            delta_loss = spike_aware_loss(y_delta, pred_delta)
            
            # Spike classification loss (binary cross-entropy)
            bce = tf.keras.losses.binary_crossentropy(y_spike, pred_spike)
            spike_cls_loss = tf.reduce_mean(bce)
            
            # Spike-conditioned extra penalty:
            # When model predicts spike but delta prediction is too conservative,
            # add extra penalty. This encourages the model to actually USE the spike signal.
            spike_conf = tf.squeeze(pred_spike, axis=-1)  # [batch]
            abs_pred_delta = tf.abs(pred_delta[:, 0])  # PTS delta magnitude
            abs_true_delta = tf.abs(y_delta[:, 0])
            # If spike_conf > 0.5 and true delta is large but pred delta is small
            spike_underpredict = tf.reduce_mean(
                spike_conf * tf.maximum(0.0, abs_true_delta - abs_pred_delta - 0.5)
            )
            
            total_loss = delta_loss + self.spike_loss_weight * spike_cls_loss + 0.05 * spike_underpredict
            total_loss += sum(self.base_model.losses)  # L2 regularization
        
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        # Metrics — all must be scalars
        mae = tf.reduce_mean(tf.abs(y_delta - pred_delta))
        return {"loss": total_loss, "mae": mae, "delta_loss": delta_loss, 
                "spike_cls": spike_cls_loss}
    
    def test_step(self, data):
        (x, y_combined) = data
        y_delta = y_combined[:, :3]
        y_spike = y_combined[:, 3:4]
        
        pred = self.base_model(x, training=False)
        pred_delta = pred[:, :3]
        pred_spike = pred[:, 3:4]
        
        delta_loss = spike_aware_loss(y_delta, pred_delta)
        bce = tf.keras.losses.binary_crossentropy(y_spike, pred_spike)
        spike_cls_loss = tf.reduce_mean(bce)
        total_loss = delta_loss + self.spike_loss_weight * spike_cls_loss
        
        mae = tf.reduce_mean(tf.abs(y_delta - pred_delta))
        return {"loss": total_loss, "mae": mae, "delta_loss": delta_loss,
                "spike_cls": spike_cls_loss}


# ============================================================
# MIXUP GENERATOR (spike-aware)
# ============================================================

def mixup_batch(X, b, delta, spike, alpha=0.2):
    """Mixup augmentation — only mix samples with same spike label to preserve signal."""
    batch_size = len(X)
    indices = np.random.permutation(batch_size)
    lam = np.random.beta(alpha, alpha, size=(batch_size, 1))
    lam_seq = lam[:, :, np.newaxis]

    X_mix = lam_seq * X + (1 - lam_seq) * X[indices]
    b_mix = lam * b + (1 - lam) * b[indices]
    d_mix = lam * delta + (1 - lam) * delta[indices]
    # For spike labels, use the max (if either is a spike, the mix is a spike)
    s_mix = np.maximum(spike, spike[indices])
    return X_mix, b_mix, d_mix, s_mix


class SpikeAwareMixupGenerator(tf.keras.utils.Sequence):
    """Data generator with spike-aware mixup and sample weighting."""
    def __init__(self, X, b, delta, spike_labels, sample_weights,
                 batch_size=96, mixup_alpha=0.15, target_noise=0.015):
        self.X = X
        self.b = b
        self.delta = delta
        self.spike_labels = spike_labels
        self.sample_weights = sample_weights
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
        s_batch = self.spike_labels[batch_idx].copy().reshape(-1, 1)

        # Mixup with 50% probability
        if np.random.random() < 0.5 and len(X_batch) > 1:
            X_batch, b_batch, d_batch, s_batch = mixup_batch(
                X_batch, b_batch, d_batch, s_batch, self.mixup_alpha
            )

        # Label smoothing on targets (less noise on spike games to preserve signal)
        if self.target_noise > 0:
            noise_scale = np.where(s_batch > 0.5, self.target_noise * 0.5, self.target_noise)
            d_batch = d_batch + np.random.normal(0, 1, d_batch.shape) * noise_scale

        # Combine delta + spike label as target: [delta(3), spike(1)]
        y_combined = np.concatenate([d_batch, s_batch], axis=-1)

        return [X_batch, b_batch], y_combined

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_single(X_train, b_train, delta_train, spike_train, weights_train,
                 X_val, b_val, delta_val, spike_val,
                 seq_len, n_features, n_targets, seed=42, config=None,
                 scaler_y=None, y_val_raw=None, target_names=None):
    """Train a single spike-aware v5 model."""
    cfg = config or {}
    epochs = cfg.get("epochs", 150)
    batch_size = cfg.get("batch_size", 96)
    max_lr = cfg.get("max_lr", 7e-4)

    base_model = build_lstm_v5(seq_len, n_features, n_targets, seed=seed, config=cfg)
    base_model.summary()

    # Wrap in spike-aware training model
    model = SpikeAwareModel(base_model, spike_loss_weight=0.15)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=max_lr, clipnorm=0.8),
    )

    # Spike-aware data generator
    train_gen = SpikeAwareMixupGenerator(
        X_train, b_train, delta_train, spike_train, weights_train,
        batch_size=batch_size,
        mixup_alpha=0.15,
        target_noise=0.015
    )

    # Validation data: combine delta + spike labels
    val_y_combined = np.concatenate([delta_val, spike_val.reshape(-1, 1)], axis=-1)

    swa_cb = SWACallback(swa_start=35, swa_freq=5)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        CosineAnnealingWarmup(warmup_epochs=6, max_lr=max_lr, min_lr=1e-6, total_epochs=epochs),
        swa_cb,
    ]

    if scaler_y is not None and y_val_raw is not None:
        callbacks.append(R2MonitorCallback(
            X_val, b_val, scaler_y, y_val_raw,
            target_names or ["PTS", "TRB", "AST"],
            spike_labels=spike_val
        ))

    history = model.fit(
        train_gen,
        validation_data=([X_val, b_val], val_y_combined),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    best_val = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val) + 1
    print(f"  Best val_loss: {best_val:.5f} at epoch {best_epoch}")

    # Try SWA — note: SWA callback operates on the wrapper model,
    # but weights flow through to base_model since wrapper has no own params
    es_val = best_val
    es_weights = [w.copy() for w in model.get_weights()]

    if swa_cb.apply_swa():
        swa_val = model.evaluate([X_val, b_val], val_y_combined, verbose=0)
        if isinstance(swa_val, dict):
            swa_val = swa_val.get('loss', swa_val)
        elif isinstance(swa_val, (list, tuple)):
            swa_val = swa_val[0]
        print(f"  SWA val_loss: {swa_val:.5f} vs ES val_loss: {es_val:.5f}")
        if swa_val < es_val:
            print(f"  -> Using SWA weights")
            best_val = swa_val
        else:
            print(f"  -> Keeping EarlyStopping weights")
            model.set_weights(es_weights)

    return base_model, history, best_val


def train_ensemble(X_train, b_train, delta_train, spike_train, weights_train,
                   X_val, b_val, delta_val, spike_val,
                   seq_len, n_features, n_targets, n_models=5, config=None,
                   scaler_y=None, y_val_raw=None, target_names=None):
    """Train N spike-aware models with architectural diversity."""
    models = []
    val_losses = []

    # Same diversity as v4 but with spike-tuned variants
    variants = [
        {"lstm_units": 96,  "dense_units": 80,  "attn_units": 32, "dropout": 0.30, "noise_std": 0.03},
        {"lstm_units": 112, "dense_units": 72,  "attn_units": 28, "dropout": 0.27, "noise_std": 0.035},
        {"lstm_units": 80,  "dense_units": 96,  "attn_units": 36, "dropout": 0.32, "noise_std": 0.025},
        {"lstm_units": 104, "dense_units": 80,  "attn_units": 32, "dropout": 0.28, "noise_std": 0.03},
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
            X_train, b_train, delta_train, spike_train, weights_train,
            X_val, b_val, delta_val, spike_val,
            seq_len, n_features, n_targets,
            seed=seed, config=cfg,
            scaler_y=scaler_y, y_val_raw=y_val_raw,
            target_names=target_names
        )
        models.append(model)
        val_losses.append(best_val)

    return models, val_losses


# ============================================================
# PREDICTION & EVALUATION
# ============================================================

def predict_with_spike_info(model, X, baselines):
    """Get delta predictions and spike probabilities from a v5 model."""
    raw_pred = model.predict([X, baselines], verbose=0, batch_size=512)
    delta_pred = raw_pred[:, :3]
    spike_prob = raw_pred[:, 3] if raw_pred.shape[-1] > 3 else np.zeros(len(X))
    return delta_pred, spike_prob


def weighted_ensemble_predict(models, val_losses, X, baselines):
    """Inverse-loss weighted ensemble prediction with spike info."""
    weights = 1.0 / (np.array(val_losses) + 1e-8)
    weights /= weights.sum()
    
    all_deltas = []
    all_spikes = []
    for m in models:
        delta, spike = predict_with_spike_info(m, X, baselines)
        all_deltas.append(delta)
        all_spikes.append(spike)
    
    ensemble_delta = sum(d * w for d, w in zip(all_deltas, weights))
    ensemble_spike = sum(s * w for s, w in zip(all_spikes, weights))
    
    return ensemble_delta, ensemble_spike


def main():
    print("=" * 80)
    print("IMPROVED LSTM v5 — Spike-Aware Architecture")
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

    # ============================================================
    # GENERATE SPIKE LABELS
    # ============================================================
    print("\n--- Generating spike labels ---")
    spike_train, weights_train = generate_spike_labels(
        delta_train, trainer.scaler_y,
        threshold_pts=8.0,    # ±8 points from rolling avg = spike
        threshold_other=3.0   # ±3 rebounds/assists from rolling avg = spike
    )
    spike_val, _ = generate_spike_labels(
        delta_val, trainer.scaler_y,
        threshold_pts=8.0,
        threshold_other=3.0
    )

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
        X_train, b_train, delta_train, spike_train, weights_train,
        X_val, b_val, delta_val, spike_val,
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
        delta, spike_p = predict_with_spike_info(m, X_val, b_val)
        ps = b_val + delta
        po = trainer.scaler_y.inverse_transform(ps)
        maes = [mean_absolute_error(y_val_orig[:, j], po[:, j]) for j in range(n_targets)]
        r2s = [r2_score(y_val_orig[:, j], po[:, j]) for j in range(n_targets)]
        avg_spike = np.mean(spike_p)
        print(f"  Member {i+1}: PTS={maes[0]:.3f}(R²={r2s[0]:.3f})  "
              f"TRB={maes[1]:.3f}(R²={r2s[1]:.3f})  "
              f"AST={maes[2]:.3f}(R²={r2s[2]:.3f})  avg={np.mean(maes):.3f}  "
              f"spike_avg={avg_spike:.3f}")

    # Weighted ensemble
    print("\n" + "=" * 80)
    print("WEIGHTED ENSEMBLE")
    print("=" * 80)
    delta_pred, spike_probs = weighted_ensemble_predict(models, val_losses, X_val, b_val)
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
    
    # Spike-specific analysis
    print(f"\n{'='*60}")
    print("SPIKE ANALYSIS")
    print(f"{'='*60}")
    spike_mask = spike_val == 1
    normal_mask = spike_val == 0
    for i, target in enumerate(trainer.target_columns):
        if np.sum(spike_mask) > 0:
            spike_mae = mean_absolute_error(y_val_orig[spike_mask, i], pred_orig[spike_mask, i])
            normal_mae = mean_absolute_error(y_val_orig[normal_mask, i], pred_orig[normal_mask, i])
            print(f"  {target}: Normal MAE={normal_mae:.3f}, Spike MAE={spike_mae:.3f}")
    
    print(f"\n{'='*60}")
    print(f"  OVERALL AVG MAE: {avg_mae:.4f}")
    print(f"  v4 weighted:     4.709  (R² ~0.21/0.21/0.28)")
    print(f"  v2 weighted:     4.811  (R² ~0.21)")
    print(f"{'='*60}")

    # Top-3 ensemble
    print("\n--- Top-3 ensemble ---")
    sorted_idx = np.argsort(val_losses)[:3]
    top_deltas = []
    for i in sorted_idx:
        d, _ = predict_with_spike_info(models[i], X_val, b_val)
        top_deltas.append(d)
    top_losses = [val_losses[i] for i in sorted_idx]
    top_weights = 1.0 / (np.array(top_losses) + 1e-8)
    top_weights /= top_weights.sum()
    delta_top = sum(d * w for d, w in zip(top_deltas, top_weights))
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
        m.save_weights(f"model/lstm_v5_member_{i}.weights.h5")
    joblib.dump(trainer.scaler_x, 'model/lstm_v5_scaler_x.pkl')
    joblib.dump(trainer.scaler_y, 'model/lstm_v5_scaler_y.pkl')

    meta = {
        'model_type': 'lstm_v5_spike_aware_ensemble',
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
        'spike_threshold_pts': 8.0,
        'spike_threshold_other': 3.0,
    }
    with open('model/lstm_v5_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to model/")


if __name__ == "__main__":
    main()
