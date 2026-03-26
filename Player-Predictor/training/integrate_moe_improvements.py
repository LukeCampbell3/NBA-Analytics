#!/usr/bin/env python3
"""
MoE Improvements Integration Script (UPDATED - DELTA-ONLY + WORKING OUTLIERS)

Key upgrades:
- ✅ NO baseline softening: the network outputs DELTA ONLY (never baseline-added inside the model)
- ✅ Training math is unambiguous: delta_true = y - baseline_pred, train against delta_pred
- ✅ Outlier detection works: per-batch robust MAD-z spike labels with guaranteed non-zero positives
- ✅ Supervised outlier head with focal loss; routing bias is driven by outlier_prob (not raw spike features)
- ✅ Optional "within deviation" band-loss (hinge outside ±band per target)
- ✅ Keeps Phase 1+2 anti-collapse/specialization + Phase 3 diversity
- ✅ Supports generators yielding (seq, baseline, extra/meta...) safely

IMPORTANT ASSUMPTION:
- baseline_input is baseline_pred in the SAME UNITS as y_true and shape [B, n_targets].
  If you're feeding baseline FEATURES instead, fix that first.

FIXES IN THIS REVISION:
- ✅ Prevent KeyError for missing config keys (log_moe_metrics_every_n_epochs, warn_on_collapse, etc.)
- ✅ Robust median/quantile helpers rewritten (no broken transpose logic)
"""

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd  # parity with your environment (may be used downstream)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from improved_baseline_trainer import (
    ImprovedBaselineTrainer,
    BaselineAwareKerasModel,  # kept for compatibility with your codebase
    TimeAwareBatchGenerator,
    ValR2Callback,
)
from moe_metrics import MoEMetricsTracker


# -----------------------------
# Utility losses / helpers
# -----------------------------

def _huber(delta=1.0):
    return tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)


def focal_bce(y_true, y_prob, gamma=2.0, alpha=0.25, eps=1e-7):
    """
    Binary focal loss. y_true in {0,1}, y_prob in [0,1]
    """
    y_true = tf.cast(y_true, tf.float32)
    y_prob = tf.clip_by_value(tf.cast(y_prob, tf.float32), eps, 1.0 - eps)

    p_t = y_true * y_prob + (1.0 - y_true) * (1.0 - y_prob)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
    return loss


def tf_median(x, axis=0, keepdims=True):
    """
    Pure-TF median along a given axis using sort + gather.
    Works for any rank >=1.
    """
    x = tf.cast(x, tf.float32)
    x_sorted = tf.sort(x, axis=axis)

    n = tf.shape(x_sorted)[axis]
    mid = n // 2
    is_even = tf.equal(n % 2, 0)

    def _even():
        a = tf.gather(x_sorted, mid - 1, axis=axis)
        b = tf.gather(x_sorted, mid, axis=axis)
        return 0.5 * (a + b)

    def _odd():
        return tf.gather(x_sorted, mid, axis=axis)

    med = tf.cond(is_even, _even, _odd)

    if keepdims:
        med = tf.expand_dims(med, axis=axis)
    return med


def tf_quantile_1d(x, q):
    """
    Pure-TF quantile for 1D tensor x. q in [0,1]. Uses nearest-rank style.
    """
    x = tf.sort(tf.reshape(tf.cast(x, tf.float32), [-1]))
    n = tf.shape(x)[0]
    idx = tf.cast(tf.round(q * tf.cast(n - 1, tf.float32)), tf.int32)
    idx = tf.clip_by_value(idx, 0, n - 1)
    return tf.gather(x, idx)


def robust_spike_labels_from_residuals(residuals, z_thr=2.75, min_pos_frac=0.03, eps=1e-6):
    """
    residuals: [B, T] (delta_true = y - baseline)
    Returns:
      spike_any: [B, 1] float32 in {0,1}  (any target is a spike)
      spike_per_target: [B, T] float32 in {0,1}
      robust_z: [B, T] float32
    Uses per-batch robust z via MAD; ensures >=min_pos_frac positives by quantile fallback.
    """
    res = tf.cast(residuals, tf.float32)

    med = tf_median(res, axis=0, keepdims=True)  # [1, T]
    mad = tf_median(tf.abs(res - med), axis=0, keepdims=True) + eps  # [1, T]

    robust_z = tf.abs(res - med) / (1.4826 * mad)  # [B, T]
    spike_per_target = tf.cast(robust_z > z_thr, tf.float32)  # [B, T]
    spike_any = tf.cast(tf.reduce_any(spike_per_target > 0.0, axis=1, keepdims=True), tf.float32)  # [B,1]

    pos_rate = tf.reduce_mean(spike_any)

    def _quantile_fallback():
        scores = tf.reduce_max(robust_z, axis=1)  # [B]
        cutoff = tf_quantile_1d(scores, 1.0 - float(min_pos_frac))  # scalar
        spike_any_fb = tf.cast(scores >= cutoff, tf.float32)[:, None]  # [B,1]
        # keep per-target labels as-is; the supervised head trains on spike_any.
        return spike_any_fb, spike_per_target

    spike_any, spike_per_target = tf.cond(
        pos_rate < tf.constant(0.01, tf.float32),
        true_fn=_quantile_fallback,
        false_fn=lambda: (spike_any, spike_per_target),
    )

    return spike_any, spike_per_target, robust_z


# -----------------------------
# Delta-trained wrapper model
# -----------------------------

class DeltaTrainedKerasModel(tf.keras.Model):
    """
    DELTA-ONLY training wrapper.

    Net outputs:
      - delta_pred: [B, T] (or [B, T*4] if probabilistic)
      - outlier_prob: [B, 1] supervised by robust spike labels derived from delta_true

    Training targets:
      delta_true = y_true - baseline_pred

    Optional band-loss on absolute prediction:
      y_hat = baseline_pred + delta_pred_means
    """

    def __init__(
        self,
        net,
        target_columns,
        use_probabilistic=False,
        huber_delta=1.0,
        band=None,                 # list/np array length T or None
        band_weight=0.0,           # set >0 to enforce "within deviation"
        outlier_gamma=2.0,
        outlier_alpha=0.25,
        outlier_weight=0.5,
        spike_z_thr=2.75,
        spike_min_pos_frac=0.03,
    ):
        super().__init__()
        self.net = net
        self.target_columns = list(target_columns or [])
        self.T = len(self.target_columns)
        self.use_probabilistic = bool(use_probabilistic)

        self.huber = _huber(delta=float(huber_delta))

        self.band = None if band is None else tf.constant(np.array(band, dtype=np.float32))
        self.band_weight = float(band_weight)

        self.outlier_gamma = float(outlier_gamma)
        self.outlier_alpha = float(outlier_alpha)
        self.outlier_weight = float(outlier_weight)

        self.spike_z_thr = float(spike_z_thr)
        self.spike_min_pos_frac = float(spike_min_pos_frac)

        # Trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.delta_loss_tracker = tf.keras.metrics.Mean(name="delta_loss")
        self.band_loss_tracker = tf.keras.metrics.Mean(name="band_loss")
        self.outlier_loss_tracker = tf.keras.metrics.Mean(name="outlier_loss")
        self.abs_mae_tracker = tf.keras.metrics.Mean(name="abs_mae")
        self.spike_rate_tracker = tf.keras.metrics.Mean(name="spike_rate")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.delta_loss_tracker,
            self.band_loss_tracker,
            self.outlier_loss_tracker,
            self.abs_mae_tracker,
            self.spike_rate_tracker,
        ]

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    @staticmethod
    def _split_inputs(x):
        """
        Accepts:
          - (seq, base)
          - (seq, base, extra...)
          - [seq, base]
          - [seq, base, extra...]
        Returns (seq, base, extra_or_none)
        """
        if isinstance(x, (tuple, list)):
            if len(x) < 2:
                raise ValueError(f"Expected at least 2 inputs (seq, baseline_pred). Got {len(x)}")
            extra = x[2] if len(x) > 2 else None
            return x[0], x[1], extra
        raise ValueError(f"Expected inputs as tuple/list, got {type(x)}")

    def _extract_delta_means(self, delta_pred):
        """
        delta_pred:
          - deterministic: [B, T]
          - probabilistic: [B, T*4] (mean at i*4)
        """
        if not self.use_probabilistic:
            return delta_pred
        return tf.stack([delta_pred[:, i * 4] for i in range(self.T)], axis=1)

    def call(self, inputs, training=False):
        x_seq, x_base, _ = self._split_inputs(inputs)
        delta_pred, outlier_prob = self.net([x_seq, x_base], training=training)
        delta_means = self._extract_delta_means(delta_pred)
        y_hat = tf.cast(x_base, tf.float32) + tf.cast(delta_means, tf.float32)
        return y_hat

    def train_step(self, data):
        x, y_true = data
        x_seq, x_base, _ = self._split_inputs(x)

        y_true = tf.cast(y_true, tf.float32)
        x_base = tf.cast(x_base, tf.float32)

        with tf.GradientTape() as tape:
            delta_pred, outlier_prob = self.net([x_seq, x_base], training=True)
            delta_means = self._extract_delta_means(delta_pred)

            delta_true = y_true - x_base

            # Main delta loss (Huber)
            per_elem = self.huber(delta_true, delta_means)  # [B,T]
            delta_loss = tf.reduce_mean(per_elem)

            # Absolute prediction
            y_hat = x_base + delta_means
            abs_mae = tf.reduce_mean(tf.abs(y_true - y_hat))

            # Band loss: penalize only outside ±band
            band_loss = 0.0
            if self.band is not None and self.band_weight > 0.0:
                err = tf.abs(y_true - y_hat)  # [B,T]
                band = tf.reshape(self.band, [1, self.T])
                band_violation = tf.nn.relu(err - band)
                band_loss = tf.reduce_mean(band_violation)

            # Robust spike labels (guaranteed non-zero positives)
            spike_any, _, _ = robust_spike_labels_from_residuals(
                delta_true,
                z_thr=self.spike_z_thr,
                min_pos_frac=self.spike_min_pos_frac,
            )
            self.spike_rate_tracker.update_state(tf.reduce_mean(spike_any))

            # Outlier focal loss
            outlier_prob = tf.cast(outlier_prob, tf.float32)
            outlier_loss = tf.reduce_mean(
                focal_bce(spike_any, outlier_prob, gamma=self.outlier_gamma, alpha=self.outlier_alpha)
            )

            # MoE auxiliary losses already registered on self.net
            reg_loss = tf.add_n(self.net.losses) if self.net.losses else 0.0

            total = delta_loss + (self.band_weight * band_loss) + (self.outlier_weight * outlier_loss) + reg_loss

        grads = tape.gradient(total, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        self.loss_tracker.update_state(total)
        self.delta_loss_tracker.update_state(delta_loss)
        self.band_loss_tracker.update_state(band_loss)
        self.outlier_loss_tracker.update_state(outlier_loss)
        self.abs_mae_tracker.update_state(abs_mae)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_true = data
        x_seq, x_base, _ = self._split_inputs(x)

        y_true = tf.cast(y_true, tf.float32)
        x_base = tf.cast(x_base, tf.float32)

        delta_pred, outlier_prob = self.net([x_seq, x_base], training=False)
        delta_means = self._extract_delta_means(delta_pred)

        delta_true = y_true - x_base
        per_elem = self.huber(delta_true, delta_means)
        delta_loss = tf.reduce_mean(per_elem)

        y_hat = x_base + delta_means
        abs_mae = tf.reduce_mean(tf.abs(y_true - y_hat))

        band_loss = 0.0
        if self.band is not None and self.band_weight > 0.0:
            err = tf.abs(y_true - y_hat)
            band = tf.reshape(self.band, [1, self.T])
            band_violation = tf.nn.relu(err - band)
            band_loss = tf.reduce_mean(band_violation)

        spike_any, _, _ = robust_spike_labels_from_residuals(
            delta_true,
            z_thr=self.spike_z_thr,
            min_pos_frac=self.spike_min_pos_frac,
        )
        self.spike_rate_tracker.update_state(tf.reduce_mean(spike_any))

        outlier_prob = tf.cast(outlier_prob, tf.float32)
        outlier_loss = tf.reduce_mean(
            focal_bce(spike_any, outlier_prob, gamma=self.outlier_gamma, alpha=self.outlier_alpha)
        )

        reg_loss = tf.add_n(self.net.losses) if self.net.losses else 0.0
        total = delta_loss + (self.band_weight * band_loss) + (self.outlier_weight * outlier_loss) + reg_loss

        self.loss_tracker.update_state(total)
        self.delta_loss_tracker.update_state(delta_loss)
        self.band_loss_tracker.update_state(band_loss)
        self.outlier_loss_tracker.update_state(outlier_loss)
        self.abs_mae_tracker.update_state(abs_mae)

        return {m.name: m.result() for m in self.metrics}


# -----------------------------
# Replay buffer (kept; not fully wired here)
# -----------------------------

class ExpertReplayBuffer:
    """
    Simple FIFO replay buffer that stores samples per expert.
    Helps maintain expert specialization during curriculum/hard mining.
    """
    def __init__(self, num_experts, buffer_size_per_expert=500):
        self.num_experts = num_experts
        self.buffer_size = buffer_size_per_expert
        self.buffers = {i: [] for i in range(num_experts)}

    def add_samples(self, X, baselines, y, expert_assignments, expert_outputs):
        for i in range(len(X)):
            expert_id = int(expert_assignments[i])
            if 0 <= expert_id < self.num_experts:
                sample = (
                    X[i].numpy(),
                    baselines[i].numpy(),
                    y[i].numpy(),
                    expert_outputs[i].numpy(),
                )
                self.buffers[expert_id].append(sample)
                if len(self.buffers[expert_id]) > self.buffer_size:
                    self.buffers[expert_id].pop(0)

    def sample_replay(self, batch_size, expert_distribution=None):
        all_samples = []
        for expert_id in range(self.num_experts):
            if self.buffers[expert_id]:
                all_samples.extend([(expert_id, s) for s in self.buffers[expert_id]])

        if not all_samples:
            return None

        n_samples = min(batch_size, len(all_samples))
        sampled = np.random.choice(len(all_samples), size=n_samples, replace=False)

        X_list, b_list, y_list, old_out_list = [], [], [], []
        for idx in sampled:
            expert_id, (x, b, y_val, old_out) = all_samples[idx]
            X_list.append(x)
            b_list.append(b)
            y_list.append(y_val)
            old_out_list.append(old_out)

        return (
            np.array(X_list),
            np.array(b_list),
            np.array(y_list),
            np.array(old_out_list),
        )

    def get_buffer_stats(self):
        return {f"expert_{i}_buffer_size": len(self.buffers[i]) for i in range(self.num_experts)}


# -----------------------------
# Trainer
# -----------------------------

class MoEImprovedTrainer(ImprovedBaselineTrainer):
    """
    Enhanced trainer with MoE improvements integrated.

    Major change:
    - Model outputs DELTA ONLY + supervised outlier_prob.
    - Outlier supervision is generated per-batch from delta_true (robust MAD z-score).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        total_experts = self.config.get("num_experts", 8) + self.config.get("num_spike_experts", 3)
        self.moe_metrics = MoEMetricsTracker(
            num_experts=total_experts,
            num_spike_experts=self.config.get("num_spike_experts", 3),
            dead_threshold=0.02,
        )

        self.replay_buffer = None
        entropy_target_frac = self.config.get("entropy_target_frac", 0.85)

        # Default deviation bands (override via config['band_per_target'])
        if self.config.get("band_per_target", None) is None:
            band = []
            for c in self.target_columns:
                cu = c.upper()
                if "PTS" in cu:
                    band.append(float(self.config.get("band_pts", 4.0)))
                elif "TRB" in cu or "REB" in cu:
                    band.append(float(self.config.get("band_trb", 2.0)))
                elif "AST" in cu:
                    band.append(float(self.config.get("band_ast", 2.0)))
                else:
                    band.append(float(self.config.get("band_default", 2.0)))
            self.config["band_per_target"] = band

        # ---- Ensure required config keys exist BEFORE they are accessed anywhere ----
        self.config.update(
            {
                # ===== METRICS / LOGGING DEFAULTS (prevents KeyError) =====
                "log_moe_metrics_every_n_epochs": int(self.config.get("log_moe_metrics_every_n_epochs", 1)),
                "warn_on_collapse": bool(self.config.get("warn_on_collapse", True)),
                "dead_threshold": float(self.config.get("dead_threshold", 0.02)),
                "collapse_warning_threshold": float(self.config.get("collapse_warning_threshold", 0.30)),
                "entropy_warning_threshold": float(self.config.get("entropy_warning_threshold", 0.50)),

                # ===== PHASE 1: ANTI-COLLAPSE =====
                "use_load_balancing": True,
                "load_balance_schedule": "ramp",
                "load_balance_weight_start": float(self.config.get("load_balance_weight_start", 0.01)),  # Increased from 0.001 to 0.01
                "load_balance_weight_mid": float(self.config.get("load_balance_weight_mid", 0.05)),  # Increased from 0.003 to 0.05
                "load_balance_weight_final": float(self.config.get("load_balance_weight_final", 0.1)),  # Increased from 0.008 to 0.1
                "load_balance_ramp_epochs": self.config.get("load_balance_ramp_epochs", [3, 15]),  # Extended ramp period
                "use_capacity_enforcement": True,
                "capacity_factor": float(self.config.get("capacity_factor", 2.0)),  # Increased from 1.25 to 2.0
                "overflow_penalty": float(self.config.get("overflow_penalty", 0.02)),  # Increased from 0.01
                "track_overflow": True,
                "use_importance_loss": True,
                "use_load_loss": True,
                "importance_weight": float(self.config.get("importance_weight", 0.02)),  # Increased from 0.01
                "load_weight": float(self.config.get("load_weight", 0.02)),  # Increased from 0.01

                # ===== ROUTER EXPLORATION / STABILITY =====
                "router_temperature": 5.0,  # Increased from 1.35/2.0 to 5.0 for more exploration
                "router_noise_std": 0.05,  # Increased from 0.02 to 0.05
                "router_z_loss_weight": 0.001,  # Enable z-loss for stability

                # ===== PHASE 2: SPECIALIZATION =====
                "use_prototype_experts": True,
                "prototype_key_dim": int(self.config.get("prototype_key_dim", 256)),
                "compactness_coef": float(self.config.get("compactness_coef", 0.005)),  # Reduced from 0.01
                "separation_coef": float(self.config.get("separation_coef", 0.001)),
                "separation_margin": float(self.config.get("separation_margin", 0.5)),
                "compactness_schedule": self.config.get("compactness_schedule", "ramp"),
                "compactness_start": float(self.config.get("compactness_start", 0.0)),  # Start at 0 to avoid early collapse
                "compactness_final": float(self.config.get("compactness_final", 0.005)),  # Reduced from 0.01
                "compactness_ramp_epochs": self.config.get("compactness_ramp_epochs", [10, 25]),  # Delayed start

                # ===== PHASE 3: DIVERSITY =====
                "use_diversity_regularization": bool(self.config.get("use_diversity_regularization", True)),
                "diversity_coef": float(self.config.get("diversity_coef", 0.001)),
                "diversity_sample_fraction": float(self.config.get("diversity_sample_fraction", 0.5)),

                # ===== PHASE 4 (kept) =====
                "use_expert_replay": bool(self.config.get("use_expert_replay", True)),
                "replay_buffer_size": int(self.config.get("replay_buffer_size", 500)),
                "replay_mix_fraction": float(self.config.get("replay_mix_fraction", 0.2)),
                "consistency_coef": float(self.config.get("consistency_coef", 0.01)),
                "consistency_start_epoch": int(self.config.get("consistency_start_epoch", 10)),

                # ===== ENTROPY REG =====
                "entropy_target_frac": float(entropy_target_frac),

                # ===== OUTLIER SUPERVISION =====
                "spike_z_thr": float(self.config.get("spike_z_thr", 2.75)),
                "spike_min_pos_frac": float(self.config.get("spike_min_pos_frac", 0.03)),
                "outlier_gamma": float(self.config.get("outlier_gamma", 2.0)),
                "outlier_alpha": float(self.config.get("outlier_alpha", 0.25)),
                "outlier_weight": float(self.config.get("outlier_weight", 0.5)),

                # ===== WITHIN-DEVIATION BAND TRAINING =====
                "band_weight": float(self.config.get("band_weight", 0.35)),  # set 0.0 to disable
            }
        )

        print("\n" + "=" * 70)
        print("MoE IMPROVEMENTS ENABLED (DELTA-ONLY + SUPERVISED OUTLIERS)")
        print("=" * 70)
        print(f"Total experts: {total_experts}")
        print(f"Router temperature: {self.config['router_temperature']}")
        print(f"Router noise std (train-only): {self.config['router_noise_std']}")
        print(f"Entropy target frac: {self.config['entropy_target_frac']}")
        print(f"Band per target: {self.config.get('band_per_target')}")
        print(f"Band weight: {self.config.get('band_weight')}")
        print(
            f"Outlier focal: gamma={self.config.get('outlier_gamma')} "
            f"alpha={self.config.get('outlier_alpha')} weight={self.config.get('outlier_weight')}"
        )
        print(
            f"Spike labels: z_thr={self.config.get('spike_z_thr')} "
            f"min_pos_frac={self.config.get('spike_min_pos_frac')}"
        )
        print(f"log_moe_metrics_every_n_epochs: {self.config.get('log_moe_metrics_every_n_epochs')}")
        print(f"warn_on_collapse: {self.config.get('warn_on_collapse')}")
        print("=" * 70 + "\n")

        if self.config.get("use_expert_replay", False):
            self.replay_buffer = ExpertReplayBuffer(
                num_experts=total_experts,
                buffer_size_per_expert=self.config["replay_buffer_size"],
            )
            print("✅ Phase 4 replay buffer initialized\n")

    def get_load_balance_weight(self, epoch):
        if self.config["load_balance_schedule"] == "constant":
            return self.config["load_balance_weight_final"]

        ramp_epochs = self.config["load_balance_ramp_epochs"]
        start_weight = self.config["load_balance_weight_start"]
        mid_weight = self.config["load_balance_weight_mid"]
        final_weight = self.config["load_balance_weight_final"]

        if epoch < ramp_epochs[0]:
            return start_weight
        elif epoch < ramp_epochs[1]:
            progress = (epoch - ramp_epochs[0]) / (ramp_epochs[1] - ramp_epochs[0])
            return start_weight + progress * (mid_weight - start_weight)
        else:
            return final_weight

    def get_compactness_weight(self, epoch):
        if self.config.get("compactness_schedule") == "constant":
            return self.config["compactness_coef"]

        ramp_epochs = self.config["compactness_ramp_epochs"]
        start_weight = self.config["compactness_start"]
        final_weight = self.config["compactness_final"]

        if epoch < ramp_epochs[0]:
            return start_weight
        elif epoch < ramp_epochs[1]:
            progress = (epoch - ramp_epochs[0]) / (ramp_epochs[1] - ramp_epochs[0])
            return start_weight + progress * (final_weight - start_weight)
        else:
            return final_weight

    def build_model_with_phase2(self):
        from tensorflow.keras.layers import (
            Input,
            Dense,
            Embedding,
            Lambda,
            Concatenate,
            Add,
            MultiHeadAttention,
            LayerNormalization,
            Dropout,
            GlobalAveragePooling1D,
            Softmax,
            GaussianNoise,
        )
        from tensorflow.keras.models import Model

        print("🏗️ Building Hybrid MoE (DELTA-ONLY + supervised outlier head)...")

        # Inputs
        sequence_input = Input(shape=(self.config["seq_len"], len(self.feature_columns)), name="sequence_input")

        # IMPORTANT: baseline_input MUST be baseline_pred in target space: [B, T]
        baseline_input = Input(shape=(len(self.target_columns),), name="baseline_pred_input")

        # ID slices
        player_ids = Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32), name="player_ids")(sequence_input)
        team_ids = Lambda(lambda x: tf.cast(x[:, :, 1], tf.int32), name="team_ids")(sequence_input)
        opponent_ids = Lambda(lambda x: tf.cast(x[:, :, 2], tf.int32), name="opponent_ids")(sequence_input)

        player_embed = Embedding(len(self.player_mapping), 16, name="player_embed")(player_ids)
        team_embed = Embedding(len(self.team_mapping), 8, name="team_embed")(team_ids)
        opponent_embed = Embedding(len(self.opponent_mapping), 8, name="opponent_embed")(opponent_ids)

        numeric_features = Lambda(lambda x: x[:, :, 3:], name="numeric_features")(sequence_input)

        combined = Concatenate(axis=-1, name="combined_features")(
            [player_embed, team_embed, opponent_embed, numeric_features]
        )

        x = Dense(self.config["d_model"], name="input_projection")(combined)

        for i in range(self.config["n_layers"]):
            attn = MultiHeadAttention(
                num_heads=self.config["n_heads"],
                key_dim=self.config["d_model"] // self.config["n_heads"],
                name=f"attention_{i}",
            )(x, x)

            x = Add(name=f"add_attn_{i}")([x, attn])
            x = LayerNormalization(name=f"norm_attn_{i}")(x)

            ff = Dense(self.config["d_model"] * 2, activation="relu", name=f"ff1_{i}")(x)
            ff = Dropout(self.config["dropout"], name=f"dropout_ff_{i}")(ff)
            ff = Dense(self.config["d_model"], name=f"ff2_{i}")(ff)

            x = Add(name=f"add_ff_{i}")([x, ff])
            x = LayerNormalization(name=f"norm_ff_{i}")(x)

        sequence_repr = GlobalAveragePooling1D(name="sequence_pooling")(x)
        sequence_repr_with_baseline = Concatenate(name="sequence_baseline_concat")([sequence_repr, baseline_input])

        # Supervised outlier head (trained in DeltaTrainedKerasModel from robust delta labels)
        outlier_logit = Dense(1, activation=None, name="outlier_logit")(sequence_repr_with_baseline)
        outlier_prob = tf.keras.layers.Activation("sigmoid", name="outlier_prob")(outlier_logit)

        # (Optional) keep spike indicators for visibility (not required for routing anymore)
        if hasattr(self, "spike_features") and self.spike_features:
            spike_features_last = Lambda(
                lambda x: x[:, -1, -len(self.spike_features):],
                name="spike_features",
            )(sequence_input)
            spike_indicators = Dense(
                len(self.target_columns),
                activation="sigmoid",
                name="per_target_spike_indicators",
            )(spike_features_last)
        else:
            spike_indicators = None

        # Router keys
        total_experts = self.config["num_experts"] + self.config["num_spike_experts"]
        key_dim = self.config["prototype_key_dim"]
        router_query = Dense(key_dim, activation=None, name="router_query")(sequence_repr_with_baseline)

        expert_key_table = Embedding(
            input_dim=total_experts,
            output_dim=key_dim,
            embeddings_initializer=tf.keras.initializers.Orthogonal(gain=0.5),  # Changed from RandomNormal to Orthogonal
            name="expert_key_table",
        )

        expert_indices = Lambda(lambda q: tf.range(total_experts), name="expert_indices")(router_query)
        expert_keys_matrix = expert_key_table(expert_indices)  # [E, key_dim]

        query_norm = tf.nn.l2_normalize(router_query, axis=-1)
        keys_norm = tf.nn.l2_normalize(expert_keys_matrix, axis=-1)
        router_logits = tf.matmul(query_norm, keys_norm, transpose_b=True)  # [B, E]
        
        # EMERGENCY FIX: Scale down logits to reduce extreme confidence
        # Normalized dot product gives [-1, 1], scale to [-0.1, 0.1]
        router_logits = router_logits * 0.1  # Reduce magnitude 10x

        temperature = self.config.get("router_temperature", 1.0)
        router_logits_scaled = router_logits / temperature

        # Spike-expert mask
        spike_expert_mask = tf.concat(
            [
                tf.zeros([tf.shape(router_logits_scaled)[0], self.config["num_experts"]]),
                tf.ones([tf.shape(router_logits_scaled)[0], self.config["num_spike_experts"]]),
            ],
            axis=1,
        )

        # Routing bias driven by SUPERVISED outlier probability
        routing_strength = self.config.get("spike_routing_strength", 1.0)
        spike_routing_bias = outlier_prob * spike_expert_mask * routing_strength  # [B,E]

        adjusted_router_logits = router_logits_scaled + spike_routing_bias
        adjusted_router_logits = GaussianNoise(
            self.config.get("router_noise_std", 0.02),
            name="router_train_noise",
        )(adjusted_router_logits)

        router_probs = Softmax(name="router_probs")(adjusted_router_logits)

        # ===== ROUTER Z-LOSS: Prevent logit instability =====
        # Penalize large log-sum-exp of router logits to maintain numerical stability
        z_loss_weight = self.config.get("router_z_loss_weight", 0.0)
        if z_loss_weight > 0:
            # Compute log-sum-exp of router logits (before noise and bias)
            lse = tf.reduce_logsumexp(router_logits_scaled, axis=-1)  # [B]
            # Square it and average across batch
            z_loss = tf.reduce_mean(tf.square(lse))
            model.add_loss(z_loss_weight * z_loss)
            model.add_metric(z_loss, name="router_z_loss")
            model.add_metric(tf.reduce_mean(lse), name="router_lse_mean")
            model.add_metric(tf.reduce_max(lse), name="router_lse_max")

        # Experts output DELTA only
        expert_outputs = []
        for i in range(total_experts):
            expert_type = "spike" if i >= self.config["num_experts"] else "regular"
            expert_dim = self.config["spike_expert_capacity"] if expert_type == "spike" else self.config["expert_dim"]

            h = Dense(expert_dim, activation="relu", name=f"expert_{expert_type}_{i}_1")(sequence_repr_with_baseline)
            h = Dropout(self.config["dropout"], name=f"expert_{expert_type}_{i}_dropout")(h)

            if self.config.get("use_probabilistic", False):
                out = Dense(len(self.target_columns) * 4, name=f"expert_{expert_type}_{i}_out")(h)
            else:
                out = Dense(len(self.target_columns), name=f"expert_{expert_type}_{i}_out")(h)

            expert_outputs.append(out)

        expert_stack = tf.stack(expert_outputs, axis=1)  # [B, E, D]
        router_probs_expanded = tf.expand_dims(router_probs, -1)  # [B, E, 1]

        # ===== PHASE 3: DIVERSITY LOSS =====
        diversity_loss_tensor = None
        expert_output_correlation_tensor = None
        if self.config.get("use_diversity_regularization", False):
            sample_frac = self.config.get("diversity_sample_fraction", 0.5)
            batch_size = tf.shape(expert_stack)[0]
            sample_size = tf.cast(tf.cast(batch_size, tf.float32) * sample_frac, tf.int32)
            sample_size = tf.maximum(sample_size, 1)

            indices = tf.random.shuffle(tf.range(batch_size))[:sample_size]
            sampled_outputs = tf.gather(expert_stack, indices)  # [S, E, D]

            correlations = []
            for i in range(total_experts):
                for j in range(i + 1, total_experts):
                    out_i = sampled_outputs[:, i, :]
                    out_j = sampled_outputs[:, j, :]

                    out_i_norm = tf.nn.l2_normalize(out_i, axis=-1)
                    out_j_norm = tf.nn.l2_normalize(out_j, axis=-1)
                    cos_sim = tf.reduce_sum(out_i_norm * out_j_norm, axis=-1)  # [S]
                    correlations.append(tf.reduce_mean(cos_sim))

            if correlations:
                diversity_loss_tensor = tf.reduce_mean(correlations)
                expert_output_correlation_tensor = diversity_loss_tensor

        # Weighted expert sum -> delta output
        delta_output = tf.reduce_sum(expert_stack * router_probs_expanded, axis=1)

        # Model outputs: (delta_pred, outlier_prob)
        model = Model([sequence_input, baseline_input], [delta_output, outlier_prob], name="HybridMoE_DeltaOnly")

        # Spike sanity metrics (if present)
        if spike_indicators is not None:
            model.add_metric(tf.reduce_mean(spike_indicators), name="spike_ind_mean")
            model.add_metric(tf.reduce_max(spike_indicators), name="spike_ind_max")

        # Dynamic weights
        self._lb_w = tf.Variable(
            self.config.get("load_balance_weight_start", 0.001),
            trainable=False,
            dtype=tf.float32,
            name="lb_w",
        )
        self._comp_w = tf.Variable(
            self.config.get("compactness_start", 0.0025),
            trainable=False,
            dtype=tf.float32,
            name="comp_w",
        )

        # Add diversity loss
        if diversity_loss_tensor is not None:
            model.add_loss(self.config["diversity_coef"] * diversity_loss_tensor)
            model.add_metric(diversity_loss_tensor, name="diversity_loss")
            model.add_metric(expert_output_correlation_tensor, name="expert_output_correlation")

        # Phase 1 losses
        if self.config.get("use_importance_loss", True):
            p_mean = tf.reduce_mean(router_probs, axis=0)  # [E]
            importance_loss = total_experts * tf.reduce_sum(tf.square(p_mean))
            model.add_loss(self._lb_w * self.config.get("importance_weight", 0.01) * importance_loss)
            model.add_metric(importance_loss, name="importance_loss")

        if self.config.get("use_load_loss", True):
            top1 = tf.argmax(router_probs, axis=1)  # [B]
            counts = tf.reduce_sum(tf.one_hot(top1, total_experts, dtype=tf.float32), axis=0)  # [E]
            load = counts / (tf.reduce_sum(counts) + 1e-8)
            load_loss = total_experts * tf.reduce_sum(tf.square(load))
            model.add_loss(self._lb_w * self.config.get("load_weight", 0.01) * load_loss)
            model.add_metric(load_loss, name="load_loss")
            
            # Additional auxiliary load loss: penalize standard deviation of usage
            # This directly targets the variance in expert usage
            usage_std = tf.math.reduce_std(router_probs, axis=0)  # [E]
            aux_load_loss = tf.reduce_mean(usage_std) * total_experts
            model.add_loss(self._lb_w * 0.05 * aux_load_loss)  # Moderate weight
            model.add_metric(aux_load_loss, name="aux_load_loss")

        if self.config.get("use_capacity_enforcement", True):
            batch_size = tf.shape(router_probs)[0]
            capacity = tf.cast(
                tf.math.ceil(self.config["capacity_factor"] * tf.cast(batch_size, tf.float32) / total_experts),
                tf.float32,
            )
            top1 = tf.argmax(router_probs, axis=1)
            counts = tf.reduce_sum(tf.one_hot(top1, total_experts, dtype=tf.float32), axis=0)
            overflow = tf.maximum(0.0, counts - capacity)
            overflow_rate = tf.reduce_sum(overflow) / (tf.cast(batch_size, tf.float32) + 1e-8)
            model.add_loss(self._lb_w * self.config.get("overflow_penalty", 0.01) * overflow_rate)
            model.add_metric(overflow_rate, name="overflow_rate")

        # Phase 2: compactness + separation
        assigned_experts = tf.argmax(router_probs, axis=1)
        assigned_keys = tf.gather(expert_keys_matrix, assigned_experts)
        compactness_loss = tf.reduce_mean(tf.reduce_sum(tf.square(router_query - assigned_keys), axis=1))
        model.add_loss(self._comp_w * compactness_loss)
        model.add_metric(compactness_loss, name="compactness_loss")

        keys_norm2 = tf.nn.l2_normalize(expert_keys_matrix, axis=-1)
        sim_mat = tf.matmul(keys_norm2, keys_norm2, transpose_b=True)  # [E,E]
        mask = tf.linalg.band_part(tf.ones_like(sim_mat), 0, -1) - tf.linalg.band_part(tf.ones_like(sim_mat), 0, 0)
        sims = tf.boolean_mask(sim_mat, tf.cast(mask, tf.bool))
        margin = self.config["separation_margin"]
        separation_loss = tf.reduce_mean(tf.maximum(0.0, sims - (-margin)))
        model.add_loss(self.config["separation_coef"] * separation_loss)
        model.add_metric(separation_loss, name="separation_loss")
        model.add_metric(tf.reduce_mean(sims), name="key_similarity_mean")
        model.add_metric(tf.reduce_max(sims), name="key_similarity_max")

        # Entropy regularization - EMERGENCY FIX: Massively increase weight to force diversity
        router_entropy = -tf.reduce_mean(tf.reduce_sum(router_probs * tf.math.log(router_probs + 1e-8), axis=1))
        frac = float(self.config.get("entropy_target_frac", 0.85))
        entropy_target = frac * float(np.log(total_experts))
        
        # CRITICAL: Use squared penalty when entropy is below target
        entropy_deficit = tf.nn.relu(entropy_target - router_entropy)  # Only penalize when below target
        entropy_loss = tf.square(entropy_deficit)
        
        # EMERGENCY FIX: Increase weight from 0.001 to 2.0 (2000x increase!)
        entropy_weight = 2.0  # Was: self.config.get("router_z_loss_weight", 0.001)
        model.add_loss(entropy_weight * entropy_loss)
        model.add_metric(entropy_loss, name="entropy_loss")
        model.add_metric(router_entropy, name="router_entropy")
        model.add_metric(entropy_target, name="entropy_target")
        model.add_metric(entropy_deficit, name="entropy_deficit")

        for i in range(total_experts):
            model.add_metric(tf.reduce_mean(router_probs[:, i]), name=f"expert_{i}_usage")

        # Router probe
        self._router_probe = Model(
            inputs=[sequence_input, baseline_input],
            outputs=router_probs,
            name="router_probe",
        )

        print("✅ Model built: DELTA-ONLY + supervised outlier routing (no baseline softening possible)")
        return model

    def create_moe_metrics_callback(self):
        class MoEMetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self, moe_metrics, log_every_n=1, warn_on_collapse=True):
                super().__init__()
                self.moe_metrics = moe_metrics
                self.log_every_n = int(log_every_n)
                self.warn_on_collapse = bool(warn_on_collapse)
                self.trainer_ref = None
                self.X_val = None
                self.baselines_val = None

            def on_epoch_end(self, epoch, logs=None):
                if epoch % self.log_every_n != 0:
                    return

                n = min(4096, self.X_val.shape[0]) if self.X_val is not None else 4096
                Xs = self.X_val[:n]
                Bs = self.baselines_val[:n]

                probs = self.trainer_ref._router_probe.predict([Xs, Bs], batch_size=256, verbose=0)

                self.moe_metrics.reset_epoch()
                self.moe_metrics.expert_counts += np.sum(probs, axis=0)
                self.moe_metrics.total_samples += probs.shape[0]

                metrics = self.moe_metrics.compute_epoch_metrics()
                if logs is not None:
                    for k, v in metrics.items():
                        logs[f"moe_{k}"] = float(v)

                self.moe_metrics.print_summary(metrics, epoch=epoch)

                if self.warn_on_collapse:
                    health = self.moe_metrics.get_health_status(metrics)
                    if health == "critical":
                        print("\n" + "!" * 70)
                        print("CRITICAL: MoE COLLAPSE DETECTED")
                        print("Consider:")
                        print("  - Increase load_balance_weight_final")
                        print("  - Increase router_temperature (more exploration)")
                        print("  - Reduce compactness early (too much locking)")
                        print("!" * 70 + "\n")

        return MoEMetricsCallback(
            self.moe_metrics,
            log_every_n=self.config.get("log_moe_metrics_every_n_epochs", 1),
            warn_on_collapse=self.config.get("warn_on_collapse", True),
        )

    def train_simple(self, X_train, baselines_train, y_train, X_val, baselines_val, y_val, metadata_train, metadata_val):
        print("\n" + "=" * 70)
        print("TRAINING WITH MOE IMPROVEMENTS (DELTA-ONLY + OUTLIER SUPERVISION)")
        print("=" * 70)

        # Model outputs (delta_pred, outlier_prob)
        net = self.build_model_with_phase2()

        # Wrap for delta-only training + band-loss + focal outlier supervision
        model = DeltaTrainedKerasModel(
            net=net,
            target_columns=self.target_columns,
            use_probabilistic=self.config.get("use_probabilistic", False),
            huber_delta=float(self.config.get("huber_delta", 1.0)),
            band=self.config.get("band_per_target", None),
            band_weight=float(self.config.get("band_weight", 0.0)),
            outlier_gamma=float(self.config.get("outlier_gamma", 2.0)),
            outlier_alpha=float(self.config.get("outlier_alpha", 0.25)),
            outlier_weight=float(self.config.get("outlier_weight", 0.5)),
            spike_z_thr=float(self.config.get("spike_z_thr", 2.75)),
            spike_min_pos_frac=float(self.config.get("spike_min_pos_frac", 0.03)),
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer)

        train_gen = TimeAwareBatchGenerator(
            X_train,
            baselines_train,
            y_train,
            metadata_train,
            batch_size=self.config["batch_size"],
            shuffle_buckets=self.config["shuffle_buckets"],
            stride=self.config.get("batch_stride", self.config["batch_size"] // 4),
        )
        val_gen = TimeAwareBatchGenerator(
            X_val,
            baselines_val,
            y_val,
            metadata_val,
            batch_size=self.config["batch_size"],
            shuffle_buckets=False,
            stride=self.config.get("batch_stride", self.config["batch_size"] // 4),
        )

        steps = train_gen.steps_per_epoch()
        val_steps = max(1, val_gen.steps_per_epoch() // 2)

        moe_callback = self.create_moe_metrics_callback()
        moe_callback.X_val = X_val
        moe_callback.baselines_val = baselines_val
        moe_callback.trainer_ref = self

        class Phase12Callback(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer

            def on_epoch_begin(self, epoch, logs=None):
                lb_weight = self.trainer.get_load_balance_weight(epoch)
                comp_weight = self.trainer.get_compactness_weight(epoch)
                self.trainer._lb_w.assign(lb_weight)
                self.trainer._comp_w.assign(comp_weight)

                print(f"\n[Phase 1+2] Epoch {epoch}:")
                print(f"  load_balance_weight = {lb_weight:.4f}")
                print(f"  compactness_weight = {comp_weight:.4f}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                mode="min",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=6,
                factor=0.65,
                min_lr=1e-6,
                mode="min",
                verbose=1,
            ),
            # ValR2Callback expects absolute predictions; wrapper model.call returns y_hat, so predict_delta=False.
            ValR2Callback(X_val, baselines_val, y_val, predict_delta=False),
            moe_callback,
            Phase12Callback(self),
        ]

        print(f"\nTraining for up to {self.config['phase1_epochs']} epochs...")
        history = model.fit(
            train_gen.generate(),
            steps_per_epoch=steps,
            validation_data=val_gen.generate(),
            validation_steps=val_steps,
            epochs=self.config["phase1_epochs"],
            callbacks=callbacks,
            verbose=1,
        )

        return model, history


def main():
    print("\n" + "=" * 70)
    print("MoE IMPROVEMENTS INTEGRATION (DELTA-ONLY + OUTLIERS)")
    print("=" * 70)
    trainer = MoEImprovedTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
