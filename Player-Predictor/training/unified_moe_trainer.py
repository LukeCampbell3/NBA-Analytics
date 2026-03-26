#!/usr/bin/env python3
"""
Unified Hybrid Spike-Driver MoE Trainer with Expert Collapse Prevention

This file combines:
- hybrid_spike_moe_trainer.py (data preparation, spike features)
- integrate_moe_improvements.py (expert collapse fixes)

KEY ANTI-COLLAPSE FEATURES:
1. Router temperature: 5.0 (high exploration)
2. Strong load balance: 0.01 → 0.1 (10x stronger)
3. Router Z-Loss: 0.001 (prevents logit explosion)
4. Orthogonal expert key initialization
5. Delayed compactness schedule (starts epoch 10)
6. Auxiliary variance penalty
7. Diversity regularization
8. Expert replay buffer

Expected: All 11 experts at 5-15% usage each
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization, Embedding,
    Lambda, Concatenate, Add, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Softmax, GaussianNoise
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import base trainer components
from improved_baseline_trainer import (
    ImprovedBaselineTrainer,
    TimeAwareBatchGenerator,
    ValR2Callback
)
from moe_metrics import MoEMetricsTracker


# ============================================================================
# ROUTER NOISE LAYER (TF2-safe)
# ============================================================================

class RouterNoise(tf.keras.layers.Layer):
    """TF2-safe router noise injection (replaces tf.cond(learning_phase()))"""
    def __init__(self, stddev=0.05, **kwargs):
        super().__init__(**kwargs)
        self.stddev = float(stddev)
    
    def call(self, x, training=None):
        if training:
            return x + tf.random.normal(tf.shape(x), stddev=self.stddev)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _huber(delta=1.0):
    return tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)


def _iter_training_csvs(player_dir: Path):
    """Yield candidate processed season files for a player directory."""
    patterns = ("*_processed.csv", "*_processed_processed.csv", "*.csv")
    seen = set()
    for pattern in patterns:
        for file_path in player_dir.glob(pattern):
            resolved = str(file_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            yield file_path


def _contains_training_csvs(data_dir: Path) -> bool:
    """Return True when a directory looks like a player-organized training root."""
    if not data_dir.exists() or not data_dir.is_dir():
        return False
    for player_dir in data_dir.iterdir():
        if not player_dir.is_dir():
            continue
        if any(_iter_training_csvs(player_dir)):
            return True
    return False


def _resolve_training_data_dir() -> Path:
    """
    Locate the processed training dataset across legacy and current repo layouts.

    Expected layout:
      <data_root>/<Player_Name>/<season>_processed.csv
      <data_root>/<Player_Name>/<season>_processed_processed.csv
    """
    module_root = Path(__file__).resolve().parent.parent
    search_roots = [
        module_root,
        module_root.parent,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    ]
    candidate_suffixes = [
        Path("Data"),
        Path("Data-Proc"),
        Path("Data-Proc-OG"),
        Path("Data-org"),
        Path("data"),
        Path("data") / "processed",
        Path("data copy"),
        Path("data copy") / "processed",
    ]
    env_candidates = [
        os.environ.get("PLAYER_PREDICTOR_DATA_DIR"),
        os.environ.get("NBA_ANALYTICS_DATA_DIR"),
    ]

    candidates = []
    for raw_path in env_candidates:
        if raw_path:
            candidates.append(Path(raw_path).expanduser())
    for root in search_roots:
        candidates.extend(root / suffix for suffix in candidate_suffixes)

    seen = set()
    existing_dirs = []
    searched_dirs = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate
        dedupe_key = str(resolved).lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        searched_dirs.append(resolved)
        if not resolved.exists() or not resolved.is_dir():
            continue
        existing_dirs.append(resolved)
        if _contains_training_csvs(resolved):
            return resolved

    searched = ", ".join(str(path) for path in searched_dirs)
    if existing_dirs:
        existing = ", ".join(str(path) for path in existing_dirs)
        raise FileNotFoundError(
            "Found candidate data directories, but none contained player season CSVs. "
            f"Existing directories: {existing}. Searched: {searched}. "
            "Expected layout is <data_root>/<Player>/<season>_processed.csv. "
            "Set PLAYER_PREDICTOR_DATA_DIR to the correct folder or rebuild Data-Proc with "
            "scripts/update_nba_processed_data.py."
        )
    raise FileNotFoundError(
        f"No data directory found. Searched: {searched}. "
        "Set PLAYER_PREDICTOR_DATA_DIR to your processed data root if it lives elsewhere."
    )


def focal_bce(y_true, y_prob, gamma=2.0, alpha=0.25, eps=1e-7):
    """Binary focal loss for outlier detection"""
    y_true = tf.cast(y_true, tf.float32)
    y_prob = tf.clip_by_value(tf.cast(y_prob, tf.float32), eps, 1.0 - eps)
    p_t = y_true * y_prob + (1.0 - y_true) * (1.0 - y_prob)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
    return loss


def tf_median(x, axis=0, keepdims=True):
    """Pure-TF median using sort + gather"""
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
    """Pure-TF quantile for 1D tensor"""
    x = tf.sort(tf.reshape(tf.cast(x, tf.float32), [-1]))
    n = tf.shape(x)[0]
    idx = tf.cast(tf.round(q * tf.cast(n - 1, tf.float32)), tf.int32)
    idx = tf.clip_by_value(idx, 0, n - 1)
    return tf.gather(x, idx)


def robust_spike_labels_from_residuals(residuals, z_thr=2.75, min_pos_frac=0.03, eps=1e-6):
    """Generate spike labels from residuals using robust MAD z-score"""
    res = tf.cast(residuals, tf.float32)
    med = tf_median(res, axis=0, keepdims=True)
    mad = tf_median(tf.abs(res - med), axis=0, keepdims=True) + eps
    robust_z = tf.abs(res - med) / (1.4826 * mad)
    spike_per_target = tf.cast(robust_z > z_thr, tf.float32)
    spike_any = tf.cast(tf.reduce_any(spike_per_target > 0.0, axis=1, keepdims=True), tf.float32)
    
    pos_rate = tf.reduce_mean(spike_any)
    
    def _quantile_fallback():
        scores = tf.reduce_max(robust_z, axis=1)
        cutoff = tf_quantile_1d(scores, 1.0 - float(min_pos_frac))
        spike_any_fb = tf.cast(scores >= cutoff, tf.float32)[:, None]
        return spike_any_fb, spike_per_target
    
    spike_any, spike_per_target = tf.cond(
        pos_rate < tf.constant(0.01, tf.float32),
        true_fn=_quantile_fallback,
        false_fn=lambda: (spike_any, spike_per_target),
    )
    
    return spike_any, spike_per_target, robust_z


# ============================================================================
# DELTA-TRAINED KERAS MODEL
# ============================================================================

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
        variance_weight=0.0,       # weight for variance encouragement loss
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

        self.variance_weight = float(variance_weight)

        # Trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.delta_loss_tracker = tf.keras.metrics.Mean(name="delta_loss")
        self.band_loss_tracker = tf.keras.metrics.Mean(name="band_loss")
        self.outlier_loss_tracker = tf.keras.metrics.Mean(name="outlier_loss")
        self.abs_mae_tracker = tf.keras.metrics.Mean(name="abs_mae")
        self.spike_rate_tracker = tf.keras.metrics.Mean(name="spike_rate")
        self.var_loss_tracker = tf.keras.metrics.Mean(name="var_loss")


    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.delta_loss_tracker,
            self.band_loss_tracker,
            self.outlier_loss_tracker,
            self.abs_mae_tracker,
            self.spike_rate_tracker,
            self.var_loss_tracker,
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
        delta_pred: [B, T] deterministic or [B, T*4] probabilistic
        Returns: [B, T] means
        """
        # Shape-based check works in graph mode (no Python bool tracing issue)
        pred_dim = delta_pred.shape[-1] if delta_pred.shape[-1] is not None else tf.shape(delta_pred)[-1]
        if self.use_probabilistic and (delta_pred.shape[-1] is None or delta_pred.shape[-1] > self.T):
            indices = tf.range(0, self.T * 4, 4)
            return tf.gather(delta_pred, indices, axis=1)
        return delta_pred

    def call(self, inputs, training=False):
        x_seq, x_base, _ = self._split_inputs(inputs)
        outputs = self.net([x_seq, x_base], training=training)
        delta_pred = outputs[0]
        delta_means = self._extract_delta_means(delta_pred)
        y_hat = tf.cast(x_base, tf.float32) + tf.cast(delta_means, tf.float32)
        return y_hat

    def train_step(self, data):
        x, y_true = data
        x_seq, x_base, _ = self._split_inputs(x)

        y_true = tf.cast(y_true, tf.float32)
        x_base = tf.cast(x_base, tf.float32)

        with tf.GradientTape() as tape:
            outputs = self.net([x_seq, x_base], training=True)
            delta_pred = outputs[0]
            outlier_prob = outputs[1]
            delta_means = self._extract_delta_means(delta_pred)

            delta_true = y_true - x_base

            # Main delta loss (MSE — encourages larger predictions unlike Huber)
            per_elem = tf.square(delta_true - delta_means)  # [B,T]
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

            # Variance encouragement: penalize when predicted delta std is
            # much smaller than true delta std. This prevents the model from
            # collapsing to near-zero corrections.
            var_loss = tf.constant(0.0)
            if self.variance_weight > 0.0:
                pred_var = tf.math.reduce_variance(delta_means, axis=0)  # [T]
                true_var = tf.math.reduce_variance(delta_true, axis=0)   # [T]
                # Penalize ratio: want pred_var / true_var >= 0.7
                # Use squared hinge: penalty when ratio < target_ratio
                ratio = pred_var / (true_var + 1e-6)  # [T]
                target_ratio = 0.7
                shortfall = tf.maximum(0.0, target_ratio - ratio)  # [T]
                var_loss = tf.reduce_mean(tf.square(shortfall))

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

            total = (delta_loss 
                     + (self.band_weight * band_loss) 
                     + (self.outlier_weight * outlier_loss) 
                     + (self.variance_weight * var_loss)
                     + reg_loss)

        grads = tape.gradient(total, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        self.loss_tracker.update_state(total)
        self.delta_loss_tracker.update_state(delta_loss)
        self.band_loss_tracker.update_state(band_loss)
        self.outlier_loss_tracker.update_state(outlier_loss)
        self.abs_mae_tracker.update_state(abs_mae)
        self.var_loss_tracker.update_state(var_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_true = data
        x_seq, x_base, _ = self._split_inputs(x)

        y_true = tf.cast(y_true, tf.float32)
        x_base = tf.cast(x_base, tf.float32)

        outputs = self.net([x_seq, x_base], training=False)
        delta_pred = outputs[0]
        outlier_prob = outputs[1]
        delta_means = self._extract_delta_means(delta_pred)

        delta_true = y_true - x_base
        per_elem = tf.square(delta_true - delta_means)
        delta_loss = tf.reduce_mean(per_elem)

        y_hat = x_base + delta_means
        abs_mae = tf.reduce_mean(tf.abs(y_true - y_hat))

        band_loss = 0.0
        if self.band is not None and self.band_weight > 0.0:
            err = tf.abs(y_true - y_hat)
            band = tf.reshape(self.band, [1, self.T])
            band_violation = tf.nn.relu(err - band)
            band_loss = tf.reduce_mean(band_violation)

        var_loss = tf.constant(0.0)
        if self.variance_weight > 0.0:
            pred_var = tf.math.reduce_variance(delta_means, axis=0)
            true_var = tf.math.reduce_variance(delta_true, axis=0)
            ratio = pred_var / (true_var + 1e-6)
            target_ratio = 0.7
            shortfall = tf.maximum(0.0, target_ratio - ratio)
            var_loss = tf.reduce_mean(tf.square(shortfall))

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
        total = (delta_loss 
                 + (self.band_weight * band_loss) 
                 + (self.outlier_weight * outlier_loss) 
                 + (self.variance_weight * var_loss)
                 + reg_loss)

        self.loss_tracker.update_state(total)
        self.delta_loss_tracker.update_state(delta_loss)
        self.band_loss_tracker.update_state(band_loss)
        self.outlier_loss_tracker.update_state(outlier_loss)
        self.abs_mae_tracker.update_state(abs_mae)
        self.var_loss_tracker.update_state(var_loss)

        return {m.name: m.result() for m in self.metrics}





# ============================================================================
# EXPERT REPLAY BUFFER (Phase 4)
# ============================================================================

class ExpertReplayBuffer:
    """Stores samples per expert for stability"""
    def __init__(self, num_experts, buffer_size_per_expert=500):
        self.num_experts = num_experts
        self.buffer_size = buffer_size_per_expert
        self.buffers = {i: {'X': [], 'baselines': [], 'y': []} for i in range(num_experts)}
    
    def add_samples(self, X, baselines, y, expert_assignments, expert_outputs):
        """Add samples to expert buffers (FIFO)"""
        for i in range(self.num_experts):
            mask = expert_assignments == i
            if np.any(mask):
                self.buffers[i]['X'].extend(X[mask])
                self.buffers[i]['baselines'].extend(baselines[mask])
                self.buffers[i]['y'].extend(y[mask])
                
                # Keep only last buffer_size samples
                if len(self.buffers[i]['X']) > self.buffer_size:
                    self.buffers[i]['X'] = self.buffers[i]['X'][-self.buffer_size:]
                    self.buffers[i]['baselines'] = self.buffers[i]['baselines'][-self.buffer_size:]
                    self.buffers[i]['y'] = self.buffers[i]['y'][-self.buffer_size:]
    
    def sample_replay(self, batch_size, expert_distribution=None):
        """Sample from buffers according to expert distribution"""
        if expert_distribution is None:
            expert_distribution = np.ones(self.num_experts) / self.num_experts
        
        samples_per_expert = (expert_distribution * batch_size).astype(int)
        X_replay, b_replay, y_replay = [], [], []
        
        for i in range(self.num_experts):
            n_samples = samples_per_expert[i]
            if n_samples > 0 and len(self.buffers[i]['X']) > 0:
                indices = np.random.choice(len(self.buffers[i]['X']), 
                                          min(n_samples, len(self.buffers[i]['X'])), 
                                          replace=False)
                X_replay.extend([self.buffers[i]['X'][j] for j in indices])
                b_replay.extend([self.buffers[i]['baselines'][j] for j in indices])
                y_replay.extend([self.buffers[i]['y'][j] for j in indices])
        
        if len(X_replay) > 0:
            return np.array(X_replay), np.array(b_replay), np.array(y_replay)
        return None, None, None


# ============================================================================
# UNIFIED MOE TRAINER
# ============================================================================

class UnifiedMoETrainer(ImprovedBaselineTrainer):
    """
    Unified trainer combining spike features + expert collapse prevention
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.replay_buffer = None
        
        # Set deviation bands per target (in SCALED space now — ~0.5 std)
        if self.config.get("band_per_target", None) is None:
            band = []
            for c in self.target_columns:
                band.append(0.5)  # 0.5 std in normalized space
            self.config["band_per_target"] = band
        
        # CRITICAL: Anti-collapse configuration
        self.config.update({
            # Metrics/logging
            "log_moe_metrics_every_n_epochs": 1,
            "warn_on_collapse": True,
            "dead_threshold": 0.02,
            "collapse_warning_threshold": 0.30,
            "entropy_warning_threshold": 0.50,
            
            # PHASE 1: ANTI-COLLAPSE (minimal — prediction quality is priority)
            "use_load_balancing": True,
            "load_balance_schedule": "ramp",
            "load_balance_weight_start": 0.001,
            "load_balance_weight_mid": 0.005,
            "load_balance_weight_final": 0.01,
            "load_balance_ramp_epochs": [3, 15],
            "use_capacity_enforcement": True,
            "capacity_factor": 2.0,
            "overflow_penalty": 0.001,
            "track_overflow": True,
            "use_importance_loss": True,
            "use_load_loss": True,
            "importance_weight": 0.005,
            "load_weight": 0.005,
            
            # ROUTER EXPLORATION (conservative)
            "router_temperature": 1.5,
            "router_noise_std": 0.02,
            "router_z_loss_weight": 0.0001,
            
            # PHASE 2: SPECIALIZATION (delayed)
            "use_prototype_experts": True,
            "prototype_key_dim": 256,
            "compactness_coef": 0.001,
            "separation_coef": 0.0005,
            "separation_margin": 0.5,
            "compactness_schedule": "ramp",
            "compactness_start": 0.0,
            "compactness_final": 0.001,
            "compactness_ramp_epochs": [10, 25],
            
            # PHASE 3: DIVERSITY (very light)
            "use_diversity_regularization": True,
            "diversity_coef": 0.00005,
            "diversity_sample_fraction": 0.5,
            
            # PHASE 4: STABILITY
            "use_expert_replay": False,
            "replay_buffer_size": 500,
            "replay_mix_fraction": 0.2,
            "consistency_coef": 0.005,
            "consistency_start_epoch": 10,
            
            # Entropy regularization
            "entropy_target_frac": 0.85,
            
            # Outlier supervision (very light)
            "spike_z_thr": 2.75,
            "spike_min_pos_frac": 0.03,
            "outlier_gamma": 2.0,
            "outlier_alpha": 0.25,
            "outlier_weight": 0.0,
            
            # Band training (disabled — was hurting variance)
            "band_weight": 0.0,
            
            # Huber delta — smaller = more MSE-like = more variance
            "huber_delta": 0.5,
            
            # Variance encouragement (penalize low prediction variance)
            "variance_weight": 0.0,
            
            # Training params (override parent)
            "epochs": 120,
            "lr": 0.001,
            "patience": 25,
            "batch_size": 128,
            
            # Simplified architecture to reduce overfitting
            "num_experts": 4,
            "num_spike_experts": 2,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.2,
            
            # Use simple LSTM model instead of MoE (better for small datasets)
            "use_simple_model": True,
            
            # Use stacked LSTM + GBM meta-learner (best accuracy)
            "use_stacked_model": True,
        })
        
        # MoE configuration with ANTI-COLLAPSE defaults (AFTER config update)
        total_experts = self.config.get("num_experts", 8) + self.config.get("num_spike_experts", 3)
        
        self.moe_metrics = MoEMetricsTracker(
            num_experts=total_experts,
            num_spike_experts=self.config.get("num_spike_experts", 3),
            dead_threshold=0.02,
        )
        
        print("\n" + "="*80)
        print("UNIFIED MOE TRAINER - EXPERT COLLAPSE PREVENTION ENABLED")
        print("="*80)
        print(f"Total experts: {total_experts}")
        print(f"Router temperature: {self.config['router_temperature']}")
        print(f"Load balance: {self.config['load_balance_weight_start']} → {self.config['load_balance_weight_final']}")
        print(f"Router Z-Loss: {self.config['router_z_loss_weight']}")
        print(f"Compactness: delayed until epoch {self.config['compactness_ramp_epochs'][0]}")
        print(f"Capacity factor: {self.config['capacity_factor']}")
        print(f"Band weight: {self.config['band_weight']}, Outlier weight: {self.config['outlier_weight']}")
        print(f"Diversity coef: {self.config['diversity_coef']}")
        print(f"Variance weight: {self.config['variance_weight']}")
        print(f"N_layers: {self.config['n_layers']}, d_model: {self.config['d_model']}")
        print(f"Num experts: {self.config['num_experts']} + {self.config['num_spike_experts']} spike")
        print(f"LR: {self.config['lr']}, Batch: {self.config['batch_size']}, Epochs: {self.config['epochs']}")
        print("NOTE: Baselines & targets now in same scaled space (scaler_y)")
        print("="*80 + "\n")
        
        if self.config.get("use_expert_replay", False):
            self.replay_buffer = ExpertReplayBuffer(
                num_experts=total_experts,
                buffer_size_per_expert=self.config["replay_buffer_size"],
            )
            print("Expert replay buffer initialized\n")
        else:
            print("Expert replay buffer disabled\n")
        
        # Epoch-aware weights (graph-safe, updated by callback each epoch)
        self.lb_weight_var = tf.Variable(
            float(self.config["load_balance_weight_start"]),
            trainable=False,
            dtype=tf.float32,
            name="lb_weight_var",
        )
        self.zloss_weight_var = tf.Variable(
            float(self.config["router_z_loss_weight"]),
            trainable=False,
            dtype=tf.float32,
            name="zloss_weight_var",
        )
        print(f"✅ Epoch-aware weight variables initialized")
        print(f"   LB weight: {float(self.lb_weight_var.numpy()):.4f}")
        print(f"   Z-loss weight: {float(self.zloss_weight_var.numpy()):.4f}\n")
    
    def get_load_balance_weight(self, epoch):
        """Ramp load balance weight over training"""
        if self.config["load_balance_schedule"] == "constant":
            return self.config["load_balance_weight_final"]
        
        ramp_epochs = self.config["load_balance_ramp_epochs"]
        start = self.config["load_balance_weight_start"]
        mid = self.config["load_balance_weight_mid"]
        final = self.config["load_balance_weight_final"]
        
        if epoch < ramp_epochs[0]:
            return start
        elif epoch < ramp_epochs[1]:
            progress = (epoch - ramp_epochs[0]) / (ramp_epochs[1] - ramp_epochs[0])
            return start + progress * (mid - start)
        else:
            return final
    
    def get_compactness_weight(self, epoch):
        """Ramp compactness weight (delayed start to avoid early collapse)"""
        if self.config.get("compactness_schedule") == "constant":
            return self.config["compactness_coef"]
        
        ramp_epochs = self.config["compactness_ramp_epochs"]
        start = self.config["compactness_start"]
        final = self.config["compactness_final"]
        
        if epoch < ramp_epochs[0]:
            return start
        elif epoch < ramp_epochs[1]:
            progress = (epoch - ramp_epochs[0]) / (ramp_epochs[1] - ramp_epochs[0])
            return start + progress * (final - start)
        else:
            return final

    def build_simple_lstm_model(self):
        """Build a simple LSTM + Dense model (no MoE) for delta prediction.
        Much better generalization with small datasets (~12K samples)."""
        print("\n🔧 Building simple LSTM model (no MoE)...")
        
        seq_input = Input(shape=(self.config["seq_len"], len(self.feature_columns)), name="sequence_input")
        base_input = Input(shape=(len(self.target_columns),), name="baseline_pred_input")
        
        # LSTM encoder
        x = tf.keras.layers.LSTM(128, return_sequences=True, name="lstm1")(seq_input)
        x = Dropout(0.2)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=False, name="lstm2")(x)
        x = Dropout(0.2)(x)
        
        # Combine with baseline
        combined = Concatenate(name="lstm_baseline_concat")([x, base_input])
        
        # Dense head
        x = Dense(128, activation="relu", name="dense1")(combined)
        x = LayerNormalization(name="ln1")(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu", name="dense2")(x)
        x = Dropout(0.1)(x)
        delta_pred = Dense(len(self.target_columns), activation=None, name="delta_pred")(x)
        
        n_targets = len(self.target_columns)
        
        # Dummy outputs as Lambda layers (Keras-compatible)
        outlier_prob = Lambda(lambda z: tf.zeros([tf.shape(z)[0], 1]), name="outlier_prob_dummy")(delta_pred)
        spike_indicators = Lambda(lambda z: tf.zeros([tf.shape(z)[0], n_targets]), name="spike_ind_dummy")(delta_pred)
        routing_probs = Lambda(lambda z: tf.ones([tf.shape(z)[0], 1]), name="routing_probs_dummy")(delta_pred)
        router_entropy = Lambda(lambda z: tf.zeros([tf.shape(z)[0]]), name="router_entropy_dummy")(delta_pred)
        router_z_loss = Lambda(lambda z: tf.zeros([tf.shape(z)[0]]), name="router_z_loss_dummy")(delta_pred)
        expert_usage = Lambda(lambda z: tf.ones([1]), name="expert_usage_dummy")(delta_pred)
        expert_outputs_stacked = Lambda(lambda z: tf.expand_dims(z, axis=1), name="expert_outputs_dummy")(delta_pred)
        
        model = Model(
            inputs=[seq_input, base_input],
            outputs=[
                delta_pred,
                outlier_prob,
                spike_indicators,
                routing_probs,
                router_entropy,
                router_z_loss,
                expert_usage,
                expert_outputs_stacked
            ],
            name="simple_lstm_model"
        )
        
        print(f"   Simple LSTM model built")
        print(f"   Parameters: {model.count_params():,}")
        return model

    
    def create_hybrid_features(self, df):
        """Create spike-driver features (from hybrid_spike_moe_trainer.py)"""
        print("\n🔧 Creating hybrid spike-driver features...")
        
        # Ensure critical features exist
        df = self._add_missing_critical_features(df)
        
        # Create spike drivers
        df = self._create_spike_drivers(df)
        df = self._create_market_context_features(df)
        
        print(f"✅ Created {len(df.columns)} total features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df[numeric_cols] = df.groupby("Player")[numeric_cols].transform(lambda x: x.ffill().bfill())
            df[numeric_cols] = df[numeric_cols].fillna(0.0)
        return df
    
    def _add_missing_critical_features(self, df):
        """Add missing critical features"""
        required = ['MP', 'FGA', 'AST', 'USG%']
        
        for feat in required:
            if feat not in df.columns:
                if feat == 'MP':
                    df['MP'] = 30.0
                elif feat == 'FGA':
                    # Estimate FGA from PTS if available (rough: PTS / 2)
                    if 'PTS' in df.columns:
                        df['FGA'] = (df['PTS'] / 2.0).clip(lower=3.0)
                    else:
                        df['FGA'] = 10.0
                elif feat == 'AST':
                    df['AST'] = 3.0
                elif feat == 'USG%':
                    df['USG%'] = 20.0
        
        return df

    def _create_market_context_features(self, df):
        """Create stable numeric features from the processed market contract."""
        df = df.copy()
        fetched_raw = df["Market_Fetched_At_UTC"] if "Market_Fetched_At_UTC" in df.columns else pd.Series(pd.NaT, index=df.index)
        date_raw = df["Date"] if "Date" in df.columns else pd.Series(pd.NaT, index=df.index)
        fetched_at = pd.to_datetime(fetched_raw, errors="coerce", utc=True)
        event_dates = pd.to_datetime(date_raw, errors="coerce", utc=True)
        market_age_hours = ((event_dates - fetched_at).dt.total_seconds() / 3600.0).clip(lower=0.0)
        df["Market_Fetched_Age_Hours"] = market_age_hours.fillna(0.0)

        target_specs = [
            ("PTS", "PTS_lag1"),
            ("TRB", "TRB_lag1"),
            ("AST", "AST_lag1"),
        ]

        for target, lag_col in target_specs:
            market_col = f"Market_{target}"
            synth_col = f"Synthetic_Market_{target}"
            source_col = f"Market_Source_{target}"
            books_col = f"Market_{target}_books"
            over_col = f"Market_{target}_over_price"
            under_col = f"Market_{target}_under_price"
            std_col = f"Market_{target}_line_std"
            baseline_col = f"{target}_rolling_avg"

            if market_col not in df.columns:
                df[market_col] = pd.to_numeric(df.get(baseline_col), errors="coerce")
            if synth_col not in df.columns:
                df[synth_col] = pd.to_numeric(df.get(market_col), errors="coerce")
            if source_col not in df.columns:
                df[source_col] = "missing"

            for col in [books_col, over_col, under_col, std_col]:
                if col not in df.columns:
                    df[col] = 0.0

            market_numeric = pd.to_numeric(df[market_col], errors="coerce")
            synth_numeric = pd.to_numeric(df[synth_col], errors="coerce")
            baseline_numeric = pd.to_numeric(df.get(baseline_col), errors="coerce")
            lag_numeric = pd.to_numeric(df.get(lag_col), errors="coerce")
            books_numeric = pd.to_numeric(df.get(books_col), errors="coerce").fillna(0.0)
            over_numeric = pd.to_numeric(df.get(over_col), errors="coerce").fillna(0.0)
            under_numeric = pd.to_numeric(df.get(under_col), errors="coerce").fillna(0.0)
            std_numeric = pd.to_numeric(df.get(std_col), errors="coerce").fillna(0.0)
            source_text = df[source_col].fillna("missing").astype(str).str.lower()

            df[market_col] = market_numeric
            df[synth_col] = synth_numeric
            df[f"{target}_market_gap"] = market_numeric - baseline_numeric
            df[f"Market_{target}_abs_gap"] = np.abs(df[f"{target}_market_gap"])
            df[f"Market_{target}_lag_gap"] = market_numeric - lag_numeric
            df[f"Market_{target}_vs_synth"] = market_numeric - synth_numeric
            df[f"Market_{target}_price_spread"] = np.abs(over_numeric - under_numeric) / 100.0
            df[f"Market_{target}_price_lean"] = (under_numeric - over_numeric) / 100.0
            df[f"Market_{target}_consensus_conf"] = books_numeric / (books_numeric + 3.0)
            df[f"Market_{target}_dispersion_penalty"] = std_numeric / (np.abs(market_numeric) + 1.0)
            df[f"Market_{target}_has_line"] = market_numeric.notna().astype(float)
            df[f"Market_{target}_source_real"] = source_text.eq("real").astype(float)
            df[f"Market_{target}_source_synthetic"] = source_text.eq("synthetic").astype(float)
            df[f"Market_{target}_source_baseline"] = source_text.eq("baseline_fallback").astype(float)
            df[f"Market_{target}_source_missing"] = source_text.eq("missing").astype(float)
            df[f"Market_{target}_source_quality"] = (
                1.00 * df[f"Market_{target}_source_real"]
                + 0.65 * df[f"Market_{target}_source_synthetic"]
                + 0.25 * df[f"Market_{target}_source_baseline"]
            )

        df["Market_Any_Real"] = df[[f"Market_{target}_source_real" for target, _ in target_specs]].max(axis=1)
        df["Market_Any_Synthetic"] = df[[f"Market_{target}_source_synthetic" for target, _ in target_specs]].max(axis=1)
        df["Market_Any_Baseline"] = df[[f"Market_{target}_source_baseline" for target, _ in target_specs]].max(axis=1)
        df["Market_Consensus_Avg"] = df[[f"Market_{target}_consensus_conf" for target, _ in target_specs]].mean(axis=1)
        df["Market_Disagreement_Avg"] = df[[f"Market_{target}_abs_gap" for target, _ in target_specs]].mean(axis=1)

        return df
    
    def _create_spike_drivers(self, df):
        """Create spike detection features"""
        df = df.copy()
        
        # Sort by player and game order (processed CSVs have Game_Index, not Date)
        if 'Date' in df.columns:
            df = df.sort_values(['Player', 'Date']).reset_index(drop=True)
        elif 'Game_Index' in df.columns:
            df = df.sort_values(['Player', 'Game_Index']).reset_index(drop=True)
        else:
            df = df.sort_values(['Player', 'Game_Num']).reset_index(drop=True)
        
        # Minutes played features
        df['MP_rolling_avg'] = df.groupby('Player')['MP'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['MP_trend'] = df.groupby('Player')['MP'].transform(
            lambda x: x.diff().rolling(window=3, min_periods=1).mean()
        )
        df['High_MP_Flag'] = (df['MP'] > df['MP_rolling_avg'] * 1.15).astype(float)
        
        # Shot attempt features
        df['FGA_rolling_avg'] = df.groupby('Player')['FGA'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['FGA_trend'] = df.groupby('Player')['FGA'].transform(
            lambda x: x.diff().rolling(window=3, min_periods=1).mean()
        )
        
        # Assist features
        df['AST_rolling_avg'] = df.groupby('Player')['AST'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['AST_trend'] = df.groupby('Player')['AST'].transform(
            lambda x: x.diff().rolling(window=3, min_periods=1).mean()
        )
        df['AST_variance'] = df.groupby('Player')['AST'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        
        # Usage rate features
        if 'USG%' in df.columns:
            df['USG_rolling_avg'] = df.groupby('Player')['USG%'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            df['USG_AST_ratio'] = df['USG%'] / (df['AST'] + 1)
            df['USG_AST_ratio_trend'] = df.groupby('Player')['USG_AST_ratio'].transform(
                lambda x: x.diff().rolling(window=3, min_periods=1).mean()
            )
        else:
            df['USG_rolling_avg'] = 20.0
            df['USG_AST_ratio'] = 5.0
            df['USG_AST_ratio_trend'] = 0.0

        # Add context features that help distinguish stable continuation from
        # fragile trend and opportunity shifts. These are derived only from the
        # currently available processed data, so they are safe to add now.
        if 'PTS' in df.columns:
            df['PTS_rolling_std'] = df.groupby('Player')['PTS'].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )
        if 'TRB' in df.columns:
            df['TRB_rolling_std'] = df.groupby('Player')['TRB'].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )
        if 'AST' in df.columns:
            df['AST_rolling_std'] = df.groupby('Player')['AST'].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )

        df['PTS_resid_lag'] = df['PTS_lag1'] - df['PTS_rolling_avg']
        df['TRB_resid_lag'] = df['TRB_lag1'] - df['TRB_rolling_avg']
        df['AST_resid_lag'] = df['AST_lag1'] - df['AST_rolling_avg']

        df['MP_pressure'] = df['MP'] / (df['MP_rolling_avg'].abs() + 1e-3)
        df['FGA_pressure'] = df['FGA'] / (df['FGA_rolling_avg'].abs() + 1e-3)
        df['USG_pressure'] = df['USG%'] / (df['USG_rolling_avg'].abs() + 1e-3)

        df['Opportunity_Trend_Score'] = (
            0.40 * df['MP_trend']
            + 0.40 * df['FGA_trend']
            + 0.20 * df['USG_AST_ratio_trend']
        )
        df['Role_Instability_Score'] = (
            np.abs(df['MP_trend'])
            + np.abs(df['FGA_trend'])
            + np.abs(df['USG_AST_ratio_trend'])
        )
        df['PTS_Trend_Trust'] = (
            np.abs(df['PTS_resid_lag']) / (df['PTS_rolling_std'].abs() + 1e-3)
        )
        df['TRB_Trend_Trust'] = (
            np.abs(df['TRB_resid_lag']) / (df['TRB_rolling_std'].abs() + 1e-3)
        )
        df['AST_Trend_Trust'] = (
            np.abs(df['AST_resid_lag']) / (df['AST_rolling_std'].abs() + 1e-3)
        )
        df['Low_Rest_Flag'] = (df['Rest_Days'] <= 1).astype(float)
        df['Strong_Defense_Flag'] = (df['oppDfRtg_3'] < df['oppDfRtg_3'].median()).astype(float)
        df['Context_Pressure_Score'] = (
            0.40 * df['Low_Rest_Flag']
            + 0.35 * df['Did_Not_Play']
            + 0.25 * df['Strong_Defense_Flag']
        )
        df['Opportunity_Score'] = (
            0.35 * df['MP_pressure']
            + 0.35 * df['FGA_pressure']
            + 0.20 * df['USG_pressure']
            + 0.10 * df['High_MP_Flag']
        )
        df['PTS_Elasticity_Score'] = (
            np.abs(df['PTS_resid_lag']) * (1.0 + np.abs(df['Opportunity_Trend_Score']))
        )
        df['PTS_Downside_Pressure'] = (
            0.50 * (df['PTS_resid_lag'] < 0).astype(float)
            + 0.30 * df['Low_Rest_Flag']
            + 0.20 * df['Strong_Defense_Flag']
        )

        # High playmaker flag
        df['High_Playmaker_Flag'] = (
            (df['AST'] > df['AST_rolling_avg'] * 1.2) & 
            (df['AST'] > 5)
        ).astype(float)
        
        # Fill NaN values
        spike_cols = [
            'MP_trend', 'High_MP_Flag', 'FGA_trend', 'AST_trend',
            'AST_variance', 'USG_AST_ratio_trend', 'High_Playmaker_Flag',
            'PTS_rolling_std', 'TRB_rolling_std', 'AST_rolling_std',
            'PTS_resid_lag', 'TRB_resid_lag', 'AST_resid_lag',
            'MP_pressure', 'FGA_pressure', 'USG_pressure',
            'Opportunity_Trend_Score', 'Role_Instability_Score',
            'PTS_Trend_Trust', 'TRB_Trend_Trust', 'AST_Trend_Trust',
            'Low_Rest_Flag', 'Strong_Defense_Flag', 'Context_Pressure_Score',
            'Opportunity_Score', 'PTS_Elasticity_Score', 'PTS_Downside_Pressure'
        ]
        for col in spike_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        return df
    
    def prepare_data(self):
        """Prepare data with spike features"""
        print("\n Loading and preparing data...")
        
        # Load data from all players
        data_dir = _resolve_training_data_dir()
        print(f"   Using data directory: {data_dir}")
        all_dfs = []
        
        for player_dir in data_dir.iterdir():
            if not player_dir.is_dir():
                continue
            
            player_name = player_dir.name
            processed_files = list(_iter_training_csvs(player_dir))
            
            for file in processed_files:
                df = pd.read_csv(file)
                df['Player'] = player_name
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No data files found!")
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Loaded {len(df)} games from {len(all_dfs)} files")
        
        # Fill NaN/inf in all numeric columns before feature engineering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        
        # Create hybrid features
        df = self.create_hybrid_features(df)
        
        # Create sequences
        X, baselines, y, df_filtered = self.create_sequences(df)
        
        print(f"Created {len(X)} sequences")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        return X, baselines, y, df_filtered
    
    def create_sequences(self, df):
        """Create sequences for training"""
        seq_len = self.config["seq_len"]
        
        # Ensure required columns exist
        required_cols = ['Player', 'PTS', 'TRB', 'AST']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by player and game order
        if 'Date' in df.columns:
            df = df.sort_values(['Player', 'Date']).reset_index(drop=True)
        elif 'Game_Index' in df.columns:
            df = df.sort_values(['Player', 'Game_Index']).reset_index(drop=True)
        else:
            df = df.sort_values(['Player', 'Game_Num']).reset_index(drop=True)
        
        # Create player/team/opponent mappings
        self.player_mapping = {p: i for i, p in enumerate(df['Player'].unique())}
        
        # Preserve team/opponent identity even when the processed source only exposes numeric IDs.
        if 'Team' in df.columns:
            team_values = df['Team'].astype(str)
            self.team_mapping = {t: i for i, t in enumerate(team_values.unique())}
            df['Team_ID'] = team_values.map(self.team_mapping).fillna(0).astype(int)
        elif 'Team_ID' in df.columns:
            raw_team_ids = pd.Series(df['Team_ID']).fillna(-1).astype(int)
            unique_team_ids = sorted(raw_team_ids.unique().tolist())
            self.team_mapping = {str(team_id): i for i, team_id in enumerate(unique_team_ids)}
            team_id_map = {team_id: i for i, team_id in enumerate(unique_team_ids)}
            df['Team_ID'] = raw_team_ids.map(team_id_map).fillna(0).astype(int)
        else:
            self.team_mapping = {'UNK': 0}
            df['Team_ID'] = 0
        
        if 'Opponent' in df.columns:
            opp_values = df['Opponent'].astype(str)
            self.opponent_mapping = {o: i for i, o in enumerate(opp_values.unique())}
            df['Opponent_ID'] = opp_values.map(self.opponent_mapping).fillna(0).astype(int)
        elif 'Opponent_ID' in df.columns:
            raw_opp_ids = pd.Series(df['Opponent_ID']).fillna(-1).astype(int)
            unique_opp_ids = sorted(raw_opp_ids.unique().tolist())
            self.opponent_mapping = {str(opp_id): i for i, opp_id in enumerate(unique_opp_ids)}
            opp_id_map = {opp_id: i for i, opp_id in enumerate(unique_opp_ids)}
            df['Opponent_ID'] = raw_opp_ids.map(opp_id_map).fillna(0).astype(int)
        else:
            self.opponent_mapping = {'UNK': 0}
            df['Opponent_ID'] = 0

        # Some processed sources incorrectly mirror opponent IDs to team IDs.
        # If that happens almost everywhere, collapse opponent identity and rely
        # on matchup covariates like oppDfRtg_3 instead of a misleading embedding.
        if len(df) and float((df['Team_ID'] == df['Opponent_ID']).mean()) > 0.98:
            print("   Warning: Opponent_ID mirrors Team_ID in this dataset; neutralizing opponent embedding input.")
            self.opponent_mapping = {'UNK': 0}
            df['Opponent_ID'] = 0
        
        # Encode categorical
        df['Player_ID'] = df['Player'].map(self.player_mapping)
        
        # Define feature columns
        categorical_features = ['Player_ID', 'Team_ID', 'Opponent_ID']

        # Numeric features (exclude targets and identifiers)
        exclude_cols = ['Player', 'Date', 'PTS', 'TRB', 'AST', 'Player_ID', 'Team_ID', 'Opponent_ID', 'Market_Fetched_At_UTC']
        numeric_features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        self.feature_columns = categorical_features + numeric_features
        self.baseline_features = ['PTS_rolling_avg', 'TRB_rolling_avg', 'AST_rolling_avg']
        
        # Ensure baseline features exist
        for bf in self.baseline_features:
            if bf not in df.columns:
                target = bf.replace('_rolling_avg', '')
                if target in df.columns:
                    df[bf] = df.groupby('Player')[target].transform(
                        lambda x: x.rolling(window=5, min_periods=1).mean()
                    )
                else:
                    df[bf] = 0.0
        
        # ============================================================
        # CRITICAL FIX: Extract baselines BEFORE scaling so they stay
        # in the same space as the raw targets (PTS, TRB, AST).
        # Previously baselines were extracted AFTER StandardScaler,
        # meaning delta_true = raw_target - z_scored_baseline ≈ raw_target
        # which made the model learn identity instead of corrections.
        # ============================================================
        baseline_raw = df[self.baseline_features].values.copy()  # [N, 3] in original space
        target_raw = df[self.target_columns].values.copy()       # [N, 3] in original space
        
        # Fit a separate scaler for targets so model trains in normalized space
        self.scaler_y = StandardScaler()
        target_raw_clean = np.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_y.fit(target_raw_clean)
        
        # Scale baselines with the SAME target scaler (they're in the same units)
        baseline_raw_clean = np.nan_to_num(baseline_raw, nan=0.0, posinf=0.0, neginf=0.0)
        baseline_scaled = self.scaler_y.transform(baseline_raw_clean)
        target_scaled = self.scaler_y.transform(target_raw_clean)
        
        # Store scaled baselines and targets back for sequence extraction
        df[['_baseline_scaled_0', '_baseline_scaled_1', '_baseline_scaled_2']] = baseline_scaled
        df[['_target_scaled_0', '_target_scaled_1', '_target_scaled_2']] = target_scaled
        
        print(f"   Target scaler means: {self.scaler_y.mean_}")
        print(f"   Target scaler stds:  {self.scaler_y.scale_}")
        print(f"   Sample raw baseline[0]: {baseline_raw_clean[0]}")
        print(f"   Sample scaled baseline[0]: {baseline_scaled[0]}")
        print(f"   Sample raw target[0]: {target_raw_clean[0]}")
        print(f"   Sample scaled target[0]: {target_scaled[0]}")
        
        # Scale numeric features (for input sequences)
        numeric_data = df[numeric_features].values
        numeric_data = np.nan_to_num(numeric_data, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_x = StandardScaler()
        numeric_scaled = self.scaler_x.fit_transform(numeric_data)
        numeric_scaled = np.nan_to_num(numeric_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store scaled values back for sequence creation
        df[numeric_features] = numeric_scaled
        
        # Create sequences
        X_list, baseline_list, y_list, indices = [], [], [], []
        baseline_scaled_cols = ['_baseline_scaled_0', '_baseline_scaled_1', '_baseline_scaled_2']
        target_scaled_cols = ['_target_scaled_0', '_target_scaled_1', '_target_scaled_2']
        
        for player in df['Player'].unique():
            player_df = df[df['Player'] == player].reset_index(drop=True)
            
            for i in range(len(player_df) - seq_len):
                seq = player_df.iloc[i:i+seq_len]
                target_row = player_df.iloc[i+seq_len]
                
                # Features: categorical + scaled numeric (already scaled in df)
                cat_features = seq[categorical_features].values
                num_features = seq[numeric_features].values
                X_seq = np.concatenate([cat_features, num_features], axis=1)
                
                # Baseline from last game in sequence — NOW IN SCALED SPACE
                baseline = seq.iloc[-1][baseline_scaled_cols].values
                
                # Target — NOW IN SCALED SPACE (same scaler as baseline)
                y_target = target_row[target_scaled_cols].values
                
                X_list.append(X_seq)
                baseline_list.append(baseline)
                y_list.append(y_target)
                indices.append(i+seq_len)
        
        X = np.array(X_list, dtype=np.float32)
        baselines = np.array(baseline_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        # Final NaN/inf cleanup
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        baselines = np.nan_to_num(baselines, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store spike features for model building
        self.spike_features = [
            "MP_trend", "High_MP_Flag", "FGA_trend", "AST_trend",
            "AST_variance", "USG_AST_ratio_trend", "High_Playmaker_Flag"
        ]
        
        return X, baselines, y, df

    # ============================== PATCH: REPLACE build_model_with_anti_collapse() ==============================
    def build_model_with_anti_collapse(self):
        """
        Build MoE model with anti-collapse mechanisms + ACTUAL auxiliary losses applied in-graph.

        Adds:
        - Router noise via RouterNoise(training=...)
        - Router Z-loss via model.add_loss(z_w * zloss)
        - Switch-style load balancing:
            * importance loss: CV^2 of mean routing probs
            * load loss: CV^2 of hard (argmax) routing counts
          weighted by self.lb_weight_var (epoch-ramped by callback)
        - Proper per-expert usage metrics (expert_i_usage) + entropy/zloss metrics
        """
        print("\n Building MoE model with anti-collapse mechanisms...")

        # Inputs
        sequence_input = Input(shape=(self.config["seq_len"], len(self.feature_columns)), name="sequence_input")
        baseline_input = Input(shape=(len(self.target_columns),), name="baseline_pred_input")

        # Extract ID slices
        player_ids = Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32), name="player_ids")(sequence_input)
        team_ids = Lambda(lambda x: tf.cast(x[:, :, 1], tf.int32), name="team_ids")(sequence_input)
        opponent_ids = Lambda(lambda x: tf.cast(x[:, :, 2], tf.int32), name="opponent_ids")(sequence_input)

        # Embeddings
        player_embed = Embedding(len(self.player_mapping), 16, name="player_embed")(player_ids)
        team_embed = Embedding(len(self.team_mapping), 8, name="team_embed")(team_ids)
        opponent_embed = Embedding(len(self.opponent_mapping), 8, name="opponent_embed")(opponent_ids)

        # Numeric features
        numeric_features = Lambda(lambda x: x[:, :, 3:], name="numeric_features")(sequence_input)

        # Combine all features
        combined = Concatenate(axis=-1, name="combined_features")([
            player_embed, team_embed, opponent_embed, numeric_features
        ])

        # Transformer encoder
        x = Dense(self.config["d_model"], name="input_projection")(combined)
        for i in range(self.config["n_layers"]):
            attn = MultiHeadAttention(
                num_heads=self.config["n_heads"],
                key_dim=self.config["d_model"] // self.config["n_heads"],
                name=f"attention_{i}"
            )(x, x)
            x = Add(name=f"add_attn_{i}")([x, attn])
            x = LayerNormalization(name=f"norm_attn_{i}")(x)

            ff = Dense(self.config["d_model"] * 2, activation="relu", name=f"ff1_{i}")(x)
            ff = Dropout(self.config["dropout"], name=f"dropout_ff_{i}")(ff)
            ff = Dense(self.config["d_model"], name=f"ff2_{i}")(ff)
            x = Add(name=f"add_ff_{i}")([x, ff])
            x = LayerNormalization(name=f"norm_ff_{i}")(x)

        # Sequence representation
        sequence_repr = GlobalAveragePooling1D(name="sequence_pooling")(x)
        sequence_repr_with_baseline = Concatenate(name="sequence_baseline_concat")([sequence_repr, baseline_input])

        # Supervised outlier head
        outlier_logit = Dense(1, activation=None, name="outlier_logit")(sequence_repr_with_baseline)
        outlier_prob = tf.keras.layers.Activation("sigmoid", name="outlier_prob")(outlier_logit)

        # Spike indicators (for visibility)
        if hasattr(self, "spike_features") and self.spike_features:
            spike_features_last = Lambda(
                lambda x: x[:, -1, -len(self.spike_features):],
                name="spike_features_last"
            )(sequence_input)
            spike_indicators = Dense(len(self.target_columns), activation="sigmoid", name="spike_indicators")(
                spike_features_last
            )
        else:
            spike_indicators = Lambda(
                lambda x: tf.zeros([tf.shape(x)[0], len(self.target_columns)]),
                name="spike_indicators_dummy"
            )(sequence_repr)

        # ========================================================================
        # MOE ROUTING (with real auxiliary losses)
        # ========================================================================
        num_experts = self.config.get("num_experts", 8)
        num_spike_experts = self.config.get("num_spike_experts", 3)
        total_experts = num_experts + num_spike_experts
        key_dim = self.config.get("prototype_key_dim", 256)

        # Orthogonal expert keys (prevents early convergence)
        expert_keys = Embedding(
            total_experts,
            key_dim,
            embeddings_initializer=tf.keras.initializers.Orthogonal(gain=0.5),
            name="expert_keys"
        )(tf.range(total_experts))

        # Query
        query = Dense(key_dim, name="router_query")(sequence_repr_with_baseline)

        # Logits + temperature
        routing_logits = tf.matmul(query, expert_keys, transpose_b=True)  # [B, E]
        temperature = float(self.config["router_temperature"])
        routing_logits_scaled = Lambda(lambda z: z / temperature, name="routing_logits_scaled")(routing_logits)

        # TF2-safe noise injection (replaces tf.cond(learning_phase()))
        routing_logits_noisy = RouterNoise(stddev=float(self.config["router_noise_std"]), name="router_logits_noisy")(
            routing_logits_scaled
        )

        # Probabilities
        routing_probs = Softmax(name="routing_probs")(routing_logits_noisy)  # [B, E]

        # Router entropy (per-example)
        router_entropy = Lambda(
            lambda p: -tf.reduce_sum(p * tf.math.log(p + 1e-10), axis=1),
            name="router_entropy"
        )(routing_probs)

        # Router Z-loss (per-example): square(logsumexp(logits))
        router_lse = Lambda(lambda logits: tf.reduce_logsumexp(logits, axis=1), name="router_lse")(routing_logits)
        router_z_loss = Lambda(lambda lse: tf.square(lse), name="router_z_loss")(router_lse)

        # Expert usage (mean probs across batch): [E]
        expert_usage = Lambda(lambda p: tf.reduce_mean(p, axis=0), name="expert_usage")(routing_probs)

        # Top-k routing (k=2)
        top_k = 2
        top_k_probs, top_k_indices = tf.nn.top_k(routing_probs, k=top_k)
        top_k_probs_normalized = top_k_probs / tf.reduce_sum(top_k_probs, axis=1, keepdims=True)

        # ========================================================================
        # EXPERT NETWORKS
        # ========================================================================
        regular_experts = []
        for i in range(num_experts):
            expert = tf.keras.Sequential([
                Dense(128, activation="relu", name=f"regular_expert_{i}_fc1"),
                Dropout(self.config["dropout"], name=f"regular_expert_{i}_dropout"),
                Dense(64, activation="relu", name=f"regular_expert_{i}_fc2"),
                Dense(len(self.target_columns), activation=None, name=f"regular_expert_{i}_output")
            ], name=f"regular_expert_{i}")
            regular_experts.append(expert)

        spike_experts = []
        for i in range(num_spike_experts):
            expert = tf.keras.Sequential([
                Dense(128, activation="relu", name=f"spike_expert_{i}_fc1"),
                Dropout(self.config["dropout"], name=f"spike_expert_{i}_dropout"),
                Dense(64, activation="relu", name=f"spike_expert_{i}_fc2"),
                Dense(len(self.target_columns), activation=None, name=f"spike_expert_{i}_output")
            ], name=f"spike_expert_{i}")
            spike_experts.append(expert)

        all_experts = regular_experts + spike_experts

        expert_outputs = []
        for expert in all_experts:
            expert_outputs.append(expert(sequence_repr_with_baseline))
        expert_outputs_stacked = Lambda(lambda xs: tf.stack(xs, axis=1), name="expert_outputs_stacked")(
            expert_outputs
        )  # [B, E, T]

        def combine_experts(args):
            outputs, indices, weights = args
            batch_size = tf.shape(outputs)[0]
            batch_indices = tf.tile(tf.range(batch_size)[:, None], [1, top_k])
            gather_indices = tf.stack([batch_indices, indices], axis=-1)
            selected_outputs = tf.gather_nd(outputs, gather_indices)  # [B, k, T]
            weights_expanded = tf.expand_dims(weights, axis=-1)       # [B, k, 1]
            return tf.reduce_sum(selected_outputs * weights_expanded, axis=1)

        delta_pred = Lambda(combine_experts, name="delta_pred")([
            expert_outputs_stacked,
            top_k_indices,
            top_k_probs_normalized
        ])

        # ========================================================================
        # BUILD MODEL
        # ========================================================================
        model = Model(
            inputs=[sequence_input, baseline_input],
            outputs=[
                delta_pred,
                outlier_prob,
                spike_indicators,
                routing_probs,
                router_entropy,
                router_z_loss,
                expert_usage,
                expert_outputs_stacked
            ],
            name="unified_moe_model"
        )

        # ========================================================================
        # METRICS (so your callback can actually see them)
        # ========================================================================
        for i in range(total_experts):
            model.add_metric(
                tf.reduce_mean(routing_probs[:, i]),
                name=f"expert_{i}_usage",
                aggregation="mean",
            )
        model.add_metric(tf.reduce_mean(router_entropy), name="router_entropy_mean", aggregation="mean")
        model.add_metric(tf.reduce_mean(router_z_loss), name="router_z_loss_mean", aggregation="mean")

        # ========================================================================
        # AUX LOSSES (ACTUALLY APPLIED)
        # ========================================================================

        # 1) Router Z-loss weight (prevents logit explosion)
        model.add_loss(self.zloss_weight_var * tf.reduce_mean(router_z_loss))

        # 2) Switch-style load balancing: importance + load (CV^2)
        #    importance: mean routing prob per expert
        importance = tf.reduce_mean(routing_probs, axis=0)  # [E]
        imp_mean = tf.reduce_mean(importance)
        imp_var = tf.reduce_mean(tf.square(importance - imp_mean))
        imp_cv2 = imp_var / (tf.square(imp_mean) + 1e-9)

        #    load: hard counts via argmax routing (top-1 proxy; stable + cheap)
        hard = tf.one_hot(tf.argmax(routing_probs, axis=1), depth=total_experts, dtype=tf.float32)  # [B,E]
        load = tf.reduce_mean(hard, axis=0)  # [E]
        load_mean = tf.reduce_mean(load)
        load_var = tf.reduce_mean(tf.square(load - load_mean))
        load_cv2 = load_var / (tf.square(load_mean) + 1e-9)

        # Apply weights (epoch-ramped via lb_weight_var)
        # Use your importance_weight/load_weight as relative scalars.
        lb_w = self.lb_weight_var
        model.add_loss(lb_w * float(self.config.get("importance_weight", 0.02)) * imp_cv2)
        model.add_loss(lb_w * float(self.config.get("load_weight", 0.02)) * load_cv2)

        # Optional: entropy encouragement toward uniform (gentle)
        # (Keeps routing broad without forcing perfect uniformity.)
        entropy_target_frac = float(self.config.get("entropy_target_frac", 0.85))
        max_ent = tf.math.log(tf.cast(total_experts, tf.float32))
        ent_ratio = tf.reduce_mean(router_entropy) / (max_ent + 1e-9)
        ent_penalty = tf.square(tf.maximum(0.0, entropy_target_frac - ent_ratio))
        model.add_loss(0.01 * lb_w * ent_penalty)

        # ====================================================================
        # 3) Capacity enforcement  (overflow penalty)
        #    Count how many times each expert appears in top-k assignments.
        #    If any expert exceeds  capacity_factor * ceil(B*k / E)  tokens,
        #    penalise the squared overflow.
        # ====================================================================
        capacity_factor = float(self.config.get("capacity_factor", 2.0))
        overflow_w = float(self.config.get("overflow_penalty", 0.02))

        # top_k_indices: [B, k] int32  – already computed above
        # Flatten to [B*k] and one-hot → per-expert hard counts
        flat_indices = tf.reshape(top_k_indices, [-1])                       # [B*k]
        topk_onehot = tf.one_hot(flat_indices, depth=total_experts,
                                 dtype=tf.float32)                           # [B*k, E]
        expert_counts = tf.reduce_sum(topk_onehot, axis=0)                   # [E]

        batch_size_f = tf.cast(tf.shape(routing_probs)[0], tf.float32)
        # Ideal uniform tokens per expert, then multiply by capacity headroom
        tokens_per_expert_ideal = batch_size_f * tf.cast(top_k, tf.float32) / tf.cast(total_experts, tf.float32)
        capacity_threshold = capacity_factor * tokens_per_expert_ideal       # scalar

        # Squared hinge: only penalise counts that exceed the threshold
        overflow = tf.maximum(0.0, expert_counts - capacity_threshold)       # [E]
        overflow_loss = tf.reduce_mean(tf.square(overflow))
        # Normalise by threshold² so the scale is independent of batch size
        overflow_loss = overflow_loss / (tf.square(capacity_threshold) + 1e-9)

        model.add_loss(lb_w * overflow_w * overflow_loss)
        model.add_metric(overflow_loss, name="overflow_loss", aggregation="mean")

        # ====================================================================
        # 4) Diversity regularisation  (Phase 3)
        #    Penalise pairwise cosine similarity between expert output vectors
        #    so experts are encouraged to produce distinct predictions.
        #    Computed on a random sub-sample of the batch for efficiency.
        # ====================================================================
        diversity_coef = float(self.config.get("diversity_coef", 0.001))
        sample_frac = float(self.config.get("diversity_sample_fraction", 0.5))

        # expert_outputs_stacked: [B, E, T]  – already computed above
        # Sub-sample rows for efficiency
        n_sample = tf.cast(
            tf.maximum(1, tf.cast(batch_size_f * sample_frac, tf.int32)),
            tf.int32,
        )
        sample_idx = tf.random.shuffle(tf.range(tf.shape(expert_outputs_stacked)[0]))[:n_sample]
        sampled_outputs = tf.gather(expert_outputs_stacked, sample_idx)      # [S, E, T]

        # L2-normalise each expert's output vector along the target axis
        normed = tf.math.l2_normalize(sampled_outputs, axis=-1)              # [S, E, T]

        # Pairwise cosine similarity matrix per sample: [S, E, E]
        cos_sim = tf.matmul(normed, normed, transpose_b=True)               # [S, E, E]

        # Zero out the diagonal (self-similarity = 1, not informative)
        eye = tf.eye(total_experts, dtype=tf.float32)                        # [E, E]
        cos_sim_off = cos_sim * (1.0 - eye)                                  # [S, E, E]

        # Mean of squared off-diagonal similarities
        n_pairs = tf.cast(total_experts * (total_experts - 1), tf.float32)
        diversity_loss = tf.reduce_sum(tf.square(cos_sim_off)) / (
            tf.cast(n_sample, tf.float32) * n_pairs + 1e-9
        )

        model.add_loss(lb_w * diversity_coef * diversity_loss)
        model.add_metric(diversity_loss, name="diversity_loss", aggregation="mean")

        # ====================================================================

        print(f"   Model built with {total_experts} experts")
        print(f"   Router temperature: {temperature}")
        print(f"   Expert key initialization: Orthogonal(gain=0.5)")
        print(f"   Top-k routing: k={top_k}")
        print(f"   Aux losses: z-loss + CV² + entropy + overflow + diversity")

        return model

    
    def create_moe_metrics_callback(self):
        """Create callback to monitor expert usage and detect collapse"""
        class MoEMonitorCallback(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
                self.total_experts = (
                    trainer.config.get("num_experts", 8) + 
                    trainer.config.get("num_spike_experts", 3)
                )

            def on_epoch_begin(self, epoch, logs=None):
                # Update epoch-aware LB weight
                new_w = float(self.trainer.get_load_balance_weight(epoch))
                self.trainer.lb_weight_var.assign(new_w)

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}

                # Pull expert usage metrics that are now guaranteed to exist
                expert_usages = []
                for i in range(self.total_experts):
                    key = f"expert_{i}_usage"
                    if key in logs:
                        expert_usages.append(float(logs[key]))

                ent = logs.get("router_entropy_mean", None)

                if expert_usages:
                    max_usage = max(expert_usages)
                    dead_experts = sum(1 for u in expert_usages if u < 0.02)

                    # Check for collapse
                    if max_usage > 0.30:
                        print(f"\n⚠️  WARNING: Expert collapse detected!")
                        print(f"   Expert with max usage: {max_usage*100:.1f}%")
                        print(f"   Dead experts (< 2%): {dead_experts}/{self.total_experts}")

                    if ent is not None:
                        max_entropy = float(np.log(self.total_experts))
                        entropy_ratio = float(ent) / (max_entropy + 1e-9)

                        if entropy_ratio < 0.50:
                            print(f"\n⚠️  WARNING: Low router entropy!")
                            print(f"   Entropy: {float(ent):.3f} ({entropy_ratio*100:.1f}% of max)")

                # Log every N epochs
                if (epoch + 1) % self.trainer.config.get("log_moe_metrics_every_n_epochs", 1) == 0:
                    lb_w = float(self.trainer.lb_weight_var.numpy())
                    print(f"\n📊 Expert Usage (Epoch {epoch+1}) | LB_w={lb_w:.4f}:")
                    for i, usage in enumerate(expert_usages):
                        expert_type = "spike" if i >= self.trainer.config.get("num_experts", 8) else "regular"
                        print(f"   Expert {i} ({expert_type}): {usage*100:.2f}%")

                    if ent is not None:
                        print(f"   Router entropy mean: {float(ent):.3f}")

                    z = logs.get("router_z_loss_mean", None)
                    if z is not None:
                        print(f"   Router Z-loss mean: {float(z):.3f}")
                    
                    ov = logs.get("overflow_loss", None)
                    if ov is not None:
                        print(f"   Overflow loss: {float(ov):.5f}")
                    
                    div = logs.get("diversity_loss", None)
                    if div is not None:
                        print(f"   Diversity loss: {float(div):.5f}")

        return MoEMonitorCallback(self)

    # ================================================================
    # STACKING: GBM + Meta-Learner methods
    # ================================================================
    
    def prepare_gbm_features(self, X_seq, baselines):
        """Flatten sequence data for GBM: last timestep + stats + trend + last3 + baselines."""
        n_samples, seq_len, n_feat = X_seq.shape
        last = X_seq[:, -1, :]
        seq_mean = X_seq.mean(axis=1)
        seq_std = X_seq.std(axis=1)
        if seq_len >= 6:
            trend = X_seq[:, -3:, :].mean(axis=1) - X_seq[:, :3, :].mean(axis=1)
        else:
            trend = np.zeros_like(last)
        last3 = X_seq[:, -3:, :].reshape(n_samples, -1)
        return np.hstack([last, seq_mean, seq_std, trend, last3, baselines])
    
    def train_gbm_models(self, X_gbm_train, delta_train, X_gbm_val, delta_val, n_targets):
        """Train one LightGBM per target. Returns (models, val_predictions)."""
        try:
            import lightgbm as lgb
            use_lgb = True
            print("  Using LightGBM")
        except ImportError:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                use_lgb = False
                print("  LightGBM not available, using sklearn GBM")
            except ImportError:
                print("  No GBM available, skipping")
                return None, None
        
        models = []
        val_preds = np.zeros_like(delta_val)
        
        for t in range(n_targets):
            tname = self.target_columns[t]
            print(f"  Training GBM for {tname}...")
            
            if use_lgb:
                params = {
                    'objective': 'regression', 'metric': 'mae',
                    'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6,
                    'min_child_samples': 20, 'subsample': 0.8,
                    'colsample_bytree': 0.8, 'reg_alpha': 0.1,
                    'reg_lambda': 0.1, 'verbose': -1, 'n_jobs': -1,
                }
                train_data = lgb.Dataset(X_gbm_train, label=delta_train[:, t])
                val_data = lgb.Dataset(X_gbm_val, label=delta_val[:, t], reference=train_data)
                model = lgb.train(
                    params, train_data, num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
                )
                val_preds[:, t] = model.predict(X_gbm_val)
                print(f"    best iteration: {model.best_iteration}")
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=20,
                    validation_fraction=0.15, n_iter_no_change=30, verbose=0
                )
                model.fit(X_gbm_train, delta_train[:, t])
                val_preds[:, t] = model.predict(X_gbm_val)
            models.append(model)
        
        return models, val_preds
    
    def train_stacking_meta(self, lstm_delta, gbm_delta, baselines_val, true_delta, n_targets):
        """Train Ridge meta-learner per target. Features: LSTM delta, GBM delta, baselines."""
        meta_models = []
        for t in range(n_targets):
            tname = self.target_columns[t]
            feats = [lstm_delta[:, t:t+1]]
            if gbm_delta is not None:
                feats.append(gbm_delta[:, t:t+1])
            feats.append(baselines_val)
            X_meta = np.hstack(feats)
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_meta, true_delta[:, t])
            meta_models.append(ridge)
            pred = ridge.predict(X_meta)
            mae = np.mean(np.abs(pred - true_delta[:, t]))
            print(f"  {tname} meta-learner MAE (scaled): {mae:.4f}, coefs: {ridge.coef_[:4]}...")
        return meta_models
    
    def train_unified(self, X_train, baselines_train, y_train, X_val, baselines_val, y_val):
        """
        Train the unified model (simple LSTM or MoE with anti-collapse)
        """
        use_simple = self.config.get("use_simple_model", False)
        
        print("\n" + "="*80)
        print(f"TRAINING {'SIMPLE LSTM' if use_simple else 'UNIFIED MOE'} MODEL")
        print("="*80)
        
        # Build model
        if use_simple:
            net = self.build_simple_lstm_model()
        else:
            net = self.build_model_with_anti_collapse()
        
        if use_simple:
            # ============================================================
            # SIMPLE PATH: Build a plain Keras model that takes [seq, base]
            # and outputs delta predictions. Train with raw MSE loss.
            # This avoids DeltaTrainedKerasModel overhead.
            # ============================================================
            # Extract just the delta_pred output (first output)
            simple_model = Model(
                inputs=net.inputs,
                outputs=net.outputs[0],  # delta_pred only
                name="simple_lstm_delta"
            )
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.get("lr", 0.001),
                clipnorm=1.0
            )
            simple_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Compute delta targets
            delta_train = y_train - baselines_train
            delta_val = y_val - baselines_val
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get("patience", 25),
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    verbose=1
                ),
                ModelCheckpoint(
                    'model/unified_moe_best.weights.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ),
            ]
            
            print(f"\n Starting training (simple LSTM, MSE loss on delta)...")
            print(f"   Train samples: {len(X_train)}")
            print(f"   Val samples: {len(X_val)}")
            print(f"   Delta train std: {np.std(delta_train, axis=0)}")
            
            history = simple_model.fit(
                [X_train, baselines_train], delta_train,
                validation_data=([X_val, baselines_val], delta_val),
                epochs=self.config.get("epochs", 120),
                batch_size=self.config.get("batch_size", 128),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save
            simple_model.save_weights('model/unified_moe_final.weights.h5')
            
            # For evaluation, wrap in a lambda that adds baseline back
            class SimpleModelWrapper:
                def __init__(self, model):
                    self.model = model
                def predict(self, inputs, **kwargs):
                    delta = self.model.predict(inputs, **kwargs)
                    return [delta]  # Return as list to match MoE interface
                def save_weights(self, path):
                    self.model.save_weights(path)
            
            model = SimpleModelWrapper(simple_model)
            
        else:
            # ============================================================
            # MOE PATH: Use DeltaTrainedKerasModel wrapper
            # ============================================================
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
                variance_weight=float(self.config.get("variance_weight", 0.0)),
            )
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.get("lr", 0.001),
                clipnorm=1.0
            )
            model.compile(optimizer=optimizer)
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get("patience", 15),
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    verbose=1
                ),
                ModelCheckpoint(
                    'model/unified_moe_best.weights.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ),
                self.create_moe_metrics_callback()
            ]
            
            print(f"\n Starting training (MoE)...")
            print(f"   Train samples: {len(X_train)}")
            print(f"   Val samples: {len(X_val)}")
            print(f"   Epochs: {self.config.get('epochs', 50)}")
            print(f"   Batch size: {self.config.get('batch_size', 64)}")
            
            history = model.fit(
                [X_train, baselines_train],
                y_train,
                validation_data=([X_val, baselines_val], y_val),
                epochs=self.config.get("epochs", 50),
                batch_size=self.config.get("batch_size", 64),
                callbacks=callbacks,
                verbose=1
            )
            
            model.save_weights('model/unified_moe_final.weights.h5')
        
        print("\n Training complete!")
        print(f"   Best model saved to: model/unified_moe_best.weights.h5")
        print(f"   Final model saved to: model/unified_moe_final.weights.h5")
        
        # Save metadata
        metadata = {
            'model_type': 'unified_moe_with_anti_collapse',
            'num_experts': self.config.get("num_experts", 8),
            'num_spike_experts': self.config.get("num_spike_experts", 3),
            'total_experts': self.config.get("num_experts", 8) + self.config.get("num_spike_experts", 3),
            'router_temperature': self.config["router_temperature"],
            'load_balance_final': self.config["load_balance_weight_final"],
            'router_z_loss_weight': self.config["router_z_loss_weight"],
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'baseline_features': self.baseline_features,
            'player_mapping': self.player_mapping,
            'team_mapping': self.team_mapping,
            'opponent_mapping': self.opponent_mapping,
            'config': self.config
        }
        
        with open('model/unified_moe_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save scaler
        joblib.dump(self.scaler_x, 'model/unified_moe_scaler_x.pkl')
        joblib.dump(self.scaler_y, 'model/unified_moe_scaler_y.pkl')
        
        print(" Metadata and scalers saved")
        
        return history, model


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function"""
    print("\n" + "="*80)
    print("UNIFIED MOE TRAINER - EXPERT COLLAPSE PREVENTION")
    print("="*80)
    
    # Initialize trainer (config is set inside __init__ via self.config dict)
    trainer = UnifiedMoETrainer()
    
    # Prepare data
    X, baselines, y, df = trainer.prepare_data()
    
    # Split data (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    baselines_train, baselines_val = baselines[:split_idx], baselines[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    
    use_stacked = trainer.config.get("use_stacked_model", False)
    
    if use_stacked:
        # ============================================================
        # STACKED PIPELINE: LSTM + GBM + Ridge meta-learner
        # ============================================================
        print("\n" + "="*80)
        print("STACKED MODEL: LSTM + GBM + Ridge Meta-Learner")
        print("="*80)
        
        delta_train = y_train - baselines_train
        delta_val = y_val - baselines_val
        n_targets = y_train.shape[1]
        
        # Phase 1: Train LSTM
        print("\n--- PHASE 1: LSTM ---")
        history, lstm_wrapper = trainer.train_unified(
            X_train, baselines_train, y_train,
            X_val, baselines_val, y_val
        )
        lstm_delta = lstm_wrapper.predict([X_val, baselines_val], verbose=0, batch_size=256)
        if isinstance(lstm_delta, (list, tuple)):
            lstm_delta = lstm_delta[0]
        if hasattr(lstm_delta, 'numpy'):
            lstm_delta = lstm_delta.numpy()
        
        # Phase 2: Train GBM
        print("\n--- PHASE 2: GBM ---")
        X_gbm_train = trainer.prepare_gbm_features(X_train, baselines_train)
        X_gbm_val = trainer.prepare_gbm_features(X_val, baselines_val)
        print(f"  GBM feature dim: {X_gbm_train.shape[1]}")
        gbm_models, gbm_delta = trainer.train_gbm_models(
            X_gbm_train, delta_train, X_gbm_val, delta_val, n_targets
        )
        
        # Phase 3: Train meta-learner
        print("\n--- PHASE 3: Stacking Meta-Learner ---")
        meta_models = trainer.train_stacking_meta(
            lstm_delta, gbm_delta, baselines_val, delta_val, n_targets
        )
        
        # Save GBM and meta-learner models
        joblib.dump(gbm_models, 'model/stacked_gbm_models.pkl')
        joblib.dump(meta_models, 'model/stacked_meta_models.pkl')
        print("  GBM and meta-learner models saved")
        
        # Phase 4: Evaluate all approaches
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        
        y_val_orig = trainer.scaler_y.inverse_transform(y_val)
        b_val_orig = trainer.scaler_y.inverse_transform(baselines_val)
        
        # Baseline
        print("\n--- BASELINE (rolling avg) ---")
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], b_val_orig[:, i])
            print(f"  {t}: {mae:.4f}")
        
        # LSTM only
        print("\n--- LSTM ONLY ---")
        lstm_preds = trainer.scaler_y.inverse_transform(baselines_val + lstm_delta)
        lstm_total = sum(mean_absolute_error(y_val_orig[:, i], lstm_preds[:, i])
                        for i in range(n_targets))
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], lstm_preds[:, i])
            print(f"  {t}: {mae:.4f}")
        print(f"  Avg MAE: {lstm_total / n_targets:.4f}")
        
        # GBM only
        if gbm_delta is not None:
            print("\n--- GBM ONLY ---")
            gbm_preds = trainer.scaler_y.inverse_transform(baselines_val + gbm_delta)
            gbm_total = sum(mean_absolute_error(y_val_orig[:, i], gbm_preds[:, i])
                           for i in range(n_targets))
            for i, t in enumerate(trainer.target_columns):
                mae = mean_absolute_error(y_val_orig[:, i], gbm_preds[:, i])
                print(f"  {t}: {mae:.4f}")
            print(f"  Avg MAE: {gbm_total / n_targets:.4f}")
        
        # Stacked meta-learner
        print("\n--- STACKED META-LEARNER ---")
        meta_delta = np.zeros_like(lstm_delta)
        for t_idx in range(n_targets):
            feats = [lstm_delta[:, t_idx:t_idx+1]]
            if gbm_delta is not None:
                feats.append(gbm_delta[:, t_idx:t_idx+1])
            feats.append(baselines_val)
            X_meta = np.hstack(feats)
            meta_delta[:, t_idx] = meta_models[t_idx].predict(X_meta)
        
        meta_preds = trainer.scaler_y.inverse_transform(baselines_val + meta_delta)
        meta_total = 0
        for i, t in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_orig[:, i], meta_preds[:, i])
            r2 = r2_score(y_val_orig[:, i], meta_preds[:, i])
            baseline_mae = mean_absolute_error(y_val_orig[:, i], b_val_orig[:, i])
            improvement = baseline_mae - mae
            pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
            meta_total += mae
            print(f"  {t}: MAE={mae:.4f}  R²={r2:.4f}  (baseline: {baseline_mae:.4f}, improvement: {improvement:+.4f} / {pct:+.1f}%)")
        print(f"  Avg MAE: {meta_total / n_targets:.4f}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        results = {
            'stacked': meta_total / n_targets,
            'lstm': lstm_total / n_targets,
        }
        if gbm_delta is not None:
            results['gbm'] = gbm_total / n_targets
        for name, avg in sorted(results.items(), key=lambda x: x[1]):
            marker = " <-- BEST" if avg == min(results.values()) else ""
            print(f"  {name:20s}: {avg:.4f}{marker}")
        print(f"  {'target (old ens)':20s}: 4.795")
        print("="*60)
        
    else:
        # ============================================================
        # ORIGINAL PIPELINE: Simple LSTM or MoE
        # ============================================================
        history, model = trainer.train_unified(
            X_train, baselines_train, y_train,
            X_val, baselines_val, y_val
        )
        
        print("\n" + "="*80)
        print("VALIDATION SET EVALUATION")
        print("="*80)
        
        use_simple = trainer.config.get("use_simple_model", False)
        preds_raw = model.predict([X_val, baselines_val], verbose=0, batch_size=128)
        
        if isinstance(preds_raw, (list, tuple)):
            preds_raw = preds_raw[0] if isinstance(preds_raw[0], np.ndarray) else preds_raw[0].numpy()
        if hasattr(preds_raw, 'numpy'):
            preds_raw = preds_raw.numpy()
        
        if use_simple:
            preds_scaled = baselines_val + preds_raw
        else:
            preds_scaled = preds_raw
        
        pred_original = trainer.scaler_y.inverse_transform(preds_scaled)
        y_val_original = trainer.scaler_y.inverse_transform(y_val)
        baselines_val_original = trainer.scaler_y.inverse_transform(baselines_val)
        
        for i, target in enumerate(trainer.target_columns):
            mae = mean_absolute_error(y_val_original[:, i], pred_original[:, i])
            r2 = r2_score(y_val_original[:, i], pred_original[:, i])
            baseline_mae = mean_absolute_error(y_val_original[:, i], baselines_val_original[:, i])
            improvement = baseline_mae - mae
            pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
            print(f"\n{target}:")
            print(f"  Model MAE:    {mae:.4f}")
            print(f"  Model R²:     {r2:.4f}")
            print(f"  Baseline MAE: {baseline_mae:.4f}")
            print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%)")
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
