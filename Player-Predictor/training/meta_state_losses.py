#!/usr/bin/env python3
"""
Meta-State MoE Loss Functions
Delta-focused losses with uncertainty separation
"""

import tensorflow as tf
import numpy as np


def create_meta_state_loss(config, target_columns):
    """
    Create comprehensive loss for meta-state MoE
    
    Components:
    1. Delta Huber loss (explicit delta learning)
    2. Student-t NLL on reconstructed μ (probabilistic)
    3. Event classification loss (BCE)
    4. Epistemic supervision (weak ranking)
    5. Calibration loss (quantile coverage)
    """
    
    n_targets = len(target_columns)
    
    def loss_fn(y_true, y_pred, baselines, event_true=None):
        """
        y_pred structure: [delta_mean (3), sigma_ale (3), u_epi (3), event_logits (4), gate (1)]
        Total: 14 outputs
        
        CRITICAL: Model outputs DELTA, not absolute prediction
        """
        
        # Extract outputs
        delta_pred = y_pred[:, :n_targets]  # Delta predictions from model
        sigma_ale = y_pred[:, n_targets:2*n_targets]
        u_epi = y_pred[:, 2*n_targets:3*n_targets]
        event_logits = y_pred[:, 3*n_targets:3*n_targets+4]
        gate = y_pred[:, -1:]
        
        # Reconstruct absolute prediction
        mu = baselines + delta_pred
        
        # Delta space (for delta-focused losses)
        delta_true = y_true - baselines
        
        # ===== 1. Delta Huber Loss (explicit delta learning) =====
        delta_loss = tf.keras.losses.huber(delta_true, delta_pred, delta=2.0)
        
        # ===== 2. Student-t NLL on reconstructed μ =====
        df = tf.cast(config.get("student_t_df", 4.0), tf.float32)
        resid = (y_true - mu) / (sigma_ale + 1e-6)
        
        log_like = (
            tf.math.lgamma((df + 1.0) / 2.0) -
            tf.math.lgamma(df / 2.0) -
            0.5 * tf.math.log(df * np.pi) -
            tf.math.log(sigma_ale + 1e-6) -
            ((df + 1.0) / 2.0) * tf.math.log(1.0 + tf.square(resid) / df)
        )
        nll = -tf.reduce_mean(log_like)
        
        # ===== 3. Event Classification Loss =====
        event_loss = 0.0
        if event_true is not None:
            event_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(event_true, event_logits)
            )
        
        # ===== 4. Epistemic Supervision (weak ranking) =====
        # Epistemic should correlate with prediction difficulty
        abs_residuals = tf.abs(y_true - mu)
        residual_difficulty = tf.reduce_mean(abs_residuals, axis=1)  # Per sample
        epi_magnitude = tf.reduce_mean(u_epi, axis=1)  # Per sample
        
        # Rank correlation (Spearman-like)
        residual_ranks = tf.cast(tf.argsort(tf.argsort(residual_difficulty)), tf.float32)
        epi_ranks = tf.cast(tf.argsort(tf.argsort(epi_magnitude)), tf.float32)
        
        # Normalize ranks to [0, 1]
        n_samples = tf.cast(tf.shape(residual_ranks)[0], tf.float32)
        residual_ranks_norm = residual_ranks / (n_samples + 1e-6)
        epi_ranks_norm = epi_ranks / (n_samples + 1e-6)
        
        # Correlation loss (maximize correlation = minimize negative correlation)
        rank_corr = tf.reduce_mean(residual_ranks_norm * epi_ranks_norm)
        epistemic_loss = -rank_corr
        
        # ===== 5. Calibration Loss (quantile coverage) =====
        # 68% of errors should fall within 1 sigma_ale
        # 95% of errors should fall within 2 sigma_ale
        abs_errors = tf.abs(y_true - mu)
        within_1sigma = tf.cast(abs_errors < sigma_ale, tf.float32)
        within_2sigma = tf.cast(abs_errors < 2.0 * sigma_ale, tf.float32)
        
        coverage_1sigma = tf.reduce_mean(within_1sigma)
        coverage_2sigma = tf.reduce_mean(within_2sigma)
        
        # Target: 68% and 95% coverage
        calibration_loss = (
            tf.square(coverage_1sigma - 0.68) +
            tf.square(coverage_2sigma - 0.95)
        )
        
        # ===== 6. Weak Mean Loss (backup) =====
        mean_loss = tf.reduce_mean(tf.abs(y_true - mu))
        
        # ===== Combine Losses =====
        total = (
            config.get("delta_huber_weight", 0.5) * delta_loss +
            config.get("nll_weight", 1.0) * nll +
            config.get("event_loss_weight", 0.1) * event_loss +
            config.get("epistemic_weight", 0.01) * epistemic_loss +
            config.get("calibration_weight", 0.05) * calibration_loss +
            config.get("mean_loss_weight", 0.05) * mean_loss
        )
        
        # Ensure scalar output
        return tf.reduce_mean(total) if hasattr(total, 'shape') and len(total.shape) > 0 else total
    
    return loss_fn


def create_delta_variance_loss(config, target_columns):
    """
    Per-bucket delta variance loss (from original trainer)
    Ensures predictions have realistic variance
    """
    
    def variance_loss(y_true, y_pred, baselines, bucket_ids):
        n = len(target_columns)
        
        # Extract delta from y_pred (model outputs delta now)
        delta_pred = y_pred[:, :n]
        
        # True deltas
        delta_true = y_true - baselines
        
        # Per-bucket variance computation
        bucket_ids = tf.cast(bucket_ids, tf.int32)
        num_buckets = tf.reduce_max(bucket_ids) + 1
        
        ones = tf.ones_like(bucket_ids, dtype=tf.float32)
        counts = tf.math.unsorted_segment_sum(ones, bucket_ids, num_buckets)
        counts = tf.maximum(counts, 1.0)
        
        # Per-bucket means
        sum_pred = tf.math.unsorted_segment_sum(delta_pred, bucket_ids, num_buckets)
        sum_true = tf.math.unsorted_segment_sum(delta_true, bucket_ids, num_buckets)
        mean_pred = sum_pred / tf.expand_dims(counts, 1)
        mean_true = sum_true / tf.expand_dims(counts, 1)
        
        # Per-bucket variances
        sumsq_pred = tf.math.unsorted_segment_sum(tf.square(delta_pred), bucket_ids, num_buckets)
        sumsq_true = tf.math.unsorted_segment_sum(tf.square(delta_true), bucket_ids, num_buckets)
        ex2_pred = sumsq_pred / tf.expand_dims(counts, 1)
        ex2_true = sumsq_true / tf.expand_dims(counts, 1)
        
        var_pred = tf.maximum(0.0, ex2_pred - tf.square(mean_pred))
        var_true = tf.maximum(0.0, ex2_true - tf.square(mean_true))
        
        # Variance ratio targets
        target_ratios = tf.constant(config.get("target_delta_variance_ratios", [0.40, 0.50, 0.35]), tf.float32)
        ratios = var_pred / (var_true + 1e-6)
        
        # Penalize deficit
        deficit = tf.maximum(0.0, target_ratios - ratios)
        
        # Only use buckets with >=3 samples
        valid = tf.cast(counts >= 3.0, tf.float32)
        bucket_loss = tf.reduce_mean(tf.square(deficit), axis=1)
        
        denom = tf.reduce_sum(valid) + 1e-6
        return tf.reduce_sum(bucket_loss * valid) / denom
    
    return variance_loss


def create_baseline_correlation_penalty(config, target_columns):
    """
    Penalize correlation between baseline and delta predictions
    Prevents "cancellation" where model learns baseline - constant
    """
    
    def cov_penalty(y_true, y_pred, baselines):
        n = len(target_columns)
        
        # Extract delta from y_pred (model outputs delta now)
        delta_pred = y_pred[:, :n]
        
        # Center
        b = baselines - tf.reduce_mean(baselines, axis=0, keepdims=True)
        dp = delta_pred - tf.reduce_mean(delta_pred, axis=0, keepdims=True)
        
        # Covariance
        cov = tf.reduce_mean(b * dp, axis=0)
        
        # Stat-specific weights (TRB/AST need more pressure)
        weights = tf.constant([0.5, 2.5, 2.5], tf.float32)
        cov_loss = tf.reduce_mean(tf.square(cov) * weights)
        
        # Negative correlation penalty
        b_std = tf.math.reduce_std(baselines, axis=0) + 1e-9
        dp_std = tf.math.reduce_std(delta_pred, axis=0) + 1e-9
        corr = cov / (b_std * dp_std)
        
        neg_corr_penalty = tf.reduce_mean(tf.maximum(0.0, -corr) * weights)
        
        return cov_loss + config.get("neg_corr_penalty_weight", 0.5) * neg_corr_penalty
    
    return cov_penalty
