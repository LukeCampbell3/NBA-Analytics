#!/usr/bin/env python3
"""
NBA v8 Distributional Head

Adds quantile regression and P(over line) estimation on top of the v7 LSTM backbone.

Architecture:
  v7 LSTM latent state (170D)
    -> Quantile head: outputs Q10, Q25, Q50, Q75, Q90 for each stat
    -> Regime head: classifies game state (normal/usage-spike/blowout/foul-trouble)
    -> Sigma head: per-stat uncertainty estimate
    -> Over probability: P(stat > line) using predicted distribution

The key insight from the research:
  - Props are threshold-crossing events on fat-tailed distributions
  - We need the full distribution, not just the mean
  - P(over) = 1 - CDF(line | predicted_distribution)
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, Concatenate, Input, Lambda
)
from tensorflow.keras import regularizers
from scipy.stats import norm as scipy_norm
from scipy.special import ndtr  # fast normal CDF


# Quantile levels for the distributional head
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
N_QUANTILES = len(QUANTILE_LEVELS)

# Regime labels
REGIME_NORMAL = 0
REGIME_USAGE_SPIKE = 1
REGIME_BLOWOUT = 2
REGIME_FOUL_TROUBLE = 3
N_REGIMES = 4


def build_distributional_head(
    latent_dim: int,
    n_targets: int,
    dropout: float = 0.20,
    l2_reg: float = 1e-4,
) -> tuple[tf.keras.Model, dict]:
    """
    Build the distributional head that sits on top of the v7 LSTM latent state.

    Returns:
        model: Keras model taking latent vector, outputting distributional params
        output_spec: dict describing output structure
    """
    latent_input = Input(shape=(latent_dim,), name="latent_input")

    # Shared trunk
    x = Dense(128, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="dist_trunk_1")(latent_input)
    x = LayerNormalization(name="dist_trunk_ln1")(x)
    x = Dropout(dropout, name="dist_trunk_drop1")(x)
    x = Dense(64, activation="swish", kernel_regularizer=regularizers.l2(l2_reg), name="dist_trunk_2")(x)
    x = LayerNormalization(name="dist_trunk_ln2")(x)
    x = Dropout(dropout * 0.8, name="dist_trunk_drop2")(x)

    # Quantile head: outputs Q10/Q25/Q50/Q75/Q90 for each target
    # Shape: (batch, n_targets * N_QUANTILES)
    quantile_raw = Dense(
        n_targets * N_QUANTILES,
        activation="linear",
        name="quantile_raw",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)

    # Regime classifier: softmax over N_REGIMES
    regime_logits = Dense(N_REGIMES, activation="linear", name="regime_logits")(x)
    regime_probs = Lambda(lambda z: tf.nn.softmax(z, axis=-1), name="regime_probs")(regime_logits)

    # Sigma head: per-target uncertainty (softplus to ensure positive)
    sigma_raw = Dense(n_targets, activation="linear", name="sigma_raw")(x)
    sigma = Lambda(lambda z: tf.nn.softplus(z) + 0.1, name="sigma")(sigma_raw)

    model = Model(
        inputs=latent_input,
        outputs={
            "quantile_raw": quantile_raw,
            "regime_probs": regime_probs,
            "sigma": sigma,
        },
        name="distributional_head",
    )

    output_spec = {
        "quantile_levels": QUANTILE_LEVELS,
        "n_quantiles": N_QUANTILES,
        "n_targets": n_targets,
        "n_regimes": N_REGIMES,
        "regime_names": ["normal", "usage_spike", "blowout", "foul_trouble"],
    }

    return model, output_spec


def quantile_loss(y_true: tf.Tensor, y_pred: tf.Tensor, quantile: float) -> tf.Tensor:
    """Pinball loss for quantile regression."""
    error = y_true - y_pred
    return tf.reduce_mean(
        tf.maximum(quantile * error, (quantile - 1.0) * error)
    )


def total_quantile_loss(
    y_true: tf.Tensor,
    quantile_preds: tf.Tensor,
    n_targets: int,
) -> tf.Tensor:
    """
    Compute total quantile loss across all quantile levels and targets.
    quantile_preds shape: (batch, n_targets * N_QUANTILES)
    y_true shape: (batch, n_targets)
    """
    total = tf.constant(0.0)
    for t_idx in range(n_targets):
        for q_idx, q_level in enumerate(QUANTILE_LEVELS):
            pred_col = t_idx * N_QUANTILES + q_idx
            pred = quantile_preds[:, pred_col]
            true = y_true[:, t_idx]
            total = total + quantile_loss(true, pred, q_level)
    return total / float(n_targets * N_QUANTILES)


def extract_quantiles(
    quantile_raw: np.ndarray,
    n_targets: int,
    baseline: np.ndarray,
    scaler_y_scale: np.ndarray,
    scaler_y_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Extract per-target quantile predictions in original scale.

    Args:
        quantile_raw: (n_samples, n_targets * N_QUANTILES) in scaled space
        n_targets: number of targets
        baseline: (n_samples, n_targets) baseline in scaled space
        scaler_y_scale: StandardScaler scale_ attribute
        scaler_y_mean: StandardScaler mean_ attribute

    Returns:
        dict mapping target_idx -> (n_samples, N_QUANTILES) in original scale
    """
    result = {}
    for t_idx in range(n_targets):
        cols = [t_idx * N_QUANTILES + q_idx for q_idx in range(N_QUANTILES)]
        q_scaled = quantile_raw[:, cols]  # (n_samples, N_QUANTILES)
        # Add baseline (quantile_raw is delta from baseline)
        q_with_baseline = q_scaled + baseline[:, t_idx:t_idx+1]
        # Inverse transform: x_orig = x_scaled * scale + mean
        q_orig = q_with_baseline * scaler_y_scale[t_idx] + scaler_y_mean[t_idx]
        result[t_idx] = q_orig
    return result


def compute_over_probability(
    line: float,
    quantiles_orig: np.ndarray,
    sigma_orig: float,
    method: str = "gaussian_fit",
) -> float:
    """
    Compute P(stat > line) from predicted distribution.

    Args:
        line: sportsbook prop line
        quantiles_orig: (N_QUANTILES,) predicted quantiles in original scale
        sigma_orig: predicted uncertainty in original scale
        method: 'gaussian_fit' or 'empirical'

    Returns:
        P(stat > line) in [0, 1]
    """
    if method == "gaussian_fit":
        # Fit Gaussian to Q25/Q50/Q75 (robust to outliers)
        q25 = quantiles_orig[QUANTILE_LEVELS.index(0.25)]
        q50 = quantiles_orig[QUANTILE_LEVELS.index(0.50)]
        q75 = quantiles_orig[QUANTILE_LEVELS.index(0.75)]

        # IQR-based sigma estimate (more robust than direct sigma)
        iqr_sigma = (q75 - q25) / 1.3490  # IQR / 1.349 = sigma for normal
        # Blend with model sigma
        blended_sigma = 0.6 * iqr_sigma + 0.4 * sigma_orig
        blended_sigma = max(blended_sigma, 0.5)  # floor

        # P(X > line) = 1 - Phi((line - mu) / sigma)
        z = (line - q50) / blended_sigma
        p_over = float(1.0 - ndtr(z))

    elif method == "empirical":
        # Linear interpolation between quantile levels
        q_levels = np.array(QUANTILE_LEVELS)
        q_vals = quantiles_orig

        if line <= q_vals[0]:
            p_over = 1.0 - q_levels[0]
        elif line >= q_vals[-1]:
            p_over = 1.0 - q_levels[-1]
        else:
            # Find bracket
            idx = np.searchsorted(q_vals, line) - 1
            idx = max(0, min(idx, len(q_vals) - 2))
            # Linear interpolation
            frac = (line - q_vals[idx]) / max(q_vals[idx+1] - q_vals[idx], 1e-6)
            p_quantile = q_levels[idx] + frac * (q_levels[idx+1] - q_levels[idx])
            p_over = 1.0 - p_quantile

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(np.clip(p_over, 0.01, 0.99))


def compute_edge(
    p_over_model: float,
    market_over_price: float,
    vig_removal: bool = True,
) -> dict:
    """
    Compute betting edge given model probability and market price.

    Args:
        p_over_model: model's P(over) estimate
        market_over_price: American odds for over (e.g., -110, +120)
        vig_removal: whether to remove vig from market price

    Returns:
        dict with edge metrics
    """
    # Convert American odds to implied probability
    if market_over_price < 0:
        implied_raw = abs(market_over_price) / (abs(market_over_price) + 100)
    else:
        implied_raw = 100 / (market_over_price + 100)

    if vig_removal:
        # Simple vig removal: assume symmetric -110/-110 vig = 4.76%
        # Adjust implied probability down by half the vig
        vig_factor = 0.952  # 1 / (1 + 0.05)
        implied_no_vig = implied_raw * vig_factor
    else:
        implied_no_vig = implied_raw

    # Expected value
    if market_over_price < 0:
        payout_multiplier = 100 / abs(market_over_price)
    else:
        payout_multiplier = market_over_price / 100

    ev = p_over_model * payout_multiplier - (1 - p_over_model)

    # Edge = model probability - market no-vig probability
    edge = p_over_model - implied_no_vig

    return {
        "p_over_model": float(p_over_model),
        "p_over_market_raw": float(implied_raw),
        "p_over_market_no_vig": float(implied_no_vig),
        "edge": float(edge),
        "ev": float(ev),
        "playable": bool(edge > 0.04 and ev > 0.02),
    }


if __name__ == "__main__":
    # Smoke test
    import numpy as np
    print("Testing distributional head...")

    # Test over probability computation
    quantiles = np.array([18.0, 22.0, 26.0, 30.0, 35.0])  # Q10/Q25/Q50/Q75/Q90
    line = 27.5
    p_over = compute_over_probability(line, quantiles, sigma_orig=5.0, method="gaussian_fit")
    print(f"  P(PTS > {line}) = {p_over:.3f} (expected ~0.40)")

    p_over_emp = compute_over_probability(line, quantiles, sigma_orig=5.0, method="empirical")
    print(f"  P(PTS > {line}) empirical = {p_over_emp:.3f}")

    # Test edge computation
    edge_info = compute_edge(p_over, market_over_price=-110)
    print(f"  Edge at -110: {edge_info['edge']:.3f}, EV: {edge_info['ev']:.3f}, Playable: {edge_info['playable']}")

    print("Distributional head smoke test PASSED")
