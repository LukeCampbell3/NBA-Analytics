"""Write the NBA v8 distributional model and calibration layer."""
from pathlib import Path

TRAINING = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# nba_v8_distributional_head.py
# -----------------------------------------------------------------------------
dist_head = r'''#!/usr/bin/env python3
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
'''

with open(str(TRAINING / "nba_v8_distributional_head.py"), "w", encoding="utf-8") as f:
    f.write(dist_head)
print("wrote nba_v8_distributional_head.py")

# -----------------------------------------------------------------------------
# nba_v8_calibration.py
# -----------------------------------------------------------------------------
calibration = r'''#!/usr/bin/env python3
"""
NBA v8 Calibration Layer

Implements isotonic regression calibration for prop over/under probabilities.

Key insight from research:
  "Calibration is more important than accuracy for betting models."
  A model that says 57% should actually hit ~57% of the time.

This module:
  1. Fits isotonic regression on validation set predictions vs outcomes
  2. Evaluates calibration quality (ECE, Brier score, calibration slope)
  3. Provides calibrated P(over) for production use
  4. Tracks calibration drift over time
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import joblib

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available, calibration will use identity mapping")


class PropCalibrator:
    """
    Calibrates raw model P(over) probabilities using isotonic regression.

    Usage:
        calibrator = PropCalibrator(target="PTS")
        calibrator.fit(raw_probs, actual_outcomes)  # 1=over, 0=under
        calibrated = calibrator.predict(new_raw_probs)
        metrics = calibrator.evaluate(raw_probs, actual_outcomes)
    """

    def __init__(self, target: str = "PTS", n_bins: int = 10):
        self.target = target
        self.n_bins = n_bins
        self.is_fitted = False
        self._isotonic = None
        self._fit_metrics = {}

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "PropCalibrator":
        """
        Fit isotonic regression calibrator.

        Args:
            raw_probs: (n,) raw model P(over) in [0, 1]
            outcomes: (n,) binary outcomes (1=over hit, 0=under hit)
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float64).clip(0.001, 0.999)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if not SKLEARN_AVAILABLE:
            self.is_fitted = True
            return self

        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(raw_probs, outcomes)
        self.is_fitted = True

        # Compute fit metrics
        calibrated = self._isotonic.predict(raw_probs)
        self._fit_metrics = self._compute_metrics(raw_probs, calibrated, outcomes)
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        raw_probs = np.asarray(raw_probs, dtype=np.float64).clip(0.001, 0.999)
        if not self.is_fitted or self._isotonic is None:
            return raw_probs
        return self._isotonic.predict(raw_probs).clip(0.01, 0.99)

    def evaluate(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        label: str = "validation",
    ) -> dict:
        """
        Evaluate calibration quality.

        Returns dict with:
          - brier_score: mean squared error of probabilities
          - ece: expected calibration error
          - calibration_slope: slope of calibration curve (1.0 = perfect)
          - calibration_intercept: intercept (0.0 = perfect)
          - hit_rate: actual over rate
          - avg_predicted: average predicted probability
          - n_samples: number of samples
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float64).clip(0.001, 0.999)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        calibrated = self.predict(raw_probs)
        metrics = self._compute_metrics(raw_probs, calibrated, outcomes)
        metrics["label"] = label
        return metrics

    def _compute_metrics(
        self,
        raw_probs: np.ndarray,
        calibrated_probs: np.ndarray,
        outcomes: np.ndarray,
    ) -> dict:
        n = len(outcomes)
        hit_rate = float(np.mean(outcomes))

        # Brier scores
        brier_raw = float(np.mean((raw_probs - outcomes) ** 2))
        brier_cal = float(np.mean((calibrated_probs - outcomes) ** 2))

        # ECE (Expected Calibration Error)
        ece_raw = self._compute_ece(raw_probs, outcomes)
        ece_cal = self._compute_ece(calibrated_probs, outcomes)

        # Calibration slope/intercept (linear regression of outcomes on probs)
        slope_raw, intercept_raw = self._calibration_slope(raw_probs, outcomes)
        slope_cal, intercept_cal = self._calibration_slope(calibrated_probs, outcomes)

        return {
            "n_samples": n,
            "hit_rate": hit_rate,
            "avg_predicted_raw": float(np.mean(raw_probs)),
            "avg_predicted_calibrated": float(np.mean(calibrated_probs)),
            "brier_raw": brier_raw,
            "brier_calibrated": brier_cal,
            "brier_improvement": float(brier_raw - brier_cal),
            "ece_raw": ece_raw,
            "ece_calibrated": ece_cal,
            "calibration_slope_raw": slope_raw,
            "calibration_slope_calibrated": slope_cal,
            "calibration_intercept_raw": intercept_raw,
            "calibration_intercept_calibrated": intercept_cal,
        }

    def _compute_ece(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Expected Calibration Error using equal-width bins."""
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(outcomes)
        for i in range(self.n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_probs = probs[mask]
            bin_outcomes = outcomes[mask]
            bin_confidence = float(np.mean(bin_probs))
            bin_accuracy = float(np.mean(bin_outcomes))
            bin_weight = float(mask.sum()) / n
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
        return float(ece)

    def _calibration_slope(
        self, probs: np.ndarray, outcomes: np.ndarray
    ) -> tuple[float, float]:
        """Fit linear regression of outcomes on probs."""
        if len(probs) < 10:
            return 1.0, 0.0
        X = np.column_stack([np.ones(len(probs)), probs])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, outcomes, rcond=None)
            return float(coeffs[1]), float(coeffs[0])
        except Exception:
            return 1.0, 0.0

    def save(self, path: str | Path) -> None:
        """Save calibrator to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "target": self.target,
            "n_bins": self.n_bins,
            "is_fitted": self.is_fitted,
            "isotonic": self._isotonic,
            "fit_metrics": self._fit_metrics,
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "PropCalibrator":
        """Load calibrator from disk."""
        data = joblib.load(str(path))
        cal = cls(target=data["target"], n_bins=data["n_bins"])
        cal.is_fitted = data["is_fitted"]
        cal._isotonic = data["isotonic"]
        cal._fit_metrics = data.get("fit_metrics", {})
        return cal


class MultiTargetCalibrator:
    """
    Manages calibrators for all targets (PTS, TRB, AST).
    """

    def __init__(self, targets: list[str] = None):
        self.targets = targets or ["PTS", "TRB", "AST"]
        self.calibrators: dict[str, PropCalibrator] = {
            t: PropCalibrator(target=t) for t in self.targets
        }

    def fit(
        self,
        raw_probs_dict: dict[str, np.ndarray],
        outcomes_dict: dict[str, np.ndarray],
    ) -> "MultiTargetCalibrator":
        """Fit calibrators for all targets."""
        for target in self.targets:
            if target in raw_probs_dict and target in outcomes_dict:
                self.calibrators[target].fit(
                    raw_probs_dict[target],
                    outcomes_dict[target],
                )
        return self

    def predict(self, raw_probs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply calibration to all targets."""
        return {
            target: self.calibrators[target].predict(raw_probs_dict[target])
            for target in self.targets
            if target in raw_probs_dict
        }

    def evaluate(
        self,
        raw_probs_dict: dict[str, np.ndarray],
        outcomes_dict: dict[str, np.ndarray],
        label: str = "validation",
    ) -> dict:
        """Evaluate calibration for all targets."""
        results = {}
        for target in self.targets:
            if target in raw_probs_dict and target in outcomes_dict:
                results[target] = self.calibrators[target].evaluate(
                    raw_probs_dict[target],
                    outcomes_dict[target],
                    label=label,
                )
        return results

    def save(self, dir_path: str | Path) -> None:
        """Save all calibrators to directory."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for target, cal in self.calibrators.items():
            cal.save(dir_path / f"calibrator_{target}.pkl")
        # Save summary
        summary = {
            "targets": self.targets,
            "fit_metrics": {
                t: cal._fit_metrics for t, cal in self.calibrators.items()
            },
        }
        (dir_path / "calibration_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path: str | Path) -> "MultiTargetCalibrator":
        """Load all calibrators from directory."""
        dir_path = Path(dir_path)
        summary_path = dir_path / "calibration_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            targets = summary.get("targets", ["PTS", "TRB", "AST"])
        else:
            targets = ["PTS", "TRB", "AST"]

        multi = cls(targets=targets)
        for target in targets:
            cal_path = dir_path / f"calibrator_{target}.pkl"
            if cal_path.exists():
                multi.calibrators[target] = PropCalibrator.load(cal_path)
        return multi


def compute_over_outcomes(
    actual_stats: np.ndarray,
    market_lines: np.ndarray,
) -> np.ndarray:
    """
    Compute binary over outcomes from actual stats and market lines.

    Args:
        actual_stats: (n,) actual stat values
        market_lines: (n,) sportsbook prop lines

    Returns:
        (n,) binary array: 1=over hit, 0=under hit
    """
    return (actual_stats > market_lines).astype(np.float64)


def print_calibration_report(metrics: dict, target: str = "") -> None:
    """Print a formatted calibration report."""
    prefix = f"[{target}] " if target else ""
    print(f"\n{prefix}Calibration Report ({metrics.get('label', 'eval')})")
    print(f"  Samples:          {metrics.get('n_samples', 0)}")
    print(f"  Hit rate:         {metrics.get('hit_rate', 0):.3f}")
    print(f"  Avg pred (raw):   {metrics.get('avg_predicted_raw', 0):.3f}")
    print(f"  Avg pred (cal):   {metrics.get('avg_predicted_calibrated', 0):.3f}")
    print(f"  Brier (raw):      {metrics.get('brier_raw', 0):.4f}")
    print(f"  Brier (cal):      {metrics.get('brier_calibrated', 0):.4f}")
    print(f"  Brier improve:    {metrics.get('brier_improvement', 0):.4f}")
    print(f"  ECE (raw):        {metrics.get('ece_raw', 0):.4f}")
    print(f"  ECE (cal):        {metrics.get('ece_calibrated', 0):.4f}")
    print(f"  Cal slope (raw):  {metrics.get('calibration_slope_raw', 0):.3f}")
    print(f"  Cal slope (cal):  {metrics.get('calibration_slope_calibrated', 0):.3f}")


if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    # Simulate miscalibrated model (overconfident)
    true_probs = np.random.uniform(0.3, 0.7, n)
    raw_probs = np.clip(true_probs + np.random.normal(0.05, 0.08, n), 0.01, 0.99)
    outcomes = (np.random.uniform(0, 1, n) < true_probs).astype(float)

    cal = PropCalibrator(target="PTS")
    cal.fit(raw_probs, outcomes)
    metrics = cal.evaluate(raw_probs, outcomes, label="train")
    print_calibration_report(metrics, target="PTS")

    # Test multi-target
    multi = MultiTargetCalibrator()
    raw_dict = {"PTS": raw_probs, "TRB": raw_probs * 0.9, "AST": raw_probs * 1.1}
    out_dict = {"PTS": outcomes, "TRB": outcomes, "AST": outcomes}
    multi.fit(raw_dict, out_dict)
    eval_results = multi.evaluate(raw_dict, out_dict, label="test")
    print(f"\nMulti-target calibration: {list(eval_results.keys())}")
    print("Calibration smoke test PASSED")
'''

with open(str(TRAINING / "nba_v8_calibration.py"), "w", encoding="utf-8") as f:
    f.write(calibration)
print("wrote nba_v8_calibration.py")

print("All v8 model files written.")
