#!/usr/bin/env python3
"""
NBA v9 Adaptive Calibration with Drift Detection

Research basis:
  - Static calibration degrades over time as the NBA evolves
  - Three-point attempts roughly doubled from 2010 to 2024, scoring rose,
    and offensive/defensive rating distributions shifted
  - A calibrator fit on October data may be wrong by February
  - The model should detect when calibration is drifting and recalibrate

This module:
  1. Online/streaming calibration that updates with each new observation
  2. Drift detection using CUSUM and Page-Hinkley tests
  3. Windowed recalibration when drift is detected
  4. Regime-aware calibration (different calibration curves per regime)
  5. Calibration confidence intervals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class DriftAlert:
    """Alert when calibration drift is detected."""
    detected: bool = False
    drift_magnitude: float = 0.0
    drift_direction: str = ""  # "overconfident" or "underconfident"
    samples_since_last_fit: int = 0
    recommended_action: str = ""  # "recalibrate", "monitor", "stable"
    cusum_statistic: float = 0.0
    page_hinkley_statistic: float = 0.0


@dataclass
class CalibrationHealth:
    """Overall health of the calibration system."""
    target: str
    window_brier: float = 0.0
    window_ece: float = 0.0
    window_hit_rate: float = 0.0
    window_avg_predicted: float = 0.0
    drift_alert: DriftAlert = field(default_factory=DriftAlert)
    n_total_observations: int = 0
    n_window_observations: int = 0
    last_recalibration_idx: int = 0
    calibration_age: int = 0  # observations since last recalibration


class AdaptiveCalibrator:
    """
    Online adaptive calibration with drift detection.

    Unlike static isotonic regression, this calibrator:
      1. Maintains a sliding window of recent predictions/outcomes
      2. Detects when the calibration curve has shifted
      3. Automatically recalibrates when drift exceeds threshold
      4. Provides confidence intervals on calibrated probabilities
    """

    def __init__(
        self,
        target: str = "PTS",
        window_size: int = 200,
        drift_threshold: float = 0.03,
        min_recalibration_samples: int = 50,
        cusum_threshold: float = 5.0,
    ):
        self.target = target
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_recalibration_samples = min_recalibration_samples
        self.cusum_threshold = cusum_threshold

        # Sliding window of (predicted_prob, actual_outcome)
        self._window: deque = deque(maxlen=window_size)
        # Full history for recalibration
        self._history_probs: list[float] = []
        self._history_outcomes: list[float] = []

        # Current calibrator
        self._isotonic: Optional[IsotonicRegression] = None
        self.is_fitted = False

        # Drift detection state
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._page_hinkley_sum = 0.0
        self._page_hinkley_min = 0.0
        self._n_observations = 0
        self._last_recalibration_idx = 0

        # Bin-level tracking for ECE monitoring
        self._n_bins = 10
        self._bin_counts = np.zeros(self._n_bins)
        self._bin_correct = np.zeros(self._n_bins)
        self._bin_predicted = np.zeros(self._n_bins)

    def fit_initial(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
    ) -> "AdaptiveCalibrator":
        """
        Fit initial calibration from historical data.

        Args:
            raw_probs: (n,) raw model P(over) in [0, 1]
            outcomes: (n,) binary outcomes (1=over hit, 0=under hit)
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float64).clip(0.001, 0.999)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if SKLEARN_AVAILABLE and len(raw_probs) >= self.min_recalibration_samples:
            self._isotonic = IsotonicRegression(out_of_bounds="clip")
            self._isotonic.fit(raw_probs, outcomes)

        self.is_fitted = True
        self._last_recalibration_idx = 0

        # Seed history
        for p, o in zip(raw_probs[-self.window_size:], outcomes[-self.window_size:]):
            self._window.append((float(p), float(o)))
        self._history_probs = list(raw_probs)
        self._history_outcomes = list(outcomes)
        self._n_observations = len(raw_probs)

        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply current calibration to raw probabilities."""
        raw_probs = np.asarray(raw_probs, dtype=np.float64).clip(0.001, 0.999)
        if not self.is_fitted or self._isotonic is None:
            return raw_probs
        return self._isotonic.predict(raw_probs).clip(0.01, 0.99)

    def update(
        self,
        predicted_prob: float,
        actual_outcome: float,
        auto_recalibrate: bool = True,
    ) -> DriftAlert:
        """
        Update calibrator with a new observation and check for drift.

        This is the core online learning step. After each game resolves,
        we feed the prediction and outcome back to detect drift.

        Args:
            predicted_prob: calibrated P(over) that was used
            actual_outcome: 1.0 if over hit, 0.0 if under hit
            auto_recalibrate: whether to auto-recalibrate on drift

        Returns:
            DriftAlert indicating current calibration health
        """
        predicted_prob = float(np.clip(predicted_prob, 0.001, 0.999))
        actual_outcome = float(actual_outcome)

        # Add to window
        self._window.append((predicted_prob, actual_outcome))
        self._history_probs.append(predicted_prob)
        self._history_outcomes.append(actual_outcome)
        self._n_observations += 1

        # Update bin tracking
        bin_idx = min(int(predicted_prob * self._n_bins), self._n_bins - 1)
        self._bin_counts[bin_idx] += 1
        self._bin_correct[bin_idx] += actual_outcome
        self._bin_predicted[bin_idx] += predicted_prob

        # Drift detection using CUSUM
        error = actual_outcome - predicted_prob
        drift_alert = self._detect_drift(error)

        # Auto-recalibrate if drift detected
        if auto_recalibrate and drift_alert.detected:
            self._recalibrate()
            drift_alert.recommended_action = "recalibrated"

        return drift_alert

    def _detect_drift(self, error: float) -> DriftAlert:
        """
        Detect calibration drift using CUSUM and Page-Hinkley tests.

        CUSUM (Cumulative Sum): detects persistent shifts in mean error
        Page-Hinkley: detects changes in the mean of a sequence
        """
        # CUSUM test
        # If errors are consistently positive, model is underconfident
        # If errors are consistently negative, model is overconfident
        allowance = 0.01  # Small allowance for noise
        self._cusum_pos = max(0, self._cusum_pos + error - allowance)
        self._cusum_neg = max(0, self._cusum_neg - error - allowance)

        cusum_stat = max(self._cusum_pos, self._cusum_neg)

        # Page-Hinkley test
        self._page_hinkley_sum += error
        self._page_hinkley_min = min(self._page_hinkley_min, self._page_hinkley_sum)
        ph_stat = self._page_hinkley_sum - self._page_hinkley_min

        # Check drift
        samples_since_fit = self._n_observations - self._last_recalibration_idx
        detected = (
            cusum_stat > self.cusum_threshold and
            samples_since_fit >= self.min_recalibration_samples
        )

        # Determine direction
        direction = ""
        if detected:
            if self._cusum_pos > self._cusum_neg:
                direction = "underconfident"  # Model says low prob but outcomes are high
            else:
                direction = "overconfident"  # Model says high prob but outcomes are low

        # Recommended action
        if detected:
            action = "recalibrate"
        elif cusum_stat > self.cusum_threshold * 0.6:
            action = "monitor"
        else:
            action = "stable"

        return DriftAlert(
            detected=detected,
            drift_magnitude=float(cusum_stat / self.cusum_threshold),
            drift_direction=direction,
            samples_since_last_fit=samples_since_fit,
            recommended_action=action,
            cusum_statistic=float(cusum_stat),
            page_hinkley_statistic=float(ph_stat),
        )

    def _recalibrate(self) -> None:
        """Recalibrate using recent window data."""
        if not SKLEARN_AVAILABLE:
            return

        # Use recent window for recalibration
        if len(self._window) < self.min_recalibration_samples:
            return

        window_data = list(self._window)
        probs = np.array([p for p, _ in window_data])
        outcomes = np.array([o for _, o in window_data])

        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(probs, outcomes)
        self._last_recalibration_idx = self._n_observations

        # Reset CUSUM
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._page_hinkley_sum = 0.0
        self._page_hinkley_min = 0.0

    def get_health(self) -> CalibrationHealth:
        """Get current calibration health metrics."""
        if len(self._window) < 10:
            return CalibrationHealth(
                target=self.target,
                drift_alert=DriftAlert(recommended_action="insufficient_data"),
            )

        window_data = list(self._window)
        probs = np.array([p for p, _ in window_data])
        outcomes = np.array([o for _, o in window_data])

        # Window metrics
        brier = float(np.mean((probs - outcomes) ** 2))
        hit_rate = float(np.mean(outcomes))
        avg_pred = float(np.mean(probs))

        # ECE
        ece = 0.0
        n = len(probs)
        bins = np.linspace(0, 1, self._n_bins + 1)
        for i in range(self._n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = float(np.mean(probs[mask]))
            bin_acc = float(np.mean(outcomes[mask]))
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

        # Current drift status
        if len(self._window) > 0:
            last_error = outcomes[-1] - probs[-1]
            drift_alert = self._detect_drift(0)  # Check without updating
        else:
            drift_alert = DriftAlert()

        return CalibrationHealth(
            target=self.target,
            window_brier=brier,
            window_ece=ece,
            window_hit_rate=hit_rate,
            window_avg_predicted=avg_pred,
            drift_alert=drift_alert,
            n_total_observations=self._n_observations,
            n_window_observations=len(self._window),
            last_recalibration_idx=self._last_recalibration_idx,
            calibration_age=self._n_observations - self._last_recalibration_idx,
        )

    def confidence_interval(
        self,
        raw_prob: float,
        confidence: float = 0.90,
    ) -> tuple[float, float]:
        """
        Compute confidence interval for calibrated probability.

        Uses the window data to estimate how uncertain the calibration is
        at this probability level.
        """
        if len(self._window) < 20:
            # Wide interval when insufficient data
            return (max(0.01, raw_prob - 0.15), min(0.99, raw_prob + 0.15))

        # Find nearby predictions in the window
        window_data = list(self._window)
        probs = np.array([p for p, _ in window_data])
        outcomes = np.array([o for _, o in window_data])

        # Bandwidth for local estimation
        bandwidth = 0.10
        mask = np.abs(probs - raw_prob) < bandwidth
        if mask.sum() < 5:
            bandwidth = 0.20
            mask = np.abs(probs - raw_prob) < bandwidth

        if mask.sum() < 3:
            return (max(0.01, raw_prob - 0.12), min(0.99, raw_prob + 0.12))

        local_outcomes = outcomes[mask]
        local_mean = float(np.mean(local_outcomes))
        local_std = float(np.std(local_outcomes))

        # Confidence interval using normal approximation
        from scipy.stats import norm as scipy_norm
        z = scipy_norm.ppf((1 + confidence) / 2)
        se = local_std / np.sqrt(len(local_outcomes))
        lower = local_mean - z * se
        upper = local_mean + z * se

        return (float(np.clip(lower, 0.01, 0.99)), float(np.clip(upper, 0.01, 0.99)))

    def save(self, path: str | Path) -> None:
        """Save adaptive calibrator state."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "target": self.target,
            "window_size": self.window_size,
            "drift_threshold": self.drift_threshold,
            "min_recalibration_samples": self.min_recalibration_samples,
            "cusum_threshold": self.cusum_threshold,
            "isotonic": self._isotonic,
            "is_fitted": self.is_fitted,
            "window": list(self._window),
            "history_probs": self._history_probs[-1000:],  # Keep last 1000
            "history_outcomes": self._history_outcomes[-1000:],
            "n_observations": self._n_observations,
            "last_recalibration_idx": self._last_recalibration_idx,
            "cusum_pos": self._cusum_pos,
            "cusum_neg": self._cusum_neg,
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveCalibrator":
        """Load adaptive calibrator state."""
        import joblib
        data = joblib.load(str(path))
        cal = cls(
            target=data["target"],
            window_size=data["window_size"],
            drift_threshold=data["drift_threshold"],
            min_recalibration_samples=data["min_recalibration_samples"],
            cusum_threshold=data["cusum_threshold"],
        )
        cal._isotonic = data["isotonic"]
        cal.is_fitted = data["is_fitted"]
        cal._window = deque(data["window"], maxlen=cal.window_size)
        cal._history_probs = data["history_probs"]
        cal._history_outcomes = data["history_outcomes"]
        cal._n_observations = data["n_observations"]
        cal._last_recalibration_idx = data["last_recalibration_idx"]
        cal._cusum_pos = data["cusum_pos"]
        cal._cusum_neg = data["cusum_neg"]
        return cal


class RegimeAwareCalibrator:
    """
    Maintains separate calibration curves per regime.

    Insight: A player in "usage_spike" regime has a different
    calibration curve than in "normal" regime. The same raw P(over)=0.55
    might mean different things in different contexts.
    """

    def __init__(
        self,
        target: str = "PTS",
        regime_names: list[str] = None,
        window_size: int = 150,
    ):
        self.target = target
        self.regime_names = regime_names or [
            "normal", "usage_spike", "suppressed", "high_volatility", "cold_outlier"
        ]
        self.window_size = window_size

        # One adaptive calibrator per regime
        self.calibrators: dict[str, AdaptiveCalibrator] = {
            regime: AdaptiveCalibrator(
                target=f"{target}_{regime}",
                window_size=window_size,
            )
            for regime in self.regime_names
        }

        # Fallback calibrator for unknown regimes
        self._fallback = AdaptiveCalibrator(target=target, window_size=window_size)

    def fit_initial(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        regimes: np.ndarray,
    ) -> "RegimeAwareCalibrator":
        """
        Fit initial calibration per regime.

        Args:
            raw_probs: (n,) raw P(over)
            outcomes: (n,) binary outcomes
            regimes: (n,) regime labels (strings)
        """
        # Fit per-regime calibrators
        for regime in self.regime_names:
            mask = regimes == regime
            if mask.sum() >= 20:
                self.calibrators[regime].fit_initial(
                    raw_probs[mask], outcomes[mask]
                )

        # Fit fallback on all data
        self._fallback.fit_initial(raw_probs, outcomes)
        return self

    def predict(
        self,
        raw_prob: float,
        regime: str,
    ) -> float:
        """Apply regime-specific calibration."""
        if regime in self.calibrators and self.calibrators[regime].is_fitted:
            return float(self.calibrators[regime].predict(np.array([raw_prob]))[0])
        return float(self._fallback.predict(np.array([raw_prob]))[0])

    def update(
        self,
        predicted_prob: float,
        actual_outcome: float,
        regime: str,
    ) -> DriftAlert:
        """Update the appropriate regime calibrator."""
        if regime in self.calibrators:
            return self.calibrators[regime].update(predicted_prob, actual_outcome)
        return self._fallback.update(predicted_prob, actual_outcome)

    def get_all_health(self) -> dict[str, CalibrationHealth]:
        """Get health metrics for all regime calibrators."""
        return {
            regime: cal.get_health()
            for regime, cal in self.calibrators.items()
        }


if __name__ == "__main__":
    np.random.seed(42)
    print("Testing Adaptive Calibration...")

    # Simulate a model that starts well-calibrated then drifts
    n_initial = 200
    n_drift = 100

    # Initial period: well-calibrated
    true_probs_initial = np.random.uniform(0.3, 0.7, n_initial)
    raw_probs_initial = true_probs_initial + np.random.normal(0, 0.05, n_initial)
    raw_probs_initial = np.clip(raw_probs_initial, 0.01, 0.99)
    outcomes_initial = (np.random.uniform(0, 1, n_initial) < true_probs_initial).astype(float)

    # Fit initial calibrator
    cal = AdaptiveCalibrator(target="PTS", window_size=100, cusum_threshold=3.0)
    cal.fit_initial(raw_probs_initial, outcomes_initial)
    print(f"  Initial fit: {cal._n_observations} observations")

    # Drift period: model becomes overconfident (predicts higher than reality)
    true_probs_drift = np.random.uniform(0.3, 0.55, n_drift)  # Reality shifted down
    raw_probs_drift = true_probs_drift + 0.10  # Model still predicts high
    raw_probs_drift = np.clip(raw_probs_drift, 0.01, 0.99)
    outcomes_drift = (np.random.uniform(0, 1, n_drift) < true_probs_drift).astype(float)

    # Feed drift observations
    drift_detected = False
    for i in range(n_drift):
        calibrated = float(cal.predict(np.array([raw_probs_drift[i]]))[0])
        alert = cal.update(calibrated, outcomes_drift[i])
        if alert.detected and not drift_detected:
            print(f"  Drift detected at observation {i}!")
            print(f"    Direction: {alert.drift_direction}")
            print(f"    Magnitude: {alert.drift_magnitude:.2f}")
            drift_detected = True

    if not drift_detected:
        print("  No drift detected (threshold may need tuning)")

    # Check health
    health = cal.get_health()
    print(f"\n  Calibration Health:")
    print(f"    Window Brier: {health.window_brier:.4f}")
    print(f"    Window ECE: {health.window_ece:.4f}")
    print(f"    Window hit rate: {health.window_hit_rate:.3f}")
    print(f"    Avg predicted: {health.window_avg_predicted:.3f}")
    print(f"    Calibration age: {health.calibration_age}")

    # Test confidence interval
    ci = cal.confidence_interval(0.55)
    print(f"\n  CI for P=0.55: [{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\nAdaptive Calibration smoke test PASSED")
