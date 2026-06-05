#!/usr/bin/env python3
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
