#!/usr/bin/env python3
"""
NBA v9 Residual Chaos Measurement

Research basis:
  "How I would quantify chaos in your NBA system:
   I would not measure chaos by asking 'Can the model win?'
   I would measure it as unexplained entropy after controlling for
   market and known basketball state."

This module answers the fundamental question:
  "Is the NBA chaotic, or is my model missing state?"

Framework:
  1. Market baseline: closing odds implied probability
  2. Basketball model: team, player, matchup, rest, pace, injury, rotation
  3. Market + model fusion: market probability + model residual correction
  4. Residual test: actual outcome - calibrated probability
  5. Chaos estimate: how random the residuals remain after every known signal

Metrics:
  - Brier score decomposition (reliability, resolution, uncertainty)
  - Log loss
  - Expected calibration error
  - Mutual information after controlling for market
  - Residual autocorrelation (are errors predictable?)
  - Permutation entropy of residual signs
  - Rolling out-of-sample degradation
  - CLV correlation with final profit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import math


@dataclass
class ChaosMetrics:
    """Comprehensive chaos/predictability measurement for the system."""
    # Core metrics
    brier_score: float = 0.0
    log_loss: float = 0.0
    ece: float = 0.0

    # Brier decomposition
    brier_reliability: float = 0.0    # How well-calibrated (lower = better)
    brier_resolution: float = 0.0     # How much model separates outcomes (higher = better)
    brier_uncertainty: float = 0.0    # Inherent uncertainty of the problem

    # Residual analysis
    residual_mean: float = 0.0        # Should be ~0 if calibrated
    residual_std: float = 0.0         # Irreducible noise level
    residual_autocorrelation: float = 0.0  # Are errors predictable? (should be ~0)
    residual_skewness: float = 0.0    # Asymmetry in errors

    # Entropy metrics
    permutation_entropy: float = 0.0  # Randomness of residual sign sequence
    max_permutation_entropy: float = 0.0
    normalized_entropy: float = 0.0   # 1.0 = pure chaos, 0.0 = fully predictable

    # Information metrics
    mutual_information_vs_market: float = 0.0  # Does model add info beyond market?
    model_lift_over_market: float = 0.0        # Brier improvement over market baseline

    # Stability metrics
    rolling_brier_std: float = 0.0    # How stable is performance over time?
    degradation_rate: float = 0.0     # Is model getting worse over time?

    # Practical interpretation
    chaos_level: str = ""             # "low", "moderate", "high", "extreme"
    exploitable_signal: float = 0.0   # Estimated fraction of exploitable signal
    interpretation: str = ""          # Human-readable summary


@dataclass
class ResidualDiagnostics:
    """Detailed residual analysis for a specific stat/market segment."""
    target: str
    n_observations: int = 0
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    actual_outcomes: np.ndarray = field(default_factory=lambda: np.array([]))
    market_probs: np.ndarray = field(default_factory=lambda: np.array([]))


def compute_brier_decomposition(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, float]:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.

    Brier = Reliability - Resolution + Uncertainty

    - Reliability: measures calibration (lower = better calibrated)
    - Resolution: measures how well model separates outcomes (higher = better)
    - Uncertainty: inherent unpredictability of the base rate

    This decomposition tells you WHERE the model is failing:
      - High reliability = poor calibration (fixable)
      - Low resolution = model doesn't separate winners from losers (harder)
      - High uncertainty = the problem itself is hard (irreducible)
    """
    n = len(outcomes)
    base_rate = np.mean(outcomes)
    uncertainty = base_rate * (1 - base_rate)

    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        n_k = mask.sum()
        if n_k == 0:
            continue
        bin_mean_pred = np.mean(predicted_probs[mask])
        bin_mean_outcome = np.mean(outcomes[mask])
        weight = n_k / n

        reliability += weight * (bin_mean_pred - bin_mean_outcome) ** 2
        resolution += weight * (bin_mean_outcome - base_rate) ** 2

    return float(reliability), float(resolution), float(uncertainty)


def compute_permutation_entropy(
    residuals: np.ndarray,
    order: int = 3,
    delay: int = 1,
) -> tuple[float, float]:
    """
    Compute permutation entropy of residual signs.

    Permutation entropy measures the complexity/randomness of a time series.
    - High PE (close to max) = residuals are random (good - no exploitable pattern)
    - Low PE = residuals have structure (bad - model is missing something)

    Args:
        residuals: time series of prediction errors
        order: embedding dimension (3-5 typical)
        delay: time delay

    Returns:
        (permutation_entropy, max_possible_entropy)
    """
    n = len(residuals)
    if n < order * delay + 1:
        max_pe = np.log2(math.factorial(order))
        return max_pe, max_pe  # Assume random if too few samples

    # Create ordinal patterns
    n_patterns = math.factorial(order)
    pattern_counts = {}

    for i in range(n - (order - 1) * delay):
        # Extract subsequence
        indices = [i + j * delay for j in range(order)]
        subsequence = residuals[indices]
        # Get ordinal pattern (rank order)
        pattern = tuple(np.argsort(subsequence))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Compute entropy
    total = sum(pattern_counts.values())
    pe = 0.0
    for count in pattern_counts.values():
        p = count / total
        if p > 0:
            pe -= p * np.log2(p)

    max_pe = np.log2(n_patterns)
    return float(pe), float(max_pe)


def compute_residual_autocorrelation(
    residuals: np.ndarray,
    max_lag: int = 5,
) -> float:
    """
    Compute autocorrelation of residuals.

    If residuals are autocorrelated, the model is missing temporal structure.
    Ideally, residuals should be white noise (autocorrelation ≈ 0).

    Returns the maximum absolute autocorrelation across lags 1..max_lag.
    """
    n = len(residuals)
    if n < max_lag + 10:
        return 0.0

    residuals_centered = residuals - np.mean(residuals)
    var = np.var(residuals_centered)
    if var < 1e-10:
        return 0.0

    max_autocorr = 0.0
    for lag in range(1, max_lag + 1):
        autocorr = np.mean(residuals_centered[lag:] * residuals_centered[:-lag]) / var
        max_autocorr = max(max_autocorr, abs(autocorr))

    return float(max_autocorr)


def compute_mutual_information_discrete(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 5,
) -> float:
    """
    Estimate mutual information between model predictions and outcomes,
    after controlling for market information.

    This answers: "Does the model know something the market doesn't?"

    High MI = model adds information beyond market
    Low MI = model is just echoing market prices
    """
    n = len(outcomes)
    if n < 50:
        return 0.0

    # Compute model residual (what model adds beyond market)
    model_residual = model_probs - market_probs

    # Discretize residual into bins
    bins = np.linspace(np.percentile(model_residual, 5),
                       np.percentile(model_residual, 95), n_bins + 1)
    residual_binned = np.digitize(model_residual, bins) - 1
    residual_binned = np.clip(residual_binned, 0, n_bins - 1)

    # Compute MI between binned residual and outcomes
    # MI(X;Y) = H(Y) - H(Y|X)
    h_y = _binary_entropy(np.mean(outcomes))

    # H(Y|X) = sum_x P(X=x) * H(Y|X=x)
    h_y_given_x = 0.0
    for b in range(n_bins):
        mask = residual_binned == b
        if mask.sum() < 3:
            continue
        p_x = mask.sum() / n
        h_y_x = _binary_entropy(np.mean(outcomes[mask]))
        h_y_given_x += p_x * h_y_x

    mi = max(0.0, h_y - h_y_given_x)
    return float(mi)


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compute_rolling_degradation(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
    window: int = 50,
) -> tuple[float, float]:
    """
    Compute rolling Brier score to detect model degradation over time.

    Returns:
        (rolling_brier_std, degradation_rate)
        - rolling_brier_std: how variable performance is
        - degradation_rate: slope of rolling Brier (positive = getting worse)
    """
    n = len(outcomes)
    if n < window * 2:
        return 0.0, 0.0

    rolling_briers = []
    for i in range(window, n):
        window_probs = predicted_probs[i - window:i]
        window_outcomes = outcomes[i - window:i]
        brier = float(np.mean((window_probs - window_outcomes) ** 2))
        rolling_briers.append(brier)

    rolling_briers = np.array(rolling_briers)
    std = float(np.std(rolling_briers))

    # Linear regression for trend
    x = np.arange(len(rolling_briers))
    if len(x) > 1:
        slope = float(np.polyfit(x, rolling_briers, 1)[0])
    else:
        slope = 0.0

    return std, slope


class ResidualChaosAnalyzer:
    """
    Comprehensive chaos/predictability analyzer for the NBA prop system.

    This is the diagnostic tool that answers:
      "Is the remaining unpredictability truly chaotic (irreducible),
       or is the model missing exploitable state?"

    Usage:
        analyzer = ResidualChaosAnalyzer()
        analyzer.add_observations(predicted_probs, outcomes, market_probs)
        metrics = analyzer.compute_chaos_metrics()
        report = analyzer.generate_report()
    """

    def __init__(self, target: str = "ALL"):
        self.target = target
        self._predicted_probs: list[float] = []
        self._outcomes: list[float] = []
        self._market_probs: list[float] = []
        self._timestamps: list[str] = []

    def add_observations(
        self,
        predicted_probs: np.ndarray,
        outcomes: np.ndarray,
        market_probs: np.ndarray = None,
        timestamps: list[str] = None,
    ) -> None:
        """Add a batch of observations for analysis."""
        predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if market_probs is None:
            market_probs = np.full_like(predicted_probs, 0.5)
        else:
            market_probs = np.asarray(market_probs, dtype=np.float64)

        self._predicted_probs.extend(predicted_probs.tolist())
        self._outcomes.extend(outcomes.tolist())
        self._market_probs.extend(market_probs.tolist())

        if timestamps:
            self._timestamps.extend(timestamps)
        else:
            self._timestamps.extend([""] * len(predicted_probs))

    def compute_chaos_metrics(self) -> ChaosMetrics:
        """Compute comprehensive chaos metrics."""
        if len(self._outcomes) < 30:
            return ChaosMetrics(
                chaos_level="insufficient_data",
                interpretation="Need at least 30 observations for meaningful analysis.",
            )

        probs = np.array(self._predicted_probs)
        outcomes = np.array(self._outcomes)
        market = np.array(self._market_probs)

        # Residuals
        residuals = outcomes - probs

        # Core metrics
        brier = float(np.mean((probs - outcomes) ** 2))
        log_loss = float(-np.mean(
            outcomes * np.log(np.clip(probs, 1e-6, 1)) +
            (1 - outcomes) * np.log(np.clip(1 - probs, 1e-6, 1))
        ))

        # ECE
        ece = self._compute_ece(probs, outcomes)

        # Brier decomposition
        reliability, resolution, uncertainty = compute_brier_decomposition(probs, outcomes)

        # Residual analysis
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))
        residual_autocorr = compute_residual_autocorrelation(residuals)
        residual_skew = float(pd.Series(residuals).skew())

        # Permutation entropy
        pe, max_pe = compute_permutation_entropy(residuals)
        normalized_entropy = pe / max_pe if max_pe > 0 else 1.0

        # Mutual information vs market
        mi = compute_mutual_information_discrete(probs, market, outcomes)

        # Model lift over market
        market_brier = float(np.mean((market - outcomes) ** 2))
        model_lift = market_brier - brier  # Positive = model is better

        # Rolling degradation
        rolling_std, degradation = compute_rolling_degradation(probs, outcomes)

        # Interpret chaos level
        chaos_level, exploitable_signal, interpretation = self._interpret_chaos(
            brier=brier,
            normalized_entropy=normalized_entropy,
            residual_autocorr=residual_autocorr,
            model_lift=model_lift,
            resolution=resolution,
            uncertainty=uncertainty,
        )

        return ChaosMetrics(
            brier_score=brier,
            log_loss=log_loss,
            ece=ece,
            brier_reliability=reliability,
            brier_resolution=resolution,
            brier_uncertainty=uncertainty,
            residual_mean=residual_mean,
            residual_std=residual_std,
            residual_autocorrelation=residual_autocorr,
            residual_skewness=residual_skew,
            permutation_entropy=pe,
            max_permutation_entropy=max_pe,
            normalized_entropy=normalized_entropy,
            mutual_information_vs_market=mi,
            model_lift_over_market=model_lift,
            rolling_brier_std=rolling_std,
            degradation_rate=degradation,
            chaos_level=chaos_level,
            exploitable_signal=exploitable_signal,
            interpretation=interpretation,
        )

    def _compute_ece(self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(outcomes)
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = float(np.mean(probs[mask]))
            bin_acc = float(np.mean(outcomes[mask]))
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
        return float(ece)

    def _interpret_chaos(
        self,
        brier: float,
        normalized_entropy: float,
        residual_autocorr: float,
        model_lift: float,
        resolution: float,
        uncertainty: float,
    ) -> tuple[str, float, str]:
        """
        Interpret chaos metrics into actionable conclusions.

        Returns: (chaos_level, exploitable_signal, interpretation)
        """
        # Chaos level based on normalized entropy and residual structure
        if normalized_entropy > 0.95 and residual_autocorr < 0.05:
            chaos_level = "high"
        elif normalized_entropy > 0.85 and residual_autocorr < 0.10:
            chaos_level = "moderate"
        elif normalized_entropy > 0.70:
            chaos_level = "moderate"
        else:
            chaos_level = "low"

        # Exploitable signal estimate
        # Based on resolution relative to uncertainty
        if uncertainty > 0:
            signal_ratio = resolution / uncertainty
        else:
            signal_ratio = 0.0

        exploitable_signal = min(1.0, signal_ratio)

        # Interpretation
        parts = []

        if model_lift > 0.01:
            parts.append(f"Model adds {model_lift:.4f} Brier improvement over market.")
        elif model_lift > 0:
            parts.append("Model marginally improves on market baseline.")
        else:
            parts.append("Model does NOT improve on market baseline - edge may not exist.")

        if residual_autocorr > 0.10:
            parts.append(
                f"Residuals show autocorrelation ({residual_autocorr:.3f}) - "
                "model is missing temporal structure."
            )
        else:
            parts.append("Residuals appear independent - no obvious missing temporal signal.")

        if normalized_entropy > 0.90:
            parts.append(
                "Residual entropy is near-maximum - remaining variance is effectively random."
            )
        elif normalized_entropy > 0.75:
            parts.append(
                "Moderate residual entropy - some structure may remain exploitable."
            )
        else:
            parts.append(
                "Low residual entropy - significant exploitable structure remains."
            )

        if brier < 0.22:
            parts.append("Overall Brier score is competitive for prop betting.")
        elif brier < 0.25:
            parts.append("Brier score is acceptable but has room for improvement.")
        else:
            parts.append("Brier score is high - model needs significant improvement.")

        interpretation = " ".join(parts)
        return chaos_level, exploitable_signal, interpretation

    def generate_report(self) -> str:
        """Generate a human-readable chaos analysis report."""
        metrics = self.compute_chaos_metrics()

        lines = [
            "=" * 60,
            f"RESIDUAL CHAOS ANALYSIS - {self.target}",
            "=" * 60,
            "",
            f"Observations: {len(self._outcomes)}",
            "",
            "--- Core Metrics ---",
            f"  Brier Score:     {metrics.brier_score:.4f}",
            f"  Log Loss:        {metrics.log_loss:.4f}",
            f"  ECE:             {metrics.ece:.4f}",
            "",
            "--- Brier Decomposition ---",
            f"  Reliability:     {metrics.brier_reliability:.4f} (calibration error, lower=better)",
            f"  Resolution:      {metrics.brier_resolution:.4f} (separation power, higher=better)",
            f"  Uncertainty:     {metrics.brier_uncertainty:.4f} (inherent difficulty)",
            "",
            "--- Residual Analysis ---",
            f"  Mean:            {metrics.residual_mean:+.4f} (should be ~0)",
            f"  Std:             {metrics.residual_std:.4f} (irreducible noise)",
            f"  Autocorrelation: {metrics.residual_autocorrelation:.4f} (should be ~0)",
            f"  Skewness:        {metrics.residual_skewness:+.3f}",
            "",
            "--- Entropy & Information ---",
            f"  Permutation Entropy:  {metrics.permutation_entropy:.3f} / {metrics.max_permutation_entropy:.3f}",
            f"  Normalized Entropy:   {metrics.normalized_entropy:.3f} (1.0=pure chaos)",
            f"  MI vs Market:         {metrics.mutual_information_vs_market:.4f} bits",
            f"  Model Lift vs Market: {metrics.model_lift_over_market:+.4f} Brier",
            "",
            "--- Stability ---",
            f"  Rolling Brier Std:    {metrics.rolling_brier_std:.4f}",
            f"  Degradation Rate:     {metrics.degradation_rate:+.6f} per observation",
            "",
            "--- Conclusion ---",
            f"  Chaos Level:          {metrics.chaos_level.upper()}",
            f"  Exploitable Signal:   {metrics.exploitable_signal:.1%}",
            f"  {metrics.interpretation}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def save_report(self, path: str | Path) -> None:
        """Save chaos analysis report and metrics to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_chaos_metrics()
        report = self.generate_report()

        # Save report text
        (path.parent / f"{path.stem}_report.txt").write_text(report, encoding="utf-8")

        # Save metrics as JSON
        metrics_dict = {
            k: v for k, v in metrics.__dict__.items()
            if not isinstance(v, np.ndarray)
        }
        (path.parent / f"{path.stem}_metrics.json").write_text(
            json.dumps(metrics_dict, indent=2, default=str), encoding="utf-8"
        )


if __name__ == "__main__":
    np.random.seed(42)
    print("Testing Residual Chaos Analyzer...")

    # Simulate a moderately predictive model
    n = 300
    true_probs = np.random.uniform(0.35, 0.65, n)
    # Model is decent but not perfect
    model_probs = true_probs + np.random.normal(0, 0.08, n)
    model_probs = np.clip(model_probs, 0.05, 0.95)
    # Market is slightly worse
    market_probs = true_probs + np.random.normal(0.02, 0.10, n)
    market_probs = np.clip(market_probs, 0.05, 0.95)
    # Outcomes
    outcomes = (np.random.uniform(0, 1, n) < true_probs).astype(float)

    analyzer = ResidualChaosAnalyzer(target="PTS")
    analyzer.add_observations(model_probs, outcomes, market_probs)

    report = analyzer.generate_report()
    print(report)

    print("\nResidual Chaos Analyzer smoke test PASSED")
