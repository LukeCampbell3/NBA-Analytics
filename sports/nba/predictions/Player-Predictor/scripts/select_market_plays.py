#!/usr/bin/env python3
"""
Rank upcoming market plays using historical model-vs-market edge behavior.

Inputs:
- upcoming slate CSV from scripts/build_upcoming_slate.py
- historical row-level CSV from scripts/backtest_inference_accuracy.py --csv-out

The selector:
- computes target-specific disagreement percentiles
- applies conservative per-target thresholds
- maps each candidate to an expected win rate from historical backtests
- ranks filtered plays for decision use
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parent.parent
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import build_candidate_id
from research.market_quality.priced_event_ledger import build_priced_event_ledger_frame

try:
    from decision_engine.line_decision import (
        LineDecisionConfig,
        build_line_decision_lookup,
        estimate_line_decision,
    )
except Exception:  # pragma: no cover - fallback for standalone execution
    LineDecisionConfig = None

    def build_line_decision_lookup(history_df: pd.DataFrame) -> dict[str, dict]:
        return {}

    def estimate_line_decision(**kwargs) -> dict[str, Any]:
        direction = str(kwargs.get("direction", "NO_TRADE")).upper().strip()
        prior_win_rate = float(np.clip(kwargs.get("prior_direction_win_rate", 0.5), 0.0, 1.0))
        prior_neutral = float(np.clip(kwargs.get("prior_neutral_rate", 0.0), 0.0, 1.0))
        prior_loss = float(np.clip(1.0 - prior_win_rate - prior_neutral, 0.0, 1.0))
        if direction == "OVER":
            over_prob = prior_win_rate
            under_prob = prior_loss
        elif direction == "UNDER":
            over_prob = prior_loss
            under_prob = prior_win_rate
        else:
            over_prob = 0.5
            under_prob = 0.5
        return {
            "over_prob": float(over_prob),
            "under_prob": float(under_prob),
            "no_trade_prob": float(prior_neutral),
            "chosen_direction_prob": float(prior_win_rate),
            "opposite_direction_prob": float(prior_loss),
            "chosen_direction_conditional_prob": float(prior_win_rate),
            "opposite_direction_conditional_prob": float(prior_loss),
            "conditional_prob_gap": float(prior_win_rate - prior_loss),
            "trade_prob_floor": float(kwargs.get("config").min_trade_prob if kwargs.get("config") is not None else 0.63),
            "trade_eligible": direction in {"OVER", "UNDER"},
            "action": direction if direction in {"OVER", "UNDER"} else "NO_TRADE",
            "source": "module_missing",
            "support_rows": 0.0,
            "support_strength": 0.0,
            "sigma_pressure": 1.0,
            "instability_score": 1.0,
            "fragility_score": 1.0,
            "empirical_blend_weight": 0.0,
        }

try:
    from decision_engine.uncertainty import (
        BELIEF_UNCERTAINTY_LOWER,
        BELIEF_UNCERTAINTY_UPPER,
        belief_confidence_factor,
        normalize_belief_uncertainty,
    )
except Exception:  # pragma: no cover - fallback for standalone execution
    BELIEF_UNCERTAINTY_LOWER = 0.75
    BELIEF_UNCERTAINTY_UPPER = 1.15

    def normalize_belief_uncertainty(value, default: float = 1.0, lower: float = BELIEF_UNCERTAINTY_LOWER, upper: float = BELIEF_UNCERTAINTY_UPPER):
        span = max(float(upper) - float(lower), 1e-9)
        numeric = pd.to_numeric(value, errors="coerce") if isinstance(value, pd.Series) else safe_float(value, default=default)
        if isinstance(numeric, pd.Series):
            return ((numeric.fillna(float(default)) - float(lower)) / span).clip(lower=0.0, upper=1.0)
        return float(np.clip((float(numeric) - float(lower)) / span, 0.0, 1.0))

    def belief_confidence_factor(value, default: float = 1.0, lower: float = BELIEF_UNCERTAINTY_LOWER, upper: float = BELIEF_UNCERTAINTY_UPPER):
        normalized = normalize_belief_uncertainty(value, default=default, lower=lower, upper=upper)
        if isinstance(normalized, pd.Series):
            return (1.0 - normalized).clip(lower=0.0, upper=1.0)
        return float(np.clip(1.0 - float(normalized), 0.0, 1.0))


TARGETS = ["PTS", "TRB", "AST"]
TARGET_THRESHOLDS = {
    "PTS": {"consider_pct": 0.75, "strong_pct": 0.90},
    "TRB": {"consider_pct": 0.85, "strong_pct": 0.95},
    "AST": {"consider_pct": 0.85, "strong_pct": 0.95},
}
HEURISTIC_EDGE_SCALES = {
    "PTS": 3.0,
    "TRB": 1.2,
    "AST": 1.0,
}
DEFAULT_BETA_PRIOR_ALPHA = 1.0
DEFAULT_BETA_PRIOR_BETA = 1.0
DEFAULT_CALIBRATION_BINS = 12
DEFAULT_CALIBRATION_MIN_ROWS = 40
DEFAULT_MARKET_REGRESSION_FLOOR = 0.25
DEFAULT_MARKET_REGRESSION_CEILING = 0.95
DEFAULT_REBOUND_DIAGNOSTICS_CONFIG = {
    "enabled": True,
    "upper_band": {
        "enabled": True,
        "q75_margin_trigger": 0.25,
        "min_recent_games": 5,
        "penalty_strength": 0.12,
    },
    "low_line_role_volatility": {
        "enabled": True,
        "max_minutes_band_width": 8.0,
        "min_minutes_floor": 18.0,
        "bench_role_penalty": 0.08,
        "penalty_strength": 0.10,
    },
    "rebound_supply": {
        "enabled": True,
        "min_projected_missed_fga_environment": 78.0,
        "high_efficiency_penalty_threshold": 0.505,
        "penalty_strength": 0.08,
    },
    "rebound_share": {
        "enabled": True,
        "max_teammate_competition_score": 0.65,
        "max_rebound_share_std": 0.08,
        "penalty_strength": 0.07,
    },
    "opposite_side_discovery": {
        "enabled": True,
        "evaluate_under_when_over_penalized": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and rank upcoming market plays.")
    parser.add_argument(
        "--slate-csv",
        type=Path,
        default=Path("model/analysis/upcoming_market_slate.csv"),
        help="Upcoming slate CSV from build_upcoming_slate.py",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("model/analysis/latest_market_comparison_strict_rows.csv"),
        help="Historical row-level backtest CSV",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("model/analysis/upcoming_market_play_selector.json"),
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("model/analysis/upcoming_market_play_selector.csv"),
        help="Output ranked plays CSV path",
    )
    parser.add_argument(
        "--disable-volatility-adjustment",
        action="store_true",
        help="Disable volatility/spike-aware risk adjustment and use raw gap logic only.",
    )
    parser.add_argument(
        "--belief-uncertainty-lower",
        type=float,
        default=BELIEF_UNCERTAINTY_LOWER,
        help="Lower anchor used when converting latent belief uncertainty into a confidence penalty.",
    )
    parser.add_argument(
        "--belief-uncertainty-upper",
        type=float,
        default=BELIEF_UNCERTAINTY_UPPER,
        help="Upper anchor used when converting latent belief uncertainty into a confidence penalty.",
    )
    parser.add_argument(
        "--market-regression-floor",
        type=float,
        default=DEFAULT_MARKET_REGRESSION_FLOOR,
        help="Minimum shrinkage lambda used when regressing prediction toward market line.",
    )
    parser.add_argument(
        "--market-regression-ceiling",
        type=float,
        default=DEFAULT_MARKET_REGRESSION_CEILING,
        help="Maximum shrinkage lambda used when regressing prediction toward market line.",
    )
    parser.add_argument(
        "--disable-line-decision-sidecar",
        action="store_true",
        help="Disable the line-aware decision sidecar and keep the legacy selector probability mapping only.",
    )
    parser.add_argument(
        "--line-decision-no-trade-threshold",
        type=float,
        default=0.45,
        help="Minimum sidecar neutral/no-trade probability that blocks a trade.",
    )
    parser.add_argument(
        "--line-decision-min-trade-prob",
        type=float,
        default=LineDecisionConfig().min_trade_prob if LineDecisionConfig is not None else 0.63,
        help="Minimum chosen-direction probability required for a trade to remain eligible.",
    )
    parser.add_argument(
        "--line-decision-min-prob-gap",
        type=float,
        default=LineDecisionConfig().min_trade_prob_gap if LineDecisionConfig is not None else 0.06,
        help="Minimum chosen-vs-opposite probability gap required for a trade to remain eligible.",
    )
    return parser.parse_args()


def active_only_mask(df: pd.DataFrame) -> pd.Series:
    minutes = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0)
    return (
        (pd.to_numeric(df.get("did_not_play"), errors="coerce").fillna(0.0) < 0.5)
        & ~(
            (pd.to_numeric(df.get("actual_PTS"), errors="coerce").fillna(0.0) == 0.0)
            & (pd.to_numeric(df.get("actual_TRB"), errors="coerce").fillna(0.0) == 0.0)
            & (pd.to_numeric(df.get("actual_AST"), errors="coerce").fillna(0.0) == 0.0)
            & (minutes <= 0.0)
        )
    )


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _beta_posterior_stats(
    wins: int,
    losses: int,
    alpha_prior: float,
    beta_prior: float,
) -> dict[str, float]:
    alpha = float(alpha_prior) + float(max(0, wins))
    beta = float(beta_prior) + float(max(0, losses))
    total = max(1e-9, alpha + beta)
    mean = float(alpha / total)
    variance = float((alpha * beta) / ((total ** 2) * (total + 1.0)))
    std = float(np.sqrt(max(0.0, variance)))
    return {
        "alpha": alpha,
        "beta": beta,
        "mean": mean,
        "variance": variance,
        "ci_low": float(np.clip(mean - 1.96 * std, 0.0, 1.0)),
        "ci_high": float(np.clip(mean + 1.96 * std, 0.0, 1.0)),
    }


def _global_beta_prior(target_frames: dict[str, pd.DataFrame]) -> tuple[float, float, float]:
    wins = 0
    losses = 0
    for frame in target_frames.values():
        resolved = frame["actual_minus_market"].to_numpy(dtype=float) != 0.0
        correct = frame["directional_correct"].to_numpy(dtype=bool)
        wins += int((resolved & correct).sum())
        losses += int((resolved & (~correct)).sum())

    if wins + losses <= 0:
        return DEFAULT_BETA_PRIOR_ALPHA, DEFAULT_BETA_PRIOR_BETA, 0.5

    global_mean = float((wins + DEFAULT_BETA_PRIOR_ALPHA) / (wins + losses + DEFAULT_BETA_PRIOR_ALPHA + DEFAULT_BETA_PRIOR_BETA))
    prior_strength = float(np.clip(np.sqrt(wins + losses), 8.0, 32.0))
    alpha_prior = max(0.1, global_mean * prior_strength)
    beta_prior = max(0.1, (1.0 - global_mean) * prior_strength)
    return float(alpha_prior), float(beta_prior), global_mean


def _bucket_summary(frame: pd.DataFrame, alpha_prior: float, beta_prior: float) -> dict[str, Any] | None:
    if frame.empty:
        return None

    resolved = frame["actual_minus_market"].to_numpy(dtype=float) != 0.0
    correct = frame["directional_correct"].to_numpy(dtype=bool)
    wins = int((resolved & correct).sum())
    losses = int((resolved & (~correct)).sum())
    pushes = int((~resolved).sum())
    rows = int(len(frame))
    resolved_rows = wins + losses
    resolved_rate = float(resolved_rows / rows) if rows > 0 else 0.0
    push_rate = float(pushes / rows) if rows > 0 else 0.0

    posterior = _beta_posterior_stats(wins=wins, losses=losses, alpha_prior=alpha_prior, beta_prior=beta_prior)
    win_rate = float(np.clip(posterior["mean"] * resolved_rate, 0.0, 1.0 - push_rate))
    loss_rate = float(np.clip(1.0 - win_rate - push_rate, 0.0, 1.0))

    return {
        "rows": rows,
        "resolved_rows": int(resolved_rows),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "resolved_rate": resolved_rate,
        "win_rate": win_rate,
        "push_rate": push_rate,
        "loss_rate": loss_rate,
        "posterior_alpha": float(posterior["alpha"]),
        "posterior_beta": float(posterior["beta"]),
        "posterior_mean": float(posterior["mean"]),
        "posterior_variance": float(posterior["variance"]),
        "posterior_ci_low": float(posterior["ci_low"]),
        "posterior_ci_high": float(posterior["ci_high"]),
    }


def _fit_monotonic_calibration(
    frame: pd.DataFrame,
    gaps_sorted: np.ndarray,
    alpha_prior: float,
    beta_prior: float,
    min_rows: int = DEFAULT_CALIBRATION_MIN_ROWS,
    max_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any] | None:
    if frame.empty or gaps_sorted.size == 0:
        return None

    resolved_mask = frame["actual_minus_market"].to_numpy(dtype=float) != 0.0
    if int(resolved_mask.sum()) < int(min_rows):
        return None

    resolved_frame = frame.loc[resolved_mask].copy()
    gap_values = resolved_frame["abs_gap"].to_numpy(dtype=float)
    percentiles = np.searchsorted(gaps_sorted, gap_values, side="right") / max(1, gaps_sorted.size)
    outcomes = resolved_frame["directional_correct"].to_numpy(dtype=bool)

    n_bins = int(np.clip(np.sqrt(len(resolved_frame)), 6, max_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_ids = np.clip(np.digitize(percentiles, edges[1:-1], right=False), 0, n_bins - 1)

    rates = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        counts[idx] = count
        if count <= 0:
            continue
        wins = int(outcomes[mask].sum())
        losses = int(count - wins)
        posterior = _beta_posterior_stats(wins=wins, losses=losses, alpha_prior=alpha_prior, beta_prior=beta_prior)
        rates[idx] = float(posterior["mean"])

    valid = np.flatnonzero(~np.isnan(rates))
    if valid.size == 0:
        return None

    interpolated = np.interp(np.arange(n_bins), valid, rates[valid])
    monotonic = np.maximum.accumulate(interpolated)
    monotonic = np.clip(monotonic, 0.35, 0.95)

    return {
        "bin_centers": centers,
        "bin_rates": monotonic,
        "bin_counts": counts,
        "resolved_rows": int(len(resolved_frame)),
    }


def _apply_calibration_curve(calibration_curve: dict[str, Any] | None, percentile: float) -> float | None:
    if not calibration_curve:
        return None
    centers = np.asarray(calibration_curve.get("bin_centers", []), dtype=float)
    rates = np.asarray(calibration_curve.get("bin_rates", []), dtype=float)
    if centers.size == 0 or rates.size == 0:
        return None
    pct = float(np.clip(percentile, 0.0, 1.0))
    return float(np.interp(pct, centers, rates))


def build_history_lookup(history_df: pd.DataFrame) -> dict[str, dict]:
    active_history = history_df.loc[active_only_mask(history_df)].copy()
    target_frames: dict[str, pd.DataFrame] = {}

    for target in TARGETS:
        market_col = f"market_{target}"
        pred_col = f"pred_{target}"
        actual_col = f"actual_{target}"
        if market_col not in active_history.columns or pred_col not in active_history.columns or actual_col not in active_history.columns:
            continue

        covered = active_history.loc[pd.to_numeric(active_history[market_col], errors="coerce").notna()].copy()
        if covered.empty:
            continue

        pred_minus_market = pd.to_numeric(covered[pred_col], errors="coerce") - pd.to_numeric(covered[market_col], errors="coerce")
        actual_minus_market = pd.to_numeric(covered[actual_col], errors="coerce") - pd.to_numeric(covered[market_col], errors="coerce")
        abs_gap = pred_minus_market.abs()
        called = pred_minus_market != 0
        correct = ((pred_minus_market > 0) & (actual_minus_market > 0)) | ((pred_minus_market < 0) & (actual_minus_market < 0))
        working = pd.DataFrame(
            {
                "pred_minus_market": pred_minus_market,
                "actual_minus_market": actual_minus_market,
                "abs_gap": abs_gap,
                "directional_called": called,
                "directional_correct": correct,
            }
        )
        working = working.loc[working["directional_called"] & working["abs_gap"].notna()].copy()
        if working.empty:
            continue
        target_frames[target] = working

    alpha_prior, beta_prior, global_mean = _global_beta_prior(target_frames)
    lookup: dict[str, dict] = {
        "__meta__": {
            "global_prior_alpha": float(alpha_prior),
            "global_prior_beta": float(beta_prior),
            "global_prior_mean": float(global_mean),
        }
    }

    for target, working in target_frames.items():
        quartile_cut = float(working["abs_gap"].quantile(0.75))
        decile_cut = float(working["abs_gap"].quantile(0.90))
        gaps_sorted = np.sort(working["abs_gap"].to_numpy(dtype=float))

        all_bucket = _bucket_summary(working, alpha_prior=alpha_prior, beta_prior=beta_prior)
        top_quartile = _bucket_summary(working.loc[working["abs_gap"] >= quartile_cut], alpha_prior=alpha_prior, beta_prior=beta_prior)
        top_decile = _bucket_summary(working.loc[working["abs_gap"] >= decile_cut], alpha_prior=alpha_prior, beta_prior=beta_prior)
        calibration_curve = _fit_monotonic_calibration(
            working,
            gaps_sorted=gaps_sorted,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )

        lookup[target] = {
            "rows": int(len(working)),
            "quartile_cut": quartile_cut,
            "decile_cut": decile_cut,
            "gaps_sorted": gaps_sorted,
            "all": all_bucket,
            "top_quartile": top_quartile,
            "top_decile": top_decile,
            "calibration_curve": calibration_curve,
            "prior_alpha": float(alpha_prior),
            "prior_beta": float(beta_prior),
        }
    return lookup


def percentile_of_gap(gaps_sorted: np.ndarray, gap: float) -> float:
    if gaps_sorted.size == 0:
        return 0.0
    rank = np.searchsorted(gaps_sorted, gap, side="right")
    return float(rank / gaps_sorted.size)


def classify_play(target: str, percentile: float) -> str:
    thresholds = TARGET_THRESHOLDS[target]
    if percentile >= thresholds["strong_pct"]:
        return "strong"
    if percentile >= thresholds["consider_pct"]:
        return "consider"
    return "pass"


def expected_rate_for(target: str, percentile: float, history_info: dict[str, Any]) -> dict[str, Any]:
    thresholds = TARGET_THRESHOLDS[target]
    bucket_key = "all"
    bucket = history_info.get("all")

    if percentile >= thresholds["strong_pct"] and history_info.get("top_decile") is not None:
        bucket_key = "top_decile"
        bucket = history_info.get("top_decile")
    elif percentile >= thresholds["consider_pct"] and history_info.get("top_quartile") is not None:
        bucket_key = "top_quartile"
        bucket = history_info.get("top_quartile")

    if not bucket:
        return {
            "bucket": "empty",
            "base_expected_win_rate": 0.5,
            "expected_win_rate": 0.5,
            "expected_push_rate": 0.0,
            "expected_loss_rate": 0.5,
            "posterior_alpha": DEFAULT_BETA_PRIOR_ALPHA,
            "posterior_beta": DEFAULT_BETA_PRIOR_BETA,
            "posterior_variance": 0.25,
            "posterior_ci_low": 0.0,
            "posterior_ci_high": 1.0,
            "calibrated_conditional_win_rate": None,
            "calibration_weight": 0.0,
            "calibration_source": "empty",
            "bucket_rows": 0,
        }

    base_win_rate = float(bucket.get("win_rate", 0.5))
    push_rate = float(np.clip(bucket.get("push_rate", 0.0), 0.0, 1.0))
    resolved_rate = float(np.clip(bucket.get("resolved_rate", 1.0 - push_rate), 0.0, 1.0))
    calibrated_conditional_rate = _apply_calibration_curve(history_info.get("calibration_curve"), percentile)

    if calibrated_conditional_rate is None:
        win_rate = base_win_rate
        calibration_weight = 0.0
        calibration_source = "bayesian_bucket"
    else:
        calibrated_win_rate = float(np.clip(calibrated_conditional_rate * resolved_rate, 0.0, 1.0 - push_rate))
        calibration_rows = int((history_info.get("calibration_curve") or {}).get("resolved_rows", 0))
        bucket_rows = int(bucket.get("rows", 0))
        calibration_weight = float(np.clip(calibration_rows / max(1.0, calibration_rows + 0.75 * bucket_rows), 0.20, 0.85))
        win_rate = float(calibration_weight * calibrated_win_rate + (1.0 - calibration_weight) * base_win_rate)
        calibration_source = "bayesian_isotonic_blend"

    non_push = max(0.0, 1.0 - push_rate)
    win_rate = float(np.clip(win_rate, 0.0, non_push))
    loss_rate = float(np.clip(non_push - win_rate, 0.0, 1.0))

    return {
        "bucket": bucket_key,
        "base_expected_win_rate": base_win_rate,
        "expected_win_rate": win_rate,
        "expected_push_rate": push_rate,
        "expected_loss_rate": loss_rate,
        "posterior_alpha": float(bucket.get("posterior_alpha", DEFAULT_BETA_PRIOR_ALPHA)),
        "posterior_beta": float(bucket.get("posterior_beta", DEFAULT_BETA_PRIOR_BETA)),
        "posterior_variance": float(bucket.get("posterior_variance", 0.25)),
        "posterior_ci_low": float(bucket.get("posterior_ci_low", 0.0)),
        "posterior_ci_high": float(bucket.get("posterior_ci_high", 1.0)),
        "calibrated_conditional_win_rate": calibrated_conditional_rate,
        "calibration_weight": calibration_weight,
        "calibration_source": calibration_source,
        "bucket_rows": int(bucket.get("rows", 0)),
    }


def heuristic_percentile_and_rate(target: str, abs_gap: float) -> dict[str, Any]:
    scale = float(HEURISTIC_EDGE_SCALES[target])
    gap_pct = float(np.clip(abs_gap / scale, 0.01, 0.99))
    win_rate = float(np.clip(0.50 + 0.30 * gap_pct, 0.50, 0.82))
    push_rate = float(np.clip(0.06 - 0.04 * gap_pct, 0.01, 0.06))
    loss_rate = float(np.clip(1.0 - win_rate - push_rate, 0.0, 1.0))
    return {
        "gap_percentile": gap_pct,
        "bucket": "heuristic",
        "base_expected_win_rate": win_rate,
        "expected_win_rate": win_rate,
        "expected_push_rate": push_rate,
        "expected_loss_rate": loss_rate,
        "posterior_alpha": DEFAULT_BETA_PRIOR_ALPHA,
        "posterior_beta": DEFAULT_BETA_PRIOR_BETA,
        "posterior_variance": 0.25,
        "posterior_ci_low": 0.0,
        "posterior_ci_high": 1.0,
        "calibrated_conditional_win_rate": None,
        "calibration_weight": 0.0,
        "calibration_source": "heuristic",
        "bucket_rows": 0,
    }


def _prediction_shrink_lambda(
    belief_conf: float,
    feasibility: float,
    fallback_blend: float,
    floor: float,
    ceiling: float,
) -> float:
    fallback = _clip01(fallback_blend, default=0.0)
    confidence = float(np.clip(belief_conf * feasibility * (1.0 - fallback), 0.0, 1.0))
    lower = float(np.clip(floor, 0.0, 1.0))
    upper = float(np.clip(ceiling, lower, 1.0))
    return float(np.clip(lower + (upper - lower) * confidence, lower, upper))


def _clip01(value: float, default: float = 0.0) -> float:
    numeric = safe_float(value, default=default)
    return float(np.clip(numeric, 0.0, 1.0))


def _downgrade_recommendation(label: str) -> str:
    order = {
        "elite": "strong",
        "strong": "consider",
        "consider": "pass",
        "pass": "pass",
    }
    return order.get(str(label).strip().lower(), str(label))


def _merge_nested_dict(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    override_payload = overrides if isinstance(overrides, dict) else {}
    for key, default_value in defaults.items():
        override_value = override_payload.get(key)
        if isinstance(default_value, dict):
            merged[key] = _merge_nested_dict(default_value, override_value if isinstance(override_value, dict) else {})
        else:
            merged[key] = default_value if key not in override_payload else override_value
    for key, value in override_payload.items():
        if key not in merged:
            merged[key] = value
    return merged


def _resolve_rebound_diagnostics_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return _merge_nested_dict(DEFAULT_REBOUND_DIAGNOSTICS_CONFIG, config if isinstance(config, dict) else {})


def _market_type(target: str, direction: str) -> str:
    return f"{str(target).upper().strip()}_{str(direction).upper().strip()}"


def _bool_like(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "t", "yes", "y"}:
            return True
        if token in {"0", "false", "f", "no", "n", ""}:
            return False
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(np.nan).iloc[0]
    if pd.notna(numeric):
        return bool(float(numeric) > 0.0)
    return bool(default)


def _american_break_even_prob(odds: float | int | None) -> float:
    price = safe_float(odds, default=np.nan)
    if not np.isfinite(price) or abs(price) < 1e-9:
        return np.nan
    if price > 0:
        return float(100.0 / (price + 100.0))
    return float(abs(price) / (abs(price) + 100.0))


def _market_side_price(row: pd.Series, target: str, direction: str) -> float:
    return safe_float(row.get(f"market_{str(direction).lower()}_price_{target}"), default=np.nan)


def _upper_band_diagnostic(
    target: str,
    direction: str,
    market_line: float,
    row: pd.Series,
    rebound_config: dict[str, Any],
) -> dict[str, Any]:
    q75 = safe_float(row.get("trb_q75_recent"), default=np.nan)
    q90 = safe_float(row.get("trb_q90_recent"), default=np.nan)
    median = safe_float(row.get("trb_median_recent"), default=np.nan)
    recent_games_count = int(max(0, round(safe_float(row.get("recent_games_count"), default=0.0))))
    out = {
        "recent_games_count": recent_games_count,
        "line_minus_trb_q75": float(market_line - q75) if np.isfinite(market_line) and np.isfinite(q75) else np.nan,
        "line_minus_trb_q90": float(market_line - q90) if np.isfinite(market_line) and np.isfinite(q90) else np.nan,
        "upper_band_line_penalty": 0.0,
        "upper_band_line_flag": False,
        "upper_band_line_reason": "not_applicable",
    }
    if _market_type(target, direction) != "TRB_OVER":
        return out
    cfg = rebound_config.get("upper_band", {})
    if not bool(cfg.get("enabled", True)):
        out["upper_band_line_reason"] = "disabled"
        return out
    if not np.isfinite(market_line):
        out["upper_band_line_reason"] = "invalid_market_line"
        return out
    if recent_games_count < int(cfg.get("min_recent_games", 5)):
        out["upper_band_line_reason"] = "insufficient_recent_games"
        return out
    if not np.isfinite(q75):
        out["upper_band_line_reason"] = "missing_recent_quantiles"
        return out
    trigger_line = float(q75 + float(cfg.get("q75_margin_trigger", 0.25)))
    if market_line <= trigger_line:
        out["upper_band_line_reason"] = "line_not_above_recent_upper_band"
        return out
    scale = max(0.75, (q90 - q75) if np.isfinite(q90) and q90 > q75 else max(0.75, q75 - median if np.isfinite(median) else 0.75))
    severity = float(np.clip((market_line - trigger_line) / scale, 0.0, 1.0))
    out["upper_band_line_penalty"] = float(np.clip(float(cfg.get("penalty_strength", 0.12)) * severity, 0.0, 1.0))
    out["upper_band_line_flag"] = bool(out["upper_band_line_penalty"] > 0.0)
    out["upper_band_line_reason"] = (
        f"line={market_line:.2f}>q75_trigger={trigger_line:.2f};"
        f"q75={q75:.2f};q90={q90:.2f}" if np.isfinite(q90) else f"line={market_line:.2f}>q75_trigger={trigger_line:.2f};q75={q75:.2f}"
    )
    return out


def _low_line_role_volatility_diagnostic(
    target: str,
    direction: str,
    market_line: float,
    row: pd.Series,
    rebound_config: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "low_line_role_volatility_flag": False,
        "low_line_role_volatility_penalty": 0.0,
        "low_line_role_volatility_reason": "not_applicable",
    }
    if _market_type(target, direction) != "TRB_OVER":
        return out
    cfg = rebound_config.get("low_line_role_volatility", {})
    if not bool(cfg.get("enabled", True)):
        out["low_line_role_volatility_reason"] = "disabled"
        return out

    trb_median = safe_float(row.get("trb_median_recent"), default=np.nan)
    trb_q75 = safe_float(row.get("trb_q75_recent"), default=np.nan)
    low_line = (
        (np.isfinite(trb_median) and market_line <= trb_median)
        or (np.isfinite(trb_q75) and market_line <= trb_q75)
    )
    if not low_line:
        out["low_line_role_volatility_reason"] = "line_not_low_band"
        return out

    expected_band_width = safe_float(row.get("expected_minutes_band_width"), default=np.nan)
    minutes_floor = safe_float(row.get("minutes_floor_recent"), default=np.nan)
    bench_role = _bool_like(row.get("bench_role_flag"), default=False)
    rotation_score = _clip01(row.get("rotation_volatility_score"), default=0.50)
    max_band_width = float(cfg.get("max_minutes_band_width", 8.0))
    min_minutes_floor = float(cfg.get("min_minutes_floor", 18.0))
    rotation_threshold = float(cfg.get("rotation_volatility_threshold", 0.58))

    reasons: list[str] = []
    width_severity = 0.0
    if np.isfinite(expected_band_width) and expected_band_width > max_band_width:
        reasons.append(f"minutes_band_width={expected_band_width:.2f}>{max_band_width:.2f}")
        width_severity = float(np.clip((expected_band_width - max_band_width) / max(2.0, max_band_width), 0.0, 1.0))
    floor_severity = 0.0
    if np.isfinite(minutes_floor) and minutes_floor < min_minutes_floor:
        reasons.append(f"minutes_floor={minutes_floor:.2f}<{min_minutes_floor:.2f}")
        floor_severity = float(np.clip((min_minutes_floor - minutes_floor) / max(1.0, min_minutes_floor), 0.0, 1.0))
    bench_severity = 1.0 if bench_role else 0.0
    if bench_role:
        reasons.append("bench_role_flag")
    rotation_severity = 0.0
    if rotation_score > rotation_threshold:
        reasons.append(f"rotation_volatility={rotation_score:.2f}>{rotation_threshold:.2f}")
        rotation_severity = float(np.clip((rotation_score - rotation_threshold) / max(1e-6, 1.0 - rotation_threshold), 0.0, 1.0))
    if not reasons:
        out["low_line_role_volatility_reason"] = "minutes_band_stable"
        return out

    severity = float(np.clip(max(width_severity, floor_severity, bench_severity, rotation_severity), 0.0, 1.0))
    penalty = float(cfg.get("penalty_strength", 0.10)) * severity
    if bench_role:
        penalty += float(cfg.get("bench_role_penalty", 0.08))
    out["low_line_role_volatility_flag"] = True
    out["low_line_role_volatility_penalty"] = float(np.clip(penalty, 0.0, 1.0))
    out["low_line_role_volatility_reason"] = "; ".join(reasons)
    return out


def _rebound_supply_diagnostic(
    target: str,
    direction: str,
    row: pd.Series,
    rebound_config: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "rebound_supply_penalty": 0.0,
        "rebound_supply_reason": "not_applicable",
    }
    if _market_type(target, direction) != "TRB_OVER":
        return out
    cfg = rebound_config.get("rebound_supply", {})
    if not bool(cfg.get("enabled", True)):
        out["rebound_supply_reason"] = "disabled"
        return out

    projected_total = safe_float(row.get("projected_missed_fga_total"), default=np.nan)
    team_fg_pct = safe_float(row.get("projected_team_fg_pct"), default=np.nan)
    opp_fg_pct = safe_float(row.get("projected_opponent_fg_pct"), default=np.nan)
    pace_env = _clip01(row.get("pace_rebound_environment"), default=_clip01(row.get("rebound_supply_score"), default=0.50))
    ft_suppression = _clip01(row.get("free_throw_rebound_suppression"), default=0.0)
    min_env = float(cfg.get("min_projected_missed_fga_environment", 78.0))
    high_eff_threshold = float(cfg.get("high_efficiency_penalty_threshold", 0.505))

    env_shortfall = 0.0
    reasons: list[str] = []
    if np.isfinite(projected_total) and projected_total < min_env:
        reasons.append(f"missed_fga_total={projected_total:.2f}<{min_env:.2f}")
        env_shortfall = float(np.clip((min_env - projected_total) / max(6.0, min_env), 0.0, 1.0))
    both_efficient = bool(
        np.isfinite(team_fg_pct)
        and np.isfinite(opp_fg_pct)
        and team_fg_pct >= high_eff_threshold
        and opp_fg_pct >= high_eff_threshold
    )
    efficiency_severity = 0.0
    if both_efficient:
        reasons.append(f"fg_efficiency={team_fg_pct:.3f}/{opp_fg_pct:.3f}>={high_eff_threshold:.3f}")
        efficiency_severity = float(np.clip(((team_fg_pct - high_eff_threshold) + (opp_fg_pct - high_eff_threshold)) / 0.10, 0.0, 1.0))
    if not reasons:
        out["rebound_supply_reason"] = "missed_shot_environment_supportive"
        return out

    severity = float(np.clip(0.55 * env_shortfall + 0.25 * efficiency_severity + 0.10 * (1.0 - pace_env) + 0.10 * ft_suppression, 0.0, 1.0))
    out["rebound_supply_penalty"] = float(np.clip(float(cfg.get("penalty_strength", 0.08)) * severity, 0.0, 1.0))
    out["rebound_supply_reason"] = "; ".join(reasons)
    return out


def _rebound_share_diagnostic(
    target: str,
    direction: str,
    row: pd.Series,
    rebound_config: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "rebound_share_competition_penalty": 0.0,
        "rebound_share_reason": "not_applicable",
    }
    if _market_type(target, direction) != "TRB_OVER":
        return out
    cfg = rebound_config.get("rebound_share", {})
    if not bool(cfg.get("enabled", True)):
        out["rebound_share_reason"] = "disabled"
        return out

    competition = _clip01(row.get("teammate_rebound_competition_score"), default=_clip01(row.get("teammate_rebound_competition"), default=0.50))
    share_std = safe_float(row.get("player_rebound_share_std"), default=np.nan)
    wing_leakage = _clip01(row.get("wing_rebound_leakage_score"), default=0.50)
    center_pressure = _clip01(row.get("center_rebound_share_pressure"), default=0.50)
    frontcourt_overlap = _clip01(row.get("frontcourt_rebound_overlap_score"), default=0.50)
    max_competition = float(cfg.get("max_teammate_competition_score", 0.65))
    max_share_std = float(cfg.get("max_rebound_share_std", 0.08))

    reasons: list[str] = []
    competition_severity = 0.0
    if competition > max_competition:
        reasons.append(f"competition={competition:.2f}>{max_competition:.2f}")
        competition_severity = float(np.clip((competition - max_competition) / max(1e-6, 1.0 - max_competition), 0.0, 1.0))
    share_std_severity = 0.0
    if np.isfinite(share_std) and share_std > max_share_std:
        reasons.append(f"share_std={share_std:.3f}>{max_share_std:.3f}")
        share_std_severity = float(np.clip((share_std - max_share_std) / max(0.02, max_share_std), 0.0, 1.0))
    leakage_severity = float(np.clip((wing_leakage - 0.58) / 0.22, 0.0, 1.0))
    overlap_severity = float(np.clip((0.55 * center_pressure + 0.45 * frontcourt_overlap - 0.62) / 0.25, 0.0, 1.0))
    if leakage_severity > 0.0:
        reasons.append(f"wing_leakage={wing_leakage:.2f}")
    if overlap_severity > 0.0:
        reasons.append(f"frontcourt_overlap={frontcourt_overlap:.2f}")
    if not reasons:
        out["rebound_share_reason"] = "rebound_share_concentrated"
        return out

    severity = float(np.clip(0.45 * competition_severity + 0.30 * share_std_severity + 0.15 * leakage_severity + 0.10 * overlap_severity, 0.0, 1.0))
    out["rebound_share_competition_penalty"] = float(np.clip(float(cfg.get("penalty_strength", 0.07)) * severity, 0.0, 1.0))
    out["rebound_share_reason"] = "; ".join(reasons)
    return out


def _supply_dependent_context(
    row: pd.Series,
    target: str,
    direction: str,
    market_line: float,
    rebound_diagnostics_config: dict[str, Any] | None = None,
    american_odds: int = -110,
) -> dict[str, Any]:
    target_key = str(target).upper().strip()
    direction_key = str(direction).upper().strip()
    rebound_config = _resolve_rebound_diagnostics_config(rebound_diagnostics_config)
    active = bool(rebound_config.get("enabled", True)) and _market_type(target_key, direction_key) == "TRB_OVER"
    over_price = _market_side_price(row, target_key, "OVER")
    under_price = _market_side_price(row, target_key, "UNDER")
    fallback_break_even = _american_break_even_prob(american_odds)
    upper_band_diag = _upper_band_diagnostic(target_key, direction_key, market_line, row, rebound_config)
    low_line_diag = _low_line_role_volatility_diagnostic(target_key, direction_key, market_line, row, rebound_config)
    supply_diag = _rebound_supply_diagnostic(target_key, direction_key, row, rebound_config)
    share_diag = _rebound_share_diagnostic(target_key, direction_key, row, rebound_config)

    base_context = {
        "supply_dependency_active": bool(active),
        "supply_dependency_score": 0.0,
        "supply_dependency_classification": "not_applicable",
        "rebound_supply_score": _clip01(row.get("rebound_supply_score"), default=0.50),
        "rebound_share_stability": _clip01(row.get("rebound_share_stability"), default=0.50),
        "rebound_share_stability_score": _clip01(row.get("rebound_share_stability_score"), default=_clip01(row.get("rebound_share_stability"), default=0.50)),
        "team_shooting_efficiency_stress": _clip01(row.get("team_shooting_efficiency_stress"), default=0.50),
        "opponent_shooting_efficiency_stress": _clip01(row.get("opponent_shooting_efficiency_stress"), default=0.50),
        "wing_rebound_leakage_score": _clip01(row.get("wing_rebound_leakage_score"), default=0.50),
        "teammate_rebound_competition": _clip01(row.get("teammate_rebound_competition"), default=0.50),
        "teammate_rebound_competition_score": _clip01(row.get("teammate_rebound_competition_score"), default=_clip01(row.get("teammate_rebound_competition"), default=0.50)),
        "rebound_share_estimate": _clip01(row.get("rebound_share_estimate"), default=0.50),
        "player_team_rebound_share_recent": safe_float(row.get("player_team_rebound_share_recent"), default=np.nan),
        "player_rebound_share_std": safe_float(row.get("player_rebound_share_std"), default=np.nan),
        "center_rebound_share_pressure": _clip01(row.get("center_rebound_share_pressure"), default=0.50),
        "frontcourt_rebound_overlap_score": _clip01(row.get("frontcourt_rebound_overlap_score"), default=0.50),
        "projected_team_missed_fga": safe_float(row.get("projected_team_missed_fga"), default=np.nan),
        "projected_opponent_missed_fga": safe_float(row.get("projected_opponent_missed_fga"), default=np.nan),
        "projected_team_missed_fta": safe_float(row.get("projected_team_missed_fta"), default=np.nan),
        "projected_opponent_missed_fta": safe_float(row.get("projected_opponent_missed_fta"), default=np.nan),
        "projected_missed_fga_total": safe_float(row.get("projected_missed_fga_total"), default=np.nan),
        "projected_missed_fta_total": safe_float(row.get("projected_missed_fta_total"), default=np.nan),
        "projected_available_rebound_events": safe_float(row.get("projected_available_rebound_events"), default=np.nan),
        "expected_rebound_chances": safe_float(row.get("expected_rebound_chances"), default=np.nan),
        "team_rebound_pool_size": safe_float(row.get("team_rebound_pool_size"), default=np.nan),
        "pace_rebound_environment": _clip01(row.get("pace_rebound_environment"), default=_clip01(row.get("rebound_supply_score"), default=0.50)),
        "long_rebound_profile": _clip01(row.get("long_rebound_profile"), default=0.50),
        "free_throw_rebound_suppression": _clip01(row.get("free_throw_rebound_suppression"), default=0.0),
        "projected_team_fg_pct": safe_float(row.get("projected_team_fg_pct"), default=np.nan),
        "projected_opponent_fg_pct": safe_float(row.get("projected_opponent_fg_pct"), default=np.nan),
        "trb_median_recent": safe_float(row.get("trb_median_recent"), default=np.nan),
        "trb_q75_recent": safe_float(row.get("trb_q75_recent"), default=np.nan),
        "trb_q90_recent": safe_float(row.get("trb_q90_recent"), default=np.nan),
        "recent_games_count": int(max(0, round(safe_float(row.get("recent_games_count"), default=0.0)))),
        "minutes_floor_recent": safe_float(row.get("minutes_floor_recent"), default=np.nan),
        "minutes_p25_recent": safe_float(row.get("minutes_p25_recent"), default=np.nan),
        "minutes_median_recent": safe_float(row.get("minutes_median_recent"), default=np.nan),
        "minutes_range_recent": safe_float(row.get("minutes_range_recent"), default=np.nan),
        "expected_minutes_band_low": safe_float(row.get("expected_minutes_band_low"), default=np.nan),
        "expected_minutes_band_high": safe_float(row.get("expected_minutes_band_high"), default=np.nan),
        "expected_minutes_band_width": safe_float(row.get("expected_minutes_band_width"), default=np.nan),
        "bench_role_flag": _bool_like(row.get("bench_role_flag"), default=False),
        "starter_status_recent": safe_float(row.get("starter_status_recent"), default=np.nan),
        "starter_status_change_count": int(max(0, round(safe_float(row.get("starter_status_change_count"), default=0.0)))),
        "rotation_volatility_score": _clip01(row.get("rotation_volatility_score"), default=0.50),
        "blowout_minutes_sensitivity": _clip01(row.get("blowout_minutes_sensitivity"), default=0.50),
        "foul_rate_minutes_loss_risk": _clip01(row.get("foul_rate_minutes_loss_risk"), default=0.50),
        "coach_trust_score": _clip01(row.get("coach_trust_score"), default=0.50),
        "market_over_price": over_price,
        "market_under_price": under_price,
        "side_break_even_prob": _american_break_even_prob(over_price) if np.isfinite(over_price) else fallback_break_even,
        "opposite_side_break_even": _american_break_even_prob(under_price),
        "line_minus_trb_q75": upper_band_diag["line_minus_trb_q75"],
        "line_minus_trb_q90": upper_band_diag["line_minus_trb_q90"],
        "upper_band_line_penalty": float(upper_band_diag["upper_band_line_penalty"]),
        "upper_band_line_flag": bool(upper_band_diag["upper_band_line_flag"]),
        "upper_band_line_reason": str(upper_band_diag["upper_band_line_reason"]),
        "low_line_role_volatility_flag": bool(low_line_diag["low_line_role_volatility_flag"]),
        "low_line_role_volatility_penalty": float(low_line_diag["low_line_role_volatility_penalty"]),
        "low_line_role_volatility_reason": str(low_line_diag["low_line_role_volatility_reason"]),
        "rebound_supply_penalty": float(supply_diag["rebound_supply_penalty"]),
        "rebound_supply_reason": str(supply_diag["rebound_supply_reason"]),
        "rebound_share_competition_penalty": float(share_diag["rebound_share_competition_penalty"]),
        "rebound_share_reason": str(share_diag["rebound_share_reason"]),
        "role_pathway_shift_score": 0.0,
        "adjustment_pressure": 0.0,
        "boundary_shadow": False,
        "price_dependent": False,
        "severe_veto": False,
        "trb_over_bucket": "NOT_APPLICABLE",
        "trb_over_bucket_reasons": "",
        "trb_over_bucket_count": 0,
        "total_rebound_penalty": 0.0,
        "adjusted_stress_prob": np.nan,
        "adjusted_lcb_edge": np.nan,
        "opposite_side_candidate_flag": False,
        "opposite_side_discovery_enabled": bool(rebound_config.get("opposite_side_discovery", {}).get("enabled", True)),
        "opposite_side_discovery_when_penalized": bool(
            rebound_config.get("opposite_side_discovery", {}).get("evaluate_under_when_over_penalized", True)
        ),
        "opposite_side_reason": "",
        "opposite_side_market_type": "TRB_UNDER",
        "opposite_side_line": market_line,
        "opposite_side_odds": under_price,
        "opposite_side_break_even": _american_break_even_prob(under_price),
        "opposite_side_stress_prob": np.nan,
        "opposite_side_lcb_edge": np.nan,
        "opposite_side_decision": "not_evaluated",
        "rebound_diagnostic_segment": "NOT_APPLICABLE",
    }
    if not active:
        return base_context

    pred_ast = safe_float(row.get("pred_AST"), default=np.nan)
    baseline_ast = safe_float(row.get("baseline_AST"), default=np.nan)
    if np.isfinite(pred_ast) and np.isfinite(baseline_ast):
        role_denominator = max(1.5, abs(baseline_ast) * 0.75 + 1.0)
        role_shift_score = float(np.clip((pred_ast - baseline_ast) / role_denominator, 0.0, 1.0))
    else:
        role_shift_score = _clip01(row.get("role_shift_risk"), default=0.0)
    base_context["role_pathway_shift_score"] = role_shift_score

    bucket_tokens: list[str] = []
    bucket_reasons: list[str] = []
    if base_context["upper_band_line_flag"]:
        bucket_tokens.append("TRB_OVER_UPPER_BAND")
        bucket_reasons.append(str(base_context["upper_band_line_reason"]))
    if base_context["low_line_role_volatility_flag"]:
        bucket_tokens.append("TRB_OVER_LOW_LINE_ROLE_VOLATILE")
        bucket_reasons.append(str(base_context["low_line_role_volatility_reason"]))
    if float(base_context["rebound_supply_penalty"]) > 0.0:
        bucket_tokens.append("TRB_OVER_SUPPLY_DEPENDENT")
        bucket_reasons.append(str(base_context["rebound_supply_reason"]))
    if float(base_context["rebound_share_competition_penalty"]) > 0.0:
        bucket_tokens.append("TRB_OVER_SHARE_COMPETITION")
        bucket_reasons.append(str(base_context["rebound_share_reason"]))
    if not bucket_tokens:
        bucket_tokens = ["TRB_OVER_STABLE"]
        bucket_reasons = ["no_special_rebound_penalty"]

    low_line_cfg = rebound_config.get("low_line_role_volatility", {})
    structural_minutes_floor = 0.75 * float(low_line_cfg.get("min_minutes_floor", 18.0))
    structurally_unsafe = bool(
        base_context["low_line_role_volatility_flag"]
        and (
            (
                np.isfinite(base_context["minutes_floor_recent"])
                and float(base_context["minutes_floor_recent"]) < structural_minutes_floor
            )
            or (
                np.isfinite(base_context["expected_minutes_band_low"])
                and float(base_context["expected_minutes_band_low"]) < structural_minutes_floor
            )
            or (
                bool(base_context["bench_role_flag"])
                and float(base_context["rotation_volatility_score"]) >= 0.75
            )
        )
    )

    total_penalty = float(
        np.clip(
            float(base_context["upper_band_line_penalty"])
            + float(base_context["low_line_role_volatility_penalty"])
            + float(base_context["rebound_supply_penalty"])
            + float(base_context["rebound_share_competition_penalty"]),
            0.0,
            0.95,
        )
    )
    max_total_penalty = (
        float(rebound_config.get("upper_band", {}).get("penalty_strength", 0.12))
        + float(rebound_config.get("low_line_role_volatility", {}).get("penalty_strength", 0.10))
        + float(rebound_config.get("low_line_role_volatility", {}).get("bench_role_penalty", 0.08))
        + float(rebound_config.get("rebound_supply", {}).get("penalty_strength", 0.08))
        + float(rebound_config.get("rebound_share", {}).get("penalty_strength", 0.07))
    )
    adjustment_pressure = float(np.clip(total_penalty / max(max_total_penalty, 1e-6), 0.0, 1.0))
    base_context.update(
        {
            "supply_dependency_score": adjustment_pressure,
            "supply_dependency_classification": "balanced_playable" if bucket_tokens == ["TRB_OVER_STABLE"] else "price_dependent",
            "adjustment_pressure": adjustment_pressure,
            "boundary_shadow": False,
            "price_dependent": bool(total_penalty > 0.0),
            "severe_veto": structurally_unsafe,
            "trb_over_bucket": "|".join(bucket_tokens),
            "trb_over_bucket_reasons": " | ".join(bucket_reasons),
            "trb_over_bucket_count": int(len(bucket_tokens)),
            "total_rebound_penalty": total_penalty,
            "rebound_diagnostic_segment": bucket_tokens[0],
        }
    )
    return base_context


def _augment_risk_profile_for_supply(
    risk_profile: dict[str, float | bool],
    supply_context: dict[str, Any],
) -> dict[str, float | bool]:
    out = dict(risk_profile)
    if not bool(supply_context.get("supply_dependency_active", False)):
        return out
    adjustment_pressure = _clip01(supply_context.get("adjustment_pressure"), default=0.0)
    if adjustment_pressure <= 0.0:
        return out

    out["volatility_score"] = float(
        np.clip(
            float(out.get("volatility_score", 0.0))
            + 0.20 * adjustment_pressure
            + 0.12 * _clip01(supply_context.get("low_line_role_volatility_penalty"), default=0.0),
            0.0,
            1.0,
        )
    )
    out["risk_penalty"] = float(
        np.clip(float(out.get("risk_penalty", 0.0)) + 0.26 * adjustment_pressure, 0.0, 0.95)
    )
    return out


def _apply_supply_dependent_adjustments(
    abs_gap: float,
    expected_rate: float,
    recommendation: str,
    supply_context: dict[str, Any],
) -> tuple[float, float, str, dict[str, Any]]:
    if not bool(supply_context.get("supply_dependency_active", False)):
        return abs_gap, expected_rate, recommendation, supply_context

    total_penalty = float(np.clip(supply_context.get("total_rebound_penalty", 0.0), 0.0, 0.95))
    adjustment_pressure = _clip01(supply_context.get("adjustment_pressure"), default=0.0)
    adjusted_gap = float(max(0.0, abs_gap * (1.0 - min(0.70, 1.35 * total_penalty))))
    adjusted_rate = float(np.clip(expected_rate - total_penalty, 0.40, 0.95))
    adjusted_recommendation = recommendation

    if float(supply_context.get("upper_band_line_penalty", 0.0)) > 0.0 or float(supply_context.get("rebound_share_competition_penalty", 0.0)) > 0.0:
        adjusted_recommendation = _downgrade_recommendation(adjusted_recommendation)
    if float(supply_context.get("low_line_role_volatility_penalty", 0.0)) > 0.0 and adjusted_recommendation in {"elite", "strong", "consider"}:
        adjusted_recommendation = _downgrade_recommendation(adjusted_recommendation)
    if total_penalty >= 0.10 and adjusted_recommendation in {"elite", "strong"}:
        adjusted_recommendation = "consider"
    if adjusted_rate < 0.53 and adjusted_recommendation == "consider":
        adjusted_recommendation = "pass"
    if bool(supply_context.get("severe_veto", False)):
        adjusted_recommendation = "pass"

    updated_context = dict(supply_context)
    updated_context["price_dependent"] = bool(total_penalty > 0.0)
    updated_context["boundary_shadow"] = bool(total_penalty >= 0.12)
    return adjusted_gap, adjusted_rate, adjusted_recommendation, updated_context


def _probability_recommendation(probability: float, lcb_edge: float) -> str:
    if probability >= 0.62 and lcb_edge >= 0.04:
        return "strong"
    if probability >= 0.56 and lcb_edge >= 0.01:
        return "consider"
    return "pass"


def _finalize_rebound_diagnostic_state(
    supply_context: dict[str, Any],
    *,
    expected_rate: float,
    expected_push_rate: float,
    uncertainty_penalty: float,
    line_decision: dict[str, Any],
    recommendation: str,
) -> tuple[dict[str, Any], str]:
    if not bool(supply_context.get("supply_dependency_active", False)):
        return supply_context, recommendation

    out = dict(supply_context)
    break_even = safe_float(out.get("side_break_even_prob"), default=np.nan)
    if not np.isfinite(break_even):
        break_even = _american_break_even_prob(-110)
    total_penalty = float(np.clip(out.get("total_rebound_penalty", 0.0), 0.0, 0.95))
    adjusted_stress_prob = float(np.clip(expected_rate, 0.0, 1.0 - max(0.0, expected_push_rate)))
    adjusted_lcb_edge = float(adjusted_stress_prob - break_even)

    if bool(out.get("severe_veto", False)):
        classification = "boundary_shadow_hard_pass"
        next_recommendation = "pass"
    elif total_penalty <= 0.0:
        classification = "balanced_playable"
        next_recommendation = recommendation
    elif adjusted_lcb_edge > 0.015:
        classification = "balanced_playable"
        next_recommendation = recommendation
    elif adjusted_lcb_edge > -0.015:
        classification = "price_dependent"
        next_recommendation = _downgrade_recommendation(recommendation) if recommendation in {"elite", "strong"} else recommendation
    else:
        classification = "boundary_shadow"
        next_recommendation = "pass" if recommendation == "consider" else _downgrade_recommendation(recommendation)

    out["adjusted_stress_prob"] = adjusted_stress_prob
    out["adjusted_lcb_edge"] = adjusted_lcb_edge
    out["supply_dependency_classification"] = classification
    out["boundary_shadow"] = bool(classification in {"boundary_shadow", "boundary_shadow_hard_pass"})
    out["price_dependent"] = bool(classification == "price_dependent")

    stress_enabled = bool(out.get("opposite_side_discovery_enabled", True)) and bool(
        out.get("opposite_side_discovery_when_penalized", True)
    )
    if not stress_enabled or total_penalty <= 0.0:
        out["opposite_side_decision"] = "not_penalized"
        return out, next_recommendation

    opposite_break_even = safe_float(out.get("opposite_side_break_even"), default=np.nan)
    opposite_price = safe_float(out.get("market_under_price"), default=np.nan)
    if not np.isfinite(opposite_price) or not np.isfinite(opposite_break_even):
        out["opposite_side_decision"] = "reject_price_unavailable"
        out["opposite_side_reason"] = "under_price_unavailable"
        return out, next_recommendation

    opposite_base_prob = float(np.clip(line_decision.get("opposite_direction_prob", 0.0), 0.0, 1.0))
    resolved_share = max(1e-6, 1.0 - max(0.0, expected_push_rate))
    forecastability = bool(
        _clip01(out.get("coach_trust_score"), default=0.50) >= 0.35
        and _clip01(out.get("rotation_volatility_score"), default=0.50) <= 0.85
    )
    opposite_stress_prob = float(
        np.clip(
            opposite_base_prob
            + 0.85 * total_penalty
            + 0.20 * float(out.get("rebound_supply_penalty", 0.0))
            + 0.15 * float(out.get("rebound_share_competition_penalty", 0.0)),
            0.0,
            resolved_share,
        )
    )
    opposite_lcb_edge = float(opposite_stress_prob - opposite_break_even - uncertainty_penalty)
    out["opposite_side_candidate_flag"] = True
    out["opposite_side_reason"] = f"under_prob={opposite_stress_prob:.3f};break_even={opposite_break_even:.3f};penalty={total_penalty:.3f}"
    out["opposite_side_stress_prob"] = opposite_stress_prob
    out["opposite_side_lcb_edge"] = opposite_lcb_edge

    if not forecastability:
        out["opposite_side_decision"] = "reject_forecastability"
    elif opposite_stress_prob <= opposite_break_even:
        out["opposite_side_decision"] = "reject_break_even"
    elif opposite_lcb_edge <= 0.005:
        out["opposite_side_decision"] = "reject_lcb_edge"
    else:
        out["opposite_side_decision"] = "promote_under_candidate"
    return out, next_recommendation


def _build_opposite_side_candidate_row(base_row: dict[str, Any], supply_context: dict[str, Any]) -> dict[str, Any] | None:
    if str(supply_context.get("opposite_side_decision")) != "promote_under_candidate":
        return None
    market_line = safe_float(base_row.get("market_line"), default=np.nan)
    abs_gap = safe_float(base_row.get("adjusted_abs_edge"), default=safe_float(base_row.get("abs_edge"), default=0.0))
    gap_scale = float(np.clip(float(supply_context.get("opposite_side_lcb_edge", 0.0)) / 0.06, 0.20, 0.80))
    mirrored_gap = float(max(0.05, abs_gap * gap_scale))
    prediction = float(market_line - mirrored_gap) if np.isfinite(market_line) else np.nan
    edge = float(prediction - market_line) if np.isfinite(prediction) and np.isfinite(market_line) else np.nan
    expected_push_rate = safe_float(base_row.get("expected_push_rate"), default=0.0)
    expected_win_rate = float(np.clip(supply_context.get("opposite_side_stress_prob", 0.0), 0.0, 1.0 - max(0.0, expected_push_rate)))
    expected_loss_rate = float(np.clip(1.0 - expected_win_rate - expected_push_rate, 0.0, 1.0))
    recommendation = _probability_recommendation(expected_win_rate, float(supply_context.get("opposite_side_lcb_edge", 0.0)))
    if recommendation == "pass":
        recommendation = "consider"
    direction_conditional_prob = float(expected_win_rate / max(1e-6, 1.0 - expected_push_rate))

    candidate = dict(base_row)
    candidate.update(
        {
            "direction": "UNDER",
            "prediction": prediction,
            "raw_prediction": prediction,
            "raw_prediction_effective": prediction,
            "line_decision_action": "UNDER",
            "line_action_direction": "UNDER",
            "line_action_is_opposite": True,
            "line_decision_trade_eligible": True,
            "line_over_prob": float(np.clip(1.0 - expected_win_rate - expected_push_rate, 0.0, 1.0)),
            "line_under_prob": expected_win_rate,
            "line_no_trade_prob": expected_push_rate,
            "line_chosen_direction_prob": expected_win_rate,
            "line_opposite_direction_prob": float(np.clip(1.0 - expected_win_rate - expected_push_rate, 0.0, 1.0)),
            "line_chosen_direction_conditional_prob": direction_conditional_prob,
            "line_opposite_direction_conditional_prob": float(np.clip(1.0 - direction_conditional_prob, 0.0, 1.0)),
            "line_preferred_direction": "UNDER",
            "line_preferred_direction_prob": expected_win_rate,
            "line_preferred_direction_conditional_prob": direction_conditional_prob,
            "line_action_prob": expected_win_rate,
            "line_action_conditional_prob": direction_conditional_prob,
            "line_action_expected_win_rate": expected_win_rate,
            "line_decision_source": f"{base_row.get('line_decision_source', 'rebound_diag')}+rebound_under_discovery",
            "edge": edge,
            "raw_edge": edge,
            "abs_edge": abs(edge) if np.isfinite(edge) else np.nan,
            "recommendation": recommendation,
            "raw_recommendation": recommendation,
            "expected_win_rate": expected_win_rate,
            "expected_win_rate_pre_sidecar": expected_win_rate,
            "raw_expected_win_rate": expected_win_rate,
            "bayesian_expected_win_rate": expected_win_rate,
            "expected_loss_rate": expected_loss_rate,
            "raw_expected_loss_rate": expected_loss_rate,
            "adjusted_abs_edge": abs(edge) if np.isfinite(edge) else np.nan,
            "market_side_price": safe_float(supply_context.get("market_under_price"), default=np.nan),
            "market_side_break_even": safe_float(supply_context.get("opposite_side_break_even"), default=np.nan),
            "supply_dependency_classification": "opposite_side_under_candidate",
            "rebound_diagnostic_segment": "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY",
            "trb_over_bucket": "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY",
            "trb_over_bucket_reasons": str(supply_context.get("opposite_side_reason", "")),
            "trb_over_bucket_count": 1,
            "opposite_side_decision": "promoted_to_candidate",
        }
    )
    return candidate


def _risk_profile(
    row: pd.Series,
    target: str,
    pred: float,
    direction: str,
    belief_uncertainty_lower: float,
    belief_uncertainty_upper: float,
) -> dict[str, float | bool]:
    sigma = max(0.0, safe_float(row.get(f"{target}_uncertainty_sigma"), default=0.0))
    pred_scale = max(1.0, abs(pred))
    sigma_ratio = sigma / pred_scale
    sigma_norm = float(np.clip(sigma_ratio / 0.45, 0.0, 1.0))
    spike_probability = _clip01(row.get(f"{target}_spike_probability"), default=0.50)
    belief = float(
        normalize_belief_uncertainty(
            row.get("belief_uncertainty"),
            default=0.50,
            lower=float(belief_uncertainty_lower),
            upper=float(belief_uncertainty_upper),
        )
    )
    volatility_regime = _clip01(row.get("volatility_regime_risk"), default=sigma_norm)
    feasibility = _clip01(row.get("feasibility"), default=0.60)
    minutes_instability = float(np.clip(1.0 - feasibility, 0.0, 1.0))
    if direction == "UNDER":
        tail_imbalance = spike_probability
    elif direction == "OVER":
        tail_imbalance = 1.0 - spike_probability
    else:
        tail_imbalance = 0.50

    volatility_score = float(
        np.clip(
            0.28 * sigma_norm
            + 0.28 * spike_probability
            + 0.18 * belief
            + 0.14 * volatility_regime
            + 0.12 * minutes_instability,
            0.0,
            1.0,
        )
    )
    risk_penalty = float(np.clip(0.80 * volatility_score + 0.20 * tail_imbalance, 0.0, 0.90))
    spike_flag = bool(
        (spike_probability >= 0.72 and sigma_norm >= 0.30)
        or (volatility_score >= 0.67)
        or (belief >= 0.75 and sigma_norm >= 0.20)
    )
    return {
        "sigma_ratio": sigma_ratio,
        "sigma_norm": sigma_norm,
        "spike_probability": spike_probability,
        "belief_uncertainty": belief,
        "volatility_regime_risk": volatility_regime,
        "minutes_instability": minutes_instability,
        "tail_imbalance": tail_imbalance,
        "volatility_score": volatility_score,
        "risk_penalty": risk_penalty,
        "spike_flag": spike_flag,
    }


def _apply_volatility_adjustments(
    abs_gap: float,
    expected_rate: float,
    recommendation: str,
    risk_profile: dict[str, float | bool],
) -> tuple[float, float, str]:
    risk_penalty = float(risk_profile["risk_penalty"])
    spike_flag = bool(risk_profile["spike_flag"])
    adjusted_gap = float(max(0.0, abs_gap * (1.0 - 0.60 * risk_penalty)))
    if spike_flag:
        adjusted_gap *= 0.85

    margin = float(expected_rate - 0.50)
    adjusted_rate = float(0.50 + margin * max(0.0, 1.0 - 0.90 * risk_penalty))
    if spike_flag:
        adjusted_rate -= 0.0125
    adjusted_rate = float(np.clip(adjusted_rate, 0.50, 0.95))

    adjusted_recommendation = recommendation
    if spike_flag:
        if recommendation == "strong":
            adjusted_recommendation = "consider"
    elif risk_penalty >= 0.55 and recommendation == "strong":
        adjusted_recommendation = "consider"
    return adjusted_gap, adjusted_rate, adjusted_recommendation


def build_play_rows(
    slate_df: pd.DataFrame,
    history_lookup: dict[str, dict],
    line_decision_lookup: dict[str, dict] | None = None,
    volatility_adjustment: bool = True,
    belief_uncertainty_lower: float = BELIEF_UNCERTAINTY_LOWER,
    belief_uncertainty_upper: float = BELIEF_UNCERTAINTY_UPPER,
    market_regression_floor: float = DEFAULT_MARKET_REGRESSION_FLOOR,
    market_regression_ceiling: float = DEFAULT_MARKET_REGRESSION_CEILING,
    line_decision_enabled: bool = True,
    line_decision_config: LineDecisionConfig | None = None,
    rebound_diagnostics_config: dict[str, Any] | None = None,
    american_odds: int = -110,
    selector_run_time: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    resolved_rebound_config = _resolve_rebound_diagnostics_config(rebound_diagnostics_config)
    for _, row in slate_df.iterrows():
        belief_raw = safe_float(row.get("belief_uncertainty"), default=1.0)
        belief = float(
            normalize_belief_uncertainty(
                belief_raw,
                default=1.0,
                lower=float(belief_uncertainty_lower),
                upper=float(belief_uncertainty_upper),
            )
        )
        belief_conf = float(
            belief_confidence_factor(
                belief_raw,
                default=1.0,
                lower=float(belief_uncertainty_lower),
                upper=float(belief_uncertainty_upper),
            )
        )
        feas = max(0.0, safe_float(row.get("feasibility"), default=0.0))
        fallback_blend = safe_float(row.get("fallback_blend"), default=0.0)
        for target in TARGETS:
            history_info = history_lookup.get(target)
            raw_pred = safe_float(row.get(f"pred_{target}"))
            market = safe_float(row.get(f"market_{target}"))
            if np.isnan(raw_pred) or np.isnan(market):
                continue

            prediction_shrink_lambda = _prediction_shrink_lambda(
                belief_conf=belief_conf,
                feasibility=feas,
                fallback_blend=fallback_blend,
                floor=float(market_regression_floor),
                ceiling=float(market_regression_ceiling),
            )
            pred = float(market + prediction_shrink_lambda * (raw_pred - market))
            edge = pred - market
            raw_edge = raw_pred - market
            if edge == 0.0:
                direction = "PUSH"
            elif edge > 0.0:
                direction = "OVER"
            else:
                direction = "UNDER"
            # Use the same edge scale as historical calibration (raw model-vs-market)
            # when computing percentiles/recommendations. The shrunk edge remains the
            # execution margin used downstream for risk and sizing.
            abs_gap = abs(edge)
            raw_abs_gap = abs(raw_edge)
            if history_info is None:
                heuristic = heuristic_percentile_and_rate(target, abs_gap)
                gap_pct = float(heuristic["gap_percentile"])
                expected_triplet = heuristic
            else:
                gap_pct = percentile_of_gap(history_info["gaps_sorted"], raw_abs_gap)
                expected_triplet = expected_rate_for(target, gap_pct, history_info)
            recommendation = classify_play(target, gap_pct)
            risk_profile = _risk_profile(
                row,
                target,
                pred,
                direction,
                belief_uncertainty_lower=float(belief_uncertainty_lower),
                belief_uncertainty_upper=float(belief_uncertainty_upper),
            )
            supply_context = _supply_dependent_context(
                row,
                target,
                direction,
                market,
                rebound_diagnostics_config=resolved_rebound_config,
                american_odds=american_odds,
            )
            risk_profile = _augment_risk_profile_for_supply(risk_profile, supply_context)
            base_expected_rate = float(expected_triplet["expected_win_rate"])
            adjusted_abs_gap = abs_gap
            adjusted_expected_rate = base_expected_rate
            adjusted_recommendation = recommendation
            if volatility_adjustment:
                adjusted_abs_gap, adjusted_expected_rate, adjusted_recommendation = _apply_volatility_adjustments(
                    abs_gap,
                    base_expected_rate,
                    recommendation,
                    risk_profile,
                )
            adjusted_abs_gap, adjusted_expected_rate, adjusted_recommendation, supply_context = _apply_supply_dependent_adjustments(
                adjusted_abs_gap,
                adjusted_expected_rate,
                adjusted_recommendation,
                supply_context,
            )
            adjusted_gap_pct = gap_pct

            posterior_variance = float(expected_triplet.get("posterior_variance", 0.25))
            posterior_std = float(np.sqrt(max(0.0, posterior_variance)))
            uncertainty_discount = float(np.clip(0.35 * posterior_std, 0.0, 0.07))
            adjusted_expected_rate = float(np.clip(adjusted_expected_rate - uncertainty_discount, 0.40, 0.95))

            historical_push_rate = float(np.clip(expected_triplet.get("expected_push_rate", 0.0), 0.0, 1.0))
            prior_expected_loss_rate = float(np.clip(1.0 - adjusted_expected_rate - historical_push_rate, 0.0, 1.0))
            if direction == "PUSH":
                line_decision = {
                    "over_prob": 0.0,
                    "under_prob": 0.0,
                    "no_trade_prob": 1.0,
                    "chosen_direction_prob": 0.0,
                    "opposite_direction_prob": 0.0,
                    "chosen_direction_conditional_prob": 0.0,
                    "opposite_direction_conditional_prob": 0.0,
                    "conditional_prob_gap": 0.0,
                    "trade_prob_floor": float((line_decision_config or LineDecisionConfig()).min_trade_prob if LineDecisionConfig is not None else 0.63),
                    "trade_eligible": False,
                    "action": "NO_TRADE",
                    "source": "zero_edge",
                    "support_rows": 0.0,
                    "support_strength": 0.0,
                    "sigma_pressure": 1.0,
                    "instability_score": 1.0,
                    "fragility_score": 1.0,
                    "empirical_blend_weight": 0.0,
                }
            elif line_decision_enabled:
                line_decision = estimate_line_decision(
                    lookup=line_decision_lookup or {},
                    target=target,
                    prediction=pred,
                    market_line=market,
                    direction=direction,
                    gap_percentile=gap_pct,
                    uncertainty_sigma=safe_float(row.get(f"{target}_uncertainty_sigma"), default=np.nan),
                    belief_confidence_factor=belief_conf,
                    feasibility=feas,
                    history_rows=row.get("history_rows"),
                    market_books=row.get(f"market_books_{target}"),
                    fallback_blend=fallback_blend,
                    prior_direction_win_rate=adjusted_expected_rate,
                    prior_neutral_rate=historical_push_rate,
                    config=line_decision_config,
                )
            else:
                if direction == "OVER":
                    line_over_prob = adjusted_expected_rate
                    line_under_prob = prior_expected_loss_rate
                else:
                    line_over_prob = prior_expected_loss_rate
                    line_under_prob = adjusted_expected_rate
                line_decision = {
                    "over_prob": float(line_over_prob),
                    "under_prob": float(line_under_prob),
                    "no_trade_prob": float(historical_push_rate),
                    "chosen_direction_prob": float(adjusted_expected_rate),
                    "opposite_direction_prob": float(prior_expected_loss_rate),
                    "chosen_direction_conditional_prob": float(adjusted_expected_rate / max(1e-9, adjusted_expected_rate + prior_expected_loss_rate)),
                    "opposite_direction_conditional_prob": float(prior_expected_loss_rate / max(1e-9, adjusted_expected_rate + prior_expected_loss_rate)),
                    "conditional_prob_gap": float((adjusted_expected_rate - prior_expected_loss_rate) / max(1e-9, adjusted_expected_rate + prior_expected_loss_rate)),
                    "trade_prob_floor": float((line_decision_config or LineDecisionConfig()).min_trade_prob if LineDecisionConfig is not None else 0.63),
                    "trade_eligible": True,
                    "action": direction,
                    "source": "disabled",
                    "support_rows": 0.0,
                    "support_strength": 0.0,
                    "sigma_pressure": 0.0,
                    "instability_score": 0.0,
                    "fragility_score": 0.0,
                    "empirical_blend_weight": 0.0,
                }
            if bool(supply_context.get("severe_veto", False)):
                line_decision = dict(line_decision)
                line_decision["trade_eligible"] = False
                line_decision["action"] = "NO_TRADE"
                line_decision["no_trade_prob"] = float(
                    np.clip(
                        safe_float(line_decision.get("no_trade_prob"), default=historical_push_rate)
                        + 0.18 * _clip01(supply_context.get("adjustment_pressure"), default=0.0),
                        0.0,
                        0.90,
                    )
                )
                source_text = str(line_decision.get("source", "unknown"))
                line_decision["source"] = f"{source_text}+rebound_supply_veto"

            supply_context, adjusted_recommendation = _finalize_rebound_diagnostic_state(
                supply_context,
                expected_rate=adjusted_expected_rate,
                expected_push_rate=historical_push_rate,
                uncertainty_penalty=uncertainty_discount,
                line_decision=line_decision,
                recommendation=adjusted_recommendation,
            )

            line_over_prob = float(np.clip(line_decision.get("over_prob", 0.5), 0.0, 1.0))
            line_under_prob = float(np.clip(line_decision.get("under_prob", 0.5), 0.0, 1.0))
            line_no_trade_prob = float(np.clip(line_decision.get("no_trade_prob", 0.0), 0.0, 1.0))
            chosen_direction_prob = float(np.clip(line_decision.get("chosen_direction_prob", adjusted_expected_rate), 0.0, 1.0))
            opposite_direction_prob = float(np.clip(line_decision.get("opposite_direction_prob", prior_expected_loss_rate), 0.0, 1.0))
            chosen_direction_conditional_prob = float(
                np.clip(line_decision.get("chosen_direction_conditional_prob", adjusted_expected_rate), 0.0, 1.0)
            )
            opposite_direction_conditional_prob = float(
                np.clip(line_decision.get("opposite_direction_conditional_prob", prior_expected_loss_rate), 0.0, 1.0)
            )
            expected_push_rate = historical_push_rate
            trade_eligible = bool(line_decision.get("trade_eligible", False))
            action_direction = str(line_decision.get("action", "NO_TRADE")).upper().strip()
            if action_direction not in {"OVER", "UNDER"}:
                action_direction = direction
            action_is_opposite = bool(
                trade_eligible
                and action_direction in {"OVER", "UNDER"}
                and action_direction != direction
            )
            action_conditional_prob = chosen_direction_conditional_prob if action_direction == direction else opposite_direction_conditional_prob
            action_side_prob = line_over_prob if action_direction == "OVER" else line_under_prob
            action_expected_rate_raw = float(np.clip(action_conditional_prob * max(0.0, 1.0 - historical_push_rate), 0.0, 0.99))
            sidecar_blend_weight = float(np.clip(line_decision.get("empirical_blend_weight", 0.0), 0.0, 1.0))
            if action_is_opposite:
                sidecar_blend_weight = max(0.55, sidecar_blend_weight)
            sidecar_effective_rate = float(
                np.clip(
                    sidecar_blend_weight * action_expected_rate_raw
                    + (1.0 - sidecar_blend_weight) * adjusted_expected_rate,
                    0.50,
                    0.95,
                )
            )
            effective_direction = action_direction if trade_eligible else direction
            effective_prediction = pred
            effective_raw_prediction = raw_pred
            if action_is_opposite:
                effective_prediction = float((2.0 * market) - pred)
                effective_raw_prediction = float((2.0 * market) - raw_pred)
            effective_edge = float(effective_prediction - market)
            effective_raw_edge = float(effective_raw_prediction - market)
            effective_abs_gap = abs(effective_edge)
            effective_risk_profile = risk_profile
            if action_is_opposite:
                effective_risk_profile = _risk_profile(
                    row,
                    target,
                    effective_prediction,
                    effective_direction,
                    belief_uncertainty_lower=float(belief_uncertainty_lower),
                    belief_uncertainty_upper=float(belief_uncertainty_upper),
                )
            effective_expected_rate = sidecar_effective_rate if trade_eligible else adjusted_expected_rate
            expected_loss_rate = float(np.clip(1.0 - effective_expected_rate - historical_push_rate, 0.0, 1.0))
            if not trade_eligible:
                adjusted_recommendation = "pass"
            elif adjusted_recommendation == "pass" and float(supply_context.get("total_rebound_penalty", 0.0)) <= 0.0:
                adjusted_recommendation = "consider"
            elif adjusted_recommendation == "elite" and effective_expected_rate < 0.62:
                adjusted_recommendation = "strong"
            elif adjusted_recommendation in {"elite", "strong"} and effective_expected_rate < 0.58:
                adjusted_recommendation = "consider"

            confidence_score = (
                effective_abs_gap
                * belief_conf
                * feas
                * (1.0 - float(effective_risk_profile["risk_penalty"]))
                * (1.0 - min(0.75, posterior_std))
                * (1.0 - min(0.80, line_no_trade_prob))
            )
            market_over_price = safe_float(row.get(f"market_over_price_{target}"), default=np.nan)
            market_under_price = safe_float(row.get(f"market_under_price_{target}"), default=np.nan)
            market_side_price = _market_side_price(row, target, effective_direction)
            market_side_break_even = _american_break_even_prob(market_side_price) if np.isfinite(market_side_price) else np.nan
            row_payload = {
                "player": row["player"],
                "player_id": safe_float(row.get("player_id"), default=np.nan),
                "player_name": row.get("player"),
                "team": row.get("team"),
                "opponent": row.get("opponent"),
                "game_id": row.get("market_event_id"),
                "market_date": row.get("market_date"),
                "market_player_raw": row.get("market_player_raw"),
                "market_event_id": row.get("market_event_id"),
                "market_commence_time_utc": row.get("market_commence_time_utc"),
                "market_home_team": row.get("market_home_team"),
                "market_away_team": row.get("market_away_team"),
                "target": target,
                "direction": effective_direction,
                "side": effective_direction,
                "model_direction": direction,
                "market_type": _market_type(target, effective_direction),
                "prediction": effective_prediction,
                "raw_prediction": raw_pred,
                "raw_prediction_effective": effective_raw_prediction,
                "line_decision_action": str(line_decision.get("action", "NO_TRADE")),
                "line_action_direction": action_direction,
                "line_action_is_opposite": action_is_opposite,
                "line_decision_trade_eligible": bool(line_decision.get("trade_eligible", False)),
                "line_over_prob": line_over_prob,
                "line_under_prob": line_under_prob,
                "line_no_trade_prob": line_no_trade_prob,
                "line_chosen_direction_prob": chosen_direction_prob,
                "line_opposite_direction_prob": opposite_direction_prob,
                "line_chosen_direction_conditional_prob": chosen_direction_conditional_prob,
                "line_opposite_direction_conditional_prob": opposite_direction_conditional_prob,
                "line_preferred_direction": str(line_decision.get("preferred_direction", action_direction)),
                "line_preferred_direction_prob": float(line_decision.get("preferred_direction_prob", action_side_prob)),
                "line_preferred_direction_conditional_prob": float(
                    line_decision.get("preferred_direction_conditional_prob", action_conditional_prob)
                ),
                "line_action_prob": action_side_prob,
                "line_action_conditional_prob": action_conditional_prob,
                "line_action_expected_win_rate": action_expected_rate_raw,
                "line_action_empirical_blend_weight": sidecar_blend_weight,
                "line_opposite_context_weight": float(line_decision.get("opposite_context_weight", 0.0)),
                "line_conditional_prob_gap": float(line_decision.get("conditional_prob_gap", 0.0)),
                "line_trade_prob_floor": float(line_decision.get("trade_prob_floor", 0.0)),
                "line_decision_source": str(line_decision.get("source", "unknown")),
                "line_decision_support_rows": float(line_decision.get("support_rows", 0.0)),
                "line_decision_support_strength": float(line_decision.get("support_strength", 0.0)),
                "line_decision_sigma_pressure": float(line_decision.get("sigma_pressure", 1.0)),
                "line_decision_instability_score": float(line_decision.get("instability_score", 1.0)),
                "line_decision_fragility_score": float(line_decision.get("fragility_score", 1.0)),
                "line_decision_empirical_blend_weight": float(line_decision.get("empirical_blend_weight", 0.0)),
                "prediction_shrink_lambda": prediction_shrink_lambda,
                "market_line": market,
                "line": market,
                "market_side_price": market_side_price,
                "market_side_break_even": market_side_break_even,
                "line_at_prediction": market,
                "line_at_odds_snapshot": market,
                "selector_run_time": selector_run_time,
                "prediction_snapshot_time": selector_run_time,
                "odds_snapshot_time": row.get("market_fetched_at_utc", row.get("odds_snapshot_time")),
                "provider": row.get("market_provider", ""),
                "book": row.get("market_book", "aggregate_market_snapshot"),
                "snapshot_id": row.get("market_snapshot_id", ""),
                "price_source": row.get("market_price_source", ""),
                "price_source_type": row.get("market_price_source_type", ""),
                "price_source_hint": row.get("market_price_source_hint", row.get("market_price_source_type", "")),
                "over_price": market_over_price,
                "under_price": market_under_price,
                "edge": effective_edge,
                "raw_edge": effective_raw_edge,
                "abs_edge": effective_abs_gap,
                "raw_gap_percentile": gap_pct,
                "gap_percentile": adjusted_gap_pct,
                "recommendation": adjusted_recommendation,
                "raw_recommendation": recommendation,
                "expected_win_rate": effective_expected_rate,
                "expected_win_rate_pre_sidecar": adjusted_expected_rate,
                "raw_expected_win_rate": float(expected_triplet.get("base_expected_win_rate", base_expected_rate)),
                "bayesian_expected_win_rate": base_expected_rate,
                "expected_push_rate": expected_push_rate,
                "historical_push_rate": historical_push_rate,
                "expected_fragile_rate": float(np.clip(line_no_trade_prob - historical_push_rate, 0.0, 1.0)),
                "raw_expected_push_rate": historical_push_rate,
                "expected_loss_rate": expected_loss_rate,
                "raw_expected_loss_rate": prior_expected_loss_rate,
                "posterior_alpha": float(expected_triplet.get("posterior_alpha", DEFAULT_BETA_PRIOR_ALPHA)),
                "posterior_beta": float(expected_triplet.get("posterior_beta", DEFAULT_BETA_PRIOR_BETA)),
                "posterior_variance": posterior_variance,
                "posterior_ci_low": float(expected_triplet.get("posterior_ci_low", 0.0)),
                "posterior_ci_high": float(expected_triplet.get("posterior_ci_high", 1.0)),
                "calibrated_conditional_win_rate": expected_triplet.get("calibrated_conditional_win_rate"),
                "calibration_weight": float(expected_triplet.get("calibration_weight", 0.0)),
                "calibration_source": str(expected_triplet.get("calibration_source", "unknown")),
                "calibration_bucket": str(expected_triplet.get("bucket", "unknown")),
                "calibration_bucket_rows": int(expected_triplet.get("bucket_rows", 0)),
                "confidence_score": confidence_score,
                "belief_uncertainty": belief_raw,
                "belief_uncertainty_normalized": belief,
                "belief_confidence_factor": belief_conf,
                "feasibility": feas,
                "fallback_blend": fallback_blend,
                "market_books": safe_float(row.get(f"market_books_{target}"), default=np.nan),
                "market_over_price": market_over_price,
                "market_under_price": market_under_price,
                "baseline": safe_float(row.get(f"baseline_{target}"), default=np.nan),
                "baseline_edge": safe_float(row.get(f"baseline_edge_{target}"), default=np.nan),
                "uncertainty_sigma": safe_float(row.get(f"{target}_uncertainty_sigma"), default=np.nan),
                "spike_probability": safe_float(row.get(f"{target}_spike_probability"), default=np.nan),
                "sigma_ratio": float(effective_risk_profile["sigma_ratio"]),
                "volatility_score": float(effective_risk_profile["volatility_score"]),
                "risk_penalty": float(effective_risk_profile["risk_penalty"]),
                "tail_imbalance": float(effective_risk_profile["tail_imbalance"]),
                "spike_flag": bool(effective_risk_profile["spike_flag"]),
                "supply_dependency_active": bool(supply_context.get("supply_dependency_active", False)),
                "supply_dependency_score": float(supply_context.get("supply_dependency_score", 0.0)),
                "supply_dependency_classification": str(supply_context.get("supply_dependency_classification", "not_applicable")),
                "rebound_supply_score": float(supply_context.get("rebound_supply_score", 0.50)),
                "rebound_supply_penalty": float(supply_context.get("rebound_supply_penalty", 0.0)),
                "rebound_supply_reason": str(supply_context.get("rebound_supply_reason", "")),
                "rebound_share_stability": float(supply_context.get("rebound_share_stability", 0.50)),
                "rebound_share_stability_score": float(supply_context.get("rebound_share_stability_score", supply_context.get("rebound_share_stability", 0.50))),
                "rebound_share_estimate": float(supply_context.get("rebound_share_estimate", 0.50)),
                "player_team_rebound_share_recent": safe_float(supply_context.get("player_team_rebound_share_recent"), default=np.nan),
                "player_rebound_share_std": safe_float(supply_context.get("player_rebound_share_std"), default=np.nan),
                "rebound_share_competition_penalty": float(supply_context.get("rebound_share_competition_penalty", 0.0)),
                "rebound_share_reason": str(supply_context.get("rebound_share_reason", "")),
                "team_shooting_efficiency_stress": float(supply_context.get("team_shooting_efficiency_stress", 0.50)),
                "opponent_shooting_efficiency_stress": float(supply_context.get("opponent_shooting_efficiency_stress", 0.50)),
                "wing_rebound_leakage_score": float(supply_context.get("wing_rebound_leakage_score", 0.50)),
                "teammate_rebound_competition": float(supply_context.get("teammate_rebound_competition", 0.50)),
                "teammate_rebound_competition_score": float(supply_context.get("teammate_rebound_competition_score", supply_context.get("teammate_rebound_competition", 0.50))),
                "center_rebound_share_pressure": float(supply_context.get("center_rebound_share_pressure", 0.50)),
                "frontcourt_rebound_overlap_score": float(supply_context.get("frontcourt_rebound_overlap_score", 0.50)),
                "role_pathway_shift_score": float(supply_context.get("role_pathway_shift_score", 0.0)),
                "recent_games_count": int(supply_context.get("recent_games_count", 0)),
                "line_minus_trb_q75": safe_float(supply_context.get("line_minus_trb_q75"), default=np.nan),
                "line_minus_trb_q90": safe_float(supply_context.get("line_minus_trb_q90"), default=np.nan),
                "upper_band_line_penalty": float(supply_context.get("upper_band_line_penalty", 0.0)),
                "upper_band_line_flag": bool(supply_context.get("upper_band_line_flag", False)),
                "upper_band_line_reason": str(supply_context.get("upper_band_line_reason", "")),
                "low_line_role_volatility_flag": bool(supply_context.get("low_line_role_volatility_flag", False)),
                "low_line_role_volatility_penalty": float(supply_context.get("low_line_role_volatility_penalty", 0.0)),
                "low_line_role_volatility_reason": str(supply_context.get("low_line_role_volatility_reason", "")),
                "minutes_floor_recent": safe_float(supply_context.get("minutes_floor_recent"), default=np.nan),
                "minutes_p25_recent": safe_float(supply_context.get("minutes_p25_recent"), default=np.nan),
                "minutes_median_recent": safe_float(supply_context.get("minutes_median_recent"), default=np.nan),
                "minutes_range_recent": safe_float(supply_context.get("minutes_range_recent"), default=np.nan),
                "expected_minutes_band_low": safe_float(supply_context.get("expected_minutes_band_low"), default=np.nan),
                "expected_minutes_band_high": safe_float(supply_context.get("expected_minutes_band_high"), default=np.nan),
                "expected_minutes_band_width": safe_float(supply_context.get("expected_minutes_band_width"), default=np.nan),
                "bench_role_flag": bool(supply_context.get("bench_role_flag", False)),
                "starter_status_recent": safe_float(supply_context.get("starter_status_recent"), default=np.nan),
                "starter_status_change_count": int(supply_context.get("starter_status_change_count", 0)),
                "rotation_volatility_score": float(supply_context.get("rotation_volatility_score", 0.50)),
                "blowout_minutes_sensitivity": float(supply_context.get("blowout_minutes_sensitivity", 0.50)),
                "foul_rate_minutes_loss_risk": safe_float(supply_context.get("foul_rate_minutes_loss_risk"), default=np.nan),
                "coach_trust_score": safe_float(supply_context.get("coach_trust_score"), default=np.nan),
                "projected_team_missed_fga": safe_float(supply_context.get("projected_team_missed_fga"), default=np.nan),
                "projected_opponent_missed_fga": safe_float(supply_context.get("projected_opponent_missed_fga"), default=np.nan),
                "projected_team_missed_fta": safe_float(supply_context.get("projected_team_missed_fta"), default=np.nan),
                "projected_opponent_missed_fta": safe_float(supply_context.get("projected_opponent_missed_fta"), default=np.nan),
                "projected_missed_fga_total": safe_float(supply_context.get("projected_missed_fga_total"), default=np.nan),
                "projected_missed_fta_total": safe_float(supply_context.get("projected_missed_fta_total"), default=np.nan),
                "projected_available_rebound_events": safe_float(supply_context.get("projected_available_rebound_events"), default=np.nan),
                "expected_rebound_chances": safe_float(supply_context.get("expected_rebound_chances"), default=np.nan),
                "team_rebound_pool_size": safe_float(supply_context.get("team_rebound_pool_size"), default=np.nan),
                "pace_rebound_environment": float(supply_context.get("pace_rebound_environment", 0.50)),
                "long_rebound_profile": float(supply_context.get("long_rebound_profile", 0.50)),
                "free_throw_rebound_suppression": float(supply_context.get("free_throw_rebound_suppression", 0.0)),
                "projected_team_fg_pct": safe_float(supply_context.get("projected_team_fg_pct"), default=np.nan),
                "projected_opponent_fg_pct": safe_float(supply_context.get("projected_opponent_fg_pct"), default=np.nan),
                "trb_median_recent": safe_float(supply_context.get("trb_median_recent"), default=np.nan),
                "trb_q75_recent": safe_float(supply_context.get("trb_q75_recent"), default=np.nan),
                "trb_q90_recent": safe_float(supply_context.get("trb_q90_recent"), default=np.nan),
                "trb_over_bucket": str(supply_context.get("trb_over_bucket", "NOT_APPLICABLE")),
                "trb_over_bucket_reasons": str(supply_context.get("trb_over_bucket_reasons", "")),
                "trb_over_bucket_count": int(supply_context.get("trb_over_bucket_count", 0)),
                "total_rebound_penalty": float(supply_context.get("total_rebound_penalty", 0.0)),
                "adjusted_stress_prob": safe_float(supply_context.get("adjusted_stress_prob"), default=np.nan),
                "adjusted_lcb_edge": safe_float(supply_context.get("adjusted_lcb_edge"), default=np.nan),
                "opposite_side_candidate_flag": bool(supply_context.get("opposite_side_candidate_flag", False)),
                "opposite_side_reason": str(supply_context.get("opposite_side_reason", "")),
                "opposite_side_market_type": str(supply_context.get("opposite_side_market_type", "TRB_UNDER")),
                "opposite_side_line": safe_float(supply_context.get("opposite_side_line"), default=np.nan),
                "opposite_side_odds": safe_float(supply_context.get("opposite_side_odds"), default=np.nan),
                "opposite_side_break_even": safe_float(supply_context.get("opposite_side_break_even"), default=np.nan),
                "opposite_side_stress_prob": safe_float(supply_context.get("opposite_side_stress_prob"), default=np.nan),
                "opposite_side_lcb_edge": safe_float(supply_context.get("opposite_side_lcb_edge"), default=np.nan),
                "opposite_side_decision": str(supply_context.get("opposite_side_decision", "not_evaluated")),
                "rebound_diagnostic_segment": str(supply_context.get("rebound_diagnostic_segment", "NOT_APPLICABLE")),
                "adjusted_abs_edge": effective_abs_gap,
                "history_rows": int(row.get("history_rows", 0)),
                "last_history_date": row.get("last_history_date"),
                "csv": row.get("csv"),
            }
            rows.append(row_payload)
            opposite_candidate = _build_opposite_side_candidate_row(row_payload, supply_context)
            if opposite_candidate is not None:
                rows.append(opposite_candidate)
    plays = pd.DataFrame.from_records(rows)
    if plays.empty:
        return plays
    if "candidate_id" not in plays.columns:
        plays["candidate_id"] = build_candidate_id(plays)
    plays = build_priced_event_ledger_frame(plays, record_scope="candidate")
    plays = plays.sort_values(
        ["recommendation", "expected_win_rate", "confidence_score", "abs_edge"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return plays


def recommendation_rank(label: str) -> int:
    order = {"elite": 0, "strong": 1, "consider": 2, "pass": 3}
    return order.get(label, 4)


def main() -> None:
    args = parse_args()
    slate_path = args.slate_csv.resolve()
    history_path = args.history_csv.resolve()
    if not slate_path.exists():
        raise FileNotFoundError(f"Slate CSV not found: {slate_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"History CSV not found: {history_path}")

    slate_df = pd.read_csv(slate_path)
    history_df = pd.read_csv(history_path)
    history_lookup = build_history_lookup(history_df)
    line_decision_lookup = build_line_decision_lookup(history_df)
    selector_run_time = pd.Timestamp.utcnow().isoformat()
    line_decision_cfg = (
        LineDecisionConfig(
            no_trade_threshold=float(args.line_decision_no_trade_threshold),
            min_trade_prob=float(args.line_decision_min_trade_prob),
            min_trade_prob_gap=float(args.line_decision_min_prob_gap),
        )
        if LineDecisionConfig is not None
        else None
    )
    plays = build_play_rows(
        slate_df,
        history_lookup,
        line_decision_lookup=line_decision_lookup,
        volatility_adjustment=not args.disable_volatility_adjustment,
        belief_uncertainty_lower=float(args.belief_uncertainty_lower),
        belief_uncertainty_upper=float(args.belief_uncertainty_upper),
        market_regression_floor=float(args.market_regression_floor),
        market_regression_ceiling=float(args.market_regression_ceiling),
        line_decision_enabled=not args.disable_line_decision_sidecar,
        line_decision_config=line_decision_cfg,
        selector_run_time=selector_run_time,
    )
    if plays.empty:
        raise RuntimeError("No playable rows were produced from the provided slate/history inputs.")

    plays["recommendation_rank"] = plays["recommendation"].map(recommendation_rank)
    plays = plays.sort_values(
        ["recommendation_rank", "expected_win_rate", "confidence_score", "abs_edge"],
        ascending=[True, False, False, False],
    ).drop(columns=["recommendation_rank"]).reset_index(drop=True)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    plays.to_csv(args.csv_out, index=False)

    summary = {
        "slate_csv": str(slate_path),
        "history_csv": str(history_path),
        "n_plays": int(len(plays)),
        "recommendation_counts": plays["recommendation"].value_counts().to_dict(),
        "line_decision_action_counts": plays["line_decision_action"].value_counts().to_dict() if "line_decision_action" in plays.columns else {},
        "trade_eligible_rows": int(pd.to_numeric(plays.get("line_decision_trade_eligible"), errors="coerce").fillna(0).astype(bool).sum()) if "line_decision_trade_eligible" in plays.columns else int(len(plays)),
        "top_strong": plays.loc[plays["recommendation"] == "strong"].head(10).to_dict(orient="records"),
        "top_consider": plays.loc[plays["recommendation"] == "consider"].head(10).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("UPCOMING MARKET PLAY SELECTOR")
    print("=" * 90)
    print(f"Slate:  {slate_path}")
    print(f"Rows:   {len(plays)}")
    print(f"Saved:  {args.csv_out}")
    print(f"JSON:   {args.json_out}")
    print("Recommendation counts:")
    for label, count in plays["recommendation"].value_counts().items():
        print(f"  {label}: {count}")

    show_cols = [
        "player",
        "target",
        "direction",
        "prediction",
        "market_line",
        "edge",
        "gap_percentile",
        "expected_win_rate",
        "expected_push_rate",
        "line_decision_action",
        "confidence_score",
        "recommendation",
    ]
    print("\nTop plays:")
    print(plays[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
