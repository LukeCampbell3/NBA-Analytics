from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TARGET_LINE_BINS = {
    "PTS": [-np.inf, 9.5, 14.5, 19.5, 24.5, 29.5, np.inf],
    "TRB": [-np.inf, 3.5, 5.5, 7.5, 9.5, 11.5, np.inf],
    "AST": [-np.inf, 2.5, 4.5, 6.5, 8.5, 10.5, np.inf],
}
GAP_PERCENTILE_BINS = [-np.inf, 0.50, 0.75, 0.90, 0.97, np.inf]
GAP_PERCENTILE_LABELS = ["P00_50", "P50_75", "P75_90", "P90_97", "P97P"]
SOURCE_WEIGHTS = {
    "exact": 1.00,
    "line": 0.82,
    "gap": 0.74,
    "direction": 0.60,
    "target": 0.40,
}


@dataclass
class LineDecisionConfig:
    empirical_support_scale: float = 180.0
    no_trade_threshold: float = 0.45
    min_trade_prob: float = 0.63
    min_trade_prob_gap: float = 0.06


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _clip01(value: Any, default: float = 0.0) -> float:
    return float(np.clip(_safe_float(value, default=default), 0.0, 1.0))


def _active_only_mask(df: pd.DataFrame) -> pd.Series:
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


def _target_line_band(target: str, market_line: pd.Series) -> pd.Series:
    bins = TARGET_LINE_BINS.get(str(target).upper(), TARGET_LINE_BINS["PTS"])
    labels = [f"L{idx}" for idx in range(len(bins) - 1)]
    bucket = pd.cut(pd.to_numeric(market_line, errors="coerce"), bins=bins, labels=labels, include_lowest=True)
    out = bucket.astype("object").astype(str)
    out = out.where(out.str.lower().ne("nan"), "")
    return out.fillna("").astype("object")


def _gap_percentile_band(values: pd.Series) -> pd.Series:
    bucket = pd.cut(pd.to_numeric(values, errors="coerce"), bins=GAP_PERCENTILE_BINS, labels=GAP_PERCENTILE_LABELS, include_lowest=True)
    out = bucket.astype("object").astype(str)
    out = out.where(out.str.lower().ne("nan"), "")
    return out.fillna("").astype("object")


def _cohort_payload(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    residuals = pd.to_numeric(frame.get("residual"), errors="coerce").dropna().to_numpy(dtype="float64")
    actual_minus_market = pd.to_numeric(frame.get("actual_minus_market"), errors="coerce").dropna().to_numpy(dtype="float64")
    if residuals.size <= 0 or actual_minus_market.size <= 0:
        return None
    push_mask = np.isclose(actual_minus_market, 0.0, atol=1e-9)
    return {
        "rows": int(len(frame)),
        "residuals_sorted": np.sort(residuals),
        "push_rate": float(np.mean(push_mask)) if push_mask.size > 0 else 0.0,
        "residual_std": float(np.std(residuals, ddof=0)) if residuals.size > 1 else 0.0,
    }


def _empirical_dist_for_margin(
    payload: dict[str, Any] | None,
    margin: float,
    market_line: float,
) -> tuple[float, float, float]:
    if not payload:
        return 0.5, 0.5, 0.0
    residuals_sorted = np.asarray(payload.get("residuals_sorted", []), dtype="float64")
    rows = int(payload.get("rows", 0))
    if residuals_sorted.size <= 0 or rows <= 0:
        return 0.5, 0.5, 0.0
    threshold = float(market_line - (market_line + margin))
    under_prob = float(np.searchsorted(residuals_sorted, threshold, side="left") / rows)
    over_prob = float(1.0 - (np.searchsorted(residuals_sorted, threshold, side="right") / rows))
    neutral_prob = float(np.clip(payload.get("push_rate", 0.0), 0.0, 1.0)) if np.isclose(market_line, round(market_line), atol=1e-9) else 0.0
    mass = max(1e-9, over_prob + under_prob)
    over_prob = float(np.clip(over_prob, 0.0, 1.0))
    under_prob = float(np.clip(under_prob, 0.0, 1.0))
    neutral_prob = float(np.clip(neutral_prob, 0.0, 1.0))
    available = max(0.0, 1.0 - neutral_prob)
    over_prob = float(available * over_prob / mass)
    under_prob = float(available * under_prob / mass)
    return over_prob, under_prob, neutral_prob


def _normalize_dist(over_prob: float, under_prob: float, neutral_prob: float) -> tuple[float, float, float]:
    probs = np.asarray([over_prob, under_prob, neutral_prob], dtype="float64")
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if total <= 1e-9:
        return 0.5, 0.5, 0.0
    probs = probs / total
    return float(probs[0]), float(probs[1]), float(probs[2])


def build_line_decision_lookup(history_df: pd.DataFrame) -> dict[str, Any]:
    active_history = history_df.loc[_active_only_mask(history_df)].copy()
    lookup: dict[str, Any] = {}
    for target in ("PTS", "TRB", "AST"):
        pred_col = f"pred_{target}"
        market_col = f"market_{target}"
        actual_col = f"actual_{target}"
        if pred_col not in active_history.columns or market_col not in active_history.columns or actual_col not in active_history.columns:
            continue
        frame = active_history.loc[
            pd.to_numeric(active_history[pred_col], errors="coerce").notna()
            & pd.to_numeric(active_history[market_col], errors="coerce").notna()
            & pd.to_numeric(active_history[actual_col], errors="coerce").notna()
        ].copy()
        if frame.empty:
            continue
        frame["margin"] = pd.to_numeric(frame[pred_col], errors="coerce") - pd.to_numeric(frame[market_col], errors="coerce")
        frame = frame.loc[frame["margin"].ne(0.0)].copy()
        if frame.empty:
            continue
        frame["direction"] = np.where(frame["margin"] > 0.0, "OVER", "UNDER")
        frame["abs_gap"] = frame["margin"].abs()
        gaps_sorted = np.sort(frame["abs_gap"].to_numpy(dtype="float64"))
        frame["gap_percentile"] = frame["abs_gap"].map(
            lambda value: float(np.searchsorted(gaps_sorted, float(value), side="right") / max(1, len(gaps_sorted)))
        )
        frame["gap_band"] = _gap_percentile_band(frame["gap_percentile"])
        frame["line_band"] = _target_line_band(target, frame[market_col])
        frame["residual"] = pd.to_numeric(frame[actual_col], errors="coerce") - pd.to_numeric(frame[pred_col], errors="coerce")
        frame["actual_minus_market"] = pd.to_numeric(frame[actual_col], errors="coerce") - pd.to_numeric(frame[market_col], errors="coerce")

        target_payload: dict[str, Any] = {
            "exact": {},
            "line": {},
            "gap": {},
            "direction": {},
            "target": _cohort_payload(frame),
        }
        for direction, part in frame.groupby("direction", sort=False):
            target_payload["direction"][str(direction)] = _cohort_payload(part)
            for line_band, line_part in part.groupby("line_band", sort=False):
                target_payload["line"][(str(direction), str(line_band))] = _cohort_payload(line_part)
            for gap_band, gap_part in part.groupby("gap_band", sort=False):
                target_payload["gap"][(str(direction), str(gap_band))] = _cohort_payload(gap_part)
            for (line_band, gap_band), exact_part in part.groupby(["line_band", "gap_band"], sort=False):
                target_payload["exact"][(str(direction), str(line_band), str(gap_band))] = _cohort_payload(exact_part)
        lookup[target] = target_payload
    return lookup


def estimate_line_decision(
    *,
    lookup: dict[str, Any],
    target: str,
    prediction: float,
    market_line: float,
    direction: str,
    gap_percentile: float,
    uncertainty_sigma: float,
    belief_confidence_factor: float,
    feasibility: float,
    history_rows: float,
    market_books: float,
    fallback_blend: float,
    prior_direction_win_rate: float,
    prior_neutral_rate: float,
    config: LineDecisionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LineDecisionConfig()
    target_key = str(target).upper().strip()
    direction_key = str(direction).upper().strip()
    margin = _safe_float(prediction, default=np.nan) - _safe_float(market_line, default=np.nan)
    if not np.isfinite(margin) or direction_key not in {"OVER", "UNDER"}:
        return {
            "over_prob": 0.5,
            "under_prob": 0.5,
            "no_trade_prob": 0.0,
            "chosen_direction_prob": 0.5,
            "opposite_direction_prob": 0.5,
            "chosen_direction_conditional_prob": 0.5,
            "opposite_direction_conditional_prob": 0.5,
            "conditional_prob_gap": 0.0,
            "trade_prob_floor": float(cfg.min_trade_prob),
            "trade_eligible": False,
            "action": "NO_TRADE",
            "source": "unavailable",
            "support_rows": 0.0,
            "support_strength": 0.0,
            "sigma_pressure": 1.0,
            "instability_score": 1.0,
            "fragility_score": 1.0,
            "empirical_blend_weight": 0.0,
        }

    target_payload = lookup.get(target_key, {})
    line_band = str(_target_line_band(target_key, pd.Series([market_line])).iloc[0])
    gap_band = str(_gap_percentile_band(pd.Series([gap_percentile])).iloc[0])

    candidates: list[tuple[str, dict[str, Any] | None, float]] = [
        ("exact", target_payload.get("exact", {}).get((direction_key, line_band, gap_band)), SOURCE_WEIGHTS["exact"]),
        ("line", target_payload.get("line", {}).get((direction_key, line_band)), SOURCE_WEIGHTS["line"]),
        ("gap", target_payload.get("gap", {}).get((direction_key, gap_band)), SOURCE_WEIGHTS["gap"]),
        ("direction", target_payload.get("direction", {}).get(direction_key), SOURCE_WEIGHTS["direction"]),
        ("target", target_payload.get("target"), SOURCE_WEIGHTS["target"]),
    ]

    empirical_over = 0.0
    empirical_under = 0.0
    empirical_neutral = 0.0
    empirical_weight_total = 0.0
    support_rows = 0.0
    source_tokens: list[str] = []
    for name, payload, source_weight in candidates:
        if not payload:
            continue
        rows = int(payload.get("rows", 0))
        if rows <= 0:
            continue
        over_prob, under_prob, neutral_prob = _empirical_dist_for_margin(payload, margin=margin, market_line=float(market_line))
        cohort_weight = float(max(1.0, np.sqrt(rows)) * source_weight)
        empirical_over += cohort_weight * over_prob
        empirical_under += cohort_weight * under_prob
        empirical_neutral += cohort_weight * neutral_prob
        empirical_weight_total += cohort_weight
        support_rows += float(rows * source_weight)
        source_tokens.append(f"{name}:{rows}")

    if empirical_weight_total <= 1e-9:
        empirical_over, empirical_under, empirical_neutral = 0.5, 0.5, 0.0
        support_rows = 0.0
        source_tokens = ["prior_only"]
    else:
        empirical_over /= empirical_weight_total
        empirical_under /= empirical_weight_total
        empirical_neutral /= empirical_weight_total
        empirical_over, empirical_under, empirical_neutral = _normalize_dist(empirical_over, empirical_under, empirical_neutral)

    prior_direction = float(np.clip(prior_direction_win_rate, 0.0, 1.0))
    prior_neutral = float(np.clip(prior_neutral_rate, 0.0, 1.0))
    prior_opposite = float(np.clip(1.0 - prior_direction - prior_neutral, 0.0, 1.0))
    if direction_key == "OVER":
        prior_over, prior_under = prior_direction, prior_opposite
    else:
        prior_over, prior_under = prior_opposite, prior_direction
    prior_over, prior_under, prior_neutral = _normalize_dist(prior_over, prior_under, prior_neutral)

    support_scale = max(1.0, float(cfg.empirical_support_scale))
    empirical_blend_weight = float(np.clip(np.log1p(max(0.0, support_rows)) / np.log1p(support_scale), 0.22, 0.78))
    base_over = empirical_blend_weight * empirical_over + (1.0 - empirical_blend_weight) * prior_over
    base_under = empirical_blend_weight * empirical_under + (1.0 - empirical_blend_weight) * prior_under
    base_neutral = empirical_blend_weight * empirical_neutral + (1.0 - empirical_blend_weight) * prior_neutral
    base_over, base_under, base_neutral = _normalize_dist(base_over, base_under, base_neutral)

    sigma_value = max(0.0, _safe_float(uncertainty_sigma, default=0.0))
    sigma_pressure = float(sigma_value / max(sigma_value + abs(margin), 1e-9))
    fragile_near_line = float(np.exp(-abs(margin) / max(0.65, 0.35 + 0.35 * sigma_value)))
    support_strength = float(
        np.clip(
            0.45 * np.clip(np.log1p(max(0.0, _safe_float(history_rows, default=0.0))) / np.log1p(120.0), 0.0, 1.0)
            + 0.20 * np.clip(np.log1p(max(0.0, _safe_float(market_books, default=0.0))) / np.log1p(8.0), 0.0, 1.0)
            + 0.35 * np.clip(np.log1p(max(0.0, support_rows)) / np.log1p(240.0), 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    belief_conf = _clip01(belief_confidence_factor, default=0.5)
    feasibility_value = _clip01(feasibility, default=0.5)
    fallback_value = _clip01(fallback_blend, default=0.0)
    instability_score = float(
        np.clip(
            0.50 * sigma_pressure
            + 0.15 * (1.0 - belief_conf)
            + 0.10 * (1.0 - feasibility_value)
            + 0.10 * fallback_value
            + 0.15 * (1.0 - support_strength),
            0.0,
            1.0,
        )
    )
    directional_mass = max(1e-9, 1.0 - base_neutral)
    ambiguity = float(np.clip(1.0 - (abs(base_over - base_under) / directional_mass), 0.0, 1.0))
    fragility_score = float(np.clip(0.55 * fragile_near_line + 0.45 * ambiguity, 0.0, 1.0))
    fragility_lift = float(
        np.clip(
            (1.0 - base_neutral)
            * (
                0.70 * fragility_score * instability_score
                + 0.25 * fragile_near_line * sigma_pressure
            ),
            0.0,
            0.65,
        )
    )
    neutral_prob = float(np.clip(base_neutral + fragility_lift, 0.0, 0.90))
    remaining_mass = max(1e-9, 1.0 - neutral_prob)
    directional_base_mass = max(1e-9, base_over + base_under)
    over_prob = float(remaining_mass * base_over / directional_base_mass)
    under_prob = float(remaining_mass * base_under / directional_base_mass)
    over_prob, under_prob, neutral_prob = _normalize_dist(over_prob, under_prob, neutral_prob)

    chosen_direction_prob = over_prob if direction_key == "OVER" else under_prob
    opposite_direction_prob = under_prob if direction_key == "OVER" else over_prob
    directional_mass_post_abstain = max(1e-9, chosen_direction_prob + opposite_direction_prob)
    chosen_direction_conditional_prob = float(chosen_direction_prob / directional_mass_post_abstain)
    opposite_direction_conditional_prob = float(opposite_direction_prob / directional_mass_post_abstain)
    effective_trade_prob_floor = float(np.clip(float(cfg.min_trade_prob), 0.50, 0.90))
    conditional_prob_gap = float(chosen_direction_conditional_prob - opposite_direction_conditional_prob)
    trade_eligible = bool(
        neutral_prob < float(cfg.no_trade_threshold)
        and chosen_direction_conditional_prob >= effective_trade_prob_floor
        and conditional_prob_gap >= float(cfg.min_trade_prob_gap)
    )
    action = direction_key if trade_eligible else "NO_TRADE"
    return {
        "over_prob": float(over_prob),
        "under_prob": float(under_prob),
        "no_trade_prob": float(neutral_prob),
        "chosen_direction_prob": float(chosen_direction_prob),
        "opposite_direction_prob": float(opposite_direction_prob),
        "chosen_direction_conditional_prob": float(chosen_direction_conditional_prob),
        "opposite_direction_conditional_prob": float(opposite_direction_conditional_prob),
        "conditional_prob_gap": float(conditional_prob_gap),
        "trade_prob_floor": float(effective_trade_prob_floor),
        "trade_eligible": bool(trade_eligible),
        "action": str(action),
        "source": "+".join(source_tokens),
        "support_rows": float(support_rows),
        "support_strength": float(support_strength),
        "sigma_pressure": float(sigma_pressure),
        "instability_score": float(instability_score),
        "fragility_score": float(fragility_score),
        "empirical_blend_weight": float(empirical_blend_weight),
    }
