"""Precision enhancements for prediction accuracy.

Four improvements applied as post-processing adjustments to the candidate pool:

1. RECENCY-WEIGHTED CALIBRATION — Applies exponential decay to historical
   bucket priors so recent performance matters more than early-season data.

2. MINUTES/ROLE INSTABILITY DETECTION — Penalizes picks where the player's
   feasibility or role has recently degraded, indicating minutes uncertainty.

3. MARKET-IMPLIED PROBABILITY BLEND — Blends the model's win probability
   with the market-implied probability from sportsbook prices to improve
   calibration when model and market agree.

4. LINE MOVEMENT / MARKET STRENGTH SIGNAL — Boosts confidence when multiple
   books agree on a line (high market_books) and penalizes thin markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PrecisionEnhancementConfig:
    """Tunable parameters for all four enhancements."""

    enabled: bool = True

    # --- 1. Recency-weighted calibration ---
    recency_enabled: bool = True
    recency_half_life_days: float = 18.0
    recency_min_factor: float = 0.3
    recency_boost_recent_winners: float = 0.015
    recency_penalty_recent_losers: float = 0.010

    # --- 2. Minutes/role instability ---
    instability_enabled: bool = True
    instability_feasibility_threshold: float = 0.55
    instability_role_shift_threshold: float = 0.40
    instability_penalty_low_feasibility: float = 0.035
    instability_penalty_high_role_shift: float = 0.025
    instability_penalty_combined: float = 0.050
    instability_fallback_blend_threshold: float = 0.25
    instability_fallback_penalty: float = 0.020

    # --- 3. Market-implied probability blend ---
    market_implied_enabled: bool = True
    market_implied_min_books: int = 3
    market_implied_blend_weight: float = 0.20
    market_implied_agreement_bonus: float = 0.015
    market_implied_disagreement_penalty: float = 0.020
    market_implied_agreement_threshold: float = 0.04

    # --- 4. Market strength / line movement ---
    market_strength_enabled: bool = True
    market_strength_high_books_threshold: int = 5
    market_strength_high_books_bonus: float = 0.010
    market_strength_low_books_threshold: int = 2
    market_strength_low_books_penalty: float = 0.015
    market_strength_low_std_bonus: float = 0.008
    market_strength_low_std_threshold: float = 0.15

    # --- Caps ---
    max_total_adjustment: float = 0.06
    min_win_rate_floor: float = 0.50


def _sf(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _american_to_implied(price: float) -> float:
    """Convert American odds to implied probability."""
    if price == 0 or not np.isfinite(price):
        return 0.5
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    else:
        return 100.0 / (price + 100.0)


def _no_vig_probability(over_price: float, under_price: float, direction: str) -> float:
    """Calculate no-vig probability for the selected side."""
    over_implied = _american_to_implied(over_price)
    under_implied = _american_to_implied(under_price)
    total = over_implied + under_implied
    if total <= 0:
        return 0.5
    if direction.upper() == "OVER":
        return over_implied / total
    else:
        return under_implied / total


def compute_enhancements(
    *,
    expected_win_rate: float,
    direction: str,
    feasibility: float = 0.7,
    role_shift_risk: float = 0.2,
    fallback_blend: float = 0.0,
    market_books: float = 0.0,
    market_over_price: float = 0.0,
    market_under_price: float = 0.0,
    market_line_std: float = 0.0,
    recency_factor: float = 1.0,
    spike_probability: float = 0.5,
    history_rows: int = 50,
    config: PrecisionEnhancementConfig | None = None,
) -> dict[str, Any]:
    """Compute all four precision enhancements for a single candidate.

    Returns a dict with:
      - total_adjustment: float to add to expected_win_rate
      - recency_adj: float
      - instability_adj: float
      - market_implied_adj: float
      - market_strength_adj: float
      - market_implied_prob: float or None
      - sources: list of strings describing which adjustments fired
    """
    cfg = config or PrecisionEnhancementConfig()
    if not cfg.enabled:
        return {
            "total_adjustment": 0.0,
            "recency_adj": 0.0,
            "instability_adj": 0.0,
            "market_implied_adj": 0.0,
            "market_strength_adj": 0.0,
            "market_implied_prob": None,
            "sources": ["disabled"],
        }

    sources: list[str] = []
    recency_adj = 0.0
    instability_adj = 0.0
    market_implied_adj = 0.0
    market_strength_adj = 0.0
    market_implied_prob = None

    wr = float(np.clip(_sf(expected_win_rate, 0.5), 0.0, 1.0))
    feas = float(np.clip(_sf(feasibility, 0.7), 0.0, 1.0))
    role_risk = float(np.clip(_sf(role_shift_risk, 0.2), 0.0, 1.0))
    fb_blend = float(np.clip(_sf(fallback_blend, 0.0), 0.0, 1.0))
    books = int(max(0, _sf(market_books, 0)))
    over_price = _sf(market_over_price, 0.0)
    under_price = _sf(market_under_price, 0.0)
    line_std = float(max(0.0, _sf(market_line_std, 0.0)))
    rec_factor = float(np.clip(_sf(recency_factor, 1.0), 0.0, 1.0))
    hrows = int(max(0, _sf(history_rows, 0)))
    dir_upper = str(direction).upper().strip()

    # ─── 1. Recency-weighted calibration ───
    if cfg.recency_enabled:
        # Players with high recency factor (recent data) and good win rate
        # get a small boost; stale data gets a penalty
        if rec_factor >= 0.8 and wr >= 0.54:
            recency_adj = cfg.recency_boost_recent_winners * rec_factor
            sources.append("recency_boost")
        elif rec_factor < cfg.recency_min_factor:
            recency_adj = -cfg.recency_penalty_recent_losers * (1.0 - rec_factor)
            sources.append("recency_stale_penalty")

    # ─── 2. Minutes/role instability ───
    if cfg.instability_enabled:
        low_feas = feas < cfg.instability_feasibility_threshold
        high_role_shift = role_risk > cfg.instability_role_shift_threshold
        high_fallback = fb_blend > cfg.instability_fallback_blend_threshold

        # Key insight: low feasibility HELPS UNDERs (player plays less = lower stats)
        # but HURTS OVERs (player needs minutes to produce).
        if dir_upper == "OVER":
            if low_feas and high_role_shift:
                instability_adj = -cfg.instability_penalty_combined
                sources.append("instability_combined")
            elif low_feas:
                instability_adj = -cfg.instability_penalty_low_feasibility
                sources.append("instability_low_feas")
            elif high_role_shift:
                instability_adj = -cfg.instability_penalty_high_role_shift
                sources.append("instability_role_shift")
            if high_fallback:
                instability_adj -= cfg.instability_fallback_penalty
                sources.append("instability_fallback")
        else:
            # For UNDERs: low feasibility is actually a POSITIVE signal
            # (less minutes = lower stats = UNDER hits more often)
            # Only penalize if role_shift is high AND feasibility is very low
            # (suggests the player might not play at all → push/void risk)
            if feas < 0.35 and high_role_shift:
                instability_adj = -cfg.instability_penalty_high_role_shift * 0.5
                sources.append("instability_dnp_risk")
            elif low_feas:
                # Small BOOST for UNDERs with low feasibility
                instability_adj = 0.008
                sources.append("instability_under_boost")

    # ─── 3. Market-implied probability blend ───
    if cfg.market_implied_enabled and books >= cfg.market_implied_min_books:
        if abs(over_price) >= 100 and abs(under_price) >= 100:
            mkt_prob = _no_vig_probability(over_price, under_price, dir_upper)
            market_implied_prob = mkt_prob

            # Blend: if market agrees with model, boost; if disagrees, penalize
            model_edge = wr - 0.5
            market_edge = mkt_prob - 0.5
            agreement = model_edge * market_edge  # positive = same direction

            if agreement > 0:
                # Model and market agree on direction
                agreement_strength = min(abs(model_edge), abs(market_edge))
                if agreement_strength >= cfg.market_implied_agreement_threshold:
                    market_implied_adj = cfg.market_implied_agreement_bonus
                    sources.append("market_agrees")
                # Also blend the probability slightly toward market
                blend_adj = cfg.market_implied_blend_weight * (mkt_prob - wr)
                market_implied_adj += float(np.clip(blend_adj, -0.02, 0.02))
            elif agreement < 0 and abs(market_edge) > cfg.market_implied_agreement_threshold:
                # Market disagrees with model direction
                market_implied_adj = -cfg.market_implied_disagreement_penalty
                sources.append("market_disagrees")

    # ─── 4. Market strength / line movement ───
    if cfg.market_strength_enabled:
        if books >= cfg.market_strength_high_books_threshold:
            market_strength_adj = cfg.market_strength_high_books_bonus
            sources.append("strong_market")
            # Additional bonus for tight consensus (low std)
            if line_std > 0 and line_std <= cfg.market_strength_low_std_threshold:
                market_strength_adj += cfg.market_strength_low_std_bonus
                sources.append("tight_consensus")
        elif books > 0 and books <= cfg.market_strength_low_books_threshold:
            market_strength_adj = -cfg.market_strength_low_books_penalty
            sources.append("thin_market")

    # ─── Clamp total ───
    total = recency_adj + instability_adj + market_implied_adj + market_strength_adj
    total = float(np.clip(total, -cfg.max_total_adjustment, cfg.max_total_adjustment))

    return {
        "total_adjustment": total,
        "recency_adj": recency_adj,
        "instability_adj": instability_adj,
        "market_implied_adj": market_implied_adj,
        "market_strength_adj": market_strength_adj,
        "market_implied_prob": market_implied_prob,
        "sources": sources if sources else ["none"],
    }


def annotate_precision_enhancements(
    candidates: pd.DataFrame,
    *,
    config: PrecisionEnhancementConfig | None = None,
) -> pd.DataFrame:
    """Annotate a candidate DataFrame with precision enhancement columns.

    Works with both NBA selector format and MLB high-precision format.
    Adds columns:
      - precision_enhancement_adj
      - precision_enhanced_win_rate
      - precision_recency_adj
      - precision_instability_adj
      - precision_market_implied_adj
      - precision_market_strength_adj
      - precision_market_implied_prob
      - precision_enhancement_sources
    """
    cfg = config or PrecisionEnhancementConfig()
    out = candidates.copy()

    if out.empty or not cfg.enabled:
        for col in [
            "precision_enhancement_adj", "precision_enhanced_win_rate",
            "precision_recency_adj", "precision_instability_adj",
            "precision_market_implied_adj", "precision_market_strength_adj",
            "precision_market_implied_prob", "precision_enhancement_sources",
        ]:
            out[col] = 0.0 if "sources" not in col else "disabled"
        return out

    # Resolve column names (NBA vs MLB format)
    def _col(row, *names, default=0.0):
        for name in names:
            val = row.get(name)
            if val is not None and str(val).strip().lower() not in ("", "nan", "none"):
                return _sf(val, default)
        return default

    adjustments = []
    for _, row in out.iterrows():
        result = compute_enhancements(
            expected_win_rate=_col(row, "expected_win_rate", "Estimated_Hit_Probability", "calibrated_hit_probability", default=0.5),
            direction=str(row.get("direction", row.get("Direction", "UNDER"))),
            feasibility=_col(row, "feasibility", default=0.7),
            role_shift_risk=_col(row, "role_shift_risk", default=0.2),
            fallback_blend=_col(row, "fallback_blend", default=0.0),
            market_books=_col(row, "market_books", "Market_Books", default=0),
            market_over_price=_col(row, "market_over_price", "Market_Over_Price", default=0),
            market_under_price=_col(row, "market_under_price", "Market_Under_Price", default=0),
            market_line_std=_col(row, "market_line_std", "Market_Line_Std", default=0),
            recency_factor=_col(row, "recency_factor", default=1.0),
            spike_probability=_col(row, "spike_probability", default=0.5),
            history_rows=int(_col(row, "history_rows", "History_Rows", default=50)),
            config=cfg,
        )
        adjustments.append(result)

    out["precision_enhancement_adj"] = [r["total_adjustment"] for r in adjustments]
    out["precision_recency_adj"] = [r["recency_adj"] for r in adjustments]
    out["precision_instability_adj"] = [r["instability_adj"] for r in adjustments]
    out["precision_market_implied_adj"] = [r["market_implied_adj"] for r in adjustments]
    out["precision_market_strength_adj"] = [r["market_strength_adj"] for r in adjustments]
    out["precision_market_implied_prob"] = [r["market_implied_prob"] for r in adjustments]
    out["precision_enhancement_sources"] = ["+".join(r["sources"]) for r in adjustments]

    # Compute enhanced win rate
    base_wr = pd.to_numeric(
        out.get("expected_win_rate", out.get("Estimated_Hit_Probability", pd.Series(0.5, index=out.index))),
        errors="coerce",
    ).fillna(0.5)
    out["precision_enhanced_win_rate"] = (
        base_wr + pd.to_numeric(out["precision_enhancement_adj"], errors="coerce").fillna(0.0)
    ).clip(lower=cfg.min_win_rate_floor, upper=0.98)

    return out
