"""Direction-aware win-probability classifier.

This module provides two capabilities:

1. **Market mispricing detection** — identifies when a market line appears
   mispriced relative to the player's recent performance and model prediction,
   creating opportunities for high-confidence OVER or UNDER picks.

2. **Direction-specific win-rate adjustment** — applies empirically-calibrated
   adjustments to expected_win_rate based on the direction (OVER vs UNDER) and
   features that historically predict direction-specific outcomes.

Historical analysis (20260406-20260430, 6096 rows) shows:
  - UNDER overall: 66.7% win rate
  - OVER overall:  49.6% win rate
  - TRB OVER edge>=1.0: 67.0% (282 rows)
  - AST OVER edge>=1.0: 63.7% (91 rows)
  - PTS OVER edge>=1.0: 55.7% (783 rows)
  - OVER edge/sigma >= 0.40: 56.1% (1412 rows)

The classifier does NOT blindly boost OVERs.  It identifies the specific
conditions under which OVERs are profitable and adjusts probabilities
accordingly, while also boosting confidence in high-edge UNDERs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DirectionClassifierConfig:
    """Tunable parameters for the direction classifier."""

    enabled: bool = True

    # --- Market mispricing detection ---
    # A line is considered potentially mispriced when the baseline (player's
    # rolling average) diverges significantly from the market line.
    mispricing_baseline_edge_min_pts: float = 2.0
    mispricing_baseline_edge_min_trb: float = 1.5
    mispricing_baseline_edge_min_ast: float = 1.0
    mispricing_max_spike_probability: float = 0.65
    mispricing_min_feasibility: float = 0.55
    mispricing_min_belief_conf: float = 0.45
    mispricing_min_history_rows: int = 20
    mispricing_over_boost: float = 0.06   # win-rate lift for mispriced OVERs
    mispricing_under_boost: float = 0.04  # win-rate lift for mispriced UNDERs

    # --- Direction-specific edge thresholds ---
    # Minimum abs_edge for an OVER to be considered "high-edge" by target.
    over_high_edge_pts: float = 1.5
    over_high_edge_trb: float = 1.0
    over_high_edge_ast: float = 1.0

    # Win-rate adjustments for high-edge OVERs by target.
    # These are calibrated from historical data.
    over_high_edge_boost_pts: float = 0.00   # PTS OVERs are marginal — no boost, rely on edge alone
    over_high_edge_boost_trb: float = 0.06   # TRB OVERs are strong at high edge
    over_high_edge_boost_ast: float = 0.05   # AST OVERs are strong at high edge

    # Edge/sigma ratio threshold — OVERs with high edge relative to
    # uncertainty are more reliable.
    over_edge_sigma_threshold: float = 0.35
    over_edge_sigma_boost: float = 0.025

    # --- UNDER enhancements ---
    # High-edge UNDERs are already strong; give them a small additional boost
    # to increase confidence in the precision pool.
    under_high_edge_boost_pts: float = 0.03
    under_high_edge_boost_trb: float = 0.04
    under_high_edge_boost_ast: float = 0.05
    under_high_edge_min: float = 1.5

    # --- Low-edge OVER penalty ---
    # OVERs with small edges are historically losing propositions (~42%).
    # Apply a penalty to prevent them from sneaking onto the board.
    over_low_edge_penalty: float = 0.06
    over_low_edge_threshold: float = 0.75

    # --- Caps ---
    max_total_adjustment: float = 0.08
    min_win_rate_floor: float = 0.50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _mispricing_edge_threshold(target: str, config: DirectionClassifierConfig) -> float:
    """Return the minimum baseline-vs-line edge for mispricing detection."""
    target_upper = str(target).upper().strip()
    if target_upper == "PTS":
        return float(config.mispricing_baseline_edge_min_pts)
    elif target_upper == "TRB":
        return float(config.mispricing_baseline_edge_min_trb)
    elif target_upper == "AST":
        return float(config.mispricing_baseline_edge_min_ast)
    return 2.0


def _over_high_edge_threshold(target: str, config: DirectionClassifierConfig) -> float:
    target_upper = str(target).upper().strip()
    if target_upper == "PTS":
        return float(config.over_high_edge_pts)
    elif target_upper == "TRB":
        return float(config.over_high_edge_trb)
    elif target_upper == "AST":
        return float(config.over_high_edge_ast)
    return 1.5


def _over_high_edge_boost(target: str, config: DirectionClassifierConfig) -> float:
    target_upper = str(target).upper().strip()
    if target_upper == "PTS":
        return float(config.over_high_edge_boost_pts)
    elif target_upper == "TRB":
        return float(config.over_high_edge_boost_trb)
    elif target_upper == "AST":
        return float(config.over_high_edge_boost_ast)
    return 0.02


def _under_high_edge_boost(target: str, config: DirectionClassifierConfig) -> float:
    target_upper = str(target).upper().strip()
    if target_upper == "PTS":
        return float(config.under_high_edge_boost_pts)
    elif target_upper == "TRB":
        return float(config.under_high_edge_boost_trb)
    elif target_upper == "AST":
        return float(config.under_high_edge_boost_ast)
    return 0.03


# ---------------------------------------------------------------------------
# Core: per-row classification
# ---------------------------------------------------------------------------

def classify_direction_quality(
    *,
    target: str,
    direction: str,
    abs_edge: float,
    raw_edge: float,
    baseline: float,
    market_line: float,
    spike_probability: float,
    uncertainty_sigma: float,
    belief_conf: float,
    feasibility: float,
    history_rows: int,
    expected_win_rate: float,
    config: DirectionClassifierConfig | None = None,
) -> dict[str, Any]:
    """Classify a single candidate and return direction-aware adjustments.

    Returns a dict with:
      - direction_adjustment: float added to expected_win_rate
      - mispricing_detected: bool
      - mispricing_direction: str ("OVER", "UNDER", or "")
      - mispricing_strength: float 0-1
      - over_quality_tier: str ("high_edge", "marginal", "low_edge", "n/a")
      - classification_source: str describing which rules fired
    """
    cfg = config or DirectionClassifierConfig()
    if not cfg.enabled:
        return {
            "direction_adjustment": 0.0,
            "mispricing_detected": False,
            "mispricing_direction": "",
            "mispricing_strength": 0.0,
            "over_quality_tier": "disabled",
            "classification_source": "disabled",
        }

    target_upper = str(target).upper().strip()
    direction_upper = str(direction).upper().strip()
    abs_edge_val = max(0.0, _safe_float(abs_edge))
    baseline_val = _safe_float(baseline)
    market_val = _safe_float(market_line)
    spike = float(np.clip(_safe_float(spike_probability, 0.5), 0.0, 1.0))
    sigma = max(0.01, _safe_float(uncertainty_sigma, 1.0))
    bconf = float(np.clip(_safe_float(belief_conf, 0.5), 0.0, 1.0))
    feas_val = float(np.clip(_safe_float(feasibility, 0.5), 0.0, 1.0))
    hrows = max(0, int(_safe_float(history_rows, 0)))
    ewr = float(np.clip(_safe_float(expected_win_rate, 0.5), 0.0, 1.0))

    adjustment = 0.0
    sources: list[str] = []
    mispricing_detected = False
    mispricing_direction = ""
    mispricing_strength = 0.0
    over_quality_tier = "n/a"

    # ---------------------------------------------------------------
    # 1. Market mispricing detection
    # ---------------------------------------------------------------
    if baseline_val > 0.0 and market_val > 0.0:
        baseline_edge = baseline_val - market_val  # positive = baseline above line
        mispricing_threshold = _mispricing_edge_threshold(target_upper, cfg)

        baseline_mispriced_over = bool(
            baseline_edge >= mispricing_threshold
            and spike <= cfg.mispricing_max_spike_probability
            and feas_val >= cfg.mispricing_min_feasibility
            and bconf >= cfg.mispricing_min_belief_conf
            and hrows >= cfg.mispricing_min_history_rows
        )
        baseline_mispriced_under = bool(
            baseline_edge <= -mispricing_threshold
            and feas_val >= cfg.mispricing_min_feasibility
            and bconf >= cfg.mispricing_min_belief_conf
            and hrows >= cfg.mispricing_min_history_rows
        )

        if baseline_mispriced_over and direction_upper == "OVER":
            # Market line is set too low relative to the player's baseline.
            # The model and baseline both agree the player should go OVER.
            strength = float(np.clip(
                (baseline_edge - mispricing_threshold) / max(mispricing_threshold, 1.0),
                0.0, 1.0,
            ))
            mispricing_detected = True
            mispricing_direction = "OVER"
            mispricing_strength = strength
            boost = cfg.mispricing_over_boost * (0.5 + 0.5 * strength)
            adjustment += boost
            sources.append("mispricing_over")

        elif baseline_mispriced_under and direction_upper == "UNDER":
            strength = float(np.clip(
                (abs(baseline_edge) - mispricing_threshold) / max(mispricing_threshold, 1.0),
                0.0, 1.0,
            ))
            mispricing_detected = True
            mispricing_direction = "UNDER"
            mispricing_strength = strength
            boost = cfg.mispricing_under_boost * (0.5 + 0.5 * strength)
            adjustment += boost
            sources.append("mispricing_under")

    # ---------------------------------------------------------------
    # 2. Direction-specific edge quality
    # ---------------------------------------------------------------
    if direction_upper == "OVER":
        high_edge_threshold = _over_high_edge_threshold(target_upper, cfg)

        if abs_edge_val >= high_edge_threshold:
            over_quality_tier = "high_edge"
            boost = _over_high_edge_boost(target_upper, cfg)
            # Scale boost by edge magnitude and confidence
            edge_scale = float(np.clip(
                (abs_edge_val - high_edge_threshold) / max(high_edge_threshold, 1.0),
                0.0, 1.0,
            ))
            confidence_scale = float(np.clip(bconf * feas_val, 0.3, 1.0))
            adjustment += boost * (0.6 + 0.4 * edge_scale) * confidence_scale
            sources.append(f"over_high_edge_{target_upper.lower()}")

            # Edge/sigma bonus: high edge relative to uncertainty is more reliable
            edge_sigma_ratio = abs_edge_val / sigma
            if edge_sigma_ratio >= cfg.over_edge_sigma_threshold:
                sigma_scale = float(np.clip(
                    (edge_sigma_ratio - cfg.over_edge_sigma_threshold) / 0.30,
                    0.0, 1.0,
                ))
                adjustment += cfg.over_edge_sigma_boost * sigma_scale
                sources.append("over_edge_sigma")

        elif abs_edge_val < cfg.over_low_edge_threshold:
            over_quality_tier = "low_edge"
            # Low-edge OVERs are historically losing (~42%).  Penalize.
            adjustment -= cfg.over_low_edge_penalty
            sources.append("over_low_edge_penalty")

        else:
            over_quality_tier = "marginal"

    elif direction_upper == "UNDER":
        if abs_edge_val >= cfg.under_high_edge_min:
            boost = _under_high_edge_boost(target_upper, cfg)
            edge_scale = float(np.clip(
                (abs_edge_val - cfg.under_high_edge_min) / max(cfg.under_high_edge_min, 1.0),
                0.0, 1.0,
            ))
            adjustment += boost * (0.5 + 0.5 * edge_scale)
            sources.append(f"under_high_edge_{target_upper.lower()}")

    # ---------------------------------------------------------------
    # 3. Clamp total adjustment
    # ---------------------------------------------------------------
    adjustment = float(np.clip(adjustment, -cfg.max_total_adjustment, cfg.max_total_adjustment))

    return {
        "direction_adjustment": adjustment,
        "mispricing_detected": mispricing_detected,
        "mispricing_direction": mispricing_direction,
        "mispricing_strength": mispricing_strength,
        "over_quality_tier": over_quality_tier,
        "classification_source": "+".join(sources) if sources else "none",
    }


# ---------------------------------------------------------------------------
# Batch: annotate a DataFrame of candidates
# ---------------------------------------------------------------------------

def annotate_direction_quality(
    candidates: pd.DataFrame,
    *,
    config: DirectionClassifierConfig | None = None,
) -> pd.DataFrame:
    """Annotate a candidate DataFrame with direction-quality columns.

    Adds columns:
      - direction_adjustment
      - direction_adjusted_win_rate
      - mispricing_detected
      - mispricing_direction
      - mispricing_strength
      - over_quality_tier
      - direction_classification_source
    """
    cfg = config or DirectionClassifierConfig()
    out = candidates.copy()

    if out.empty or not cfg.enabled:
        out["direction_adjustment"] = 0.0
        out["direction_adjusted_win_rate"] = pd.to_numeric(
            out.get("expected_win_rate"), errors="coerce"
        ).fillna(0.5)
        out["mispricing_detected"] = False
        out["mispricing_direction"] = ""
        out["mispricing_strength"] = 0.0
        out["over_quality_tier"] = "disabled" if not cfg.enabled else "n/a"
        out["direction_classification_source"] = "disabled" if not cfg.enabled else "empty"
        return out

    adjustments: list[float] = []
    mispricing_flags: list[bool] = []
    mispricing_dirs: list[str] = []
    mispricing_strengths: list[float] = []
    over_tiers: list[str] = []
    sources_list: list[str] = []

    for _, row in out.iterrows():
        result = classify_direction_quality(
            target=str(row.get("target", "")),
            direction=str(row.get("direction", "")),
            abs_edge=_safe_float(row.get("abs_edge")),
            raw_edge=_safe_float(row.get("raw_edge", row.get("edge", 0.0))),
            baseline=_safe_float(row.get("baseline")),
            market_line=_safe_float(row.get("market_line")),
            spike_probability=_safe_float(row.get("spike_probability", 0.5)),
            uncertainty_sigma=_safe_float(row.get("uncertainty_sigma", 1.0)),
            belief_conf=_safe_float(row.get("belief_confidence_factor", 0.5)),
            feasibility=_safe_float(row.get("feasibility", 0.5)),
            history_rows=int(_safe_float(row.get("history_rows", 0))),
            expected_win_rate=_safe_float(row.get("expected_win_rate", 0.5)),
            config=cfg,
        )
        adjustments.append(result["direction_adjustment"])
        mispricing_flags.append(result["mispricing_detected"])
        mispricing_dirs.append(result["mispricing_direction"])
        mispricing_strengths.append(result["mispricing_strength"])
        over_tiers.append(result["over_quality_tier"])
        sources_list.append(result["classification_source"])

    out["direction_adjustment"] = pd.Series(adjustments, index=out.index, dtype="float64")
    out["mispricing_detected"] = pd.Series(mispricing_flags, index=out.index, dtype="bool")
    out["mispricing_direction"] = pd.Series(mispricing_dirs, index=out.index, dtype="object")
    out["mispricing_strength"] = pd.Series(mispricing_strengths, index=out.index, dtype="float64")
    out["over_quality_tier"] = pd.Series(over_tiers, index=out.index, dtype="object")
    out["direction_classification_source"] = pd.Series(sources_list, index=out.index, dtype="object")

    base_win_rate = pd.to_numeric(out.get("expected_win_rate"), errors="coerce").fillna(0.5)
    out["direction_adjusted_win_rate"] = (base_win_rate + out["direction_adjustment"]).clip(
        lower=cfg.min_win_rate_floor, upper=0.95,
    )

    return out
