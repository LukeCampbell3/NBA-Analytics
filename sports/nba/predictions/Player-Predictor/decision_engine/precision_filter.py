"""Precision filter — gates picks for maximum hit rate.

Empirical analysis (6,096 graded picks, 20260406-20260430) reveals that
the edge-to-sigma ratio is the single strongest predictor of win probability,
stronger than raw edge or model-estimated win rate.

Key findings:
  - e/s >= 0.30 + UNDER = 70.6% (807 picks, +48% ROI)
  - e/s >= 0.30 + spike <= 0.55 = 78.0% (519 picks, +49% ROI)
  - e/s >= 0.40 + UNDER = 76.0% (all in that bucket)
  - e/s >= 0.70 + UNDER = 86.7% (elite tier)

This module computes a precision_score for each pick and assigns a
confidence tier (elite/strong/consider/pass) based on empirically-validated
thresholds.  The parlay builder uses these tiers to select legs.

The filter also identifies "danger zones" — combinations that historically
lose money — and flags them for exclusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PrecisionFilterConfig:
    """Empirically-calibrated thresholds."""
    enabled: bool = True

    # Edge-to-sigma ratio thresholds (the strongest signal)
    elite_edge_sigma: float = 0.50
    strong_edge_sigma: float = 0.30
    consider_edge_sigma: float = 0.15

    # Spike probability thresholds (for UNDERs)
    under_spike_penalty_threshold: float = 0.65
    under_spike_bonus_threshold: float = 0.45

    # History rows minimum for confidence
    min_history_rows_strong: int = 40
    min_history_rows_consider: int = 20

    # Danger zones (historically losing combinations)
    danger_trb_under_line_range: tuple[float, float] = (6.0, 8.0)
    danger_over_low_edge_sigma: float = 0.15
    danger_over_low_edge: float = 0.5

    # Scoring weights
    weight_edge_sigma: float = 0.35
    weight_abs_edge: float = 0.20
    weight_spike_signal: float = 0.15
    weight_history_support: float = 0.10
    weight_direction_bonus: float = 0.20


def _sf(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def score_pick_precision(
    *,
    direction: str,
    target: str,
    abs_edge: float,
    edge_to_sigma: float,
    spike_probability: float,
    history_rows: int,
    market_line: float = 0.0,
    config: PrecisionFilterConfig | None = None,
) -> dict[str, Any]:
    """Score a single pick for precision quality.

    Returns:
      - precision_score: 0-1 composite score
      - precision_tier: elite/strong/consider/pass/danger
      - precision_flags: list of reasons
    """
    cfg = config or PrecisionFilterConfig()
    if not cfg.enabled:
        return {"precision_score": 0.5, "precision_tier": "consider", "precision_flags": ["disabled"]}

    dir_upper = str(direction).upper().strip()
    tgt_upper = str(target).upper().strip()
    edge = max(0.0, _sf(abs_edge))
    e_s = max(0.0, _sf(edge_to_sigma))
    spike = float(np.clip(_sf(spike_probability, 0.5), 0.0, 1.0))
    hrows = int(max(0, _sf(history_rows)))
    line = _sf(market_line)

    flags: list[str] = []

    # --- Danger zone detection ---
    is_danger = False

    # OVER with low edge-to-sigma is historically a coin flip or worse
    if dir_upper == "OVER" and e_s < cfg.danger_over_low_edge_sigma:
        is_danger = True
        flags.append("danger_over_low_es")

    # OVER with low absolute edge
    if dir_upper == "OVER" and edge < cfg.danger_over_low_edge:
        is_danger = True
        flags.append("danger_over_low_edge")

    # TRB UNDER in the 6-8 line range (historically weak at 61.8%)
    if tgt_upper == "TRB" and dir_upper == "UNDER":
        lo, hi = cfg.danger_trb_under_line_range
        if lo <= line < hi:
            flags.append("weak_trb_under_6_8")

    if is_danger:
        return {"precision_score": 0.0, "precision_tier": "danger", "precision_flags": flags}

    # --- Component scores ---

    # Edge-to-sigma (strongest signal, 0-1 normalized)
    es_score = float(np.clip(e_s / 0.70, 0.0, 1.0))

    # Absolute edge (0-1 normalized)
    edge_score = float(np.clip(edge / 3.0, 0.0, 1.0))

    # Spike signal (lower spike = better for UNDERs)
    if dir_upper == "UNDER":
        spike_score = float(np.clip(1.0 - spike / 0.80, 0.0, 1.0))
        if spike <= cfg.under_spike_bonus_threshold:
            flags.append("low_spike_bonus")
        elif spike >= cfg.under_spike_penalty_threshold:
            flags.append("high_spike_penalty")
    else:
        # For OVERs, high spike is actually good (player has upside variance)
        spike_score = float(np.clip(spike / 0.80, 0.0, 1.0))

    # History support
    history_score = float(np.clip(hrows / 80.0, 0.0, 1.0))

    # Direction bonus (UNDERs are inherently more reliable)
    direction_score = 0.8 if dir_upper == "UNDER" else 0.3

    # --- Composite score ---
    precision_score = (
        cfg.weight_edge_sigma * es_score
        + cfg.weight_abs_edge * edge_score
        + cfg.weight_spike_signal * spike_score
        + cfg.weight_history_support * history_score
        + cfg.weight_direction_bonus * direction_score
    )
    precision_score = float(np.clip(precision_score, 0.0, 1.0))

    # --- Tier assignment ---
    if e_s >= cfg.elite_edge_sigma and hrows >= cfg.min_history_rows_strong:
        tier = "elite"
        flags.append("elite_es")
    elif e_s >= cfg.strong_edge_sigma and hrows >= cfg.min_history_rows_consider:
        tier = "strong"
        flags.append("strong_es")
    elif e_s >= cfg.consider_edge_sigma:
        tier = "consider"
    else:
        tier = "pass"
        flags.append("low_es")

    # Upgrade/downgrade based on spike for UNDERs
    if dir_upper == "UNDER" and tier in ("strong", "elite"):
        if spike <= cfg.under_spike_bonus_threshold:
            precision_score = min(1.0, precision_score + 0.05)
        elif spike >= cfg.under_spike_penalty_threshold:
            if tier == "elite":
                tier = "strong"
            precision_score = max(0.0, precision_score - 0.05)
            flags.append("spike_downgrade")

    return {
        "precision_score": precision_score,
        "precision_tier": tier,
        "precision_flags": flags,
    }


def annotate_precision_filter(
    candidates: pd.DataFrame,
    *,
    config: PrecisionFilterConfig | None = None,
) -> pd.DataFrame:
    """Annotate candidates with precision filter scores and tiers.

    Adds columns:
      - pf_score: 0-1 precision score
      - pf_tier: elite/strong/consider/pass/danger
      - pf_flags: comma-separated flags
    """
    cfg = config or PrecisionFilterConfig()
    out = candidates.copy()

    if out.empty or not cfg.enabled:
        out["pf_score"] = 0.5
        out["pf_tier"] = "consider"
        out["pf_flags"] = ""
        return out

    scores = []
    tiers = []
    all_flags = []

    for _, row in out.iterrows():
        # Compute edge_to_sigma if not present
        edge = _sf(row.get("abs_edge"))
        sigma = _sf(row.get("uncertainty_sigma"), default=3.0)
        e_s = _sf(row.get("edge_to_sigma"))
        if e_s == 0.0 and sigma > 0:
            e_s = edge / max(sigma, 0.01)

        result = score_pick_precision(
            direction=str(row.get("direction", "")),
            target=str(row.get("target", "")),
            abs_edge=edge,
            edge_to_sigma=e_s,
            spike_probability=_sf(row.get("spike_probability"), 0.5),
            history_rows=int(_sf(row.get("history_rows"), 30)),
            market_line=_sf(row.get("market_line")),
            config=cfg,
        )
        scores.append(result["precision_score"])
        tiers.append(result["precision_tier"])
        all_flags.append(",".join(result["precision_flags"]))

    out["pf_score"] = scores
    out["pf_tier"] = tiers
    out["pf_flags"] = all_flags
    return out
