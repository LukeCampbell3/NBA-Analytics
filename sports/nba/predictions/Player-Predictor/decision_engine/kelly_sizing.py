"""Fractional Kelly sizing for precision-filtered picks.

Replaces flat-stake sizing with edge-proportional bet sizing using the
Kelly Criterion at 25% fraction (quarter-Kelly).  This reduces variance
by ~75% while retaining ~75% of long-term growth rate.

The key insight: a pick with 76% win rate at -110 odds has a much larger
optimal stake than a pick with 55% win rate.  Flat sizing treats them
equally, leaving significant growth on the table for high-confidence picks
and over-exposing on marginal ones.

Sizing tiers:
  - Elite (pf_tier=elite, WR~82%): 3-4% of bankroll
  - Strong (pf_tier=strong, WR~76%): 2-3% of bankroll
  - Consider (WR~65%): 1-1.5% of bankroll
  - Marginal (WR~55%): 0.5% of bankroll (minimum)
  - Danger/Pass: 0% (no bet)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class KellySizingConfig:
    """Configuration for fractional Kelly sizing."""
    enabled: bool = True
    kelly_fraction: float = 0.25          # quarter-Kelly (conservative)
    american_odds: int = -110             # standard juice
    max_single_bet_pct: float = 0.04     # never risk more than 4% on one bet
    min_single_bet_pct: float = 0.005    # minimum 0.5% to be worth placing
    max_total_exposure_pct: float = 0.15  # max 15% of bankroll at risk per day
    parlay_stake_pct: float = 0.02       # 2% per parlay ticket
    danger_stake: float = 0.0            # don't bet danger-tier picks


def _payout_per_unit(odds: int) -> float:
    """Convert American odds to profit per unit risked."""
    if odds < 0:
        return 100.0 / abs(odds)
    return abs(odds) / 100.0


def kelly_fraction(
    win_probability: float,
    payout: float,
    fraction: float = 0.25,
) -> float:
    """Compute fractional Kelly stake as a fraction of bankroll.

    Full Kelly: f* = (bp - q) / b
    where b = payout per unit, p = win prob, q = 1 - p

    We multiply by `fraction` (default 0.25) for quarter-Kelly.
    """
    p = float(np.clip(win_probability, 0.01, 0.99))
    q = 1.0 - p
    b = float(max(0.01, payout))

    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0

    return float(np.clip(full_kelly * fraction, 0.0, 1.0))


def compute_stake(
    *,
    win_probability: float,
    precision_tier: str = "consider",
    precision_score: float = 0.5,
    bankroll: float = 1000.0,
    config: KellySizingConfig | None = None,
) -> dict[str, Any]:
    """Compute the optimal stake for a single pick.

    Returns:
      - stake_fraction: fraction of bankroll to risk
      - stake_dollars: dollar amount
      - kelly_raw: raw full-Kelly fraction
      - kelly_adjusted: after quarter-Kelly and caps
      - sizing_tier: label for the stake level
    """
    cfg = config or KellySizingConfig()

    if not cfg.enabled:
        return {
            "stake_fraction": 0.01,
            "stake_dollars": bankroll * 0.01,
            "kelly_raw": 0.0,
            "kelly_adjusted": 0.01,
            "sizing_tier": "flat",
        }

    tier = str(precision_tier).lower().strip()
    if tier == "danger":
        return {
            "stake_fraction": cfg.danger_stake,
            "stake_dollars": 0.0,
            "kelly_raw": 0.0,
            "kelly_adjusted": 0.0,
            "sizing_tier": "no_bet",
        }

    payout = _payout_per_unit(cfg.american_odds)
    raw_kelly = kelly_fraction(win_probability, payout, fraction=1.0)
    adjusted = kelly_fraction(win_probability, payout, fraction=cfg.kelly_fraction)

    # Apply caps
    adjusted = float(np.clip(adjusted, cfg.min_single_bet_pct, cfg.max_single_bet_pct))

    # Tier-based floor/ceiling adjustments
    if tier == "elite":
        adjusted = max(adjusted, 0.025)  # at least 2.5% for elite
    elif tier == "strong":
        adjusted = max(adjusted, 0.015)  # at least 1.5% for strong
    elif tier == "pass":
        adjusted = min(adjusted, 0.005)  # cap at 0.5% for pass

    # Precision score multiplier (0.8-1.2x based on score)
    score_mult = 0.8 + 0.4 * float(np.clip(precision_score, 0.0, 1.0))
    adjusted *= score_mult
    adjusted = float(np.clip(adjusted, cfg.min_single_bet_pct, cfg.max_single_bet_pct))

    # Sizing tier label
    if adjusted >= 0.03:
        sizing_tier = "max"
    elif adjusted >= 0.02:
        sizing_tier = "high"
    elif adjusted >= 0.012:
        sizing_tier = "medium"
    elif adjusted >= 0.007:
        sizing_tier = "low"
    else:
        sizing_tier = "minimum"

    return {
        "stake_fraction": adjusted,
        "stake_dollars": bankroll * adjusted,
        "kelly_raw": raw_kelly,
        "kelly_adjusted": adjusted,
        "sizing_tier": sizing_tier,
    }


def size_daily_board(
    picks: list[dict],
    *,
    bankroll: float = 1000.0,
    config: KellySizingConfig | None = None,
) -> list[dict]:
    """Apply Kelly sizing to a list of picks, respecting total exposure cap.

    Modifies picks in-place and returns them with added sizing fields.
    """
    cfg = config or KellySizingConfig()

    # First pass: compute raw stakes
    for pick in picks:
        wr = float(pick.get("expected_win_rate", pick.get("precision_enhanced_win_rate", 0.5)))
        tier = str(pick.get("pf_tier", pick.get("stake_tier", "consider")))
        score = float(pick.get("pf_score", 0.5))

        result = compute_stake(
            win_probability=wr,
            precision_tier=tier,
            precision_score=score,
            bankroll=bankroll,
            config=cfg,
        )
        pick.update(result)

    # Second pass: enforce total exposure cap
    total_exposure = sum(p.get("stake_fraction", 0) for p in picks)
    if total_exposure > cfg.max_total_exposure_pct:
        scale = cfg.max_total_exposure_pct / total_exposure
        for pick in picks:
            pick["stake_fraction"] *= scale
            pick["stake_dollars"] = bankroll * pick["stake_fraction"]
            pick["kelly_adjusted"] *= scale

    return picks
