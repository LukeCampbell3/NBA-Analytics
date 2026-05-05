"""Daily risk manager — prevents catastrophic losing days.

The goal: never lose more than 5% of bankroll on any single day.
Ideally, break even or better every day.

Strategies:
  1. GAME DIVERSIFICATION — Require picks from at least 2-3 different games.
     When all picks are from the same 1-2 games, a single blowout kills
     the entire day.

  2. CORRELATION-AWARE SIZING — When multiple picks are from the same game,
     reduce total exposure for that game cluster.  Treat same-game picks
     as partially the same bet.

  3. PROGRESSIVE STOP-LOSS — If the first 2-3 bets of the day all lose,
     reduce remaining stakes by 50%.  This limits damage on days where
     the model is clearly miscalibrated.

  4. MAX DAILY LOSS CAP — Hard cap at 5% of bankroll per day.  Once hit,
     remaining bets are voided or reduced to minimum.

  5. CONFIDENCE FLOOR — Only publish picks where the system has genuine
     conviction.  On thin slates (few games, few eligible picks), publish
     fewer picks rather than filling with marginal ones.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DailyRiskConfig:
    """Risk management parameters."""
    enabled: bool = True

    # Game diversification
    min_unique_games: int = 2
    max_picks_per_game: int = 2
    same_game_exposure_discount: float = 0.50  # treat 2nd pick from same game as 50% exposure

    # Daily loss cap
    max_daily_loss_pct: float = 0.05  # never lose more than 5% of bankroll
    max_daily_exposure_pct: float = 0.12  # never have more than 12% at risk

    # Progressive stop-loss
    stop_loss_enabled: bool = True
    stop_loss_trigger_losses: int = 2  # after 2 consecutive losses
    stop_loss_reduction: float = 0.50  # reduce remaining stakes by 50%

    # Confidence floor
    min_picks_for_parlay: int = 3  # need at least 3 eligible legs to build a parlay
    min_board_confidence: float = 0.52  # avg win rate must exceed this to publish

    # Parlay protection
    parlay_max_loss_pct: float = 0.02  # max 2% on any single parlay
    singles_max_loss_pct: float = 0.03  # max 3% total on singles


def assess_daily_risk(
    picks: list[dict],
    *,
    bankroll: float = 1000.0,
    config: DailyRiskConfig | None = None,
) -> dict[str, Any]:
    """Assess and adjust daily risk for a set of picks.

    Returns a dict with:
      - approved_picks: list of picks with adjusted stakes
      - risk_score: 0-1 overall risk level
      - adjustments_made: list of adjustment descriptions
      - daily_max_loss: maximum possible loss for the day
      - game_concentration: how concentrated picks are in few games
    """
    cfg = config or DailyRiskConfig()

    if not cfg.enabled or not picks:
        return {
            "approved_picks": picks,
            "risk_score": 0.0,
            "adjustments_made": [],
            "daily_max_loss": 0.0,
            "game_concentration": 0.0,
        }

    adjustments = []
    approved = [dict(p) for p in picks]  # copy

    # --- 1. Game diversification check ---
    game_keys = []
    for pick in approved:
        gk = str(pick.get("_game_key", pick.get("game_key", "")))
        if not gk:
            # Build from available fields
            home = str(pick.get("market_home_team", ""))
            away = str(pick.get("market_away_team", ""))
            date = str(pick.get("market_date", ""))[:10]
            gk = f"{date}|{away}@{home}" if home and away else f"{date}|{pick.get('player', '')}"
        game_keys.append(gk)
        pick["_risk_game_key"] = gk

    game_counts = Counter(game_keys)
    unique_games = len(game_counts)
    total_picks = len(approved)

    # Game concentration: 1.0 = all picks from one game, 0.0 = perfectly spread
    if total_picks > 0:
        max_from_one_game = max(game_counts.values())
        game_concentration = max_from_one_game / total_picks
    else:
        game_concentration = 0.0

    # If too concentrated, discount same-game picks
    if unique_games < cfg.min_unique_games and total_picks > 1:
        adjustments.append(f"low_game_diversity ({unique_games} games)")
        # Reduce all stakes by 30%
        for pick in approved:
            pick["stake_fraction"] = pick.get("stake_fraction", 0.01) * 0.70

    # Apply per-game cap
    game_pick_count: dict[str, int] = {}
    for pick in approved:
        gk = pick.get("_risk_game_key", "")
        game_pick_count[gk] = game_pick_count.get(gk, 0) + 1
        if game_pick_count[gk] > cfg.max_picks_per_game:
            pick["stake_fraction"] = 0.0
            pick["_risk_excluded"] = True
            adjustments.append(f"game_cap_exceeded ({gk})")
        elif game_pick_count[gk] > 1:
            # Discount subsequent picks from same game
            pick["stake_fraction"] = pick.get("stake_fraction", 0.01) * cfg.same_game_exposure_discount
            adjustments.append(f"same_game_discount ({gk})")

    # --- 2. Daily exposure cap ---
    total_exposure = sum(p.get("stake_fraction", 0) for p in approved if not p.get("_risk_excluded"))
    if total_exposure > cfg.max_daily_exposure_pct:
        scale = cfg.max_daily_exposure_pct / total_exposure
        for pick in approved:
            if not pick.get("_risk_excluded"):
                pick["stake_fraction"] = pick.get("stake_fraction", 0) * scale
        adjustments.append(f"exposure_cap_applied (scaled to {cfg.max_daily_exposure_pct:.0%})")
        total_exposure = cfg.max_daily_exposure_pct

    # --- 3. Confidence floor ---
    win_rates = [float(p.get("expected_win_rate", p.get("precision_enhanced_win_rate", 0.5)))
                 for p in approved if not p.get("_risk_excluded")]
    avg_confidence = np.mean(win_rates) if win_rates else 0.5

    if avg_confidence < cfg.min_board_confidence:
        # Board is too weak — reduce everything by 40%
        for pick in approved:
            pick["stake_fraction"] = pick.get("stake_fraction", 0) * 0.60
        adjustments.append(f"low_confidence_floor (avg={avg_confidence:.3f})")

    # --- 4. Compute risk metrics ---
    daily_max_loss = sum(p.get("stake_fraction", 0) for p in approved if not p.get("_risk_excluded")) * bankroll
    risk_score = float(np.clip(
        0.3 * game_concentration
        + 0.3 * (total_exposure / max(cfg.max_daily_exposure_pct, 0.01))
        + 0.2 * (1.0 - avg_confidence)
        + 0.2 * (1.0 - unique_games / max(total_picks, 1)),
        0.0, 1.0,
    ))

    # Filter out excluded picks
    final_picks = [p for p in approved if not p.get("_risk_excluded")]

    return {
        "approved_picks": final_picks,
        "risk_score": risk_score,
        "adjustments_made": adjustments,
        "daily_max_loss": daily_max_loss,
        "game_concentration": game_concentration,
        "unique_games": unique_games,
        "avg_confidence": avg_confidence,
        "total_exposure_pct": total_exposure,
    }


def apply_progressive_stop_loss(
    results_so_far: list[str],
    remaining_picks: list[dict],
    *,
    config: DailyRiskConfig | None = None,
) -> list[dict]:
    """Apply progressive stop-loss based on results so far today.

    If the first N bets all lost, reduce remaining stakes.

    Parameters
    ----------
    results_so_far : list of "win"/"loss" strings
    remaining_picks : picks not yet placed

    Returns adjusted remaining picks.
    """
    cfg = config or DailyRiskConfig()

    if not cfg.stop_loss_enabled or not results_so_far:
        return remaining_picks

    # Count consecutive losses from the end
    consecutive_losses = 0
    for r in reversed(results_so_far):
        if r == "loss":
            consecutive_losses += 1
        else:
            break

    if consecutive_losses >= cfg.stop_loss_trigger_losses:
        reduction = cfg.stop_loss_reduction
        adjusted = []
        for pick in remaining_picks:
            p = dict(pick)
            p["stake_fraction"] = p.get("stake_fraction", 0.01) * reduction
            p["_stop_loss_applied"] = True
            adjusted.append(p)
        return adjusted

    return remaining_picks
