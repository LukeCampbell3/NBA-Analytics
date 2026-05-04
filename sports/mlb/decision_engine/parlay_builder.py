"""MLB daily parlay construction and singles board engine.

Builds optimised parlay tickets and a ranked singles board from the
MLB high-precision prediction pool.  Adapted from the NBA parlay builder
with MLB-specific correlation structure and target characteristics.

MLB-specific considerations:
  • Count targets (H, TB, R, K) are Poisson-distributed with low lines
    (0.5, 1.5, 2.5), making UNDER 0.5 lines very common and high-probability.
  • Pitcher performance affects all hitters in a game → same-game hitter
    parlays carry hidden correlation through the opposing pitcher.
  • Games are more independent than NBA (no back-to-backs, less fatigue).
  • Weather/park factors create game-level correlation.

Output structure (daily):
  1. PRIMARY PARLAY   – 2-3 legs, highest-confidence picks from
                        different games, preferring different teams.
  2. SINGLES BOARD    – Remaining high-value picks ranked by expected
                        profit, sized by confidence tier.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MLBParlayConfig:
    """Tunable parameters for the MLB parlay builder."""

    # --- Leg eligibility ---
    min_leg_hit_prob: float = 0.70
    min_leg_ev: float = 0.0
    min_leg_abs_edge: float = 0.20
    min_leg_history_rows: int = 10
    max_leg_days_since_history: int = 5
    allowed_targets: tuple[str, ...] = ("H", "TB", "R", "K")
    min_confidence_tier: str = "consider"  # consider, strong, elite

    # --- Primary parlay ---
    primary_min_legs: int = 2
    primary_max_legs: int = 3
    primary_prefer_different_games: bool = True
    primary_max_same_game_legs: int = 1
    primary_min_avg_hit_prob: float = 0.72
    primary_max_tickets: int = 1

    # --- Singles board ---
    singles_max_picks: int = 8
    singles_min_hit_prob: float = 0.65
    singles_min_ev: float = -0.05

    # --- Correlation penalties ---
    same_player_penalty: float = 1.0   # hard block
    same_game_penalty: float = 0.10    # higher than NBA — pitcher correlation
    same_team_penalty: float = 0.04
    same_target_penalty: float = 0.02
    different_game_bonus: float = 0.04

    # --- Sizing ---
    parlay_stake_fraction: float = 0.02
    single_elite_stake: float = 0.015
    single_strong_stake: float = 0.010
    single_consider_stake: float = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _str(value: Any, default: str = "") -> str:
    s = str(value or "").strip()
    return s if s and s.lower() not in ("nan", "none", "null") else default


CONFIDENCE_RANK = {"elite": 0, "strong": 1, "consider": 2, "pass": 3}


def _game_key(row: dict | pd.Series) -> str:
    gid = _str(row.get("Game_ID", row.get("game_id")))
    if gid:
        return gid
    team = _str(row.get("Team", row.get("team")))
    opp = _str(row.get("Opponent", row.get("opponent")))
    date = _str(row.get("Game_Date", row.get("game_date", row.get("market_date", ""))))[:10]
    if team and opp:
        teams = f"{min(team, opp)}@{max(team, opp)}"
        return f"{date}|{teams}"
    return f"{date}|{_str(row.get('Player', row.get('player')))}"


# ---------------------------------------------------------------------------
# Leg eligibility
# ---------------------------------------------------------------------------

def _is_eligible_leg(row: dict, cfg: MLBParlayConfig) -> tuple[bool, str]:
    """Check if an MLB candidate qualifies as a parlay leg."""
    hit_prob = _sf(row.get("Estimated_Hit_Probability", row.get("calibrated_hit_probability")))
    ev = _sf(row.get("Expected_Value_Per_Unit", row.get("expected_value_per_unit")), default=float("nan"))
    edge = _sf(row.get("Abs_Edge", row.get("abs_edge")))
    hrows = int(_sf(row.get("History_Rows", row.get("history_rows"))))
    days_since = int(_sf(row.get("Days_Since_History", row.get("days_since_history")), default=999))
    target = _str(row.get("Target", row.get("target"))).upper()
    tier = _str(row.get("Confidence_Tier", row.get("confidence_tier"))).lower()

    if target not in cfg.allowed_targets:
        return False, "excluded_target"
    if hit_prob < cfg.min_leg_hit_prob:
        return False, "low_hit_prob"
    if np.isfinite(ev) and ev < cfg.min_leg_ev:
        return False, "low_ev"
    if edge < cfg.min_leg_abs_edge:
        return False, "low_edge"
    if hrows < cfg.min_leg_history_rows:
        return False, "low_history"
    if days_since > cfg.max_leg_days_since_history:
        return False, "stale_history"
    if CONFIDENCE_RANK.get(tier, 3) > CONFIDENCE_RANK.get(cfg.min_confidence_tier, 2):
        return False, "low_confidence_tier"

    return True, "eligible"


def _leg_score(row: dict) -> float:
    """Score a leg for ranking.  Higher = better parlay candidate."""
    hit_prob = _sf(row.get("Estimated_Hit_Probability", row.get("calibrated_hit_probability")), 0.5)
    ev = _sf(row.get("Expected_Value_Per_Unit", row.get("expected_value_per_unit")), 0.0)
    edge = _sf(row.get("Abs_Edge", row.get("abs_edge")))
    tier = _str(row.get("Confidence_Tier", row.get("confidence_tier"))).lower()
    precision = _sf(row.get("Precision_Score", row.get("precision_score")), 0.5)

    tier_score = {
        "elite": 1.0,
        "strong": 0.7,
        "consider": 0.4,
    }.get(tier, 0.2)

    return (
        0.35 * hit_prob
        + 0.20 * min(edge / 2.0, 1.0)
        + 0.15 * min(max(ev, 0) / 1.0, 1.0)
        + 0.15 * tier_score
        + 0.15 * precision
    )


# ---------------------------------------------------------------------------
# Parlay scoring
# ---------------------------------------------------------------------------

def _score_parlay(legs: list[dict], cfg: MLBParlayConfig) -> dict[str, Any]:
    """Score a candidate parlay combination."""
    n = len(legs)
    hit_probs = [
        _sf(leg.get("Estimated_Hit_Probability", leg.get("calibrated_hit_probability")), 0.5)
        for leg in legs
    ]
    joint_prob = math.prod(hit_probs)
    avg_prob = sum(hit_probs) / n

    penalty = 0.0
    bonus = 0.0
    players = set()
    games = set()
    teams = set()
    same_player = False
    same_game = False

    for i, j in combinations(range(n), 2):
        pi = _str(legs[i].get("Player", legs[i].get("player")))
        pj = _str(legs[j].get("Player", legs[j].get("player")))
        gi = _game_key(legs[i])
        gj = _game_key(legs[j])
        ti_team = _str(legs[i].get("Team", legs[i].get("team")))
        tj_team = _str(legs[j].get("Team", legs[j].get("team")))
        ti_target = _str(legs[i].get("Target", legs[i].get("target")))
        tj_target = _str(legs[j].get("Target", legs[j].get("target")))

        if pi and pi == pj:
            same_player = True
            penalty += cfg.same_player_penalty
        if gi and gi == gj:
            same_game = True
            penalty += cfg.same_game_penalty
        else:
            bonus += cfg.different_game_bonus
        if ti_team and ti_team == tj_team:
            penalty += cfg.same_team_penalty
        if ti_target and ti_target == tj_target:
            penalty += cfg.same_target_penalty

    for leg in legs:
        players.add(_str(leg.get("Player", leg.get("player"))))
        games.add(_game_key(leg))
        teams.add(_str(leg.get("Team", leg.get("team"))))

    if same_player:
        return {"score": -1.0, "blocked": True, "reason": "same_player"}

    correlation_factor = max(0.50, 1.0 - penalty + bonus)
    adjusted_prob = joint_prob * correlation_factor
    adjusted_prob = min(adjusted_prob, min(hit_probs))

    leg_scores = [_leg_score(leg) for leg in legs]
    avg_quality = sum(leg_scores) / n

    game_diversity = len(games) / n
    team_diversity = len(teams) / n

    score = (
        adjusted_prob
        * (0.65 + 0.35 * avg_quality)
        * (0.80 + 0.20 * game_diversity)
    )

    return {
        "score": score,
        "blocked": False,
        "reason": "",
        "joint_prob": joint_prob,
        "adjusted_prob": adjusted_prob,
        "avg_hit_prob": avg_prob,
        "avg_quality": avg_quality,
        "correlation_factor": correlation_factor,
        "game_diversity": game_diversity,
        "team_diversity": team_diversity,
        "same_game": same_game,
        "n_games": len(games),
        "n_teams": len(teams),
        "leg_count": n,
    }


# ---------------------------------------------------------------------------
# Core: build daily board
# ---------------------------------------------------------------------------

@dataclass
class MLBDailyBoard:
    """Output of the MLB parlay builder for a single day."""
    primary_parlays: list[dict[str, Any]] = field(default_factory=list)
    singles: list[dict[str, Any]] = field(default_factory=list)
    all_candidates: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_mlb_daily_board(
    candidates: pd.DataFrame,
    *,
    config: MLBParlayConfig | None = None,
) -> MLBDailyBoard:
    """Build the daily MLB parlay + singles board.

    Parameters
    ----------
    candidates : pd.DataFrame
        The high-precision prediction pool (output of select_high_precision_predictions).
    config : MLBParlayConfig, optional
        Tunable parameters.

    Returns
    -------
    MLBDailyBoard with primary_parlays, singles, and diagnostics.
    """
    cfg = config or MLBParlayConfig()
    board = MLBDailyBoard()

    if candidates.empty:
        board.diagnostics = {"status": "empty_candidates"}
        return board

    # --- 1. Annotate eligibility ---
    rows: list[dict] = []
    for _, row in candidates.iterrows():
        d = row.to_dict()
        eligible, reason = _is_eligible_leg(d, cfg)
        d["parlay_eligible"] = eligible
        d["parlay_reject_reason"] = reason if not eligible else ""
        d["parlay_leg_score"] = _leg_score(d) if eligible else 0.0
        d["_game_key"] = _game_key(d)
        rows.append(d)

    board.all_candidates = rows
    eligible_legs = sorted(
        [r for r in rows if r["parlay_eligible"]],
        key=lambda r: r["parlay_leg_score"],
        reverse=True,
    )

    board.diagnostics = {
        "total_candidates": len(rows),
        "eligible_legs": len(eligible_legs),
        "eligible_by_target": dict(Counter(
            _str(r.get("Target", r.get("target"))).upper()
            for r in eligible_legs
        )),
    }

    # --- 2. Build primary parlays ---
    used_players: set[str] = set()

    primary_candidates: list[dict] = []
    for leg_count in range(cfg.primary_max_legs, cfg.primary_min_legs - 1, -1):
        if len(eligible_legs) < leg_count:
            continue

        best_parlay = None
        best_score = -1.0

        pool = eligible_legs[:min(15, len(eligible_legs))]

        for combo in combinations(range(len(pool)), leg_count):
            legs = [pool[i] for i in combo]

            if cfg.primary_prefer_different_games:
                game_keys = [_game_key(leg) for leg in legs]
                game_counts = Counter(game_keys)
                if any(c > cfg.primary_max_same_game_legs for c in game_counts.values()):
                    continue

            result = _score_parlay(legs, cfg)
            if result["blocked"]:
                continue
            if result["avg_hit_prob"] < cfg.primary_min_avg_hit_prob:
                continue
            if result["score"] > best_score:
                best_score = result["score"]
                best_parlay = {
                    "type": "primary",
                    "legs": legs,
                    "leg_count": leg_count,
                    **result,
                }

        if best_parlay:
            primary_candidates.append(best_parlay)

    # Select best primary — prefer 3-leg if score is close
    if primary_candidates:
        primary_candidates.sort(key=lambda p: p["score"], reverse=True)
        selected = primary_candidates[0]

        for candidate in primary_candidates:
            if candidate["leg_count"] == 3 and candidate["score"] >= selected["score"] * 0.80:
                selected = candidate
                break

        board.primary_parlays.append(selected)
        for leg in selected["legs"]:
            used_players.add(_str(leg.get("Player", leg.get("player"))))

    # --- 3. Build singles board ---
    parlay_players = set()
    for parlay in board.primary_parlays:
        for leg in parlay.get("legs", []):
            parlay_players.add(_str(leg.get("Player", leg.get("player"))))

    singles_pool = []
    for r in rows:
        hit_prob = _sf(r.get("Estimated_Hit_Probability", r.get("calibrated_hit_probability")))
        ev = _sf(r.get("Expected_Value_Per_Unit", r.get("expected_value_per_unit")), default=float("nan"))
        target = _str(r.get("Target", r.get("target"))).upper()

        if target not in cfg.allowed_targets:
            continue
        if hit_prob < cfg.singles_min_hit_prob:
            continue
        if np.isfinite(ev) and ev < cfg.singles_min_ev:
            continue

        player = _str(r.get("Player", r.get("player")))
        if player in parlay_players:
            r["singles_note"] = "also_in_parlay"

        r["singles_score"] = _leg_score(r)
        singles_pool.append(r)

    singles_pool.sort(key=lambda r: r["singles_score"], reverse=True)

    seen_players: set[str] = set()
    seen_buckets: set[str] = set()
    for r in singles_pool:
        player = _str(r.get("Player", r.get("player")))
        bucket = _str(r.get("Market_Bucket", r.get("market_bucket")))

        if player in seen_players:
            continue
        if bucket and bucket in seen_buckets:
            continue

        seen_players.add(player)
        if bucket:
            seen_buckets.add(bucket)

        tier = _str(r.get("Confidence_Tier", r.get("confidence_tier"))).lower()
        if tier == "elite":
            r["stake_tier"] = "elite"
            r["stake_fraction"] = cfg.single_elite_stake
        elif tier == "strong":
            r["stake_tier"] = "strong"
            r["stake_fraction"] = cfg.single_strong_stake
        else:
            r["stake_tier"] = "consider"
            r["stake_fraction"] = cfg.single_consider_stake

        board.singles.append(r)
        if len(board.singles) >= cfg.singles_max_picks:
            break

    # --- 4. Summary ---
    board.diagnostics.update({
        "primary_parlays": len(board.primary_parlays),
        "singles": len(board.singles),
        "primary_avg_hit_prob": (
            board.primary_parlays[0]["avg_hit_prob"]
            if board.primary_parlays else None
        ),
        "primary_joint_prob": (
            board.primary_parlays[0]["joint_prob"]
            if board.primary_parlays else None
        ),
        "primary_leg_count": (
            board.primary_parlays[0]["leg_count"]
            if board.primary_parlays else None
        ),
    })

    return board


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_mlb_daily_board(board: MLBDailyBoard, bankroll: float = 1000.0) -> str:
    """Format the MLB daily board as a human-readable string."""
    lines: list[str] = []
    payout_per_unit = 100.0 / 110.0

    lines.append("=" * 70)
    lines.append("MLB DAILY BOARD")
    lines.append("=" * 70)

    for i, parlay in enumerate(board.primary_parlays, 1):
        legs = parlay.get("legs", [])
        n = len(legs)
        parlay_odds_mult = (1 + payout_per_unit) ** n
        stake = bankroll * 0.02
        potential_profit = stake * (parlay_odds_mult - 1)

        lines.append(f"\n{'─' * 70}")
        lines.append(f"PRIMARY PARLAY #{i}  ({n}-leg)")
        lines.append(f"  Joint hit probability: {parlay.get('joint_prob', 0):.1%}")
        lines.append(f"  Adjusted probability: {parlay.get('adjusted_prob', 0):.1%}")
        lines.append(f"  Avg leg hit rate: {parlay.get('avg_hit_prob', 0):.1%}")
        lines.append(f"  Games: {parlay.get('n_games', 0)} different | Teams: {parlay.get('n_teams', 0)} different")
        lines.append(f"  Stake: ${stake:.0f} → potential profit: ${potential_profit:.0f}")
        lines.append(f"  {'─' * 60}")

        for j, leg in enumerate(legs, 1):
            player = _str(leg.get("Player", leg.get("player")), "?")
            target = _str(leg.get("Target", leg.get("target")), "?")
            direction = _str(leg.get("Direction", leg.get("direction")), "?")
            line = leg.get("Market_Line", leg.get("market_line", "?"))
            hit_prob = _sf(leg.get("Estimated_Hit_Probability", leg.get("calibrated_hit_probability")))
            edge = _sf(leg.get("Abs_Edge", leg.get("abs_edge")))
            team = _str(leg.get("Team", leg.get("team")), "?")
            opp = _str(leg.get("Opponent", leg.get("opponent")), "?")
            lines.append(
                f"  Leg {j}: {player} ({team} vs {opp}) "
                f"{target} {direction} {line}  "
                f"(Hit: {hit_prob:.1%}, edge: {edge:.2f})"
            )

    if board.singles:
        lines.append(f"\n{'─' * 70}")
        lines.append("SINGLES BOARD")
        lines.append(f"  {'─' * 60}")

        for i, pick in enumerate(board.singles, 1):
            player = _str(pick.get("Player", pick.get("player")), "?")
            target = _str(pick.get("Target", pick.get("target")), "?")
            direction = _str(pick.get("Direction", pick.get("direction")), "?")
            line = pick.get("Market_Line", pick.get("market_line", "?"))
            hit_prob = _sf(pick.get("Estimated_Hit_Probability", pick.get("calibrated_hit_probability")))
            ev = _sf(pick.get("Expected_Value_Per_Unit", pick.get("expected_value_per_unit")))
            tier = pick.get("stake_tier", "?")
            frac = pick.get("stake_fraction", 0)
            stake = bankroll * frac
            team = _str(pick.get("Team", pick.get("team")), "?")
            note = f"  [{pick['singles_note']}]" if pick.get("singles_note") else ""
            ev_str = f"EV: {ev:+.3f}" if np.isfinite(ev) else "EV: n/a"
            lines.append(
                f"  {i}. {player} ({team}) {target} {direction} {line}  "
                f"Hit: {hit_prob:.1%} | {ev_str} | {tier} ${stake:.0f}{note}"
            )

    lines.append(f"\n{'─' * 70}")
    lines.append("DIAGNOSTICS")
    diag = board.diagnostics
    lines.append(f"  Candidates: {diag.get('total_candidates', 0)}")
    lines.append(f"  Eligible parlay legs: {diag.get('eligible_legs', 0)}")
    lines.append(f"  By target: {diag.get('eligible_by_target', {})}")
    lines.append(f"  Primary parlays: {diag.get('primary_parlays', 0)}")
    lines.append(f"  Singles: {diag.get('singles', 0)}")

    return "\n".join(lines)
