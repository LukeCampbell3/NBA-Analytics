"""Daily parlay construction engine.

Builds optimised parlay tickets and a ranked singles board from the
precision-pool candidate set.  The engine is designed around the empirical
finding that UNDER picks with edge ≥ 0.5 hit at ~72% and produce 2-leg
parlays that hit at ~63% (+130% ROI) and 3-leg parlays at ~58% (+303% ROI).

Output structure (daily):
  1. PRIMARY PARLAY   – 2-3 legs, highest-confidence UNDER picks from
                        different players and preferably different games.
  2. SECONDARY PARLAY – 2 legs, may include a high-edge TRB/AST OVER
                        if one qualifies alongside a strong UNDER.
  3. SINGLES BOARD    – Remaining high-value picks ranked by expected
                        profit, sized by confidence tier.

Design principles:
  • Parlays must hit → every leg must individually clear a high bar.
  • Decorrelation → prefer legs from different games to avoid correlated
    outcomes (e.g. blowout kills all same-game unders).
  • Fail-closed → if the candidate pool is thin, emit fewer/no parlays
    rather than filling with marginal legs.
  • Transparent → every decision is annotated for audit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParlayConfig:
    """Tunable parameters for the parlay builder."""

    # --- Leg eligibility ---
    min_leg_win_rate: float = 0.52
    min_leg_ev: float = 0.0
    min_leg_abs_edge: float = 0.25
    min_leg_confidence: float = 0.01
    max_leg_risk_penalty: float = 0.60
    max_leg_spike_probability: float = 0.72
    min_leg_feasibility: float = 0.45

    # --- OVER leg eligibility (stricter) ---
    over_min_win_rate: float = 0.55
    over_min_abs_edge: float = 1.0
    over_allowed_targets: tuple[str, ...] = ("TRB", "AST")
    over_max_spike_probability: float = 0.60

    # --- Primary parlay ---
    primary_min_legs: int = 2
    primary_max_legs: int = 3
    primary_prefer_different_games: bool = True
    primary_max_same_game_legs: int = 1
    primary_min_avg_win_rate: float = 0.53
    primary_max_tickets: int = 1

    # --- Secondary parlay ---
    secondary_enabled: bool = False
    secondary_legs: int = 2
    secondary_max_tickets: int = 1
    secondary_allow_over_legs: bool = True

    # --- Singles board ---
    singles_max_picks: int = 6
    singles_min_win_rate: float = 0.51
    singles_min_ev: float = -0.01

    # --- Correlation penalties ---
    same_player_penalty: float = 1.0   # hard block (infinite penalty)
    same_game_penalty: float = 0.08
    same_target_penalty: float = 0.02
    different_game_bonus: float = 0.03

    # --- Sizing ---
    parlay_stake_fraction: float = 0.02    # 2% of bankroll per parlay
    single_high_stake: float = 0.015       # 1.5% for high-confidence singles
    single_med_stake: float = 0.010        # 1.0% for medium singles
    single_low_stake: float = 0.005        # 0.5% for lower singles
    single_high_threshold: float = 0.65    # win rate threshold for high tier
    single_med_threshold: float = 0.60     # win rate threshold for medium tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _game_key(row: dict | pd.Series) -> str:
    for col in ("game_key", "market_event_id"):
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("", "nan", "none"):
            return val
    home = str(row.get("market_home_team", "")).strip()
    away = str(row.get("market_away_team", "")).strip()
    date = str(row.get("market_date", "")).strip()[:10]
    if home and away:
        teams = f"{min(home, away)}@{max(home, away)}"
        return f"{date}|{teams}"
    return f"{date}|{row.get('player', '')}"


# ---------------------------------------------------------------------------
# Leg eligibility
# ---------------------------------------------------------------------------

def _is_eligible_leg(row: dict, cfg: ParlayConfig) -> tuple[bool, str]:
    """Check if a candidate qualifies as a parlay leg."""
    wr = _sf(row.get("expected_win_rate"))
    ev = _sf(row.get("ev"))
    edge = _sf(row.get("abs_edge"))
    conf = _sf(row.get("final_confidence"))
    risk = _sf(row.get("risk_penalty"), 0.5)
    spike = _sf(row.get("spike_probability"), 0.5)
    feas = _sf(row.get("feasibility"), 0.5)
    direction = str(row.get("direction", "")).upper().strip()
    target = str(row.get("target", "")).upper().strip()

    if direction == "OVER":
        if wr < cfg.over_min_win_rate:
            return False, "over_low_wr"
        if edge < cfg.over_min_abs_edge:
            return False, "over_low_edge"
        if target not in cfg.over_allowed_targets:
            return False, "over_excluded_target"
        if spike > cfg.over_max_spike_probability:
            return False, "over_high_spike"
    else:
        if wr < cfg.min_leg_win_rate:
            return False, "low_wr"

    if ev < cfg.min_leg_ev:
        return False, "low_ev"
    if edge < cfg.min_leg_abs_edge:
        return False, "low_edge"
    if conf < cfg.min_leg_confidence:
        return False, "low_confidence"
    if risk > cfg.max_leg_risk_penalty:
        return False, "high_risk"
    if spike > cfg.max_leg_spike_probability:
        return False, "high_spike"
    if feas < cfg.min_leg_feasibility:
        return False, "low_feasibility"

    # Precision filter gate: reject "danger" tier picks
    pf_tier = str(row.get("pf_tier", "")).lower()
    if pf_tier == "danger":
        return False, "precision_danger"

    return True, "eligible"


def _leg_score(row: dict) -> float:
    """Score a leg for ranking.  Higher = better parlay candidate.
    
    Heavily weights abs_edge and direction because historical data shows:
    - UNDER with high edge = most profitable segment
    - Edge magnitude is the strongest predictor of win rate
    """
    # Use enhanced win rate if available, otherwise base
    wr = _sf(row.get("precision_enhanced_win_rate", row.get("expected_win_rate")), 0.5)
    ev = _sf(row.get("ev"))
    edge = _sf(row.get("abs_edge"))
    conf = _sf(row.get("final_confidence"))
    quality = _sf(row.get("parlay_leg_quality_score", row.get("final_pool_quality_score")), 0.5)
    direction = str(row.get("direction", "")).upper()
    enhancement_adj = _sf(row.get("precision_enhancement_adj"), 0.0)
    
    # UNDER bonus: UNDERs historically win at 68% vs OVERs at 50%
    direction_bonus = 0.08 if direction == "UNDER" else 0.0
    
    # Edge is the strongest signal — normalize to 0-1 range
    edge_score = min(edge / 2.5, 1.0)
    
    # Enhancement bonus: picks that passed all four checks get a lift
    enhancement_score = float(np.clip(enhancement_adj * 5.0, -0.5, 0.5)) + 0.5

    # Precision filter score (0-1, from empirical edge-to-sigma analysis)
    pf_score = _sf(row.get("pf_score"), 0.5)
    
    return (
        0.20 * wr
        + 0.18 * edge_score
        + 0.18 * pf_score
        + 0.12 * min(ev + 0.05, 0.35)
        + 0.08 * conf
        + 0.08 * quality
        + 0.08 * direction_bonus / 0.08
        + 0.08 * enhancement_score
    )


# ---------------------------------------------------------------------------
# Parlay scoring
# ---------------------------------------------------------------------------

def _score_parlay(legs: list[dict], cfg: ParlayConfig) -> dict[str, Any]:
    """Score a candidate parlay combination."""
    n = len(legs)
    win_rates = [_sf(leg.get("expected_win_rate"), 0.5) for leg in legs]
    joint_prob = math.prod(win_rates)
    avg_wr = sum(win_rates) / n

    # Correlation adjustments
    penalty = 0.0
    bonus = 0.0
    players = set()
    games = set()
    targets = set()
    same_player = False
    same_game = False

    for i, j in combinations(range(n), 2):
        pi = str(legs[i].get("player", "")).strip()
        pj = str(legs[j].get("player", "")).strip()
        gi = _game_key(legs[i])
        gj = _game_key(legs[j])
        ti = str(legs[i].get("target", "")).strip()
        tj = str(legs[j].get("target", "")).strip()

        if pi and pi == pj:
            same_player = True
            penalty += cfg.same_player_penalty
        if gi and gi == gj:
            same_game = True
            penalty += cfg.same_game_penalty
        else:
            bonus += cfg.different_game_bonus
        if ti and ti == tj:
            penalty += cfg.same_target_penalty

    for leg in legs:
        players.add(str(leg.get("player", "")))
        games.add(_game_key(leg))
        targets.add(str(leg.get("target", "")))

    # Hard block: never parlay same player
    if same_player:
        return {"score": -1.0, "blocked": True, "reason": "same_player"}

    # Adjusted probability
    correlation_factor = max(0.50, 1.0 - penalty + bonus)
    adjusted_prob = joint_prob * correlation_factor
    adjusted_prob = min(adjusted_prob, min(win_rates))  # can't exceed weakest leg

    # Quality components
    leg_scores = [_leg_score(leg) for leg in legs]
    avg_quality = sum(leg_scores) / n

    # Diversity bonus
    game_diversity = len(games) / n
    target_diversity = len(targets) / n

    # Final score: probability × quality × diversity
    score = (
        adjusted_prob
        * (0.70 + 0.30 * avg_quality)
        * (0.85 + 0.15 * game_diversity)
    )

    return {
        "score": score,
        "blocked": False,
        "reason": "",
        "joint_prob": joint_prob,
        "adjusted_prob": adjusted_prob,
        "avg_win_rate": avg_wr,
        "avg_quality": avg_quality,
        "correlation_factor": correlation_factor,
        "game_diversity": game_diversity,
        "target_diversity": target_diversity,
        "same_game": same_game,
        "n_games": len(games),
        "n_targets": len(targets),
        "leg_count": n,
    }


# ---------------------------------------------------------------------------
# Core: build daily board
# ---------------------------------------------------------------------------

@dataclass
class DailyBoard:
    """Output of the parlay builder for a single day."""
    primary_parlays: list[dict[str, Any]] = field(default_factory=list)
    secondary_parlays: list[dict[str, Any]] = field(default_factory=list)
    singles: list[dict[str, Any]] = field(default_factory=list)
    all_candidates: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_daily_board(
    candidates: pd.DataFrame,
    *,
    config: ParlayConfig | None = None,
) -> DailyBoard:
    """Build the daily parlay + singles board from the candidate pool.

    Parameters
    ----------
    candidates : pd.DataFrame
        The post-selection candidate pool (output of compute_final_board or
        the precision-pool ranked candidates).
    config : ParlayConfig, optional
        Tunable parameters.  Defaults are calibrated from historical data.

    Returns
    -------
    DailyBoard with primary_parlays, secondary_parlays, singles, and diagnostics.
    """
    cfg = config or ParlayConfig()
    board = DailyBoard()

    if candidates.empty:
        board.diagnostics = {"status": "empty_candidates"}
        return board

    # --- 0. Apply precision enhancements ---
    try:
        from .precision_enhancements import annotate_precision_enhancements
        candidates = annotate_precision_enhancements(candidates)
    except Exception:
        pass  # graceful fallback if enhancements unavailable

    # --- 0b. Apply precision filter ---
    try:
        from .precision_filter import annotate_precision_filter
        candidates = annotate_precision_filter(candidates)
    except Exception:
        pass  # graceful fallback

    # --- 0c. Apply streak regression ---
    try:
        from .streak_regression import annotate_streak_regression
        candidates = annotate_streak_regression(candidates)
    except Exception:
        pass  # graceful fallback

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
    eligible_legs = [r for r in rows if r["parlay_eligible"]]
    under_legs = [r for r in eligible_legs if str(r.get("direction", "")).upper() == "UNDER"]
    over_legs = [r for r in eligible_legs if str(r.get("direction", "")).upper() == "OVER"]

    # Sort by leg score descending
    under_legs.sort(key=lambda r: r["parlay_leg_score"], reverse=True)
    over_legs.sort(key=lambda r: r["parlay_leg_score"], reverse=True)

    board.diagnostics = {
        "total_candidates": len(rows),
        "eligible_legs": len(eligible_legs),
        "under_legs": len(under_legs),
        "over_legs": len(over_legs),
    }

    # --- 2. Build primary parlays (UNDER-focused) ---
    used_players: set[str] = set()
    used_indices: set[int] = set()

    primary_candidates: list[dict] = []
    for leg_count in range(cfg.primary_max_legs, cfg.primary_min_legs - 1, -1):
        if len(under_legs) < leg_count:
            continue

        best_parlay = None
        best_score = -1.0

        # Try top N legs to keep combinatorics manageable
        pool = under_legs[:min(12, len(under_legs))]

        for combo in combinations(range(len(pool)), leg_count):
            legs = [pool[i] for i in combo]

            # Check same-game constraint
            if cfg.primary_prefer_different_games:
                game_keys = [_game_key(leg) for leg in legs]
                from collections import Counter
                game_counts = Counter(game_keys)
                if any(c > cfg.primary_max_same_game_legs for c in game_counts.values()):
                    continue

            result = _score_parlay(legs, cfg)
            if result["blocked"]:
                continue
            if result["avg_win_rate"] < cfg.primary_min_avg_win_rate:
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

    # Select best primary parlay (prefer 3-leg if score is close to 2-leg)
    if primary_candidates:
        primary_candidates.sort(key=lambda p: p["score"], reverse=True)
        selected = primary_candidates[0]

        # Prefer 3-leg if it exists and score is within 15% of 2-leg
        for candidate in primary_candidates:
            if candidate["leg_count"] == 3 and candidate["score"] >= selected["score"] * 0.85:
                selected = candidate
                break

        board.primary_parlays.append(selected)
        for leg in selected["legs"]:
            used_players.add(str(leg.get("player", "")))

    # --- 3. Build secondary parlay (may include OVER) ---
    if cfg.secondary_enabled and len(eligible_legs) >= cfg.secondary_legs:
        secondary_pool = []
        for leg in eligible_legs:
            player = str(leg.get("player", ""))
            if player in used_players:
                continue
            secondary_pool.append(leg)

        if len(secondary_pool) >= cfg.secondary_legs:
            best_secondary = None
            best_sec_score = -1.0

            pool = secondary_pool[:min(10, len(secondary_pool))]
            for combo in combinations(range(len(pool)), cfg.secondary_legs):
                legs = [pool[i] for i in combo]
                result = _score_parlay(legs, cfg)
                if result["blocked"]:
                    continue
                if result["score"] > best_sec_score:
                    best_sec_score = result["score"]
                    best_secondary = {
                        "type": "secondary",
                        "legs": legs,
                        "leg_count": cfg.secondary_legs,
                        **result,
                    }

            if best_secondary:
                board.secondary_parlays.append(best_secondary)
                for leg in best_secondary["legs"]:
                    used_players.add(str(leg.get("player", "")))

    # --- 4. Build singles board ---
    parlay_players = set()
    for parlay in board.primary_parlays + board.secondary_parlays:
        for leg in parlay.get("legs", []):
            parlay_players.add(str(leg.get("player", "")))

    singles_pool = []
    for r in rows:
        wr = _sf(r.get("expected_win_rate"))
        ev = _sf(r.get("ev"))
        player = str(r.get("player", ""))
        if wr < cfg.singles_min_win_rate:
            continue
        if ev < cfg.singles_min_ev:
            continue
        # Don't duplicate parlay legs as singles (they're already covered)
        if player in parlay_players:
            # Still include but mark as parlay-covered
            r["singles_note"] = "also_in_parlay"
        r["singles_score"] = (
            0.35 * wr
            + 0.25 * min(ev, 0.30)
            + 0.20 * min(_sf(r.get("abs_edge")) / 3.0, 1.0)
            + 0.10 * _sf(r.get("final_confidence"))
            + 0.10 * _sf(r.get("parlay_leg_quality_score", r.get("final_pool_quality_score")), 0.5)
        )
        singles_pool.append(r)

    singles_pool.sort(key=lambda r: r["singles_score"], reverse=True)

    # Deduplicate by player (keep best per player)
    seen_players: set[str] = set()
    for r in singles_pool:
        player = str(r.get("player", ""))
        if player in seen_players:
            continue
        seen_players.add(player)

        # Assign stake via Kelly sizing
        wr = _sf(r.get("precision_enhanced_win_rate", r.get("expected_win_rate")), 0.5)
        pf_tier = str(r.get("pf_tier", "consider"))
        pf_score = _sf(r.get("pf_score"), 0.5)

        # Add streak regression adjustment to win rate
        streak_adj = _sf(r.get("streak_regression_adj"), 0.0)
        wr_adjusted = float(np.clip(wr + streak_adj, 0.50, 0.95))

        try:
            from .kelly_sizing import compute_stake
            sizing = compute_stake(
                win_probability=wr_adjusted,
                precision_tier=pf_tier,
                precision_score=pf_score,
                bankroll=1000.0,
            )
            r["stake_tier"] = sizing["sizing_tier"]
            r["stake_fraction"] = sizing["stake_fraction"]
            r["kelly_raw"] = sizing["kelly_raw"]
        except Exception:
            # Fallback to simple tier sizing
            if wr >= cfg.single_high_threshold:
                r["stake_tier"] = "high"
                r["stake_fraction"] = cfg.single_high_stake
            elif wr >= cfg.single_med_threshold:
                r["stake_tier"] = "medium"
                r["stake_fraction"] = cfg.single_med_stake
            else:
                r["stake_tier"] = "low"
                r["stake_fraction"] = cfg.single_low_stake

        board.singles.append(r)
        if len(board.singles) >= cfg.singles_max_picks:
            break

    # --- 5. Summary diagnostics ---
    board.diagnostics.update({
        "primary_parlays": len(board.primary_parlays),
        "secondary_parlays": len(board.secondary_parlays),
        "singles": len(board.singles),
        "primary_avg_wr": (
            board.primary_parlays[0]["avg_win_rate"]
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

def format_daily_board(board: DailyBoard, bankroll: float = 1000.0) -> str:
    """Format the daily board as a human-readable string."""
    lines: list[str] = []
    payout_per_unit = 100.0 / 110.0

    lines.append("=" * 70)
    lines.append("DAILY BOARD")
    lines.append("=" * 70)

    # Primary parlays
    for i, parlay in enumerate(board.primary_parlays, 1):
        legs = parlay.get("legs", [])
        n = len(legs)
        parlay_odds_mult = (1 + payout_per_unit) ** n
        stake = bankroll * (parlay.get("stake_fraction", 0.02) if "stake_fraction" in parlay else 0.02)
        potential_profit = stake * (parlay_odds_mult - 1)

        lines.append(f"\n{'─' * 70}")
        lines.append(f"PRIMARY PARLAY #{i}  ({n}-leg)")
        lines.append(f"  Joint probability: {parlay.get('joint_prob', 0):.1%}")
        lines.append(f"  Adjusted probability: {parlay.get('adjusted_prob', 0):.1%}")
        lines.append(f"  Avg leg win rate: {parlay.get('avg_win_rate', 0):.1%}")
        lines.append(f"  Game diversity: {parlay.get('n_games', 0)} different games")
        lines.append(f"  Stake: ${stake:.0f} → potential profit: ${potential_profit:.0f}")
        lines.append(f"  {'─' * 60}")

        for j, leg in enumerate(legs, 1):
            player = leg.get("player", "?")
            target = leg.get("target", "?")
            direction = leg.get("direction", "?")
            line = leg.get("market_line", "?")
            wr = _sf(leg.get("expected_win_rate"))
            edge = _sf(leg.get("abs_edge"))
            lines.append(f"  Leg {j}: {player} {target} {direction} {line}  (WR: {wr:.1%}, edge: {edge:.1f})")

    # Secondary parlays
    for i, parlay in enumerate(board.secondary_parlays, 1):
        legs = parlay.get("legs", [])
        n = len(legs)
        parlay_odds_mult = (1 + payout_per_unit) ** n
        stake = bankroll * 0.015
        potential_profit = stake * (parlay_odds_mult - 1)

        lines.append(f"\n{'─' * 70}")
        lines.append(f"SECONDARY PARLAY #{i}  ({n}-leg)")
        lines.append(f"  Joint probability: {parlay.get('joint_prob', 0):.1%}")
        lines.append(f"  Stake: ${stake:.0f} → potential profit: ${potential_profit:.0f}")
        lines.append(f"  {'─' * 60}")

        for j, leg in enumerate(legs, 1):
            player = leg.get("player", "?")
            target = leg.get("target", "?")
            direction = leg.get("direction", "?")
            line = leg.get("market_line", "?")
            wr = _sf(leg.get("expected_win_rate"))
            lines.append(f"  Leg {j}: {player} {target} {direction} {line}  (WR: {wr:.1%})")

    # Singles
    if board.singles:
        lines.append(f"\n{'─' * 70}")
        lines.append("SINGLES BOARD")
        lines.append(f"  {'─' * 60}")

        for i, pick in enumerate(board.singles, 1):
            player = pick.get("player", "?")
            target = pick.get("target", "?")
            direction = pick.get("direction", "?")
            line = pick.get("market_line", "?")
            wr = _sf(pick.get("expected_win_rate"))
            ev = _sf(pick.get("ev"))
            tier = pick.get("stake_tier", "?")
            frac = pick.get("stake_fraction", 0)
            stake = bankroll * frac
            note = f"  [{pick['singles_note']}]" if pick.get("singles_note") else ""
            lines.append(
                f"  {i}. {player} {target} {direction} {line}  "
                f"WR: {wr:.1%} | EV: {ev:+.3f} | {tier} ${stake:.0f}{note}"
            )

    # Diagnostics
    lines.append(f"\n{'─' * 70}")
    lines.append("DIAGNOSTICS")
    diag = board.diagnostics
    lines.append(f"  Candidates: {diag.get('total_candidates', 0)}")
    lines.append(f"  Eligible parlay legs: {diag.get('eligible_legs', 0)} (UNDER: {diag.get('under_legs', 0)}, OVER: {diag.get('over_legs', 0)})")
    lines.append(f"  Primary parlays: {diag.get('primary_parlays', 0)}")
    lines.append(f"  Secondary parlays: {diag.get('secondary_parlays', 0)}")
    lines.append(f"  Singles: {diag.get('singles', 0)}")

    return "\n".join(lines)
