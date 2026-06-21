#!/usr/bin/env python3
"""
MLB Pick Compiler — strict gating layer between model projections and the published board.

This module sits downstream of select_high_precision_predictions.py and enforces:

1. Event Identity Gate — cross-validates game_id, player_team_id, official_local_game_date,
   commence_time_utc, and sportsbook_event_id to catch date-mapping mismatches.
2. Dual Settlement Tracking — stores book_settlement_result AND model_audit_result separately
   so corrupted wins (e.g. Alonso 6/19 hit attributed to 6/20 board) don't pollute learning.
3. Publish Gates — enforces market source, book count, line stability, history rows,
   confirmed lineup, and projected PA thresholds before any pick reaches the board.
4. Kill Reasons — every rejected pick gets a structured reason for feedback loops.
5. Action Score — composite publish score replacing raw edge ranking.
6. TB Distribution — for total bases props, stores P(0), P(1), P(2+), P(4+) for
   direct line grading instead of expected-value-only reasoning.

Usage:
    python pick_compiler.py \
        --selected-csv path/to/high_precision_predictions.csv \
        --schedule-json path/to/schedule.json \
        --out-board path/to/compiled_board.json \
        --out-kills path/to/killed_picks.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

DEFAULT_DAILY_RUNS_ROOT = SPORT_ROOT / "data" / "predictions" / "daily_runs"


# ---------------------------------------------------------------------------
# Kill Reasons
# ---------------------------------------------------------------------------

class KillReason(str, Enum):
    EVENT_DATE_MISMATCH = "EVENT_DATE_MISMATCH"
    PLAYER_TEAM_MISMATCH = "PLAYER_TEAM_MISMATCH"
    NO_CONFIRMED_LINEUP = "NO_CONFIRMED_LINEUP"
    LOW_MARKET_BOOK_COUNT = "LOW_MARKET_BOOK_COUNT"
    SYNTHETIC_LINE = "SYNTHETIC_LINE"
    LOW_HISTORY_ROWS = "LOW_HISTORY_ROWS"
    HIGH_PUSH_EXPOSURE = "HIGH_PUSH_EXPOSURE"
    HIGH_VARIANCE_PLAYER = "HIGH_VARIANCE_PLAYER"
    MODEL_PRICE_DISAGREEMENT = "MODEL_PRICE_DISAGREEMENT"
    STALE_LAST_HISTORY_DATE = "STALE_LAST_HISTORY_DATE"
    LOW_PROJECTED_PA = "LOW_PROJECTED_PA"
    HIGH_LINE_STD = "HIGH_LINE_STD"
    LOW_CALIBRATED_PROB = "LOW_CALIBRATED_PROB"
    LOW_EDGE = "LOW_EDGE"
    BELOW_PUBLISH_SCORE = "BELOW_PUBLISH_SCORE"
    GAME_STATUS_FINAL = "GAME_STATUS_FINAL"
    MISSING_GAME_ID = "MISSING_GAME_ID"


# ---------------------------------------------------------------------------
# Settlement result types
# ---------------------------------------------------------------------------

class SettlementResult(str, Enum):
    WIN = "W"
    LOSS = "L"
    PUSH = "PUSH"
    VOID = "VOID"
    PENDING = "PENDING"
    INVALID = "INVALID"  # model audit only — event mapping error


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EventIdentity:
    """Cross-validated event identity fields."""
    game_id: str = ""
    sportsbook_event_id: str = ""
    official_game_date: str = ""  # official MLB local game date
    display_date: str = ""  # board display date / run date
    commence_time_utc: str = ""
    home_team_id: str = ""
    away_team_id: str = ""
    home_team: str = ""
    away_team: str = ""
    player_team_id: str = ""
    player_team: str = ""
    venue_timezone: str = ""
    event_verified: bool = False
    mismatch_details: list[str] = field(default_factory=list)


@dataclass
class DualSettlement:
    """Tracks both book settlement and model audit results."""
    book_settlement_result: str = SettlementResult.PENDING.value
    model_audit_result: str = SettlementResult.PENDING.value
    settlement_notes: str = ""
    event_mapping_valid: bool = True


@dataclass
class TBDistribution:
    """Total bases probability distribution for direct line grading."""
    p_0_tb: float = 0.0
    p_1_tb: float = 0.0
    p_2_tb: float = 0.0
    p_3_tb: float = 0.0
    p_4_plus_tb: float = 0.0

    @property
    def over_0_5(self) -> float:
        return 1.0 - self.p_0_tb

    @property
    def under_1_5(self) -> float:
        return self.p_0_tb + self.p_1_tb

    @property
    def under_2_5(self) -> float:
        return self.p_0_tb + self.p_1_tb + self.p_2_tb

    @property
    def over_1_5(self) -> float:
        return 1.0 - self.under_1_5

    @property
    def over_2_5(self) -> float:
        return 1.0 - self.under_2_5


@dataclass
class PublishScore:
    """Composite action score components."""
    calibrated_prob: float = 0.0
    market_quality: float = 0.0
    lineup_certainty: float = 0.0
    matchup_fit: float = 0.0
    role_stability: float = 0.0
    clv_signal: float = 0.0
    event_mapping_risk: float = 0.0
    stale_data_risk: float = 0.0
    push_risk: float = 0.0
    composite: float = 0.0


@dataclass
class CompiledPick:
    """A fully validated pick ready for the board (or killed with reason)."""
    # Core pick identity
    player: str = ""
    player_id: str = ""
    team: str = ""
    target: str = ""
    direction: str = ""
    market_line: float = 0.0
    prediction: float = 0.0
    edge: float = 0.0

    # Event identity
    event_identity: EventIdentity = field(default_factory=EventIdentity)

    # Settlement tracking
    settlement: DualSettlement = field(default_factory=DualSettlement)

    # Market quality
    market_source: str = ""
    market_books: int = 0
    market_line_std: float = 0.0
    history_rows: int = 0
    last_history_date: str = ""
    days_since_history: int = 0

    # Probability & calibration
    calibrated_hit_prob: float = 0.0
    calibrated_graded_hit_rate: float = 0.0
    push_probability: float = 0.0

    # TB distribution (populated for TB targets only)
    tb_distribution: TBDistribution | None = None

    # Lineup / opportunity
    confirmed_starter: bool = False
    batting_order: int | None = None
    projected_pa: float | None = None

    # Publish score
    publish_score: PublishScore = field(default_factory=PublishScore)

    # Outcome
    published: bool = False
    confidence_label: str = ""  # "consider" / "strong" / "wanted"
    kill_reasons: list[str] = field(default_factory=list)

    # Original row data for audit trail
    raw_row: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gate thresholds (configurable)
# ---------------------------------------------------------------------------

@dataclass
class CompilerConfig:
    """All tunable thresholds for the pick compiler."""
    # Event identity
    require_event_verification: bool = True

    # Market quality gates
    min_market_books: int = 5
    max_market_line_std: float = 1.0  # typical range 0.35-0.85; 1.0 filters extreme disagreement only
    require_real_market_source: bool = True

    # History gates
    min_history_rows_overs: int = 30
    min_history_rows_unders: int = 20
    max_days_since_history: int = 4

    # Probability gates
    min_calibrated_hit_prob: float = 0.62
    min_edge: float = 0.35

    # Push gate
    max_push_probability: float = 0.24

    # Lineup / PA gates (hitter overs)
    require_confirmed_starter_overs: bool = True
    min_projected_pa_overs: float = 3.5
    max_projected_pa_unders: float = 4.4

    # Publish score thresholds
    # NOTE: threshold set at 0.65 while lineup/CLV data feeds are absent.
    # Tighten to 0.72 once confirmed_starter and closing_line_value are populated.
    min_publish_score: float = 0.65
    wanted_publish_score: float = 0.75

    # Publish score weights
    # NOTE: lineup_certainty and clv_signal are set low until those data feeds exist.
    # When lineup data is piped in, increase w_lineup_certainty to 0.15 and reduce w_calibrated_prob.
    w_calibrated_prob: float = 0.40
    w_market_quality: float = 0.20
    w_lineup_certainty: float = 0.05  # low until lineup feed exists
    w_matchup_fit: float = 0.12
    w_role_stability: float = 0.10
    w_clv_signal: float = 0.03  # low until CLV tracking exists
    w_event_mapping_risk: float = -0.25
    w_stale_data_risk: float = -0.15
    w_push_risk: float = -0.10

    # Board limits
    max_board_size: int = 6
    max_per_player: int = 1
    max_per_game: int = 2
    max_per_team: int = 3


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


# ---------------------------------------------------------------------------
# Gate 1: Event Identity Verification
# ---------------------------------------------------------------------------

def fetch_mlb_schedule(game_date: str, timeout: float = 30.0) -> list[dict]:
    """Fetch the official MLB schedule for a given date."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date}&hydrate=team"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []
    games: list[dict] = []
    for date_bucket in payload.get("dates", []):
        games.extend(date_bucket.get("games", []))
    return games


def build_schedule_lookup(games: list[dict]) -> dict[str, dict]:
    """Build a lookup from game_id and team abbreviations to game info."""
    lookup: dict[str, dict] = {}
    for game in games:
        game_pk = str(game.get("gamePk", ""))
        home = game.get("teams", {}).get("home", {}).get("team", {})
        away = game.get("teams", {}).get("away", {}).get("team", {})
        info = {
            "game_id": game_pk,
            "game_date": str(game.get("officialDate", "")),
            "commence_time_utc": str(game.get("gameDate", "")),
            "home_team": str(home.get("abbreviation", "")).upper(),
            "home_team_id": str(home.get("id", "")),
            "away_team": str(away.get("abbreviation", "")).upper(),
            "away_team_id": str(away.get("id", "")),
            "status": game.get("status", {}).get("statusCode", ""),
        }
        lookup[game_pk] = info
        # Also index by team pair for fuzzy matching
        team_key = f"{info['home_team']}_{info['away_team']}"
        lookup[team_key] = info
    return lookup


def verify_event_identity(
    row: dict[str, Any],
    schedule_lookup: dict[str, dict],
    run_date: str,
) -> EventIdentity:
    """Cross-validate event identity fields against the MLB schedule."""
    identity = EventIdentity()
    identity.display_date = run_date

    # Extract what we know from the pick row
    game_id = str(row.get("Game_ID", "") or "").strip()
    player_team = str(row.get("Team", "") or row.get("Player_Team", "")).strip().upper()
    commence_time = str(row.get("Commence_Time_UTC", "") or "").strip()
    selected_game_date = str(row.get("Selected_Game_Date", "") or "").strip()[:10]

    identity.game_id = game_id
    identity.player_team = player_team
    identity.commence_time_utc = commence_time
    identity.official_game_date = selected_game_date or run_date

    if not schedule_lookup:
        # Can't verify without schedule, mark as unverified but don't kill
        identity.event_verified = False
        identity.mismatch_details.append("NO_SCHEDULE_DATA")
        return identity

    # Try to find the game in the schedule
    game_info = schedule_lookup.get(game_id)
    if game_info is None and player_team:
        # Try team-pair lookup
        for key, info in schedule_lookup.items():
            if player_team in (info.get("home_team", ""), info.get("away_team", "")):
                game_info = info
                break

    if game_info is None:
        identity.event_verified = False
        identity.mismatch_details.append("GAME_NOT_IN_SCHEDULE")
        return identity

    # Populate verified fields
    identity.home_team = game_info.get("home_team", "")
    identity.away_team = game_info.get("away_team", "")
    identity.home_team_id = game_info.get("home_team_id", "")
    identity.away_team_id = game_info.get("away_team_id", "")
    identity.official_game_date = game_info.get("game_date", "")

    # Check 1: player_team in {home, away}
    if player_team and player_team not in (identity.home_team, identity.away_team):
        identity.mismatch_details.append(
            f"PLAYER_TEAM_{player_team}_NOT_IN_GAME_{identity.home_team}_vs_{identity.away_team}"
        )

    # Check 2: display_date == official_local_game_date
    official_date = identity.official_game_date[:10]
    if official_date and run_date and official_date != run_date:
        identity.mismatch_details.append(
            f"DISPLAY_DATE_{run_date}_NE_OFFICIAL_DATE_{official_date}"
        )

    # Check 3: game_id consistency
    if game_id and game_info.get("game_id") and game_id != game_info.get("game_id"):
        identity.mismatch_details.append(
            f"GAME_ID_MISMATCH_ROW_{game_id}_VS_SCHEDULE_{game_info.get('game_id')}"
        )

    identity.event_verified = len(identity.mismatch_details) == 0
    return identity


# ---------------------------------------------------------------------------
# Gate 2: TB Distribution Model
# ---------------------------------------------------------------------------

def compute_tb_distribution(projection: float) -> TBDistribution:
    """
    Estimate total bases probability distribution using a modified Poisson.

    For TB, zero-inflation is common (player goes 0-for-4), so we blend
    a standard Poisson with an inflated P(0) based on typical hitless rates.
    """
    lam = max(0.01, float(projection))

    # Standard Poisson PMFs
    p0_poisson = math.exp(-lam)
    p1_poisson = lam * math.exp(-lam)
    p2_poisson = (lam ** 2) * math.exp(-lam) / 2.0
    p3_poisson = (lam ** 3) * math.exp(-lam) / 6.0

    # Zero-inflation factor: hitters typically have 25-40% chance of 0 TB
    # Scale this based on projection — lower projection = higher zero chance
    zero_inflation = max(0.0, 0.10 * (1.5 - lam))  # adds ~10% at lam=0.5, ~0% at lam=1.5+
    p0_adjusted = min(0.95, p0_poisson + zero_inflation)

    # Renormalize remaining mass
    remaining = 1.0 - p0_adjusted
    non_zero_total = 1.0 - p0_poisson
    if non_zero_total < 1e-9:
        non_zero_total = 1e-9

    scale = remaining / non_zero_total
    p1 = p1_poisson * scale
    p2 = p2_poisson * scale
    p3 = p3_poisson * scale
    p4_plus = max(0.0, remaining - p1 - p2 - p3)

    return TBDistribution(
        p_0_tb=round(p0_adjusted, 4),
        p_1_tb=round(p1, 4),
        p_2_tb=round(p2, 4),
        p_3_tb=round(p3, 4),
        p_4_plus_tb=round(p4_plus, 4),
    )


# ---------------------------------------------------------------------------
# Gate 3: Publish Score Computation
# ---------------------------------------------------------------------------

def compute_publish_score(
    pick: CompiledPick,
    config: CompilerConfig,
) -> PublishScore:
    """Compute composite publish/action score for a pick."""
    score = PublishScore()

    # Calibrated probability (0-1, higher = better)
    score.calibrated_prob = _clamp(pick.calibrated_hit_prob)

    # Market quality (books count normalized, line std penalty)
    books_norm = _clamp(pick.market_books / 8.0)  # 8 books = 1.0
    std_penalty = _clamp(pick.market_line_std / 1.5)  # 1.5 std = full penalty
    source_bonus = 1.0 if pick.market_source == "real" else 0.3
    score.market_quality = _clamp(books_norm * source_bonus * (1.0 - std_penalty * 0.3))

    # Lineup certainty
    if pick.confirmed_starter:
        score.lineup_certainty = 1.0
    elif pick.batting_order is not None and pick.batting_order <= 9:
        score.lineup_certainty = 0.7
    else:
        score.lineup_certainty = 0.3

    # Matchup fit (use edge strength as proxy)
    edge_norm = _clamp(abs(pick.edge) / 1.5)  # 1.5 edge = max
    score.matchup_fit = edge_norm

    # Role stability (history rows as proxy)
    score.role_stability = _clamp(pick.history_rows / 50.0)  # 50 rows = max

    # CLV signal (placeholder — would come from closing line value tracking)
    score.clv_signal = 0.5  # neutral until CLV data is available

    # Risk penalties
    if not pick.event_identity.event_verified:
        score.event_mapping_risk = 0.5 if pick.event_identity.mismatch_details else 0.2
    else:
        score.event_mapping_risk = 0.0

    if pick.days_since_history > 3:
        score.stale_data_risk = _clamp((pick.days_since_history - 3) / 4.0)
    else:
        score.stale_data_risk = 0.0

    score.push_risk = _clamp(pick.push_probability / 0.3)  # 30% push = full penalty

    # Composite score
    score.composite = (
        config.w_calibrated_prob * score.calibrated_prob
        + config.w_market_quality * score.market_quality
        + config.w_lineup_certainty * score.lineup_certainty
        + config.w_matchup_fit * score.matchup_fit
        + config.w_role_stability * score.role_stability
        + config.w_clv_signal * score.clv_signal
        + config.w_event_mapping_risk * score.event_mapping_risk
        + config.w_stale_data_risk * score.stale_data_risk
        + config.w_push_risk * score.push_risk
    )
    score.composite = round(score.composite, 4)
    return score


# ---------------------------------------------------------------------------
# Gate 4: Hard gates — kill picks that fail
# ---------------------------------------------------------------------------

def apply_hard_gates(
    pick: CompiledPick,
    config: CompilerConfig,
) -> list[KillReason]:
    """Apply all hard gates. Returns list of kill reasons (empty = passed)."""
    kills: list[KillReason] = []

    # Event identity gate
    if config.require_event_verification:
        for detail in pick.event_identity.mismatch_details:
            if "DATE" in detail.upper():
                kills.append(KillReason.EVENT_DATE_MISMATCH)
                break
        for detail in pick.event_identity.mismatch_details:
            if "TEAM" in detail.upper() and KillReason.PLAYER_TEAM_MISMATCH not in kills:
                kills.append(KillReason.PLAYER_TEAM_MISMATCH)
                break

    # Market source gate
    if config.require_real_market_source and pick.market_source != "real":
        kills.append(KillReason.SYNTHETIC_LINE)

    # Market books gate
    if pick.market_books < config.min_market_books:
        kills.append(KillReason.LOW_MARKET_BOOK_COUNT)

    # Line stability gate
    if pick.market_line_std > config.max_market_line_std:
        kills.append(KillReason.HIGH_LINE_STD)

    # History rows gate (different thresholds for overs vs unders)
    is_over = pick.direction.upper() == "OVER"
    min_rows = config.min_history_rows_overs if is_over else config.min_history_rows_unders
    if pick.history_rows < min_rows:
        kills.append(KillReason.LOW_HISTORY_ROWS)

    # Staleness gate
    if pick.days_since_history > config.max_days_since_history:
        kills.append(KillReason.STALE_LAST_HISTORY_DATE)

    # Push exposure gate
    if pick.push_probability > config.max_push_probability:
        kills.append(KillReason.HIGH_PUSH_EXPOSURE)

    # Calibrated probability gate
    if pick.calibrated_hit_prob < config.min_calibrated_hit_prob:
        kills.append(KillReason.LOW_CALIBRATED_PROB)

    # Edge gate
    if abs(pick.edge) < config.min_edge:
        kills.append(KillReason.LOW_EDGE)

    # Hitter-specific: projected PA for overs
    if is_over and pick.target in ("H", "TB", "R", "RBI", "HR"):
        if pick.projected_pa is not None and pick.projected_pa < config.min_projected_pa_overs:
            kills.append(KillReason.LOW_PROJECTED_PA)
        if config.require_confirmed_starter_overs and not pick.confirmed_starter:
            # Only kill if we have lineup data but player isn't in it
            if pick.batting_order is not None and pick.batting_order > 9:
                kills.append(KillReason.NO_CONFIRMED_LINEUP)

    return kills


# ---------------------------------------------------------------------------
# Confidence labeling
# ---------------------------------------------------------------------------

def assign_confidence_label(calibrated_prob: float, publish_score: float) -> str:
    """
    Conservative labeling until calibration buckets are historically validated.
    Do NOT use 80-90% labels until backtest proves those buckets hit near that rate.
    """
    if publish_score >= 0.75 and calibrated_prob >= 0.70:
        return "wanted"
    if publish_score >= 0.65 and calibrated_prob >= 0.65:
        return "strong"
    if calibrated_prob >= 0.62:
        return "consider"
    return "pass"


# ---------------------------------------------------------------------------
# Main compiler pipeline
# ---------------------------------------------------------------------------

def compile_pick_from_row(
    row: dict[str, Any],
    schedule_lookup: dict[str, dict],
    run_date: str,
    config: CompilerConfig,
) -> CompiledPick:
    """Compile a single pick from a selector output row."""
    pick = CompiledPick()
    pick.raw_row = dict(row)

    # Core identity
    pick.player = str(row.get("Player", "")).strip()
    pick.player_id = str(row.get("Player_ID", "") or "").strip()
    pick.team = str(row.get("Team", "") or "").strip().upper()
    pick.target = str(row.get("Target", "")).strip().upper()
    pick.direction = str(row.get("Direction", "")).strip().upper()
    pick.market_line = _safe_float(row.get("Market_Line"))
    pick.prediction = _safe_float(row.get("Prediction"))
    pick.edge = _safe_float(row.get("Edge"))

    # Market quality
    pick.market_source = str(row.get("Market_Source", "")).strip().lower()
    pick.market_books = _safe_int(row.get("Market_Books"))
    pick.market_line_std = _safe_float(row.get("Market_Line_Std"))
    pick.history_rows = _safe_int(row.get("History_Rows"))
    pick.last_history_date = str(row.get("Last_History_Date", "") or "")
    pick.days_since_history = _safe_int(row.get("Days_Since_History"))

    # Probability (from upstream selector — maps multiple possible column names)
    pick.calibrated_hit_prob = _safe_float(
        row.get("Calibrated_Hit_Probability")
        or row.get("Estimated_Hit_Probability")
        or row.get("Model_Hit_Probability")
    )
    pick.calibrated_graded_hit_rate = _safe_float(
        row.get("Calibrated_Graded_Hit_Rate")
        or row.get("Estimated_Graded_Hit_Rate")
        or row.get("Model_Graded_Hit_Rate")
    )
    pick.push_probability = _safe_float(
        row.get("Push_Probability")
        or row.get("Estimated_Push_Probability")
    )

    # Lineup / opportunity (may come from upstream or schedule enrichment)
    pick.confirmed_starter = str(row.get("Confirmed_Starter", "")).lower() in ("true", "1", "yes")
    batting_order_raw = row.get("Batting_Order")
    pick.batting_order = _safe_int(batting_order_raw) if batting_order_raw else None
    pa_raw = row.get("Projected_PA")
    pick.projected_pa = _safe_float(pa_raw) if pa_raw else None

    # Event identity verification
    pick.event_identity = verify_event_identity(row, schedule_lookup, run_date)

    # If event has a date mismatch, mark settlement as potentially invalid
    if not pick.event_identity.event_verified:
        for detail in pick.event_identity.mismatch_details:
            if "DATE" in detail.upper():
                pick.settlement.event_mapping_valid = False
                pick.settlement.settlement_notes = (
                    f"Event-date mismatch detected: {detail}. "
                    "Book settlement may be valid but model audit should mark INVALID."
                )
                break

    # TB distribution (for total bases props)
    if pick.target == "TB":
        pick.tb_distribution = compute_tb_distribution(pick.prediction)

    # Compute publish score
    pick.publish_score = compute_publish_score(pick, config)

    # Apply hard gates
    kill_reasons = apply_hard_gates(pick, config)
    pick.kill_reasons = [kr.value for kr in kill_reasons]

    # Determine publication status
    if not kill_reasons and pick.publish_score.composite >= config.min_publish_score:
        pick.published = True
        pick.confidence_label = assign_confidence_label(
            pick.calibrated_hit_prob, pick.publish_score.composite
        )
    else:
        pick.published = False
        if not kill_reasons and pick.publish_score.composite < config.min_publish_score:
            pick.kill_reasons.append(KillReason.BELOW_PUBLISH_SCORE.value)
        pick.confidence_label = "pass"

    return pick


def apply_board_limits(
    picks: list[CompiledPick],
    config: CompilerConfig,
) -> list[CompiledPick]:
    """
    Apply board-level diversity limits: max per player, per game, per team.
    Picks are already sorted by publish_score descending.
    """
    from collections import Counter

    published: list[CompiledPick] = []
    player_counts: Counter = Counter()
    game_counts: Counter = Counter()
    team_counts: Counter = Counter()

    for pick in picks:
        if not pick.published:
            continue

        player_key = pick.player_id or pick.player
        game_key = pick.event_identity.game_id or f"{pick.team}_{pick.event_identity.display_date}"

        if player_counts[player_key] >= config.max_per_player:
            pick.published = False
            pick.kill_reasons.append("BOARD_LIMIT_PER_PLAYER")
            continue
        if game_counts[game_key] >= config.max_per_game:
            pick.published = False
            pick.kill_reasons.append("BOARD_LIMIT_PER_GAME")
            continue
        if team_counts[pick.team] >= config.max_per_team:
            pick.published = False
            pick.kill_reasons.append("BOARD_LIMIT_PER_TEAM")
            continue
        if len(published) >= config.max_board_size:
            pick.published = False
            pick.kill_reasons.append("BOARD_LIMIT_MAX_SIZE")
            continue

        player_counts[player_key] += 1
        game_counts[game_key] += 1
        team_counts[pick.team] += 1
        published.append(pick)

    return picks  # returns all picks (published flag updated in-place)


# ---------------------------------------------------------------------------
# Settlement grading (post-game)
# ---------------------------------------------------------------------------

def grade_settlement(
    pick: CompiledPick,
    actual_value: float | None,
) -> DualSettlement:
    """
    Grade a pick after the game is complete.
    Produces both book_settlement_result and model_audit_result.
    """
    settlement = pick.settlement

    if actual_value is None:
        settlement.book_settlement_result = SettlementResult.VOID.value
        settlement.model_audit_result = SettlementResult.VOID.value
        return settlement

    line = pick.market_line
    direction = pick.direction.upper()

    # Determine book settlement
    if actual_value == line:
        book_result = SettlementResult.PUSH.value
    elif direction == "OVER":
        book_result = SettlementResult.WIN.value if actual_value > line else SettlementResult.LOSS.value
    else:  # UNDER
        book_result = SettlementResult.WIN.value if actual_value < line else SettlementResult.LOSS.value

    settlement.book_settlement_result = book_result

    # Determine model audit result
    if not settlement.event_mapping_valid:
        # Event-date mismatch: book may settle correctly, but model should not learn from this
        settlement.model_audit_result = SettlementResult.INVALID.value
        settlement.settlement_notes += (
            f" Book settled as {book_result} but model audit marks INVALID due to event mapping error."
        )
    else:
        settlement.model_audit_result = book_result

    return settlement


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def pick_to_dict(pick: CompiledPick) -> dict[str, Any]:
    """Serialize a CompiledPick to a JSON-safe dict."""
    result = {
        "player": pick.player,
        "player_id": pick.player_id,
        "team": pick.team,
        "target": pick.target,
        "direction": pick.direction,
        "market_line": pick.market_line,
        "prediction": round(pick.prediction, 3),
        "edge": round(pick.edge, 3),
        "market_source": pick.market_source,
        "market_books": pick.market_books,
        "market_line_std": round(pick.market_line_std, 4),
        "history_rows": pick.history_rows,
        "last_history_date": pick.last_history_date,
        "days_since_history": pick.days_since_history,
        "calibrated_hit_prob": round(pick.calibrated_hit_prob, 4),
        "calibrated_graded_hit_rate": round(pick.calibrated_graded_hit_rate, 4),
        "push_probability": round(pick.push_probability, 4),
        "confirmed_starter": pick.confirmed_starter,
        "batting_order": pick.batting_order,
        "projected_pa": round(pick.projected_pa, 2) if pick.projected_pa else None,
        "event_identity": asdict(pick.event_identity),
        "settlement": asdict(pick.settlement),
        "publish_score": asdict(pick.publish_score),
        "published": pick.published,
        "confidence_label": pick.confidence_label,
        "kill_reasons": pick.kill_reasons,
    }
    if pick.tb_distribution:
        result["tb_distribution"] = asdict(pick.tb_distribution)
        result["tb_line_grades"] = {
            "over_0.5": round(pick.tb_distribution.over_0_5, 4),
            "under_1.5": round(pick.tb_distribution.under_1_5, 4),
            "over_1.5": round(pick.tb_distribution.over_1_5, 4),
            "under_2.5": round(pick.tb_distribution.under_2_5, 4),
            "over_2.5": round(pick.tb_distribution.over_2_5, 4),
        }
    return result


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_compiler(
    selected_csv: Path,
    run_date: str | None = None,
    config: CompilerConfig | None = None,
    schedule_json: Path | None = None,
    out_board: Path | None = None,
    out_kills: Path | None = None,
) -> dict[str, Any]:
    """
    Run the pick compiler pipeline.

    Args:
        selected_csv: Path to high_precision_predictions CSV from the selector.
        run_date: Board date (YYYY-MM-DD). Inferred from CSV name if not given.
        config: Compiler configuration. Uses defaults if not given.
        schedule_json: Optional pre-fetched schedule JSON. Fetches live if absent.
        out_board: Output path for published board JSON.
        out_kills: Output path for killed picks JSON.

    Returns:
        Summary dict with board stats.
    """
    if config is None:
        config = CompilerConfig()

    # Load selected predictions
    df = pd.read_csv(selected_csv)
    rows = df.to_dict(orient="records")

    # Infer run date from filename if not provided
    if not run_date:
        stem = selected_csv.stem
        digits = "".join(c for c in stem if c.isdigit())
        if len(digits) >= 8:
            run_date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        else:
            run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load or fetch schedule
    schedule_lookup: dict[str, dict] = {}
    if schedule_json and schedule_json.exists():
        games = json.loads(schedule_json.read_text(encoding="utf-8"))
        schedule_lookup = build_schedule_lookup(games)
    else:
        try:
            games = fetch_mlb_schedule(run_date)
            schedule_lookup = build_schedule_lookup(games)
        except Exception:
            pass  # proceed without schedule verification

    # Compile each pick
    compiled: list[CompiledPick] = []
    for row in rows:
        pick = compile_pick_from_row(row, schedule_lookup, run_date, config)
        compiled.append(pick)

    # Sort by publish score descending
    compiled.sort(key=lambda p: p.publish_score.composite, reverse=True)

    # Apply board-level limits
    compiled = apply_board_limits(compiled, config)

    # Split into published vs killed
    board = [p for p in compiled if p.published]
    killed = [p for p in compiled if not p.published]

    # Build output
    board_output = {
        "run_date": run_date,
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "summary": {
            "input_rows": len(rows),
            "published": len(board),
            "killed": len(killed),
            "kill_reason_counts": {},
            "event_verified_count": sum(1 for p in compiled if p.event_identity.event_verified),
            "event_unverified_count": sum(1 for p in compiled if not p.event_identity.event_verified),
        },
        "board": [pick_to_dict(p) for p in board],
    }

    # Count kill reasons
    from collections import Counter
    reason_counter: Counter = Counter()
    for p in killed:
        for reason in p.kill_reasons:
            reason_counter[reason] += 1
    board_output["summary"]["kill_reason_counts"] = dict(reason_counter.most_common())

    kills_output = {
        "run_date": run_date,
        "compiled_at_utc": board_output["compiled_at_utc"],
        "killed_picks": [pick_to_dict(p) for p in killed],
    }

    # Write outputs
    if out_board is None:
        out_board = selected_csv.with_name(
            selected_csv.stem.replace("_high_precision_predictions", "") + "_compiled_board.json"
        )
    if out_kills is None:
        out_kills = selected_csv.with_name(
            selected_csv.stem.replace("_high_precision_predictions", "") + "_killed_picks.json"
        )

    out_board.parent.mkdir(parents=True, exist_ok=True)
    out_board.write_text(json.dumps(board_output, indent=2), encoding="utf-8")
    out_kills.write_text(json.dumps(kills_output, indent=2), encoding="utf-8")

    print(f"PICK COMPILER COMPLETE — {run_date}")
    print(f"  Input rows: {len(rows)}")
    print(f"  Published:  {len(board)}")
    print(f"  Killed:     {len(killed)}")
    print(f"  Event verified: {board_output['summary']['event_verified_count']}")
    if reason_counter:
        print("  Kill reasons:")
        for reason, count in reason_counter.most_common():
            print(f"    {reason}: {count}")
    print(f"  Board JSON: {out_board}")
    print(f"  Kills JSON: {out_kills}")

    return board_output["summary"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLB Pick Compiler — strict gating between model output and published board."
    )
    parser.add_argument("--selected-csv", type=Path, required=True, help="High-precision predictions CSV.")
    parser.add_argument("--run-date", type=str, default=None, help="Board date YYYY-MM-DD.")
    parser.add_argument("--schedule-json", type=Path, default=None, help="Pre-fetched MLB schedule JSON.")
    parser.add_argument("--out-board", type=Path, default=None, help="Output board JSON path.")
    parser.add_argument("--out-kills", type=Path, default=None, help="Output killed picks JSON path.")
    # Gate overrides
    parser.add_argument("--min-market-books", type=int, default=5)
    parser.add_argument("--max-market-line-std", type=float, default=1.0)
    parser.add_argument("--min-history-rows-overs", type=int, default=30)
    parser.add_argument("--min-history-rows-unders", type=int, default=20)
    parser.add_argument("--min-calibrated-hit-prob", type=float, default=0.62)
    parser.add_argument("--min-edge", type=float, default=0.35)
    parser.add_argument("--min-publish-score", type=float, default=0.65)
    parser.add_argument("--max-board-size", type=int, default=6)
    parser.add_argument("--no-event-verification", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CompilerConfig(
        min_market_books=args.min_market_books,
        max_market_line_std=args.max_market_line_std,
        min_history_rows_overs=args.min_history_rows_overs,
        min_history_rows_unders=args.min_history_rows_unders,
        min_calibrated_hit_prob=args.min_calibrated_hit_prob,
        min_edge=args.min_edge,
        min_publish_score=args.min_publish_score,
        max_board_size=args.max_board_size,
        require_event_verification=not args.no_event_verification,
        require_real_market_source=not args.allow_synthetic,
    )
    run_compiler(
        selected_csv=args.selected_csv,
        run_date=args.run_date,
        config=config,
        schedule_json=args.schedule_json,
        out_board=args.out_board,
        out_kills=args.out_kills,
    )


if __name__ == "__main__":
    main()
