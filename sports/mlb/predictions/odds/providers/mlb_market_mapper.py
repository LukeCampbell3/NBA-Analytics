#!/usr/bin/env python3
"""
MLB Market Mapper

Resolves raw SportsGameOdds statIDs into canonical market types using
player role context. Handles ambiguous markets explicitly.

Ambiguity rules:
  H: batter_hits if batter context, pitcher_hits_allowed if pitcher context
  K: pitcher_strikeouts if pitcher context, batter_strikeouts if batter context
  RBI: always batter_rbis (only batters have RBI props)

Role detection priority:
  1. Explicit player position/role from provider metadata
  2. Known starting pitcher list from event
  3. Player team relation + stat line range heuristic
  4. diagnostic_only when confidence is low

Every mapped row includes:
  raw_stat_id, canonical_market_type, market_mapping_confidence, market_mapping_reason
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set


# Canonical market types
SUPPORTED_CANONICAL_MARKETS = {
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs",
    "batter_home_runs",
    "batter_strikeouts",
    "batter_stolen_bases",
    "batter_walks",
    "batter_singles",
    "batter_doubles",
    "batter_triples",
    "batter_hits_runs_rbis",
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_earned_runs",
    "pitcher_walks_allowed",
    "pitcher_outs_recorded",
    "pitcher_pitches_thrown",
}

# Unambiguous statID → canonical (no role context needed)
UNAMBIGUOUS_MAP = {
    # Real SportsGameOdds v2 statIDs (camelCase with batting_/pitching_ prefix)
    "batting_hits": "batter_hits",
    "batting_totalBases": "batter_total_bases",
    "batting_RBI": "batter_rbis",
    "batting_rbi": "batter_rbis",
    "batting_homeRuns": "batter_home_runs",
    "batting_strikeouts": "batter_strikeouts",
    "batting_stolenBases": "batter_stolen_bases",
    "batting_basesOnBalls": "batter_walks",
    "batting_singles": "batter_singles",
    "batting_doubles": "batter_doubles",
    "batting_triples": "batter_triples",
    "batting_hits+runs+rbi": "batter_hits_runs_rbis",
    "pitching_strikeouts": "pitcher_strikeouts",
    "pitching_hits": "pitcher_hits_allowed",
    "pitching_earnedRuns": "pitcher_earned_runs",
    "pitching_outs": "pitcher_outs_recorded",
    "pitching_basesOnBalls": "pitcher_walks_allowed",
    "pitching_pitchesThrown": "pitcher_pitches_thrown",
    "points": "batter_runs",
    # Legacy/simple statIDs (from earlier API versions or other providers)
    "RBI": "batter_rbis",
    "rbi": "batter_rbis",
    "rbis": "batter_rbis",
    "TB": "batter_total_bases",
    "total_bases": "batter_total_bases",
    "R": "batter_runs",
    "runs": "batter_runs",
    "runs_scored": "batter_runs",
    "HR": "batter_home_runs",
    "home_runs": "batter_home_runs",
    "ER": "pitcher_earned_runs",
    "earned_runs": "pitcher_earned_runs",
    "HA": "pitcher_hits_allowed",
    "hits_allowed": "pitcher_hits_allowed",
    "BB": "pitcher_walks_allowed",
    "walks": "pitcher_walks_allowed",
    "IP": "pitcher_innings_pitched",
    "OUTS": "pitcher_outs_recorded",
    "SB": "batter_stolen_bases",
    "stolen_bases": "batter_stolen_bases",
}

# Ambiguous statIDs that require role context
AMBIGUOUS_STAT_IDS = {"H", "K", "hits", "strikeouts"}

# Pitcher-typical line ranges for heuristic fallback
PITCHER_H_LINE_RANGE = (3.0, 12.0)  # Pitcher hits allowed typically 3.5-8.5
BATTER_H_LINE_RANGE = (0.0, 3.5)   # Batter hits typically 0.5-2.5
PITCHER_K_LINE_RANGE = (3.0, 15.0)  # Pitcher Ks typically 4.5-10.5
BATTER_K_LINE_RANGE = (0.0, 2.5)   # Batter Ks typically 0.5-2.5


class MarketMappingResult:
    """Result of a market mapping attempt."""

    __slots__ = ("raw_stat_id", "canonical_market_type", "confidence", "reason", "is_production")

    def __init__(self, raw_stat_id: str, canonical_market_type: str,
                 confidence: str, reason: str):
        self.raw_stat_id = raw_stat_id
        self.canonical_market_type = canonical_market_type
        self.confidence = confidence  # "high", "medium", "low", "ambiguous"
        self.reason = reason
        self.is_production = (
            confidence in ("high", "medium")
            and canonical_market_type in SUPPORTED_CANONICAL_MARKETS
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_stat_id": self.raw_stat_id,
            "canonical_market_type": self.canonical_market_type,
            "market_mapping_confidence": self.confidence,
            "market_mapping_reason": self.reason,
            "is_production_market": self.is_production,
        }


def map_market(
    raw_stat_id: str,
    player_id: str = "",
    player_name: str = "",
    line: float = 0.0,
    event_pitchers: Optional[Set[str]] = None,
    player_position: str = "",
    player_team_id: str = "",
) -> MarketMappingResult:
    """Map a raw statID to canonical market type with confidence.

    Args:
        raw_stat_id: The raw statID from the provider (e.g. "H", "K", "RBI")
        player_id: The statEntityID/playerID from the provider
        player_name: Player display name
        line: The over/under line value
        event_pitchers: Set of playerIDs known to be pitchers in this event
        player_position: Explicit position if available (e.g. "P", "SP", "RP")
        player_team_id: Player's teamID for context
    """
    # 1. Check unambiguous map first
    if raw_stat_id in UNAMBIGUOUS_MAP:
        canonical = UNAMBIGUOUS_MAP[raw_stat_id]
        if canonical in SUPPORTED_CANONICAL_MARKETS:
            return MarketMappingResult(raw_stat_id, canonical, "high", "unambiguous_stat_id")
        else:
            return MarketMappingResult(raw_stat_id, canonical, "high", "unambiguous_unsupported")

    # 2. Handle ambiguous statIDs
    if raw_stat_id in ("H", "hits"):
        return _resolve_H(raw_stat_id, player_id, player_name, line, event_pitchers, player_position)

    if raw_stat_id in ("K", "strikeouts"):
        return _resolve_K(raw_stat_id, player_id, player_name, line, event_pitchers, player_position)

    # 3. Try normalized lookup
    normalized = raw_stat_id.lower().replace("-", "_").replace(" ", "_")
    if normalized in UNAMBIGUOUS_MAP:
        canonical = UNAMBIGUOUS_MAP[normalized]
        if canonical in SUPPORTED_CANONICAL_MARKETS:
            return MarketMappingResult(raw_stat_id, canonical, "high", "normalized_unambiguous")
        else:
            return MarketMappingResult(raw_stat_id, canonical, "high", "normalized_unsupported")

    # 4. Unknown/unsupported
    return MarketMappingResult(
        raw_stat_id, f"unsupported_{raw_stat_id}",
        "low", "unsupported_stat_id"
    )


def _resolve_H(
    raw_stat_id: str,
    player_id: str,
    player_name: str,
    line: float,
    event_pitchers: Optional[Set[str]],
    player_position: str,
) -> MarketMappingResult:
    """Resolve ambiguous H (hits) statID."""

    # Priority 1: Explicit position
    if player_position:
        pos = player_position.upper()
        if pos in ("P", "SP", "RP", "CL"):
            return MarketMappingResult(raw_stat_id, "pitcher_hits_allowed", "high", "explicit_pitcher_position")
        else:
            return MarketMappingResult(raw_stat_id, "batter_hits", "high", "explicit_batter_position")

    # Priority 2: Known pitcher list from event
    if event_pitchers and player_id:
        if player_id in event_pitchers:
            return MarketMappingResult(raw_stat_id, "pitcher_hits_allowed", "high", "event_pitcher_list")
        else:
            return MarketMappingResult(raw_stat_id, "batter_hits", "medium", "not_in_pitcher_list")

    # Priority 3: Line range heuristic
    if line >= 3.5:
        # Lines 3.5+ are almost certainly pitcher hits allowed
        return MarketMappingResult(raw_stat_id, "pitcher_hits_allowed", "medium", f"line_heuristic_pitcher_range_{line}")
    elif line <= 2.5:
        # Lines 0.5-2.5 are almost certainly batter hits
        return MarketMappingResult(raw_stat_id, "batter_hits", "medium", f"line_heuristic_batter_range_{line}")
    else:
        # Line 3.0 is ambiguous
        return MarketMappingResult(raw_stat_id, "diagnostic_only", "ambiguous", f"ambiguous_H_market_line_{line}")


def _resolve_K(
    raw_stat_id: str,
    player_id: str,
    player_name: str,
    line: float,
    event_pitchers: Optional[Set[str]],
    player_position: str,
) -> MarketMappingResult:
    """Resolve ambiguous K (strikeouts) statID."""

    # Priority 1: Explicit position
    if player_position:
        pos = player_position.upper()
        if pos in ("P", "SP", "RP", "CL"):
            return MarketMappingResult(raw_stat_id, "pitcher_strikeouts", "high", "explicit_pitcher_position")
        else:
            return MarketMappingResult(raw_stat_id, "batter_strikeouts", "high", "explicit_batter_position")

    # Priority 2: Known pitcher list from event
    if event_pitchers and player_id:
        if player_id in event_pitchers:
            return MarketMappingResult(raw_stat_id, "pitcher_strikeouts", "high", "event_pitcher_list")
        else:
            return MarketMappingResult(raw_stat_id, "batter_strikeouts", "medium", "not_in_pitcher_list")

    # Priority 3: Line range heuristic
    # Pitcher K lines are typically 3.5-12.5; batter K lines are 0.5-2.5
    if line >= 3.0:
        return MarketMappingResult(raw_stat_id, "pitcher_strikeouts", "medium", f"line_heuristic_pitcher_range_{line}")
    elif line <= 2.5:
        return MarketMappingResult(raw_stat_id, "batter_strikeouts", "medium", f"line_heuristic_batter_range_{line}")
    else:
        return MarketMappingResult(raw_stat_id, "diagnostic_only", "ambiguous", f"ambiguous_K_market_line_{line}")


def extract_event_pitchers(event: Dict[str, Any]) -> Set[str]:
    """Extract known pitcher playerIDs from event data.

    Uses odds structure to identify pitchers: players with K/H props at
    pitcher-typical lines, or explicit pitcher metadata if available.
    """
    pitchers: Set[str] = set()
    odds_dict = event.get("odds", {})
    players_dict = event.get("players", {})

    for odd_id, odd_data in odds_dict.items():
        if not isinstance(odd_data, dict):
            continue
        if odd_data.get("betTypeID") != "ou":
            continue

        stat_id = odd_data.get("statID", "")
        player_id = odd_data.get("playerID", odd_data.get("statEntityID", ""))
        line_str = odd_data.get("bookOverUnder", odd_data.get("fairOverUnder", ""))

        if not player_id or not line_str:
            continue

        try:
            line = float(line_str)
        except (ValueError, TypeError):
            continue

        # Pitcher indicators:
        # - K with line >= 3.5 (pitcher strikeouts)
        # - H with line >= 3.5 (pitcher hits allowed)
        # - ER (earned runs) — always pitcher
        # - OUTS/IP — always pitcher
        if stat_id in ("ER", "earned_runs", "OUTS", "IP", "outs_recorded", "innings_pitched"):
            pitchers.add(player_id)
        elif stat_id in ("K", "strikeouts") and line >= 3.5:
            pitchers.add(player_id)
        elif stat_id in ("H", "hits") and line >= 3.5:
            pitchers.add(player_id)

    return pitchers
