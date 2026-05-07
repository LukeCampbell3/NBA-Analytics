"""Name and market normalization for sportsbook matching."""
from __future__ import annotations

import re
import unicodedata


def normalize_player_name(name: str) -> str:
    """Normalize a player name for fuzzy matching across sportsbooks."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove suffixes
    for suffix in (" jr", " sr", " ii", " iii", " iv"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return " ".join(text.split())


# Canonical market keys used across the system
NBA_TARGET_TO_MARKET = {
    "PTS": "player_points",
    "TRB": "player_rebounds",
    "REB": "player_rebounds",
    "AST": "player_assists",
    "3PM": "player_threes",
    "3PT": "player_threes",
    "STL": "player_steals",
    "BLK": "player_blocks",
    "TO": "player_turnovers",
    "PRA": "player_points_rebounds_assists",
}

MLB_TARGET_TO_MARKET = {
    "H": "batter_hits",
    "TB": "batter_total_bases",
    "R": "batter_runs_scored",
    "RBI": "batter_rbis",
    "HR": "batter_home_runs",
    "K": "pitcher_strikeouts",
    "BB": "pitcher_walks",
    "ER": "pitcher_earned_runs",
    "OUTS": "pitcher_outs",
}


def target_to_market_key(target: str, sport: str) -> str:
    """Convert our internal target name to a canonical market key."""
    t = target.upper().strip()
    if sport.lower() in ("nba", "basketball"):
        return NBA_TARGET_TO_MARKET.get(t, f"player_{t.lower()}")
    if sport.lower() in ("mlb", "baseball"):
        return MLB_TARGET_TO_MARKET.get(t, f"batter_{t.lower()}")
    return t.lower()


def normalize_side(side: str) -> str:
    """Normalize Over/Under to uppercase."""
    s = str(side or "").strip().upper()
    if s in ("OVER", "O"):
        return "OVER"
    if s in ("UNDER", "U"):
        return "UNDER"
    return s
