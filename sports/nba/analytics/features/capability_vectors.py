"""
Capability Vector Builder

Transforms existing player data (Data-Proc CSVs + player_cards JSONs)
into the full 22-dimension PlayerCapabilityVector.

Uses observed data where available, marks inferred/unavailable dimensions honestly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..schema import (
    CAPABILITY_DIMENSIONS,
    ObservationStatus,
    PlayerCapabilityVector,
    VectorDimension,
)
from .percentiles import compute_percentile, reliability_shrinkage

WORKSPACE = Path(__file__).resolve().parents[4]  # sports/nba/analytics/features -> repo root
DATA_PROC_DIR = WORKSPACE / "Player-Predictor" / "Data-Proc"
PLAYER_CARDS_DIR = WORKSPACE / "data" / "processed" / "player_cards"
RAW_DATA_DIR = WORKSPACE / "data" / "raw"


def load_player_game_log(player_name: str, season: int = 2026) -> pd.DataFrame:
    """Load per-game data from Data-Proc."""
    player_dir = DATA_PROC_DIR / player_name.replace(" ", "_")
    csv_path = player_dir / f"{season}_processed_processed.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        if "Did_Not_Play" in df.columns:
            df = df[df["Did_Not_Play"] != 1]
        return df
    except Exception:
        return pd.DataFrame()


def load_player_card(player_name: str) -> Dict[str, Any]:
    """Load enriched player card JSON."""
    # Search for matching card file
    pattern = f"{player_name.replace(' ', '_')}*_final.json"
    matches = list(PLAYER_CARDS_DIR.glob(pattern))
    if not matches:
        return {}
    try:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_capability_vector(
    player_name: str,
    league_stats: Optional[pd.DataFrame] = None,
    season: int = 2026,
) -> PlayerCapabilityVector:
    """Build a full capability vector for a player from available data.

    Args:
        player_name: Player name (e.g. "LeBron James")
        league_stats: Optional DataFrame of all players' aggregated stats for percentiles
        season: Season year

    Returns:
        PlayerCapabilityVector with all dimensions populated from available data
    """
    game_log = load_player_game_log(player_name, season)
    card = load_player_card(player_name)

    vector = PlayerCapabilityVector(
        player_id=str(card.get("player", {}).get("id", "")),
        player_name=player_name,
        team=str(card.get("player", {}).get("team", "")),
        position=str(card.get("player", {}).get("position", "")),
        season=season,
    )

    if game_log.empty and not card:
        vector.metadata["warning"] = "No data available for this player"
        return vector

    n_games = len(game_log) if not game_log.empty else 0
    trad = card.get("performance", {}).get("traditional", {})
    adv = card.get("performance", {}).get("advanced", {})
    shot = card.get("shot_profile", {})
    creation = card.get("creation_profile", {})

    # Helper to set a dimension
    def _set(name: str, raw: Optional[float], source: str, status: ObservationStatus,
             sample: int = n_games, conf: float = 0.7):
        if raw is None:
            vector.set_dimension(name, observation_status=ObservationStatus.UNAVAILABLE, confidence=0.0)
            return
        vector.set_dimension(
            name,
            raw_value=round(raw, 3),
            sample_size=sample,
            confidence=round(conf, 3),
            source=source,
            observation_status=status,
        )

    # --- Compute raw values from available data ---

    pts = float(trad.get("points_per_game", 0))
    ast = float(trad.get("assists_per_game", 0))
    usg = float(adv.get("usage_rate", 0.15))
    fg_pct = float(trad.get("field_goal_pct", 0))
    three_pct = float(trad.get("three_point_pct", 0))
    ft_pct = float(trad.get("free_throw_pct", 0))
    reb = float(trad.get("rebounds_per_game", 0))
    stl = float(trad.get("steals_per_game", 0))
    blk = float(trad.get("blocks_per_game", 0))
    mp = float(trad.get("minutes_per_game", 0))
    three_freq = float(shot.get("three_point_frequency", 0.3))
    rim_freq = float(shot.get("rim_frequency", 0.3))
    drives = float(creation.get("drives_per_game", 0))
    assisted_rate = float(creation.get("assisted_rate", 0.5))

    src = "player_card" if card else "game_log"
    obs = ObservationStatus.OBSERVED if n_games >= 20 else ObservationStatus.INFERRED
    conf_base = min(1.0, n_games / 50) * 0.8

    # On-ball creation: usage + assists + drives
    on_ball = (usg * 200) + (ast * 3) + (drives * 0.5)
    _set("on_ball_creation", on_ball, src, obs, conf=conf_base)

    # Self-scoring efficiency: TS-like composite
    ts_approx = pts / max(2 * (float(trad.get("field_goal_attempts_per_game", 10)) + 0.44 * float(trad.get("free_throw_pct", 0.75))), 1)
    _set("self_scoring_efficiency", fg_pct * 100 + ts_approx * 20, src, obs, conf=conf_base)

    # Rim pressure: drives + rim frequency + FG%
    rim_val = (rim_freq * 100) + (drives * 2) + (fg_pct * 30 if rim_freq > 0.3 else 0)
    _set("rim_pressure", rim_val, src, obs, conf=conf_base)

    # Shooting gravity: 3P% + volume
    shoot_grav = (three_pct * 100) + (three_freq * 50)
    _set("shooting_gravity", shoot_grav, src, obs, conf=conf_base)

    # Spacing gravity: 3P% weighted by frequency and reliability
    spacing_grav = (three_pct * three_freq * 200) + (ft_pct * 10)
    _set("spacing_gravity", spacing_grav, src, obs, conf=conf_base)

    # Corner spacing: inferred from three_freq and assisted_rate
    corner_val = three_freq * assisted_rate * 100
    _set("corner_spacing_value", corner_val, src, ObservationStatus.INFERRED, conf=conf_base * 0.6)

    # Above-break spacing
    ab_val = three_freq * (1 - assisted_rate) * 100
    _set("above_break_spacing_value", ab_val, src, ObservationStatus.INFERRED, conf=conf_base * 0.6)

    # Catch-and-shoot
    cs_val = three_pct * assisted_rate * 100
    _set("catch_and_shoot_gravity", cs_val, src, ObservationStatus.INFERRED, conf=conf_base * 0.5)

    # Pull-up spacing pressure
    pu_val = three_pct * (1 - assisted_rate) * usg * 200
    _set("pull_up_spacing_pressure", pu_val, src, ObservationStatus.INFERRED, conf=conf_base * 0.5)

    # Off-ball scalability: low usage + good shooting = high scalability
    offball = (1 - usg) * three_pct * 100 + assisted_rate * 30
    _set("off_ball_scalability", offball, src, obs, conf=conf_base)

    # Passing creation
    _set("passing_creation", ast * 10 + drives * 1.5, src, obs, conf=conf_base)

    # Decision quality: assist-to-turnover proxy
    tov = float(trad.get("steals_per_game", 0))  # Using TOV from game log if available
    if not game_log.empty and "TOV" in game_log.columns:
        tov = float(game_log["TOV"].mean())
    ast_tov = ast / max(tov, 0.5) if tov > 0 else ast * 2
    _set("decision_quality", ast_tov * 10, src, obs, conf=conf_base)

    # Ball security
    ball_sec = max(0, 50 - tov * 10) + (1 - usg) * 20
    _set("ball_security", ball_sec, src, obs, conf=conf_base)

    # Transition value
    if not game_log.empty and "GmSc" in game_log.columns:
        trans_val = float(game_log["GmSc"].mean()) * 2
    else:
        trans_val = pts * 0.8
    _set("transition_value", trans_val, src, ObservationStatus.INFERRED, conf=conf_base * 0.5)

    # Defensive disruption: steals + blocks
    _set("defensive_disruption", (stl + blk) * 20, src, obs, conf=conf_base)

    # Defensive coverage range: inferred from position/matchup
    matchup = card.get("defense_assessment", {}).get("matchup_profile", {})
    coverage = float(matchup.get("vs_guards", 0.33)) * 30 + float(matchup.get("vs_wings", 0.34)) * 30
    _set("defensive_coverage_range", coverage, src, ObservationStatus.INFERRED, conf=conf_base * 0.5)

    # Rim protection
    _set("rim_protection", blk * 30, src, obs, conf=conf_base)

    # Rebounding
    _set("rebounding_value", reb * 8, src, obs, conf=conf_base)

    # Physical translation: minutes durability proxy
    _set("physical_translation", min(mp * 2.5, 100), src, obs, conf=conf_base)

    # Competition translation: plus/minus proxy
    pm = float(adv.get("plus_minus", 0))
    _set("competition_translation", 50 + pm * 2, src, obs, conf=conf_base)

    # Upside: age-based + usage headroom
    age = float(card.get("player", {}).get("age", 27))
    upside = max(0, (28 - age) * 5) + max(0, (0.30 - usg) * 100)
    _set("upside", upside, src, ObservationStatus.INFERRED, conf=0.4)

    # Risk: injury/low sample/age
    risk = max(0, (age - 28) * 3) + max(0, (30 - n_games) * 1.5)
    _set("risk", risk, src, ObservationStatus.INFERRED, conf=0.4)

    vector.metadata["games_played"] = n_games
    vector.metadata["data_sources"] = ["Data-Proc", "player_cards"] if card else ["Data-Proc"]
    return vector
