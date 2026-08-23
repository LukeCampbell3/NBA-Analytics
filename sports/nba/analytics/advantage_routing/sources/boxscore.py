"""Real box-score source adapter -- reads the existing, already-cached
Player-Predictor per-game CSVs (Player-Predictor/Data-Proc/<Player>/
<season>_processed_processed.csv). Every value returned here is
OBSERVED: these are real per-game box scores already ingested elsewhere
in this repository, not fetched or invented by this package.

This is the ONLY source in the advantage-routing pipeline for
possession-ending box-score counts (PTS/TRB/AST/TOV/FGA/FTA/MP/USG%).
It does NOT and cannot provide touches, post-ups, drives, or any
tracking-only quantity -- see sources/bball_ref.py and
docs/advantage-routing.md for what else is (and is not) reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_PROC_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc"

BOXSCORE_SOURCE_LABEL = "Player-Predictor/Data-Proc (real per-game box scores)"

REQUIRED_COLUMNS = (
    "Date", "PTS", "TRB", "AST", "STL", "TOV", "FGA", "FTA", "MP", "USG%",
)


@dataclass(frozen=True)
class PlayerBoxScoreTable:
    player_name: str
    season: str
    games: pd.DataFrame  # one row per real game actually played
    source_path: Path

    @property
    def games_played(self) -> int:
        return int(len(self.games))


def _player_dir_name(player_name: str) -> str:
    """Player-Predictor's own directory naming: spaces -> underscores,
    hyphens preserved (e.g. "Collin Murray-Boyles" -> "Collin_Murray-Boyles")."""
    return player_name.strip().replace(" ", "_")


def _season_file_stub(season: str) -> str:
    """Player-Predictor stamps files by the season's END year (e.g.
    season "2025-26" -> "2026"). Falls back to the raw season string if
    it isn't in that YYYY-YY shape."""
    if "-" in season and len(season.split("-")[0]) == 4:
        start_year = int(season.split("-")[0])
        return str(start_year + 1)
    return season


def load_player_boxscores(player_name: str, season: str = "2025-26") -> Optional[PlayerBoxScoreTable]:
    """Loads a real player's per-game box scores for one season. Returns
    None (never a fabricated empty table) if no such file exists --
    callers must treat that as "no real box-score source for this
    player/season", not "zero games"."""
    player_dir = DATA_PROC_ROOT / _player_dir_name(player_name)
    csv_path = player_dir / f"{_season_file_stub(season)}_processed_processed.csv"
    if not csv_path.is_file():
        return None

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required real columns: {missing}")

    if "Did_Not_Play" in df.columns:
        df = df[df["Did_Not_Play"].fillna(0) != 1].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    return PlayerBoxScoreTable(player_name=player_name, season=season, games=df, source_path=csv_path)


def list_available_players(season: str = "2025-26") -> list[str]:
    """Every player with a real, already-cached box-score file for this
    season -- used to populate the frontend's player selector without
    guessing who has data."""
    stub = _season_file_stub(season)
    if not DATA_PROC_ROOT.is_dir():
        return []
    names = []
    for player_dir in sorted(DATA_PROC_ROOT.iterdir()):
        if not player_dir.is_dir():
            continue
        if (player_dir / f"{stub}_processed_processed.csv").is_file():
            names.append(player_dir.name.replace("_", " "))
    return names
