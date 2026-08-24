#!/usr/bin/env python3
"""Real historical NFL team-game outcomes AND real closing market lines,
from nflverse's public games.parquet (the same real, free source already
used by fetch_historical_nfl_props.py and build_nfl_week_pool.py for
schedule/kickoff data in this repo).

WHY THIS EXISTS: same-game parlay support (moneyline + game total +
player props combined) needs a real team win-probability model and a
real game-total model -- neither exists anywhere in this codebase today
(every NFL/MLB/NBA prediction system here is player-props only). This is
the real, verified historical foundation those models need: for every
real completed NFL regular-season game since 2016, the real final score
AND the real closing moneyline/spread/total line the market settled on.

This module deliberately stops at "fetch and persist the real dataset" --
it does not fit a model. That is real, separate, sequenced follow-up
work (see this session's scoped decision: real dependence/correlation
modeling must be validated before any same-game combination goes live).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "sports" / "nfl" / "data" / "reference" / "nfl_team_game_history.csv"

OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_type",
    "game_id",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_won",
    "total_points",
    "spread_line",
    "home_spread_odds",
    "away_spread_odds",
    "total_line",
    "over_odds",
    "under_odds",
    "home_moneyline",
    "away_moneyline",
]


def fetch_real_team_game_history(*, min_season: Optional[int] = None) -> pd.DataFrame:
    """Every real, completed NFL regular-season game nflverse has a real
    final score AND a real closing total_line for -- never a row with a
    guessed or interpolated score or line. `min_season` (if given) trims
    to real games at/after that season; omit it for the full real
    history nflverse currently publishes."""
    schedule = pd.read_parquet(SCHEDULE_URL)
    real_games = schedule.loc[
        schedule["game_type"].eq("REG")
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
        & schedule["total_line"].notna()
    ].copy()
    if min_season is not None:
        real_games = real_games.loc[real_games["season"] >= int(min_season)].copy()

    real_games["home_won"] = (real_games["home_score"] > real_games["away_score"]).astype(int)
    real_games["total_points"] = real_games["home_score"] + real_games["away_score"]
    return real_games[[column for column in OUTPUT_COLUMNS if column in real_games.columns]].sort_values(
        ["season", "week", "game_id"]
    ).reset_index(drop=True)


def persist_team_game_history(frame: pd.DataFrame, *, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-season", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = fetch_real_team_game_history(min_season=args.min_season)
    out_path = persist_team_game_history(frame, output_path=args.output)
    print(f"wrote {len(frame)} real completed games with real closing lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
