#!/usr/bin/env python3
"""Refresh NFL weekly stats and refit the already validated model architectures."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.latent_pipeline import refit_deployment_artifact  # noqa: E402
from sports.nfl.predictions.pbp_stats import load_aggregated_season  # noqa: E402
from sports.nfl.predictions.pipeline import NFLVERSE_PLAYER_STATS_URL, load_weekly_stats  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports/nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats", type=Path, default=NFL_ROOT / "data/raw/player_stats.parquet"
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=NFL_ROOT / "model/nfl_yardage_latent_hybrid.joblib",
    )
    parser.add_argument(
        "--deployment-stats",
        type=Path,
        default=NFL_ROOT / "data/raw/player_stats_deployment.parquet",
    )
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_day = date.fromisoformat(args.run_date) if args.run_date else date.today()
    latest = pd.read_parquet(NFLVERSE_PLAYER_STATS_URL)
    latest.to_parquet(args.stats, index=False)
    stats = load_weekly_stats(args.stats, start_season=2018)
    maximum_base_season = int(stats["season"].max())
    supplements: list[pd.DataFrame] = []
    last_required_season = run_day.year - 1
    for season in range(maximum_base_season + 1, last_required_season + 1):
        supplements.append(
            load_aggregated_season(
                season,
                cache_path=NFL_ROOT / f"data/raw/player_stats_{season}_pbp.parquet",
            )
        )
    current_cache = NFL_ROOT / f"data/raw/player_stats_{run_day.year}_pbp.parquet"
    should_refresh_current = run_day.month >= 9 and (run_day.weekday() == 1 or args.force)
    if should_refresh_current:
        try:
            supplements.append(
                load_aggregated_season(
                    run_day.year,
                    cache_path=current_cache,
                    refresh=True,
                )
            )
        except Exception as error:
            if not current_cache.is_file():
                print(f"Current-season play-by-play is not available yet: {error}")
            else:
                print(f"Current-season refresh failed; using cached data: {error}")
                supplements.append(pd.read_parquet(current_cache))
    elif current_cache.is_file():
        supplements.append(pd.read_parquet(current_cache))
    if supplements:
        stats = (
            pd.concat([stats, *supplements], ignore_index=True)
            .drop_duplicates(["player_id", "season", "week"], keep="last")
            .sort_values(["season", "week", "player_id"])
        )
    args.deployment_stats.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(args.deployment_stats, index=False)
    newest = stats.sort_values(["season", "week"]).iloc[-1]
    artifact = joblib.load(args.artifact)
    previous = artifact.get("refit_through") or {
        "season": artifact.get("holdout_season", 0),
        "week": 0,
    }
    current_key = (int(newest["season"]), int(newest["week"]))
    previous_key = (int(previous.get("season", 0)), int(previous.get("week", 0)))
    if not args.force and current_key <= previous_key:
        print(
            f"NFL yardage artifact is current through {previous_key[0]} "
            f"week {previous_key[1]}; deployment features end at "
            f"{current_key[0]} week {current_key[1]}."
        )
        return 0
    refreshed = refit_deployment_artifact(stats, artifact)
    joblib.dump(refreshed, args.artifact, compress=3)
    print(f"NFL yardage artifact refit through {current_key[0]} week {current_key[1]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
