#!/usr/bin/env python3
"""Build validated Monte Carlo fantasy draft rankings for the NFL site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.fantasy.model import (  # noqa: E402
    FantasyConfig,
    build_draft_rankings,
    fit_accuracy_layer,
)


SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--players", type=int, default=200)
    parser.add_argument("--history", type=Path, default=NFL_ROOT / "data/raw/player_stats_deployment.parquet")
    parser.add_argument("--roster", type=Path, default=NFL_ROOT / "data/reference/current_skill_roster.csv")
    parser.add_argument(
        "--depth-chart",
        type=Path,
        default=NFL_ROOT / "data/reference/current_depth_chart.csv",
    )
    parser.add_argument("--schedule", default=SCHEDULE_URL)
    parser.add_argument("--output", type=Path, default=NFL_ROOT / "web/data/fantasy_draft_rankings.json")
    parser.add_argument("--validation-output", type=Path, default=NFL_ROOT / "data/evaluation/fantasy_draft_validation.json")
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    history = pd.read_parquet(args.history)
    roster = pd.read_csv(args.roster)
    depth_chart = pd.read_csv(args.depth_chart)
    schedule = pd.read_parquet(args.schedule)
    validation, accuracy_bundle = fit_accuracy_layer(history)
    write_json(args.validation_output, validation)
    if validation["status"] != "passed":
        print(json.dumps(validation, indent=2))
        print("Fantasy rankings were not published because validation failed.")
        return 2
    payload = build_draft_rankings(
        history,
        roster,
        schedule,
        config=FantasyConfig(
            season=args.season,
            simulations=args.simulations,
            published_players=args.players,
        ),
        accuracy_bundle=accuracy_bundle,
        depth_chart=depth_chart,
    )
    if payload["lineup_validation"]["status"] != "passed":
        print(json.dumps(payload["lineup_validation"], indent=2))
        print("Fantasy rankings were not published because lineup validation failed.")
        return 3
    payload["validation"] = validation
    write_json(args.output, payload)
    print(
        f"Published {payload['players_published']} of {payload['players_simulated']} "
        f"simulated players to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
