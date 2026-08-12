#!/usr/bin/env python3
"""Refresh the normalized current NFL skill-position roster."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
NFL_ROOT = REPO_ROOT / "sports" / "nfl"
ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{season}.parquet"
)
SKILL_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--history",
        type=Path,
        default=NFL_ROOT / "data/raw/player_stats_deployment.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NFL_ROOT / "data/reference/current_skill_roster.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=NFL_ROOT / "data/reference/current_skill_roster_manifest.json",
    )
    return parser.parse_args()


def history_availability(
    player_ids: pd.Series, *, history_path: Path, prior_roster_path: Path
) -> pd.Series:
    if history_path.is_file():
        known_ids = set(
            pd.read_parquet(history_path, columns=["player_id"])["player_id"].astype(str)
        )
        return player_ids.astype(str).isin(known_ids)
    if prior_roster_path.is_file():
        prior = pd.read_csv(prior_roster_path, usecols=["gsis_id", "history_available"])
        availability = prior.set_index(prior["gsis_id"].astype(str))["history_available"]
        return player_ids.astype(str).map(availability).eq(True)
    return pd.Series(False, index=player_ids.index, dtype=bool)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw = pd.read_parquet(ROSTER_URL.format(season=args.season))
    latest_week = int(raw["week"].max())
    roster = raw.loc[
        raw["week"].eq(latest_week) & raw["position"].isin(SKILL_POSITIONS)
    ].copy()
    roster = roster.dropna(subset=["gsis_id", "full_name", "team", "position"])
    roster = roster.drop_duplicates("gsis_id", keep="last")
    roster["history_available"] = history_availability(
        roster["gsis_id"], history_path=args.history, prior_roster_path=args.output
    )
    columns = [
        "season",
        "week",
        "gsis_id",
        "full_name",
        "team",
        "position",
        "depth_chart_position",
        "status",
        "years_exp",
        "rookie_year",
        "history_available",
    ]
    output = roster[[column for column in columns if column in roster.columns]].sort_values(
        ["team", "position", "full_name"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "schema_version": 1,
        "source": ROSTER_URL.format(season=args.season),
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "season": args.season,
        "week": latest_week,
        "skill_positions": sorted(SKILL_POSITIONS),
        "players": int(len(output)),
        "players_with_model_history": int(output["history_available"].sum()),
        "position_counts": {
            str(key): int(value) for key, value in output["position"].value_counts().items()
        },
        "output": args.output.resolve().relative_to(REPO_ROOT).as_posix(),
    }
    write_manifest(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
