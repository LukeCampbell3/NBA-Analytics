#!/usr/bin/env python3
"""Freeze the latest nflverse depth chart used by the fantasy lineup model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
NFL_ROOT = REPO_ROOT / "sports" / "nfl"
URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=NFL_ROOT / "data/reference/current_depth_chart.csv")
    parser.add_argument("--manifest", type=Path, default=NFL_ROOT / "data/reference/current_depth_chart_manifest.json")
    args = parser.parse_args()
    source = URL.format(season=args.season)
    frame = pd.read_parquet(source)
    latest = str(frame["dt"].max())
    output = frame.loc[
        frame["dt"].astype(str).eq(latest) & frame["pos_abb"].isin({"QB", "RB", "WR", "TE"})
    ].sort_values(["team", "pos_abb", "pos_rank", "player_name"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "schema_version": 1,
        "source": source,
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "depth_chart_as_of_utc": latest,
        "season": args.season,
        "players": int(len(output)),
        "teams": int(output["team"].nunique()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
