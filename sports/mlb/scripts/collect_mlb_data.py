#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.collect import CollectorConfig, collect_raw_mlb_data, default_collection_window


def parse_args() -> argparse.Namespace:
    start_default, end_default = default_collection_window(days_back=30)
    parser = argparse.ArgumentParser(description="Collect raw MLB hitter/pitcher game logs from MLB Stats API.")
    parser.add_argument("--start-date", type=str, default=start_default, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=end_default, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--game-type", type=str, default="R", help="MLB game type (default: R regular season).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Raw output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = collect_raw_mlb_data(
        CollectorConfig(
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            game_type=str(args.game_type),
        ),
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

