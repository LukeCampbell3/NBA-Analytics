#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.evaluation import PoolScoreConfig, score_all_unscored_pools, score_prediction_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score MLB prediction pools against actual game results."
    )
    parser.add_argument(
        "--pool-csv",
        type=Path,
        default=None,
        help="Single prediction pool CSV to score.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Raw logs directory containing hitter_game_logs.csv and pitcher_game_logs.csv.",
    )
    parser.add_argument(
        "--scored-csv-out",
        type=Path,
        default=None,
        help="Optional scored rows CSV output path for single-pool mode.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=None,
        help="Optional summary JSON output path for single-pool mode.",
    )
    parser.add_argument(
        "--all-unscored",
        action="store_true",
        help="Batch-score all unscored daily pools under daily-runs-root.",
    )
    parser.add_argument(
        "--daily-runs-root",
        type=Path,
        default=ROOT / "data" / "predictions" / "daily_runs",
        help="Daily prediction runs root for batch mode.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Score only pools with game_date <= as-of-date (YYYY-MM-DD) in batch mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scored outputs in batch mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_unscored:
        if not args.as_of_date:
            raise ValueError("--as-of-date is required when --all-unscored is used.")
        payload = score_all_unscored_pools(
            daily_runs_root=args.daily_runs_root,
            raw_dir=args.raw_dir,
            as_of_date=str(args.as_of_date),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.pool_csv is None:
        raise ValueError("Provide --pool-csv for single-pool scoring, or use --all-unscored.")

    payload = score_prediction_pool(
        PoolScoreConfig(
            pool_csv=args.pool_csv,
            raw_dir=args.raw_dir,
            scored_csv_out=args.scored_csv_out,
            summary_json_out=args.summary_json_out,
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
