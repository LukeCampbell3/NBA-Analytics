#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.inference import DailyInferenceConfig, generate_daily_prediction_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MLB daily prediction pool for a target game date."
    )
    parser.add_argument(
        "--run-date",
        type=str,
        required=True,
        help="Target game date to predict (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (defaults to run-date year).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Processed features directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Model artifact directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "predictions" / "daily_runs",
        help="Output directory for pool CSV/JSON.",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default="R",
        help="MLB schedule game type (default: R regular season).",
    )
    parser.add_argument(
        "--min-history-rows",
        type=int,
        default=5,
        help="Minimum historical rows required per player.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_date = str(args.run_date)
    season = int(args.season) if args.season is not None else int(run_date[:4])
    run_stamp = run_date.replace("-", "")
    out_dir = (args.out_dir / run_stamp).resolve()

    payload = generate_daily_prediction_pool(
        DailyInferenceConfig(
            run_date=run_date,
            season=season,
            processed_dir=args.processed_dir,
            model_dir=args.model_dir,
            out_dir=out_dir,
            game_type=str(args.game_type),
            min_history_rows=int(args.min_history_rows),
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
