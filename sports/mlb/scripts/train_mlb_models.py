#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.training import TrainConfig, train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLB hitter/pitcher models from processed data.")
    parser.add_argument("--season", type=int, default=2026, help="Season year.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Processed data directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Output model directory.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="Minimum row count per role required before training.",
    )
    parser.add_argument(
        "--min-real-market-rows",
        type=int,
        default=1,
        help="Minimum matched real market rows per role required before training.",
    )
    parser.add_argument(
        "--min-real-market-dates",
        type=int,
        default=1,
        help="Minimum matched real market dates per role required before training.",
    )
    parser.add_argument(
        "--allow-synthetic-market-only",
        action="store_true",
        help="Bypass the real-market-line gate and allow synthetic-only training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = train_all_models(
        TrainConfig(
            processed_dir=args.processed_dir,
            model_dir=args.model_dir,
            season=int(args.season),
            min_rows=int(args.min_rows),
            require_real_market_lines=not bool(args.allow_synthetic_market_only),
            min_real_market_rows=int(args.min_real_market_rows),
            min_real_market_dates=int(args.min_real_market_dates),
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

