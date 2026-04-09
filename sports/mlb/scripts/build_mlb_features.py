#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.features import FeatureBuildConfig, build_processed_mlb_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MLB processed features from raw game logs.")
    parser.add_argument("--season", type=int, default=2026, help="Season year to use in output file naming.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw", help="Raw input directory.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Processed output directory.",
    )
    parser.add_argument(
        "--market-file",
        type=Path,
        default=None,
        help="Optional market lines CSV with Date/Player/Player_Type and Market_* columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_processed_mlb_features(
        FeatureBuildConfig(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            season=int(args.season),
            market_file=args.market_file,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

