#!/usr/bin/env python3
"""Grade NFL predictions against authentic historical player-prop lines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_backtest import (  # noqa: E402
    evaluate_market_backtest,
    load_market_archive,
)


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=Path, required=True, help="CSV/parquet historical prop archive.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "backtest_rows.csv",
    )
    parser.add_argument("--minimum-edge", type=float, default=0.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_backtest_report.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_backtest_rows.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, low_memory=False)
    markets = load_market_archive(args.lines)
    report, rows = evaluate_market_backtest(
        predictions, markets, minimum_edge_yards=args.minimum_edge
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(args.rows, index=False)
    print(json.dumps(report["overall"], indent=2))
    print(f"Market validation: {report['status']}")
    print(f"Report: {args.report}")
    return 0 if report["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
