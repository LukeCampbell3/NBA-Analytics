#!/usr/bin/env python3
"""
Build the daily MLB parlay + singles board from the high-precision prediction pool.

This script reads the output of select_high_precision_predictions.py and produces:
  1. A primary parlay ticket (2-3 legs, highest confidence, different games)
  2. A ranked singles board with stake sizing

Usage:
    python sports/mlb/scripts/build_daily_parlay_board.py
    python sports/mlb/scripts/build_daily_parlay_board.py --pool-csv path/to/predictions.csv
    python sports/mlb/scripts/build_daily_parlay_board.py --run-date 2026-05-03

Output:
    - Prints the formatted daily board to stdout
    - Writes JSON summary to the daily_runs directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DAILY_RUNS = SPORT_ROOT / "data" / "predictions" / "daily_runs"

sys.path.insert(0, str(SPORT_ROOT))
from decision_engine.parlay_builder import (
    MLBDailyBoard,
    MLBParlayConfig,
    build_mlb_daily_board,
    format_mlb_daily_board,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the daily MLB parlay + singles board.")
    parser.add_argument("--pool-csv", type=Path, default=None, help="Explicit high-precision predictions CSV.")
    parser.add_argument("--run-date", type=str, default=None, help="Run date (YYYYMMDD or YYYY-MM-DD).")
    parser.add_argument("--daily-runs", type=Path, default=DEFAULT_DAILY_RUNS, help="Daily runs root directory.")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for stake sizing display.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress formatted board output.")
    return parser.parse_args()


def find_pool_csv(args: argparse.Namespace) -> Path | None:
    if args.pool_csv and args.pool_csv.exists():
        return args.pool_csv

    run_date = (args.run_date or "").replace("-", "")
    if run_date:
        date_dir = args.daily_runs / run_date
        candidates = sorted(date_dir.glob("*_high_precision_predictions.csv"))
        if candidates:
            return candidates[-1]

    # Find latest
    for date_dir in sorted(args.daily_runs.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        candidates = sorted(date_dir.glob("*_high_precision_predictions.csv"))
        if candidates:
            return candidates[-1]

    return None


def board_to_json(board: MLBDailyBoard) -> dict:
    """Convert the board to a JSON-serializable dict."""

    def _leg_dict(leg: dict) -> dict:
        return {
            "player": str(leg.get("Player", leg.get("player", ""))),
            "team": str(leg.get("Team", leg.get("team", ""))),
            "opponent": str(leg.get("Opponent", leg.get("opponent", ""))),
            "target": str(leg.get("Target", leg.get("target", ""))),
            "direction": str(leg.get("Direction", leg.get("direction", ""))),
            "market_line": float(leg.get("Market_Line", leg.get("market_line", 0))),
            "hit_probability": float(leg.get("Estimated_Hit_Probability", leg.get("calibrated_hit_probability", 0))),
            "abs_edge": float(leg.get("Abs_Edge", leg.get("abs_edge", 0))),
            "confidence_tier": str(leg.get("Confidence_Tier", leg.get("confidence_tier", ""))),
        }

    parlays = []
    for parlay in board.primary_parlays:
        parlays.append({
            "type": parlay.get("type", "primary"),
            "leg_count": parlay.get("leg_count", 0),
            "joint_probability": parlay.get("joint_prob", 0),
            "adjusted_probability": parlay.get("adjusted_prob", 0),
            "avg_hit_probability": parlay.get("avg_hit_prob", 0),
            "n_games": parlay.get("n_games", 0),
            "n_teams": parlay.get("n_teams", 0),
            "score": parlay.get("score", 0),
            "legs": [_leg_dict(leg) for leg in parlay.get("legs", [])],
        })

    singles = []
    for pick in board.singles:
        singles.append({
            **_leg_dict(pick),
            "stake_tier": pick.get("stake_tier", ""),
            "stake_fraction": pick.get("stake_fraction", 0),
            "ev_per_unit": float(pick.get("Expected_Value_Per_Unit", pick.get("expected_value_per_unit", 0)) or 0),
            "in_parlay": bool(pick.get("singles_note") == "also_in_parlay"),
        })

    return {
        "parlays": parlays,
        "singles": singles,
        "diagnostics": board.diagnostics,
    }


def main() -> None:
    args = parse_args()
    pool_csv = find_pool_csv(args)

    if pool_csv is None:
        print("No high-precision predictions CSV found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {pool_csv}")
    df = pd.read_csv(pool_csv)
    print(f"Candidates: {len(df)}")

    board = build_mlb_daily_board(df)

    if not args.quiet:
        print(format_mlb_daily_board(board, bankroll=args.bankroll))

    # Write JSON
    out_json = args.out_json
    if out_json is None:
        out_json = pool_csv.parent / f"{pool_csv.stem}_parlay_board.json"

    payload = board_to_json(board)
    payload["source_csv"] = str(pool_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nBoard JSON: {out_json}")


if __name__ == "__main__":
    main()
