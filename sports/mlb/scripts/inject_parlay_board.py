#!/usr/bin/env python3
"""Inject MLB parlay board data into an existing daily_predictions.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(SPORT_ROOT))

from decision_engine.parlay_builder import build_mlb_daily_board, MLBParlayConfig


def _sf(v, d=0.0):
    try:
        f = float(v)
        return f if f == f else d
    except Exception:
        return d


def board_to_payload(board):
    parlays = []
    for p in board.primary_parlays:
        legs = []
        for leg in p.get("legs", []):
            legs.append({
                "player": str(leg.get("Player", leg.get("player", ""))),
                "team": str(leg.get("Team", leg.get("team", ""))),
                "opponent": str(leg.get("Opponent", leg.get("opponent", ""))),
                "target": str(leg.get("Target", leg.get("target", ""))),
                "direction": str(leg.get("Direction", leg.get("direction", ""))),
                "market_line": _sf(leg.get("Market_Line", leg.get("market_line"))),
                "hit_probability": _sf(leg.get("Estimated_Hit_Probability", leg.get("calibrated_hit_probability"))),
                "abs_edge": _sf(leg.get("Abs_Edge", leg.get("abs_edge"))),
                "confidence_tier": str(leg.get("Confidence_Tier", leg.get("confidence_tier", ""))),
            })
        parlays.append({
            "type": p.get("type", "primary"),
            "leg_count": p.get("leg_count", len(legs)),
            "joint_probability": _sf(p.get("joint_prob")),
            "adjusted_probability": _sf(p.get("adjusted_prob")),
            "avg_hit_probability": _sf(p.get("avg_hit_prob")),
            "score": _sf(p.get("score")),
            "n_games": p.get("n_games", 0),
            "n_teams": p.get("n_teams", 0),
            "legs": legs,
        })
    return {"parlays": parlays, "diagnostics": board.diagnostics}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, required=True, help="daily_predictions.json to update")
    p.add_argument("--pool-csv", type=Path, default=None, help="High-precision predictions CSV")
    args = p.parse_args()

    json_path = args.json.resolve()
    if not json_path.exists():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    pool_csv = args.pool_csv
    if pool_csv is None:
        daily_runs = SPORT_ROOT / "data" / "predictions" / "daily_runs"
        for d in sorted(daily_runs.iterdir(), reverse=True):
            if d.is_dir():
                candidates = sorted(d.glob("*_high_precision_predictions.csv"))
                if candidates:
                    pool_csv = candidates[-1]
                    break

    if pool_csv is None or not pool_csv.exists():
        print("No high-precision CSV found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(pool_csv)
    board = build_mlb_daily_board(df, config=MLBParlayConfig())
    payload["parlay_board"] = board_to_payload(board)

    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Injected MLB parlay board ({len(board.primary_parlays)} parlays) into {json_path}")


if __name__ == "__main__":
    main()
