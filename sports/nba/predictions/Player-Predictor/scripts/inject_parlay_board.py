#!/usr/bin/env python3
"""Inject parlay board data into an existing daily_predictions.json.

This reads the selector CSV, runs the parlay builder, and merges the
parlay board into the web JSON payload so the frontend can render it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PREDICTOR_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(PREDICTOR_ROOT))

from decision_engine.parlay_builder import build_daily_board, ParlayConfig


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
                "player": str(leg.get("player", "")).replace("_", " "),
                "player_display_name": str(leg.get("player", "")).replace("_", " "),
                "target": str(leg.get("target", "")),
                "direction": str(leg.get("direction", "")),
                "market_line": _sf(leg.get("market_line")),
                "win_rate": _sf(leg.get("expected_win_rate")),
                "expected_win_rate": _sf(leg.get("expected_win_rate")),
                "abs_edge": _sf(leg.get("abs_edge")),
                "game_key": str(leg.get("game_key", leg.get("_game_key", ""))),
            })
        parlays.append({
            "type": p.get("type", "primary"),
            "leg_count": p.get("leg_count", len(legs)),
            "joint_probability": _sf(p.get("joint_prob")),
            "adjusted_probability": _sf(p.get("adjusted_prob")),
            "avg_win_rate": _sf(p.get("avg_win_rate")),
            "score": _sf(p.get("score")),
            "n_games": p.get("n_games", 0),
            "legs": legs,
        })
    return {"parlays": parlays, "diagnostics": board.diagnostics}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, required=True, help="daily_predictions.json to update")
    p.add_argument("--selector-csv", type=Path, default=None, help="Selector CSV (auto-detected if omitted)")
    args = p.parse_args()

    json_path = args.json.resolve()
    if not json_path.exists():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    # Find selector CSV
    selector_csv = args.selector_csv
    if selector_csv is None:
        daily_runs = PREDICTOR_ROOT / "model" / "analysis" / "daily_runs"
        for d in sorted(daily_runs.iterdir(), reverse=True):
            if d.is_dir() and d.name.isdigit():
                candidates = sorted(d.glob("upcoming_market_play_selector_*.csv"))
                if candidates:
                    selector_csv = candidates[-1]
                    break

    if selector_csv is None or not selector_csv.exists():
        print("No selector CSV found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(selector_csv)
    board = build_daily_board(df, config=ParlayConfig())
    payload["parlay_board"] = board_to_payload(board)

    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Injected parlay board ({len(board.primary_parlays)} parlays) into {json_path}")


if __name__ == "__main__":
    main()
