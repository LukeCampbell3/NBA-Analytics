#!/usr/bin/env python3
"""Publish an explicit no-slate NBA payload during the scheduled offseason."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PLAYER_PREDICTOR_ROOT = SCRIPT_PATH.parent.parent
WORKSPACE_ROOT = SCRIPT_PATH.parents[5]
DEFAULT_CALIBRATOR = (
    PLAYER_PREDICTOR_ROOT / "model/analysis/calibration/selected_board_calibrator.json"
)
DEFAULT_OUTPUT = WORKSPACE_ROOT / "sports/nba/web/data/daily_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--calibrator-json", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_confidence_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload.get("confidence_calibration") if isinstance(payload, dict) else None
    if (
        not isinstance(calibration, dict)
        or calibration.get("status") != "passed"
        or calibration.get("method") != "segment_monotonic_safety"
    ):
        raise ValueError("NBA offseason publication requires the validated confidence calibrator.")
    return calibration


def build_payload(run_date: str, calibration: dict[str, Any]) -> dict[str, Any]:
    resolved_date = date.fromisoformat(run_date)
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": resolved_date.isoformat(),
        "season": resolved_date.year,
        "through_date": resolved_date.isoformat(),
        "published_board_source": "offseason_no_slate",
        "display_board_source": "offseason_no_slate",
        "current_market_rows": 0,
        "model_run_id": "offseason_no_slate",
        "policy_profile": "production_board_objective_b12",
        "publication_status": "suppressed",
        "publication_message": "Board withheld because the NBA is outside its scheduled playing season.",
        "publication_gate": {
            "status": "suppressed",
            "selected_source": None,
            "suppressed_source": "offseason_no_slate",
            "blockers": ["offseason_no_slate"],
        },
        "confidence_calibration": calibration,
        "policy": {},
        "summary": {
            "play_count": 0,
            "avg_expected_win_rate": None,
            "avg_ev": None,
            "avg_edge": None,
            "total_bet_fraction": 0.0,
            "expected_profit_fraction": 0.0,
            "by_target": {},
            "by_recommendation": {},
            "parlay_tagged_plays": 0,
            "parlay_pairs": 0,
        },
        "parlay_summary": {
            "selected_pair_count": 0,
            "tagged_play_count": 0,
            "status": "withheld",
        },
        "parlay_pairs": [],
        "plays": [],
        "shadow_runs": [],
    }


def main() -> None:
    args = parse_args()
    calibration = load_confidence_calibration(args.calibrator_json.resolve())
    payload = build_payload(args.run_date, calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"NBA offseason payload: {args.output}")


if __name__ == "__main__":
    main()
