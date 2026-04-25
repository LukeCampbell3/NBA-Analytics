#!/usr/bin/env python3
"""
Convert the MLB high-precision selection artifacts into the web payload consumed by
the MLB predictions pages.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs" / "20260410"
DEFAULT_CSV = DEFAULT_RUN_DIR / "daily_prediction_pool_20260410_high_precision_predictions.csv"
DEFAULT_SUMMARY = DEFAULT_RUN_DIR / "daily_prediction_pool_20260410_high_precision_predictions_summary.json"
DEFAULT_OUT = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the MLB web prediction payload.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV, help="High-precision selection CSV.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY, help="High-precision selection summary JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Destination web payload JSON.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_splits(source: dict[str, int], total: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key, count in source.items():
        count_int = int(count)
        out[str(key)] = {
            "count": count_int,
            "share": (count_int / total) if total else 0.0,
        }
    return out


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    total = len(rows)

    through_date = max((row.get("Last_History_Date", "") for row in rows), default="")
    plays = []
    for row in rows:
        is_home = to_int(row.get("Is_Home", "0"))
        team = row.get("Team", "")
        opponent = row.get("Opponent", "")
        home_team = team if is_home else opponent
        away_team = opponent if is_home else team
        plays.append(
            {
                "rank": to_int(row.get("Rank")),
                "player": row.get("Player", ""),
                "player_display_name": row.get("Player", ""),
                "player_id": row.get("Player_ID", ""),
                "team": team,
                "opponent": opponent,
                "market_home_team": home_team,
                "market_away_team": away_team,
                "market_date": row.get("Game_Date", ""),
                "commence_time_utc": row.get("Commence_Time_UTC", ""),
                "game_id": row.get("Game_ID", ""),
                "game_status_code": row.get("Game_Status_Code", ""),
                "direction": row.get("Direction", ""),
                "target": row.get("Target", ""),
                "prediction": to_float(row.get("Prediction")),
                "market_line": to_float(row.get("Market_Line")),
                "abs_edge": to_float(row.get("Abs_Edge")),
                "estimated_hit_probability": to_float(row.get("Estimated_Hit_Probability")),
                "estimated_graded_hit_rate": to_float(row.get("Estimated_Graded_Hit_Rate")),
                "precision_score": to_float(row.get("Precision_Score")),
                "confidence_tier": row.get("Confidence_Tier", "consider"),
            }
        )

    payload = {
        "sport": "MLB",
        "board_title": "MLB Prediction Bounties",
        "run_date": rows[0].get("Prediction_Run_Date", "") if rows else "",
        "through_date": through_date,
        "model_run_id": "mlb_high_precision_selector_v1",
        "policy_profile": "high_precision_hits",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "play_count": total,
            "supported_rows": int(summary.get("rows_supported", 0)),
            "rows_after_filters": int(summary.get("rows_after_filters", 0)),
            "rejected_rows": max(0, int(summary.get("rows_supported", 0)) - int(summary.get("rows_after_filters", 0))),
            "avg_expected_hit_rate": float(summary.get("avg_hit_probability", 0.0)),
            "avg_graded_hit_rate": float(summary.get("avg_graded_hit_rate", 0.0)),
            "avg_abs_edge": float(summary.get("avg_abs_edge", 0.0)),
            "avg_precision_score": float(summary.get("avg_precision_score", 0.0)),
        },
        "selection": summary.get("selection", {}),
        "filter_rejections": summary.get("filter_rejections", {}),
        "by_target": build_splits(summary.get("by_target", {}), total),
        "by_direction": build_splits(summary.get("by_direction", {}), total),
        "plays": plays,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote MLB web payload -> {args.output}")


if __name__ == "__main__":
    main()
