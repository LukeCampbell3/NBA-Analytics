#!/usr/bin/env python3
"""
Convert the MLB high-precision selection artifacts into the web payload consumed by
the MLB predictions pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
DEFAULT_OUT = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"
DEFAULT_OUT_DIST = REPO_ROOT / "dist" / "mlb" / "data" / "daily_predictions.json"
MLB_MANIFEST_PATH = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB" / "update_manifest_2026.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.parlay_analysis import annotate_parlay_board, evaluate_historical_parlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the MLB web prediction payload.")
    parser.add_argument(
        "--daily-runs-root",
        type=Path,
        default=DEFAULT_DAILY_RUNS_ROOT,
        help="Root directory containing MLB daily prediction run folders.",
    )
    parser.add_argument("--input-csv", type=Path, default=None, help="High-precision selection CSV.")
    parser.add_argument("--summary-json", type=Path, default=None, help="High-precision selection summary JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Destination web payload JSON.")
    parser.add_argument(
        "--output-dist",
        type=Path,
        default=DEFAULT_OUT_DIST,
        help="Optional destination for the published dist payload JSON.",
    )
    return parser.parse_args()


def find_latest_selected_csv(daily_runs_root: Path) -> Path:
    candidates = sorted(
        daily_runs_root.glob("**/daily_prediction_pool_*_high_precision_predictions.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No MLB high-precision selection CSV was found under {daily_runs_root}"
        )
    return candidates[0]


def infer_summary_path(selected_csv: Path) -> Path:
    return selected_csv.with_name(f"{selected_csv.stem}_summary.json")


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
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


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    lam = max(0.0, float(lam))
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    log_p = (-lam) + (k * math.log(lam)) - math.lgamma(k + 1)
    return math.exp(log_p)


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return min(1.0, sum(poisson_pmf(i, lam) for i in range(k + 1)))


def estimate_count_hit_probabilities(prediction: float, market_line: float, direction: str) -> tuple[float, float, float]:
    lam = max(0.0, float(prediction))
    rounded = round(float(market_line))
    is_integer_line = abs(float(market_line) - rounded) < 1e-9

    if is_integer_line:
        push_probability = poisson_pmf(int(rounded), lam)
        if direction == "OVER":
            hit_probability = 1.0 - poisson_cdf(int(rounded), lam)
        else:
            hit_probability = poisson_cdf(int(rounded) - 1, lam)
    else:
        floor_line = math.floor(float(market_line))
        push_probability = 0.0
        if direction == "OVER":
            hit_probability = 1.0 - poisson_cdf(int(floor_line), lam)
        else:
            hit_probability = poisson_cdf(int(floor_line), lam)

    settle_probability = max(1e-9, 1.0 - push_probability)
    graded_hit_rate = hit_probability / settle_probability
    return (
        max(0.0, min(1.0, hit_probability)),
        max(0.0, min(1.0, push_probability)),
        max(0.0, min(1.0, graded_hit_rate)),
    )


def build_mlb_parlay_validation(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"available": False, "reason": f"manifest not found: {manifest_path}"}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "reason": f"failed reading manifest: {exc}"}

    written = manifest.get("written", {})
    if not isinstance(written, dict) or not written:
        return {"available": False, "reason": "manifest does not contain processed MLB player files"}

    target_map = {
        "H": ("H", "Market_H", "H_market_gap"),
        "HR": ("HR", "Market_HR", "HR_market_gap"),
        "RBI": ("RBI", "Market_RBI", "RBI_market_gap"),
    }
    rows: list[dict] = []

    for player_name, item in written.items():
        raw_path = item.get("path")
        if not raw_path:
            continue
        source_path = Path(raw_path)
        if not source_path.exists():
            fallback = manifest_path.parent / player_name / "2026_processed_processed.csv"
            source_path = fallback if fallback.exists() else source_path
        if not source_path.exists():
            continue

        try:
            frame = pd.read_csv(source_path)
        except Exception:
            continue
        if frame.empty:
            continue

        for _, row in frame.iterrows():
            market_date = str(row.get("Date", "")).strip()
            player = str(row.get("Player", "") or player_name).strip()
            team = str(row.get("Team", "")).strip()
            opponent = str(row.get("Opponent", "")).strip()
            game_id = str(row.get("Game_ID", "")).strip()
            for target, (actual_col, market_col, gap_col) in target_map.items():
                try:
                    market_line = float(row.get(market_col))
                    gap = float(row.get(gap_col))
                    actual = float(row.get(actual_col))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(market_line) or not math.isfinite(gap) or not math.isfinite(actual) or abs(gap) < 1e-9:
                    continue
                prediction = market_line + gap
                direction = "OVER" if gap > 0 else "UNDER"
                _, _, graded_hit_rate = estimate_count_hit_probabilities(prediction, market_line, direction)
                if direction == "OVER":
                    result = "win" if actual > market_line else "push" if actual == market_line else "loss"
                else:
                    result = "win" if actual < market_line else "push" if actual == market_line else "loss"
                rows.append(
                    {
                        "market_date": market_date,
                        "player": player,
                        "player_display_name": player,
                        "team": team,
                        "opponent": opponent,
                        "game_id": game_id,
                        "target": target,
                        "direction": direction,
                        "estimated_graded_hit_rate": graded_hit_rate,
                        "result": result,
                    }
                )

    history = pd.DataFrame(rows)
    if history.empty:
        return {"available": False, "reason": "processed MLB history did not yield usable pair rows"}

    summary = evaluate_historical_parlays(
        history,
        sport="mlb",
        date_col="market_date",
        probability_col="estimated_graded_hit_rate",
        result_col="result",
        max_pairs_per_day=1,
    )
    summary["source_manifest"] = str(manifest_path)
    summary["history_row_count"] = int(len(history))
    return summary


def main() -> None:
    args = parse_args()
    selected_csv = args.input_csv.resolve() if args.input_csv else find_latest_selected_csv(args.daily_runs_root.resolve())
    summary_json = args.summary_json.resolve() if args.summary_json else infer_summary_path(selected_csv)
    rows = load_rows(selected_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8-sig"))
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
                "edge": to_float(row.get("Edge")),
                "abs_edge": to_float(row.get("Abs_Edge")),
                "estimated_hit_probability": to_float(row.get("Estimated_Hit_Probability")),
                "estimated_graded_hit_rate": to_float(row.get("Estimated_Graded_Hit_Rate")),
                "precision_score": to_float(row.get("Precision_Score")),
                "value_score": to_float(row.get("Precision_Score")) * to_float(row.get("Abs_Edge")),
                "confidence_tier": row.get("Confidence_Tier", "consider"),
            }
        )

    parlay_payload = annotate_parlay_board(
        plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
    )
    plays = parlay_payload["plays"]

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
            "avg_edge": (sum(to_float(row.get("Edge")) for row in rows) / total) if total else 0.0,
            "avg_abs_edge": float(summary.get("avg_abs_edge", 0.0)),
            "avg_value_score": (sum(to_float(row.get("Precision_Score")) * to_float(row.get("Abs_Edge")) for row in rows) / total) if total else 0.0,
            "avg_precision_score": float(summary.get("avg_precision_score", 0.0)),
        },
        "selection": summary.get("selection", {}),
        "filter_rejections": summary.get("filter_rejections", {}),
        "by_target": build_splits(summary.get("by_target", {}), total),
        "by_direction": build_splits(summary.get("by_direction", {}), total),
        "parlay_summary": parlay_payload["summary"],
        "parlay_pairs": parlay_payload["pairs"],
        "parlay_validation": build_mlb_parlay_validation(MLB_MANIFEST_PATH),
        "plays": plays,
    }
    payload["summary"]["parlay_tagged_plays"] = int(payload["parlay_summary"].get("tagged_play_count", 0))
    payload["summary"]["parlay_pairs"] = int(payload["parlay_summary"].get("selected_pair_count", 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_dist:
        args.output_dist.parent.mkdir(parents=True, exist_ok=True)
        args.output_dist.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote MLB web payload -> {args.output}")
    if args.output_dist:
        print(f"Wrote MLB dist payload -> {args.output_dist}")


if __name__ == "__main__":
    main()
