#!/usr/bin/env python3
"""Generate prediction and settled CSV files for MLB historical backtesting."""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_TO_ACTUAL_COL = {
    "H": "H",
    "TB": "TB",
    "R": "R",
    "K": "K",
}

SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_DAILY_RUNS_ROOT = SCRIPT_ROOT.parent / "data" / "predictions" / "daily_runs"
DEFAULT_PROCESSED_ROOT = SCRIPT_ROOT.parent.parent.parent / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_PREDICTIONS_OUT = SCRIPT_ROOT.parent / "data" / "predictions" / "mlb_historical_backtest_predictions.csv"
DEFAULT_SETTLED_OUT = SCRIPT_ROOT.parent / "data" / "predictions" / "mlb_historical_backtest_settled.csv"
DEFAULT_CALIBRATION_FILE = SCRIPT_ROOT.parent / "data" / "predictions" / "calibration" / "historical_pool_universe_2026.csv"


def normalize_player_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def grade_result(actual: float, market_line: float, direction: str) -> str:
    if direction == "OVER":
        if actual > market_line:
            return "win"
        if actual == market_line:
            return "push"
        return "loss"
    if actual < market_line:
        return "win"
    if actual == market_line:
        return "push"
    return "loss"


def build_actual_lookup(processed_root: Path) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    usecols = ["Date", "Player", "Game_ID", "H", "TB", "R", "K"]
    for path in processed_root.glob("*/2026_processed_processed.csv"):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in usecols)
        except Exception:
            continue
        if frame.empty or "Date" not in frame.columns or "Player" not in frame.columns or "Game_ID" not in frame.columns:
            continue

        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["Player_Key"] = frame["Player"].astype(str).map(normalize_player_key)
        frame["Game_ID"] = frame["Game_ID"].astype(str)

        for target, actual_col in TARGET_TO_ACTUAL_COL.items():
            if actual_col not in frame.columns:
                continue
            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            mask = frame["Date"].notna() & frame["Player_Key"].ne("") & frame["Game_ID"].ne("") & actual.notna()
            if not bool(mask.any()):
                continue
            part = frame.loc[mask, ["Date", "Player_Key", "Game_ID"]].copy()
            part["Actual"] = actual.loc[mask].astype(float)
            for _, row in part.iterrows():
                lookup[(str(row["Date"]), str(row["Player_Key"]), target, str(row["Game_ID"]))] = float(row["Actual"])
    return lookup


def infer_direction(prediction: float, market_line: float) -> str:
    return "OVER" if prediction > market_line else "UNDER"


def build_calibration_backtest(calibration_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    df = pd.read_csv(calibration_path)
    preds: list[dict[str, object]] = []
    settled: list[dict[str, object]] = []

    for _, row in df.iterrows():
        game_id = str(row.get("Game_ID", ""))
        player_key = normalize_player_key(row.get("Player_ID") or row.get("Player"))
        target = str(row.get("Target", "")).strip().upper()
        market_line = to_float(row.get("Market_Line"))
        prediction_value = to_float(row.get("Prediction"))
        over_price = to_float(row.get("Market_Over_Price"))
        under_price = to_float(row.get("Market_Under_Price"))
        actual = to_float(row.get("Actual"))
        run_date = str(row.get("Prediction_Run_Date") or row.get("Game_Date", ""))[:10]

        if not run_date or not player_key or not game_id or target not in TARGET_TO_ACTUAL_COL:
            continue
        if market_line is None or prediction_value is None or actual is None:
            continue

        direction = infer_direction(prediction_value, market_line)
        american_odds = over_price if direction == "OVER" else under_price
        if american_odds is None:
            continue

        result = grade_result(actual, market_line, direction)
        if result == "push":
            continue

        event_id = f"{run_date}|{player_key}|{target}|{direction}|{game_id}"
        actual_outcome = 1 if result == "win" else 0

        preds.append({
            "event_id": event_id,
            "predicted_prob": prediction_value,
            "american_odds": american_odds,
            "prediction_time": run_date,
            "player": str(row.get("Player", "")),
            "target": target,
            "direction": direction,
            "market_line": market_line,
            "game_id": game_id,
            "source_file": str(calibration_path),
        })
        settled.append({"event_id": event_id, "actual_outcome": actual_outcome})

    return preds, sorted({row["event_id"]: row for row in settled}.values(), key=lambda x: x["event_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MLB prediction and settled CSVs for backtesting.")
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS_ROOT, help="Root for archived MLB daily runs.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT, help="Root for MLB processed player files.")
    parser.add_argument("--calibration-file", type=Path, default=None, help="Optional calibration CSV containing market prices and actual outcomes.")
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS_OUT, help="Output predictions CSV path.")
    parser.add_argument("--settled-out", type=Path, default=DEFAULT_SETTLED_OUT, help="Output settled outcome CSV path.")
    args = parser.parse_args()

    predictions_out = args.predictions_out.resolve()
    settled_out = args.settled_out.resolve()

    preds: list[dict[str, object]] = []
    settled: list[dict[str, object]] = []

    if args.calibration_file:
        calibration_path = args.calibration_file.resolve()
        preds, settled = build_calibration_backtest(calibration_path)
    else:
        daily_runs_root = args.daily_runs_root.resolve()
        processed_root = args.processed_root.resolve()
        actual_lookup = build_actual_lookup(processed_root)
        selected_paths = sorted(daily_runs_root.glob("*/daily_prediction_pool_*_high_precision_predictions.csv"))

        for path in selected_paths:
            frame = pd.read_csv(path)
            if frame.empty or "Game_Date" not in frame.columns:
                continue

            for _, row in frame.iterrows():
                run_date = str(row.get("Game_Date", ""))[:10]
                player_key = normalize_player_key(row.get("Player_ID") or row.get("Player"))
                game_id = str(row.get("Game_ID", ""))
                target = str(row.get("Target", "")).strip().upper()
                direction = str(row.get("Direction", "")).strip().upper()
                market_line = to_float(row.get("Market_Line"))
                if not run_date or not player_key or not game_id or target not in TARGET_TO_ACTUAL_COL or market_line is None:
                    continue

                event_id = f"{run_date}|{player_key}|{target}|{direction}|{game_id}"
                predicted_prob = to_float(row.get("Prediction"))
                if direction == "OVER":
                    american_odds = to_float(row.get("Market_Over_Price"))
                else:
                    american_odds = to_float(row.get("Market_Under_Price"))
                actual = actual_lookup.get((run_date, player_key, target, game_id))
                if actual is None:
                    continue

                result = grade_result(actual, market_line, direction)
                if result == "push":
                    continue

                actual_outcome = 1 if result == "win" else 0

                preds.append({
                    "event_id": event_id,
                    "predicted_prob": predicted_prob,
                    "american_odds": american_odds,
                    "prediction_time": run_date,
                    "player": str(row.get("Player", "")),
                    "target": target,
                    "direction": direction,
                    "market_line": market_line,
                    "game_id": game_id,
                    "source_file": str(path),
                })
                settled.append({"event_id": event_id, "actual_outcome": actual_outcome})

    if preds:
        pd.DataFrame(preds).to_csv(predictions_out, index=False)
    if settled:
        pd.DataFrame(settled).drop_duplicates(subset=["event_id"]).to_csv(settled_out, index=False)

    print(f"Wrote predictions: {predictions_out} ({len(preds)} rows)")
    print(f"Wrote settled: {settled_out} ({len(settled)} unique event_ids)")


if __name__ == "__main__":
    main()
