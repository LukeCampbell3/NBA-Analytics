#!/usr/bin/env python3
"""
Validate archived MLB final pools against settled historical results.

This script grades each archived high-precision MLB selection using the current
processed MLB files, then reports:

1. overall hit rate across archived final-pool picks
2. hit rate / ROI on bets that were both real-market and side-price confirmed
3. per-date summaries so we can inspect board quality over time
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
PROCESSED_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_REPORT_JSON = REPO_ROOT / "sports" / "validation" / "mlb_historical_final_pool_validation.json"
TARGET_TO_ACTUAL_COL = {
    "H": "H",
    "TB": "TB",
    "R": "R",
    "K": "K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate archived MLB final pools against settled historical results.")
    parser.add_argument("--daily-runs-root", type=Path, default=DAILY_RUNS_ROOT, help="Root directory containing archived MLB daily runs.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT, help="Root directory containing processed MLB player files.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON, help="Destination JSON report path.")
    return parser.parse_args()


def normalize_player_key(value: str) -> str:
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


def american_profit_per_unit(price: float | None) -> float | None:
    if price is None:
        return None
    value = float(price)
    if not math.isfinite(value) or abs(value) < 1e-9:
        return None
    if value > 0:
        return value / 100.0
    return 100.0 / abs(value)


def settled_units(result: str, side_price: float | None) -> float | None:
    profit_if_win = american_profit_per_unit(side_price)
    if profit_if_win is None:
        return None
    if result == "win":
        return float(profit_if_win)
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    return None


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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "play_count": 0,
            "graded_play_count": 0,
            "hit_rate": None,
            "line_placeable_count": 0,
            "price_confirmed_count": 0,
            "priced_play_count": 0,
            "priced_graded_count": 0,
            "priced_hit_rate": None,
            "priced_roi": None,
            "avg_units_per_priced_pool": None,
        }

    graded = frame.loc[frame["result"].isin(["win", "loss"])].copy()
    priced = frame.loc[frame["units"].notna()].copy()
    priced_graded = priced.loc[priced["result"].isin(["win", "loss"])].copy()
    pool_units = priced.groupby("run_date", dropna=False)["units"].sum() if not priced.empty else pd.Series(dtype="float64")
    return {
        "play_count": int(len(frame)),
        "graded_play_count": int(len(graded)),
        "hit_rate": float((graded["result"] == "win").mean()) if not graded.empty else None,
        "line_placeable_count": int(frame["line_placeable"].sum()),
        "price_confirmed_count": int(frame["price_confirmed"].sum()),
        "priced_play_count": int(len(priced)),
        "priced_graded_count": int(len(priced_graded)),
        "priced_hit_rate": float((priced_graded["result"] == "win").mean()) if not priced_graded.empty else None,
        "priced_roi": float(priced["units"].mean()) if not priced.empty else None,
        "avg_units_per_priced_pool": float(pool_units.mean()) if not pool_units.empty else None,
    }


def main() -> None:
    args = parse_args()
    actual_lookup = build_actual_lookup(args.processed_root.resolve())
    rows: list[dict[str, Any]] = []
    by_date: list[dict[str, Any]] = []

    selected_paths = sorted(args.daily_runs_root.glob("*/daily_prediction_pool_*_high_precision_predictions.csv"))
    for path in selected_paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Game_Date" not in frame.columns:
            continue

        date_rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            run_date = str(row.get("Game_Date", ""))[:10]
            player_key = normalize_player_key(row.get("Player_ID") or row.get("Player"))
            game_id = str(row.get("Game_ID", "") or "")
            target = str(row.get("Target", "")).strip().upper()
            direction = str(row.get("Direction", "")).strip().upper()
            market_line = to_float(row.get("Market_Line"))
            if not run_date or not player_key or not game_id or target not in TARGET_TO_ACTUAL_COL or market_line is None:
                continue

            actual = actual_lookup.get((run_date, player_key, target, game_id))
            result = ""
            if actual is not None:
                result = grade_result(float(actual), float(market_line), direction)

            market_source = str(row.get("Market_Source", "")).strip().lower()
            books = int(to_float(row.get("Market_Books")) or 0)
            selected_side_price = to_float(row.get("Market_Over_Price")) if direction == "OVER" else to_float(row.get("Market_Under_Price"))
            line_placeable = bool(market_source == "real" and books > 0)
            price_confirmed = bool(selected_side_price is not None and math.isfinite(selected_side_price) and abs(selected_side_price) > 1e-9)
            units = settled_units(result, selected_side_price) if result else None

            record = {
                "run_date": run_date,
                "player": str(row.get("Player", "")),
                "target": target,
                "direction": direction,
                "market_line": float(market_line),
                "actual": None if actual is None else float(actual),
                "result": result,
                "line_placeable": line_placeable,
                "price_confirmed": price_confirmed,
                "units": units,
                "source_file": str(path),
            }
            rows.append(record)
            date_rows.append(record)

        date_summary = summarize_rows(date_rows)
        date_summary["run_date"] = path.parent.name
        by_date.append(date_summary)

    report = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "daily_runs_root": str(args.daily_runs_root.resolve()),
        "processed_root": str(args.processed_root.resolve()),
        "source_file_count": len(selected_paths),
        "overall": summarize_rows(rows),
        "by_date": by_date,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["overall"]
    print("MLB HISTORICAL FINAL POOL VALIDATION")
    print(f"Play count:              {overall['play_count']}")
    print(f"Graded play count:       {overall['graded_play_count']}")
    print(f"Overall hit rate:        {overall['hit_rate']}")
    print(f"Line placeable count:    {overall['line_placeable_count']}")
    print(f"Price confirmed count:   {overall['price_confirmed_count']}")
    print(f"Priced play count:       {overall['priced_play_count']}")
    print(f"Priced hit rate:         {overall['priced_hit_rate']}")
    print(f"Priced ROI:              {overall['priced_roi']}")
    print(f"Avg units / priced pool: {overall['avg_units_per_priced_pool']}")
    print(f"Report JSON:             {args.report_json}")


if __name__ == "__main__":
    main()
