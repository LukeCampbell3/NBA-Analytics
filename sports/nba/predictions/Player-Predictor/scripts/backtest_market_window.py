#!/usr/bin/env python3
"""
Backtest model-vs-market performance over a historical date window.

This script:
- loads historical market lines from the normalized wide Covers history
- rebuilds predictions with the current predictor for each market row
- resolves actual PTS/TRB/AST from official NBA player logs
- reports hit rate vs market lines over the requested window

It is intentionally lightweight and works even when the structured model
artifacts are unavailable locally; in that case StructuredStackInference
falls back to its artifact-free heuristic mode and the summary records that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelogs

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "inference"))

from build_upcoming_slate import (  # noqa: E402
    DEFAULT_TARGET_PREDICTION_CALIBRATOR,
    build_records,
    load_market_wide,
    normalize_name,
)
from structured_stack_inference import StructuredStackInference  # noqa: E402


TARGET_MAP = {
    "PTS": "PTS",
    "TRB": "REB",
    "AST": "AST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest model-vs-market performance over a date window.")
    parser.add_argument("--season", type=int, required=True, help="Season end year, e.g. 2026 for 2025-26.")
    parser.add_argument("--start-date", type=str, required=True, help="Inclusive market date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, required=True, help="Inclusive market date YYYY-MM-DD.")
    parser.add_argument(
        "--market-history-path",
        type=Path,
        default=REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "history_player_props_wide.parquet",
        help="Normalized historical market wide parquet/csv.",
    )
    parser.add_argument(
        "--target-prediction-calibrator-json",
        type=Path,
        default=DEFAULT_TARGET_PREDICTION_CALIBRATOR,
        help="Optional target-level short-term prediction calibrator JSON.",
    )
    parser.add_argument(
        "--disable-target-prediction-calibration",
        action="store_true",
        help="Disable target-level prediction calibration and keep raw predictor outputs.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable model run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production manifest.")
    parser.add_argument(
        "--actual-season-types",
        nargs="+",
        default=["Regular Season", "Playoffs"],
        help="NBA season types to query for actual outcomes.",
    )
    parser.add_argument(
        "--wide-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "market_window_backtest_wide.csv",
        help="Wide player-date comparison output.",
    )
    parser.add_argument(
        "--rows-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "market_window_backtest_rows.csv",
        help="Long target-level results output.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "market_window_backtest_summary.json",
        help="Summary JSON output.",
    )
    return parser.parse_args()


def resolve_manifest_path(model_dir: Path, run_id: str | None, latest: bool) -> Path | None:
    if run_id:
        return model_dir / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return model_dir / "latest_structured_lstm_stack.json"
    return model_dir / "production_structured_lstm_stack.json"


def load_window_market_history(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    df = load_market_wide(path)
    mask = (df["Market_Date"] >= start_date.normalize()) & (df["Market_Date"] <= end_date.normalize())
    df = df.loc[mask].copy()
    if "Market_Fetched_At_UTC" in df.columns:
        df["Market_Fetched_At_UTC"] = pd.to_datetime(df["Market_Fetched_At_UTC"], errors="coerce", utc=True)
        df = df.sort_values(["Market_Date", "Player", "Market_Fetched_At_UTC"])
    else:
        df = df.sort_values(["Market_Date", "Player"])
    df = df.drop_duplicates(subset=["Market_Date", "Player"], keep="last")
    line_cols = [col for col in ["Market_PTS", "Market_TRB", "Market_AST"] if col in df.columns]
    if line_cols:
        df = df.loc[df[line_cols].notna().any(axis=1)].copy()
    if df.empty:
        raise RuntimeError(f"No market rows found in {path} for window {start_date.date()}..{end_date.date()}.")
    return df.sort_values(["Market_Date", "Player"]).reset_index(drop=True)


def fetch_actual_logs(season: int, season_types: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    season_label = f"{season - 1}-{str(season)[-2:]}"
    frames: list[pd.DataFrame] = []
    for season_type in season_types:
        df = playergamelogs.PlayerGameLogs(
            season_nullable=season_label,
            season_type_nullable=season_type,
            per_mode_simple_nullable="PerGame",
            timeout=30,
        ).get_data_frames()[0]
        if df.empty:
            continue
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()
        df = df.loc[(df["GAME_DATE"] >= start_date.normalize()) & (df["GAME_DATE"] <= end_date.normalize())].copy()
        if df.empty:
            continue
        df["player_key"] = df["PLAYER_NAME"].astype(str).map(normalize_name)
        df["season_type"] = str(season_type)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["GAME_DATE", "PLAYER_ID", "season_type"]).drop_duplicates(
        subset=["GAME_DATE", "PLAYER_ID"],
        keep="last",
    )
    return out.reset_index(drop=True)


def build_actual_lookup(actual_logs: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    if actual_logs.empty:
        return lookup
    for _, row in actual_logs.iterrows():
        date_key = pd.to_datetime(row["GAME_DATE"], errors="coerce")
        if pd.isna(date_key):
            continue
        key = (str(date_key.date()), str(row["player_key"]))
        lookup[key] = {
            "PTS": float(pd.to_numeric(pd.Series([row.get("PTS")]), errors="coerce").fillna(np.nan).iloc[0]),
            "TRB": float(pd.to_numeric(pd.Series([row.get("REB")]), errors="coerce").fillna(np.nan).iloc[0]),
            "AST": float(pd.to_numeric(pd.Series([row.get("AST")]), errors="coerce").fillna(np.nan).iloc[0]),
        }
    return lookup


def classify_result(prediction: float, market_line: float, actual: float) -> tuple[str, str]:
    if pd.isna(prediction) or pd.isna(market_line) or pd.isna(actual):
        return "NO_TRADE", "missing"
    if float(prediction) > float(market_line):
        direction = "OVER"
    elif float(prediction) < float(market_line):
        direction = "UNDER"
    else:
        return "NO_TRADE", "push"
    if abs(float(actual) - float(market_line)) <= 1e-9:
        return direction, "push"
    if direction == "OVER":
        return direction, "win" if float(actual) > float(market_line) else "loss"
    return direction, "win" if float(actual) < float(market_line) else "loss"


def summarize_rows(rows_df: pd.DataFrame, group_col: str | None = None) -> list[dict[str, Any]]:
    if rows_df.empty:
        return []
    summaries: list[dict[str, Any]] = []
    grouped = [(None, rows_df)] if group_col is None else rows_df.groupby(group_col, dropna=False)
    for key, part in grouped:
        resolved = part.loc[part["result"].isin(["win", "loss"])].copy()
        wins = int((resolved["result"] == "win").sum())
        losses = int((resolved["result"] == "loss").sum())
        pushes = int((part["result"] == "push").sum())
        missing = int((part["result"] == "missing").sum())
        resolved_count = int(len(resolved))
        summary = {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "missing": missing,
            "resolved": resolved_count,
            "hit_rate": float(wins / resolved_count) if resolved_count > 0 else np.nan,
        }
        if group_col is not None:
            summary[group_col] = key
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    if start_date > end_date:
        raise ValueError(f"Invalid window: {start_date.date()} is after {end_date.date()}.")

    market_df = load_window_market_history(args.market_history_path.resolve(), start_date, end_date)
    model_dir = REPO_ROOT / "model"
    manifest_path = resolve_manifest_path(model_dir, args.run_id, args.latest)
    predictor = StructuredStackInference(model_dir=str(model_dir), manifest_path=manifest_path)
    calibrator_path = None if args.disable_target_prediction_calibration else args.target_prediction_calibrator_json
    records, skipped = build_records(
        predictor,
        market_df,
        args.season,
        target_prediction_calibrator_path=calibrator_path,
    )
    if not records:
        raise RuntimeError(f"No prediction rows built. Skipped={len(skipped)} sample={skipped[:5]}")

    wide_df = pd.DataFrame.from_records(records).sort_values(["market_date", "player"]).reset_index(drop=True)
    wide_df["player_key"] = wide_df["csv"].map(lambda value: Path(str(value)).parent.name if str(value) not in {"", "nan"} else normalize_name(str(value)))

    actual_logs = fetch_actual_logs(args.season, [str(item) for item in args.actual_season_types], start_date, end_date)
    actual_lookup = build_actual_lookup(actual_logs)

    for target in ["PTS", "TRB", "AST"]:
        actual_values: list[float] = []
        for _, row in wide_df.iterrows():
            lookup_key = (str(row["market_date"]), str(row["player_key"]))
            actual_map = actual_lookup.get(lookup_key, {})
            actual_values.append(float(actual_map.get(target, np.nan)))
        wide_df[f"actual_{target}"] = pd.Series(actual_values, index=wide_df.index, dtype="float64")

    row_records: list[dict[str, Any]] = []
    for _, row in wide_df.iterrows():
        for target in ["PTS", "TRB", "AST"]:
            pred = pd.to_numeric(pd.Series([row.get(f"pred_{target}")]), errors="coerce").fillna(np.nan).iloc[0]
            market = pd.to_numeric(pd.Series([row.get(f"market_{target}")]), errors="coerce").fillna(np.nan).iloc[0]
            actual = pd.to_numeric(pd.Series([row.get(f"actual_{target}")]), errors="coerce").fillna(np.nan).iloc[0]
            direction, result = classify_result(float(pred) if pd.notna(pred) else np.nan, float(market) if pd.notna(market) else np.nan, float(actual) if pd.notna(actual) else np.nan)
            row_records.append(
                {
                    "market_date": str(row.get("market_date")),
                    "player": str(row.get("player")),
                    "player_key": str(row.get("player_key")),
                    "target": target,
                    "direction": direction,
                    "prediction": float(pred) if pd.notna(pred) else np.nan,
                    "market_line": float(market) if pd.notna(market) else np.nan,
                    "actual": float(actual) if pd.notna(actual) else np.nan,
                    "edge": float(pred - market) if pd.notna(pred) and pd.notna(market) else np.nan,
                    "result": result,
                }
            )

    rows_df = pd.DataFrame.from_records(row_records)
    overall_summary = summarize_rows(rows_df)
    by_target_summary = summarize_rows(rows_df, group_col="target")
    by_date_summary = summarize_rows(rows_df, group_col="market_date")

    artifact_free = bool(getattr(predictor, "artifact_free", False))
    summary_payload = {
        "window": {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
        },
        "market_rows": int(len(wide_df)),
        "target_rows": int(len(rows_df)),
        "predictor": {
            "artifact_free": artifact_free,
            "artifact_free_reason": getattr(predictor, "artifact_free_reason", None),
            "run_id": predictor.metadata.get("run_id") if isinstance(getattr(predictor, "metadata", None), dict) else None,
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "target_prediction_calibrator_json": str(calibrator_path) if calibrator_path is not None else None,
        },
        "actual_logs_rows": int(len(actual_logs)),
        "skipped_market_rows": int(len(skipped)),
        "overall": overall_summary[0] if overall_summary else {},
        "by_target": by_target_summary,
        "by_date": by_date_summary,
    }

    args.wide_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.rows_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.wide_csv_out, index=False)
    rows_df.to_csv(args.rows_csv_out, index=False)
    args.summary_json_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    overall = summary_payload["overall"]
    print("\n" + "=" * 88)
    print("MARKET WINDOW BACKTEST")
    print("=" * 88)
    print(f"Window:                 {start_date.date()} -> {end_date.date()}")
    print(f"Market rows:            {len(wide_df)}")
    print(f"Target rows:            {len(rows_df)}")
    print(f"Resolved:               {overall.get('resolved', 0)}")
    print(f"Wins-Losses-Pushes:     {overall.get('wins', 0)}-{overall.get('losses', 0)}-{overall.get('pushes', 0)}")
    print(f"Hit rate:               {overall.get('hit_rate', np.nan):.4f}")
    print(f"Predictor artifactfree: {artifact_free}")
    print(f"Wide CSV:               {args.wide_csv_out}")
    print(f"Rows CSV:               {args.rows_csv_out}")
    print(f"Summary JSON:           {args.summary_json_out}")


if __name__ == "__main__":
    main()
