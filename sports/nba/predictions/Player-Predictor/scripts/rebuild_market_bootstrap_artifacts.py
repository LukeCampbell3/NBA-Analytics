#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "inference"))

from backtest_market_window import (  # noqa: E402
    build_actual_lookup,
    classify_result,
    fetch_actual_logs,
    load_window_market_history,
    resolve_manifest_path,
)
from build_upcoming_slate import build_records  # noqa: E402
from select_market_plays import build_history_lookup, build_play_rows  # noqa: E402
from decision_engine.line_decision import LineDecisionConfig, build_line_decision_lookup  # noqa: E402
from structured_stack_inference import StructuredStackInference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild resolved market-history and selector replay artifacts.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument(
        "--market-history-path",
        type=Path,
        default=REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "history_player_props_wide.parquet",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument(
        "--history-wide-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv",
    )
    parser.add_argument(
        "--target-rows-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "market_window_backtest_rows.csv",
    )
    parser.add_argument(
        "--selector-rows-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "selector_replay_rows.csv",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "market_bootstrap_summary.json",
    )
    return parser.parse_args()


def parse_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        mins, secs = text.split(":", 1)
        try:
            return float(mins) + float(secs) / 60.0
        except Exception:
            return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def build_actual_detail_lookup(actual_logs: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if actual_logs.empty:
        return out
    working = actual_logs.copy()
    working["GAME_DATE"] = pd.to_datetime(working["GAME_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["player_key"] = working["PLAYER_NAME"].astype(str).str.replace(" ", "_", regex=False).str.replace(".", "", regex=False).str.replace("'", "", regex=False)
    working["minutes"] = working["MIN"].map(parse_minutes)
    working["did_not_play"] = (pd.to_numeric(working["minutes"], errors="coerce").fillna(0.0) <= 0.0).astype(float)
    for _, row in working.iterrows():
        key = (str(row["GAME_DATE"]), str(row["player_key"]))
        out[key] = {
            "minutes": float(pd.to_numeric(pd.Series([row.get("minutes")]), errors="coerce").fillna(0.0).iloc[0]),
            "did_not_play": float(pd.to_numeric(pd.Series([row.get("did_not_play")]), errors="coerce").fillna(0.0).iloc[0]),
        }
    return out


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    market_df = load_window_market_history(args.market_history_path.resolve(), start_date, end_date)
    predictor = StructuredStackInference(
        model_dir=str(REPO_ROOT / "model"),
        manifest_path=resolve_manifest_path(REPO_ROOT / "model", args.run_id, args.latest),
    )
    records, skipped = build_records(
        predictor,
        market_df,
        args.season,
        target_prediction_calibrator_path=None,
    )
    if not records:
        raise RuntimeError(f"No market records were built. Skipped={len(skipped)}")

    wide_df = pd.DataFrame.from_records(records).sort_values(["market_date", "player"]).reset_index(drop=True)
    wide_df["player_key"] = wide_df["csv"].map(lambda value: Path(str(value)).parent.name if pd.notna(value) else "")

    actual_logs = fetch_actual_logs(args.season, ["Regular Season", "Playoffs"], start_date, end_date)
    actual_lookup = build_actual_lookup(actual_logs)
    actual_detail_lookup = build_actual_detail_lookup(actual_logs)
    for target in ["PTS", "TRB", "AST"]:
        actual_values: list[float] = []
        for _, row in wide_df.iterrows():
            actual_map = actual_lookup.get((str(row["market_date"]), str(row["player_key"])), {})
            actual_values.append(float(actual_map.get(target, np.nan)))
        wide_df[f"actual_{target}"] = pd.Series(actual_values, index=wide_df.index, dtype="float64")

    minutes_values: list[float] = []
    dnp_values: list[float] = []
    for _, row in wide_df.iterrows():
        detail = actual_detail_lookup.get((str(row["market_date"]), str(row["player_key"])), {})
        minutes_values.append(float(detail.get("minutes", 0.0)))
        dnp_values.append(float(detail.get("did_not_play", 0.0)))
    wide_df["minutes"] = pd.Series(minutes_values, index=wide_df.index, dtype="float64")
    wide_df["did_not_play"] = pd.Series(dnp_values, index=wide_df.index, dtype="float64")

    target_rows: list[dict] = []
    for _, row in wide_df.iterrows():
        for target in ["PTS", "TRB", "AST"]:
            pred = float(pd.to_numeric(pd.Series([row.get(f"pred_{target}")]), errors="coerce").fillna(np.nan).iloc[0])
            market_line = float(pd.to_numeric(pd.Series([row.get(f"market_{target}")]), errors="coerce").fillna(np.nan).iloc[0])
            actual = float(pd.to_numeric(pd.Series([row.get(f"actual_{target}")]), errors="coerce").fillna(np.nan).iloc[0])
            direction, result = classify_result(pred, market_line, actual)
            target_rows.append(
                {
                    "run_date": str(row["market_date"]),
                    "market_date": str(row["market_date"]),
                    "player": str(row["player"]),
                    "target": target,
                    "direction": direction,
                    "prediction": pred,
                    "market_line": market_line,
                    "actual": actual,
                    "edge": float(pred - market_line) if np.isfinite(pred) and np.isfinite(market_line) else np.nan,
                    "result": result,
                }
            )
    target_rows_df = pd.DataFrame.from_records(target_rows)

    selector_rows: list[dict] = []
    line_cfg = LineDecisionConfig()
    market_dates = sorted(pd.to_datetime(wide_df["market_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    for market_date in market_dates:
        current_slate = wide_df.loc[wide_df["market_date"] == market_date].copy()
        history_slice = wide_df.loc[pd.to_datetime(wide_df["market_date"], errors="coerce") < pd.Timestamp(market_date)].copy()
        if current_slate.empty or history_slice.empty:
            continue
        history_lookup = build_history_lookup(history_slice)
        line_lookup = build_line_decision_lookup(history_slice)
        plays = build_play_rows(
            current_slate,
            history_lookup,
            line_decision_lookup=line_lookup,
            line_decision_enabled=True,
            line_decision_config=line_cfg,
        )
        if plays.empty:
            continue
        actual_cols = ["player", "market_date", "actual_PTS", "actual_TRB", "actual_AST"]
        joined = plays.merge(current_slate[actual_cols], how="left", on=["player", "market_date"])
        for _, row in joined.iterrows():
            target = str(row["target"]).upper().strip()
            actual_value = float(pd.to_numeric(pd.Series([row.get(f"actual_{target}")]), errors="coerce").fillna(np.nan).iloc[0])
            direction, result = classify_result(
                float(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").fillna(np.nan).iloc[0]),
                float(pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").fillna(np.nan).iloc[0]),
                actual_value,
            )
            selector_rows.append(
                {
                    "run_date": market_date,
                    "market_date": market_date,
                    "player": str(row["player"]),
                    "target": target,
                    "direction": direction,
                    "market_line": float(pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").fillna(np.nan).iloc[0]),
                    "prediction": float(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").fillna(np.nan).iloc[0]),
                    "expected_win_rate": float(pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").fillna(np.nan).iloc[0]),
                    "board_play_win_prob": float(pd.to_numeric(pd.Series([row.get("line_chosen_direction_prob")]), errors="coerce").fillna(np.nan).iloc[0]),
                    "ev": float(pd.to_numeric(pd.Series([row.get("ev")]), errors="coerce").fillna(np.nan).iloc[0]) if "ev" in row.index else np.nan,
                    "recommendation": str(row.get("recommendation", "")),
                    "actual": actual_value,
                    "result": result,
                }
            )

    selector_rows_df = pd.DataFrame.from_records(selector_rows)

    for path in [args.history_wide_out, args.target_rows_out, args.selector_rows_out, args.summary_json_out]:
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(args.history_wide_out.resolve(), index=False)
    target_rows_df.to_csv(args.target_rows_out.resolve(), index=False)
    selector_rows_df.to_csv(args.selector_rows_out.resolve(), index=False)

    summary = {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "predictor_run_id": predictor.metadata.get("run_id"),
        "predictor_artifact_free": bool(getattr(predictor, "artifact_free", False)),
        "wide_rows": int(len(wide_df)),
        "target_rows": int(len(target_rows_df)),
        "selector_rows": int(len(selector_rows_df)),
        "skipped_rows": int(len(skipped)),
    }
    args.summary_json_out.resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("MARKET BOOTSTRAP ARTIFACTS REBUILT")
    print("=" * 88)
    print(f"History wide:  {args.history_wide_out.resolve()}")
    print(f"Target rows:   {args.target_rows_out.resolve()}")
    print(f"Selector rows: {args.selector_rows_out.resolve()}")
    print(f"Summary JSON:  {args.summary_json_out.resolve()}")
    print(f"Predictor run: {predictor.metadata.get('run_id')}")
    print(f"Wide rows:     {len(wide_df)}")
    print(f"Selector rows: {len(selector_rows_df)}")


if __name__ == "__main__":
    main()
