#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.inference import DailyInferenceConfig, generate_daily_prediction_pool


def _safe_float(value):
    try:
        out = float(value)
        if np.isnan(out):
            return None
        return out
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the strongest MLB predictions from a daily prediction pool."
    )
    parser.add_argument(
        "--pool-csv",
        type=Path,
        default=None,
        help="Path to daily_prediction_pool_YYYYMMDD.csv.",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="If provided, first build pool for this date (YYYY-MM-DD) then select best rows.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year for pool build mode (defaults to run-date year).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Processed feature directory used in pool build mode.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Model directory used in pool build mode.",
    )
    parser.add_argument(
        "--daily-runs-root",
        type=Path,
        default=ROOT / "data" / "predictions" / "daily_runs",
        help="Daily run root used in pool build mode.",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default="R",
        help="MLB schedule game type used in pool build mode.",
    )
    parser.add_argument(
        "--inference-min-history-rows",
        type=int,
        default=5,
        help="Minimum history rows for candidate generation when building pool.",
    )
    parser.add_argument(
        "--force-rebuild-pool",
        action="store_true",
        help="Force rebuild of pool when using --run-date, even if CSV already exists.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults next to pool CSV.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output summary JSON path. Defaults next to pool CSV.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Maximum number of best predictions to keep.",
    )
    parser.add_argument(
        "--min-abs-edge",
        type=float,
        default=0.35,
        help="Minimum absolute model-vs-market edge to keep.",
    )
    parser.add_argument(
        "--min-history-rows",
        type=int,
        default=10,
        help="Minimum player history rows required.",
    )
    parser.add_argument(
        "--max-per-player",
        type=int,
        default=1,
        help="Maximum kept rows per player.",
    )
    parser.add_argument(
        "--allow-baseline",
        action="store_true",
        help="Allow baseline-selected rows (default drops baseline-only rows).",
    )
    return parser.parse_args()


def _resolve_or_build_pool(args: argparse.Namespace) -> tuple[Path, dict | None]:
    if args.pool_csv is not None:
        pool_csv = args.pool_csv.resolve()
        if not pool_csv.exists():
            raise FileNotFoundError(f"Pool CSV not found: {pool_csv}")
        return pool_csv, None

    if not args.run_date:
        raise ValueError("Provide either --pool-csv or --run-date.")

    run_date = pd.to_datetime(str(args.run_date), errors="coerce")
    if pd.isna(run_date):
        raise ValueError(f"Invalid --run-date: {args.run_date}")
    run_date = run_date.normalize()
    run_stamp = run_date.strftime("%Y%m%d")
    daily_runs_root = args.daily_runs_root.resolve()
    out_dir = (daily_runs_root / run_stamp).resolve()
    expected_pool = out_dir / f"daily_prediction_pool_{run_stamp}.csv"

    if expected_pool.exists() and not bool(args.force_rebuild_pool):
        return expected_pool, None

    season = int(args.season) if args.season is not None else int(run_date.year)
    pool_summary = generate_daily_prediction_pool(
        DailyInferenceConfig(
            run_date=run_date.strftime("%Y-%m-%d"),
            season=season,
            processed_dir=args.processed_dir.resolve(),
            model_dir=args.model_dir.resolve(),
            out_dir=out_dir,
            game_type=str(args.game_type),
            min_history_rows=int(args.inference_min_history_rows),
        )
    )
    pool_csv = Path(str(pool_summary["pool_csv"])).resolve()
    if not pool_csv.exists():
        raise FileNotFoundError(f"Built pool CSV is missing: {pool_csv}")
    return pool_csv, pool_summary


def main() -> None:
    args = parse_args()
    pool_csv, pool_summary = _resolve_or_build_pool(args)

    df = pd.read_csv(pool_csv)
    required = [
        "Player",
        "Player_Type",
        "Target",
        "Prediction",
        "Market_Line",
        "Edge",
        "History_Rows",
        "Model_Selected",
        "Model_Val_MAE",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Pool CSV is missing required columns: {missing}")

    work = df.copy()
    rows_before_cleanup = int(len(work))
    work["Prediction"] = pd.to_numeric(work["Prediction"], errors="coerce")
    work["Market_Line"] = pd.to_numeric(work["Market_Line"], errors="coerce")
    work["Edge"] = pd.to_numeric(work["Edge"], errors="coerce")
    work["History_Rows"] = pd.to_numeric(work["History_Rows"], errors="coerce")
    work["Model_Val_MAE"] = pd.to_numeric(work["Model_Val_MAE"], errors="coerce")
    work = work.loc[
        work["Prediction"].notna()
        & work["Market_Line"].notna()
        & work["Edge"].notna()
        & work["History_Rows"].notna()
    ].copy()
    rows_after_numeric_cleanup = int(len(work))

    work["Abs_Edge"] = work["Edge"].abs()
    work["Direction"] = np.where(work["Edge"] > 0.0, "OVER", np.where(work["Edge"] < 0.0, "UNDER", "PUSH"))
    work["Model_Selected"] = work["Model_Selected"].astype(str).str.lower().str.strip()
    work["Target"] = work["Target"].astype(str).str.upper().str.strip()

    if not args.allow_baseline:
        work = work.loc[work["Model_Selected"] != "baseline"].copy()

    work = work.loc[
        (work["Abs_Edge"] >= float(args.min_abs_edge))
        & (work["History_Rows"] >= int(args.min_history_rows))
        & work["Direction"].isin(["OVER", "UNDER"])
    ].copy()

    if work.empty:
        raise RuntimeError(
            "No rows passed best-pick filters. "
            "Try lowering --min-abs-edge or --min-history-rows."
        )

    mae_floor = 1e-6
    hist_weight = np.log1p(work["History_Rows"].astype(float))
    work["Quality_Score"] = (work["Abs_Edge"] / work["Model_Val_MAE"].clip(lower=mae_floor)) * hist_weight

    work = work.sort_values(
        ["Quality_Score", "Abs_Edge", "History_Rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    max_per_player = int(max(1, args.max_per_player))
    work["__rank_player"] = work.groupby("Player", sort=False).cumcount() + 1
    work = work.loc[work["__rank_player"] <= max_per_player].copy()
    work = work.drop(columns=["__rank_player"])
    work = work.head(int(max(1, args.top_n))).copy()
    work.insert(0, "Rank", np.arange(1, len(work) + 1, dtype=int))

    default_out_csv = pool_csv.with_name(f"{pool_csv.stem}_best_predictions.csv")
    default_out_json = pool_csv.with_name(f"{pool_csv.stem}_best_predictions_summary.json")
    out_csv = args.out_csv.resolve() if args.out_csv is not None else default_out_csv
    out_json = args.out_json.resolve() if args.out_json is not None else default_out_json
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "Rank",
        "Prediction_Run_Date",
        "Game_Date",
        "Commence_Time_UTC",
        "Game_ID",
        "Player",
        "Player_ID",
        "Player_Type",
        "Team",
        "Opponent",
        "Is_Home",
        "Target",
        "Direction",
        "Prediction",
        "Market_Line",
        "Edge",
        "Abs_Edge",
        "History_Rows",
        "Last_History_Date",
        "Model_Selected",
        "Model_Members",
        "Model_Val_MAE",
        "Model_Val_RMSE",
        "Quality_Score",
    ]
    present_cols = [col for col in keep_cols if col in work.columns]
    out_df = work[present_cols].copy()
    out_df.to_csv(out_csv, index=False)

    summary = {
        "pool_csv": str(pool_csv),
        "out_csv": str(out_csv),
        "rows_in_pool": int(len(df)),
        "rows_before_cleanup": rows_before_cleanup,
        "rows_after_numeric_cleanup": rows_after_numeric_cleanup,
        "selection": {
            "top_n": int(args.top_n),
            "min_abs_edge": float(args.min_abs_edge),
            "min_history_rows": int(args.min_history_rows),
            "max_per_player": int(args.max_per_player),
            "allow_baseline": bool(args.allow_baseline),
        },
        "pool_build": {
            "built_in_this_run": bool(pool_summary is not None),
            "run_date": str(args.run_date) if args.run_date else None,
            "season": int(args.season) if args.season is not None else None,
            "daily_runs_root": str(args.daily_runs_root.resolve()) if args.run_date else None,
            "summary": pool_summary,
        },
        "rows_selected": int(len(out_df)),
        "avg_abs_edge": _safe_float(out_df["Abs_Edge"].mean()) if "Abs_Edge" in out_df.columns else None,
        "avg_quality_score": _safe_float(out_df["Quality_Score"].mean()) if "Quality_Score" in out_df.columns else None,
        "by_target": out_df["Target"].value_counts().to_dict() if "Target" in out_df.columns else {},
        "by_direction": out_df["Direction"].value_counts().to_dict() if "Direction" in out_df.columns else {},
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
