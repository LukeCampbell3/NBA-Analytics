#!/usr/bin/env python3
"""
Daily unattended market pipeline runner.

This script is designed to run once per day, typically around 2am local time.
It performs:
1. refresh current-season official game logs through yesterday
2. collect recent Covers prop lines for both recent historical games and the
   current/upcoming slate
3. align historical market lines onto the current season processed files
4. build a filtered current-slate market snapshot
5. run the market decision pipeline and write dated outputs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_ROOT = REPO_ROOT.parent
MARKET_ROOT = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba"
ANALYSIS_ROOT = REPO_ROOT / "model" / "analysis" / "daily_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily market data + prediction pipeline.")
    parser.add_argument("--season", type=int, default=None, help="Season end year. Defaults from current date.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production for the final board.")
    parser.add_argument("--history-csv", type=Path, default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv", help="Historical row-level backtest CSV for edge calibration.")
    parser.add_argument("--lookback-days", type=int, default=10, help="How many recent days of historical market lines to collect.")
    parser.add_argument("--future-days", type=int, default=2, help="How many days ahead of today to keep in the current slate snapshot.")
    parser.add_argument("--collect-scan-count", type=int, default=500, help="Maximum Covers matchup ids to scan for the nightly collection window.")
    parser.add_argument("--run-date", type=str, default=None, help="Optional YYYY-MM-DD override for local run date.")
    parser.add_argument("--skip-update-data", action="store_true", help="Skip official game-log refresh.")
    parser.add_argument("--skip-collect-market", action="store_true", help="Skip Covers market collection.")
    parser.add_argument("--skip-align", action="store_true", help="Skip market alignment onto processed files.")
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow market pipeline to run with heuristic-only predictions when model load fails.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use for child steps.")
    return parser.parse_args()


def infer_season(local_date: pd.Timestamp) -> int:
    return local_date.year + 1 if local_date.month >= 9 else local_date.year


def run_step(label: str, args: list[str]) -> None:
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    print("Command:", " ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def filter_current_market_snapshot(source_path: Path, out_path: Path, run_date: pd.Timestamp, future_days: int) -> tuple[int, dict]:
    if not source_path.exists():
        raise FileNotFoundError(f"Market snapshot not found: {source_path}")
    if source_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        df = pd.read_csv(source_path)
    if df.empty:
        raise RuntimeError(f"Market snapshot is empty: {source_path}")
    if "Market_Date" not in df.columns:
        raise RuntimeError(f"Market snapshot is missing Market_Date: {source_path}")

    market_dates = pd.to_datetime(df["Market_Date"], errors="coerce")
    start_date = run_date.normalize()
    end_date = (run_date + pd.Timedelta(days=int(future_days))).normalize()
    filtered = df.loc[(market_dates >= start_date) & (market_dates <= end_date)].copy()
    if filtered.empty:
        fallback_date = market_dates.max()
        if pd.isna(fallback_date):
            raise RuntimeError(f"No valid Market_Date values found in {source_path}")
        filtered = df.loc[market_dates == fallback_date].copy()
        if filtered.empty:
            raise RuntimeError(
                f"No current/upcoming market rows found between {start_date.date()} and {end_date.date()} in {source_path}"
            )
        snapshot_meta = {
            "mode": "stale_fallback",
            "requested_start_date": str(start_date.date()),
            "requested_end_date": str(end_date.date()),
            "selected_market_date": str(pd.Timestamp(fallback_date).date()),
            "selected_row_count": int(len(filtered)),
        }
    else:
        snapshot_meta = {
            "mode": "requested_window",
            "requested_start_date": str(start_date.date()),
            "requested_end_date": str(end_date.date()),
            "selected_market_date_min": str(pd.Timestamp(filtered["Market_Date"].min()).date()),
            "selected_market_date_max": str(pd.Timestamp(filtered["Market_Date"].max()).date()),
            "selected_row_count": int(len(filtered)),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        filtered.to_parquet(out_path, index=False)
    else:
        filtered.to_csv(out_path, index=False)
    return int(len(filtered)), snapshot_meta


def main() -> None:
    args = parse_args()
    local_date = pd.Timestamp(args.run_date).normalize() if args.run_date else pd.Timestamp.now().normalize()
    season = args.season or infer_season(local_date)
    yesterday = (local_date - pd.Timedelta(days=1)).date()
    lookback_start = (local_date - pd.Timedelta(days=int(args.lookback_days))).date()
    future_end = (local_date + pd.Timedelta(days=int(args.future_days))).date()

    run_stamp = local_date.strftime("%Y%m%d")
    run_dir = ANALYSIS_ROOT / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_update_data:
        run_step(
            f"Update Official NBA Data Through {yesterday}",
            [
                args.python,
                "scripts/update_nba_processed_data.py",
                "--season",
                str(season),
                "--through-date",
                str(yesterday),
            ],
        )

    if not args.skip_collect_market:
        run_step(
            f"Collect Covers Props {lookback_start} -> {future_end}",
            [
                args.python,
                "scripts/collect_covers_historical_props.py",
                "--date-from",
                str(lookback_start),
                "--date-to",
                str(future_end),
                "--scan-count",
                str(int(args.collect_scan_count)),
            ],
        )

    if not args.skip_align:
        run_step(
            f"Align Historical Market Lines For Season {season}",
            [
                args.python,
                "scripts/align_historical_market_lines.py",
                "--season",
                str(season),
                "--skip-market-anchor",
            ],
        )

    latest_market_path = MARKET_ROOT / "latest_player_props_wide.parquet"
    current_snapshot_path = run_dir / f"current_market_snapshot_{run_stamp}.parquet"
    current_rows, snapshot_meta = filter_current_market_snapshot(latest_market_path, current_snapshot_path, local_date, args.future_days)

    final_csv = run_dir / f"final_market_plays_{run_stamp}.csv"
    final_json = run_dir / f"final_market_plays_{run_stamp}.json"
    slate_csv = run_dir / f"upcoming_market_slate_{run_stamp}.csv"
    selector_csv = run_dir / f"upcoming_market_play_selector_{run_stamp}.csv"

    run_step(
        "Run Market Decision Pipeline",
        [
            args.python,
            "scripts/run_market_pipeline.py",
            "--season",
            str(season),
            "--history-csv",
            str(args.history_csv),
            "--market-wide-path",
            str(current_snapshot_path),
            "--slate-csv-out",
            str(slate_csv),
            "--selector-csv-out",
            str(selector_csv),
            "--final-csv-out",
            str(final_csv),
            "--final-json-out",
            str(final_json),
            *(["--allow-heuristic-fallback"] if args.allow_heuristic_fallback else []),
            *(["--latest"] if args.latest else []),
        ],
    )

    manifest = {
        "run_date": str(local_date.date()),
        "season": int(season),
        "through_date": str(yesterday),
        "lookback_start": str(lookback_start),
        "future_end": str(future_end),
        "history_csv": str(args.history_csv),
        "current_market_snapshot": str(current_snapshot_path),
        "current_market_rows": int(current_rows),
        "current_market_snapshot_meta": snapshot_meta,
        "final_csv": str(final_csv),
        "final_json": str(final_json),
        "slate_csv": str(slate_csv),
        "selector_csv": str(selector_csv),
        "used_latest_manifest": bool(args.latest),
        "skip_update_data": bool(args.skip_update_data),
        "skip_collect_market": bool(args.skip_collect_market),
        "skip_align": bool(args.skip_align),
        "allow_heuristic_fallback": bool(args.allow_heuristic_fallback),
        "updated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = run_dir / f"daily_market_pipeline_manifest_{run_stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_step(
        "Export Static Daily Predictions Page Data",
        [
            args.python,
            "scripts/export_daily_predictions_web.py",
            "--manifest",
            str(manifest_path),
            "--out-dist",
            str(SITE_ROOT / "dist" / "data" / "daily_predictions.json"),
        ],
    )

    print("\n" + "=" * 90)
    print("DAILY MARKET PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Run date:             {local_date.date()}")
    print(f"Season:               {season}")
    print(f"Current market rows:  {current_rows}")
    print(f"Snapshot mode:        {snapshot_meta['mode']}")
    if snapshot_meta["mode"] == "stale_fallback":
        print(f"Selected market date: {snapshot_meta['selected_market_date']}")
    print(f"Run directory:        {run_dir}")
    print(f"Final board:          {final_csv}")
    print(f"Manifest:             {manifest_path}")


if __name__ == "__main__":
    main()
