#!/usr/bin/env python3
"""
Align historical market lines to existing processed player data.

Priority by target:
1. exact real market line from collected market history (Player + Date)
2. synthetic market anchor from trained model bundle
3. baseline fallback from existing rolling average

This produces a production-safe, fully populated market contract for every row
with explicit provenance in Market_Source_*.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from update_nba_processed_data import apply_market_anchor, load_market_anchor_bundle, normalize_name  # noqa: E402


DATA_DIR = REPO_ROOT / "Data-Proc"
MARKET_ROOT = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba"
MANIFEST_DIR = DATA_DIR
TARGETS = ["PTS", "TRB", "AST"]

REAL_MARKET_FIELDS = [
    "Market_PTS",
    "Market_TRB",
    "Market_AST",
    "Market_PTS_books",
    "Market_TRB_books",
    "Market_AST_books",
    "Market_PTS_over_price",
    "Market_TRB_over_price",
    "Market_AST_over_price",
    "Market_PTS_under_price",
    "Market_TRB_under_price",
    "Market_AST_under_price",
    "Market_PTS_line_std",
    "Market_TRB_line_std",
    "Market_AST_line_std",
    "Market_Fetched_At_UTC",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align historical market lines to processed player rows.")
    parser.add_argument("--season", type=int, required=True, help="Season end year, e.g. 2026 for 2025-26.")
    parser.add_argument("--history-wide-path", type=Path, default=MARKET_ROOT / "history_player_props_wide.parquet", help="Append-only market history wide file.")
    parser.add_argument("--market-anchor-path", type=Path, default=None, help="Optional explicit market anchor bundle path.")
    parser.add_argument("--skip-market-anchor", action="store_true", help="Disable synthetic market fills from the anchor model.")
    parser.add_argument("--player-limit", type=int, default=None, help="Optional player limit for smoke tests.")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Optional manifest path.")
    return parser.parse_args()


def load_market_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["Player"] = df["Player"].astype(str).map(normalize_name)
    df["Market_Date"] = pd.to_datetime(df["Market_Date"], errors="coerce").dt.date.astype(str)
    if "Market_Fetched_At_UTC" in df.columns:
        df["Market_Fetched_At_UTC"] = pd.to_datetime(df["Market_Fetched_At_UTC"], errors="coerce", utc=True)
    else:
        df["Market_Fetched_At_UTC"] = pd.NaT
    sort_cols = ["Player", "Market_Date", "Market_Fetched_At_UTC"]
    df = df.sort_values(sort_cols).drop_duplicates(subset=["Player", "Market_Date"], keep="last")
    return df


def gather_player_csvs(season: int, player_limit: int | None) -> list[Path]:
    player_dirs = sorted([path for path in DATA_DIR.iterdir() if path.is_dir()])
    if player_limit is not None:
        player_dirs = player_dirs[:player_limit]
    paths: list[Path] = []
    for player_dir in player_dirs:
        candidate = player_dir / f"{season}_processed_processed.csv"
        if candidate.exists():
            paths.append(candidate)
    return paths


def merge_real_market(df: pd.DataFrame, market_history: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    if "Date" not in out.columns:
        for col in REAL_MARKET_FIELDS:
            if col not in out.columns:
                out[col] = np.nan
        for target in TARGETS:
            source_col = f"Market_Source_{target}"
            if source_col not in out.columns:
                out[source_col] = ""
        return out, 0
    out["Market_Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
    if market_history.empty:
        for col in REAL_MARKET_FIELDS:
            if col not in out.columns:
                out[col] = np.nan
        out = out.drop(columns=["Market_Date"])
        return out, 0
    real_keep = ["Player", "Market_Date"] + [col for col in REAL_MARKET_FIELDS if col in market_history.columns]
    real = market_history[real_keep].copy()
    merged = out.merge(real, how="left", on=["Player", "Market_Date"], suffixes=("", "_real"))
    for target in TARGETS:
        source_col = f"Market_Source_{target}"
        market_col = f"Market_{target}"
        if source_col not in merged.columns:
            merged[source_col] = ""
        merged.loc[merged[market_col].notna(), source_col] = "real"
    matched = int(merged["Market_PTS"].notna().sum()) if "Market_PTS" in merged.columns else 0
    merged = merged.drop(columns=["Market_Date"])
    return merged, matched


def apply_baseline_fallback(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    summary = {"rows_filled": 0, "targets": {}}
    for target in TARGETS:
        market_col = f"Market_{target}"
        synth_col = f"Synthetic_Market_{target}"
        source_col = f"Market_Source_{target}"
        baseline_col = f"{target}_rolling_avg"
        if market_col not in out.columns:
            out[market_col] = np.nan
        if synth_col not in out.columns:
            out[synth_col] = np.nan
        if source_col not in out.columns:
            out[source_col] = pd.Series([""] * len(out), index=out.index, dtype="object")
        else:
            out[source_col] = out[source_col].fillna("").astype("object")
        missing_mask = out[market_col].isna()
        fill_values = pd.to_numeric(out.get(baseline_col), errors="coerce")
        out.loc[missing_mask, market_col] = fill_values[missing_mask]
        out.loc[missing_mask & out[synth_col].isna(), synth_col] = fill_values[missing_mask & out[synth_col].isna()]
        out.loc[missing_mask, source_col] = "baseline_fallback"
        rows_filled = int(missing_mask.sum())
        summary["targets"][target] = {"rows_filled": rows_filled}
        summary["rows_filled"] += rows_filled
    return out, summary


def finalize_market_metadata(df: pd.DataFrame, aligned_at_utc: str) -> pd.DataFrame:
    out = df.copy()
    aligned_ts = pd.Timestamp(aligned_at_utc)
    if "Market_Fetched_At_UTC" not in out.columns:
        out["Market_Fetched_At_UTC"] = pd.NaT
    fetched = pd.to_datetime(out["Market_Fetched_At_UTC"], errors="coerce", utc=True)
    fetched = fetched.fillna(aligned_ts)
    out["Market_Fetched_At_UTC"] = fetched.map(lambda value: value.isoformat() if pd.notna(value) else "")
    return out


def recompute_market_gaps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for target in TARGETS:
        baseline_col = f"{target}_rolling_avg"
        market_col = f"Market_{target}"
        gap_col = f"{target}_market_gap"
        if baseline_col in out.columns and market_col in out.columns:
            out[gap_col] = pd.to_numeric(out[market_col], errors="coerce") - pd.to_numeric(out[baseline_col], errors="coerce")
        else:
            out[gap_col] = np.nan
    return out


def process_player_csv(path: Path, market_history: pd.DataFrame, market_anchor_bundle, aligned_at_utc: str) -> dict:
    df = pd.read_csv(path)
    existing_market_cols = [
        col
        for col in df.columns
        if col.startswith("Market_") or col.startswith("Synthetic_Market_") or col.endswith("_market_gap")
    ]
    if existing_market_cols:
        df = df.drop(columns=existing_market_cols)
    if "Player" not in df.columns:
        df["Player"] = path.parent.name
    df["Player"] = df["Player"].astype(str).map(normalize_name)
    merged, real_matches = merge_real_market(df, market_history)
    if market_anchor_bundle is not None:
        merged, synth_summary = apply_market_anchor(merged, market_anchor_bundle)
    else:
        synth_summary = {"rows_filled": 0, "targets": {}}
    merged, baseline_summary = apply_baseline_fallback(merged)
    merged = finalize_market_metadata(merged, aligned_at_utc)
    merged = recompute_market_gaps(merged)
    merged.to_csv(path, index=False)
    return {
        "path": str(path),
        "player": path.parent.name,
        "rows": int(len(merged)),
        "real_market_rows": int(real_matches),
        "synthetic_rows_filled": int(synth_summary.get("rows_filled", 0)),
        "baseline_rows_filled": int(baseline_summary.get("rows_filled", 0)),
    }


def main() -> None:
    args = parse_args()
    aligned_at_utc = utc_now_iso()
    market_history = load_market_history(args.history_wide_path)
    market_anchor_bundle = None
    market_anchor_meta = {"available": False, "path": None, "targets": []}
    if not args.skip_market_anchor:
        market_anchor_bundle, market_anchor_meta = load_market_anchor_bundle(args.market_anchor_path)

    csv_paths = gather_player_csvs(args.season, args.player_limit)
    if not csv_paths:
        raise RuntimeError(f"No processed CSVs found for season {args.season}")

    results = []
    for path in csv_paths:
        results.append(process_player_csv(path, market_history, market_anchor_bundle, aligned_at_utc))

    manifest = {
        "updated_at_utc": aligned_at_utc,
        "season": args.season,
        "history_wide_path": str(args.history_wide_path),
        "history_rows": int(len(market_history)),
        "market_anchor": market_anchor_meta,
        "players_processed": int(len(results)),
        "real_market_rows": int(sum(item["real_market_rows"] for item in results)),
        "synthetic_rows_filled": int(sum(item["synthetic_rows_filled"] for item in results)),
        "baseline_rows_filled": int(sum(item["baseline_rows_filled"] for item in results)),
        "results": results,
    }

    manifest_path = args.manifest_out or (MANIFEST_DIR / f"historical_market_alignment_{args.season}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("HISTORICAL MARKET ALIGNMENT COMPLETE")
    print("=" * 80)
    print(f"Season:                {args.season}")
    print(f"Players processed:     {len(results)}")
    print(f"History rows loaded:   {len(market_history)}")
    print(f"Real market matches:   {manifest['real_market_rows']}")
    print(f"Synthetic rows filled: {manifest['synthetic_rows_filled']}")
    print(f"Baseline rows filled:  {manifest['baseline_rows_filled']}")
    print(f"Manifest:              {manifest_path}")


if __name__ == "__main__":
    main()
