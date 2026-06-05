#!/usr/bin/env python3
"""Build v9.6 market snapshot sequences with open/prelock/close labels."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_odds_quality import add_american_odds_quality, odds_quality_report


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
KEYS = ["date", "book", "player", "market"]


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _read(path: Path) -> pd.DataFrame:
    path = _resolve(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def _load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        resolved = _resolve(path)
        if resolved.is_dir():
            files = sorted(resolved.glob("*.csv")) + sorted(resolved.glob("*.parquet"))
            frames.extend(_read(file) for file in files)
        else:
            frames.append(_read(resolved))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if "snapshot_date" in rows.columns:
        if "date" not in rows.columns:
            rows["date"] = rows["snapshot_date"]
        else:
            rows["date"] = rows["date"].combine_first(rows["snapshot_date"])
    if "current_line" not in rows.columns and "line" in rows.columns:
        rows["current_line"] = rows["line"]
    if "current_over_odds" not in rows.columns and "over_odds" in rows.columns:
        rows["current_over_odds"] = rows["over_odds"]
    if "current_under_odds" not in rows.columns and "under_odds" in rows.columns:
        rows["current_under_odds"] = rows["under_odds"]
    if "book" not in rows.columns:
        rows["book"] = "unknown"
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date.astype(str)
    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_time"], errors="coerce", utc=True, format="mixed")
    if "game_start_time" in rows.columns:
        rows["game_start_ts"] = pd.to_datetime(rows["game_start_time"], errors="coerce", utc=True, format="mixed")
    else:
        rows["game_start_ts"] = pd.NaT
    for col in ["line", "current_line", "over_odds", "under_odds", "current_over_odds", "current_under_odds"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = add_american_odds_quality(rows)
    return rows.dropna(subset=["date", "snapshot_ts", "player", "market", "line", "over_odds", "under_odds"])


def _label_snapshot_types(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.sort_values(KEYS + ["snapshot_ts"]).copy()
    counts = rows.groupby(KEYS, dropna=False)["snapshot_ts"].transform("size")
    rows["snapshot_type"] = np.where(counts <= 1, "single_snapshot", "intraday")
    multi = rows[counts > 1]
    first_idx = multi.groupby(KEYS, dropna=False).head(1).index
    last_idx = multi.groupby(KEYS, dropna=False).tail(1).index
    rows.loc[first_idx, "snapshot_type"] = "open"
    rows.loc[last_idx, "snapshot_type"] = "close"
    if "game_start_ts" in rows.columns and rows["game_start_ts"].notna().any():
        prelock = rows[rows["snapshot_ts"] < rows["game_start_ts"]].groupby(KEYS, dropna=False).tail(1).index
        rows.loc[prelock, "snapshot_type"] = "prelock"
    return rows


def _derive_current_close(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    rows = rows.sort_values(KEYS + ["snapshot_ts"]).copy()
    # Current = latest pre-lock snapshot when game start is known; otherwise fall back to open.
    if "game_start_ts" in rows.columns and rows["game_start_ts"].notna().any():
        prelock_rows = rows[rows["snapshot_ts"] < rows["game_start_ts"]].copy()
        current = prelock_rows.groupby(KEYS, dropna=False).tail(1)
        open_rows = rows.groupby(KEYS, dropna=False).head(1)
        current_keys = current[KEYS].drop_duplicates()
        missing_open = open_rows.merge(current_keys, on=KEYS, how="left", indicator=True)
        missing_open = missing_open[missing_open["_merge"].eq("left_only")].drop(columns=["_merge"])
        current = pd.concat([current, missing_open], ignore_index=False).sort_values(KEYS + ["snapshot_ts"])
    else:
        current = rows.groupby(KEYS, dropna=False).head(1)
    # Close = last snapshot per group (the price the market converged to)
    close = rows.groupby(KEYS, dropna=False).tail(1)
    counts = rows.groupby(KEYS, dropna=False).size().reset_index(name="_snapshot_count")
    merged = current.merge(
        close[KEYS + ["line", "over_odds", "under_odds", "snapshot_ts"]],
        on=KEYS,
        how="left",
        suffixes=("", "_close"),
    )
    merged["current_line"] = merged["line"]
    merged["close_line"] = merged["line_close"]
    merged["close_over_odds"] = merged["over_odds_close"]
    merged["close_under_odds"] = merged["under_odds_close"]
    merged = merged.merge(counts, on=KEYS, how="left")
    if "game_start_ts" in merged.columns:
        has_game_start = merged["game_start_ts"].notna()
    else:
        has_game_start = pd.Series(False, index=merged.index)
    merged["close_status"] = np.select(
        [
            merged["_snapshot_count"] <= 1,
            ~has_game_start,
        ],
        [
            "provisional_single_snapshot_not_clv",
            "sequence_close_game_start_missing",
        ],
        default="true_sequence_close",
    )
    return merged


def build_report(rows: pd.DataFrame, valid_rows: pd.DataFrame, sequenced: pd.DataFrame, attachable: pd.DataFrame, min_valid_rate: float) -> dict:
    quality_all = odds_quality_report(rows)
    snapshot_counts = sequenced["snapshot_type"].value_counts().to_dict() if not sequenced.empty else {}
    close_status_counts = attachable["close_status"].value_counts().to_dict() if "close_status" in attachable.columns and not attachable.empty else {}
    true_clv_rows = int(attachable["close_status"].eq("true_sequence_close").sum()) if "close_status" in attachable.columns else 0
    gate_checks = {
        "valid_american_odds_rate": quality_all.get("valid_american_odds_rate", 0.0) >= min_valid_rate,
        "true_market_rows_min": int(len(attachable)) >= 1000,
        "true_clv_rows_min": true_clv_rows >= 500,
        "multiple_snapshot_types": len(snapshot_counts) >= 2,
        "prelock_snapshot_present": snapshot_counts.get("prelock", 0) > 0,
        "close_snapshot_present": snapshot_counts.get("close", 0) > 0,
        "game_start_time_present_for_clv": true_clv_rows > 0,
    }
    return {
        "status": "built_v9_6_market_snapshot_sequence",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(rows)),
        "valid_rows": int(len(valid_rows)),
        "sequence_rows": int(len(sequenced)),
        "attachable_rows": int(len(attachable)),
        "true_clv_rows": true_clv_rows,
        "snapshot_type_counts": snapshot_counts,
        "close_status_counts": close_status_counts,
        "odds_quality": quality_all,
        "quality_gate_pass": gate_checks["valid_american_odds_rate"],
        "promotion_gate_checks": gate_checks,
        "promotion_status": "eligible_for_market_review" if all(gate_checks.values()) else "blocked_market_snapshot_coverage",
        "promotion_blockers": {
            "require_true_market_rows_min": 1000,
            "require_gated_true_market_rows_min": 150,
            "require_true_clv_rows_min": 500,
            "require_valid_american_odds_rate_min": min_valid_rate,
            "require_multiple_snapshot_types": True,
            "require_prelock_snapshot": True,
            "require_close_snapshot": True,
            "require_model_bss_vs_market_positive": True,
            "require_clv_correlation_positive": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v9.6 market snapshot sequence")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence")
    parser.add_argument("--min-valid-american-odds-rate", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _normalize(_load_inputs(args.inputs))
    valid_rows = rows[rows["is_valid_american_odds"]].copy()
    sequenced = _label_snapshot_types(valid_rows)
    attachable = _derive_current_close(sequenced)
    outdir = _resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sequenced.drop(columns=["snapshot_ts", "game_start_ts"], errors="ignore").to_csv(outdir / "market_snapshot_sequence.csv", index=False)
    attachable.drop(columns=["snapshot_ts", "game_start_ts", "snapshot_ts_close"], errors="ignore").to_csv(outdir / "market_snapshot_attachable.csv", index=False)
    report = build_report(rows, valid_rows, sequenced, attachable, args.min_valid_american_odds_rate)
    (outdir / "market_snapshot_sequence_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
