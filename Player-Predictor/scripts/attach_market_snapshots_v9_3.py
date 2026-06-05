#!/usr/bin/env python3
"""
Attach real sportsbook odds snapshots to v9.2 artifacts.

v9.3 is intentionally not a new model stack. It is v9.2 plus market
validation infrastructure: current no-vig probabilities, actual available
odds, closing prices, and CLV-ready fields.
"""
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

REQUIRED_FIELDS = [
    "snapshot_time",
    "book",
    "market",
    "player",
    "line",
    "over_odds",
    "under_odds",
    "no_vig_over",
    "no_vig_under",
    "open_line",
    "current_line",
    "close_line",
    "close_over_odds",
    "close_under_odds",
]


def _resolve_path(path_text: str | Path, base: Path | None = None) -> Path:
    text = str(path_text)
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    path = Path(text)
    if path.is_absolute():
        return path
    return ((base or REPO_ROOT) / path).resolve()


def _american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    over = _american_to_implied(over_odds)
    under = _american_to_implied(under_odds)
    total = over + under
    if not np.isfinite(total) or total <= 0:
        return 0.5, 0.5
    return over / total, under / total


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    return pd.read_csv(path)


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(manifest: dict, manifest_path: Path) -> pd.DataFrame:
    output = _resolve_path(manifest.get("output", manifest_path.parent), manifest_path.parent)
    rows_path = output / "data" / "prop_training_rows.csv"
    if not rows_path.exists():
        rows_path = manifest_path.parent / "data" / "prop_training_rows.csv"
    rows = pd.read_csv(rows_path)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date.astype(str)
    return rows


def _normalize_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    snapshots = snapshots.copy()
    rename_map = {
        "closing_over_odds": "close_over_odds",
        "closing_under_odds": "close_under_odds",
        "closing_line": "close_line",
        "current_over_odds": "over_odds",
        "current_under_odds": "under_odds",
    }
    for old, new in rename_map.items():
        if old in snapshots.columns and new not in snapshots.columns:
            snapshots[new] = snapshots[old]
    if "current_line" not in snapshots.columns and "line" in snapshots.columns:
        snapshots["current_line"] = snapshots["line"]
    if "date" in snapshots.columns:
        snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce").dt.date.astype(str)
    for col in ["line", "current_line", "open_line", "close_line", "over_odds", "under_odds", "close_over_odds", "close_under_odds"]:
        if col in snapshots.columns:
            snapshots[col] = pd.to_numeric(snapshots[col], errors="coerce")
    if {"over_odds", "under_odds"}.issubset(snapshots.columns):
        snapshots = add_american_odds_quality(snapshots)
        missing_no_vig = ("no_vig_over" not in snapshots.columns) or ("no_vig_under" not in snapshots.columns)
        if missing_no_vig:
            valid = snapshots["is_valid_american_odds"]
            snapshots["no_vig_over"] = np.nan
            snapshots["no_vig_under"] = np.nan
            pairs = snapshots.loc[valid].apply(lambda row: _no_vig(row["over_odds"], row["under_odds"]), axis=1)
            if len(pairs):
                snapshots.loc[valid, "no_vig_over"], snapshots.loc[valid, "no_vig_under"] = zip(*pairs)
    return snapshots


def _validate_snapshots(snapshots: pd.DataFrame) -> dict:
    missing = [col for col in REQUIRED_FIELDS if col not in snapshots.columns]
    invalid = {}
    for col in ["over_odds", "under_odds", "close_over_odds", "close_under_odds", "line", "current_line", "close_line"]:
        if col in snapshots.columns:
            invalid[col] = int(snapshots[col].isna().sum())
    for col in ["no_vig_over", "no_vig_under"]:
        if col in snapshots.columns:
            values = pd.to_numeric(snapshots[col], errors="coerce")
            invalid[col] = int((values.isna() | (values <= 0) | (values >= 1)).sum())
    status = "pass" if not missing and all(v == 0 for v in invalid.values()) else "fail"
    return {"status": status, "missing_required_fields": missing, "invalid_value_counts": invalid}


def _choose_join_keys(rows: pd.DataFrame, snapshots: pd.DataFrame) -> list[str]:
    candidates = [
        ["game_id", "player_id", "market"],
        ["game_id", "player", "market"],
        ["date", "player_id", "market"],
        ["date", "player", "market"],
    ]
    for keys in candidates:
        if all(key in rows.columns and key in snapshots.columns for key in keys):
            return keys
    raise ValueError("Could not find a supported join key set for odds snapshots")


def _dedupe_snapshots(snapshots: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    snapshots = snapshots.copy()
    snapshots["_snapshot_ts"] = pd.to_datetime(snapshots["snapshot_time"], errors="coerce", utc=True)
    snapshots = snapshots.sort_values(keys + ["_snapshot_ts"], na_position="first")
    return snapshots.drop_duplicates(keys, keep="last").drop(columns=["_snapshot_ts"])


def attach_snapshots(rows: pd.DataFrame, snapshots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    snapshots = _normalize_snapshots(snapshots)
    validation = _validate_snapshots(snapshots)
    quality = odds_quality_report(snapshots)
    if validation["status"] != "pass":
        return rows.copy(), {
            "status": "blocked_schema_validation_failed",
            "schema_validation": validation,
            "odds_quality": quality,
            "matched_rows": 0,
            "match_rate": 0.0,
        }
    keys = _choose_join_keys(rows, snapshots)
    rows = rows.copy()
    snapshots = snapshots.copy()
    for key in keys:
        rows[key] = rows[key].astype(str)
        snapshots[key] = snapshots[key].astype(str)
    snapshots = _dedupe_snapshots(snapshots, keys)
    optional_market_fields = [
        "close_status",
        "source",
        "close_no_vig_over",
        "close_no_vig_under",
        "clv_limit",
    ]
    market_cols = list(dict.fromkeys(keys + REQUIRED_FIELDS + optional_market_fields + [c for c in ["game_id", "player_id", "date"] if c in snapshots.columns]))
    market_cols = [c for c in market_cols if c in snapshots.columns]
    merged = rows.merge(
        snapshots[market_cols],
        on=keys,
        how="left",
        suffixes=("", "_snapshot"),
        indicator=True,
    )
    matched = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])
    for col in REQUIRED_FIELDS:
        snap_col = f"{col}_snapshot"
        if snap_col in merged.columns:
            if col in merged.columns:
                if merged[snap_col].notna().any():
                    merged[col] = merged[snap_col].combine_first(merged[col])
            else:
                merged[col] = merged[snap_col]
            merged = merged.drop(columns=[snap_col])
    for col in optional_market_fields:
        snap_col = f"{col}_snapshot"
        if snap_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[snap_col].combine_first(merged[col])
            else:
                merged[col] = merged[snap_col]
            merged = merged.drop(columns=[snap_col])
    if "current_line" in merged.columns:
        merged["line"] = merged["current_line"].combine_first(merged["line"])
    merged["market_no_vig_over"] = pd.to_numeric(merged["no_vig_over"], errors="coerce").combine_first(merged.get("market_no_vig_over", pd.Series(index=merged.index)))
    merged["market_no_vig_under"] = pd.to_numeric(merged["no_vig_under"], errors="coerce").combine_first(merged.get("market_no_vig_under", pd.Series(index=merged.index)))
    return merged, {
        "status": "attached",
        "schema_validation": validation,
        "odds_quality": quality,
        "join_keys": keys,
        "snapshot_rows": int(len(snapshots)),
        "matched_rows": int(matched.sum()),
        "total_rows": int(len(rows)),
        "match_rate": float(matched.mean()) if len(rows) else 0.0,
        "books": sorted(str(v) for v in snapshots["book"].dropna().unique()) if "book" in snapshots.columns else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create v9.3 artifacts by attaching real odds snapshots to v9.2")
    parser.add_argument("--v9-2-manifest", type=Path, default=ROOT / "model" / "props" / "v9_2" / "manifest.json")
    parser.add_argument("--odds-snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_3")
    parser.add_argument("--min-match-rate", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = args.v9_2_manifest.resolve()
    source_manifest = _load_manifest(source_manifest_path)
    rows = _load_rows(source_manifest, source_manifest_path)
    snapshots = _read_table(args.odds_snapshots)
    merged, attachment = attach_snapshots(rows, snapshots)
    status = "real_market_validation_candidate" if attachment.get("match_rate", 0.0) >= args.min_match_rate else "blocked_low_market_snapshot_match_rate"

    data_dir = args.output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(data_dir / "prop_training_rows.csv", index=False)

    manifest = {
        "model_version": "prop_engine_v9_3_market_validated_distribution",
        "status": status,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_v9_2_manifest": str(args.v9_2_manifest),
        "odds_snapshots": str(args.odds_snapshots),
        "output": str(args.output),
        "rows": int(len(merged)),
        "players": int(merged["player"].nunique()),
        "date_min": str(pd.to_datetime(merged["date"], errors="coerce").min().date()),
        "date_max": str(pd.to_datetime(merged["date"], errors="coerce").max().date()),
        "artifacts": {
            "data": "data/prop_training_rows.csv",
            "market_odds_schema": "Player-Predictor/configs/market_odds_snapshot_schema_v1.json",
        },
        "market_attachment": attachment,
        "live_promotion_blockers": [
            "CLV must be positive and correlated with model edge out of sample",
            "model must beat true current no-vig Brier",
            "ROI must be computed from actual available odds, not neutral -110 defaults",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": status, "market_attachment": attachment, "output": str(args.output)}, indent=2, default=str))


if __name__ == "__main__":
    main()
