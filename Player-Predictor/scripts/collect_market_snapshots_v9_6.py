#!/usr/bin/env python3
"""Collect repeated v9.6 market snapshots and rebuild sequence artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from build_market_snapshot_sequence_v9_6 import (
    _derive_current_close,
    _label_snapshot_types,
    _load_inputs,
    _normalize,
    _resolve,
    build_report,
)
from fetch_nba_market_snapshots import (
    DEFAULT_OUTDIR as DEFAULT_ROTOWIRE_OUTDIR,
    DEFAULT_URL,
    build_book_snapshots,
    build_canonical_snapshots,
    extract_rotowire_bundles,
    utc_now_iso,
    utc_stamp,
    write_outputs,
)
from market_odds_quality import add_american_odds_quality, odds_quality_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_OUTDIR = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence"
DEFAULT_COLLECTION_FILE = DEFAULT_SEQUENCE_OUTDIR / "collected_book_snapshots.csv"
DEDUP_KEYS = [
    "snapshot_time",
    "book",
    "game_id",
    "player_id",
    "player",
    "market",
    "line",
    "over_odds",
    "under_odds",
]


def fetch_rotowire_book_rows(url: str, timeout_seconds: float, rotowire_outdir: Path) -> tuple[pd.DataFrame, dict]:
    fetched_at = utc_now_iso()
    response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "NBA-Analytics/1.0"})
    response.raise_for_status()
    market_date, bundles = extract_rotowire_bundles(response.text)
    book_rows = add_american_odds_quality(build_book_snapshots(market_date, bundles, fetched_at))
    canonical = add_american_odds_quality(build_canonical_snapshots(book_rows))
    manifest = {
        "provider": "rotowire",
        "source_url": url,
        "snapshot_stamp": utc_stamp(),
        "fetched_at_utc": fetched_at,
        "market_date": market_date,
        "book_rows": int(len(book_rows)),
        "canonical_rows": int(len(canonical)),
        "books": sorted(book_rows["book"].dropna().unique().tolist()) if not book_rows.empty else [],
        "odds_quality": odds_quality_report(book_rows),
        "close_status": "provisional_current_snapshot_not_closing",
        "clv_ready": False,
        "notes": [
            "Current RotoWire snapshots are valid decision-time price observations when fetched before lock.",
            "RotoWire does not expose game_start_time here, so repeated rows are not CLV-promotion-safe until enriched.",
        ],
    }
    write_outputs(_resolve(rotowire_outdir), book_rows, canonical, manifest)
    return book_rows, manifest


def append_collection(existing_path: Path, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    existing_path = _resolve(existing_path)
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
    else:
        existing = pd.DataFrame()
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    keys = [key for key in DEDUP_KEYS if key in combined.columns]
    if keys:
        normalized_keys = combined[keys].fillna("").astype(str)
        combined = combined.loc[~normalized_keys.duplicated(keep="last")].copy()
    appended = len(combined) - len(existing)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(existing_path, index=False)
    return combined, max(0, appended)


def build_sequence(inputs: list[Path], outdir: Path, min_valid_rate: float) -> dict:
    rows = _normalize(_load_inputs(inputs))
    valid_rows = rows[rows["is_valid_american_odds"]].copy()
    sequenced = _label_snapshot_types(valid_rows)
    attachable = _derive_current_close(sequenced)
    outdir = _resolve(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sequenced.drop(columns=["snapshot_ts", "game_start_ts"], errors="ignore").to_csv(outdir / "market_snapshot_sequence.csv", index=False)
    attachable.drop(columns=["snapshot_ts", "game_start_ts", "snapshot_ts_close"], errors="ignore").to_csv(outdir / "market_snapshot_attachable.csv", index=False)
    report = build_report(rows, valid_rows, sequenced, attachable, min_valid_rate)
    (outdir / "market_snapshot_sequence_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect v9.6 market snapshots and rebuild sequence files")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rotowire-outdir", type=Path, default=DEFAULT_ROTOWIRE_OUTDIR)
    parser.add_argument("--sequence-outdir", type=Path, default=DEFAULT_SEQUENCE_OUTDIR)
    parser.add_argument("--collection-file", type=Path, default=DEFAULT_COLLECTION_FILE)
    parser.add_argument("--include-inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--min-valid-american-odds-rate", type=float, default=0.98)
    parser.add_argument("--skip-fetch", action="store_true", help="Only rebuild the sequence from existing collection and include-inputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetched_rows = pd.DataFrame()
    fetch_manifest = {}
    if not args.skip_fetch:
        fetched_rows, fetch_manifest = fetch_rotowire_book_rows(args.url, args.timeout_seconds, args.rotowire_outdir)
    collection, appended_rows = append_collection(args.collection_file, fetched_rows) if not args.skip_fetch else (_read_existing(args.collection_file), 0)
    inputs = list(args.include_inputs) + [args.collection_file]
    sequence_report = build_sequence(inputs, args.sequence_outdir, args.min_valid_american_odds_rate)
    collection_report = {
        "status": "collected_v9_6_market_snapshot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fetched_rows": int(len(fetched_rows)),
        "appended_rows": int(appended_rows),
        "collected_rows": int(len(collection)),
        "game_start_time_available": bool("game_start_time" in collection.columns and collection["game_start_time"].notna().any()),
        "fetch_manifest": fetch_manifest,
        "sequence_report": sequence_report,
    }
    outdir = _resolve(args.sequence_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "collection_report.json").write_text(json.dumps(collection_report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(collection_report, indent=2, default=str))


def _read_existing(path: Path) -> pd.DataFrame:
    path = _resolve(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


if __name__ == "__main__":
    main()
