#!/usr/bin/env python3
"""Recover exact MLB sportsbook snapshots retained in Git history.

The live fetcher keeps a compact current history, while older normalized CSVs
may be removed from the working tree. Git still retains those per-book offers.
This utility reconstructs them without changing the checked-out revision and
writes a history file accepted by ``generate_daily_prediction_pool.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = "sports/mlb/data/raw/market_odds/mlb/odds_api_io/normalized"
DEFAULT_CURRENT_ROOT = REPO_ROOT / DEFAULT_SNAPSHOT_ROOT
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "sports"
    / "mlb"
    / "data"
    / "raw"
    / "market_odds"
    / "mlb"
    / "historical_recovered"
    / "history_player_props_long.csv"
)
OFFER_KEY_COLUMNS = [
    "fetched_at_utc",
    "event_id",
    "bookmaker_key",
    "market_key",
    "player_name_norm",
    "line",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover normalized MLB market snapshots from Git history.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--snapshot-root", default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--current-root", type=Path, default=DEFAULT_CURRENT_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def discover_snapshot_refs(repo_root: Path, snapshot_root: str) -> list[tuple[str, str]]:
    commits = [
        line.strip()
        for line in run_git(repo_root, "log", "--all", "--format=%H", "--", snapshot_root).splitlines()
        if line.strip()
    ]
    refs_by_blob: dict[str, tuple[str, str]] = {}
    for commit in commits:
        listing = run_git(repo_root, "ls-tree", "-r", commit, "--", snapshot_root)
        for line in listing.splitlines():
            metadata, separator, path = line.partition("\t")
            if not separator or "/player_props_long_" not in path or not path.endswith(".csv"):
                continue
            parts = metadata.split()
            if len(parts) < 3:
                continue
            blob = parts[2]
            refs_by_blob.setdefault(blob, (commit, path))
    return sorted(refs_by_blob.values(), key=lambda ref: ref[1])


def load_historical_frames(repo_root: Path, refs: Iterable[tuple[str, str]]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for commit, path in refs:
        content = run_git(repo_root, "show", f"{commit}:{path}")
        try:
            frame = pd.read_csv(io.StringIO(content))
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        frame["history_origin"] = "git"
        frame["history_git_commit"] = commit
        frame["history_git_path"] = path
        frames.append(frame)
    return frames


def load_current_frames(current_root: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(current_root.glob("player_props_long_*.csv")):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        frame["history_origin"] = "working_tree"
        frame["history_git_commit"] = ""
        frame["history_git_path"] = path.as_posix()
        frames.append(frame)
    return frames


def combine_snapshots(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame for frame in frames if not frame.empty]
    if not materialized:
        return pd.DataFrame()
    combined = pd.concat(materialized, ignore_index=True, sort=False)
    available_keys = [column for column in OFFER_KEY_COLUMNS if column in combined.columns]
    if available_keys:
        combined = combined.drop_duplicates(subset=available_keys, keep="last")
    sort_columns = [
        column
        for column in ["fetched_at_utc", "event_date_et", "event_id", "player_name_norm", "market_key", "bookmaker_key"]
        if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable")
    return combined.reset_index(drop=True)


def build_report(frame: pd.DataFrame, ref_count: int) -> dict[str, object]:
    fetched = pd.to_datetime(frame.get("fetched_at_utc"), errors="coerce", utc=True)
    event_dates = pd.to_datetime(frame.get("event_date_et"), errors="coerce")
    return {
        "source": "exact sportsbook offers recovered from Git history and current normalized snapshots",
        "git_blob_count": ref_count,
        "row_count": int(len(frame)),
        "capture_count": int(fetched.nunique()) if fetched is not None else 0,
        "first_capture_utc": fetched.min().isoformat() if fetched is not None and fetched.notna().any() else None,
        "last_capture_utc": fetched.max().isoformat() if fetched is not None and fetched.notna().any() else None,
        "event_date_count": int(event_dates.dt.normalize().nunique()) if event_dates is not None else 0,
        "first_event_date": event_dates.min().date().isoformat() if event_dates is not None and event_dates.notna().any() else None,
        "last_event_date": event_dates.max().date().isoformat() if event_dates is not None and event_dates.notna().any() else None,
        "bookmaker_count": int(frame["bookmaker_key"].nunique()) if "bookmaker_key" in frame.columns else 0,
        "market_counts": (
            {str(key): int(value) for key, value in frame["market_key"].value_counts().sort_index().items()}
            if "market_key" in frame.columns
            else {}
        ),
    }


def portable_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    refs = discover_snapshot_refs(repo_root, args.snapshot_root)
    frames = load_historical_frames(repo_root, refs)
    frames.extend(load_current_frames(args.current_root.resolve()))
    combined = combine_snapshots(frames)
    if combined.empty:
        raise SystemExit("No normalized player prop snapshots were found.")

    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    report = build_report(combined, len(refs))
    report["output_csv"] = portable_path(out_csv, repo_root)
    report_path = (args.report_json or out_csv.with_suffix(".json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("MLB HISTORICAL MARKET RECOVERY")
    print(f"Rows:             {report['row_count']}")
    print(f"Git snapshots:    {report['git_blob_count']}")
    print(f"Capture count:    {report['capture_count']}")
    print(f"Event dates:      {report['event_date_count']}")
    print(f"Date range:       {report['first_event_date']} through {report['last_event_date']}")
    print(f"Bookmakers:       {report['bookmaker_count']}")
    print(f"Output CSV:       {out_csv}")
    print(f"Report JSON:      {report_path}")


if __name__ == "__main__":
    main()
