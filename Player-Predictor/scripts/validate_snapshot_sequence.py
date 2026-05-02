#!/usr/bin/env python3
"""
Replay saved point-in-time market snapshots and grade the resulting boards.

This expands the validation sample using the normalized snapshot files already
captured under data copy/raw/market_odds/nba/normalized. Because multiple
snapshots can cover the same eventual games, the output reports both:
1. occurrence-level results across all replayed boards
2. deduplicated unique-play results by event/play signature
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from validate_historical_daily_runs import grade_board


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = REPO_ROOT / "model" / "analysis"
VALIDATION_ROOT = ANALYSIS_ROOT / "historical_validation"
MARKET_ROOT = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba"
NORMALIZED_ROOT = MARKET_ROOT / "normalized"
DEFAULT_HISTORY_CSV = ANALYSIS_ROOT / "refreshed_market_comparison_strict_rows.csv"
DEFAULT_HISTORY_WIDE = MARKET_ROOT / "history_player_props_wide.parquet"
SNAPSHOT_RE = re.compile(r"player_props_wide_(\d{8}T\d{6}Z)\.(?:parquet|csv)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a sequence of saved normalized market snapshots.")
    parser.add_argument("--season", type=int, default=2026, help="Season end year.")
    parser.add_argument("--snapshot-dir", type=Path, default=NORMALIZED_ROOT, help="Directory containing saved normalized snapshots.")
    parser.add_argument("--snapshot-glob", type=str, default="player_props_wide_*.parquet", help="Snapshot filename pattern.")
    parser.add_argument("--policy-profiles", nargs="+", default=["production_high_precision", "production_calibrated"], help="Policy profiles to replay against each snapshot.")
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV, help="Historical calibration CSV to use when replaying.")
    parser.add_argument("--history-wide-path", type=Path, default=DEFAULT_HISTORY_WIDE, help="Wide market history used to reconstruct point-in-time snapshots.")
    parser.add_argument("--output-tag", type=str, default="snapshot_sequence_validation", help="Output folder tag under model/analysis/historical_validation.")
    parser.add_argument("--start-stamp", type=str, default=None, help="Optional inclusive snapshot stamp floor (YYYYMMDDTHHMMSSZ).")
    parser.add_argument("--end-stamp", type=str, default=None, help="Optional inclusive snapshot stamp ceiling (YYYYMMDDTHHMMSSZ).")
    parser.add_argument("--completed-through-date", type=str, default=None, help="Optional latest completed market date (YYYY-MM-DD). Snapshots with later market dates are skipped.")
    parser.add_argument("--max-snapshots", type=int, default=None, help="Optional cap after sorting by snapshot stamp.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    parser.add_argument("--allow-heuristic-fallback", action="store_true", help="Allow replay to continue with heuristic-only predictions if model load fails.")
    parser.add_argument("--reconstruct-from-history-wide", action="store_true", help="Rebuild each snapshot from history_player_props_wide at the same cutoff timestamp.")
    return parser.parse_args()


def snapshot_stamp(path: Path) -> str:
    match = SNAPSHOT_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized snapshot filename: {path.name}")
    return match.group(1)


def load_snapshot(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_snapshot_timestamp(stamp: str) -> pd.Timestamp:
    return pd.to_datetime(stamp, format="%Y%m%dT%H%M%SZ", utc=True)


def list_snapshots(args: argparse.Namespace) -> list[Path]:
    snapshot_dir = args.snapshot_dir.resolve()
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")
    items = []
    completed_through = pd.Timestamp(args.completed_through_date).normalize() if args.completed_through_date else None
    for path in sorted(snapshot_dir.glob(args.snapshot_glob), key=lambda item: item.name):
        stamp = snapshot_stamp(path)
        if args.start_stamp and stamp < args.start_stamp:
            continue
        if args.end_stamp and stamp > args.end_stamp:
            continue
        if completed_through is not None:
            df = load_snapshot(path)
            market_dates = pd.to_datetime(df.get("Market_Date"), errors="coerce").dropna()
            if market_dates.empty:
                continue
            if market_dates.max().normalize() > completed_through:
                continue
        items.append(path)
    if args.max_snapshots is not None:
        items = items[: int(args.max_snapshots)]
    if not items:
        raise RuntimeError("No snapshot files matched the requested filters.")
    return items


def summarize_snapshot(path: Path) -> dict:
    return summarize_frame(load_snapshot(path), path)


def summarize_frame(df: pd.DataFrame, path: Path | None = None) -> dict:
    market_dates = pd.to_datetime(df.get("Market_Date"), errors="coerce")
    valid_dates = market_dates.dropna()
    commence = pd.to_datetime(df.get("Market_Commence_Time_UTC", pd.Series(dtype=object)), errors="coerce", utc=True)
    stamp = None
    if path is not None:
        try:
            stamp = snapshot_stamp(path)
        except ValueError:
            stamp = None
    return {
        "snapshot_path": str(path) if path is not None else None,
        "snapshot_stamp": stamp,
        "rows": int(len(df)),
        "market_date_min": None if valid_dates.empty else str(valid_dates.min().date()),
        "market_date_max": None if valid_dates.empty else str(valid_dates.max().date()),
        "market_dates": sorted(valid_dates.dt.strftime("%Y-%m-%d").unique().tolist()),
        "event_rows": int(df.get("Market_Event_ID", pd.Series(dtype=object)).notna().sum()),
        "home_team_rows": int(df.get("Market_Home_Team", pd.Series(dtype=object)).notna().sum()),
        "away_team_rows": int(df.get("Market_Away_Team", pd.Series(dtype=object)).notna().sum()),
        "commence_rows": int(commence.notna().sum()),
    }


def load_history_wide(path: Path) -> pd.DataFrame:
    history_df = load_snapshot(path.resolve())
    if history_df.empty:
        raise RuntimeError(f"History wide file is empty: {path}")
    history_df = history_df.copy()
    history_df["Market_Date"] = pd.to_datetime(history_df["Market_Date"], errors="coerce")
    history_df["Market_Fetched_At_UTC"] = pd.to_datetime(history_df["Market_Fetched_At_UTC"], errors="coerce", utc=True)
    return history_df


def reconstruct_snapshot_from_history(
    source_snapshot_path: Path,
    history_df: pd.DataFrame,
    out_path: Path,
) -> tuple[Path, dict]:
    source_df = load_snapshot(source_snapshot_path)
    source_meta = summarize_frame(source_df, source_snapshot_path)
    if not source_meta["market_date_min"] or not source_meta["market_date_max"]:
        raise RuntimeError(f"Snapshot has no valid Market_Date values: {source_snapshot_path}")
    cutoff_source = parse_snapshot_timestamp(snapshot_stamp(source_snapshot_path))
    cutoff = cutoff_source + pd.Timedelta(minutes=1)
    start_date = pd.Timestamp(source_meta["market_date_min"])
    end_date = pd.Timestamp(source_meta["market_date_max"])
    rebuilt = history_df.loc[history_df["Market_Date"].between(start_date, end_date)].copy()
    rebuilt = rebuilt.loc[rebuilt["Market_Fetched_At_UTC"].notna() & (rebuilt["Market_Fetched_At_UTC"] <= cutoff)].copy()
    rebuilt = rebuilt.sort_values(["Market_Date", "Player", "Market_Fetched_At_UTC"]).drop_duplicates(subset=["Market_Date", "Player"], keep="last")
    if rebuilt.empty:
        raise RuntimeError(f"Reconstructed snapshot is empty for {source_snapshot_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        rebuilt.to_parquet(out_path, index=False)
    else:
        rebuilt.to_csv(out_path, index=False)
    rebuilt_meta = summarize_frame(rebuilt, out_path)
    rebuilt_meta["cutoff_utc_source"] = cutoff_source.isoformat()
    rebuilt_meta["cutoff_utc_applied"] = cutoff.isoformat()
    rebuilt_meta["source_snapshot_path"] = str(source_snapshot_path)
    return out_path, rebuilt_meta


def summarize_results(frame: pd.DataFrame) -> dict:
    if frame.empty or "result" not in frame.columns:
        return {"rows": 0, "graded_rows": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": None}
    wins = int((frame["result"] == "win").sum())
    losses = int((frame["result"] == "loss").sum())
    pushes = int((frame["result"] == "push").sum())
    graded_rows = wins + losses + pushes
    decisions = wins + losses
    return {
        "rows": int(len(frame)),
        "graded_rows": int(graded_rows),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": None if decisions == 0 else (wins / decisions),
    }


def build_play_signature(frame: pd.DataFrame) -> pd.Series:
    event_id = frame.get("market_event_id", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    market_date = frame.get("market_date", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    player = frame.get("player", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    target = frame.get("target", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    direction = frame.get("direction", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    market_line = pd.to_numeric(frame.get("market_line"), errors="coerce").round(3).fillna(-9999).astype(str)
    return event_id + "|" + market_date + "|" + player + "|" + target + "|" + direction + "|" + market_line


def replay_snapshot(
    snapshot_path: Path,
    policy_profile: str,
    season: int,
    history_csv: Path,
    python_exe: str,
    allow_heuristic_fallback: bool,
    out_dir: Path,
) -> tuple[Path, Path, pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_csv = out_dir / f"{policy_profile}_final_market_plays.csv"
    final_json = out_dir / f"{policy_profile}_final_market_plays.json"
    slate_csv = out_dir / f"{policy_profile}_upcoming_market_slate.csv"
    selector_csv = out_dir / f"{policy_profile}_upcoming_market_play_selector.csv"
    subprocess.run(
        [
            python_exe,
            "scripts/run_market_pipeline.py",
            "--season",
            str(season),
            "--latest",
            "--policy-profile",
            str(policy_profile),
            "--history-csv",
            str(history_csv.resolve()),
            "--market-wide-path",
            str(snapshot_path.resolve()),
            "--slate-csv-out",
            str(slate_csv),
            "--selector-csv-out",
            str(selector_csv),
            "--final-csv-out",
            str(final_csv),
            "--final-json-out",
            str(final_json),
            *(["--allow-heuristic-fallback"] if allow_heuristic_fallback else []),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    graded_df, summary = grade_board(final_csv, snapshot_path=snapshot_path)
    return final_csv, final_json, graded_df, summary


def main() -> None:
    args = parse_args()
    history_csv = args.history_csv.resolve()
    if not history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {history_csv}")
    history_wide_path = args.history_wide_path.resolve()
    history_wide_df = None
    if args.reconstruct_from_history_wide:
        if not history_wide_path.exists():
            raise FileNotFoundError(f"History wide file not found: {history_wide_path}")
        history_wide_df = load_history_wide(history_wide_path)

    snapshots = list_snapshots(args)
    output_root = VALIDATION_ROOT / args.output_tag
    output_root.mkdir(parents=True, exist_ok=True)

    rollup: dict[str, object] = {
        "season": int(args.season),
        "history_csv": str(history_csv),
        "history_wide_path": str(history_wide_path),
        "snapshot_dir": str(args.snapshot_dir.resolve()),
        "snapshot_glob": args.snapshot_glob,
        "start_stamp": args.start_stamp,
        "end_stamp": args.end_stamp,
        "completed_through_date": args.completed_through_date,
        "reconstruct_from_history_wide": bool(args.reconstruct_from_history_wide),
        "snapshot_count": int(len(snapshots)),
        "policy_profiles": list(args.policy_profiles),
        "profiles": {},
    }

    for policy_profile in args.policy_profiles:
        profile_root = output_root / policy_profile
        profile_root.mkdir(parents=True, exist_ok=True)
        snapshot_summaries: list[dict] = []
        graded_frames: list[pd.DataFrame] = []

        for snapshot_path in snapshots:
            stamp = snapshot_stamp(snapshot_path)
            replay_root = profile_root / stamp
            replay_snapshot_path = snapshot_path
            snapshot_meta = summarize_snapshot(snapshot_path)
            if args.reconstruct_from_history_wide:
                replay_snapshot_path, rebuilt_meta = reconstruct_snapshot_from_history(
                    source_snapshot_path=snapshot_path,
                    history_df=history_wide_df,
                    out_path=replay_root / "reconstructed_market_snapshot.parquet",
                )
                snapshot_meta = {
                    "source_snapshot": snapshot_meta,
                    "replay_snapshot": rebuilt_meta,
                }
            final_csv, final_json, graded_df, grade_summary = replay_snapshot(
                snapshot_path=replay_snapshot_path,
                policy_profile=policy_profile,
                season=int(args.season),
                history_csv=history_csv,
                python_exe=args.python,
                allow_heuristic_fallback=bool(args.allow_heuristic_fallback),
                out_dir=replay_root,
            )
            graded_df = graded_df.copy()
            graded_df["snapshot_stamp"] = stamp
            graded_df["snapshot_path"] = str(replay_snapshot_path)
            graded_df["source_snapshot_path"] = str(snapshot_path)
            graded_df["policy_profile"] = policy_profile
            graded_df["play_signature"] = build_play_signature(graded_df)
            graded_frames.append(graded_df)
            snapshot_summaries.append(
                {
                    **snapshot_meta,
                    "final_csv": str(final_csv),
                    "final_json": str(final_json),
                    "grade_summary": grade_summary,
                }
            )

        combined = pd.concat(graded_frames, ignore_index=True) if graded_frames else pd.DataFrame()
        occurrence_summary = summarize_results(combined)
        if not combined.empty:
            unique_frame = combined.drop_duplicates(subset=["play_signature"]).reset_index(drop=True)
            unique_summary = summarize_results(unique_frame)
            unique_summary["unique_play_count"] = int(len(unique_frame))
            combined.to_csv(profile_root / f"{policy_profile}_graded_occurrences.csv", index=False)
            unique_frame.to_csv(profile_root / f"{policy_profile}_graded_unique_plays.csv", index=False)
        else:
            unique_frame = pd.DataFrame()
            unique_summary = {"rows": 0, "graded_rows": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": None, "unique_play_count": 0}

        profile_payload = {
            "snapshot_count": int(len(snapshot_summaries)),
            "occurrence_summary": occurrence_summary,
            "unique_play_summary": unique_summary,
            "snapshots": snapshot_summaries,
        }
        (profile_root / f"{policy_profile}_summary.json").write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")
        rollup["profiles"][policy_profile] = profile_payload

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SNAPSHOT SEQUENCE VALIDATION COMPLETE")
    print("=" * 90)
    print(f"Snapshots:    {len(snapshots)}")
    print(f"Output root:  {output_root}")
    print(f"Summary JSON: {summary_path}")
    for policy_profile in args.policy_profiles:
        payload = rollup["profiles"][policy_profile]
        occurrence = payload["occurrence_summary"]
        unique = payload["unique_play_summary"]
        print(
            f"{policy_profile}: "
            f"occurrences {occurrence['wins']}/{occurrence['wins'] + occurrence['losses']} "
            f"({occurrence['win_rate']:.3f})"
            if occurrence["win_rate"] is not None
            else f"{policy_profile}: occurrences unscored"
        )
        print(
            f"  unique plays {unique['wins']}/{unique['wins'] + unique['losses']} "
            f"({unique['win_rate']:.3f})"
            if unique["win_rate"] is not None
            else "  unique plays unscored"
        )


if __name__ == "__main__":
    main()
