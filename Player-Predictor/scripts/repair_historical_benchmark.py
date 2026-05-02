#!/usr/bin/env python3
"""
Rebuild dated historical validation snapshots from event-backed market history and
write a clean rebaseline summary across one or more policy profiles.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_ROOT = REPO_ROOT / "model" / "analysis" / "historical_validation"
DEFAULT_HISTORY_CSV = REPO_ROOT / "model" / "analysis" / "refreshed_market_comparison_strict_rows.csv"
DEFAULT_HISTORY_WIDE = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "history_player_props_wide.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair dated historical validation snapshots from event-backed market history.")
    parser.add_argument("--season", type=int, default=2026, help="Season end year.")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2026-03-26", "2026-03-27", "2026-03-28"],
        help="Historical run dates to rebuild in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["production_calibrated", "production_high_precision"],
        help="Policy profiles to replay against the repaired snapshots.",
    )
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV, help="Historical selector calibration CSV.")
    parser.add_argument("--history-wide-path", type=Path, default=DEFAULT_HISTORY_WIDE, help="Wide market history used to rebuild dated snapshots.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="repaired_from_history",
        help="Prefix used for the generated validation tags and summary folder.",
    )
    parser.add_argument(
        "--tune-source-profile",
        type=str,
        default="production_calibrated",
        help="Profile name to use for post-repair selector tuning. Set to empty string to skip tuning.",
    )
    parser.add_argument(
        "--tune-iterations",
        type=int,
        default=25000,
        help="Random-search iterations for the post-repair tuner.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for child scripts.")
    return parser.parse_args()


def run_step(label: str, args: list[str]) -> None:
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    print("Command:", " ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def stamp_for(run_date: str) -> str:
    return pd.Timestamp(run_date).strftime("%Y%m%d")


def summary_path_for(tag: str) -> Path:
    return VALIDATION_ROOT / tag / "historical_daily_validation_summary.json"


def run_validation_for_profile(args: argparse.Namespace, profile: str) -> tuple[str, dict]:
    tag = f"{args.output_prefix}_{profile}"
    run_step(
        f"Repair Historical Validation For {profile}",
        [
            args.python,
            "scripts/validate_historical_daily_runs.py",
            "--season",
            str(args.season),
            "--dates",
            *args.dates,
            "--history-csv",
            str(args.history_csv),
            "--history-wide-path",
            str(args.history_wide_path),
            "--policy-profile",
            str(profile),
            "--output-tag",
            tag,
            "--reconstruct-snapshot-from-history",
        ],
    )
    path = summary_path_for(tag)
    if not path.exists():
        raise FileNotFoundError(f"Validation summary not found after replay: {path}")
    return tag, json.loads(path.read_text(encoding="utf-8"))


def infer_top_n_by_date(summary_payload: dict) -> dict[str, int]:
    top_n_by_date: dict[str, int] = {}
    for item in summary_payload.get("dates", []):
        top_n_by_date[str(item["stamp"])] = int(item.get("original", {}).get("rows", 0))
    return top_n_by_date


def run_post_repair_tuning(args: argparse.Namespace, profile: str, base_summary: dict) -> Path | None:
    source_profile = (args.tune_source_profile or "").strip()
    if not source_profile:
        return None
    if source_profile != profile:
        return None

    source_tag = f"{args.output_prefix}_{profile}"
    top_n_by_date = infer_top_n_by_date(base_summary)
    if not top_n_by_date:
        return None
    out_dir = VALIDATION_ROOT / f"{args.output_prefix}_benchmark_tuned"
    run_step(
        f"Post-Repair Tuning For {profile}",
        [
            args.python,
            "scripts/tune_historical_selector_weights.py",
            "--source-tag",
            source_tag,
            "--dates",
            *top_n_by_date.keys(),
            "--top-n",
            *[str(value) for value in top_n_by_date.values()],
            "--iterations",
            str(int(args.tune_iterations)),
            "--out-dir",
            str(out_dir),
        ],
    )
    summary_path = out_dir / "benchmark_tuned_summary.json"
    return summary_path if summary_path.exists() else None


def build_rollup(profile_summaries: dict[str, dict]) -> dict:
    profiles: dict[str, dict] = {}
    for profile, payload in profile_summaries.items():
        dates = payload.get("dates", [])
        total_original_wins = sum(int(item.get("original", {}).get("wins", 0)) for item in dates)
        total_original_losses = sum(int(item.get("original", {}).get("losses", 0)) for item in dates)
        total_latest_wins = sum(int(item.get("latest", {}).get("wins", 0)) for item in dates)
        total_latest_losses = sum(int(item.get("latest", {}).get("losses", 0)) for item in dates)
        profiles[profile] = {
            "dates": [
                {
                    "run_date": item.get("run_date"),
                    "stamp": item.get("stamp"),
                    "snapshot_mode": item.get("snapshot_mode"),
                    "original": item.get("original"),
                    "latest": item.get("latest"),
                }
                for item in dates
            ],
            "original_total": {
                "wins": total_original_wins,
                "losses": total_original_losses,
                "win_rate": (total_original_wins / (total_original_wins + total_original_losses)) if (total_original_wins + total_original_losses) else None,
            },
            "latest_total": {
                "wins": total_latest_wins,
                "losses": total_latest_losses,
                "win_rate": (total_latest_wins / (total_latest_wins + total_latest_losses)) if (total_latest_wins + total_latest_losses) else None,
            },
        }
    return {"profiles": profiles}


def main() -> None:
    args = parse_args()
    if not args.history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {args.history_csv}")
    if not args.history_wide_path.exists():
        raise FileNotFoundError(f"History wide file not found: {args.history_wide_path}")

    profile_summaries: dict[str, dict] = {}
    tuning_summary_path: Path | None = None

    for profile in args.profiles:
        tag, summary_payload = run_validation_for_profile(args, profile)
        profile_summaries[profile] = summary_payload
        if profile == (args.tune_source_profile or "").strip():
            tuning_summary_path = run_post_repair_tuning(args, profile, summary_payload)

    rollup = build_rollup(profile_summaries)
    rollup["season"] = int(args.season)
    rollup["dates"] = list(args.dates)
    rollup["history_csv"] = str(args.history_csv.resolve())
    rollup["history_wide_path"] = str(args.history_wide_path.resolve())
    rollup["output_prefix"] = args.output_prefix
    rollup["profile_summary_paths"] = {
        profile: str(summary_path_for(f"{args.output_prefix}_{profile}"))
        for profile in args.profiles
    }
    rollup["tuning_summary_path"] = str(tuning_summary_path) if tuning_summary_path is not None else None

    summary_dir = VALIDATION_ROOT / args.output_prefix
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "repair_rebaseline_summary.json"
    summary_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("HISTORICAL REPAIR + REBASELINE COMPLETE")
    print("=" * 90)
    for profile in args.profiles:
        latest_total = rollup["profiles"][profile]["latest_total"]
        decisions = latest_total["wins"] + latest_total["losses"]
        print(f"{profile}: {latest_total['wins']}/{decisions} rebuilt replay wins")
    if tuning_summary_path is not None:
        print(f"Tuning summary: {tuning_summary_path}")
    print(f"Rollup summary: {summary_path}")


if __name__ == "__main__":
    main()
