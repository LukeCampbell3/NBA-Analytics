#!/usr/bin/env python3
"""Create cached NFL projection and market-selector artifacts on a cold runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
NFL_ROOT = REPO_ROOT / "sports/nfl"
STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stats", type=Path, default=NFL_ROOT / "data/raw/player_stats.parquet"
    )
    parser.add_argument(
        "--yardage-artifact",
        type=Path,
        default=NFL_ROOT / "model/nfl_yardage_latent_hybrid.joblib",
    )
    parser.add_argument(
        "--selector-artifact",
        type=Path,
        default=NFL_ROOT / "model/nfl_market_selector.joblib",
    )
    return parser.parse_args()


def run(command: list[str], *, accepted: set[int] = {0}) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode not in accepted:
        raise RuntimeError(
            f"NFL artifact bootstrap command failed with {completed.returncode}: {command}"
        )


def ensure_stats(path: Path) -> None:
    if path.is_file():
        return
    frame = pd.read_parquet(STATS_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def ensure_market_rows(stats: Path, *, force: bool) -> tuple[Path, Path]:
    lines = NFL_ROOT / "data/raw/xsportsbook_bovada_player_props.csv"
    if force or not lines.is_file():
        run([sys.executable, str(NFL_ROOT / "scripts/fetch_xsportsbook_bovada_props.py")])
    outputs: list[Path] = []
    for season, selection in ((2021, "2019,2020"), (2022, "2020,2021")):
        work = NFL_ROOT / f"tmp/{season}"
        predictions = work / "backtest_rows.csv"
        market_rows = work / "market_rows_edge0.csv"
        if force or not predictions.is_file():
            run(
                [
                    sys.executable,
                    str(NFL_ROOT / "scripts/train_nfl_predictor.py"),
                    "--source",
                    str(stats),
                    "--cache",
                    str(stats),
                    "--holdout-season",
                    str(season),
                    "--selection-seasons",
                    selection,
                    "--report",
                    str(work / "backtest_report.json"),
                    "--rows",
                    str(predictions),
                    "--artifact",
                    str(work / "model.joblib"),
                    "--web-payload",
                    str(work / "daily_predictions.json"),
                ]
            )
        if force or not market_rows.is_file():
            run(
                [
                    sys.executable,
                    str(NFL_ROOT / "scripts/backtest_nfl_markets.py"),
                    "--lines",
                    str(lines),
                    "--predictions",
                    str(predictions),
                    "--minimum-edge",
                    "0",
                    "--report",
                    str(work / "market_report_edge0.json"),
                    "--rows",
                    str(market_rows),
                ],
                accepted={0, 2},
            )
        if not market_rows.is_file():
            raise FileNotFoundError(f"NFL market rows were not created: {market_rows}")
        outputs.append(market_rows)
    return outputs[0], outputs[1]


def main() -> int:
    args = parse_args()
    ensure_stats(args.stats)
    if not args.force and args.yardage_artifact.is_file() and args.selector_artifact.is_file():
        print("NFL artifacts restored from cache; bootstrap skipped.")
        return 0

    development_rows, final_rows = ensure_market_rows(args.stats, force=args.force)
    bootstrap_dir = NFL_ROOT / "tmp/bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    selector_report = bootstrap_dir / "market_selector_report.json"
    args.selector_artifact.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(NFL_ROOT / "scripts/train_nfl_market_selector.py"),
            "--stats",
            str(args.stats),
            "--development-market-rows",
            str(development_rows),
            "--final-market-rows",
            str(final_rows),
            "--report",
            str(selector_report),
            "--artifact",
            str(args.selector_artifact),
        ]
    )
    args.yardage_artifact.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(NFL_ROOT / "scripts/train_nfl_predictor.py"),
            "--source",
            str(args.stats),
            "--cache",
            str(args.stats),
            "--holdout-season",
            "2025",
            "--report",
            str(bootstrap_dir / "yardage_backtest_report.json"),
            "--rows",
            str(bootstrap_dir / "yardage_backtest_rows.csv"),
            "--artifact",
            str(args.yardage_artifact),
            "--web-payload",
            str(bootstrap_dir / "yardage_holdout_payload.json"),
        ]
    )
    report = json.loads(
        selector_report.read_text(encoding="utf-8")
    )
    if report.get("validated_targets") != ["passing"]:
        raise RuntimeError("Cold NFL artifact bootstrap did not reproduce passing-only validation.")
    print("NFL projection and market-selector artifacts are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
