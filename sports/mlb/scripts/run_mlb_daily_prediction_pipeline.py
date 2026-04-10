#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    payload = {
        "label": label,
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(payload, indent=2))
    return payload


def _default_run_date() -> str:
    return datetime.now().date().strftime("%Y-%m-%d")


def _default_history_start(season: int) -> str:
    return f"{int(season)}-03-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLB daily prediction pipeline: collect -> build -> validate -> infer -> score."
    )
    parser.add_argument("--run-date", type=str, default=_default_run_date(), help="Pipeline run date (YYYY-MM-DD).")
    parser.add_argument(
        "--game-date",
        type=str,
        default=None,
        help="Target game date for prediction pool (defaults to run-date).",
    )
    parser.add_argument("--season", type=int, default=None, help="Season year (defaults to game-date year).")
    parser.add_argument(
        "--history-start-date",
        type=str,
        default=None,
        help="Raw collection start date (defaults to season-03-01).",
    )
    parser.add_argument(
        "--through-date",
        type=str,
        default=None,
        help="Raw collection end date for completed results (defaults to game-date-1).",
    )
    parser.add_argument("--game-type", type=str, default="R", help="MLB game type for collection/schedule.")
    parser.add_argument("--min-history-rows", type=int, default=5, help="Minimum player history rows for inference.")
    parser.add_argument(
        "--min-processed-rows",
        type=int,
        default=1,
        help="Validator minimum rows per player file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Raw data directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Processed feature directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models",
        help="Model directory.",
    )
    parser.add_argument(
        "--daily-runs-root",
        type=Path,
        default=ROOT / "data" / "predictions" / "daily_runs",
        help="Daily runs root directory.",
    )
    parser.add_argument("--skip-collect", action="store_true", help="Skip raw collection step.")
    parser.add_argument("--skip-build-features", action="store_true", help="Skip feature build step.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip processed validation step.")
    parser.add_argument("--skip-score", action="store_true", help="Skip scoring step.")
    parser.add_argument(
        "--score-all-unscored",
        action="store_true",
        help="Also score all unresolved historical pools with available actuals.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for child scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    model_dir = args.model_dir.resolve()
    daily_runs_root = args.daily_runs_root.resolve()

    run_date = datetime.strptime(str(args.run_date), "%Y-%m-%d").date()
    game_date = datetime.strptime(str(args.game_date or args.run_date), "%Y-%m-%d").date()
    season = int(args.season) if args.season is not None else int(game_date.year)
    through_date = (
        datetime.strptime(str(args.through_date), "%Y-%m-%d").date()
        if args.through_date
        else (game_date - timedelta(days=1))
    )
    history_start_date = (
        str(args.history_start_date)
        if args.history_start_date
        else _default_history_start(season)
    )

    run_stamp = run_date.strftime("%Y%m%d")
    run_dir = (daily_runs_root / run_stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []

    if not args.skip_collect:
        steps.append(
            _run(
                f"collect_raw_{history_start_date}_to_{through_date}",
                [
                    args.python,
                    str(ROOT / "scripts" / "collect_mlb_data.py"),
                    "--start-date",
                    str(history_start_date),
                    "--end-date",
                    str(through_date),
                    "--game-type",
                    str(args.game_type),
                    "--out-dir",
                    str(raw_dir),
                ],
            )
        )

    if not args.skip_build_features:
        steps.append(
            _run(
                "build_processed_features",
                [
                    args.python,
                    str(ROOT / "scripts" / "build_mlb_features.py"),
                    "--season",
                    str(season),
                    "--raw-dir",
                    str(raw_dir),
                    "--processed-dir",
                    str(processed_dir),
                ],
            )
        )

    if not args.skip_validate:
        steps.append(
            _run(
                "validate_processed_contract",
                [
                    args.python,
                    str(ROOT / "scripts" / "validate_mlb_processed_contract.py"),
                    "--data-dir",
                    str(processed_dir),
                    "--min-rows",
                    str(int(args.min_processed_rows)),
                ],
            )
        )

    steps.append(
        _run(
            "build_daily_prediction_pool",
            [
                args.python,
                str(ROOT / "scripts" / "build_mlb_daily_prediction_pool.py"),
                "--run-date",
                str(game_date),
                "--season",
                str(season),
                "--processed-dir",
                str(processed_dir),
                "--model-dir",
                str(model_dir),
                "--out-dir",
                str(daily_runs_root),
                "--game-type",
                str(args.game_type),
                "--min-history-rows",
                str(int(args.min_history_rows)),
            ],
        )
    )

    if not args.skip_score:
        today_pool_csv = (daily_runs_root / game_date.strftime("%Y%m%d") / f"daily_prediction_pool_{game_date.strftime('%Y%m%d')}.csv").resolve()
        if today_pool_csv.exists() and game_date <= through_date:
            steps.append(
                _run(
                    "score_today_pool",
                    [
                        args.python,
                        str(ROOT / "scripts" / "score_mlb_prediction_pool.py"),
                        "--pool-csv",
                        str(today_pool_csv),
                        "--raw-dir",
                        str(raw_dir),
                    ],
                )
            )
        if args.score_all_unscored:
            steps.append(
                _run(
                    "score_all_unscored_pools",
                    [
                        args.python,
                        str(ROOT / "scripts" / "score_mlb_prediction_pool.py"),
                        "--all-unscored",
                        "--daily-runs-root",
                        str(daily_runs_root),
                        "--raw-dir",
                        str(raw_dir),
                        "--as-of-date",
                        str(through_date),
                    ],
                )
            )

    manifest = {
        "run_date": str(run_date),
        "game_date": str(game_date),
        "season": int(season),
        "history_start_date": str(history_start_date),
        "through_date": str(through_date),
        "game_type": str(args.game_type),
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "model_dir": str(model_dir),
            "daily_runs_root": str(daily_runs_root),
            "run_dir": str(run_dir),
        },
        "flags": {
            "skip_collect": bool(args.skip_collect),
            "skip_build_features": bool(args.skip_build_features),
            "skip_validate": bool(args.skip_validate),
            "skip_score": bool(args.skip_score),
            "score_all_unscored": bool(args.score_all_unscored),
        },
        "steps_run": int(len(steps)),
        "steps": [
            {
                "label": step["label"],
                "cmd": step["cmd"],
                "returncode": step["returncode"],
            }
            for step in steps
        ],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = run_dir / f"daily_prediction_pipeline_manifest_{run_stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "steps": len(steps)}, indent=2))


if __name__ == "__main__":
    main()
