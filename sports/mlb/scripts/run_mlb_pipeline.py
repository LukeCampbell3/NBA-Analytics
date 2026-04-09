#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    payload = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full MLB pipeline: collect -> build -> validate -> train")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--min-processed-rows",
        type=int,
        default=11,
        help="Minimum rows required by validator per player file.",
    )
    parser.add_argument("--min-train-rows", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = []

    if not args.skip_collect:
        logs.append(
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "collect_mlb_data.py"),
                    "--start-date",
                    str(args.start_date),
                    "--end-date",
                    str(args.end_date),
                ]
            )
        )

    logs.append(
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_mlb_features.py"),
                "--season",
                str(args.season),
            ]
        )
    )

    logs.append(
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_mlb_processed_contract.py"),
                "--data-dir",
                str(ROOT / "data" / "processed"),
                "--min-rows",
                str(args.min_processed_rows),
            ]
        )
    )

    if not args.skip_train:
        logs.append(
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train_mlb_models.py"),
                    "--season",
                    str(args.season),
                    "--min-rows",
                    str(args.min_train_rows),
                ]
            )
        )

    print(json.dumps({"status": "ok", "steps": len(logs)}, indent=2))


if __name__ == "__main__":
    main()
