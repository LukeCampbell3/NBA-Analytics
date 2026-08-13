#!/usr/bin/env python3
"""Fail-closed validation for the published F1 board."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


F1_ROOT = Path(__file__).resolve().parents[1]


def validate(path: Path, expected_date: str | None = None) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unable to read payload: {error}"]
    if payload.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if payload.get("sport") != "f1":
        errors.append("sport must equal f1")
    try:
        date.fromisoformat(str(payload.get("run_date")))
    except ValueError:
        errors.append("run_date must be ISO YYYY-MM-DD")
    if expected_date and payload.get("run_date") != expected_date:
        errors.append(f"run_date is {payload.get('run_date')}, expected {expected_date}")
    if payload.get("mode") != "live_shadow":
        errors.append("F1 publication must remain in live_shadow mode")
    if not isinstance(payload.get("model"), dict) or not payload["model"].get("backtest"):
        errors.append("model backtest metadata is required")
    for key in ("projections", "plays"):
        if not isinstance(payload.get(key), list):
            errors.append(f"{key} must be a list")
    if payload.get("selection", {}).get("staking_enabled") is not False:
        errors.append("staking_enabled must remain false")
    for row in payload.get("projections") or []:
        probability = row.get("win_probability")
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            errors.append(f"invalid win probability for {row.get('driver')}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, default=F1_ROOT / "web/data/daily_predictions.json")
    parser.add_argument("--run-date")
    args = parser.parse_args()
    errors = validate(args.payload, args.run_date)
    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1
    print(f"[ok] validated {args.payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
