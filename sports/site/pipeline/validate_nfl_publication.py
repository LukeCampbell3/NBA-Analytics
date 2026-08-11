#!/usr/bin/env python3
"""Validate that the NFL frontend remains an honest research-only publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_PAYLOAD = Path("sports/nfl/web/data/daily_predictions.json")
VALIDATION_PAYLOAD = Path("sports/nfl/web/data/market_validation_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required NFL publication file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"NFL publication file must contain an object: {path}")
    return payload


def validate_nfl_publication(*, repo_root: Path, output_dir: Path) -> str:
    resolved_output = output_dir if output_dir.is_absolute() else repo_root / output_dir
    route = resolved_output / "nfl" / "predictions" / "index.html"
    if not route.is_file():
        raise FileNotFoundError(f"NFL prediction route is missing: {route}")

    source_daily = load_json(repo_root / DAILY_PAYLOAD)
    public_daily = load_json(resolved_output / "nfl/data/daily_predictions.json")
    source_validation = load_json(repo_root / VALIDATION_PAYLOAD)
    public_validation = load_json(
        resolved_output / "nfl/data/market_validation_summary.json"
    )

    if source_daily != public_daily:
        raise ValueError("NFL daily source and public payloads differ.")
    if source_validation != public_validation:
        raise ValueError("NFL validation source and public payloads differ.")

    if source_daily.get("publication_status") != "research_only":
        raise ValueError("NFL daily payload must remain research_only.")
    if source_daily.get("mode") != "historical_holdout":
        raise ValueError("NFL daily payload must identify its historical_holdout mode.")
    if source_validation.get("publication_status") != "research_only_source_blocked":
        raise ValueError("NFL market evidence must remain source-blocked research.")

    deployment = (source_validation.get("gates") or {}).get("deployment") or {}
    if deployment.get("status") != "blocked":
        raise ValueError("NFL deployment gate must remain blocked without live-source evidence.")

    run_date = str(source_daily.get("run_date") or "<missing>")
    targets = ",".join(source_validation.get("validated_targets") or []) or "none"
    return f"NFL: research_only, holdout={run_date}, validated_targets={targets}"


def main() -> None:
    args = parse_args()
    summary = validate_nfl_publication(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir,
    )
    print("NFL publication validation passed.")
    print(f"- {summary}")


if __name__ == "__main__":
    main()
