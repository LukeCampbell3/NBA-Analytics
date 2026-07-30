#!/usr/bin/env python3
"""Validate that a daily pipeline run produced a same-day static publication."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SPORT_PAYLOADS = {
    "nba": Path("sports/nba/web/data/daily_predictions.json"),
    "mlb": Path("sports/mlb/web/data/daily_predictions.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True, help="Expected publication date in YYYY-MM-DD format.")
    parser.add_argument("--sports", nargs="+", choices=sorted(SPORT_PAYLOADS), required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction payload is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prediction payload is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Prediction payload must contain a JSON object: {path}")
    return payload


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required static output is missing or empty: {path}")


def validate_publication(
    *,
    repo_root: Path,
    output_dir: Path,
    run_date: str,
    sports: list[str],
) -> list[str]:
    expected_date = date.fromisoformat(run_date).isoformat()
    resolved_output = output_dir if output_dir.is_absolute() else repo_root / output_dir

    for static_file in ("index.html", "app.js", "styles.css"):
        require_file(resolved_output / static_file)

    summaries: list[str] = []
    for sport in sports:
        source_path = repo_root / SPORT_PAYLOADS[sport]
        dist_path = resolved_output / sport / "data" / "daily_predictions.json"
        route_path = resolved_output / sport / "predictions" / "index.html"

        source_payload = load_json(source_path)
        dist_payload = load_json(dist_path)
        require_file(route_path)

        source_date = str(source_payload.get("run_date") or "")
        dist_date = str(dist_payload.get("run_date") or "")
        if source_date != expected_date:
            raise ValueError(
                f"{sport.upper()} source payload is stale: expected {expected_date}, found {source_date or '<missing>'}"
            )
        if dist_date != expected_date:
            raise ValueError(
                f"{sport.upper()} dist payload is stale: expected {expected_date}, found {dist_date or '<missing>'}"
            )

        source_status = str(source_payload.get("publication_status") or "").strip()
        dist_status = str(dist_payload.get("publication_status") or "").strip()
        if not source_status or source_status != dist_status:
            raise ValueError(
                f"{sport.upper()} publication status is missing or differs between source and dist "
                f"({source_status or '<missing>'} vs {dist_status or '<missing>'})"
            )

        plays = source_payload.get("plays")
        if not isinstance(plays, list):
            raise ValueError(f"{sport.upper()} payload must contain a plays list.")
        summaries.append(
            f"{sport.upper()}: {expected_date}, status={source_status}, plays={len(plays)}"
        )

    return summaries


def main() -> None:
    args = parse_args()
    summaries = validate_publication(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir,
        run_date=args.run_date,
        sports=list(dict.fromkeys(args.sports)),
    )
    print("Daily publication validation passed.")
    for summary in summaries:
        print(f"- {summary}")


if __name__ == "__main__":
    main()
