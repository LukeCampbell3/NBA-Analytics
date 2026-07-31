#!/usr/bin/env python3
"""Validate that a daily pipeline run produced a same-day static publication."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SPORT_PAYLOADS = {
    "nba": Path("sports/nba/web/data/daily_predictions.json"),
    "mlb": Path("sports/mlb/web/data/daily_predictions.json"),
}
MLB_POLICY_PROFILE = "walk_forward_balanced_v2"
MLB_REQUIRED_TARGETS = {"ER", "H", "HR", "K", "R", "RBI", "TB"}
MLB_MIN_BOOKS = 5
MLB_MARKET_BUCKET_CAP = 4
MLB_PUBLICATION_STATES = {"published_current_pool", "withheld_current_pool"}


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


def as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_valid_american_price(value: object) -> bool:
    price = as_float(value)
    return bool(price is not None and (price <= -100.0 or price >= 100.0))


def validate_mlb_payload(payload: dict[str, Any], *, label: str) -> None:
    policy_profile = str(payload.get("policy_profile") or "")
    if policy_profile != MLB_POLICY_PROFILE:
        raise ValueError(
            f"MLB {label} payload used policy {policy_profile or '<missing>'}; expected {MLB_POLICY_PROFILE}."
        )
    publication_state = str(payload.get("publication_state") or "")
    if publication_state not in MLB_PUBLICATION_STATES:
        raise ValueError(f"MLB {label} payload has invalid publication state {publication_state or '<missing>'}.")

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"MLB {label} payload is missing selection policy metadata.")
    targets = {str(value).strip().upper() for value in selection.get("targets", [])}
    if targets != MLB_REQUIRED_TARGETS:
        raise ValueError(
            f"MLB {label} payload targets differ from the updated pool: "
            f"expected {sorted(MLB_REQUIRED_TARGETS)}, found {sorted(targets)}."
        )
    if int(selection.get("max_per_market_bucket", 0)) != MLB_MARKET_BUCKET_CAP:
        raise ValueError(f"MLB {label} payload is not using the four-play market-bucket cap.")
    if as_float(selection.get("min_expected_value")) != 0.0:
        raise ValueError(f"MLB {label} payload must require nonnegative expected value.")
    if int(selection.get("min_market_books", 0)) < MLB_MIN_BOOKS:
        raise ValueError(f"MLB {label} payload must require at least {MLB_MIN_BOOKS} market books.")
    if not bool(selection.get("require_real_market_source")) or bool(selection.get("allow_unpriced_side")):
        raise ValueError(f"MLB {label} payload is not enforcing real, selected-side-priced markets.")

    for index, play in enumerate(payload.get("plays", []), start=1):
        if not isinstance(play, dict):
            raise ValueError(f"MLB {label} play {index} must be an object.")
        if str(play.get("market_source") or "").lower() != "real":
            raise ValueError(f"MLB {label} play {index} is not backed by a real market.")
        if int(play.get("market_books") or 0) < MLB_MIN_BOOKS:
            raise ValueError(f"MLB {label} play {index} has insufficient market-book coverage.")
        if not bool(play.get("price_confirmed")) or not is_valid_american_price(play.get("selected_side_price")):
            raise ValueError(f"MLB {label} play {index} does not have valid selected-side American odds.")
        expected_value = as_float(play.get("expected_value_per_unit"))
        if expected_value is None or expected_value < 0.0:
            raise ValueError(f"MLB {label} play {index} has negative or missing expected value.")


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
        if sport == "mlb":
            validate_mlb_payload(source_payload, label="source")
            validate_mlb_payload(dist_payload, label="dist")
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
