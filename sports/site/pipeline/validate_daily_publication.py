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
MLB_POLICY_PROFILE = "premium_over_first_v4"
MLB_REQUIRED_TARGETS = {"ER", "H", "HR", "K", "R", "RBI", "TB"}
MLB_MIN_BOOKS = 5
MLB_MIN_COMMON_BOOKS = 2
MLB_ALLOWED_SPORTSBOOKS = {"bet365", "caesars", "draftkings", "fanduel", "fanatics", "mgm"}
MLB_MARKET_BUCKET_CAP = 2
MLB_OPTIMIZED_OVER_MARKET_BUCKET_CAP = 3
MLB_PUBLICATION_STATES = {"published_current_pool", "withheld_current_pool"}
MLB_CORE_SELECTION_PROFILE = "core_market_v1"
MLB_OPTIMIZED_OVER_PROFILE = "r_tb_over_moderate_edge_v1"
MLB_OPTIMIZED_OVER_TARGETS = {"R", "TB"}
MLB_OPTIMIZED_OVER_PROFILE_STATUS = "probation"
MLB_DAILY_PICK_SOFT_CAP = 3
MLB_DAILY_PICK_HARD_CAP = 3
MLB_POST_CAP_MIN_SELECTION_SCORE = 0.80
MLB_CORE_MIN_AMERICAN_PRICE = -250.0
MLB_CORE_MAX_AMERICAN_PRICE = -200.0
MLB_MIN_OVER_PICKS = 3
MLB_MAX_OVER_PICKS = 3
MLB_MAX_UNDER_PICKS = 1


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
    return bool(
        price is not None
        and (price <= -100.0 or price >= 100.0)
        and abs(price - round(price)) <= 1e-6
    )


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
        raise ValueError(f"MLB {label} payload is not using the two-play market-bucket cap.")
    if (
        int(selection.get("optimized_over_max_per_market_bucket", 0))
        != MLB_OPTIMIZED_OVER_MARKET_BUCKET_CAP
    ):
        raise ValueError(f"MLB {label} payload changed the validated OVER market-bucket cap.")
    if as_float(selection.get("min_expected_value")) != 0.0:
        raise ValueError(f"MLB {label} payload must require nonnegative expected value.")
    if int(selection.get("min_market_books", 0)) < MLB_MIN_BOOKS:
        raise ValueError(f"MLB {label} payload must require at least {MLB_MIN_BOOKS} market books.")
    if int(selection.get("min_common_market_books", 0)) < MLB_MIN_COMMON_BOOKS:
        raise ValueError(f"MLB {label} payload must require exact coverage at a major sportsbook.")
    if not bool(selection.get("require_real_market_source")) or bool(selection.get("allow_unpriced_side")):
        raise ValueError(f"MLB {label} payload is not enforcing real, selected-side-priced markets.")

    optimized_over_targets = {
        str(value).strip().upper() for value in selection.get("optimized_over_targets", [])
    }
    if optimized_over_targets != MLB_OPTIMIZED_OVER_TARGETS:
        raise ValueError(f"MLB {label} payload is not using the validated R/TB OVER target set.")
    if str(selection.get("optimized_over_profile") or "") != MLB_OPTIMIZED_OVER_PROFILE:
        raise ValueError(f"MLB {label} payload is missing the validated OVER profile identity.")
    if str(selection.get("optimized_over_profile_status") or "") != MLB_OPTIMIZED_OVER_PROFILE_STATUS:
        raise ValueError(f"MLB {label} payload must keep the validated OVER profile in probation status.")
    exact_over_policy = {
        "over_min_abs_edge": 0.15,
        "over_max_abs_edge": 0.35,
        "over_min_model_hit_probability": 0.45,
        "over_max_model_hit_probability": 0.55,
        "over_min_expected_value": 0.10,
        "over_max_american_price": 125.0,
    }
    for key, expected in exact_over_policy.items():
        if as_float(selection.get(key)) != expected:
            raise ValueError(f"MLB {label} payload changed validated OVER threshold {key}.")
    if (
        int(selection.get("min_over_picks", 0)) != MLB_MIN_OVER_PICKS
        or int(selection.get("max_over_picks", 0)) != MLB_MAX_OVER_PICKS
        or int(selection.get("max_under_picks", 0)) != MLB_MAX_UNDER_PICKS
    ):
        raise ValueError(f"MLB {label} payload is not using the validated over-first portfolio limits.")
    if (
        as_float(selection.get("core_min_american_price")) != MLB_CORE_MIN_AMERICAN_PRICE
        or as_float(selection.get("core_max_american_price")) != MLB_CORE_MAX_AMERICAN_PRICE
    ):
        raise ValueError(f"MLB {label} payload changed the executable core price corridor.")
    if int(selection.get("daily_pick_soft_cap", 0)) != MLB_DAILY_PICK_SOFT_CAP:
        raise ValueError(f"MLB {label} payload changed the adaptive daily pick soft cap.")
    if int(selection.get("top_n", 0)) != MLB_DAILY_PICK_HARD_CAP:
        raise ValueError(f"MLB {label} payload changed the three-pick daily hard cap.")
    if as_float(selection.get("post_cap_min_selection_score")) != MLB_POST_CAP_MIN_SELECTION_SCORE:
        raise ValueError(f"MLB {label} payload changed the post-cap elite selection-score floor.")

    optimized_over_count = 0
    under_count = 0
    for index, play in enumerate(payload.get("plays", []), start=1):
        if not isinstance(play, dict):
            raise ValueError(f"MLB {label} play {index} must be an object.")
        if str(play.get("market_source") or "").lower() != "real":
            raise ValueError(f"MLB {label} play {index} is not backed by a real market.")
        if int(play.get("market_books") or 0) < MLB_MIN_BOOKS:
            raise ValueError(f"MLB {label} play {index} has insufficient market-book coverage.")
        if int(play.get("market_common_books") or 0) < MLB_MIN_COMMON_BOOKS:
            raise ValueError(f"MLB {label} play {index} is not offered at a supported major sportsbook.")
        if not bool(play.get("price_confirmed")) or not is_valid_american_price(play.get("selected_side_price")):
            raise ValueError(f"MLB {label} play {index} does not have valid selected-side American odds.")
        sportsbook_key = str(play.get("selected_sportsbook_key") or "").strip().lower()
        if sportsbook_key not in MLB_ALLOWED_SPORTSBOOKS or not str(play.get("selected_sportsbook") or "").strip():
            raise ValueError(f"MLB {label} play {index} does not identify the sportsbook for its exact price.")
        expected_value = as_float(play.get("expected_value_per_unit"))
        if expected_value is None or expected_value < 0.0:
            raise ValueError(f"MLB {label} play {index} has negative or missing expected value.")
        selection_profile = str(play.get("selection_profile") or "")
        if selection_profile not in {MLB_CORE_SELECTION_PROFILE, MLB_OPTIMIZED_OVER_PROFILE}:
            raise ValueError(f"MLB {label} play {index} has an unknown selection profile.")
        direction = str(play.get("direction") or "").strip().upper()
        if direction == "UNDER":
            under_count += 1
        target = str(play.get("target") or "").strip().upper()
        uses_optimized_over_profile = selection_profile == MLB_OPTIMIZED_OVER_PROFILE
        if direction == "OVER" and target in MLB_OPTIMIZED_OVER_TARGETS and not uses_optimized_over_profile:
            raise ValueError(f"MLB {label} play {index} bypassed the validated R/TB OVER profile.")
        if uses_optimized_over_profile:
            optimized_over_count += 1
            if direction != "OVER" or target not in MLB_OPTIMIZED_OVER_TARGETS:
                raise ValueError(f"MLB {label} play {index} misused the validated OVER profile identity.")
            abs_edge = as_float(play.get("abs_edge"))
            model_hit_probability = as_float(play.get("model_hit_probability"))
            if abs_edge is None or not 0.15 <= abs_edge <= 0.35:
                raise ValueError(f"MLB {label} play {index} falls outside the validated OVER edge corridor.")
            if model_hit_probability is None or not 0.45 <= model_hit_probability <= 0.55:
                raise ValueError(f"MLB {label} play {index} falls outside the validated OVER probability corridor.")
            if expected_value < 0.10:
                raise ValueError(f"MLB {label} play {index} falls below the validated OVER EV floor.")
            if float(play.get("selected_side_price")) > 125.0:
                raise ValueError(f"MLB {label} play {index} exceeds the validated OVER price ceiling.")
        else:
            side_price = float(play.get("selected_side_price"))
            if not MLB_CORE_MIN_AMERICAN_PRICE <= side_price <= MLB_CORE_MAX_AMERICAN_PRICE:
                raise ValueError(f"MLB {label} play {index} falls outside the executable core price corridor.")
    if optimized_over_count > 3:
        raise ValueError(f"MLB {label} payload exceeds the three-pick validated OVER cap.")
    if under_count > MLB_MAX_UNDER_PICKS:
        raise ValueError(f"MLB {label} payload exceeds the one-pick UNDER fallback cap.")

    for index, parlay in enumerate(payload.get("parlay_pairs", []), start=1):
        if not isinstance(parlay, dict):
            raise ValueError(f"MLB {label} parlay {index} must be an object.")
        if not bool(parlay.get("same_sportsbook_confirmed")):
            raise ValueError(f"MLB {label} parlay {index} is not executable at one confirmed sportsbook.")
        if not str(parlay.get("sportsbook_key") or "").strip():
            raise ValueError(f"MLB {label} parlay {index} is missing its sportsbook identity.")
        expected_return = as_float(parlay.get("expected_return_per_unit"))
        if expected_return is None or expected_return < 0.02:
            raise ValueError(f"MLB {label} parlay {index} does not clear the expected-return floor.")


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
