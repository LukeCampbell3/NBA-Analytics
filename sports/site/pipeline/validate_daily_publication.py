#!/usr/bin/env python3
"""Validate same-day source data and the protected publication output."""

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
MLB_POLICY_PROFILE = "premium_evidence_gated_v7"
MLB_REQUIRED_TARGETS = {"ER", "H", "HR", "K", "R", "RBI", "TB"}
MLB_MIN_BOOKS = 5
MLB_MIN_COMMON_BOOKS = 2
MLB_ALLOWED_SPORTSBOOKS = {"bet365", "caesars", "draftkings", "fanduel", "fanatics", "mgm"}
MLB_MARKET_BUCKET_CAP = 2
MLB_PUBLICATION_STATES = {"published_current_pool", "withheld_current_pool"}
MLB_CORE_SELECTION_PROFILE = "core_market_v1"
MLB_OPTIMIZED_OVER_PROFILE = "r_tb_over_moderate_edge_v1"
MLB_OPTIMIZED_OVER_PROFILE_STATUS = "probation"
MLB_PITCHER_K_OVER_PROFILE = "pitcher_k_over_workload_v1"
MLB_PITCHER_K_OVER_PROFILE_STATUS = "probation"
MLB_MATCHUP_NETWORK_VERSION = "batter_pitcher_profile_network_v2"
MLB_MATCHUP_ADJUSTMENT_CAPS = {"H": 0.10, "TB": 0.16, "R": 0.05, "HR": 0.025, "RBI": 0.06}
MLB_MAX_PITCHER_K_PICKS = 1
MLB_DAILY_PICK_SOFT_CAP = 3
MLB_DAILY_PICK_HARD_CAP = 3
MLB_POST_CAP_MIN_SELECTION_SCORE = 0.80
MLB_CORE_MIN_AMERICAN_PRICE = -180.0
MLB_CORE_MAX_AMERICAN_PRICE = 125.0
MLB_MIN_OVER_PICKS = 0
MLB_MAX_OVER_PICKS = 3
MLB_MAX_UNDER_PICKS = 1
MLB_MIN_CORE_HIT_PROBABILITY = 0.825
MLB_HISTORICAL_EVIDENCE_SCOPE = "real_price_confirmed_markets_only_v1"
MLB_PARLAY_PROBABILITY_FLOORS = {2: 0.40, 3: 0.25, 4: 0.18}
MLB_PARLAY_MAX_DECIMAL_PRICES = {2: 6.0, 3: 10.0, 4: 18.0}
MLB_PROFIT_BOOST_MIN_LEG_PROBABILITY = 0.18
MLB_PROFIT_BOOST_MIN_TICKET_PROBABILITY = 0.10
MLB_PROFIT_BOOST_MIN_DECIMAL_PRICE = 4.0
MLB_PROFIT_BOOST_MAX_DECIMAL_PRICE = 15.0
MLB_PROFIT_BOOST_MIN_EXPECTED_RETURN = 0.05
NBA_CALIBRATION_METHOD = "segment_monotonic_safety"
NBA_CALIBRATION_SCOPE = "FULL_CANDIDATE_POOL_REPLAY"
NBA_CALIBRATION_SEGMENTS = {
    "GLOBAL",
    "PTS_OVER",
    "PTS_UNDER",
    "TRB_OVER",
    "TRB_UNDER",
    "AST_OVER",
    "AST_UNDER",
}


def validate_mlb_daily_ticket(
    ticket: dict[str, Any],
    *,
    label: str,
    authorization_enabled: bool,
) -> None:
    ticket_tier = str(ticket.get("ticket_tier") or "consistency").strip().lower()
    is_profit_boost = ticket_tier == "profit_boost"
    if is_profit_boost and bool(ticket.get("candidate_authorized", False)):
        raise ValueError(f"MLB {label} profit boost must remain shadow-only.")
    if not authorization_enabled and bool(ticket.get("candidate_authorized", False)):
        raise ValueError(f"MLB {label} contains an authorized parlay without an active certificate.")
    legs = ticket.get("legs")
    if not isinstance(legs, list) or not 2 <= len(legs) <= 4:
        raise ValueError(f"MLB {label} daily parlay must contain two to four legs.")
    leg_count = int(ticket.get("leg_count") or 0)
    if leg_count != len(legs):
        raise ValueError(f"MLB {label} daily parlay leg count does not match its legs.")
    sportsbook_key = str(ticket.get("sportsbook_key") or "").strip().lower()
    if sportsbook_key not in MLB_ALLOWED_SPORTSBOOKS or not bool(ticket.get("same_sportsbook_confirmed")):
        raise ValueError(f"MLB {label} daily parlay is not executable at one supported sportsbook.")
    game_ids: set[str] = set()
    players: set[str] = set()
    for leg_index, leg in enumerate(legs, start=1):
        if not isinstance(leg, dict):
            raise ValueError(f"MLB {label} daily parlay leg {leg_index} must be an object.")
        if str(leg.get("direction") or "").strip().upper() != "OVER":
            raise ValueError(f"MLB {label} daily parlay leg {leg_index} is not an OVER.")
        if str(leg.get("market_source") or "").strip().lower() != "real":
            raise ValueError(f"MLB {label} daily parlay leg {leg_index} lacks a real market.")
        if not bool(leg.get("price_confirmed")) or not is_valid_american_price(leg.get("selected_side_price")):
            raise ValueError(f"MLB {label} daily parlay leg {leg_index} lacks confirmed odds.")
        if str(leg.get("selected_sportsbook_key") or "").strip().lower() != sportsbook_key:
            raise ValueError(f"MLB {label} daily parlay legs do not share one sportsbook.")
        if int(leg.get("market_books") or 0) < MLB_MIN_BOOKS or int(leg.get("market_common_books") or 0) < MLB_MIN_COMMON_BOOKS:
            raise ValueError(f"MLB {label} daily parlay leg {leg_index} lacks book coverage.")
        leg_probability = as_float(leg.get("estimated_graded_hit_rate"))
        minimum_leg_probability = MLB_PROFIT_BOOST_MIN_LEG_PROBABILITY if is_profit_boost else 0.62
        if leg_probability is None or leg_probability < minimum_leg_probability:
            raise ValueError(f"MLB {label} daily parlay leg {leg_index} misses the consistency floor.")
        if is_profit_boost:
            base_line = as_float(leg.get("base_market_line"))
            current_line = as_float(leg.get("market_line"))
            leg_expected_return = as_float(leg.get("expected_value_per_unit"))
            if str(leg.get("line_variant") or "").strip().lower() != "alternate":
                raise ValueError(f"MLB {label} profit boost leg {leg_index} is not an alternate line.")
            if base_line is None or current_line is None or current_line <= base_line:
                raise ValueError(f"MLB {label} profit boost leg {leg_index} does not raise the main line.")
            if not str(leg.get("provider_source_market_id") or "").strip():
                raise ValueError(f"MLB {label} profit boost leg {leg_index} lacks quote provenance.")
            if not str(leg.get("alternate_line_observed_at_utc") or "").strip():
                raise ValueError(f"MLB {label} profit boost leg {leg_index} lacks quote freshness provenance.")
            if int(leg.get("alternate_line_books") or 0) < 1:
                raise ValueError(f"MLB {label} profit boost leg {leg_index} lacks an executable alternate-line book.")
            if leg_expected_return is None or leg_expected_return < 0.03:
                raise ValueError(f"MLB {label} profit boost leg {leg_index} lacks positive alternate-line value.")
        game_id = str(leg.get("game_id") or "").strip()
        player = str(leg.get("player_id") or leg.get("player") or "").strip().lower()
        if not game_id or game_id in game_ids or not player or player in players:
            raise ValueError(f"MLB {label} daily parlay repeats a player or game.")
        game_ids.add(game_id)
        players.add(player)
    ticket_probability = as_float(ticket.get("projected_probability"))
    combined_decimal = as_float(ticket.get("combined_decimal_price"))
    expected_return = as_float(ticket.get("expected_return_per_unit"))
    probability_floor = (
        MLB_PROFIT_BOOST_MIN_TICKET_PROBABILITY if is_profit_boost else MLB_PARLAY_PROBABILITY_FLOORS[leg_count]
    )
    minimum_decimal = MLB_PROFIT_BOOST_MIN_DECIMAL_PRICE if is_profit_boost else 2.0
    maximum_decimal = (
        MLB_PROFIT_BOOST_MAX_DECIMAL_PRICE if is_profit_boost else MLB_PARLAY_MAX_DECIMAL_PRICES[leg_count]
    )
    minimum_expected_return = MLB_PROFIT_BOOST_MIN_EXPECTED_RETURN if is_profit_boost else 0.0
    if ticket_probability is None or ticket_probability < probability_floor:
        raise ValueError(f"MLB {label} daily parlay misses its {leg_count}-leg ticket floor.")
    if combined_decimal is None or not minimum_decimal <= combined_decimal <= maximum_decimal:
        raise ValueError(f"MLB {label} daily parlay falls outside its declared payout scope.")
    if expected_return is None or expected_return < minimum_expected_return:
        raise ValueError(f"MLB {label} daily parlay misses its expected-return floor.")
    risk_flags = {str(value) for value in ticket.get("risk_flags", [])}
    ticket_status = str(ticket.get("status") or "").strip().lower()
    if ticket_status == "ready" and risk_flags:
        raise ValueError(f"MLB {label} daily parlay is marked ready despite risk flags.")
    if ticket_status == "review" and risk_flags - {"lineup_unconfirmed"}:
        raise ValueError(f"MLB {label} daily parlay review contains a blocking risk flag.")


def validate_nba_payload(payload: dict[str, Any], *, label: str) -> None:
    calibration = payload.get("confidence_calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"NBA {label} payload is missing confidence-calibration evidence.")
    if (
        calibration.get("status") != "passed"
        or calibration.get("method") != NBA_CALIBRATION_METHOD
        or calibration.get("evidence_scope") != NBA_CALIBRATION_SCOPE
    ):
        raise ValueError(f"NBA {label} payload is not using the locked selected-board calibration policy.")
    locked = calibration.get("locked_metrics") or {}
    if int(locked.get("rows") or 0) < 1000:
        raise ValueError(f"NBA {label} confidence calibration lacks locked sample support.")
    support = calibration.get("historical_support") or {}
    if not isinstance(support, dict) or not NBA_CALIBRATION_SEGMENTS.issubset(support):
        raise ValueError(f"NBA {label} confidence calibration lacks target/direction support bounds.")

    for index, play in enumerate(payload.get("plays") or [], start=1):
        if not isinstance(play, dict):
            raise ValueError(f"NBA {label} play {index} must be an object.")
        raw = as_float(play.get("raw_model_probability"))
        calibrated = as_float(play.get("calibrated_hit_probability"))
        if raw is None or calibrated is None or not (0.0 <= raw <= 1.0 and 0.0 <= calibrated <= 1.0):
            raise ValueError(f"NBA {label} play {index} is missing confidence provenance.")
        if not bool(play.get("confidence_in_support")):
            raise ValueError(f"NBA {label} play {index} falls outside calibration support.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True, help="Expected publication date in YYYY-MM-DD format.")
    parser.add_argument("--sports", nargs="+", choices=sorted(SPORT_PAYLOADS), required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--protected-dir", type=Path, default=Path("paywall/private-content/app"))
    parser.add_argument(
        "--allow-stale-payloads",
        action="store_true",
        help="Skip stale-date enforcement when a run could not refresh the day-specific payloads.",
    )
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
    if (
        not bool(selection.get("matchup_network_enabled"))
        or str(selection.get("matchup_network_version") or "") != MLB_MATCHUP_NETWORK_VERSION
    ):
        raise ValueError(f"MLB {label} payload is not using the required batter/pitcher matchup network.")
    targets = {str(value).strip().upper() for value in selection.get("targets", [])}
    if targets != MLB_REQUIRED_TARGETS:
        raise ValueError(
            f"MLB {label} payload targets differ from the updated pool: "
            f"expected {sorted(MLB_REQUIRED_TARGETS)}, found {sorted(targets)}."
        )
    if int(selection.get("max_per_market_bucket", 0)) != MLB_MARKET_BUCKET_CAP:
        raise ValueError(f"MLB {label} payload is not using the two-play market-bucket cap.")
    if selection.get("optimized_over_max_per_market_bucket") is not None:
        raise ValueError(f"MLB {label} payload enabled the probationary OVER market-bucket cap.")
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
    if optimized_over_targets:
        raise ValueError(f"MLB {label} payload enabled the probationary R/TB OVER target set.")
    if str(selection.get("optimized_over_profile") or "") != MLB_OPTIMIZED_OVER_PROFILE:
        raise ValueError(f"MLB {label} payload is missing the validated OVER profile identity.")
    if str(selection.get("optimized_over_profile_status") or "") != MLB_OPTIMIZED_OVER_PROFILE_STATUS:
        raise ValueError(f"MLB {label} payload must keep the validated OVER profile in probation status.")
    probationary_over_fields = {
        "over_min_abs_edge",
        "over_max_abs_edge",
        "over_min_model_hit_probability",
        "over_max_model_hit_probability",
        "over_min_expected_value",
        "over_min_history_rows",
        "over_max_american_price",
    }
    enabled_over_fields = sorted(key for key in probationary_over_fields if selection.get(key) is not None)
    if enabled_over_fields:
        raise ValueError(
            f"MLB {label} payload enabled probationary OVER thresholds: {', '.join(enabled_over_fields)}."
        )
    if bool(selection.get("pitcher_k_over_profile_enabled")):
        raise ValueError(f"MLB {label} payload enabled the uncertified pitcher K profile.")
    if str(selection.get("pitcher_k_over_profile") or "") != MLB_PITCHER_K_OVER_PROFILE:
        raise ValueError(f"MLB {label} payload is missing the pitcher K profile identity.")
    if str(selection.get("pitcher_k_over_profile_status") or "") != MLB_PITCHER_K_OVER_PROFILE_STATUS:
        raise ValueError(f"MLB {label} payload must keep the pitcher K profile in probation status.")
    exact_pitcher_k_policy = {
        "pitcher_k_min_starter_history": 15.0,
        "pitcher_k_min_projected_ip": 5.25,
        "pitcher_k_min_projected_pitches": 75.0,
        "pitcher_k_max_days_since_history": 14.0,
        "pitcher_k_min_abs_edge": 0.15,
        "pitcher_k_max_abs_edge": 1.0,
        "pitcher_k_min_model_hit_probability": 0.50,
        "pitcher_k_max_model_hit_probability": 0.65,
        "pitcher_k_min_expected_value": 0.0,
        "pitcher_k_min_american_price": -130.0,
        "pitcher_k_max_american_price": 130.0,
    }
    for key, expected in exact_pitcher_k_policy.items():
        if as_float(selection.get(key)) != expected:
            raise ValueError(f"MLB {label} payload changed pitcher K threshold {key}.")
    if int(selection.get("max_pitcher_k_picks", 0)) != MLB_MAX_PITCHER_K_PICKS:
        raise ValueError(f"MLB {label} payload changed the one-pick pitcher K cap.")
    if (
        as_float(selection.get("min_hit_probability")) != MLB_MIN_CORE_HIT_PROBABILITY
        or as_float(selection.get("min_graded_hit_rate")) != MLB_MIN_CORE_HIT_PROBABILITY
    ):
        raise ValueError(f"MLB {label} payload changed the locked core probability floor.")
    if str(selection.get("historical_calibration_evidence_scope") or "") != MLB_HISTORICAL_EVIDENCE_SCOPE:
        raise ValueError(f"MLB {label} payload is not using executable-market calibration evidence.")
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

    pitcher_k_count = 0
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
        if selection_profile not in {
            MLB_CORE_SELECTION_PROFILE,
            MLB_OPTIMIZED_OVER_PROFILE,
            MLB_PITCHER_K_OVER_PROFILE,
        }:
            raise ValueError(f"MLB {label} play {index} has an unknown selection profile.")
        direction = str(play.get("direction") or "").strip().upper()
        if direction == "UNDER":
            under_count += 1
        target = str(play.get("target") or "").strip().upper()
        if str(play.get("player_type") or "").strip().lower() == "hitter":
            if str(play.get("matchup_network_version") or "") != MLB_MATCHUP_NETWORK_VERSION:
                raise ValueError(f"MLB {label} hitter play {index} is missing the current matchup network version.")
            if not str(play.get("opposing_pitcher") or "").strip():
                raise ValueError(f"MLB {label} hitter play {index} is not linked to an opposing probable starter.")
            pitcher_uncertainty = as_float(play.get("pitcher_profile_uncertainty"))
            network_confidence = as_float(play.get("matchup_network_confidence"))
            network_adjustment = as_float(play.get("matchup_network_adjustment"))
            archetype_games = as_float(play.get("archetype_neighbor_games"))
            archetype_support = as_float(play.get("archetype_neighbor_effective_support"))
            archetype_lift = as_float(play.get("archetype_neighbor_lift"))
            adjustment_cap = MLB_MATCHUP_ADJUSTMENT_CAPS.get(target)
            if (
                adjustment_cap is None
                or pitcher_uncertainty is None
                or not 0.0 <= pitcher_uncertainty <= 1.0
                or network_confidence is None
                or not 0.0 <= network_confidence <= 1.0
                or network_adjustment is None
                or abs(network_adjustment) > adjustment_cap + 1e-9
                or archetype_games is None
                or archetype_games < 0.0
                or abs(archetype_games - round(archetype_games)) > 1e-9
                or archetype_support is None
                or archetype_support < 0.0
                or archetype_lift is None
                or not -1.0 <= archetype_lift <= 1.0
            ):
                raise ValueError(f"MLB {label} hitter play {index} has invalid matchup network values.")
        uses_optimized_over_profile = selection_profile == MLB_OPTIMIZED_OVER_PROFILE
        uses_pitcher_k_profile = selection_profile == MLB_PITCHER_K_OVER_PROFILE
        if uses_optimized_over_profile:
            raise ValueError(f"MLB {label} play {index} used the disabled probationary OVER profile.")
        elif uses_pitcher_k_profile:
            raise ValueError(f"MLB {label} play {index} used the disabled probationary pitcher K profile.")
        else:
            side_price = float(play.get("selected_side_price"))
            if not MLB_CORE_MIN_AMERICAN_PRICE <= side_price <= MLB_CORE_MAX_AMERICAN_PRICE:
                raise ValueError(f"MLB {label} play {index} falls outside the executable core price corridor.")
    if pitcher_k_count > MLB_MAX_PITCHER_K_PICKS:
        raise ValueError(f"MLB {label} payload exceeds the one-pick pitcher K cap.")
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

    governance = payload.get("policy_governance")
    if not isinstance(governance, dict):
        raise ValueError(f"MLB {label} payload is missing policy-governance status.")
    if bool(governance.get("staking_enabled", False)):
        raise ValueError(f"MLB {label} policy governance cannot enable staking.")
    authorization_enabled = bool(governance.get("candidate_authorization_enabled", False))
    if not authorization_enabled:
        if str(governance.get("publication_mode")) != "SHADOW_RESEARCH_ONLY":
            raise ValueError(f"MLB {label} uncertified governance must be shadow-only.")
        if any(bool(play.get("candidate_authorized", False)) for play in payload.get("plays") or []):
            raise ValueError(f"MLB {label} contains an authorized play without an active certificate.")

    daily_parlay = payload.get("daily_parlay")
    if not isinstance(daily_parlay, dict):
        raise ValueError(f"MLB {label} payload is missing the adaptive daily parlay artifact.")
    daily_status = str(daily_parlay.get("status") or "").strip().lower()
    if daily_status not in {"ready", "review", "withheld"}:
        raise ValueError(f"MLB {label} daily parlay has an invalid status.")
    ticket = daily_parlay.get("selected_ticket")
    if ticket is None:
        if daily_status != "withheld":
            raise ValueError(f"MLB {label} daily parlay omitted its ticket without withholding it.")
    elif not isinstance(ticket, dict):
        raise ValueError(f"MLB {label} daily parlay ticket must be an object.")
    else:
        validate_mlb_daily_ticket(ticket, label=label, authorization_enabled=authorization_enabled)

    ladder = daily_parlay.get("ticket_ladder", [])
    if not isinstance(ladder, list):
        raise ValueError(f"MLB {label} daily parlay ladder must be a list.")
    ladder_ticket_ids: set[str] = set()
    for ladder_index, ladder_ticket in enumerate(ladder, start=1):
        if not isinstance(ladder_ticket, dict):
            raise ValueError(f"MLB {label} daily parlay ladder ticket {ladder_index} must be an object.")
        leg_count = int(ladder_ticket.get("leg_count") or 0)
        ticket_id = str(ladder_ticket.get("ticket_id") or f"{ladder_ticket.get('ticket_tier', '')}_{leg_count}")
        if ticket_id in ladder_ticket_ids:
            raise ValueError(f"MLB {label} daily parlay ladder repeats a ticket id.")
        ladder_ticket_ids.add(ticket_id)
        validate_mlb_daily_ticket(
            ladder_ticket,
            label=f"{label} ladder ticket {ladder_index}",
            authorization_enabled=authorization_enabled,
        )


def validate_publication(
    *,
    repo_root: Path,
    output_dir: Path,
    protected_dir: Path = Path("paywall/private-content/app"),
    run_date: str,
    sports: list[str],
    allow_stale_payloads: bool = False,
) -> list[str]:
    expected_date = date.fromisoformat(run_date).isoformat()
    resolved_output = output_dir if output_dir.is_absolute() else repo_root / output_dir
    resolved_protected = protected_dir if protected_dir.is_absolute() else repo_root / protected_dir

    for static_file in ("index.html", "app.js", "styles.css"):
        require_file(resolved_output / static_file)

    summaries: list[str] = []
    for sport in sports:
        source_path = repo_root / SPORT_PAYLOADS[sport]
        public_path = resolved_output / sport / "data" / "daily_predictions.json"
        route_path = resolved_output / sport / "predictions" / "index.html"

        source_payload = load_json(source_path)
        public_payload = load_json(public_path)
        require_file(route_path)

        source_date = str(source_payload.get("run_date") or "")
        public_date = str(public_payload.get("run_date") or "")
        if not allow_stale_payloads:
            if source_date != expected_date:
                raise ValueError(
                    f"{sport.upper()} source payload is stale: expected {expected_date}, found {source_date or '<missing>'}"
                )
            if public_date != expected_date:
                raise ValueError(
                    f"{sport.upper()} public payload is stale: expected {expected_date}, found {public_date or '<missing>'}"
                )

        source_status = str(source_payload.get("publication_status") or "").strip()
        public_status = str(public_payload.get("publication_status") or "").strip()
        if not source_status or source_status != public_status:
            source_status = "unavailable"
            public_status = "unavailable"
            if not allow_stale_payloads:
                raise ValueError(
                    f"{sport.upper()} publication status is missing or differs between source and public output "
                    f"({source_status or '<missing>'} vs {public_status or '<missing>'})"
                )

        plays = source_payload.get("plays")
        if not isinstance(plays, list):
            if allow_stale_payloads and not source_payload and not public_payload:
                plays = []
            else:
                raise ValueError(f"{sport.upper()} payload must contain a plays list.")
        if sport == "nba" and (not allow_stale_payloads or source_payload or public_payload):
            validate_nba_payload(source_payload, label="source")
            validate_nba_payload(public_payload, label="public")
        elif sport == "mlb" and not allow_stale_payloads:
            validate_mlb_payload(source_payload, label="source")
            validate_mlb_payload(public_payload, label="public")
        elif sport == "mlb" and allow_stale_payloads and (source_payload or public_payload):
            validate_mlb_payload(source_payload, label="source")
            validate_mlb_payload(public_payload, label="public")
        governance_suffix = ""
        if sport == "mlb" and source_payload.get("policy_governance"):
            governance = source_payload.get("policy_governance") or {}
            governance_suffix = f", mode={governance.get('publication_mode', 'UNKNOWN')}"
        summaries.append(
            f"{sport.upper()}: {expected_date}, status={source_status}, plays={len(plays)}{governance_suffix}"
        )

    return summaries


def main() -> None:
    args = parse_args()
    summaries = validate_publication(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir,
        protected_dir=args.protected_dir,
        run_date=args.run_date,
        sports=list(dict.fromkeys(args.sports)),
        allow_stale_payloads=args.allow_stale_payloads,
    )
    print("Daily publication validation passed.")
    for summary in summaries:
        print(f"- {summary}")


if __name__ == "__main__":
    main()
