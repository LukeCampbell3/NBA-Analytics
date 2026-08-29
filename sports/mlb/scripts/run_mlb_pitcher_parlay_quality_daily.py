#!/usr/bin/env python3
"""Daily MLB pitcher-K parlay using the ROI-aware alt-line frontier.

The predictive model and calibration evidence are unchanged. This runner
expands every real FanDuel strikeout threshold already captured by the public
provider, rejects undercut/negative-value legs, enforces real hit-probability
and payout floors, then selects the highest quoted-price model EV among the
survivors. It writes the same pitcher_parlay_predictions.json contract
consumed by the frontend.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (
    REPO_ROOT / "sports" / "mlb" / "predictions",
    REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers",
    REPO_ROOT / "sports" / "mlb" / "parlay_v2",
    REPO_ROOT / "sports" / "mlb" / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import select_mlb_pitcher_parlay as legacy_select  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402
from fanduel_public_mlb_provider import FanduelPublicMlbProvider  # noqa: E402
from pitcher_alt_line_frontier import (  # noqa: E402
    MIN_COMBO_DECIMAL_PRICE,
    MIN_COMBO_EXPECTED_VALUE,
    MIN_LEG_DECIMAL_PRICE,
    MIN_LEG_EXPECTED_VALUE,
    MIN_LEG_PROBABILITY,
    build_pitcher_k_alt_line_legs,
    build_pitcher_parlay_frontier,
)
from run_mlb_pitcher_parlay_daily import (  # noqa: E402
    DEFAULT_CALIBRATION_LEDGER,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_WEB_DATA_ROOT,
    build_starters,
    write_web_payload,
)
from run_mlb_same_game_daily import extract_scheduled_games, fetch_todays_schedule  # noqa: E402


def _serialize_parlay(combo) -> Optional[dict[str, Any]]:
    if combo is None:
        return None
    result = combo.as_dict()
    result["selection_objective"] = "max_ev_after_probability_and_payout_gates"
    result["leg_probability_floor"] = MIN_LEG_PROBABILITY
    result["leg_expected_value_floor"] = MIN_LEG_EXPECTED_VALUE
    result["leg_decimal_price_floor"] = MIN_LEG_DECIMAL_PRICE
    result["joint_probability_floor"] = legacy_select.MIN_COMBO_JOINT_PROBABILITY
    result["combo_decimal_price_floor"] = MIN_COMBO_DECIMAL_PRICE
    result["combo_expected_value_floor"] = MIN_COMBO_EXPECTED_VALUE
    result["price_efficiency_passed"] = bool(
        combo.expected_value_per_unit is not None
        and combo.expected_value_per_unit >= MIN_COMBO_EXPECTED_VALUE
        and combo.combo_decimal_price is not None
        and combo.combo_decimal_price >= MIN_COMBO_DECIMAL_PRICE
        and combo.leg_a.expected_value_per_unit is not None
        and combo.leg_a.expected_value_per_unit >= MIN_LEG_EXPECTED_VALUE
        and combo.leg_b.expected_value_per_unit is not None
        and combo.leg_b.expected_value_per_unit >= MIN_LEG_EXPECTED_VALUE
        and combo.leg_a.decimal_price is not None
        and combo.leg_a.decimal_price >= MIN_LEG_DECIMAL_PRICE
        and combo.leg_b.decimal_price is not None
        and combo.leg_b.decimal_price >= MIN_LEG_DECIMAL_PRICE
    )
    # Frontend's historical Number(null) behavior rendered missing no-vig
    # probability as 0.0%. Missing means genuinely unavailable for one-sided
    # alt thresholds, so omit the field instead of fabricating zero.
    if result.get("naive_no_vig_combo_probability") is None:
        result.pop("naive_no_vig_combo_probability", None)
    return result


def build_daily_payload(
    *,
    run_date: date,
    calibration_ledger: Optional[Path] = DEFAULT_CALIBRATION_LEDGER,
    schedule_payload: Optional[dict[str, Any]] = None,
    fanduel_provider: Optional[FanduelPublicMlbProvider] = None,
    fetch_season_stats=legacy_select.k_model.fetch_pitcher_season_stats,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": generated_at,
        "run_date": run_date.isoformat(),
        "model": "mlb_pitcher_k_parlay_v1",
        "selector": "pitcher_alt_line_roi_frontier_v3",
        "selection_policy": {
            "leg_probability_floor": MIN_LEG_PROBABILITY,
            "leg_expected_value_floor": MIN_LEG_EXPECTED_VALUE,
            "leg_decimal_price_floor": MIN_LEG_DECIMAL_PRICE,
            "joint_probability_floor": legacy_select.MIN_COMBO_JOINT_PROBABILITY,
            "combo_decimal_price_floor": MIN_COMBO_DECIMAL_PRICE,
            "combo_expected_value_floor": MIN_COMBO_EXPECTED_VALUE,
            "objective": "maximize_actual_price_model_ev_after_probability_ev_and_payout_gates",
            "cross_game_only": True,
            "undercut_fallback_allowed": False,
        },
    }

    schedule = schedule_payload if schedule_payload is not None else fetch_todays_schedule(run_date)
    games = extract_scheduled_games(schedule)
    if not games:
        payload["status"] = "no_real_games_scheduled_today"
        return payload

    starters = build_starters(games)
    payload["real_starters_posted"] = len(starters)
    if not starters:
        payload["status"] = "no_real_probable_starters_posted_yet"
        return payload

    provider = fanduel_provider or FanduelPublicMlbProvider()
    odds_result = provider.collect_player_props()
    payload["odds_status"] = odds_result.get("status")
    odds_rows = odds_result.get("odds", []) if odds_result.get("status") == "success" else []

    calibration_store = CalibrationStore(calibration_ledger) if calibration_ledger else None
    legs = build_pitcher_k_alt_line_legs(
        starters,
        odds_rows,
        season=run_date.year,
        calibration_store=calibration_store,
        calibration_as_of=generated_at,
        fetch_season_stats=fetch_season_stats,
    )
    payload["legs"] = [leg.as_dict() for leg in legs]
    payload["real_priced_legs"] = sum(1 for leg in legs if leg.price_confirmed)
    payload["distinct_real_lines"] = len(
        {(leg.pitcher_id, leg.line, leg.side) for leg in legs if leg.price_confirmed}
    )

    combo = build_pitcher_parlay_frontier(legs)

    # Keep the old max-hit rule as a diagnostic control whether or not the ROI
    # selector abstains. This makes the cost of refusing undercut odds visible
    # prospectively rather than hiding it.
    max_hit_control = legacy_select.build_pitcher_parlay(
        legs,
        min_combo_joint_probability=legacy_select.MIN_COMBO_JOINT_PROBABILITY,
    )
    payload["max_hit_control"] = _serialize_parlay(max_hit_control)

    if combo is None:
        payload["parlay"] = None
        payload["parlay_status"] = "abstain_no_roi_efficient_probability_safe_pair"
        return payload

    payload["parlay"] = _serialize_parlay(combo)
    payload["parlay_status"] = "roi_efficient_probability_safe_shadow"

    if max_hit_control is not None and max_hit_control.expected_value_per_unit is not None:
        payload["selection_ev_lift_vs_max_hit_control"] = (
            combo.expected_value_per_unit - max_hit_control.expected_value_per_unit
        )
        payload["selection_joint_probability_delta_vs_max_hit_control"] = (
            combo.naive_independence_probability - max_hit_control.naive_independence_probability
        )
        payload["selection_decimal_price_lift_vs_max_hit_control"] = (
            (combo.combo_decimal_price or 0.0) - (max_hit_control.combo_decimal_price or 0.0)
        )

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", type=date.fromisoformat, default=None)
    parser.add_argument("--calibration-ledger", type=Path, default=DEFAULT_CALIBRATION_LEDGER)
    parser.add_argument("--web-data-root", type=Path, default=DEFAULT_WEB_DATA_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_date = args.run_date or date.today()
    payload = build_daily_payload(run_date=run_date, calibration_ledger=args.calibration_ledger)
    out_path = write_web_payload(payload, web_data_root=args.web_data_root, filename=DEFAULT_OUTPUT_FILENAME)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selector": payload.get("selector"),
                "real_starters_posted": payload.get("real_starters_posted", 0),
                "real_priced_legs": payload.get("real_priced_legs", 0),
                "distinct_real_lines": payload.get("distinct_real_lines", 0),
                "parlay_status": payload.get("parlay_status"),
                "selection_ev_lift_vs_max_hit_control": payload.get("selection_ev_lift_vs_max_hit_control"),
                "selection_decimal_price_lift_vs_max_hit_control": payload.get("selection_decimal_price_lift_vs_max_hit_control"),
                "written": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
