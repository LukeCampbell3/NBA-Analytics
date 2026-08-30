#!/usr/bin/env python3
"""HIGH_HIT_PARLAY_ROI_V2 -- probability-safe and price-efficient shadow.

This remains independent from the frozen PARLAY_POLICY_V2 evidence stream.
It reuses the same structurally-vetted single-prop candidate pool and the same
cross-game probability convention as HIGH_HIT_PARLAY_V1, but closes the price
failure mode exposed by ultra-short alternate lines:

* every leg must clear a 70% real probability floor;
* every leg must have non-negative model EV and decimal price >= 1.20;
* only two-leg, cross-game combinations are considered;
* joint probability must be >= 50%;
* combined decimal payout must be >= 2.00 (+100 or better);
* combined model EV must be >= 5%;
* after all gates, maximize EV, then joint probability.

If nothing clears every gate, abstain. This V2 policy is shadow-only and does
not retroactively alter V1 or PARLAY_POLICY_V2 evidence.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import select_high_precision_predictions as shp  # noqa: E402
from build_v11_eligible_training_set import parse_v11_args  # noqa: E402
from pick_survival_model import american_profit_per_unit, to_float  # noqa: E402
from safe_ev_optimizer import effective_probability  # noqa: E402

PRODUCT_VERSION = "HIGH_HIT_PARLAY_ROI_V2"
REPO_ROOT = SCRIPT_ROOT.parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "high_hit_parlay_predictions.json"

LEG_PROBABILITY_FLOOR = 0.70
JOINT_PROBABILITY_FLOOR = 0.50
MIN_LEG_DECIMAL_PRICE = 1.20
MIN_LEG_EXPECTED_VALUE = 0.0
MIN_COMBO_DECIMAL_PRICE = 2.00
MIN_COMBO_EXPECTED_VALUE = 0.05
MAX_LEGS = 2
MAX_PUBLISHED_PARLAYS = 5


def _decimal_price(candidate: Any) -> float | None:
    side_price = to_float(getattr(candidate, "selected_side_price", None))
    if side_price is None:
        return None
    profit = american_profit_per_unit(side_price)
    return None if profit is None else 1.0 + profit


def _probability(candidate: Any) -> float | None:
    value = effective_probability(candidate)
    return None if value is None else float(value)


def _leg_expected_value(candidate: Any) -> float | None:
    probability = _probability(candidate)
    decimal = _decimal_price(candidate)
    if probability is None or decimal is None:
        return None
    return probability * decimal - 1.0


def _leg_key(candidate: Any) -> tuple[str, str, str, str]:
    return (
        str(getattr(candidate, "player_id", None) or getattr(candidate, "player", "")).strip().lower(),
        str(getattr(candidate, "target", "")).upper(),
        str(getattr(candidate, "direction", "")).upper(),
        f"{float(getattr(candidate, 'market_line', 0.0)):.3f}",
    )


def eligible_legs(
    candidates: list[Any],
    *,
    probability_floor: float = LEG_PROBABILITY_FLOOR,
    min_decimal_price: float = MIN_LEG_DECIMAL_PRICE,
    min_expected_value: float = MIN_LEG_EXPECTED_VALUE,
) -> list[Any]:
    """Real, confirmed, high-probability legs that are not price traps."""
    legs: list[Any] = []
    for candidate in candidates:
        probability = _probability(candidate)
        decimal_price = _decimal_price(candidate)
        leg_ev = _leg_expected_value(candidate)
        if probability is None or decimal_price is None or leg_ev is None:
            continue
        if not bool(getattr(candidate, "price_confirmed", False)):
            continue
        if probability < probability_floor:
            continue
        if decimal_price < min_decimal_price:
            continue
        if leg_ev < min_expected_value:
            continue
        legs.append(candidate)
    return legs


def build_combos(
    legs: list[Any],
    *,
    max_legs: int = MAX_LEGS,
    joint_probability_floor: float = JOINT_PROBABILITY_FLOOR,
    min_combo_decimal_price: float = MIN_COMBO_DECIMAL_PRICE,
    min_combo_expected_value: float = MIN_COMBO_EXPECTED_VALUE,
) -> list[dict[str, Any]]:
    """Every real probability-safe and price-efficient cross-game pair."""
    combos: list[dict[str, Any]] = []
    # V2 intentionally supports two legs only. Keep the argument for test/API
    # compatibility but never permit a caller to silently expand beyond two.
    max_size = min(2, int(max_legs))
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(legs, size):
            game_ids = {str(getattr(candidate, "game_id", "")) for candidate in combo}
            if len(game_ids) != size:
                continue
            keys = {_leg_key(candidate) for candidate in combo}
            if len(keys) != size:
                continue

            joint_probability = 1.0
            decimal_price = 1.0
            for candidate in combo:
                probability = _probability(candidate)
                price = _decimal_price(candidate)
                if probability is None or price is None:
                    joint_probability = 0.0
                    decimal_price = 0.0
                    break
                joint_probability *= probability
                decimal_price *= price

            expected_value = joint_probability * decimal_price - 1.0
            if joint_probability < joint_probability_floor:
                continue
            if decimal_price < min_combo_decimal_price:
                continue
            if expected_value < min_combo_expected_value:
                continue

            combos.append(
                {
                    "legs": list(combo),
                    "leg_count": size,
                    "joint_probability": joint_probability,
                    "decimal_price": decimal_price,
                    "expected_value_per_unit": expected_value,
                }
            )
    return combos


def select_high_hit_parlays(
    candidates: list[Any],
    *,
    leg_probability_floor: float = LEG_PROBABILITY_FLOOR,
    joint_probability_floor: float = JOINT_PROBABILITY_FLOOR,
    max_legs: int = MAX_LEGS,
    max_published: int = MAX_PUBLISHED_PARLAYS,
) -> list[dict[str, Any]]:
    """Apply probability gates first, then maximize price-efficient EV."""
    legs = eligible_legs(candidates, probability_floor=leg_probability_floor)
    combos = build_combos(legs, max_legs=max_legs, joint_probability_floor=joint_probability_floor)
    ranked = sorted(
        combos,
        key=lambda combo: (
            combo["expected_value_per_unit"],
            combo["joint_probability"],
            combo["decimal_price"],
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str, str, str]] = set()
    for combo in ranked:
        if len(selected) >= max_published:
            break
        keys = {_leg_key(candidate) for candidate in combo["legs"]}
        if keys & used_keys:
            continue
        selected.append(combo)
        used_keys |= keys
    return selected


def _leg_payload(candidate: Any) -> dict[str, Any]:
    return {
        "player": candidate.player,
        "player_id": candidate.player_id,
        "team": candidate.team,
        "game_id": candidate.game_id,
        "target": candidate.target,
        "direction": candidate.direction,
        "market_line": candidate.market_line,
        "probability": _probability(candidate),
        "leg_expected_value": _leg_expected_value(candidate),
        "v11_calibrated_hit_probability": to_float(getattr(candidate, "calibrated_hit_probability", None)),
        "safe_probability": getattr(candidate, "safe_probability", None),
        "winner_signature_model_status": getattr(candidate, "winner_signature_model_status", "disabled"),
        "decimal_price": _decimal_price(candidate),
        "american_price": to_float(getattr(candidate, "selected_side_price", None)),
        "sportsbook": getattr(candidate, "selected_sportsbook", None),
    }


def combo_payload(combo: dict[str, Any]) -> dict[str, Any]:
    return {
        "leg_count": combo["leg_count"],
        "joint_probability": combo["joint_probability"],
        "decimal_price": combo["decimal_price"],
        "expected_value_per_unit": combo["expected_value_per_unit"],
        "legs": [_leg_payload(candidate) for candidate in combo["legs"]],
    }


def best_shadow_fallback(legs: list[Any]) -> dict[str, Any] | None:
    """Best real cross-game high-hit pair when the product gate abstains."""
    candidates = []
    for pair in itertools.combinations(legs, 2):
        if len({str(getattr(candidate, "game_id", "")) for candidate in pair}) != 2:
            continue
        probabilities = [_probability(candidate) for candidate in pair]
        prices = [_decimal_price(candidate) for candidate in pair]
        if any(value is None for value in probabilities + prices):
            continue
        joint_probability = probabilities[0] * probabilities[1]
        decimal_price = prices[0] * prices[1]
        candidates.append({"legs": list(pair), "leg_count": 2, "joint_probability": joint_probability, "decimal_price": decimal_price, "expected_value_per_unit": joint_probability * decimal_price - 1.0})
    return max(candidates, key=lambda combo: (combo["joint_probability"], combo["expected_value_per_unit"], combo["decimal_price"])) if candidates else None


def build_payload(candidates: list[Any], *, run_date: str):
    eligible = eligible_legs(candidates)
    selected = select_high_hit_parlays(candidates)
    fallback = None if selected else best_shadow_fallback(eligible)
    return {
        "schema_version": 2,
        "product_version": PRODUCT_VERSION,
        "authority": "shadow_only",
        "compared_against": "premium_tight_quality_v17_shadow",
        "run_date": run_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "construction": {
            "leg_probability_floor": LEG_PROBABILITY_FLOOR,
            "joint_probability_floor": JOINT_PROBABILITY_FLOOR,
            "min_leg_decimal_price": MIN_LEG_DECIMAL_PRICE,
            "min_leg_expected_value": MIN_LEG_EXPECTED_VALUE,
            "min_combined_decimal_price": MIN_COMBO_DECIMAL_PRICE,
            "min_expected_value_per_unit": MIN_COMBO_EXPECTED_VALUE,
            "max_legs": MAX_LEGS,
            "cross_game_only": True,
            "joint_probability_method": "product_of_leg_probabilities_cross_game_independence",
            "price_method": "product_of_leg_decimal_prices",
            "ranking_after_gates": "expected_value_desc_then_joint_probability_desc",
        },
        "candidates_considered": len(candidates),
        "legs_eligible": len(eligible),
        "parlays": [combo_payload(combo) for combo in selected],
        "shadow_fallback": ({**combo_payload(fallback), "authorization_status": "SHADOW_ONLY", "selection_status": "WITHHELD_PRODUCT_GATES"} if fallback else None),
        "abstain_reason": None if selected else "no_pair_cleared_probability_price_and_ev_floors",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--run-date", type=str, required=True)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    selector_args = parse_v11_args(args.pool_csv)
    candidates, _rejected = shp.prepare_and_filter_candidates(selector_args)
    payload = build_payload(candidates, run_date=args.run_date)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "parlays"}, indent=2))
    print(f"parlays selected: {len(payload['parlays'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
