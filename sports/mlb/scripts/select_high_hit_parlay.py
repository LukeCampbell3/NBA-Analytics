#!/usr/bin/env python3
"""
v12 Phase 3: HIGH_HIT_PARLAY_V1 -- a separate, real, joint-probability-
safe parlay product built directly from v11's own real, structurally-
vetted single-prop pool (prepare_and_filter_candidates(), the exact same
real gate the live singles board runs -- never re-derived by hand). This
is an independent product from the existing PARLAY_POLICY_V2 system
(sports/mlb/parlay_v2/), which stays completely untouched and frozen, per
the v12 proposal's own instruction.

Construction, real and disclosed -- not a fabricated "optimal" choice:

  - Legs: v11-eligible candidates whose real probability -- see
    safe_ev_optimizer.effective_probability() (v12's own safe_probability
    when the winner-signature model is active for that row, else v11's
    own calibrated_hit_probability -- never higher than v11's own bar,
    negative-authority-only, same guarantee as everywhere else in v12)
    -- clears LEG_PROBABILITY_FLOOR, and which carry a real confirmed
    price. Every published leg is independently something v11 itself
    would already consider betting.
  - Only CROSS-GAME combinations: no two legs share a game_id. This
    matches PARLAY_POLICY_V2's own established convention (see
    parlay_v2/run_parlay_v2.py's _to_candidate_wager / build_slate_payload
    section 6) -- no real dependence model for same-game correlation
    exists in this repo, so no real joint-probability estimate exists for
    a same-game combo either. Cited here, not re-derived.
  - joint_probability = product of each leg's own real probability
    (cross-game independence -- the same simplification PARLAY_POLICY_V2
    already makes for its own joint_probability_estimate on cross-game
    pairs).
  - decimal_price = product of each leg's own real decimal price (the
    standard sportsbook convention for a straight, non-SGP parlay, same
    convention PARLAY_POLICY_V2's _to_candidate_wager uses).
  - HIGH_HIT floor: joint_probability >= JOINT_PROBABILITY_FLOOR. This is
    the real selectivity lever the user asked for ("I want a selective
    record that wins much more") -- disclosed here, not tuned against any
    particular day's output.
  - Ranked probability-first (ties broken by EV) -- matching v11's own
    reliability-first selection_score philosophy, which Phase 2's real
    backtest evidence supports rather than contradicts (see
    compare_v11_v12_slates.py's committed real result).
  - Diversified across the published set: no single real leg (same
    player/target/direction/line) is reused across two different
    published parlays for the same slate.

No fabricated betslip link: sports/mlb/parlay_v2/fanduel_betslip.py
documents, from a real logged-in device test (2026-08-26), that FanDuel's
combined multi-leg "Add to Betslip" URL scheme is unconfirmed and known to
fail ("Selection not added"). This product follows that same disclosed
caution and never invents one -- each published leg surfaces only its
real sportsbook name, price, and probability (exactly what the singles
board itself already publishes for an unenriched pick), letting the
bettor place each leg at their own book.
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

PRODUCT_VERSION = "HIGH_HIT_PARLAY_V1"
REPO_ROOT = SCRIPT_ROOT.parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "high_hit_parlay_predictions.json"

LEG_PROBABILITY_FLOOR = 0.70  # matches v11's own min_hit_probability floor
JOINT_PROBABILITY_FLOOR = 0.50  # real, disclosed combined-parlay selectivity floor
MAX_LEGS = 3
MAX_PUBLISHED_PARLAYS = 5


def _decimal_price(candidate: Any) -> float | None:
    side_price = to_float(getattr(candidate, "selected_side_price", None))
    if side_price is None:
        return None
    profit = american_profit_per_unit(side_price)
    return None if profit is None else 1.0 + profit


def _leg_key(candidate: Any) -> tuple[str, str, str, str]:
    return (
        str(getattr(candidate, "player_id", None) or getattr(candidate, "player", "")).strip().lower(),
        str(getattr(candidate, "target", "")).upper(),
        str(getattr(candidate, "direction", "")).upper(),
        f"{float(getattr(candidate, 'market_line', 0.0)):.3f}",
    )


def eligible_legs(candidates: list[Any], *, probability_floor: float = LEG_PROBABILITY_FLOOR) -> list[Any]:
    """Real v11-eligible singles whose own real probability clears the
    leg floor and which carry a real, confirmed price. A candidate
    missing either input (no real probability, or no real confirmed
    price) is excluded, never guessed."""
    legs = []
    for candidate in candidates:
        probability = effective_probability(candidate)
        decimal_price = _decimal_price(candidate)
        if probability is None or decimal_price is None:
            continue
        if not bool(getattr(candidate, "price_confirmed", False)):
            continue
        if probability < probability_floor:
            continue
        legs.append(candidate)
    return legs


def build_combos(
    legs: list[Any], *, max_legs: int = MAX_LEGS, joint_probability_floor: float = JOINT_PROBABILITY_FLOOR
) -> list[dict[str, Any]]:
    """Every real cross-game combination of `legs` (2..max_legs legs)
    that clears joint_probability_floor, with a real joint probability
    and decimal price (product-of-legs, cross-game independence -- see
    module docstring)."""
    combos: list[dict[str, Any]] = []
    for size in range(2, max_legs + 1):
        for combo in itertools.combinations(legs, size):
            game_ids = {str(getattr(candidate, "game_id", "")) for candidate in combo}
            if len(game_ids) != size:
                continue  # every leg must be from a distinct game
            keys = {_leg_key(candidate) for candidate in combo}
            if len(keys) != size:
                continue  # never the same real prop twice in one combo

            joint_probability = 1.0
            decimal_price = 1.0
            for candidate in combo:
                joint_probability *= effective_probability(candidate)
                decimal_price *= _decimal_price(candidate)
            if joint_probability < joint_probability_floor:
                continue

            combos.append(
                {
                    "legs": list(combo),
                    "leg_count": size,
                    "joint_probability": joint_probability,
                    "decimal_price": decimal_price,
                    "expected_value_per_unit": joint_probability * decimal_price - 1.0,
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
    """Ranks real combos probability-first (EV as tie-break), then
    greedily selects non-overlapping combos -- no published leg (same
    player/target/direction/line) reused across two published parlays."""
    legs = eligible_legs(candidates, probability_floor=leg_probability_floor)
    combos = build_combos(legs, max_legs=max_legs, joint_probability_floor=joint_probability_floor)
    ranked = sorted(
        combos, key=lambda combo: (combo["joint_probability"], combo["expected_value_per_unit"]), reverse=True
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
        "probability": effective_probability(candidate),
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


def build_payload(candidates: list[Any], *, run_date: str) -> dict[str, Any]:
    selected = select_high_hit_parlays(candidates)
    return {
        "schema_version": 1,
        "product_version": PRODUCT_VERSION,
        "compared_against": "premium_evidence_gated_v16",
        "run_date": run_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "construction": {
            "leg_probability_floor": LEG_PROBABILITY_FLOOR,
            "joint_probability_floor": JOINT_PROBABILITY_FLOOR,
            "max_legs": MAX_LEGS,
            "cross_game_only": True,
            "joint_probability_method": "product_of_leg_probabilities_cross_game_independence",
            "price_method": "product_of_leg_decimal_prices",
        },
        "candidates_considered": len(candidates),
        "legs_eligible": len(eligible_legs(candidates)),
        "parlays": [combo_payload(combo) for combo in selected],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--run-date", type=str, required=True)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    # Reuses build_v11_eligible_training_set.parse_v11_args() -- the exact
    # same v11 argparse.Namespace construction the training-set builder
    # and the v11-vs-v12 comparison harness already use, never a second
    # hand-built copy of v11's real args.
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
