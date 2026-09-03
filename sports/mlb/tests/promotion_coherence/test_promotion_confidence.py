"""Unit tests pinning the coherent-promotion decision on real recorded payloads.

The 2026-09-02 recovered payloads are the canonical coherence-gap case
this branch exists to close: the payload's own `public_quality_overlay`
returned ABSTAIN with three specific blocking reasons, but
`parlays.action` was still ACT and the parlay was published as a
promoted pick. A regression in the shadow layer here would be a
regression in the exact behavior we want to prove.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence import (
    PromotionThresholds,
    decide_coherent_promotion,
    promotion_confidence_components,
)
from sports.mlb.parlay_v2.promotion_coherence.promotion_confidence import (
    PromotionPenalties,
    _break_even_probability,
    _clip01,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
RECOVERED_SEPT2 = (
    REPO_ROOT
    / "sports/mlb/web/data/history/runs/2026-09-02/recovered-richest/daily_predictions.json"
)


def _load_sept2_payload() -> dict:
    if not RECOVERED_SEPT2.exists():
        pytest.skip(f"recovered Sept 2 payload missing: {RECOVERED_SEPT2}")
    return json.loads(RECOVERED_SEPT2.read_text())


def test_break_even_probability_matches_price_reciprocal() -> None:
    assert _break_even_probability(2.5) == pytest.approx(0.4)
    assert _break_even_probability(5.67) == pytest.approx(1.0 / 5.67)
    assert _break_even_probability(None) is None
    assert _break_even_probability(1.0) is None  # non-priced / degenerate
    assert _break_even_probability("nope") is None


def test_clip01_bounds_deductions() -> None:
    assert _clip01(-0.5) == 0.0
    assert _clip01(0.0) == 0.0
    assert _clip01(0.42) == 0.42
    assert _clip01(1.4) == 1.0


def test_promotion_margin_is_joint_minus_break_even_when_no_penalties() -> None:
    payload = {
        "parlays": {
            "action": "ACT", "eligible": True,
            "public_quality_overlay": {
                "action": "ABSTAIN",
                "joint_probability": 0.42,
                "combined_decimal_price": 4.0,
                "leg_probabilities": [0.62, 0.68],
            },
        },
    }
    comps = promotion_confidence_components(payload)
    assert comps.calibrated_joint_probability == pytest.approx(0.42)
    assert comps.break_even_probability == pytest.approx(0.25)
    assert comps.total_deductions == 0.0
    # 0.42 - 0.25 = +0.17: coherent margin positive, but the overlay still
    # says ABSTAIN and the leg floors still block -- decide_coherent_
    # promotion must ABSTAIN despite a positive margin.
    assert comps.promotion_margin == pytest.approx(0.17)


def test_missing_price_or_joint_leaves_margin_undefined() -> None:
    payload = {"parlays": {"action": "ACT", "public_quality_overlay": {"joint_probability": 0.5}}}
    comps = promotion_confidence_components(payload)
    assert comps.promotion_margin is None


def test_penalties_stack_into_total_deductions() -> None:
    payload = {
        "parlays": {
            "public_quality_overlay": {
                "joint_probability": 0.6, "combined_decimal_price": 3.0,
                "leg_probabilities": [0.7, 0.75],
            }
        }
    }
    penalties = PromotionPenalties(
        uncertainty_deduction=0.02,
        market_disagreement_deduction=0.05,
        shared_failure_deduction=0.04,
        fragility_deduction=0.03,
    )
    comps = promotion_confidence_components(payload, penalties=penalties)
    assert comps.total_deductions == pytest.approx(0.14)
    # margin = 0.6 - 0.14 - 1/3
    assert comps.promotion_margin == pytest.approx(0.6 - 0.14 - 1.0 / 3.0)


def test_penalties_are_clipped_to_unit_interval() -> None:
    payload = {
        "parlays": {
            "public_quality_overlay": {
                "joint_probability": 0.9, "combined_decimal_price": 2.0,
                "leg_probabilities": [0.95, 0.95],
            }
        }
    }
    penalties = PromotionPenalties(
        uncertainty_deduction=-0.1,      # clipped to 0
        market_disagreement_deduction=1.5,  # clipped to 1
        shared_failure_deduction=0.2,
        fragility_deduction=0.1,
    )
    comps = promotion_confidence_components(payload, penalties=penalties)
    assert comps.uncertainty_deduction == 0.0
    assert comps.market_disagreement_deduction == 1.0
    assert comps.total_deductions == pytest.approx(1.3)


# --- the canonical Sept 2 regression ------------------------------------

def test_sept2_recovered_richest_reproduces_the_coherence_gap() -> None:
    payload = _load_sept2_payload()
    parlays = payload.get("parlays", {})
    overlay = parlays.get("public_quality_overlay", {})

    # First pin the payload actually carries the gap this branch exists
    # to close, so a payload-shape drift makes this test fail loudly
    # rather than silently pass.
    assert parlays.get("action") == "ACT", "live payload no longer publishes ACT for this case"
    assert overlay.get("action") == "ABSTAIN", "overlay no longer returns ABSTAIN for this case"
    assert overlay.get("joint_probability") == pytest.approx(0.2742624079757948)
    assert overlay.get("leg_probabilities") == pytest.approx([0.4948863276165676, 0.5541927361312965])

    decision = decide_coherent_promotion(payload)
    assert decision.live_action == "ACT"
    assert decision.action == "ABSTAIN"
    assert decision.live_action_agrees is False
    # Every one of the overlay's three blocking reasons must reappear in
    # the shadow's reason list -- the coherent gate is not merely
    # ABSTAINing, it is ABSTAINing for the same reasons the overlay
    # already identified.
    assert "quality_overlay_not_act" in decision.blocking_reasons
    assert "leg_1_probability_below_70pct" in decision.blocking_reasons
    assert "leg_2_probability_below_70pct" in decision.blocking_reasons
    assert "joint_probability_below_50pct" in decision.blocking_reasons


def test_sept2_promotion_margin_positive_but_blocked_by_other_rules() -> None:
    """The Sept 2 case has a positive raw margin (joint 0.274 vs
    break-even ~0.176 at combined price 5.67), so the margin rule alone
    would NOT block. This test proves the coherent gate blocks anyway
    because leg / joint / overlay rules fail -- the whole point of
    coherence is that no single measure is sufficient."""
    payload = _load_sept2_payload()
    decision = decide_coherent_promotion(payload)
    assert decision.action == "ABSTAIN"
    margin = decision.components.promotion_margin
    assert margin is not None and margin > 0.0
    # But: the promotion-margin rule is NOT in the blocking reasons.
    assert "promotion_margin_below_floor" not in decision.blocking_reasons
    # ... so removing the margin rule alone would not have caught this
    # ticket -- coherence is the point.


def test_coherent_gate_authorizes_when_every_rule_passes() -> None:
    payload = {
        "parlays": {
            "eligible": True, "action": "ACT",
            "public_quality_overlay": {
                "action": "ACT",
                "joint_probability": 0.55,
                "combined_decimal_price": 2.5,
                "leg_probabilities": [0.72, 0.74],
                "probability_edge": 0.06,
                "expected_value_per_unit": 0.08,
            },
            "selected_parlay": {"candidate_id": "unit-test"},
        }
    }
    decision = decide_coherent_promotion(payload)
    assert decision.action == "ACT"
    assert decision.blocking_reasons == []
    assert decision.live_action_agrees is True
    assert decision.candidate_id == "unit-test"


def test_high_promotion_margin_floor_can_block_even_a_passing_overlay() -> None:
    payload = {
        "parlays": {
            "eligible": True, "action": "ACT",
            "public_quality_overlay": {
                "action": "ACT",
                "joint_probability": 0.55,
                "combined_decimal_price": 2.5,
                "leg_probabilities": [0.72, 0.74],
            }
        }
    }
    # 0.55 - 0.40 = 0.15 raw margin
    tight = PromotionThresholds(
        min_leg_probability=0.70, min_joint_probability=0.50,
        min_promotion_margin=0.20,
    )
    decision = decide_coherent_promotion(payload, thresholds=tight)
    assert decision.action == "ABSTAIN"
    assert "promotion_margin_below_floor" in decision.blocking_reasons


def test_ineligibility_is_a_first_class_blocking_reason() -> None:
    payload = {
        "parlays": {
            "eligible": False, "action": "ABSTAIN",
            "public_quality_overlay": {
                "action": "ACT",
                "joint_probability": 0.6, "combined_decimal_price": 3.0,
                "leg_probabilities": [0.75, 0.8],
            }
        }
    }
    decision = decide_coherent_promotion(payload)
    assert decision.action == "ABSTAIN"
    assert "slate_not_eligible" in decision.blocking_reasons
