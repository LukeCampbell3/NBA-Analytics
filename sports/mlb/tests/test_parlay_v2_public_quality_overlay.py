from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import apply_parlay_v2_public_quality_overlay as overlay  # noqa: E402


def _pair(p1=0.80, p2=0.78, d1=1.70, d2=1.75, joint=None):
    return {
        "joint_probability_estimate": p1 * p2 if joint is None else joint,
        "leg_1": {"model_probability_estimate": p1, "decimal_price": d1, "in_support": True},
        "leg_2": {"model_probability_estimate": p2, "decimal_price": d2, "in_support": True},
    }


def test_aug28_shape_is_withheld_by_tight_public_gate() -> None:
    result = overlay.audit_pair(_pair(p1=0.4873137823, p2=0.5649354352, d1=2.05, d2=1.7407407407, joint=0.2753008236))
    assert result["action"] == "ABSTAIN"
    assert "leg_1_probability_below_70pct" in result["blocking_reasons"]
    assert "leg_2_probability_below_70pct" in result["blocking_reasons"]
    assert "joint_probability_below_50pct" in result["blocking_reasons"]
    assert result["expected_value_per_unit"] < 0.05
    assert result["frozen_policy_mutated"] is False


def test_quality_pair_passes_without_changing_frozen_action() -> None:
    pair = _pair(p1=0.80, p2=0.78, d1=1.70, d2=1.75)
    audit = overlay.audit_pair(pair)
    assert audit["action"] == "PASS"
    payload = {"parlays": {"action": "ACT", "selected_parlay": pair, "policy_version": "FROZEN"}}
    result = overlay.apply_overlay(payload)
    assert result["parlays"]["action"] == "ACT"
    assert result["parlays"]["selected_parlay"] == pair
    assert result["parlays"]["policy_version"] == "FROZEN"
    assert result["parlays"]["public_quality_overlay"]["action"] == "PASS"


def test_negative_ev_pair_is_withheld_even_when_probability_passes() -> None:
    # 80% x 80%=64%, but 1.25 x 1.25 only pays 1.5625 decimal.
    result = overlay.audit_pair(_pair(p1=0.80, p2=0.80, d1=1.25, d2=1.25))
    assert result["action"] == "ABSTAIN"
    assert "combined_price_below_plus_100" in result["blocking_reasons"]
    assert "expected_value_below_5pct" in result["blocking_reasons"]
