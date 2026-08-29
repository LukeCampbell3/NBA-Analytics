from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import apply_tight_quality_overlay as tight  # noqa: E402


def _calibration() -> dict:
    # Small monotonic fixture: raw 0.80 maps to empirical 0.68.
    return {
        "status": "active",
        "model_version": "fixture",
        "breakpoints": [[0.60, 0.58], [0.80, 0.68], [1.00, 0.80]],
    }


def _play(*, model_p=0.80, estimated_p=0.78, price=-150, risk_flags=None, player="A") -> dict:
    decimal = 1.0 + 100.0 / abs(price) if price < 0 else 1.0 + price / 100.0
    return {
        "play_key": player,
        "player": player,
        "target": "H",
        "market_line": 0.5,
        "model_hit_probability": model_p,
        "estimated_hit_probability": estimated_p,
        "selected_side_price": price,
        "market_implied_probability": 1.0 / decimal,
        "expected_value_per_unit": estimated_p * decimal - 1.0,
        "price_confirmed": True,
        "risk_flags": risk_flags or [],
        "final_pool_quality_score": 0.7,
    }


def test_final_probability_is_negative_authority_only() -> None:
    probability, historical = tight.final_probability(_play(), _calibration())
    assert historical == pytest.approx(0.68)
    assert probability == pytest.approx(0.68)
    assert probability <= 0.78


def test_tight_play_recalculates_ev_after_probability_haircut() -> None:
    play = _play(model_p=0.80, estimated_p=0.78, price=-150)
    tightened, reasons = tight.tighten_play(play, _calibration())
    assert reasons == []
    assert tightened["estimated_hit_probability"] == pytest.approx(0.68)
    assert tightened["final_hit_probability"] == pytest.approx(0.68)
    assert tightened["expected_value_per_unit"] == pytest.approx(0.68 * (1 + 100 / 150) - 1)
    assert tightened["expected_value_per_unit"] < play["expected_value_per_unit"]


def test_dynamic_gate_accepts_sub_65pct_probability_when_exact_price_has_positive_value() -> None:
    play = _play(model_p=0.70, estimated_p=0.78, price=100)
    tightened, reasons = tight.tighten_play(play, _calibration())
    assert tightened["final_hit_probability"] < 0.65
    assert tightened["dynamic_break_even_probability"] == pytest.approx(0.50)
    assert tightened["dynamic_probability_margin"] > 0.0
    assert reasons == []


def test_dynamic_gate_has_no_pick_count_or_relative_rank_constraint() -> None:
    payload = {
        "policy_profile": "premium_evidence_gated_v16",
        "plays": [
            _play(model_p=0.70, estimated_p=0.78, price=100, player=f"play_{index}")
            for index in range(30)
        ],
    }
    result = tight.apply_overlay(payload, _calibration())
    assert len(result["plays"]) == 30
    assert result["tight_quality_overlay"]["pick_count_constraint"].startswith("none")


def test_tight_overlay_rejects_lineup_unconfirmed() -> None:
    _tightened, reasons = tight.tighten_play(
        _play(model_p=0.80, estimated_p=0.78, price=-150, risk_flags=["lineup_unconfirmed"]),
        _calibration(),
    )
    assert "lineup_unconfirmed" in reasons


def test_tight_overlay_rejects_negative_ev_after_recalibration() -> None:
    # -300 requires 75% break-even; the historical haircut to 68% makes it
    # negative-EV even though the pre-tight 78% estimate looked positive.
    tightened, reasons = tight.tighten_play(_play(model_p=0.80, estimated_p=0.78, price=-300), _calibration())
    assert tightened["expected_value_per_unit"] < 0.0
    assert "final_price_ev_negative" in reasons


def test_overlay_ranks_price_efficiency_only_after_quality_gates() -> None:
    higher_ev = _play(model_p=0.90, estimated_p=0.78, price=-110, player="higher_ev")
    lower_ev = _play(model_p=0.90, estimated_p=0.78, price=-180, player="lower_ev")
    payload = {"policy_profile": "premium_evidence_gated_v16", "plays": [lower_ev, higher_ev]}
    result = tight.apply_overlay(payload, _calibration())
    assert [play["player"] for play in result["plays"]] == ["higher_ev", "lower_ev"]
    assert result["policy_profile"] == tight.OVERLAY_VERSION
    assert result["base_policy_profile"] == "premium_evidence_gated_v16"
    assert result["tight_quality_overlay"]["parlay_v2_unchanged"] is True
