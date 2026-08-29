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
        "holdout_metrics": {"brier_improvement": 0.005},
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
    assert probability == pytest.approx(0.73)
    assert probability <= 0.78


def test_calibration_authority_is_scaled_by_demonstrated_holdout_lift() -> None:
    assert tight.calibration_authority_weight(_calibration()) == pytest.approx(0.5)
    weak = {**_calibration(), "holdout_metrics": {"brier_improvement": 0.001}}
    assert tight.calibration_authority_weight(weak) == pytest.approx(0.1)


def test_tight_play_recalculates_ev_after_probability_haircut() -> None:
    play = _play(model_p=0.80, estimated_p=0.78, price=-150)
    tightened, reasons = tight.tighten_play(play, _calibration())
    assert reasons == []
    assert tightened["estimated_hit_probability"] == pytest.approx(0.73)
    assert tightened["final_hit_probability"] == pytest.approx(0.73)
    assert tightened["expected_value_per_unit"] == pytest.approx(0.73 * (1 + 100 / 150) - 1)
    assert tightened["expected_value_per_unit"] < play["expected_value_per_unit"]


def test_dynamic_gate_accepts_sub_65pct_probability_when_exact_price_has_positive_value() -> None:
    play = _play(model_p=0.70, estimated_p=0.62, price=100)
    tightened, reasons = tight.tighten_play(play, _calibration())
    assert tightened["final_hit_probability"] < 0.65
    assert tightened["dynamic_break_even_probability"] == pytest.approx(0.50)
    assert tightened["dynamic_probability_margin"] >= 0.01
    assert reasons == []


def test_dynamic_gate_has_no_pick_count_or_relative_rank_constraint() -> None:
    payload = {
        "policy_profile": "premium_evidence_gated_v16",
        "plays": [
            _play(model_p=0.70, estimated_p=0.62, price=100, player=f"play_{index}")
            for index in range(30)
        ],
    }
    result = tight.apply_overlay(payload, _calibration())
    assert len(result["plays"]) == 30
    assert result["tight_quality_overlay"]["pick_count_constraint"].startswith("none")


def test_confidence_floor_rejects_low_probability_longshot_despite_positive_ev() -> None:
    _tightened, reasons = tight.tighten_play(
        _play(model_p=0.55, estimated_p=0.55, price=120, player="longshot"),
        _calibration(),
    )
    assert "balanced_hit_probability_below_60pct" in reasons


def test_price_frontier_requires_value_buffer_not_just_bare_break_even() -> None:
    _tightened, reasons = tight.tighten_play(
        _play(model_p=0.70, estimated_p=0.605, price=-150, player="thin_edge"),
        _calibration(),
    )
    assert "dynamic_probability_edge_below_1pct" in reasons


def test_prior_eight_play_strategy_shape_survives_without_outcome_tuning() -> None:
    # Frozen pre-settlement inputs from the 2026-08-28 early board. Outcomes
    # are deliberately absent: this protects the strategy shape without
    # selecting on which individual plays later won or lost.
    rows = [
        ("Brandon Lowe", 0.815727, 0.730682, -165),
        ("Spencer Horwitz", 0.781220, 0.713614, -155),
        ("Kyle Tucker", 0.667940, 0.669469, -160),
        ("Hunter Goodman", 0.661150, 0.666442, -175),
        ("Jake Cronenworth", 0.649603, 0.661296, -125),
        ("Munetaka Murakami", 0.656869, 0.664535, -175),
        ("Shohei Ohtani", 0.649149, 0.601889, -105),
        ("Matt Olson", 0.654641, 0.604392, -125),
    ]
    calibration = _calibration()
    calibration["breakpoints"] = [
        [0.608322, 0.602516],
        [0.670926, 0.602516],
        [0.682972, 0.617716],
        [0.741007, 0.617716],
        [0.755451, 0.674312],
        [0.830188, 0.674312],
    ]
    calibration["holdout_metrics"] = {"brier_improvement": 0.0013295173537561655}
    payload = {
        "policy_profile": "premium_evidence_gated_v16",
        "plays": [
            _play(model_p=model_p, estimated_p=estimated_p, price=price, player=player)
            for player, model_p, estimated_p, price in rows
        ],
    }
    result = tight.apply_overlay(payload, calibration)
    assert [play["player"] for play in result["plays"]]
    assert len(result["plays"]) == len(rows)


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
