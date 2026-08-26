"""Tests for the leakage-safe two-leg parlay policy (research/parlay_policy_v2).

These tests verify the *mechanism* -- gate ordering, penalty math, and
leakage-safety of the rolling regime gate -- against synthetic fixtures.
They do not assert any NBA historical hit rate: this repository does not
currently contain a settled NBA two-leg parlay dataset carrying the full
field set the policy requires (joint_sigma, joint_lcb, an actual sportsbook
SGP quote, injury/role/support state, shared-failure risk). See
sports/nba/predictions/Player-Predictor/research/parlay_policy_v2/REPORT.md.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.parlay_policy_v2.policy import (
    FORBIDDEN_LEGACY_FIELDS,
    ParlayPolicy,
    actual_quote_ev,
    american_to_decimal,
    apply_policy_frame,
    conservative_joint_probability,
    date_blocked_walk_forward,
    decimal_to_break_even,
    evaluate_candidate,
    naive_joint_probability,
    optimize_policy_grid,
    rank_eligible,
    rolling_regime_gate,
    usable_probability,
    validate_schema,
    wilson_interval,
)


def _base_candidate(**overrides: object) -> dict:
    candidate = {
        "leg_count": 2,
        "min_leg_probability": 0.74,
        "min_leg_sigma": 0.02,
        "joint_probability": 0.60,
        "joint_sigma": 0.05,
        "joint_lcb": 0.50,
        "dependency_penalty": 0.02,
        "actual_quote_decimal": 2.05,
        "shared_failure_risk": 0.15,
        "compatible_state_score": 0.80,
        "shift_risk": 0.10,
        "lineup_confirmed": True,
        "role_stable": True,
        "material_injury_uncertainty": False,
        "all_legs_in_support": True,
        "joint_model_reliable": True,
    }
    candidate.update(overrides)
    return candidate


# --- probability / uncertainty / dependency math -----------------------

def test_usable_probability_applies_lambda_sigma_penalty() -> None:
    assert usable_probability(0.80, 0.05, lam=1.0) == pytest.approx(0.75)


def test_usable_probability_clips_to_zero_one() -> None:
    assert usable_probability(0.02, 0.10, lam=1.0) == 0.0
    assert usable_probability(0.99, -5.0, lam=1.0) == 0.99  # negative sigma treated as 0, no penalty applied


def test_naive_joint_probability_multiplies_legs() -> None:
    assert naive_joint_probability([0.769, 0.769]) == pytest.approx(0.769 * 0.769)


def test_naive_joint_probability_rejects_out_of_range_leg() -> None:
    with pytest.raises(ValueError):
        naive_joint_probability([0.5, 1.2])


def test_conservative_joint_probability_is_below_naive_product() -> None:
    """Dependency penalty makes the usable joint probability strictly below
    the naive independent product, matching the observed-vs-naive gap this
    policy exists to correct for."""
    naive = naive_joint_probability([0.815, 0.815])
    conservative = conservative_joint_probability(
        joint_probability=naive, joint_sigma=0.03, uncertainty_lambda=1.0, dependency_penalty=0.03
    )
    assert conservative < naive


# --- eligibility gates ---------------------------------------------------

def test_eligible_candidate_passes_all_gates() -> None:
    result = evaluate_candidate(_base_candidate(), ParlayPolicy())
    assert result["eligible"] is True
    assert result["reasons"] == []


def test_high_ev_candidate_rejected_below_joint_probability_floor() -> None:
    """A candidate with an attractive quote can still fail on probability
    alone -- EV must never be allowed to compensate for low win probability."""
    candidate = _base_candidate(
        joint_probability=0.30, joint_sigma=0.02, dependency_penalty=0.0, actual_quote_decimal=4.50
    )
    result = evaluate_candidate(candidate, ParlayPolicy())
    assert result["eligible"] is False
    assert "JOINT_PROBABILITY" in result["reasons"]
    assert result["actual_quote_ev"] > 0  # EV alone looked attractive


def test_high_probability_candidate_rejected_on_negative_actual_quote_ev() -> None:
    """A well-calibrated, high-probability parlay must still be rejected if
    the real sportsbook quote is juiced enough to make EV negative."""
    candidate = _base_candidate(
        joint_probability=0.90, joint_sigma=0.01, dependency_penalty=0.0, actual_quote_decimal=1.05
    )
    result = evaluate_candidate(candidate, ParlayPolicy())
    assert result["eligible"] is False
    assert "ACTUAL_QUOTE_EV" in result["reasons"]


def test_actual_quote_ev_ignores_synthetic_leg_multiplication() -> None:
    """The EV gate must be computed from the real SGP quote, never from the
    product of each leg's individually-quoted decimal price."""
    leg_decimal_a, leg_decimal_b = 1.60, 1.60  # synthetic multiplied price = 2.56 -> looks great
    synthetic_price = leg_decimal_a * leg_decimal_b
    real_sgp_quote = 1.40  # sportsbooks vig same-game parlays more heavily than straight legs

    candidate = _base_candidate(
        joint_probability=0.80, joint_sigma=0.0, dependency_penalty=0.0, actual_quote_decimal=real_sgp_quote
    )
    result = evaluate_candidate(candidate, ParlayPolicy())

    assert synthetic_price > real_sgp_quote
    assert result["actual_quote_ev"] == pytest.approx(actual_quote_ev(0.80, real_sgp_quote))
    assert result["actual_quote_ev"] != pytest.approx(actual_quote_ev(0.80, synthetic_price))


def test_leg_probability_floor_rejects_weak_leg_even_if_joint_looks_fine() -> None:
    candidate = _base_candidate(min_leg_probability=0.60, min_leg_sigma=0.02)
    result = evaluate_candidate(candidate, ParlayPolicy(min_leg_probability=0.68))
    assert result["eligible"] is False
    assert "LEG_PROBABILITY" in result["reasons"]


def test_leg_count_mismatch_rejected() -> None:
    candidate = _base_candidate(leg_count=3)
    result = evaluate_candidate(candidate, ParlayPolicy(leg_count=2))
    assert result["eligible"] is False
    assert "LEG_COUNT" in result["reasons"]


def test_shared_failure_risk_gate_rejects_correlated_legs() -> None:
    """This is the PTS_OVER + PTS_OVER-style shared-failure-state gate."""
    candidate = _base_candidate(shared_failure_risk=0.50)
    result = evaluate_candidate(candidate, ParlayPolicy(max_shared_failure_risk=0.35))
    assert result["eligible"] is False
    assert "SHARED_FAILURE" in result["reasons"]


def test_state_compatibility_gate_rejects_low_score() -> None:
    candidate = _base_candidate(compatible_state_score=0.40)
    result = evaluate_candidate(candidate, ParlayPolicy(min_compatible_state_score=0.60))
    assert result["eligible"] is False
    assert "STATE_COMPATIBILITY" in result["reasons"]


def test_shift_risk_gate_rejects_unstable_regime() -> None:
    candidate = _base_candidate(shift_risk=0.60)
    result = evaluate_candidate(candidate, ParlayPolicy(max_shift_risk=0.35))
    assert result["eligible"] is False
    assert "SHIFT_RISK" in result["reasons"]


def test_joint_uncertainty_gate_rejects_wide_distribution() -> None:
    candidate = _base_candidate(joint_sigma=0.25)
    result = evaluate_candidate(candidate, ParlayPolicy(max_joint_uncertainty=0.12))
    assert result["eligible"] is False
    assert "JOINT_UNCERTAINTY" in result["reasons"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("lineup_confirmed", False, "LINEUP"),
        ("role_stable", False, "ROLE"),
        ("material_injury_uncertainty", True, "INJURY_UNCERTAINTY"),
        ("all_legs_in_support", False, "OUT_OF_SUPPORT"),
        ("joint_model_reliable", False, "JOINT_MODEL_UNRELIABLE"),
    ],
)
def test_execution_state_gates_reject_individually(field: str, value: object, reason: str) -> None:
    candidate = _base_candidate(**{field: value})
    result = evaluate_candidate(candidate, ParlayPolicy())
    assert result["eligible"] is False
    assert reason in result["reasons"]


# --- schema / leakage guards ---------------------------------------------

def test_validate_schema_flags_missing_required_fields() -> None:
    errors = validate_schema({"leg_count": 2})
    assert any(e.startswith("missing:") for e in errors)


def test_validate_schema_forbids_legacy_cross_game_path_fields() -> None:
    candidate = _base_candidate(turn=3, accel_ratio=1.1)
    errors = validate_schema(candidate)
    assert any("legacy_path_fields_forbidden" in e for e in errors)
    for name in FORBIDDEN_LEGACY_FIELDS:
        assert name not in _base_candidate()  # the clean fixture never carries them


def test_evaluate_candidate_short_circuits_on_schema_error_before_gating() -> None:
    result = evaluate_candidate({"leg_count": 2}, ParlayPolicy())
    assert result["eligible"] is False
    assert result["reasons"][0].startswith("missing:")


def test_rolling_regime_gate_never_uses_the_current_slates_own_outcome() -> None:
    """A slate's health flag must be reproducible from strictly earlier
    outcomes alone -- recomputing it with the current (and all later)
    outcomes zeroed out must not change it."""
    outcomes = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    full = rolling_regime_gate(outcomes, window=5, min_history=5, min_recent_hit_rate=0.5)
    for i in range(len(outcomes)):
        truncated_view = outcomes[:i] + [0] * (len(outcomes) - i)  # blind the gate to i.. onward
        recomputed = rolling_regime_gate(truncated_view, window=5, min_history=5, min_recent_hit_rate=0.5)
        assert recomputed[i] == full[i]


def test_rolling_regime_gate_suspends_after_losing_streak() -> None:
    outcomes = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    active = rolling_regime_gate(outcomes, window=5, min_history=5, min_recent_hit_rate=0.5)
    assert active[:5] == [True] * 5  # not enough history yet to gate
    assert active[-1] is False  # last five outcomes were all losses


# --- pricing helpers -------------------------------------------------------

def test_american_to_decimal_positive_and_negative() -> None:
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-150) == pytest.approx(1.0 + 100.0 / 150.0)


def test_american_to_decimal_rejects_zero() -> None:
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_decimal_to_break_even_matches_probability() -> None:
    assert decimal_to_break_even(2.0) == pytest.approx(0.5)


def test_decimal_to_break_even_rejects_invalid_odds() -> None:
    with pytest.raises(ValueError):
        decimal_to_break_even(1.0)


# --- frame-level application / ranking -------------------------------------

def test_apply_policy_frame_and_rank_eligible_orders_by_ev_then_lcb() -> None:
    rows = pd.DataFrame(
        [
            _base_candidate(actual_quote_decimal=2.20, joint_lcb=0.50),  # best EV
            _base_candidate(actual_quote_decimal=2.05, joint_lcb=0.55),  # lower EV, better LCB
            _base_candidate(joint_probability=0.20, dependency_penalty=0.0),  # ineligible
        ]
    )
    applied = apply_policy_frame(rows, ParlayPolicy())
    assert applied["selected"].tolist() == [True, True, False]

    ranked = rank_eligible(applied)
    assert len(ranked) == 2
    assert ranked.iloc[0]["actual_quote_decimal"] == 2.20


def test_wilson_interval_empty_sample_is_nan() -> None:
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_interval_bounds_widen_with_smaller_sample() -> None:
    lo_small, hi_small = wilson_interval(8, 10)
    lo_large, hi_large = wilson_interval(80, 100)
    assert (hi_small - lo_small) > (hi_large - lo_large)


def test_optimize_policy_grid_prefers_wilson_lower_bound_over_raw_hit_rate() -> None:
    """A tiny 5-0 cell must not beat a larger, still-strong cell just because
    100% > 74% on raw hit rate -- this is the anti-cherry-picking guarantee."""
    rows = []
    # Large, reliable cell: 30/40 = 75% at min_leg_probability=0.60
    for i in range(40):
        rows.append(_base_candidate(min_leg_probability=0.75, joint_probability=0.65, won=1 if i < 30 else 0))
    # Tiny, lucky cell: 5/5 = 100% only unlocked at a much stricter floor
    for i in range(5):
        rows.append(_base_candidate(min_leg_probability=0.85, joint_probability=0.90, won=1))
    development = pd.DataFrame(rows)

    grid = [
        {"min_leg_probability": 0.60, "min_joint_probability": 0.50},
        {"min_leg_probability": 0.80, "min_joint_probability": 0.50},
    ]
    best_policy, table = optimize_policy_grid(development, grid, min_selected=20, min_coverage=0.10)
    assert best_policy.min_leg_probability == pytest.approx(0.60)
    # the tiny cell is present but excluded from winning by the sample-size gate
    assert bool(table.loc[table["min_leg_probability"] == 0.80, "grid_eligible"].iloc[0]) is False


def test_date_blocked_walk_forward_only_trains_on_strictly_earlier_dates() -> None:
    rows = []
    for day in range(1, 6):
        for i in range(15):
            rows.append(
                _base_candidate(
                    date=f"2026-03-{day:02d}",
                    min_leg_probability=0.85,
                    won=1 if i < 10 else 0,
                )
            )
    df = pd.DataFrame(rows)
    grid = [{"min_leg_probability": 0.74, "min_joint_probability": 0.50}]

    out = date_blocked_walk_forward(df, grid, min_train_rows=15, min_selected=10, min_coverage=0.10)
    assert not out.empty
    for _, row in out.iterrows():
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["date"])
