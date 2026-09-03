"""Tests for the same-game shared-failure penalty.

Anchored in the ledger evidence: on the real pair ledger, 100% of the
279 same-game pairs are below break-even. The penalty is designed to
push their promotion margin further negative -- more likely to be
blocked by any nonzero min_promotion_margin -- while leaving cross-game
rows unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence import (
    PromotionPenalties,
    SameGamePenaltyProfile,
    apply_same_game_penalty,
    same_game_shared_failure_deduction,
)
from sports.mlb.parlay_v2.promotion_coherence.backtest_pair_ledger import (
    DEFAULT_LEDGER,
    compute_promotion_margin,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _row(**overrides):
    base = {
        "predicted_joint_probability": 0.25,
        "quoted_pair_price": 4.0,
        "same_game": True,
        "same_team": False,
        "market_pair_type": "H|H",
    }
    base.update(overrides)
    return base


# --- unit ---------------------------------------------------------------

def test_cross_game_row_receives_zero_deduction() -> None:
    row = _row(same_game=False)
    assert same_game_shared_failure_deduction(row) == 0.0


def test_same_game_base_deduction_applied() -> None:
    row = _row()  # H|H, same_team=False, no total-market
    assert same_game_shared_failure_deduction(row) == pytest.approx(0.05)


def test_same_team_adds_extra_deduction() -> None:
    row = _row(same_team=True)
    assert same_game_shared_failure_deduction(row) == pytest.approx(0.05 + 0.03)


def test_total_market_adds_extra_deduction() -> None:
    row = _row(market_pair_type="TB|R")
    assert same_game_shared_failure_deduction(row) == pytest.approx(0.05 + 0.02)


def test_all_components_stack_and_cap_at_max() -> None:
    profile = SameGamePenaltyProfile(
        base_same_game_deduction=0.10,
        same_team_additional_deduction=0.05,
        total_line_fragility_deduction=0.05,
        max_total_deduction=0.12,
    )
    row = _row(same_team=True, market_pair_type="TB|TB")
    # 0.10 + 0.05 + 0.05 = 0.20, but capped at 0.12
    assert same_game_shared_failure_deduction(row, profile=profile) == pytest.approx(0.12)


def test_custom_profile_scales_deduction() -> None:
    profile = SameGamePenaltyProfile(base_same_game_deduction=0.01)
    row = _row()
    assert same_game_shared_failure_deduction(row, profile=profile) == pytest.approx(0.01)


# --- apply_same_game_penalty ---------------------------------------------

def test_apply_returns_raw_margin_for_cross_game() -> None:
    row = _row(same_game=False, predicted_joint_probability=0.30, quoted_pair_price=4.0)
    # 0.30 - 0 - 0.25 = 0.05
    assert apply_same_game_penalty(row) == pytest.approx(0.05)


def test_apply_subtracts_deduction_for_same_game() -> None:
    row = _row(predicted_joint_probability=0.30, quoted_pair_price=4.0)
    # 0.30 - 0.05 - 0.25 = 0.00
    assert apply_same_game_penalty(row) == pytest.approx(0.00)


def test_apply_returns_none_on_bad_row() -> None:
    assert apply_same_game_penalty({}) is None
    assert apply_same_game_penalty({"predicted_joint_probability": 0.3}) is None
    assert apply_same_game_penalty({"predicted_joint_probability": 0.3,
                                   "quoted_pair_price": 1.0, "same_game": True}) is None


# --- wiring: PromotionPenalties.from_pair_row ---------------------------

def test_promotion_penalties_from_pair_row_populates_shared_failure() -> None:
    row = _row(same_team=True, market_pair_type="TB|H")
    p = PromotionPenalties.from_pair_row(row)
    # 0.05 base + 0.03 same_team + 0.02 total-market
    assert p.shared_failure_deduction == pytest.approx(0.10)
    # Other deductions stay at their defaults
    assert p.uncertainty_deduction == 0.0
    assert p.market_disagreement_deduction == 0.0
    assert p.fragility_deduction == 0.0


def test_promotion_penalties_from_cross_game_pair_row_is_all_zero() -> None:
    p = PromotionPenalties.from_pair_row(_row(same_game=False))
    assert p.shared_failure_deduction == 0.0


# --- wiring: backtest compute_promotion_margin ---------------------------

def test_compute_promotion_margin_default_does_not_apply_penalty() -> None:
    row = _row(predicted_joint_probability=0.30, quoted_pair_price=4.0)
    # 0.30 - 0.25 = 0.05 (no penalty by default)
    assert compute_promotion_margin(row) == pytest.approx(0.05)


def test_compute_promotion_margin_with_penalty_reduces_same_game_row() -> None:
    row = _row(predicted_joint_probability=0.30, quoted_pair_price=4.0)
    # 0.30 - 0.05 (base) - 0.25 = 0.00
    assert compute_promotion_margin(row, apply_same_game_penalty=True) == pytest.approx(0.00)


def test_compute_promotion_margin_with_penalty_leaves_cross_game_unchanged() -> None:
    row = _row(same_game=False, predicted_joint_probability=0.30, quoted_pair_price=4.0)
    raw = compute_promotion_margin(row)
    penalized = compute_promotion_margin(row, apply_same_game_penalty=True)
    assert raw == penalized == pytest.approx(0.05)


# --- ledger regression: penalty tightens same-game admission -----------

def test_same_game_penalty_reduces_admission_share_at_zero_floor() -> None:
    """On the real pair ledger, same-game pairs are already 100% below
    break-even (empty admission at floor 0.0). Applying the penalty
    can't make it worse -- but on a synthetic fixture with some
    barely-positive-margin same-game pairs, it must strictly reduce
    admission count at a zero floor.
    """
    rows = [
        # Same-game pair with a barely-positive raw margin -- 0.30 - 0.25 = 0.05,
        # penalized: 0.30 - 0.05 - 0.25 = 0.00. Blocked at floor 0.01.
        _row(predicted_joint_probability=0.30, quoted_pair_price=4.0),
        # Same-game pair well above break-even -- survives even penalty.
        _row(predicted_joint_probability=0.50, quoted_pair_price=4.0),
        # Cross-game pair, unaffected.
        _row(same_game=False, predicted_joint_probability=0.30, quoted_pair_price=4.0),
    ]

    from sports.mlb.parlay_v2.promotion_coherence.backtest_pair_ledger import sweep_floors
    without = {r.floor: r for r in sweep_floors(rows, [0.01])}
    with_penalty = {r.floor: r for r in sweep_floors(rows, [0.01], apply_same_game_penalty=True)}
    # 3 rows admitted without penalty at floor 0.01; 2 rows with it
    # (the barely-positive same-game row drops out).
    assert without[0.01].admitted_count == 3
    assert with_penalty[0.01].admitted_count == 2
