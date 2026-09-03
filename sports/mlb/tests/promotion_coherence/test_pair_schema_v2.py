"""Tests for the v2 pair-observation schema and its market-
disagreement deduction.

The v2 schema is additive: v1 rows still parse cleanly, with v2-added
fields returning None. `market_disagreement_deduction` returns 0.0
whenever a required signal is absent. Real-value behavior is tested on
hand-crafted rows since the live pair-ingest does not capture no-vig
market probability yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence import (
    MarketDisagreementProfile,
    PAIR_OBSERVATION_SCHEMA_VERSION_V2,
    PairObservationV2,
    PromotionPenalties,
    market_disagreement_deduction,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_PAIR_LEDGER = REPO_ROOT / "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"


def _v1_row(**overrides):
    """Approximate a real v1 pair-ledger row -- v2 fields absent."""
    base = {
        "predicted_joint_probability": 0.20,
        "quoted_pair_price": 4.5,
        "same_game": False,
        "same_team": False,
        "market_pair_type": "R|R",
        "slate_id": "20260826",
        "pair_id": "pair-1",
    }
    base.update(overrides)
    return base


def _v2_row(**overrides):
    """Fully-populated v2 row with per-leg model + no-vig market probabilities."""
    base = _v1_row()
    base.update({
        "leg_1_model_probability": 0.55,
        "leg_2_model_probability": 0.36,
        "leg_1_price": 1.9,
        "leg_2_price": 2.4,
        "leg_1_no_vig_market_probability": 0.52,
        "leg_2_no_vig_market_probability": 0.42,
    })
    base.update(overrides)
    return base


# --- schema -------------------------------------------------------------

def test_v1_row_parses_with_v2_fields_none() -> None:
    v = PairObservationV2.from_row(_v1_row())
    assert v.predicted_joint_probability == pytest.approx(0.20)
    assert v.same_game is False
    assert v.leg_1_model_probability is None
    assert v.leg_1_no_vig_market_probability is None
    assert v.has_v2_model_probabilities() is False
    assert v.has_v2_market_probabilities() is False


def test_v2_row_parses_with_all_fields_populated() -> None:
    v = PairObservationV2.from_row(_v2_row())
    assert v.leg_1_model_probability == pytest.approx(0.55)
    assert v.leg_2_no_vig_market_probability == pytest.approx(0.42)
    assert v.has_v2_model_probabilities() is True
    assert v.has_v2_market_probabilities() is True


def test_partial_v2_row_reports_missing_signals() -> None:
    # Model probabilities present, no-vig market probabilities absent.
    row = _v2_row()
    row.pop("leg_1_no_vig_market_probability")
    row.pop("leg_2_no_vig_market_probability")
    v = PairObservationV2.from_row(row)
    assert v.has_v2_model_probabilities() is True
    assert v.has_v2_market_probabilities() is False


def test_finite_helper_rejects_bad_values() -> None:
    row = _v2_row(leg_1_model_probability=float("nan"),
                  leg_2_model_probability=float("inf"))
    v = PairObservationV2.from_row(row)
    assert v.leg_1_model_probability is None
    assert v.leg_2_model_probability is None


def test_schema_version_constant_is_stable() -> None:
    assert PAIR_OBSERVATION_SCHEMA_VERSION_V2 == "PAIR_OBSERVATION_V2"


# --- market_disagreement_deduction --------------------------------------

def test_deduction_zero_on_v1_row_missing_signals() -> None:
    assert market_disagreement_deduction(_v1_row()) == 0.0


def test_deduction_zero_when_only_partial_signals_present() -> None:
    row = _v2_row()
    row.pop("leg_1_no_vig_market_probability")
    assert market_disagreement_deduction(row) == 0.0


def test_small_disagreement_below_threshold_returns_zero() -> None:
    # gaps of 0.02 each, threshold 0.03 -> both suppressed
    row = _v2_row(leg_1_model_probability=0.55, leg_1_no_vig_market_probability=0.53,
                  leg_2_model_probability=0.36, leg_2_no_vig_market_probability=0.38)
    assert market_disagreement_deduction(row) == 0.0


def test_disagreement_scales_with_gap() -> None:
    # gaps: leg_1 = 0.15, leg_2 = 0.06; both above 0.03 threshold
    # total gap 0.21, coefficient 0.30 -> 0.063
    row = _v2_row(leg_1_model_probability=0.70, leg_1_no_vig_market_probability=0.55,
                  leg_2_model_probability=0.36, leg_2_no_vig_market_probability=0.42)
    assert market_disagreement_deduction(row) == pytest.approx(0.30 * 0.21)


def test_deduction_caps_at_max() -> None:
    # gaps: 0.40 + 0.40 = 0.80; coefficient 0.30 -> 0.24, capped at 0.12
    row = _v2_row(leg_1_model_probability=0.90, leg_1_no_vig_market_probability=0.50,
                  leg_2_model_probability=0.90, leg_2_no_vig_market_probability=0.50)
    assert market_disagreement_deduction(row) == pytest.approx(0.12)


def test_custom_profile_overrides() -> None:
    row = _v2_row(leg_1_model_probability=0.60, leg_1_no_vig_market_probability=0.50,
                  leg_2_model_probability=0.40, leg_2_no_vig_market_probability=0.30)
    profile = MarketDisagreementProfile(
        disagreement_coefficient=0.5,
        disagreement_threshold=0.0,
        max_deduction=1.0,
    )
    # gaps 0.10 + 0.10 = 0.20; * 0.5 = 0.10
    assert market_disagreement_deduction(row, profile=profile) == pytest.approx(0.10)


# --- PromotionPenalties.from_pair_row wiring ----------------------------

def test_from_pair_row_populates_both_deductions_on_v2_same_game_row() -> None:
    row = _v2_row(same_game=True, market_pair_type="H|H",
                  leg_1_model_probability=0.70, leg_1_no_vig_market_probability=0.55,
                  leg_2_model_probability=0.36, leg_2_no_vig_market_probability=0.42)
    p = PromotionPenalties.from_pair_row(row)
    # shared-failure: 0.05 base (H|H is not total-market, same_team False)
    assert p.shared_failure_deduction == pytest.approx(0.05)
    # market disagreement: 0.30 * (0.15 + 0.06) = 0.063
    assert p.market_disagreement_deduction == pytest.approx(0.30 * 0.21)
    assert p.uncertainty_deduction == 0.0
    assert p.fragility_deduction == 0.0


def test_from_pair_row_populates_zero_on_v1_cross_game_row() -> None:
    p = PromotionPenalties.from_pair_row(_v1_row())
    assert p.shared_failure_deduction == 0.0
    assert p.market_disagreement_deduction == 0.0


# --- real-ledger regression: v1 rows keep the market-disagreement deduction at zero -----

def test_real_pair_ledger_rows_all_have_zero_market_disagreement_deduction() -> None:
    """Honest ledger regression: the live pair-ingest does not capture
    no-vig market probability yet, so every real row should produce a
    zero market-disagreement deduction. If this ever fails, it means
    the live capture started -- which is exactly the win the next-steps
    doc predicts and this test should then be updated to check
    non-zero values."""
    if not REAL_PAIR_LEDGER.exists():
        pytest.skip(f"real pair ledger missing: {REAL_PAIR_LEDGER}")
    n_checked = 0
    with open(REAL_PAIR_LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert market_disagreement_deduction(row) == 0.0
            n_checked += 1
            if n_checked >= 500:  # 500 rows is plenty
                break
    assert n_checked > 0
