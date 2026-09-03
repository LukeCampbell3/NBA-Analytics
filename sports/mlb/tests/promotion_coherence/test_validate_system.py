"""End-to-end validation regression.

The validation harness runs leave-one-slate-out backtests for RAW,
GLOBAL_BETA, and SLICE_CONDITIONED_BETA, evaluates each at a floor
sweep, and pronounces a threshold verdict.

These tests pin:
    * The mechanism works (RAW baseline reproduces the raw-backtest
      numbers previously reported).
    * The slice-conditioned strategy finds SOMETHING the raw does not
      -- specifically, at least one floor cell with positive OOS
      total return.
    * The strict PUSH threshold (>=100 admitted AND positive AND
      > RAW) is NOT currently met -- honest verdict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence.validate_system import (
    DEFAULT_LEDGER,
    MIN_ADMITTED_FOR_THRESHOLD,
    run_validation,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER = REPO_ROOT / DEFAULT_LEDGER


@pytest.fixture(scope="module")
def validation_report():
    if not LEDGER.exists():
        pytest.skip(f"real ledger missing: {LEDGER}")
    return run_validation(ledger_path=LEDGER)


def test_all_three_strategies_produce_folds(validation_report) -> None:
    for name in ("RAW", "GLOBAL_BETA", "SLICE_CONDITIONED_BETA"):
        folds = validation_report.strategies[name]
        assert len(folds) == 4, f"{name}: expected 4 LOSO folds, got {len(folds)}"


def test_raw_strategy_matches_earlier_backtest_ordering(validation_report) -> None:
    """RAW at floor 0.05 admits close to 117 pairs total (pinned in
    the earlier backtest as ALL_SETTLED_PAIRS strict-dominance
    number). LOSO reproduces the same total-of-parts."""
    raw_agg = validation_report.strategy_aggregates["RAW"]
    at_005 = raw_agg.per_floor_aggregate.get(0.05)
    assert at_005 is not None
    assert at_005.admitted == 117


def test_slice_conditioned_finds_a_positive_return_cell(validation_report) -> None:
    """The point of the resolution: at some floor, SLICE_CONDITIONED_
    BETA produces positive OOS return -- something no other strategy
    does on this ledger. Pinned so any future regression that erases
    the extracted signal fails loudly."""
    slice_agg = validation_report.strategy_aggregates["SLICE_CONDITIONED_BETA"]
    positive = [
        fa for fa in slice_agg.per_floor_aggregate.values()
        if fa.total_return_per_unit > 0
    ]
    assert positive, "SLICE_CONDITIONED_BETA no longer produces any positive-return cell"


def test_slice_conditioned_beats_raw_where_it_admits(validation_report) -> None:
    """Wherever the slice-conditioned strategy admits enough pairs to
    compare, its hit rate at that floor should beat RAW's baseline
    (~6-8%) meaningfully."""
    slice_agg = validation_report.strategy_aggregates["SLICE_CONDITIONED_BETA"]
    raw_agg = validation_report.strategy_aggregates["RAW"]
    matched = 0
    beat = 0
    for floor, fa in slice_agg.per_floor_aggregate.items():
        if fa.admitted < 20 or fa.hit_rate is None:
            continue
        raw_at = raw_agg.per_floor_aggregate.get(floor)
        if raw_at is None or raw_at.hit_rate is None:
            continue
        matched += 1
        if fa.hit_rate > raw_at.hit_rate:
            beat += 1
    assert matched >= 3
    # A meaningful majority of comparable floors should beat RAW hit rate
    assert beat > matched / 2, (
        f"slice-conditioned strategy no longer beats RAW hit rate on the "
        f"majority of comparable floors ({beat}/{matched})"
    )


def test_no_strategy_exceeds_strict_push_threshold_yet(validation_report) -> None:
    """The strict production-push threshold requires >=100 admitted
    pairs AND positive return AND strict improvement over RAW at the
    same floor. Currently no strategy meets ALL three -- the
    slice-conditioned strategy meets return and improvement but
    admits only ~22 pairs at its winning floor. Honest state; do not
    promote to production."""
    for name in ("RAW", "GLOBAL_BETA", "SLICE_CONDITIONED_BETA"):
        v = validation_report.threshold_verdicts[name]
        assert not v.exceeds_threshold, (
            f"{name} now exceeds the strict threshold: {v.reason}. "
            f"Time to consider promoting this strategy to production; "
            f"update this test and the analysis doc."
        )


def test_min_admitted_threshold_is_100(validation_report) -> None:
    """Anti-cherry-pick guard is fixed at 100 pairs across LOSO folds.
    Documented and pinned."""
    assert validation_report.min_admitted_for_threshold == MIN_ADMITTED_FOR_THRESHOLD == 100
