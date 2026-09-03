"""Backtest tests. Two purposes:

    1. Unit-test the sweep math on a small hand-built dataset -- so the
       report the CLI prints is trustworthy.
    2. Pin the *actual observed behavior* on the real pair-observation
       ledger this repo carries today. These regression tests do NOT
       assert the coherent rule wins; they assert what the numbers
       actually show, honestly, so a future run that drifts loudly fails
       here. If the underlying model or ledger changes and the direction
       of the effect changes, we want that to surface as a failing
       assertion we then update -- not as a silent shift in the
       shadow-replay conclusion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence.backtest_pair_ledger import (
    DEFAULT_LEDGER,
    FloorResult,
    build_report,
    build_slice_report,
    compute_promotion_margin,
    load_ledger,
    sweep_floors,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER_PATH = REPO_ROOT / DEFAULT_LEDGER


def _row(**overrides):
    base = {
        "predicted_joint_probability": 0.2,
        "quoted_pair_price": 4.0,
        "actual_pair_return": -1.0,
        "both_win": False,
        "settlement_status": "settled",
        "same_game": False,
        "market_pair_type": "R|R",
    }
    base.update(overrides)
    return base


# --- unit ---------------------------------------------------------------

def test_compute_promotion_margin_is_joint_minus_break_even() -> None:
    row = _row(predicted_joint_probability=0.30, quoted_pair_price=4.0)
    assert compute_promotion_margin(row) == pytest.approx(0.30 - 0.25)


def test_compute_promotion_margin_none_on_missing_fields() -> None:
    assert compute_promotion_margin({"quoted_pair_price": 4.0}) is None
    assert compute_promotion_margin({"predicted_joint_probability": 0.3}) is None
    assert compute_promotion_margin({"predicted_joint_probability": 0.3,
                                    "quoted_pair_price": 1.0}) is None
    assert compute_promotion_margin({"predicted_joint_probability": "n/a",
                                    "quoted_pair_price": 4.0}) is None


def test_sweep_floor_admits_only_rows_at_or_above_floor() -> None:
    # Everything below chosen to be exactly representable in IEEE-754
    # binary floating point so the >= comparisons never drift.
    # price=2 -> BE 0.5; price=4 -> BE 0.25; price=8 -> BE 0.125.
    rows = [
        _row(predicted_joint_probability=0.75,   quoted_pair_price=2.0),  # margin +0.25
        _row(predicted_joint_probability=0.375,  quoted_pair_price=4.0),  # margin +0.125
        _row(predicted_joint_probability=0.125,  quoted_pair_price=8.0),  # margin  0.000
        _row(predicted_joint_probability=0.0625, quoted_pair_price=8.0),  # margin -0.0625
    ]
    floors = [-0.5, 0.0, 0.125, 0.25, 0.5]
    results = {r.floor: r for r in sweep_floors(rows, floors)}
    assert results[-0.5].admitted_count == 4
    assert results[0.0].admitted_count == 3
    assert results[0.125].admitted_count == 2
    assert results[0.25].admitted_count == 1
    assert results[0.5].admitted_count == 0


def test_floor_result_return_math_is_a_faithful_sum() -> None:
    rows = [
        _row(actual_pair_return=+2.5, both_win=True),
        _row(actual_pair_return=-1.0),
        _row(actual_pair_return=-1.0),
    ]
    r = FloorResult.build(floor=-1.0, admitted=rows, total_count=3)
    assert r.wins == 1 and r.losses == 2
    assert r.hit_rate == pytest.approx(1 / 3)
    assert r.total_return_per_unit == pytest.approx(0.5)
    assert r.mean_return_per_unit == pytest.approx(0.5 / 3)


def test_strict_dominance_requires_min_admitted_pairs() -> None:
    """A floor that admits fewer than 20 pairs must not be reported as
    strictly dominant -- prevents an accidental one-lucky-pick 'win'
    from being sold as a rule."""
    baseline_rows = [_row(actual_pair_return=-1.0) for _ in range(100)]
    dominant_stub = _row(actual_pair_return=+10.0, both_win=True,
                         predicted_joint_probability=0.90, quoted_pair_price=1.5)
    report = build_slice_report(
        name="tiny", filter_description="",
        rows=baseline_rows + [dominant_stub],
        floors=[-1.0, +0.30],
    )
    # The +0.30 floor admits exactly one row (the lucky stub) -- must
    # not be flagged as strictly dominant.
    assert report.strict_dominance_over_baseline is None


# --- observed-behavior regression on the real ledger --------------------

@pytest.fixture(scope="module")
def ledger_report():
    if not LEDGER_PATH.exists():
        pytest.skip(f"pair observation ledger missing: {LEDGER_PATH}")
    return build_report(
        ledger_path=LEDGER_PATH,
        floors=[round(x / 100, 2) for x in range(-10, 11, 1)],
    )


def _get_slice(report, name):
    for s in report.slices:
        if s.name == name:
            return s
    pytest.fail(f"slice {name!r} not in report")


def test_real_ledger_has_the_scale_we_backtested_against(ledger_report) -> None:
    """3120 rows across 4 slates as of this branch. If this drops, the
    ledger got truncated; if it grows meaningfully, the numbers below
    may need to be re-pinned."""
    assert ledger_report.ledger_row_count >= 3000
    assert ledger_report.settled_row_count == ledger_report.ledger_row_count


def test_real_ledger_baseline_is_broadly_negative(ledger_report) -> None:
    """Honest baseline finding: accepting every scored pair loses money
    at scale on this ledger (approximately -1650 unit-returns across
    2631 admitted rows at floor -0.10, hit rate ~7-8%)."""
    all_slice = _get_slice(ledger_report, "ALL_SETTLED_PAIRS")
    baseline = all_slice.baseline_all_admitted
    assert baseline.admitted_count >= 2000
    assert baseline.total_return_per_unit < -500
    assert baseline.hit_rate is not None and 0.05 < baseline.hit_rate < 0.12


def test_real_ledger_margin_gate_reduces_exposure_but_stays_negative(ledger_report) -> None:
    """The honest finding: on this ledger the margin gate reduces
    the total loss (fewer bad bets are published) but no swept floor
    turns the pool profitable. That is exactly the behavior the coherent
    rule is entitled to claim -- 'less bad publication' -- and it is
    exactly what this backtest supports today. Anything stronger would
    require the additional deductions (market-disagreement, fragility,
    per-leg floors) the promotion-coherence proposal separately calls
    out."""
    all_slice = _get_slice(ledger_report, "ALL_SETTLED_PAIRS")
    baseline = all_slice.baseline_all_admitted

    # Some floor in the swept range reduces total loss below the
    # baseline's:
    assert any(
        (r.total_return_per_unit > baseline.total_return_per_unit)
        for r in all_slice.floor_sweep
    ), "no floor reduces loss vs baseline"

    # ... but none of them cross the zero line: this backtest does not
    # support the claim 'the margin gate makes the pool profitable'.
    assert all(
        r.total_return_per_unit <= 0.0 for r in all_slice.floor_sweep
    ), "some floor turned the ledger profitable -- re-pin this test"


def test_real_ledger_same_game_slice_admits_nothing_by_margin(ledger_report) -> None:
    """Honest finding: on this ledger every same-game pair has a
    predicted joint probability BELOW its break-even, so a positive-
    margin floor admits zero same-game pairs. This is a data-driven
    argument in favor of the proposal's shared-failure penalty on
    same-game parlays -- the margin rule alone abstains fully anyway."""
    sg = _get_slice(ledger_report, "SAME_GAME_PAIRS")
    # Every floor at or above 0.0 admits zero:
    for r in sg.floor_sweep:
        if r.floor >= 0.0:
            assert r.admitted_count == 0, f"same_game floor {r.floor} admitted {r.admitted_count} rows"
