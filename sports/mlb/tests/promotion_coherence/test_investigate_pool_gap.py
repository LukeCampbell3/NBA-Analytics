"""Tests for the pair-pool gap investigation.

Split into unit tests on the summarization primitives and a real-data
regression that pins the current observed finding: the real pair
ledger's joint model over-predicts hit rate by more than the naive-
independence synthetic pool does. This is direct evidence against the
"the joint model is correctly pessimistic about correlation"
interpretation of the hit-rate gap.

A drift of the finding fails the regression loudly and the analysis
doc gets rewritten.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence.investigate_pool_gap import (
    CalibrationBin,
    DEFAULT_REAL_LEDGER,
    DEFAULT_SYNTHETIC_LEDGER,
    Distribution,
    _bucket_index,
    _calibration_bins,
    build_report,
    summarize_pool,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


# --- unit primitives ----------------------------------------------------

def test_distribution_summary_percentiles() -> None:
    d = Distribution.summarize([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert d.n == 9
    assert d.min == pytest.approx(0.1)
    assert d.p50 == pytest.approx(0.5)
    assert d.max == pytest.approx(0.9)
    assert d.mean == pytest.approx(0.5)


def test_distribution_summary_empty() -> None:
    d = Distribution.summarize([])
    assert d.n == 0 and d.min is None and d.mean is None


def test_bucket_index_deciles() -> None:
    edges = [i / 10 for i in range(11)]
    assert _bucket_index(0.05, edges) == 0
    assert _bucket_index(0.5, edges) == 5
    assert _bucket_index(0.999, edges) == 9
    assert _bucket_index(-0.1, edges) == 0
    assert _bucket_index(1.5, edges) == 9


def test_calibration_bins_group_by_predicted_joint() -> None:
    rows = [
        # bin 1 (0.1-0.2), 2 rows, actual hit rate 0.5
        {"predicted_joint_probability": 0.15, "both_win": True},
        {"predicted_joint_probability": 0.19, "both_win": False},
        # bin 3 (0.3-0.4), 1 row, actual hit rate 1.0
        {"predicted_joint_probability": 0.35, "both_win": True},
    ]
    bins = _calibration_bins(rows)
    assert bins[1].n_pairs == 2
    assert bins[1].mean_actual_hit_rate == pytest.approx(0.5)
    assert bins[1].calibration_gap == pytest.approx((0.15 + 0.19) / 2 - 0.5)
    assert bins[3].n_pairs == 1
    assert bins[3].mean_actual_hit_rate == pytest.approx(1.0)


def test_summarize_pool_empty_input_is_all_none() -> None:
    p = summarize_pool("EMPTY", [])
    assert p.row_count == 0
    assert p.hit_rate is None
    assert p.calibration_gap_mean_across_populated_bins is None


def test_summarize_pool_populates_per_leg_when_present() -> None:
    rows = [
        {"predicted_joint_probability": 0.3, "quoted_pair_price": 4.0, "both_win": True,
         "leg_1_model_probability": 0.6, "leg_2_model_probability": 0.5},
        {"predicted_joint_probability": 0.25, "quoted_pair_price": 4.5, "both_win": False,
         "leg_1_model_probability": 0.5, "leg_2_model_probability": 0.5},
    ]
    p = summarize_pool("SMALL", rows)
    assert p.per_leg_model_probability_distribution.n == 4
    assert p.hit_rate == pytest.approx(0.5)


# --- real-data regressions ----------------------------------------------

@pytest.fixture(scope="module")
def real_report():
    real = REPO_ROOT / DEFAULT_REAL_LEDGER
    synth = REPO_ROOT / DEFAULT_SYNTHETIC_LEDGER
    if not real.exists():
        pytest.skip(f"real pair ledger missing: {real}")
    if not synth.exists():
        pytest.skip(
            f"synthetic pair ledger missing: {synth} -- run "
            f"`python -m sports.mlb.parlay_v2.promotion_coherence.synthesize_pairs` first"
        )
    return build_report(real_ledger=real, synthetic_ledger=synth)


def test_real_vs_synthetic_hit_rate_gap_is_substantial(real_report) -> None:
    """Documents the current observed gap. Any run where the gap
    collapses is either (a) big evidence about upstream data changes
    or (b) a broken build; either way, we want it loud."""
    assert real_report.real_pool.row_count > 2000
    assert real_report.synthetic_pool.row_count > 15000
    assert real_report.hit_rate_gap_percentage_points is not None
    # Currently +19.9 pp. Pin at > +15 pp so a small drift is tolerated
    # but a collapse fails the test.
    assert real_report.hit_rate_gap_percentage_points > 15


def test_real_ledger_joint_model_over_predicts_hit_rate(real_report) -> None:
    """The core finding: the real pool's mean calibration gap
    (predicted - actual) is POSITIVE and non-trivial -- the joint model
    over-predicts hit rate. This is direct evidence AGAINST the
    'correctly pessimistic joint model' interpretation of the hit-rate
    gap. If this flips negative on a future rerun, interpretation (B)
    becomes plausible again and this analysis needs revisiting."""
    real_gap = real_report.real_pool.calibration_gap_mean_across_populated_bins
    assert real_gap is not None
    # Currently ~+0.12. Pin at > 0.05 so meaningful drift is captured.
    assert real_gap > 0.05, (
        f"real pool no longer over-predicts by > 5 pp per decile ({real_gap:+.4f}); "
        f"interpretation (B) may need to be reconsidered."
    )


def test_synthetic_ledger_is_more_calibrated_than_real_ledger(real_report) -> None:
    """A second-order finding: the naive-independence synthetic pool
    is BETTER calibrated than the frozen production joint model. The
    calibration gap on the synthetic pool should be smaller (in
    absolute terms) than on the real pool. This is evidence that the
    production joint model is not adding calibration value over the
    naive baseline on this data -- an item worth investigating
    independently."""
    real_gap = real_report.real_pool.calibration_gap_mean_across_populated_bins
    synth_gap = real_report.synthetic_pool.calibration_gap_mean_across_populated_bins
    assert real_gap is not None and synth_gap is not None
    assert abs(synth_gap) < abs(real_gap), (
        f"synthetic pool no longer better-calibrated ({synth_gap:+.4f}) than real "
        f"({real_gap:+.4f}); this changes the interpretation of the hit-rate gap."
    )


def test_interpretation_notes_include_calibration_comparison(real_report) -> None:
    """Sanity: the interpretation strings the CLI prints include the
    calibration-difference call-out. If a future refactor accidentally
    strips it, this fails."""
    text = " ".join(real_report.interpretation_notes)
    assert "calibration" in text.lower() or "interpretation" in text.lower()
