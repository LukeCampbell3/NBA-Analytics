"""Tests for the discriminative-power / AUC diagnostics.

The critical unit test: AUC math on hand-built pairs, incl.
degenerate cases (empty, all positive, all negative, all equal
scores). Then a real-ledger regression that pins the current
observed AUC-by-slice picture, so a future update to the ledger
that flips a slice from inverted to signal-carrying (or vice versa)
fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence.discriminative_power import (
    DEFAULT_LEDGER,
    _classify,
    auc,
    build_report,
    slice_pool,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER = REPO_ROOT / DEFAULT_LEDGER


# --- unit ----------------------------------------------------------------

def test_auc_perfect_ranker() -> None:
    # every positive scores above every negative
    pairs = [(0.1, 0), (0.2, 0), (0.7, 1), (0.9, 1)]
    assert auc(pairs) == pytest.approx(1.0)


def test_auc_perfectly_inverted() -> None:
    pairs = [(0.9, 0), (0.7, 0), (0.2, 1), (0.1, 1)]
    assert auc(pairs) == pytest.approx(0.0)


def test_auc_random_hits_half() -> None:
    # ties → mid-rank; expected AUC 0.5
    pairs = [(0.5, 0), (0.5, 1), (0.5, 0), (0.5, 1)]
    assert auc(pairs) == pytest.approx(0.5)


def test_auc_none_on_degenerate_input() -> None:
    assert auc([]) is None
    assert auc([(0.5, 1)]) is None
    assert auc([(0.1, 0), (0.2, 0)]) is None  # only negatives
    assert auc([(0.1, 1), (0.2, 1)]) is None  # only positives


def test_classify_requires_min_100_rows_for_flag() -> None:
    assert _classify(50, 0.90) == (False, False)
    assert _classify(500, 0.60) == (True, False)   # positive signal
    assert _classify(500, 0.30) == (False, True)   # inverted
    assert _classify(500, 0.50) == (False, False)  # flat
    assert _classify(500, None) == (False, False)


def test_slice_pool_groups_and_reports_correctly() -> None:
    rows = [
        {"predicted_joint_probability": 0.3, "both_win": True,  "market_pair_type": "A"},
        {"predicted_joint_probability": 0.2, "both_win": False, "market_pair_type": "A"},
        {"predicted_joint_probability": 0.4, "both_win": True,  "market_pair_type": "B"},
    ]
    slices = slice_pool(rows, key_fn=lambda r: r.get("market_pair_type"), key_label="mkt")
    assert {s.slice_key for s in slices} == {"mkt=A", "mkt=B"}
    a = next(s for s in slices if s.slice_key == "mkt=A")
    assert a.n == 2 and a.n_positive == 1 and a.hit_rate == pytest.approx(0.5)


# --- real-ledger regression --------------------------------------------

@pytest.fixture(scope="module")
def real_dp_report():
    if not LEDGER.exists():
        pytest.skip(f"real ledger missing: {LEDGER}")
    return build_report(ledger_path=LEDGER)


def test_global_auc_is_currently_inverted(real_dp_report) -> None:
    """Pinning the honest observed value: global AUC is well below 0.5
    on the current ledger. If this flips above 0.5, the underlying
    joint model has learned real signal since the last update -- the
    slice calibrator + BACKTEST_ANALYSIS.md story get a rewrite and
    this test needs a new pin."""
    g = real_dp_report.global_slice
    assert g.auc is not None
    assert g.auc < 0.47, (
        f"global AUC {g.auc:.4f} is no longer inverted -- promote this "
        f"to a signal-carrying pin and update the analysis doc."
    )


def test_R_R_market_has_flat_or_positive_signal(real_dp_report) -> None:
    """Pin the current observation: R|R (runs paired with runs) is
    the ONLY market whose AUC is at least 0.50. If TB-heavy markets
    ever exceed R|R, the slice-conditioned calibrator's default slice
    key should be re-considered."""
    rr = next(s for s in real_dp_report.by_market_pair_type if s.slice_key.endswith("=R|R"))
    assert rr.auc is not None
    assert rr.auc >= 0.50, (
        f"R|R AUC is {rr.auc:.4f}; the assumption that R|R is the least-"
        f"inverted market no longer holds."
    )


def test_TB_TB_market_is_strongly_inverted(real_dp_report) -> None:
    """Pin the honest observation: TB|TB is the most inverted slice."""
    tbtb = next(s for s in real_dp_report.by_market_pair_type if s.slice_key.endswith("=TB|TB"))
    assert tbtb.auc is not None
    assert tbtb.auc < 0.35, (
        f"TB|TB AUC is {tbtb.auc:.4f}; not as strongly inverted as pinned."
    )


def test_no_market_slice_exceeds_positive_signal_threshold_yet(real_dp_report) -> None:
    """As of this branch, NO market pair type at n>=100 has AUC > 0.53
    (the pinned positive-signal threshold in _classify). When the
    underlying model or ledger catches up, this pin flips first."""
    for s in real_dp_report.by_market_pair_type:
        if s.n >= 100:
            assert not s.positive_signal_flag, (
                f"{s.slice_key} now carries positive signal (AUC {s.auc}); "
                f"a slice has become trustworthy for the promotion gate."
            )
