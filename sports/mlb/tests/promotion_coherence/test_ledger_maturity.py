"""Tests for the ledger-maturity monitor.

Unit tests on tmp-file fixtures + a regression on the real ledger
that pins the current maturity so a future extension is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence.ledger_maturity import (
    DEFAULT_LEDGER,
    DEFAULT_MIN_SLATES_FOR_DECISION_QUALITY,
    compute_maturity,
    maturity_message,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_ledger(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pair_observation_ledger.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_missing_ledger_reports_empty_maturity(tmp_path: Path) -> None:
    m = compute_maturity(tmp_path / "no_such.jsonl")
    assert m.row_count == 0
    assert m.settled_row_count == 0
    assert m.slates_covered == []
    assert m.decision_quality_ready is False
    assert m.slates_short_of_decision_quality == DEFAULT_MIN_SLATES_FOR_DECISION_QUALITY


def test_settled_rows_are_counted_slate_wise(tmp_path: Path) -> None:
    rows = [
        {"slate_id": "20260901", "settlement_status": "settled", "same_game": False},
        {"slate_id": "20260901", "settlement_status": "settled", "same_game": True},
        {"slate_id": "20260902", "settlement_status": "settled", "same_game": False},
        {"slate_id": "20260903", "settlement_status": "ungraded"},  # ignored
    ]
    path = _write_ledger(tmp_path, rows)
    m = compute_maturity(path)
    assert m.settled_row_count == 3
    assert m.slates_covered == ["20260901", "20260902"]
    assert m.first_slate == "20260901"
    assert m.last_slate == "20260902"
    assert m.same_game_row_count == 1
    assert m.cross_game_row_count == 2


def test_decision_quality_ready_when_slate_count_meets_threshold(tmp_path: Path) -> None:
    # Exactly 10 slates
    rows = [
        {"slate_id": f"2026090{i}", "settlement_status": "settled"}
        for i in range(10)
    ]
    path = _write_ledger(tmp_path, rows)
    m = compute_maturity(path, min_slates=10)
    assert m.decision_quality_ready is True
    assert m.slates_short_of_decision_quality == 0


def test_decision_quality_not_ready_when_slate_count_below_threshold(tmp_path: Path) -> None:
    rows = [
        {"slate_id": f"2026090{i}", "settlement_status": "settled"}
        for i in range(4)
    ]
    path = _write_ledger(tmp_path, rows)
    m = compute_maturity(path, min_slates=10)
    assert m.decision_quality_ready is False
    assert m.slates_short_of_decision_quality == 6


def test_custom_threshold_overrides_default(tmp_path: Path) -> None:
    rows = [{"slate_id": "s1", "settlement_status": "settled"}]
    path = _write_ledger(tmp_path, rows)
    m = compute_maturity(path, min_slates=1)
    assert m.decision_quality_ready is True


def test_no_vig_capture_counts_populated_and_missing_correctly(tmp_path: Path) -> None:
    rows = [
        # v1 row: no capture
        {"slate_id": "s1", "settlement_status": "settled"},
        # partial capture: leg 1 only
        {"slate_id": "s1", "settlement_status": "settled",
         "leg_1_no_vig_market_probability": 0.55},
        # full capture
        {"slate_id": "s1", "settlement_status": "settled",
         "leg_1_no_vig_market_probability": 0.55,
         "leg_2_no_vig_market_probability": 0.42},
    ]
    path = _write_ledger(tmp_path, rows)
    m = compute_maturity(path)
    assert m.rows_with_leg_1_no_vig == 2
    assert m.rows_with_both_leg_no_vig == 1


def test_maturity_message_is_multi_line_and_actionable() -> None:
    from sports.mlb.parlay_v2.promotion_coherence.ledger_maturity import LedgerMaturity
    m = LedgerMaturity(
        ledger_path="/tmp/x.jsonl", row_count=5, settled_row_count=5,
        slates_covered=["s1", "s2"], first_slate="s1", last_slate="s2",
        same_game_row_count=1, cross_game_row_count=4,
        rows_with_leg_1_no_vig=0, rows_with_both_leg_no_vig=0,
        min_slates_for_decision_quality=10,
        slates_short_of_decision_quality=8,
        decision_quality_ready=False,
        generated_at_utc="2026-09-03T00:00:00Z",
    )
    msg = maturity_message(m)
    assert "settled rows" in msg
    assert "8 more slates" in msg
    assert "target >= 10" in msg


# --- real-ledger regression ---------------------------------------------

def test_real_ledger_current_maturity_state() -> None:
    """Pins today's state of the real ledger. As new prospective slates
    ingest, `slates_covered` extends and this test's assertions get
    revised upward -- growth is welcome and visible."""
    path = REPO_ROOT / DEFAULT_LEDGER
    if not path.exists():
        pytest.skip(f"real ledger missing: {path}")
    m = compute_maturity(path)
    # Currently 4 slates -- pin at >= 3 to allow a small growth window
    # before the "target 10" gate begins to matter.
    assert len(m.slates_covered) >= 3
    assert len(m.slates_covered) < DEFAULT_MIN_SLATES_FOR_DECISION_QUALITY, (
        "real ledger has caught up to decision-quality threshold -- rerun the pair-"
        "ledger backtest and update BACKTEST_ANALYSIS.md with the real-slice numbers."
    )
    assert m.decision_quality_ready is False
    # Rows-with-no-vig should be zero today; when pair-ingest v1.1
    # captures start populating prospective slates, this rises.
    assert m.rows_with_both_leg_no_vig == 0, (
        "real ledger now carries per-leg no-vig capture -- the market-disagreement "
        "deduction can be a first-class production signal; rerun the pool-gap "
        "investigation to see the per-leg calibration axis"
    )
