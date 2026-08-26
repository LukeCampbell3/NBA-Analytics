"""Regression check for the real-data backtest scripts in parlay_policy_v2.

These exercise the module against real, committed, settled data (MLB leg-level
backtest rows; NBA production history snapshots) rather than synthetic
fixtures. See REPORT.md for what this does and does not establish.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.parlay_policy_v2.real_data_backtest_mlb import main as run_mlb_backtest
from research.parlay_policy_v2.real_data_summary_nba import main as run_nba_summary
from research.parlay_policy_v2.shadow_annotate_board import main as run_nba_shadow_annotate

NBA_HISTORY_DIR = REPO_ROOT / "sports" / "nba" / "web" / "data" / "history"


def test_mlb_real_data_backtest_runs_and_beats_baseline() -> None:
    report = run_mlb_backtest()

    assert report["real_settled_legs"] > 300  # this is the full published_real_market real leg count
    control = report["current_mlb_strategy_control"]
    assert control["available"] is True

    pool = report["full_real_eligible_pair_pool"]
    gated = report["new_policy_v2_gate"]

    # new-policy-selected subset must be strictly smaller than the full real
    # eligible pool (the gate actually filters something) ...
    assert 0 < gated["eligible"] < pool["n"]
    # ... and its real, settled hit rate must exceed the ungated pool's,
    # i.e. the gate is not selecting at random.
    assert gated["hit_rate"] > pool["hit_rate"]
    # the Wilson lower bound of the gated subset must clear the full pool's
    # raw hit rate -- a real, not just lucky-sample, improvement.
    assert gated["wilson95"][0] > pool["hit_rate"]


def test_nba_summary_reflects_small_real_sample() -> None:
    summary = run_nba_summary()
    assert summary["snapshots"], "expected at least one real NBA parlay_validation snapshot"
    for snap in summary["snapshots"]:
        # every currently-committed NBA snapshot is a small sample -- this
        # assertion exists to catch a *change*: if a future snapshot carries
        # a materially larger graded sample, this test should be revisited
        # (and REPORT.md's "no NBA hit-rate claim yet" framing reconsidered).
        assert snap["selected_graded"] <= 10


def test_nba_shadow_annotate_runs_on_a_real_board_and_produces_no_settled_field() -> None:
    board_path = NBA_HISTORY_DIR / "2026-05-26.json"
    assert board_path.exists(), "expected the committed real NBA board snapshot to still exist"

    result = run_nba_shadow_annotate(board_path)

    assert result["settled"] is False
    assert result["real_plays_on_board"] > 0
    for candidate in result["candidates"]:
        assert "won" not in candidate  # unsettled -- must never be fabricated
        assert candidate["leg_count"] == 2
        assert 0.0 <= candidate["p_joint_usable"] <= 1.0
        assert isinstance(candidate["eligible"], bool)
        # every rejection must be traceable to a real reason code
        if not candidate["eligible"]:
            assert candidate["reasons"]
