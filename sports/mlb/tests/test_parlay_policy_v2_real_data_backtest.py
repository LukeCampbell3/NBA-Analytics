"""Regression check for parlay_policy_v2's real-data backtest against MLB.

Ported from sports/nba/tests/test_parlay_policy_v2_real_data_backtest.py's
MLB half -- this exercises the module against real, committed, settled data
(mlb_walk_forward_backtest_rows.csv) rather than synthetic fixtures. See
sports/mlb/research/parlay_policy_v2/REPORT.md for what this does and does
not establish.
"""
from __future__ import annotations

from sports.mlb.research.parlay_policy_v2.real_data_backtest import main as run_mlb_backtest


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
