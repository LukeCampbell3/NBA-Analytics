from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import pitcher_bullpen_model as model  # noqa: E402


def _start(date: str, pitcher_id: int, name: str, outs: int, earned_runs: int) -> dict:
    return {"date": date, "pitcher_id": pitcher_id, "name": name, "outs": outs, "earned_runs": earned_runs}


def test_pitcher_cumulative_stats_era_and_innings() -> None:
    stats = model.PitcherCumulativeStats(pitcher_id=1, name="X", starts=3, outs=54, earned_runs=9)
    assert stats.innings_pitched == 18.0
    assert abs(stats.era - 4.5) < 1e-9  # 9 * 9 / 18 = 4.5


def test_pitcher_cumulative_stats_era_none_with_no_real_outs() -> None:
    stats = model.PitcherCumulativeStats(pitcher_id=1, name="X", starts=0, outs=0, earned_runs=0)
    assert stats.era is None


def test_pitcher_credibility_weight_ramps_from_zero_to_one() -> None:
    fresh = model.PitcherCumulativeStats(pitcher_id=1, name="X", starts=0, outs=0, earned_runs=0)
    partial = model.PitcherCumulativeStats(pitcher_id=1, name="X", starts=5, outs=90, earned_runs=20)
    full = model.PitcherCumulativeStats(pitcher_id=1, name="X", starts=15, outs=270, earned_runs=60)
    overfull = model.PitcherCumulativeStats(pitcher_id=1, name="X", starts=30, outs=540, earned_runs=120)
    assert fresh.credibility_weight == 0.0
    assert 0.0 < partial.credibility_weight < 1.0
    assert full.credibility_weight == 1.0
    assert overfull.credibility_weight == 1.0  # never exceeds full real credibility


def test_build_cumulative_pitcher_stats_is_leakage_safe() -> None:
    """The real stats snapshot available BEFORE a pitcher's 3rd start
    must never include that 3rd start's own real outs/earned runs."""
    starts = [
        _start("2026-04-01", 1, "Ace", 18, 2),
        _start("2026-04-06", 1, "Ace", 15, 3),
        _start("2026-04-11", 1, "Ace", 21, 1),  # must not leak into its own prior snapshot
    ]
    history = model.build_cumulative_pitcher_stats(starts)
    before_third = model.stats_as_of(history[1], "2026-04-11")
    assert before_third.starts == 2
    assert before_third.outs == 33  # 18 + 15, NOT +21 from the start being predicted
    assert before_third.earned_runs == 5


def test_stats_as_of_returns_none_before_any_real_start() -> None:
    starts = [_start("2026-04-05", 1, "Ace", 18, 2)]
    history = model.build_cumulative_pitcher_stats(starts)
    assert model.stats_as_of(history[1], "2026-04-01") is None


def _bullpen_appearance(date: str, team: str, outs: int, earned_runs: int) -> dict:
    return {"date": date, "team": team, "outs": outs, "earned_runs": earned_runs}


def test_build_cumulative_bullpen_stats_aggregates_per_team_and_is_leakage_safe() -> None:
    appearances = [
        _bullpen_appearance("2026-04-01", "ATL", 9, 1),
        _bullpen_appearance("2026-04-02", "ATL", 12, 4),
        _bullpen_appearance("2026-04-03", "ATL", 6, 0),  # must not leak into its own prior snapshot
    ]
    history = model.build_cumulative_bullpen_stats(appearances)
    before_third = model.stats_as_of(history["ATL"], "2026-04-03")
    assert before_third.games == 2
    assert before_third.outs == 21  # 9 + 12, NOT +6
    assert before_third.earned_runs == 5


def test_compute_empirical_starter_innings_share_from_real_totals() -> None:
    rows = [
        {"starter_outs": 18, "bullpen_outs": 9},  # 2/3 starter share
        {"starter_outs": 15, "bullpen_outs": 12},
    ]
    share = model.compute_empirical_starter_innings_share(rows)
    assert share is not None
    total_starter = 18 + 15
    total = 18 + 9 + 15 + 12
    assert abs(share - total_starter / total) < 1e-9


def test_compute_empirical_starter_innings_share_none_with_no_real_data() -> None:
    assert model.compute_empirical_starter_innings_share([]) is None
