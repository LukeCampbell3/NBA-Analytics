from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import pitcher_bullpen_model as pitching  # noqa: E402
import pitching_enriched_win_model as enriched  # noqa: E402
import team_win_model as base_model  # noqa: E402


def _team(avg_rs: float, avg_ra: float, games: int = 20) -> base_model.TeamCumulativeStats:
    return base_model.TeamCumulativeStats(team="A", games_played=games, runs_scored=avg_rs * games, runs_allowed=avg_ra * games)


def _starter(starts: int, era: float) -> pitching.PitcherCumulativeStats:
    # outs chosen so era works out cleanly: era = 9*ER/IP -> pick IP=6*starts, ER = era*IP/9
    ip = 6.0 * starts
    outs = int(round(ip * 3))
    earned_runs = int(round(era * ip / 9.0))
    return pitching.PitcherCumulativeStats(pitcher_id=1, name="X", starts=starts, outs=outs, earned_runs=earned_runs)


def _bullpen(era: float, games: int = 20) -> pitching.BullpenCumulativeStats:
    ip = 3.0 * games
    outs = int(round(ip * 3))
    earned_runs = int(round(era * ip / 9.0))
    return pitching.BullpenCumulativeStats(team="A", games=games, outs=outs, earned_runs=earned_runs)


def test_blended_expected_runs_allowed_falls_back_to_team_baseline_without_pitching_data() -> None:
    team = _team(5.0, 4.0)
    result = enriched.blended_expected_runs_allowed(team, None, None, 0.6)
    assert result == team.avg_runs_allowed


def test_blended_expected_runs_allowed_uses_pitching_signal_when_available() -> None:
    team = _team(5.0, 4.0)
    starter = _starter(15, era=2.0)  # full credibility, real strong starter
    bullpen = _bullpen(era=2.0)
    result = enriched.blended_expected_runs_allowed(team, starter, bullpen, 0.6)
    # Full starter credibility -> result should equal the pure pitching-implied figure (2.0), not the team baseline (4.0)
    assert result is not None
    assert abs(result - 2.0) < 0.05  # small rounding slack from the fixture's outs/earned_runs reconstruction


def test_blended_expected_runs_allowed_shrinks_low_sample_starter_toward_team_baseline() -> None:
    team = _team(5.0, 4.0)
    rookie_starter = _starter(1, era=1.0)  # tiny sample, extreme ERA -- must not dominate
    bullpen = _bullpen(era=4.0)
    result = enriched.blended_expected_runs_allowed(team, rookie_starter, bullpen, 0.6)
    assert result is not None
    assert result > 1.0  # nowhere near the rookie's raw (noisy) ERA
    assert result < team.avg_runs_allowed  # but still pulled somewhat toward the real pitching signal


def test_predict_moneyline_probability_enriched_returns_none_without_team_history() -> None:
    assert (
        enriched.predict_moneyline_probability_enriched(
            None, _team(5.0, 4.0),
            home_starter_stats=None, home_bullpen_stats=None,
            away_starter_stats=None, away_bullpen_stats=None,
            starter_innings_share=0.6,
        )
        is None
    )


def test_predict_moneyline_probability_enriched_favors_team_with_better_real_pitching() -> None:
    home = _team(4.5, 4.5)
    away = _team(4.5, 4.5)
    strong_starter = _starter(15, era=2.5)
    weak_starter = _starter(15, era=6.0)
    bullpen = _bullpen(era=4.0)
    prob = enriched.predict_moneyline_probability_enriched(
        home, away,
        home_starter_stats=strong_starter, home_bullpen_stats=bullpen,
        away_starter_stats=weak_starter, away_bullpen_stats=bullpen,
        starter_innings_share=0.6,
    )
    assert prob is not None
    assert prob > 0.5  # home has the real better starter, same offense/bullpen otherwise


def test_predict_run_total_enriched_returns_none_without_team_history() -> None:
    assert (
        enriched.predict_run_total_enriched(
            None, _team(5.0, 4.0),
            home_starter_stats=None, home_bullpen_stats=None,
            away_starter_stats=None, away_bullpen_stats=None,
            starter_innings_share=0.6,
        )
        is None
    )


def test_predict_run_total_enriched_is_positive_and_real() -> None:
    home = _team(5.0, 4.0)
    away = _team(4.0, 4.5)
    starter = _starter(15, era=3.5)
    bullpen = _bullpen(era=4.0)
    total = enriched.predict_run_total_enriched(
        home, away,
        home_starter_stats=starter, home_bullpen_stats=bullpen,
        away_starter_stats=starter, away_bullpen_stats=bullpen,
        starter_innings_share=0.6,
    )
    assert total is not None
    assert total > 0
