from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import team_win_model as model  # noqa: E402


def test_pythagorean_win_pct_favors_team_with_better_run_differential() -> None:
    strong = model.TeamCumulativeStats(team="A", games_played=20, runs_scored=120.0, runs_allowed=80.0)
    weak = model.TeamCumulativeStats(team="B", games_played=20, runs_scored=80.0, runs_allowed=120.0)
    assert strong.pythagorean_win_pct > 0.5
    assert weak.pythagorean_win_pct < 0.5
    assert strong.pythagorean_win_pct > weak.pythagorean_win_pct


def test_pythagorean_win_pct_is_half_with_no_real_runs_yet() -> None:
    fresh = model.TeamCumulativeStats(team="A", games_played=0, runs_scored=0.0, runs_allowed=0.0)
    assert fresh.pythagorean_win_pct == 0.5


def test_log5_probability_is_half_for_two_evenly_matched_teams() -> None:
    assert abs(model.log5_probability(0.5, 0.5) - 0.5) < 1e-9


def test_log5_probability_favors_the_better_team() -> None:
    assert model.log5_probability(0.6, 0.4) > 0.5
    assert model.log5_probability(0.4, 0.6) < 0.5


def _game(date: str, home: str, away: str, home_score: float, away_score: float, game_id: str = "") -> dict:
    return {"date": date, "home_team": home, "away_team": away, "home_score": home_score, "away_score": away_score, "game_id": game_id or f"{date}_{home}_{away}"}


def test_build_cumulative_team_stats_is_leakage_safe() -> None:
    """The stats snapshot available BEFORE a team's 3rd game must never
    include that 3rd game's own real runs."""
    games = [
        _game("2026-04-01", "A", "B", 5.0, 2.0),
        _game("2026-04-02", "A", "C", 3.0, 4.0),
        _game("2026-04-03", "A", "D", 10.0, 1.0),  # this game's own runs must not leak into its own prior snapshot
    ]
    history = model.build_cumulative_team_stats(games)
    stats_before_game3 = model.stats_as_of(history["A"], "2026-04-03")
    assert stats_before_game3.games_played == 2
    assert stats_before_game3.runs_scored == 8.0  # 5 + 3, NOT +10 from the game being predicted
    assert stats_before_game3.runs_allowed == 6.0  # 2 + 4


def test_stats_as_of_returns_none_before_any_real_history() -> None:
    games = [_game("2026-04-05", "A", "B", 5.0, 2.0)]
    history = model.build_cumulative_team_stats(games)
    assert model.stats_as_of(history["A"], "2026-04-01") is None  # before this team's first real game


def test_predict_moneyline_probability_returns_none_without_real_history() -> None:
    assert model.predict_moneyline_probability(None, model.TeamCumulativeStats("B", 5, 20.0, 15.0)) is None


def test_predict_moneyline_probability_favors_the_better_real_team() -> None:
    strong = model.TeamCumulativeStats(team="A", games_played=20, runs_scored=120.0, runs_allowed=80.0)
    weak = model.TeamCumulativeStats(team="B", games_played=20, runs_scored=80.0, runs_allowed=120.0)
    prob = model.predict_moneyline_probability(strong, weak)
    assert prob is not None
    assert prob > 0.5


def test_predict_moneyline_probability_applies_real_home_field_adjustment() -> None:
    even = model.TeamCumulativeStats(team="A", games_played=20, runs_scored=100.0, runs_allowed=100.0)
    other = model.TeamCumulativeStats(team="B", games_played=20, runs_scored=100.0, runs_allowed=100.0)
    no_adjustment = model.predict_moneyline_probability(even, other, home_field_advantage=0.0)
    with_adjustment = model.predict_moneyline_probability(even, other, home_field_advantage=0.04)
    assert with_adjustment > no_adjustment


def test_predict_run_total_returns_none_without_real_history() -> None:
    assert model.predict_run_total(None, model.TeamCumulativeStats("B", 5, 20.0, 15.0)) is None


def test_predict_run_total_is_real_and_positive_for_real_teams() -> None:
    home = model.TeamCumulativeStats(team="A", games_played=20, runs_scored=100.0, runs_allowed=90.0)
    away = model.TeamCumulativeStats(team="B", games_played=20, runs_scored=80.0, runs_allowed=95.0)
    total = model.predict_run_total(home, away)
    assert total is not None
    assert total > 0


def test_compute_empirical_home_field_advantage_from_real_dataset() -> None:
    games = [_game(f"2026-04-{i:02d}", "A", "B", 5.0, 2.0) for i in range(1, 12)]  # 11 real home wins
    advantage = model.compute_empirical_home_field_advantage(games)
    assert advantage == 0.5  # 100% home win rate in this fixture -> +0.5 over .500


def test_compute_empirical_home_field_advantage_needs_enough_real_games() -> None:
    games = [_game("2026-04-01", "A", "B", 5.0, 2.0)]  # only 1 real game -- not enough to trust
    assert model.compute_empirical_home_field_advantage(games) == 0.0
