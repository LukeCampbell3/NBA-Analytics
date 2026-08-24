from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import backtest_team_win_model as backtest  # noqa: E402


def test_american_to_probability_favorite_and_underdog() -> None:
    assert abs(backtest.american_to_probability(-200) - (200 / 300)) < 1e-9
    assert abs(backtest.american_to_probability(150) - (100 / 250)) < 1e-9
    assert backtest.american_to_probability(None) is None


def test_no_vig_home_probability_normalizes_real_vig() -> None:
    prob = backtest.no_vig_home_probability(-150, 130)
    assert prob is not None
    assert 0.0 < prob < 1.0
    raw_home = backtest.american_to_probability(-150)
    raw_away = backtest.american_to_probability(130)
    assert abs(prob - raw_home / (raw_home + raw_away)) < 1e-9


def test_no_vig_home_probability_none_when_either_side_missing() -> None:
    assert backtest.no_vig_home_probability(-150, None) is None


def test_brier_score_zero_for_perfect_predictions() -> None:
    assert backtest.brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_score_positive_for_imperfect_predictions() -> None:
    assert backtest.brier_score([0.5, 0.5], [1, 0]) == 0.25


def _rows() -> list[dict]:
    rows = []
    teams = ["AAA", "BBB", "CCC", "DDD"]
    # 40 real-shaped games so both teams clear MIN_GAMES_PLAYED_FOR_PREDICTION
    # by the time the holdout split begins.
    for i in range(40):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        home_score = 5 if i % 3 == 0 else 3
        away_score = 2
        rows.append(
            {
                "date": f"2026-04-{(i % 28) + 1:02d}",
                "game_id": f"g{i}",
                "home_team": home,
                "away_team": away,
                "home_score": str(home_score),
                "away_score": str(away_score),
                "market_home_moneyline": "-150",
                "market_away_moneyline": "130",
                "market_run_total": "7.5",
            }
        )
    return rows


def test_load_games_reads_real_rows_and_sorts_chronologically(tmp_path) -> None:
    csv_path = tmp_path / "history.csv"
    rows = _rows()
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    games = backtest.load_games(csv_path)
    assert len(games) == 40
    dates = [g["date"] for g in games]
    assert dates == sorted(dates)


def test_run_backtest_reports_insufficient_data_for_a_tiny_real_dataset() -> None:
    games = [
        {"date": "2026-04-01", "game_id": "g1", "home_team": "A", "away_team": "B", "home_score": 5.0, "away_score": 2.0, "market_home_moneyline": -150.0, "market_away_moneyline": 130.0, "market_run_total": 7.5, "total_runs": 7.0}
    ]
    report = backtest.run_backtest(games, holdout_fraction=0.3)
    assert report["status"] == "insufficient_real_data"


def test_run_backtest_produces_real_moneyline_and_total_metrics(tmp_path) -> None:
    csv_path = tmp_path / "history.csv"
    rows = _rows()
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    games = backtest.load_games(csv_path)
    report = backtest.run_backtest(games, holdout_fraction=0.3)
    assert report["status"] == "ok"
    assert report["train_games"] > 0
    assert report["holdout_games"] > 0
    assert "moneyline" in report
    assert "run_total" in report


def test_run_backtest_never_uses_holdout_games_to_derive_home_field_advantage(monkeypatch) -> None:
    """Real regression guard: home_field_advantage must be computed ONLY
    from the train split, never the holdout it's applied to."""
    captured = {}
    original = backtest.model.compute_empirical_home_field_advantage

    def _spy(games):
        captured["games"] = games
        return original(games)

    monkeypatch.setattr(backtest.model, "compute_empirical_home_field_advantage", _spy)
    games = []
    for i in range(40):
        games.append(
            {
                "date": f"2026-04-{(i % 28) + 1:02d}", "game_id": f"g{i}",
                "home_team": "A" if i % 2 == 0 else "B", "away_team": "B" if i % 2 == 0 else "A",
                "home_score": 5.0, "away_score": 2.0,
                "market_home_moneyline": -150.0, "market_away_moneyline": 130.0, "market_run_total": 7.5,
                "total_runs": 7.0,
            }
        )
    backtest.run_backtest(games, holdout_fraction=0.3)
    assert len(captured["games"]) == int(len(games) * 0.7)
