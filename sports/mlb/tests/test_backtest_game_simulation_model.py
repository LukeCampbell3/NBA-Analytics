from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import backtest_game_simulation_model as backtest  # noqa: E402


def _team_rows() -> list[dict]:
    rows = []
    teams = ["AAA", "BBB", "CCC", "DDD"]
    for i in range(60):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        home_score = 5 if i % 3 == 0 else 3
        away_score = 2
        home_f5 = min(home_score, 3)
        away_f5 = min(away_score, 1)
        rows.append(
            {
                "date": f"2026-04-{(i % 28) + 1:02d}",
                "game_id": f"g{i}",
                "home_team": home,
                "away_team": away,
                "home_score": str(home_score),
                "away_score": str(away_score),
                "home_innings_1_5": str(home_f5),
                "away_innings_1_5": str(away_f5),
                "market_home_moneyline": "-150",
                "market_away_moneyline": "130",
                "market_run_total": "7.5",
            }
        )
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_run_backtest_reports_insufficient_data_for_a_tiny_real_dataset() -> None:
    games = [
        {"date": "2026-04-01", "game_id": "g1", "home_team": "A", "away_team": "B", "home_score": 5.0, "away_score": 2.0, "market_home_moneyline": -150.0, "market_away_moneyline": 130.0, "market_run_total": 7.5, "total_runs": 7.0}
    ]
    report = backtest.run_backtest(games, [], holdout_fraction=0.3, num_trials=500)
    assert report["status"] == "insufficient_real_data"


def test_run_backtest_produces_real_calibration_metrics_for_every_market(tmp_path) -> None:
    team_rows = _team_rows()
    team_csv = tmp_path / "teams.csv"
    _write_csv(team_csv, team_rows)
    games = backtest.load_games(team_csv)
    report = backtest.run_backtest(games, [], holdout_fraction=0.3, num_trials=2000)

    assert report["status"] == "ok"
    assert report["train_games"] > 0
    assert report["holdout_games"] > 0
    assert report["runs_dispersion_ratio_used"] >= 1.0
    assert "sim_pick_accuracy" in report["moneyline"]
    assert "sim_over_pick_accuracy" in report["full_game_total"]
    assert "sim_over_pick_accuracy" in report["first_5_innings_total"]


def test_run_backtest_f5_report_notes_no_real_historical_market_price(tmp_path) -> None:
    team_rows = _team_rows()
    team_csv = tmp_path / "teams.csv"
    _write_csv(team_csv, team_rows)
    games = backtest.load_games(team_csv)
    report = backtest.run_backtest(games, [], holdout_fraction=0.3, num_trials=1000)
    assert "no real historical F5 market price" in report["first_5_innings_total"]["note"]
