from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import backtest_pitching_enriched_win_model as backtest  # noqa: E402


def _team_rows() -> list[dict]:
    rows = []
    teams = ["AAA", "BBB", "CCC", "DDD"]
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


def _pitcher_rows_for(team_rows: list[dict]) -> list[dict]:
    rows = []
    for i, g in enumerate(team_rows):
        rows.append(
            {
                "date": g["date"], "home_team": g["home_team"], "away_team": g["away_team"],
                "home_starter_id": str(100 + (i % 4)), "home_starter_name": "Home Starter",
                "home_starter_outs": "18", "home_starter_earned_runs": "2",
                "home_bullpen_outs": "9", "home_bullpen_earned_runs": "1",
                "away_starter_id": str(200 + (i % 4)), "away_starter_name": "Away Starter",
                "away_starter_outs": "15", "away_starter_earned_runs": "3",
                "away_bullpen_outs": "12", "away_bullpen_earned_runs": "2",
            }
        )
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_pitcher_rows_reads_real_rows(tmp_path) -> None:
    team_rows = _team_rows()
    pitcher_rows = _pitcher_rows_for(team_rows)
    csv_path = tmp_path / "pitchers.csv"
    _write_csv(csv_path, pitcher_rows)
    loaded = backtest.load_pitcher_rows(csv_path)
    assert len(loaded) == len(pitcher_rows)
    assert loaded[0]["home_starter_outs"] == 18


def test_flatten_starts_and_bullpen_produces_two_entries_per_side() -> None:
    team_rows = _team_rows()[:2]
    pitcher_rows = _pitcher_rows_for(team_rows)
    parsed = [
        {**row, "home_starter_id": int(row["home_starter_id"]), "home_starter_outs": int(row["home_starter_outs"]),
         "home_starter_earned_runs": int(row["home_starter_earned_runs"]), "home_bullpen_outs": int(row["home_bullpen_outs"]),
         "home_bullpen_earned_runs": int(row["home_bullpen_earned_runs"]), "away_starter_id": int(row["away_starter_id"]),
         "away_starter_outs": int(row["away_starter_outs"]), "away_starter_earned_runs": int(row["away_starter_earned_runs"]),
         "away_bullpen_outs": int(row["away_bullpen_outs"]), "away_bullpen_earned_runs": int(row["away_bullpen_earned_runs"])}
        for row in pitcher_rows
    ]
    starts, bullpen = backtest.flatten_starts_and_bullpen(parsed)
    assert len(starts) == 2 * len(parsed)
    assert len(bullpen) == 2 * len(parsed)


def test_run_backtest_reports_insufficient_data_for_a_tiny_real_dataset() -> None:
    games = [
        {"date": "2026-04-01", "game_id": "g1", "home_team": "A", "away_team": "B", "home_score": 5.0, "away_score": 2.0, "market_home_moneyline": -150.0, "market_away_moneyline": 130.0, "market_run_total": 7.5, "total_runs": 7.0}
    ]
    report = backtest.run_backtest(games, [], holdout_fraction=0.3)
    assert report["status"] == "insufficient_real_data"


def test_run_backtest_produces_real_enriched_and_base_metrics(tmp_path) -> None:
    team_rows = _team_rows()
    pitcher_rows = _pitcher_rows_for(team_rows)
    team_csv = tmp_path / "teams.csv"
    pitcher_csv = tmp_path / "pitchers.csv"
    _write_csv(team_csv, team_rows)
    _write_csv(pitcher_csv, pitcher_rows)

    games = backtest.load_games(team_csv)
    loaded_pitcher_rows = backtest.load_pitcher_rows(pitcher_csv)
    report = backtest.run_backtest(games, loaded_pitcher_rows, holdout_fraction=0.3)

    assert report["status"] == "ok"
    assert report["train_games"] > 0
    assert report["holdout_games"] > 0
    assert report["starter_innings_share_used"] is not None
    assert 0.0 < report["starter_innings_share_used"] < 1.0
    assert "base_pick_accuracy" in report["moneyline"]
    assert "enriched_pick_accuracy" in report["moneyline"]
    assert "base_mean_absolute_error" in report["run_total"]
    assert "enriched_mean_absolute_error" in report["run_total"]
    assert report["moneyline"]["games_with_real_pitching_data"] == report["moneyline"]["graded_games"]


def test_run_backtest_degrades_to_base_model_when_no_real_pitching_data_matches(tmp_path) -> None:
    """A holdout game with no real pitcher-data match must fall back to
    an enriched prediction identical to the base model's, never crash or
    silently drop the game."""
    team_rows = _team_rows()
    team_csv = tmp_path / "teams.csv"
    _write_csv(team_csv, team_rows)
    games = backtest.load_games(team_csv)
    report = backtest.run_backtest(games, [], holdout_fraction=0.3)  # no real pitcher data at all
    assert report["status"] == "ok"
    assert report["moneyline"]["games_with_real_pitching_data"] == 0
    assert abs(report["moneyline"]["base_brier_score"] - report["moneyline"]["enriched_brier_score"]) < 1e-9
