from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "nfl" / "scripts"))

import fetch_nfl_team_game_history as fetcher  # noqa: E402


def _fixture_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Real, completed REG game with a real closing total_line.
            {
                "season": 2024, "week": 1, "game_type": "REG", "game_id": "2024_01_A_B",
                "gameday": "2024-09-08", "home_team": "B", "away_team": "A",
                "home_score": 24.0, "away_score": 20.0,
                "spread_line": -3.0, "home_spread_odds": -110.0, "away_spread_odds": -110.0,
                "total_line": 45.5, "over_odds": -108.0, "under_odds": -112.0,
                "home_moneyline": -150.0, "away_moneyline": 130.0,
            },
            # Playoff game -- excluded, this dataset is REG only.
            {
                "season": 2024, "week": 19, "game_type": "WC", "game_id": "2024_19_C_D",
                "gameday": "2025-01-12", "home_team": "D", "away_team": "C",
                "home_score": 30.0, "away_score": 10.0,
                "spread_line": -6.0, "home_spread_odds": -110.0, "away_spread_odds": -110.0,
                "total_line": 44.5, "over_odds": -105.0, "under_odds": -115.0,
                "home_moneyline": -250.0, "away_moneyline": 210.0,
            },
            # Not yet played (no real score) -- must never be included with a fabricated result.
            {
                "season": 2026, "week": 1, "game_type": "REG", "game_id": "2026_01_E_F",
                "gameday": "2026-09-10", "home_team": "F", "away_team": "E",
                "home_score": None, "away_score": None,
                "spread_line": -2.5, "home_spread_odds": -110.0, "away_spread_odds": -110.0,
                "total_line": 47.0, "over_odds": -110.0, "under_odds": -110.0,
                "home_moneyline": -135.0, "away_moneyline": 115.0,
            },
            # Real game, real score, but no real closing total_line recorded -- must be excluded.
            {
                "season": 2024, "week": 2, "game_type": "REG", "game_id": "2024_02_G_H",
                "gameday": "2024-09-15", "home_team": "H", "away_team": "G",
                "home_score": 17.0, "away_score": 14.0,
                "spread_line": None, "home_spread_odds": None, "away_spread_odds": None,
                "total_line": None, "over_odds": None, "under_odds": None,
                "home_moneyline": None, "away_moneyline": None,
            },
        ]
    )


def test_fetch_real_team_game_history_keeps_only_real_completed_reg_games_with_real_lines(monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_parquet", lambda url: _fixture_schedule())
    result = fetcher.fetch_real_team_game_history()
    assert list(result["game_id"]) == ["2024_01_A_B"]


def test_fetch_real_team_game_history_computes_real_home_won_and_total_points(monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_parquet", lambda url: _fixture_schedule())
    result = fetcher.fetch_real_team_game_history()
    row = result.iloc[0]
    assert row["home_won"] == 1  # 24 > 20
    assert row["total_points"] == 44.0


def test_fetch_real_team_game_history_respects_min_season(monkeypatch) -> None:
    schedule = pd.concat(
        [
            _fixture_schedule(),
            pd.DataFrame(
                [
                    {
                        "season": 2018, "week": 1, "game_type": "REG", "game_id": "2018_01_I_J",
                        "gameday": "2018-09-09", "home_team": "J", "away_team": "I",
                        "home_score": 21.0, "away_score": 17.0,
                        "spread_line": -3.0, "home_spread_odds": -110.0, "away_spread_odds": -110.0,
                        "total_line": 42.5, "over_odds": -110.0, "under_odds": -110.0,
                        "home_moneyline": -140.0, "away_moneyline": 120.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(pd, "read_parquet", lambda url: schedule)
    result = fetcher.fetch_real_team_game_history(min_season=2020)
    assert "2018_01_I_J" not in set(result["game_id"])
    assert "2024_01_A_B" in set(result["game_id"])


def test_persist_team_game_history_writes_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_parquet", lambda url: _fixture_schedule())
    frame = fetcher.fetch_real_team_game_history()
    out_path = fetcher.persist_team_game_history(frame, output_path=tmp_path / "history.csv")
    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == 1
