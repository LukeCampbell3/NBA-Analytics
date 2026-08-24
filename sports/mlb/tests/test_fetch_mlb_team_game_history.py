from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import fetch_mlb_team_game_history as fetcher  # noqa: E402


def _fixture_summary(*, completed: bool = True, with_odds: bool = True, home_innings=None, away_innings=None) -> dict:
    home_innings = home_innings if home_innings is not None else ["0", "1", "2", "0", "2", "0", "0", "2"]
    away_innings = away_innings if away_innings is not None else ["0", "0", "0", "0", "0", "0", "0", "0", "0"]
    summary = {
        "header": {
            "competitions": [
                {
                    "date": "2026-06-15T22:00Z",
                    "status": {"type": {"completed": completed}},
                    "competitors": [
                        {"homeAway": "home", "score": "7", "team": {"abbreviation": "PHI"}, "linescores": [{"displayValue": v} for v in home_innings]},
                        {"homeAway": "away", "score": "0", "team": {"abbreviation": "MIA"}, "linescores": [{"displayValue": v} for v in away_innings]},
                    ],
                }
            ]
        },
        "pickcenter": [],
    }
    if with_odds:
        summary["pickcenter"] = [
            {
                "provider": {"name": "DraftKings"},
                "overUnder": 8.0,
                "homeTeamOdds": {"moneyLine": -198},
                "awayTeamOdds": {"moneyLine": 162},
            }
        ]
    return summary


def test_extract_team_game_row_computes_real_scores_and_f5_total() -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(), game_id="401815765")
    assert row["home_score"] == 7.0
    assert row["away_score"] == 0.0
    assert row["home_won"] == 1
    assert row["total_runs"] == 7.0
    assert row["home_innings_1_5"] == 5.0  # 0+1+2+0+2
    assert row["away_innings_1_5"] == 0.0
    assert row["first_5_innings_total"] == 5.0


def test_extract_team_game_row_carries_real_market_data() -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(), game_id="401815765")
    assert row["market_book"] == "DraftKings"
    assert row["market_run_total"] == 8.0
    assert row["market_home_moneyline"] == -198
    assert row["market_away_moneyline"] == 162


def test_extract_team_game_row_returns_none_for_incomplete_game() -> None:
    assert fetcher.extract_team_game_row(_fixture_summary(completed=False), game_id="401815765") is None


def test_extract_team_game_row_handles_rain_shortened_game_missing_f5() -> None:
    """A real rain-shortened game with fewer than 5 real innings played
    must report first_5_innings_total as None, never a partial/guessed sum."""
    row = fetcher.extract_team_game_row(
        _fixture_summary(home_innings=["0", "1", "2"], away_innings=["0", "0", "1"]), game_id="401815765"
    )
    assert row["home_innings_1_5"] is None
    assert row["away_innings_1_5"] is None
    assert row["first_5_innings_total"] is None


def test_extract_team_game_row_handles_no_real_market_data() -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(with_odds=False), game_id="401815765")
    assert row is not None
    assert row["market_book"] == ""
    assert row["market_home_moneyline"] is None


def test_fetch_and_persist_games_reuses_a_cached_real_snapshot(tmp_path, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_fetch(game_id, *, timeout_seconds=20.0):
        calls["count"] += 1
        return _fixture_summary()

    monkeypatch.setattr(fetcher, "fetch_game_summary", _fake_fetch)
    raw_root = tmp_path / "raw"
    fetcher.fetch_and_persist_games(["401815765"], raw_root=raw_root)
    fetcher.fetch_and_persist_games(["401815765"], raw_root=raw_root)
    assert calls["count"] == 1


def test_write_team_game_history_csv(tmp_path) -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(), game_id="401815765")
    out_path = fetcher.write_team_game_history_csv([row], output_path=tmp_path / "history.csv")
    content = out_path.read_text(encoding="utf-8")
    assert "PHI" in content
    assert "DraftKings" in content
