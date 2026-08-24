from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "nba" / "scripts"))

import fetch_nba_team_game_history as fetcher  # noqa: E402


def _fixture_summary(*, completed: bool = True, with_odds: bool = True) -> dict:
    home_linescores = [{"displayValue": "23"}, {"displayValue": "24"}, {"displayValue": "33"}, {"displayValue": "36"}]
    away_linescores = [{"displayValue": "22"}, {"displayValue": "31"}, {"displayValue": "23"}, {"displayValue": "27"}]
    summary = {
        "header": {
            "competitions": [
                {
                    "date": "2026-03-15T23:00Z",
                    "status": {"type": {"completed": completed}},
                    "competitors": [
                        {
                            "homeAway": "home", "score": "116",
                            "team": {"abbreviation": "OKC"},
                            "linescores": home_linescores,
                        },
                        {
                            "homeAway": "away", "score": "103",
                            "team": {"abbreviation": "MIN"},
                            "linescores": away_linescores,
                        },
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
                "spread": -8.5,
                "overUnder": 227.5,
                "homeTeamOdds": {"favorite": True, "moneyLine": -380},
                "awayTeamOdds": {"favorite": False, "moneyLine": 300},
            }
        ]
    return summary


def test_extract_team_game_row_computes_real_scores_and_half_totals() -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(), game_id="401810831")
    assert row["home_score"] == 116.0
    assert row["away_score"] == 103.0
    assert row["home_won"] == 1
    assert row["total_points"] == 219.0
    assert row["home_first_half"] == 47.0
    assert row["away_first_half"] == 53.0
    assert row["first_half_total"] == 100.0


def test_extract_team_game_row_carries_real_single_book_market_line() -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(), game_id="401810831")
    assert row["market_book"] == "DraftKings"
    assert row["market_spread"] == -8.5
    assert row["market_total"] == 227.5
    assert row["market_home_moneyline"] == -380
    assert row["market_away_moneyline"] == 300


def test_extract_team_game_row_returns_none_for_incomplete_game() -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(completed=False), game_id="401810831")
    assert row is None


def test_extract_team_game_row_handles_no_real_market_data() -> None:
    """A real game with no real pickcenter entry (market not covered by
    ESPN for this game) must report empty/None market fields, never a
    guessed line."""
    row = fetcher.extract_team_game_row(_fixture_summary(with_odds=False), game_id="401810831")
    assert row is not None
    assert row["market_book"] == ""
    assert row["market_spread"] is None
    assert row["market_home_moneyline"] is None


def test_extract_team_game_row_returns_none_without_real_competitors() -> None:
    summary = {"header": {"competitions": [{"status": {"type": {"completed": True}}, "competitors": []}]}}
    assert fetcher.extract_team_game_row(summary, game_id="x") is None


def test_fetch_and_persist_games_reuses_a_cached_real_snapshot(tmp_path, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_fetch(game_id, *, timeout_seconds=20.0):
        calls["count"] += 1
        return _fixture_summary()

    monkeypatch.setattr(fetcher, "fetch_game_summary", _fake_fetch)
    raw_root = tmp_path / "raw"
    rows_first = fetcher.fetch_and_persist_games(["401810831"], raw_root=raw_root)
    rows_second = fetcher.fetch_and_persist_games(["401810831"], raw_root=raw_root)
    assert len(rows_first) == 1
    assert len(rows_second) == 1
    assert calls["count"] == 1  # second call reused the persisted real snapshot, no refetch


def test_fetch_and_persist_games_refresh_forces_refetch(tmp_path, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_fetch(game_id, *, timeout_seconds=20.0):
        calls["count"] += 1
        return _fixture_summary()

    monkeypatch.setattr(fetcher, "fetch_game_summary", _fake_fetch)
    raw_root = tmp_path / "raw"
    fetcher.fetch_and_persist_games(["401810831"], raw_root=raw_root)
    fetcher.fetch_and_persist_games(["401810831"], raw_root=raw_root, refresh=True)
    assert calls["count"] == 2


def test_write_team_game_history_csv(tmp_path) -> None:
    row = fetcher.extract_team_game_row(_fixture_summary(), game_id="401810831")
    out_path = fetcher.write_team_game_history_csv([row], output_path=tmp_path / "history.csv")
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "OKC" in content
    assert "DraftKings" in content
