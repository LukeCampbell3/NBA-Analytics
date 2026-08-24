from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "nfl" / "scripts"))

import fetch_nfl_team_game_halves as fetcher  # noqa: E402


def _fixture_summary(*, completed: bool = True, with_odds: bool = True) -> dict:
    home_linescores = [{"displayValue": "7"}, {"displayValue": "14"}, {"displayValue": "0"}, {"displayValue": "7"}]
    away_linescores = [{"displayValue": "3"}, {"displayValue": "7"}, {"displayValue": "6"}, {"displayValue": "0"}]
    summary = {
        "header": {
            "competitions": [
                {
                    "date": "2026-01-18T00:20Z",
                    "status": {"type": {"completed": completed}},
                    "competitors": [
                        {"homeAway": "home", "score": "28", "team": {"abbreviation": "NE"}, "linescores": home_linescores},
                        {"homeAway": "away", "score": "16", "team": {"abbreviation": "HOU"}, "linescores": away_linescores},
                    ],
                }
            ]
        },
        "pickcenter": [],
    }
    if with_odds:
        summary["pickcenter"] = [{"provider": {"name": "DraftKings"}, "overUnder": 40.5, "spread": -3.0}]
    return summary


def test_extract_team_game_half_row_computes_real_scores_and_halves() -> None:
    row = fetcher.extract_team_game_half_row(_fixture_summary(), game_id="401772983")
    assert row["home_score"] == 28.0
    assert row["away_score"] == 16.0
    assert row["home_first_half"] == 21.0
    assert row["away_first_half"] == 10.0
    assert row["first_half_total"] == 31.0


def test_extract_team_game_half_row_carries_real_book_identity() -> None:
    row = fetcher.extract_team_game_half_row(_fixture_summary(), game_id="401772983")
    assert row["market_book"] == "DraftKings"
    # No real first-half-specific total line exists in ESPN's pickcenter --
    # this must stay honestly unset, never backfilled from the full-game total.
    assert row["market_first_half_total"] is None


def test_extract_team_game_half_row_returns_none_for_incomplete_game() -> None:
    assert fetcher.extract_team_game_half_row(_fixture_summary(completed=False), game_id="401772983") is None


def test_extract_team_game_half_row_handles_no_real_market_data() -> None:
    row = fetcher.extract_team_game_half_row(_fixture_summary(with_odds=False), game_id="401772983")
    assert row is not None
    assert row["market_book"] == ""


def test_fetch_and_persist_games_reuses_a_cached_real_snapshot(tmp_path, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_fetch(game_id, *, timeout_seconds=20.0):
        calls["count"] += 1
        return _fixture_summary()

    monkeypatch.setattr(fetcher, "fetch_game_summary", _fake_fetch)
    raw_root = tmp_path / "raw"
    fetcher.fetch_and_persist_games(["401772983"], raw_root=raw_root)
    fetcher.fetch_and_persist_games(["401772983"], raw_root=raw_root)
    assert calls["count"] == 1


def test_write_team_game_halves_csv(tmp_path) -> None:
    row = fetcher.extract_team_game_half_row(_fixture_summary(), game_id="401772983")
    out_path = fetcher.write_team_game_halves_csv([row], output_path=tmp_path / "halves.csv")
    content = out_path.read_text(encoding="utf-8")
    assert "NE" in content
    assert "31.0" in content
