from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import fetch_mlb_pitcher_game_data as fetcher  # noqa: E402


def _player(pid: int, name: str, *, started: bool, outs: int, earned_runs: int) -> dict:
    return {
        "person": {"fullName": name},
        "stats": {"pitching": {"gamesStarted": 1 if started else 0, "outs": outs, "earnedRuns": earned_runs}},
    }


def _fixture_boxscore() -> dict:
    return {
        "teams": {
            "home": {
                "team": {"abbreviation": "ATL"},
                "pitchers": [1, 2, 3],
                "players": {
                    "ID1": _player(1, "Chris Sale", started=True, outs=18, earned_runs=1),
                    "ID2": _player(2, "Robert Suarez", started=False, outs=3, earned_runs=0),
                    "ID3": _player(3, "Raisel Iglesias", started=False, outs=3, earned_runs=0),
                },
            },
            "away": {
                "team": {"abbreviation": "ATH"},
                "pitchers": [4, 5],
                "players": {
                    "ID4": _player(4, "Luis Severino", started=True, outs=10, earned_runs=4),
                    "ID5": _player(5, "Hogan Harris", started=False, outs=8, earned_runs=0),
                },
            },
        }
    }


def test_extract_pitcher_game_row_splits_real_starter_and_bullpen() -> None:
    row = fetcher.extract_pitcher_game_row(_fixture_boxscore(), game_pk=824940, date_str="2026-04-01")
    assert row["home_team"] == "ATL"
    assert row["away_team"] == "ATH"
    assert row["home_starter_name"] == "Chris Sale"
    assert row["home_starter_outs"] == 18
    assert row["home_starter_earned_runs"] == 1
    assert row["home_bullpen_outs"] == 6  # 3 + 3
    assert row["home_bullpen_earned_runs"] == 0
    assert row["away_starter_name"] == "Luis Severino"
    assert row["away_starter_outs"] == 10
    assert row["away_bullpen_outs"] == 8


def test_extract_pitcher_game_row_uses_real_gamesStarted_flag_not_list_order() -> None:
    """A real box score can list pitchers in any order -- the starter is
    whoever StatsAPI flags gamesStarted == 1, never assumed to be
    pitchers[0]."""
    boxscore = _fixture_boxscore()
    # Reorder the home side's pitcher list so the real starter is last.
    boxscore["teams"]["home"]["pitchers"] = [3, 2, 1]
    row = fetcher.extract_pitcher_game_row(boxscore, game_pk=824940, date_str="2026-04-01")
    assert row["home_starter_name"] == "Chris Sale"


def test_extract_pitcher_game_row_returns_none_when_no_real_starter_flag() -> None:
    boxscore = _fixture_boxscore()
    for player in boxscore["teams"]["home"]["players"].values():
        player["stats"]["pitching"]["gamesStarted"] = 0
    assert fetcher.extract_pitcher_game_row(boxscore, game_pk=824940, date_str="2026-04-01") is None


def test_extract_pitcher_game_row_returns_none_for_missing_team_abbreviation() -> None:
    boxscore = _fixture_boxscore()
    boxscore["teams"]["home"]["team"]["abbreviation"] = ""
    assert fetcher.extract_pitcher_game_row(boxscore, game_pk=824940, date_str="2026-04-01") is None


def test_fetch_and_persist_games_reuses_a_cached_real_snapshot(tmp_path, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_fetch(game_pk, *, timeout_seconds=20.0):
        calls["count"] += 1
        return _fixture_boxscore()

    monkeypatch.setattr(fetcher, "fetch_boxscore", _fake_fetch)
    raw_root = tmp_path / "raw"
    games = [{"game_pk": 824940, "date": "2026-04-01"}]
    fetcher.fetch_and_persist_games(games, raw_root=raw_root)
    fetcher.fetch_and_persist_games(games, raw_root=raw_root)
    assert calls["count"] == 1


def test_write_pitcher_game_data_csv(tmp_path) -> None:
    row = fetcher.extract_pitcher_game_row(_fixture_boxscore(), game_pk=824940, date_str="2026-04-01")
    out_path = fetcher.write_pitcher_game_data_csv([row], output_path=tmp_path / "pitchers.csv")
    content = out_path.read_text(encoding="utf-8")
    assert "Chris Sale" in content
    assert "Luis Severino" in content


def test_trim_boxscore_for_storage_keeps_only_pitching_fields() -> None:
    trimmed = fetcher._trim_boxscore_for_storage(_fixture_boxscore())
    home_players = trimmed["teams"]["home"]["players"]
    assert "ID1" in home_players
    assert home_players["ID1"]["stats"]["pitching"]["outs"] == 18
    assert "batting" not in home_players["ID1"]["stats"]
