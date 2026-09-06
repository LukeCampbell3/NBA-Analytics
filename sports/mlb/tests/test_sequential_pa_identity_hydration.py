from __future__ import annotations

import pandas as pd

from sports.mlb.advanced import integration


def _live_feed() -> dict:
    return {
        "gameData": {
            "players": {
                "ID596019": {"id": 596019, "fullName": "Example Batter"},
                "ID543037": {"id": 543037, "fullName": "Example Pitcher"},
            },
            "probablePitchers": {
                "home": {"id": 543037, "fullName": "Example Pitcher"},
                "away": {"id": 999001, "fullName": "Other Pitcher"},
            },
            "teams": {
                "home": {"abbreviation": "NYY"},
                "away": {"abbreviation": "BOS"},
            },
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "home": {"battingOrder": [596019]},
                    "away": {"battingOrder": []},
                }
            }
        },
    }


def test_identity_hydration_accepts_arrow_string_id_columns(monkeypatch):
    frame = pd.DataFrame(
        {
            "Game_ID": ["777001"],
            "Player": ["Example Batter"],
            "Team": ["BOS"],
            "Player_MLBAM_ID": pd.Series([""], dtype="string[pyarrow]"),
            "Opposing_Pitcher_ID": pd.Series([""], dtype="string[pyarrow]"),
            "Sequential_Batting_Order": pd.Series([""], dtype="string[pyarrow]"),
        }
    )
    monkeypatch.setattr(integration, "_fetch_live_feed", lambda game_id: _live_feed())

    hydrated, diagnostics = integration.hydrate_pool_identities(frame)

    assert str(hydrated["Player_MLBAM_ID"].dtype) == "Int64"
    assert str(hydrated["Opposing_Pitcher_ID"].dtype) == "Int64"
    assert str(hydrated["Sequential_Batting_Order"].dtype) == "Int64"
    assert int(hydrated.at[0, "Player_MLBAM_ID"]) == 596019
    assert int(hydrated.at[0, "Opposing_Pitcher_ID"]) == 543037
    assert int(hydrated.at[0, "Sequential_Batting_Order"]) == 1
    assert diagnostics["games_requested"] == 1
    assert diagnostics["games_resolved"] == 1
    assert diagnostics["identity_failures"] == []


def test_existing_numeric_identity_columns_are_normalized_without_loss(monkeypatch):
    frame = pd.DataFrame(
        {
            "Game_ID": ["777001"],
            "Player": ["Example Batter"],
            "Team": ["BOS"],
            "Player_MLBAM_ID": [596019.0],
            "Opposing_Pitcher_ID": [543037.0],
            "Sequential_Batting_Order": [2.0],
        }
    )
    monkeypatch.setattr(integration, "_fetch_live_feed", lambda game_id: _live_feed())

    hydrated, _ = integration.hydrate_pool_identities(frame)

    assert int(hydrated.at[0, "Player_MLBAM_ID"]) == 596019
    assert int(hydrated.at[0, "Opposing_Pitcher_ID"]) == 543037
    # Official live-feed order supersedes the carried value.
    assert int(hydrated.at[0, "Sequential_Batting_Order"]) == 1
