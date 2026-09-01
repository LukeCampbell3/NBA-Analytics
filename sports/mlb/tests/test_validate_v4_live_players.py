from __future__ import annotations

from sports.mlb.scripts.validate_v4_live_players import apply_live_identity_gate


def _feed(*, home: str = "LAA", away: str = "NYY") -> dict:
    return {
        "gameData": {
            "teams": {
                "home": {"abbreviation": home},
                "away": {"abbreviation": away},
            }
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {
                        "battingOrder": [101],
                        "players": {
                            "ID101": {"person": {"id": 101, "fullName": "Josh Lowe"}},
                        },
                    },
                    "home": {
                        "battingOrder": [202],
                        "players": {
                            "ID202": {"person": {"id": 202, "fullName": "Mike Trout"}},
                            "ID203": {"person": {"id": 203, "fullName": "Denzer Guzman"}},
                        },
                    },
                }
            }
        },
    }


def _play(player: str, *, game_id: str = "823982", link: bool = True) -> dict:
    row = {
        "player": player,
        "game_id": game_id,
        "target": "H",
        "direction": "OVER",
        "line": 0.5,
        "execution_status": "LIVE_SELECTION_AVAILABLE",
    }
    if link:
        row["sportsbook_deeplink"] = "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=2"
    return row


def test_live_gate_enriches_confirmed_starter_and_rejects_nonstarter_and_wrong_game_player() -> None:
    payload = {
        "v4_singles_shadow": {
            "eligible_count": 3,
            "plays": [
                _play("Josh Lowe"),
                _play("Denzer Guzman"),
                _play("David Hamilton"),
            ],
        }
    }

    updated = apply_live_identity_gate(payload, fetch_json=lambda _: _feed())
    shadow = updated["v4_singles_shadow"]

    assert len(shadow["plays"]) == 1
    assert shadow["plays"][0]["player"] == "Josh Lowe"
    assert shadow["plays"][0]["player_id"] == 101
    assert shadow["plays"][0]["team"] == "NYY"
    assert shadow["plays"][0]["opponent"] == "LAA"
    assert shadow["plays"][0]["is_home"] == "0"
    assert shadow["plays"][0]["lineup_status"] == "CONFIRMED_STARTER"
    reasons = {row["player"]: row["reason"] for row in shadow["identity_rejections"]}
    assert reasons["Denzer Guzman"] == "PLAYER_NOT_IN_STARTING_LINEUP"
    assert reasons["David Hamilton"] == "PLAYER_GAME_IDENTITY_MISMATCH"


def test_live_gate_rejects_carried_team_or_home_away_conflict() -> None:
    wrong_team = _play("Josh Lowe")
    wrong_team.update({"team": "LAA", "opponent": "NYY", "is_home": "1"})
    payload = {"v4_singles_shadow": {"eligible_count": 1, "plays": [wrong_team]}}
    updated = apply_live_identity_gate(payload, fetch_json=lambda _: _feed())
    assert updated["v4_singles_shadow"]["plays"] == []
    assert updated["v4_singles_shadow"]["identity_rejections"][0]["reason"] == "TEAM_GAME_IDENTITY_MISMATCH"


def test_live_gate_fails_closed_when_selection_link_is_missing() -> None:
    payload = {"v4_singles_shadow": {"eligible_count": 1, "plays": [_play("Josh Lowe", link=False)]}}
    updated = apply_live_identity_gate(payload, fetch_json=lambda _: _feed())
    shadow = updated["v4_singles_shadow"]
    assert shadow["plays"] == []
    assert shadow["identity_rejections"][0]["reason"] == "LIVE_SELECTION_UNAVAILABLE"


def test_live_gate_fails_closed_when_starting_lineup_is_not_available() -> None:
    feed = _feed()
    feed["liveData"]["boxscore"]["teams"]["away"]["battingOrder"] = []
    payload = {"v4_singles_shadow": {"eligible_count": 1, "plays": [_play("Josh Lowe")]}}
    updated = apply_live_identity_gate(payload, fetch_json=lambda _: feed)
    assert updated["v4_singles_shadow"]["plays"] == []
    assert updated["v4_singles_shadow"]["identity_rejections"][0]["reason"] == "STARTING_LINEUP_UNCONFIRMED"
