import importlib.util
from pathlib import Path

import sys


SCRIPT = Path("Player-Predictor/scripts/fetch_nba_the_odds_api_clv_snapshots.py")
sys.path.insert(0, str(SCRIPT.parent.resolve()))
SPEC = importlib.util.spec_from_file_location("fetch_nba_the_odds_api_clv_snapshots", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_event_payload_keeps_only_nba_pts_trb_ast_two_sided_american_odds():
    payload = {
        "id": "game1",
        "commence_time": "2026-01-01T00:00:00Z",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "price": -115, "point": 27.5},
                            {"name": "Under", "description": "Jayson Tatum", "price": -105, "point": 27.5},
                            {"name": "Over", "description": "Jaylen Brown", "price": -11, "point": 22.5},
                            {"name": "Under", "description": "Jaylen Brown", "price": -105, "point": 22.5},
                        ],
                    },
                    {
                        "key": "player_rebounds",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "price": 110, "point": 8.5},
                            {"name": "Under", "description": "Jayson Tatum", "price": -130, "point": 8.5},
                        ],
                    },
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -150},
                            {"name": "New York Knicks", "price": 130},
                        ],
                    },
                ],
            }
        ],
    }

    rows = MODULE.normalize_event_payload(
        payload,
        requested_snapshot_time="2025-12-31T23:45:00Z",
        snapshot_type="prelock",
        source_snapshot_time="2025-12-31T23:44:00Z",
    )

    assert len(rows) == 2
    assert set(rows["market"]) == {"PTS", "TRB"}
    assert set(rows["player"]) == {"Jayson_Tatum"}
    assert rows["is_valid_american_odds"].all()
    assert set(rows["snapshot_type"]) == {"prelock"}
    assert rows.iloc[0]["game_start_time"] == "2026-01-01T00:00:00Z"


def test_phase_time_uses_nba_commence_time_offset():
    assert MODULE._phase_time("2026-01-01T00:00:00Z", 15) == "2025-12-31T23:45:00Z"
