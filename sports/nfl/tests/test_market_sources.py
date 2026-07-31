from __future__ import annotations

import pandas as pd

from sports.nfl.predictions.market_sources import flatten_sportsgameodds_closing_lines


def _payload() -> dict:
    player_id = "JOSH_ALLEN_1_NFL"
    return {
        "success": True,
        "data": [
            {
                "eventID": "nfl-buf-week-1",
                "status": {
                    "finalized": True,
                    "startsAt": "2025-09-08T00:20:00Z",
                },
                "info": {"seasonWeek": "2025 Week 1"},
                "teams": {
                    "home": {"names": {"long": "Buffalo Bills"}},
                    "away": {"names": {"long": "Baltimore Ravens"}},
                },
                "players": {player_id: {"name": "Josh Allen"}},
                "odds": {
                    f"passing_yards-{player_id}-game-ou-over": {
                        "statID": "passing_yards",
                        "playerID": player_id,
                        "periodID": "game",
                        "betTypeID": "ou",
                        "sideID": "over",
                        "byBookmaker": {
                            "draftkings": {
                                "overUnder": "241.5",
                                "odds": "-105",
                                "closeOverUnder": "239.5",
                                "closeOdds": "-110",
                            }
                        },
                    },
                    f"passing_yards-{player_id}-game-ou-under": {
                        "statID": "passing_yards",
                        "playerID": player_id,
                        "periodID": "game",
                        "betTypeID": "ou",
                        "sideID": "under",
                        "byBookmaker": {
                            "draftkings": {
                                "overUnder": "241.5",
                                "odds": "-115",
                                "closeOverUnder": "239.5",
                                "closeOdds": "-110",
                            }
                        },
                    },
                },
            }
        ],
    }


def test_sportsgameodds_adapter_uses_only_explicit_closing_fields() -> None:
    rows, audit = flatten_sportsgameodds_closing_lines(_payload(), season=2025)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["player"] == "Josh Allen"
    assert row["market"] == "player_pass_yds"
    assert row["week"] == 1
    assert row["line"] == 239.5
    assert row["over_price"] == -110
    assert row["under_price"] == -110
    assert row["line_phase"] == "closing_pregame"
    assert bool(row["pregame_verified"])
    assert pd.isna(row["snapshot_time_utc"])
    assert audit["two_sided_price_rows"] == 1


def test_sportsgameodds_adapter_does_not_fall_back_to_current_live_values() -> None:
    payload = _payload()
    over = next(iter(payload["data"][0]["odds"].values()))
    over["byBookmaker"]["draftkings"].pop("closeOverUnder")
    rows, audit = flatten_sportsgameodds_closing_lines(payload, season=2025)
    assert rows.empty
    assert audit["book_sides_without_close"] == 1
    assert audit["dropped_one_sided_rows"] == 1
