from __future__ import annotations

import pandas as pd

from sports.nfl.predictions.market_sources import (
    flatten_sportsgameodds_closing_lines,
    flatten_sportsgameodds_consensus_closing_lines,
    flatten_xsportsbook_bovada_archive,
)


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


def test_sportsgameodds_consensus_close_is_not_labeled_as_named_book() -> None:
    payload = _payload()
    for odd in payload["data"][0]["odds"].values():
        book = odd["byBookmaker"]["draftkings"]
        odd["closeBookOverUnder"] = book["closeOverUnder"]
        odd["closeBookOdds"] = book["closeOdds"]
        book.pop("closeOverUnder")
        book.pop("closeOdds")

    rows, audit = flatten_sportsgameodds_consensus_closing_lines(payload, season=2025)

    assert len(rows) == 1
    assert rows.iloc[0]["bookmaker"] == "sportsgameodds_consensus"
    assert rows.iloc[0]["source"] == "sportsgameodds_consensus_close"
    assert bool(rows.iloc[0]["pregame_verified"])
    assert not bool(rows.iloc[0]["executable_book_verified"])
    assert audit["two_sided_price_rows"] == 1


def test_xsportsbook_adapter_keeps_lines_prices_and_discards_results() -> None:
    raw = pd.DataFrame(
        {
            "Game_Id": [10, 10, 10, 10],
            "Player": ["Quarter Back\u00a0", "Running Back", "Wide Receiver", "Wide Receiver"],
            "Player.id": [1, 2, 3, 3],
            "Betting Event": [
                " Passing Yards ", "Rushing Yards", "Receiving Yards", "Receiving Yards"
            ],
            "Team": ["LAR"] * 4,
            "Opp": ["BUF"] * 4,
            "Hteam": ["LAR"] * 4,
            "Ateam": ["BUF"] * 4,
            "Week": ["22W01"] * 4,
            "O-Line": [250.5, 55.5, 65.5, 66.5],
            "O-Odds": [-110, 105, -115, -115],
            "U-Line": [250.5, 55.5, 65.5, 66.5],
            "U-Odds": [-120, -135, -115, -115],
            "O-Result": [300, 40, 80, 80],
        }
    )
    schedule = pd.DataFrame(
        {
            "week": [1],
            "home_team": ["LA"],
            "away_team": ["BUF"],
            "commence_time_utc": [pd.Timestamp("2022-09-09T00:20:00Z")],
        }
    )
    rows, audit = flatten_xsportsbook_bovada_archive(raw, season=2022, schedule=schedule)
    assert len(rows) == 2  # conflicting receiving lines are conservatively removed
    assert set(rows["market"]) == {"player_pass_yds", "player_rush_yds"}
    assert rows["commence_time_utc"].notna().all()
    assert rows["over_price"].tolist() == [-110, 105]
    assert not rows["pregame_verified"].any()
    assert "O-Result" not in rows.columns
    assert audit["ambiguous_duplicate_rows"] == 2
