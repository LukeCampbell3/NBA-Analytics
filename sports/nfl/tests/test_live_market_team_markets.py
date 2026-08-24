from __future__ import annotations

from sports.nfl.predictions.live_market import (
    TEAM_MARKET_KEYS,
    flatten_event_team_market_odds,
)


def _team_event(markets: list[dict]) -> dict:
    return {
        "id": "game-1",
        "commence_time": "2026-09-10T00:20:00Z",
        "home_team": "Philadelphia Eagles",
        "away_team": "Dallas Cowboys",
        "bookmakers": [{"key": "fanduel", "title": "FanDuel", "markets": markets}],
    }


def test_flatten_h2h_market_produces_real_two_sided_moneyline_row() -> None:
    event = _team_event(
        [
            {
                "key": "h2h",
                "last_update": "2026-09-09T14:00:00Z",
                "outcomes": [
                    {"name": "Philadelphia Eagles", "price": -150},
                    {"name": "Dallas Cowboys", "price": 130},
                ],
            }
        ]
    )
    rows = flatten_event_team_market_odds(event, fetched_at_utc="2026-09-09T14:05:00Z")
    assert len(rows) == 1
    row = rows[0]
    assert row["target"] == "moneyline"
    assert row["market"] == "h2h"
    assert row["home_moneyline"] == -150.0
    assert row["away_moneyline"] == 130.0
    assert row["line"] is None
    assert row["bookmaker"] == "fanduel"


def test_flatten_h2h_market_drops_incomplete_one_sided_quote() -> None:
    """A real book missing one side's price (e.g. a stale/partial feed)
    must never be silently completed with a guess."""
    event = _team_event(
        [{"key": "h2h", "outcomes": [{"name": "Philadelphia Eagles", "price": -150}]}]
    )
    rows = flatten_event_team_market_odds(event, fetched_at_utc="2026-09-09T14:05:00Z")
    assert rows == []


def test_flatten_totals_market_produces_real_over_under_row() -> None:
    event = _team_event(
        [
            {
                "key": "totals",
                "last_update": "2026-09-09T14:00:00Z",
                "outcomes": [
                    {"name": "Over", "point": 47.5, "price": -110},
                    {"name": "Under", "point": 47.5, "price": -110},
                ],
            }
        ]
    )
    rows = flatten_event_team_market_odds(event, fetched_at_utc="2026-09-09T14:05:00Z")
    assert len(rows) == 1
    row = rows[0]
    assert row["target"] == "game_total"
    assert row["line"] == 47.5
    assert row["over_price"] == -110.0
    assert row["under_price"] == -110.0


def test_flatten_totals_market_drops_incomplete_one_sided_quote() -> None:
    event = _team_event([{"key": "totals", "outcomes": [{"name": "Over", "point": 47.5, "price": -110}]}])
    rows = flatten_event_team_market_odds(event, fetched_at_utc="2026-09-09T14:05:00Z")
    assert rows == []


def test_flatten_totals_market_keeps_multiple_real_alternate_lines_separate() -> None:
    event = _team_event(
        [
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 47.5, "price": -110},
                    {"name": "Under", "point": 47.5, "price": -110},
                    {"name": "Over", "point": 44.5, "price": -130},
                    {"name": "Under", "point": 44.5, "price": 110},
                ],
            }
        ]
    )
    rows = flatten_event_team_market_odds(event, fetched_at_utc="2026-09-09T14:05:00Z")
    assert {row["line"] for row in rows} == {47.5, 44.5}


def test_flatten_event_team_market_odds_ignores_player_prop_markets() -> None:
    """Same event payload shape as the player-prop test fixtures --
    proves the new team-market flattener never accidentally picks up a
    player-prop market key it wasn't asked for."""
    event = _team_event(
        [
            {
                "key": "player_pass_yds",
                "outcomes": [
                    {"name": "Over", "description": "Quarterback A", "point": 249.5, "price": -105},
                    {"name": "Under", "description": "Quarterback A", "point": 249.5, "price": -115},
                ],
            }
        ]
    )
    assert flatten_event_team_market_odds(event, fetched_at_utc="2026-09-09T14:05:00Z") == []


def test_team_market_keys_are_the_real_odds_api_standard_keys() -> None:
    assert TEAM_MARKET_KEYS == ("h2h", "totals")
