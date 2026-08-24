from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLF_PREDICTIONS_ROOT = REPO_ROOT / "sports" / "golf" / "predictions"
sys.path.insert(0, str(GOLF_PREDICTIONS_ROOT))

from odds_provider import TheOddsApiGolfProvider  # noqa: E402


def test_missing_api_key_reports_missing_credentials() -> None:
    provider = TheOddsApiGolfProvider(api_key="")
    result = provider.collect_odds()
    assert result["status"] == "missing_credentials"


def test_discover_golf_sport_keys_filters_to_active_golf_only() -> None:
    provider = TheOddsApiGolfProvider(
        api_key="fixture",
        fixture_sports=[
            {"key": "golf_the_open_championship_winner", "title": "The Open", "active": True},
            {"key": "golf_masters_tournament_winner", "title": "The Masters", "active": False},
            {"key": "baseball_mlb", "title": "MLB", "active": True},
        ],
    )
    keys = provider.discover_golf_sport_keys()
    assert [k["key"] for k in keys] == ["golf_the_open_championship_winner"]


def test_no_active_golf_market_is_reported_honestly_not_as_an_error() -> None:
    provider = TheOddsApiGolfProvider(api_key="fixture", fixture_sports=[])
    result = provider.collect_odds()
    assert result["status"] == "no_active_golf_market"
    assert result["accounting"]["sport_keys_found"] == 0


def test_collect_odds_extracts_real_outright_and_top10_markets() -> None:
    sport_key = "golf_pga_tour_championship_winner"
    provider = TheOddsApiGolfProvider(
        api_key="fixture",
        fixture_sports=[{"key": sport_key, "title": "TOUR Championship", "active": True}],
        fixture_odds={
            sport_key: [
                {
                    "id": "evt1",
                    "sport_title": "TOUR Championship",
                    "commence_time": "2026-08-27T13:00:00Z",
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "title": "DraftKings",
                            "last_update": "2026-08-24T12:00:00Z",
                            "markets": [
                                {
                                    "key": "outrights",
                                    "last_update": "2026-08-24T12:00:00Z",
                                    "outcomes": [
                                        {"name": "Scottie Scheffler", "price": 350},
                                        {"name": "Sam Burns", "price": 900},
                                    ],
                                },
                                {
                                    "key": "top_10_finish",
                                    "last_update": "2026-08-24T12:00:00Z",
                                    "outcomes": [{"name": "Scottie Scheffler", "price": -250}],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
    )
    result = provider.collect_odds()
    assert result["status"] == "success"
    odds = result["odds"]
    markets = {row.market for row in odds}
    assert markets == {"WINNER", "TOP_10"}
    winner_row = next(row for row in odds if row.market == "WINNER" and row.player_name == "Scottie Scheffler")
    assert winner_row.price_american == 350.0
    assert winner_row.sportsbook_key == "draftkings"
    assert winner_row.event_id == "evt1"
    assert result["accounting"]["rows_by_market"] == {"WINNER": 2, "TOP_10": 1}


def test_collect_odds_ignores_unrecognized_market_keys() -> None:
    sport_key = "golf_x"
    provider = TheOddsApiGolfProvider(
        api_key="fixture",
        fixture_sports=[{"key": sport_key, "title": "X", "active": True}],
        fixture_odds={
            sport_key: [
                {
                    "id": "evt1",
                    "commence_time": "2026-08-27T13:00:00Z",
                    "bookmakers": [
                        {
                            "key": "fanduel",
                            "title": "FanDuel",
                            "markets": [{"key": "h2h", "outcomes": [{"name": "Player A vs Player B", "price": -110}]}],
                        }
                    ],
                }
            ]
        },
    )
    result = provider.collect_odds()
    assert result["status"] == "no_props"
