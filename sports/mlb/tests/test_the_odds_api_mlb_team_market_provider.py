from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"))

from the_odds_api_mlb_team_market_provider import TheOddsApiMlbTeamMarketProvider  # noqa: E402


def _payload(markets: list[dict]) -> dict:
    return {
        "id": "evt1",
        "commence_time": "2026-06-15T22:00:00Z",
        "home_team": "Philadelphia Phillies",
        "away_team": "Miami Marlins",
        "bookmakers": [{"key": "draftkings", "title": "DraftKings", "last_update": "2026-06-15T20:00:00Z", "markets": markets}],
    }


def test_missing_api_key_reports_missing_credentials() -> None:
    provider = TheOddsApiMlbTeamMarketProvider(api_key="")
    result = provider.collect_team_market_odds()
    assert result["status"] == "missing_credentials"


def test_extracts_real_two_sided_moneyline() -> None:
    payload = _payload(
        [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": "Philadelphia Phillies", "price": -198},
                    {"name": "Miami Marlins", "price": 162},
                ],
            }
        ]
    )
    provider = TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[payload])
    result = provider.collect_team_market_odds()
    assert result["status"] == "success"
    row = result["odds"][0]
    assert row["target"] == "moneyline"
    assert row["home_moneyline"] == -198.0
    assert row["away_moneyline"] == 162.0
    assert row["line"] is None


def test_moneyline_drops_incomplete_one_sided_quote() -> None:
    payload = _payload([{"key": "h2h", "outcomes": [{"name": "Philadelphia Phillies", "price": -198}]}])
    provider = TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[payload])
    result = provider.collect_team_market_odds()
    assert result["status"] == "no_props"


def test_extracts_real_two_sided_run_total() -> None:
    payload = _payload(
        [
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 8.0, "price": -108},
                    {"name": "Under", "point": 8.0, "price": -111},
                ],
            }
        ]
    )
    provider = TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[payload])
    result = provider.collect_team_market_odds()
    assert result["status"] == "success"
    row = result["odds"][0]
    assert row["target"] == "game_total"
    assert row["line"] == 8.0
    assert row["over_price"] == -108.0
    assert row["under_price"] == -111.0


def test_ignores_player_prop_market_keys() -> None:
    payload = _payload(
        [
            {
                "key": "batter_total_bases",
                "outcomes": [
                    {"name": "Over", "description": "Player A", "point": 1.5, "price": -110},
                    {"name": "Under", "description": "Player A", "point": 1.5, "price": -110},
                ],
            }
        ]
    )
    provider = TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[payload])
    result = provider.collect_team_market_odds()
    assert result["status"] == "no_props"


def test_normalize_handles_team_market_rows_without_player_identity() -> None:
    payload = _payload([{"key": "h2h", "outcomes": [{"name": "Philadelphia Phillies", "price": -198}, {"name": "Miami Marlins", "price": 162}]}])
    provider = TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[payload])
    result = provider.collect_team_market_odds()
    normalized = provider.normalize(result["odds"])
    assert len(normalized) == 1
    assert normalized.iloc[0]["home_team"] == "Philadelphia Phillies"
