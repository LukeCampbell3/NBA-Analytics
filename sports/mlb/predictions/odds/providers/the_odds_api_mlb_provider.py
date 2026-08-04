#!/usr/bin/env python3
"""The Odds API v4 adapter for MLB player props."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odds_contract import ensure_contract, normalize_identity, stable_hash


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
DEFAULT_MARKETS = [
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_strikeouts",
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_walks",
    "pitcher_earned_runs",
    "pitcher_outs",
]


class TheOddsApiMlbProvider:
    """Collect MLB props one event at a time using the documented v4 API."""

    def __init__(self, api_key: str | None = None, fixture_payloads: list[dict[str, Any]] | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("THE_ODDS_API_KEY", "")
        self.fixture_payloads = fixture_payloads
        self.base_url = os.environ.get("THE_ODDS_API_BASE_URL", BASE_URL).rstrip("/")
        self.regions = os.environ.get("MLB_ODDS_API_REGIONS", "us")
        self.bookmakers = os.environ.get("MLB_ODDS_API_BOOKMAKERS", "")
        configured_markets = os.environ.get("MLB_ODDS_API_MARKETS", "")
        self.markets = [value.strip() for value in configured_markets.split(",") if value.strip()] or DEFAULT_MARKETS
        self._accounting: dict[str, Any] = {}

    def validate_config(self) -> dict[str, Any]:
        if self.fixture_payloads is not None:
            return {"status": "ok"}
        if not self.api_key:
            return {"status": "missing_credentials", "message": "THE_ODDS_API_KEY not set"}
        return {"status": "ok"}

    def get_accounting(self) -> dict[str, Any]:
        return dict(self._accounting)

    def collect_player_props(self) -> dict[str, Any]:
        config = self.validate_config()
        if config["status"] != "ok":
            return config
        if os.environ.get("PYTEST_CURRENT_TEST") and self.fixture_payloads is None and os.environ.get("MLB_ENABLE_LIVE_API_TESTS") != "1":
            return {"status": "live_api_disabled_for_tests", "message": "Live The Odds API calls are disabled during pytest"}

        requested_at = datetime.now(timezone.utc)
        try:
            payloads = self.fixture_payloads if self.fixture_payloads is not None else self._collect_live_payloads()
            rows: list[dict[str, Any]] = []
            for payload in payloads:
                rows.extend(self._extract_payload(payload, requested_at.isoformat()))
            self._accounting = {
                "events_checked": len(payloads),
                "raw_player_props_found": len(rows),
                "normalized_book_rows": len(rows),
            }
            if not rows:
                return {"status": "no_props", "message": "No supported MLB player props returned", "accounting": self._accounting}
            return {"status": "success", "odds": rows, "accounting": self._accounting}
        except requests.HTTPError as exc:
            response = exc.response
            code = response.status_code if response is not None else 0
            if code in {401, 403}:
                return {"status": "missing_credentials", "message": f"The Odds API authentication failed ({code})"}
            if code == 429:
                return {
                    "status": "rate_limited",
                    "message": "The Odds API rate limited the request",
                    "retry_after": response.headers.get("Retry-After") if response is not None else None,
                }
            return {"status": "api_error", "message": f"The Odds API HTTP {code}"}
        except requests.Timeout:
            return {"status": "source_timeout", "message": "The Odds API request timed out"}
        except (requests.RequestException, ValueError) as exc:
            return {"status": "api_error", "message": str(exc)[:200]}

    def _collect_live_payloads(self) -> list[dict[str, Any]]:
        common = {"apiKey": self.api_key, "dateFormat": "iso"}
        events_url = f"{self.base_url}/sports/{SPORT_KEY}/events"
        response = requests.get(events_url, params=common, timeout=30)
        response.raise_for_status()
        events = response.json()
        if not isinstance(events, list):
            raise ValueError("The Odds API events response was not a list")

        payloads: list[dict[str, Any]] = []
        for event in events:
            event_id = str(event.get("id", ""))
            if not event_id:
                continue
            endpoint = f"{self.base_url}/sports/{SPORT_KEY}/events/{event_id}/odds"
            params = {
                **common,
                "regions": self.regions,
                "markets": ",".join(self.markets),
                "oddsFormat": "american",
            }
            if self.bookmakers:
                params["bookmakers"] = self.bookmakers
            odds_response = requests.get(endpoint, params=params, timeout=30)
            odds_response.raise_for_status()
            payload = odds_response.json()
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def _extract_payload(self, payload: dict[str, Any], fallback_observed: str) -> list[dict[str, Any]]:
        event_id = str(payload.get("id", ""))
        start = payload.get("commence_time", "")
        home = payload.get("home_team", "")
        away = payload.get("away_team", "")
        rows: list[dict[str, Any]] = []
        for bookmaker in payload.get("bookmakers", []):
            book = bookmaker.get("key") or bookmaker.get("title") or ""
            book_updated = bookmaker.get("last_update") or fallback_observed
            for market in bookmaker.get("markets", []):
                market_key = str(market.get("key", ""))
                if market_key not in self.markets:
                    continue
                market_updated = market.get("last_update") or book_updated
                for outcome in market.get("outcomes", []):
                    side = str(outcome.get("name", "")).lower()
                    if side not in {"over", "under"}:
                        continue
                    player_name = str(outcome.get("description", "")).strip()
                    line = outcome.get("point")
                    price = outcome.get("price")
                    if not player_name or line is None or price is None:
                        continue
                    raw_hash = stable_hash({"event": event_id, "book": book, "market": market_key, "outcome": outcome})
                    rows.append(
                        {
                            "source": "the_odds_api",
                            "source_market_id": raw_hash[:24],
                            "sportsbook": book,
                            "event_id": event_id,
                            "external_event_id": event_id,
                            "player_id": normalize_identity(player_name),
                            "external_player_id": normalize_identity(player_name),
                            "player_name": player_name,
                            "home_team": home,
                            "away_team": away,
                            "game_start_utc": start,
                            "league": "MLB",
                            "market_type": market_key,
                            "side": side,
                            "line": line,
                            "price_american": price,
                            "observed_at_utc": market_updated,
                            "source_updated_at_utc": market_updated,
                            "source_url_or_endpoint": f"{self.base_url}/sports/{SPORT_KEY}/events/{event_id}/odds",
                            "acquisition_method": "api",
                            "raw_record_hash": raw_hash,
                            "parser_version": "the-odds-api-v4-parser-v1",
                        }
                    )
        return rows

    def normalize(self, raw_odds: list[dict[str, Any]]) -> pd.DataFrame:
        return ensure_contract(
            pd.DataFrame(raw_odds),
            source="the_odds_api",
            acquisition_method="api",
            source_endpoint=f"{self.base_url}/sports/{SPORT_KEY}/events/{{eventId}}/odds",
            parser_version="the-odds-api-v4-parser-v1",
        )
