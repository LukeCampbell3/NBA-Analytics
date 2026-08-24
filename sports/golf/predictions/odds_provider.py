from __future__ import annotations

"""Real PGA Tour odds via The Odds API v4, reusing this repo's existing
THE_ODDS_API_KEY (already wired into NFL/NBA/MLB workflows).

WHY THIS LOOKS DIFFERENT FROM sports/mlb's the_odds_api provider: MLB has
one persistent sport key (`baseball_mlb`) all season. Golf does not --
The Odds API publishes a new, tournament-specific sport key only once a
real market for that event exists (commonly a few days before a
tournament starts, and only for events books actually price), and no
fixed naming scheme predicts it in advance. This module discovers the
real, currently-live golf sport key(s) from `/v4/sports/` at request
time -- it never hardcodes or guesses a key.

Market reality, stated honestly rather than assumed: The Odds API's golf
coverage is centered on the outright tournament-winner market
(`markets=outrights`). Top-5/10/20-finish and make/miss-cut markets are
NOT reliably published there -- this module still requests them (in case
a book adds them for a given event) but callers must not assume they will
come back non-empty; `collect_odds()`'s `accounting` reports exactly what
came back per market so this is never silently hidden.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import requests

BASE_URL = "https://api.the-odds-api.com/v4"
REQUEST_TIMEOUT_SECONDS = 30.0
# Real market keys this module asks for. "outrights" is the only one with
# reliable real golf coverage today (see module docstring); the others are
# requested defensively and simply come back empty when a book doesn't
# offer them for a given event -- never fabricated.
GOLF_MARKET_KEYS = ("outrights", "top_5_finish", "top_10_finish", "top_20_finish", "make_cut")


@dataclass(frozen=True)
class OddsRow:
    player_name: str
    market: str  # WINNER | TOP_5 | TOP_10 | TOP_20 | MAKE_CUT
    side: str  # YES (single-sided market -- there is no real "NO" price for these)
    price_american: float
    sportsbook_key: str
    sportsbook_title: str
    event_id: str
    event_name: str
    commence_time_utc: str
    observed_at_utc: str


_MARKET_KEY_TO_TARGET = {
    "outrights": "WINNER",
    "top_5_finish": "TOP_5",
    "top_10_finish": "TOP_10",
    "top_20_finish": "TOP_20",
    "make_cut": "MAKE_CUT",
}


class TheOddsApiGolfProvider:
    def __init__(self, api_key: Optional[str] = None, *, fixture_sports: Optional[list[dict[str, Any]]] = None, fixture_odds: Optional[dict[str, list[dict[str, Any]]]] = None):
        self.api_key = api_key if api_key is not None else os.environ.get("THE_ODDS_API_KEY", "")
        self.base_url = os.environ.get("THE_ODDS_API_BASE_URL", BASE_URL).rstrip("/")
        self.regions = os.environ.get("GOLF_ODDS_API_REGIONS", "us")
        # fixture_* let tests/CI dry-runs exercise this provider's real
        # parsing logic without a live network call or a real key.
        self._fixture_sports = fixture_sports
        self._fixture_odds = fixture_odds
        self._accounting: dict[str, Any] = {}

    def validate_config(self) -> dict[str, Any]:
        if self._fixture_sports is not None:
            return {"status": "ok"}
        if not self.api_key:
            return {"status": "missing_credentials", "message": "THE_ODDS_API_KEY not set"}
        return {"status": "ok"}

    def get_accounting(self) -> dict[str, Any]:
        return dict(self._accounting)

    def discover_golf_sport_keys(self) -> list[dict[str, str]]:
        """Every real, currently-active golf sport key The Odds API lists
        right now -- e.g. a specific PGA Tour event once a book has priced
        it. Returns [] (not an error) when no real golf market is live,
        which is a genuine, expected state most of a normal week."""
        if self._fixture_sports is not None:
            payload = self._fixture_sports
        else:
            response = requests.get(f"{self.base_url}/sports/", params={"apiKey": self.api_key}, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, list):
            return []
        return [
            {"key": str(entry.get("key") or ""), "title": str(entry.get("title") or "")}
            for entry in payload
            if str(entry.get("key") or "").startswith("golf_") and entry.get("active")
        ]

    def collect_odds(self) -> dict[str, Any]:
        config = self.validate_config()
        if config["status"] != "ok":
            return config
        if os.environ.get("PYTEST_CURRENT_TEST") and self._fixture_sports is None and os.environ.get("GOLF_ENABLE_LIVE_API_TESTS") != "1":
            return {"status": "live_api_disabled_for_tests", "message": "Live The Odds API calls are disabled during pytest"}

        try:
            sport_keys = self.discover_golf_sport_keys()
            if not sport_keys:
                self._accounting = {"sport_keys_found": 0, "rows_by_market": {}}
                return {"status": "no_active_golf_market", "message": "No real golf sport key is currently active on The Odds API", "accounting": self._accounting}

            observed_at = datetime.now(timezone.utc).isoformat()
            rows: list[OddsRow] = []
            rows_by_market: dict[str, int] = {}
            for sport in sport_keys:
                events = self._fetch_odds_for_sport(sport["key"])
                for event in events:
                    extracted = self._extract_event(event, observed_at)
                    for row in extracted:
                        rows_by_market[row.market] = rows_by_market.get(row.market, 0) + 1
                    rows.extend(extracted)

            self._accounting = {"sport_keys_found": len(sport_keys), "sport_keys": [s["key"] for s in sport_keys], "rows_by_market": rows_by_market}
            if not rows:
                return {"status": "no_props", "message": "Real golf sport key(s) found but no market rows returned", "accounting": self._accounting}
            return {"status": "success", "odds": rows, "accounting": self._accounting}
        except requests.HTTPError as exc:
            response = exc.response
            code = response.status_code if response is not None else 0
            if code in {401, 403}:
                return {"status": "missing_credentials", "message": f"The Odds API authentication failed ({code})"}
            if code == 429:
                return {"status": "rate_limited", "message": "The Odds API rate limited the request", "retry_after": response.headers.get("Retry-After") if response is not None else None}
            return {"status": "api_error", "message": f"The Odds API HTTP {code}"}
        except requests.Timeout:
            return {"status": "source_timeout", "message": "The Odds API request timed out"}
        except (requests.RequestException, ValueError) as exc:
            return {"status": "api_error", "message": str(exc)[:200]}

    def _fetch_odds_for_sport(self, sport_key: str) -> list[dict[str, Any]]:
        if self._fixture_odds is not None:
            return self._fixture_odds.get(sport_key, [])
        response = requests.get(
            f"{self.base_url}/sports/{sport_key}/odds",
            params={
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": ",".join(GOLF_MARKET_KEYS),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def _extract_event(self, event: dict[str, Any], fallback_observed: str) -> list[OddsRow]:
        event_id = str(event.get("id", ""))
        event_name = str(event.get("sport_title", "") or event.get("home_team", ""))
        commence_time = str(event.get("commence_time", ""))
        rows: list[OddsRow] = []
        for bookmaker in event.get("bookmakers", []):
            book_key = str(bookmaker.get("key") or "")
            book_title = str(bookmaker.get("title") or book_key)
            book_updated = str(bookmaker.get("last_update") or fallback_observed)
            for market in bookmaker.get("markets", []):
                market_key = str(market.get("key", ""))
                target = _MARKET_KEY_TO_TARGET.get(market_key)
                if target is None:
                    continue
                market_updated = str(market.get("last_update") or book_updated)
                for outcome in market.get("outcomes", []):
                    player_name = str(outcome.get("name", "")).strip()
                    price = outcome.get("price")
                    if not player_name or price is None:
                        continue
                    rows.append(
                        OddsRow(
                            player_name=player_name,
                            market=target,
                            side="YES",
                            price_american=float(price),
                            sportsbook_key=book_key,
                            sportsbook_title=book_title,
                            event_id=event_id,
                            event_name=event_name,
                            commence_time_utc=commence_time,
                            observed_at_utc=market_updated,
                        )
                    )
        return rows
