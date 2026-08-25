#!/usr/bin/env python3
"""Anonymous read-only FanDuel MLB TEAM-market provider: real moneyline
and real full-game total, no auth, no API key -- the same real,
already-in-production technique and endpoints fanduel_public_mlb_
provider.py (player props) uses, reading the DEFAULT event page (no
prop-tab param) where FanDuel's own real site already shows Moneyline /
Run Line / Total Runs before any prop tab is even selected. Confirmed
against a real live request before writing this: event 35973141
(a real, not-yet-started MLB game) returned real OPEN MONEY_LINE and
TOTAL_POINTS_(OVER/UNDER) markets alongside dozens of real per-inning
markets -- no real "First 5 Innings" cumulative market was found on
this page, so this provider covers moneyline + full-game total only;
F5 stays sourced from TheOddsApiMlbTeamMarketProvider when a real key
is configured (see run_mlb_same_game_daily.py's fallback wiring).

WHY THIS EXISTS: TheOddsApiMlbTeamMarketProvider needs a real
THE_ODDS_API_KEY secret this repo's CI has never actually had configured
-- every real run of the same-game combo pipeline reported
`missing_credentials` and priced zero real combos. This provider needs
no key at all, so it is the real primary source for moneyline/total;
the-odds-api stays wired as a real fallback for whenever a key IS
configured, and for the F5 market this free source doesn't expose.

Output row shape matches TheOddsApiMlbTeamMarketProvider's rows exactly
(same `target`/`home_team`/`away_team`/`sportsbook`/`line`/`over_price`/
`under_price`/`home_moneyline`/`away_moneyline` keys) so select_mlb_
same_game_bets.py consumes either source identically, with no changes
to the selection layer.
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_URL = "https://api.sportsbook.fanduel.com"
CONTENT_PATH = "/sbapi/content-managed-page"
EVENT_PATH = "/sbapi/event-page"
PUBLIC_WEB_APPLICATION_KEY = "FhMFpcPWXMeyZxOx"
MLB_EVENT_TYPE_ID = 7511
TEAM_MARKET_TYPES = {"MONEY_LINE": "moneyline", "TOTAL_POINTS_(OVER/UNDER)": "game_total"}


class FanduelPublicMlbTeamMarketProvider:
    """Collects real FanDuel moneyline + full-game-total prices, no auth."""

    def __init__(
        self,
        *,
        content_payload: Optional[dict[str, Any]] = None,
        event_payloads: Optional[dict[str, dict[str, Any]]] = None,
        now: Optional[datetime] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.content_payload = content_payload
        self.event_payloads = event_payloads or {}
        self.now = now
        self.sleep_fn = sleep_fn
        self.region = str(os.environ.get("MLB_FANDUEL_REGION", "NJ")).strip().upper()
        self.public_key = str(os.environ.get("MLB_FANDUEL_PUBLIC_APP_KEY", PUBLIC_WEB_APPLICATION_KEY)).strip()
        self.enabled = os.environ.get("MLB_FANDUEL_PUBLIC_ENABLED", "1") != "0"
        self.timeout = float(os.environ.get("MLB_FANDUEL_TIMEOUT_SECONDS", "20"))
        self.max_retries = max(0, int(os.environ.get("MLB_FANDUEL_MAX_RETRIES", "2")))
        self.rate_limit_rps = max(0.25, float(os.environ.get("MLB_FANDUEL_RATE_LIMIT_RPS", "2")))
        self.max_events = max(1, int(os.environ.get("MLB_FANDUEL_MAX_EVENTS", "20")))
        self._last_request_at = 0.0
        self._accounting: dict[str, int] = {}

    def validate_config(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "message": "FanDuel public team-market adapter is disabled"}
        if not re.fullmatch(r"[A-Z]{2}", self.region):
            return {"status": "source_invalid_config", "message": "MLB_FANDUEL_REGION must be a two-letter code"}
        if not self.public_key:
            return {"status": "source_invalid_config", "message": "FanDuel public web application key is empty"}
        return {"status": "ok"}

    def get_accounting(self) -> dict[str, int]:
        return dict(self._accounting)

    def collect_team_market_odds(self) -> dict[str, Any]:
        config = self.validate_config()
        if config["status"] != "ok":
            return config
        self._accounting = {"events_discovered": 0, "events_requested": 0, "rows_collected": 0, "malformed_rows_rejected": 0}
        try:
            content = self.content_payload or self._request_json(
                CONTENT_PATH, {"page": "CUSTOM", "customPageId": "mlb", "timezone": "America/New_York"}
            )
            events = self._discover_events(content)
            self._accounting["events_discovered"] = len(events)
            if not events:
                return {"status": "no_props", "message": "FanDuel returned no upcoming MLB game events", "accounting": self._accounting}

            observed_at = self._utc_now().isoformat()
            rows: list[dict[str, Any]] = []
            for event in events[: self.max_events]:
                event_id = str(event["eventId"])
                self._accounting["events_requested"] += 1
                fixture = self.event_payloads.get(event_id)
                payload = fixture if fixture is not None else self._request_json(EVENT_PATH, {"eventId": event_id})
                rows.extend(self._extract_event_rows(payload, event, observed_at))

            self._accounting["rows_collected"] = len(rows)
            if not rows:
                return {"status": "no_props", "message": "FanDuel MLB games contained no real open team markets", "accounting": self._accounting}
            return {
                "status": "success", "odds": rows, "accounting": self._accounting,
                "cost_profile": "anonymous_public_read_only_no_subscription",
            }
        except requests.Timeout:
            return {"status": "source_timeout", "message": "FanDuel public feed timed out", "accounting": self._accounting}
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            status = "rate_limited" if status_code == 429 else "source_blocked" if status_code in {401, 403} else "api_error"
            return {"status": status, "message": f"FanDuel public feed returned HTTP {status_code}", "accounting": self._accounting}
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            return {
                "status": "schema_drift" if isinstance(exc, (ValueError, TypeError, KeyError)) else "api_error",
                "message": str(exc)[:200], "accounting": self._accounting,
            }

    def _utc_now(self) -> datetime:
        current = self.now or datetime.now(timezone.utc)
        return current if current.tzinfo else current.replace(tzinfo=timezone.utc)

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        query = {"_ak": self.public_key, **params}
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; NBA-Analytics/1.0; +read-only-market-data)",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://sportsbook.fanduel.com",
            "Referer": "https://sportsbook.fanduel.com/",
            "x-sportsbook-region": self.region,
        }
        for attempt in range(self.max_retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            minimum_interval = 1.0 / self.rate_limit_rps
            if self._last_request_at and elapsed < minimum_interval:
                self.sleep_fn(minimum_interval - elapsed)
            response = requests.get(url, params=query, headers=headers, timeout=self.timeout)
            self._last_request_at = time.monotonic()
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("FanDuel response is not a JSON object")
                return payload
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.5 * (2 ** attempt)
            self.sleep_fn(min(delay, 8.0))
        raise ValueError("FanDuel retry loop ended unexpectedly")

    def _discover_events(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        attachments = payload.get("attachments")
        if not isinstance(attachments, dict) or not isinstance(attachments.get("events"), dict):
            raise ValueError("FanDuel content response is missing attachments.events")
        current = self._utc_now()
        horizon = current + timedelta(hours=48)
        events: list[dict[str, Any]] = []
        for event in attachments["events"].values():
            if not isinstance(event, dict) or int(event.get("eventTypeId") or 0) != MLB_EVENT_TYPE_ID:
                continue
            if " @ " not in str(event.get("name") or ""):
                continue
            starts_at = self._parse_iso(event.get("openDate"))
            if starts_at is None or starts_at <= current or starts_at > horizon:
                continue
            events.append(event)
        return sorted(events, key=lambda row: str(row.get("openDate") or ""))

    @staticmethod
    def _parse_iso(value: Any) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    def _extract_event_rows(self, payload: dict[str, Any], discovered_event: dict[str, Any], observed_at: str) -> list[dict[str, Any]]:
        attachments = payload.get("attachments")
        if not isinstance(attachments, dict) or not isinstance(attachments.get("markets"), dict):
            raise ValueError("FanDuel event response is missing attachments.markets")
        event_id = str(discovered_event["eventId"])
        event = (attachments.get("events") or {}).get(event_id, discovered_event)
        if bool(event.get("inPlay")):
            return []
        away_team, home_team = self._parse_event_teams(str(event.get("name") or discovered_event.get("name") or ""))
        rows: list[dict[str, Any]] = []
        for market in attachments["markets"].values():
            if not isinstance(market, dict):
                continue
            target = TEAM_MARKET_TYPES.get(str(market.get("marketType") or "").strip())
            if target is None:
                continue
            if (
                str(market.get("marketStatus") or "").upper() != "OPEN"
                or bool(market.get("inPlay"))
                or str(market.get("eventId") or "") != event_id
            ):
                continue
            row = self._market_row(market, target=target, event_id=event_id, home_team=home_team, away_team=away_team, observed_at=observed_at)
            if row is not None:
                rows.append(row)
        return rows

    def _market_row(self, market: dict[str, Any], *, target: str, event_id: str, home_team: str, away_team: str, observed_at: str) -> Optional[dict[str, Any]]:
        runners = market.get("runners") or []
        base_row = {
            "source": "fanduel_public", "sportsbook": "fanduel", "event_id": event_id,
            "external_event_id": event_id, "home_team": home_team, "away_team": away_team,
            "league": "MLB", "target": target, "observed_at_utc": observed_at, "source_updated_at_utc": observed_at,
        }
        if target == "moneyline":
            home_price = away_price = None
            for runner in runners:
                if not isinstance(runner, dict) or str(runner.get("runnerStatus") or "").upper() != "ACTIVE":
                    continue
                price = self._price(runner)
                side = str((runner.get("result") or {}).get("type") or "").upper()
                if price is None or side not in {"HOME", "AWAY"}:
                    continue
                if side == "HOME":
                    home_price = price
                else:
                    away_price = price
            if home_price is None or away_price is None:
                self._accounting["malformed_rows_rejected"] += 1
                return None
            return {**base_row, "side": "", "line": None, "home_moneyline": home_price, "away_moneyline": away_price}

        over_price = under_price = line = None
        for runner in runners:
            if not isinstance(runner, dict) or str(runner.get("runnerStatus") or "").upper() != "ACTIVE":
                continue
            price = self._price(runner)
            side = str((runner.get("result") or {}).get("type") or "").upper()
            if price is None or side not in {"OVER", "UNDER"}:
                continue
            try:
                handicap = float(runner.get("handicap"))
            except (TypeError, ValueError):
                continue
            line = handicap
            if side == "OVER":
                over_price = price
            else:
                under_price = price
        if over_price is None or under_price is None or line is None:
            self._accounting["malformed_rows_rejected"] += 1
            return None
        return {**base_row, "side": "", "line": line, "over_price": over_price, "under_price": under_price}

    @staticmethod
    def _price(runner: dict[str, Any]) -> Optional[int]:
        try:
            return int(float(((runner.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get("americanOddsInt")))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_event_teams(name: str) -> tuple[str, str]:
        if " @ " not in name:
            return "", ""
        away, home = name.split(" @ ", 1)
        clean = lambda value: re.sub(r"\s+\([^)]*\)\s*$", "", value).strip()
        return clean(away), clean(home)
