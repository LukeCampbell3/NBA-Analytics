#!/usr/bin/env python3
"""Anonymous read-only FanDuel MLB player-prop provider."""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odds_contract import ensure_contract, stable_hash


BASE_URL = "https://api.sportsbook.fanduel.com"
CONTENT_PATH = "/sbapi/content-managed-page"
EVENT_PATH = "/sbapi/event-page"
PUBLIC_WEB_APPLICATION_KEY = "FhMFpcPWXMeyZxOx"
MLB_EVENT_TYPE_ID = 7511
PROP_TABS = ("batter-props", "pitcher-props")
MARKET_ID_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
THRESHOLD_PATTERN = re.compile(r"^(?:PLAYER_)?TO_RECORD_(\d+)\+_(HITS|TOTAL_BASES|RUNS|RBIS)$")
PITCHER_TOTAL_K_PATTERN = re.compile(r"^PITCHER_[A-Z]_TOTAL_STRIKEOUTS$")
PITCHER_ALT_K_PATTERN = re.compile(r"^PITCHER_[A-Z]_STRIKEOUTS$")
PITCHER_K_RUNNER_PATTERN = re.compile(r"^(.+?)\s+(Over|Under)$", re.IGNORECASE)
PITCHER_ALT_K_RUNNER_PATTERN = re.compile(r"^(.+?)\s+(\d+)\+\s+Strikeouts$", re.IGNORECASE)
DIRECT_THRESHOLD_MARKETS = {
    "PLAYER_TO_RECORD_A_HIT": ("batter_hits", 0.5),
    "TO_RECORD_A_RUN": ("batter_runs_scored", 0.5),
    "TO_RECORD_AN_RBI": ("batter_rbis", 0.5),
    "TO_HIT_A_HOME_RUN": ("batter_home_runs", 0.5),
}
THRESHOLD_MARKETS = {
    "HITS": "batter_hits",
    "TOTAL_BASES": "batter_total_bases",
    "RUNS": "batter_runs_scored",
    "RBIS": "batter_rbis",
}


class FanduelPublicMlbProvider:
    """Collect exact FanDuel prop prices and betslip identifiers without auth."""

    def __init__(
        self,
        *,
        content_payload: dict[str, Any] | None = None,
        event_payloads: dict[tuple[str, str], dict[str, Any]] | None = None,
        now: datetime | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.content_payload = content_payload
        self.event_payloads = event_payloads or {}
        self.now = now
        self.sleep_fn = sleep_fn
        self.region = str(os.environ.get("MLB_FANDUEL_REGION", "NJ")).strip().upper()
        self.public_key = str(
            os.environ.get("MLB_FANDUEL_PUBLIC_APP_KEY", PUBLIC_WEB_APPLICATION_KEY)
        ).strip()
        self.enabled = os.environ.get("MLB_FANDUEL_PUBLIC_ENABLED", "1") != "0"
        self.timeout = float(os.environ.get("MLB_FANDUEL_TIMEOUT_SECONDS", "20"))
        self.max_retries = max(0, int(os.environ.get("MLB_FANDUEL_MAX_RETRIES", "2")))
        self.rate_limit_rps = max(0.25, float(os.environ.get("MLB_FANDUEL_RATE_LIMIT_RPS", "2")))
        self.max_events = max(1, int(os.environ.get("MLB_FANDUEL_MAX_EVENTS", "20")))
        self._last_request_at = 0.0
        self._accounting: dict[str, int] = {}

    def validate_config(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "message": "FanDuel public adapter is disabled"}
        if not re.fullmatch(r"[A-Z]{2}", self.region):
            return {"status": "source_invalid_config", "message": "MLB_FANDUEL_REGION must be a two-letter code"}
        if not self.public_key:
            return {"status": "source_invalid_config", "message": "FanDuel public web application key is empty"}
        return {"status": "ok"}

    def get_accounting(self) -> dict[str, int]:
        return dict(self._accounting)

    def collect_player_props(self) -> dict[str, Any]:
        config = self.validate_config()
        if config["status"] != "ok":
            return config
        self._accounting = {
            "events_discovered": 0,
            "events_requested": 0,
            "event_tabs_requested": 0,
            "markets_seen": 0,
            "supported_markets": 0,
            "rows_collected": 0,
            "inactive_rows_rejected": 0,
            "malformed_rows_rejected": 0,
        }
        try:
            content = self.content_payload or self._request_json(
                CONTENT_PATH,
                {
                    "page": "CUSTOM",
                    "customPageId": "mlb",
                    "timezone": "America/New_York",
                },
            )
            events = self._discover_events(content)
            self._accounting["events_discovered"] = len(events)
            if not events:
                return {
                    "status": "no_props",
                    "message": "FanDuel returned no upcoming MLB game events",
                    "accounting": self._accounting,
                }

            observed_at = self._utc_now().isoformat()
            rows: list[dict[str, Any]] = []
            for event in events[: self.max_events]:
                event_id = str(event["eventId"])
                self._accounting["events_requested"] += 1
                for tab in PROP_TABS:
                    fixture = self.event_payloads.get((event_id, tab))
                    payload = fixture if fixture is not None else self._request_json(
                        EVENT_PATH,
                        {"eventId": event_id, "tab": tab},
                    )
                    self._accounting["event_tabs_requested"] += 1
                    rows.extend(self._extract_event_rows(payload, event, observed_at))

            deduplicated = {
                str(row["source_market_id"]): row
                for row in rows
                if str(row.get("source_market_id") or "").strip()
            }
            accepted = list(deduplicated.values())
            self._accounting["rows_collected"] = len(accepted)
            if not accepted:
                return {
                    "status": "no_props",
                    "message": "FanDuel MLB games contained no supported active player props",
                    "accounting": self._accounting,
                }
            return {
                "status": "success",
                "odds": accepted,
                "events_checked": self._accounting["events_requested"],
                "accounting": self._accounting,
                "cost_profile": "anonymous_public_read_only_no_subscription",
            }
        except requests.Timeout:
            return {"status": "source_timeout", "message": "FanDuel public feed timed out", "accounting": self._accounting}
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            status = "rate_limited" if status_code == 429 else "source_blocked" if status_code in {401, 403} else "api_error"
            return {
                "status": status,
                "message": f"FanDuel public feed returned HTTP {status_code}",
                "accounting": self._accounting,
            }
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            return {
                "status": "schema_drift" if isinstance(exc, (ValueError, TypeError, KeyError)) else "api_error",
                "message": str(exc)[:200],
                "accounting": self._accounting,
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
            starts_at = pd.to_datetime(event.get("openDate"), utc=True, errors="coerce")
            if pd.isna(starts_at):
                continue
            starts = starts_at.to_pydatetime()
            if starts <= current or starts > horizon:
                continue
            events.append(event)
        return sorted(events, key=lambda row: str(row.get("openDate") or ""))

    def _extract_event_rows(
        self,
        payload: dict[str, Any],
        discovered_event: dict[str, Any],
        observed_at: str,
    ) -> list[dict[str, Any]]:
        attachments = payload.get("attachments")
        if not isinstance(attachments, dict) or not isinstance(attachments.get("markets"), dict):
            raise ValueError("FanDuel event response is missing attachments.markets")
        event_id = str(discovered_event["eventId"])
        event = (attachments.get("events") or {}).get(event_id, discovered_event)
        if bool(event.get("inPlay")):
            return []
        away_team, home_team = self._parse_event_teams(str(event.get("name") or discovered_event.get("name") or ""))
        game_start = str(event.get("openDate") or discovered_event.get("openDate") or "")
        rows: list[dict[str, Any]] = []
        for market in attachments["markets"].values():
            if not isinstance(market, dict):
                continue
            self._accounting["markets_seen"] += 1
            if (
                str(market.get("marketStatus") or "").upper() != "OPEN"
                or bool(market.get("inPlay"))
                or str(market.get("eventId") or "") != event_id
            ):
                continue
            descriptor = self._market_descriptor(market)
            if descriptor is None:
                continue
            self._accounting["supported_markets"] += 1
            market_type, fixed_line, mode = descriptor
            market_id = str(market.get("marketId") or "").strip()
            if not MARKET_ID_PATTERN.fullmatch(market_id):
                self._accounting["malformed_rows_rejected"] += len(market.get("runners") or [])
                continue
            for runner in market.get("runners") or []:
                row = self._runner_row(
                    runner=runner,
                    mode=mode,
                    fixed_line=fixed_line,
                    market_type=market_type,
                    market_id=market_id,
                    event_id=event_id,
                    game_start=game_start,
                    home_team=home_team,
                    away_team=away_team,
                    observed_at=observed_at,
                )
                if row is not None:
                    rows.append(row)
        return rows

    @staticmethod
    def _market_descriptor(market: dict[str, Any]) -> tuple[str, float | None, str] | None:
        market_type = str(market.get("marketType") or "").strip().upper()
        direct = DIRECT_THRESHOLD_MARKETS.get(market_type)
        if direct is not None:
            return direct[0], direct[1], "batter_threshold"
        threshold = THRESHOLD_PATTERN.fullmatch(market_type)
        if threshold:
            canonical = THRESHOLD_MARKETS.get(threshold.group(2))
            if canonical:
                return canonical, float(threshold.group(1)) - 0.5, "batter_threshold"
        if PITCHER_TOTAL_K_PATTERN.fullmatch(market_type):
            return "pitcher_strikeouts", None, "pitcher_total"
        if PITCHER_ALT_K_PATTERN.fullmatch(market_type):
            return "pitcher_strikeouts", None, "pitcher_threshold"
        return None

    def _runner_row(
        self,
        *,
        runner: Any,
        mode: str,
        fixed_line: float | None,
        market_type: str,
        market_id: str,
        event_id: str,
        game_start: str,
        home_team: str,
        away_team: str,
        observed_at: str,
    ) -> dict[str, Any] | None:
        if not isinstance(runner, dict) or str(runner.get("runnerStatus") or "").upper() != "ACTIVE":
            self._accounting["inactive_rows_rejected"] += 1
            return None
        selection_id = str(runner.get("selectionId") or "").strip()
        runner_name = str(runner.get("runnerName") or "").strip()
        odds = (((runner.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get("americanOddsInt"))
        try:
            price = int(float(odds))
        except (TypeError, ValueError):
            self._accounting["malformed_rows_rejected"] += 1
            return None
        if not selection_id.isdigit() or -100 < price < 100:
            self._accounting["malformed_rows_rejected"] += 1
            return None

        player = runner_name
        side = "over"
        line = fixed_line
        if mode == "pitcher_total":
            match = PITCHER_K_RUNNER_PATTERN.fullmatch(runner_name)
            try:
                line = float(runner.get("handicap"))
            except (TypeError, ValueError):
                line = None
            if not match:
                self._accounting["malformed_rows_rejected"] += 1
                return None
            player = match.group(1).strip()
            side = match.group(2).lower()
        elif mode == "pitcher_threshold":
            match = PITCHER_ALT_K_RUNNER_PATTERN.fullmatch(runner_name)
            if not match:
                self._accounting["malformed_rows_rejected"] += 1
                return None
            player = match.group(1).strip()
            line = float(match.group(2)) - 0.5
        if not player or line is None or line < 0:
            self._accounting["malformed_rows_rejected"] += 1
            return None

        team = self._team_from_logo(runner.get("secondaryLogo"))
        opponent = away_team if team == home_team else home_team if team == away_team else ""
        deeplink = (
            f"https://sportsbook.fanduel.com/addToBetslip?marketId={market_id}"
            f"&selectionId={selection_id}"
        )
        source_market_id = f"{market_id}:{selection_id}"
        return {
            "source": "fanduel_public",
            "provider_name": "fanduel_public",
            "source_market_id": source_market_id,
            "odd_id": source_market_id,
            "sportsbook": "fanduel",
            "book": "fanduel",
            "event_id": event_id,
            "external_event_id": event_id,
            "source_event_id": event_id,
            "player_name": player,
            "player": player,
            "external_player_id": selection_id,
            "player_id_source": selection_id,
            "home_team": home_team,
            "away_team": away_team,
            "team": team,
            "opponent": opponent,
            "game_start_utc": game_start,
            "commence_time_utc": game_start,
            "league": "MLB",
            "market_type": market_type,
            "market_canonical": market_type,
            "side": side,
            "line": float(line),
            "price_american": price,
            "odds": price,
            "observed_at_utc": observed_at,
            "source_updated_at_utc": observed_at,
            "sportsbook_deeplink": deeplink,
            "raw_record_hash": stable_hash(
                {"market_id": market_id, "selection_id": selection_id, "line": line, "price": price}
            ),
        }

    @staticmethod
    def _parse_event_teams(name: str) -> tuple[str, str]:
        if " @ " not in name:
            return "", ""
        away, home = name.split(" @ ", 1)
        clean = lambda value: re.sub(r"\s+\([^)]*\)\s*$", "", value).strip()
        return clean(away), clean(home)

    @staticmethod
    def _team_from_logo(value: Any) -> str:
        match = re.search(r"/mlb/([^/?]+)\.(?:png|svg)", str(value or ""), flags=re.IGNORECASE)
        return match.group(1).replace("_", " ").title() if match else ""

    def normalize(self, raw_odds: list[dict[str, Any]]) -> pd.DataFrame:
        return ensure_contract(
            pd.DataFrame(raw_odds),
            source="fanduel_public",
            acquisition_method="anonymous_public_json",
            source_endpoint=f"{BASE_URL}{CONTENT_PATH}",
            parser_version="fanduel-public-mlb-v1",
        )
