"""Current NFL event and player-prop acquisition from The Odds API."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


SPORT_KEY = "americanfootball_nfl"
MARKET_KEYS = (
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
)
TARGET_BY_MARKET = {
    "player_pass_yds": "passing",
    "player_rush_yds": "rushing",
    "player_reception_yds": "receiving",
}


def _request_json(
    session: requests.Session,
    path: str,
    params: dict[str, Any],
    *,
    max_retries: int = 4,
) -> tuple[Any, dict[str, str]]:
    for attempt in range(max_retries + 1):
        response = session.get(
            f"https://api.the-odds-api.com{path}", params=params, timeout=45
        )
        if response.ok:
            return response.json(), {
                key.lower(): value for key, value in response.headers.items()
            }
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
            try:
                body = response.json()
                reason = body.get("error_code") or body.get("message") or "unknown"
            except Exception:
                reason = "unknown"
            raise RuntimeError(
                f"The Odds API returned HTTP {response.status_code} ({reason})."
            )
        retry_after = response.headers.get("Retry-After")
        delay = (
            float(retry_after)
            if retry_after and retry_after.replace(".", "", 1).isdigit()
            else 2**attempt
        )
        time.sleep(min(delay, 30.0))
    raise AssertionError("unreachable")


def flatten_event_odds(
    event: dict[str, Any], *, fetched_at_utc: str
) -> list[dict[str, Any]]:
    """Retain every complete two-sided book/line observation in one event."""

    rows: list[dict[str, Any]] = []
    for bookmaker in event.get("bookmakers") or []:
        bookmaker_key = str(bookmaker.get("key") or "").strip().lower()
        if not bookmaker_key:
            continue
        for market in bookmaker.get("markets") or []:
            market_key = str(market.get("key") or "").strip()
            target = TARGET_BY_MARKET.get(market_key)
            if target is None:
                continue
            grouped: dict[tuple[str, float], dict[str, Any]] = {}
            for outcome in market.get("outcomes") or []:
                player = outcome.get("description") or outcome.get("participant")
                point = outcome.get("point")
                side = str(outcome.get("name") or "").strip().lower()
                if not player or point is None or side not in {"over", "under"}:
                    continue
                try:
                    line = float(point)
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                key = (str(player).strip(), line)
                row = grouped.setdefault(
                    key,
                    {
                        "event_id": str(event.get("id") or ""),
                        "commence_time_utc": event.get("commence_time"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "player": str(player).strip(),
                        "market": market_key,
                        "target": target,
                        "line": line,
                        "bookmaker": bookmaker_key,
                        "bookmaker_title": bookmaker.get("title") or bookmaker_key,
                        "over_price": None,
                        "under_price": None,
                        "snapshot_time_utc": market.get("last_update")
                        or fetched_at_utc,
                        "fetched_at_utc": fetched_at_utc,
                        "source": "the_odds_api_live",
                    },
                )
                row[f"{side}_price"] = price
            rows.extend(
                row
                for row in grouped.values()
                if row["over_price"] is not None and row["under_price"] is not None
            )
    return rows


def fetch_live_slate(
    *,
    api_key: str,
    commence_from_utc: str,
    commence_to_utc: str,
    regions: str = "us",
    markets: tuple[str, ...] = MARKET_KEYS,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key.strip():
        raise ValueError("A non-empty The Odds API key is required.")
    active_session = session or requests.Session()
    active_session.headers.update(
        {"Accept": "application/json", "User-Agent": "NFL-Predictor/2.0"}
    )
    common = {
        "apiKey": api_key.strip(),
        "dateFormat": "iso",
        "commenceTimeFrom": commence_from_utc,
        "commenceTimeTo": commence_to_utc,
    }
    events, event_headers = _request_json(
        active_session, f"/v4/sports/{SPORT_KEY}/events", common
    )
    if not isinstance(events, list):
        raise ValueError("The Odds API events response must be a list.")

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    event_payloads: list[dict[str, Any]] = []
    quota_headers = event_headers
    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        payload, quota_headers = _request_json(
            active_session,
            f"/v4/sports/{SPORT_KEY}/events/{event_id}/odds",
            {
                "apiKey": api_key.strip(),
                "regions": regions,
                "markets": ",".join(markets),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )
        if not isinstance(payload, dict):
            continue
        event_payloads.append(payload)
        rows.extend(flatten_event_odds(payload, fetched_at_utc=fetched_at))

    raw_hash = hashlib.sha256(
        json.dumps(event_payloads, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "provider": "the_odds_api",
        "sport_key": SPORT_KEY,
        "fetched_at_utc": fetched_at,
        "events_discovered": len(events),
        "events_with_odds": len(event_payloads),
        "complete_two_sided_rows": len(rows),
        "markets": list(markets),
        "regions": regions,
        "raw_source_sha256": raw_hash,
        "quota": {
            "remaining": quota_headers.get("x-requests-remaining"),
            "used": quota_headers.get("x-requests-used"),
            "last": quota_headers.get("x-requests-last"),
        },
    }
    return rows, audit


def load_fixture_slate(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("observations"), list):
        audit = dict(payload.get("audit") or {})
        audit["replayed_from_snapshot"] = True
        audit["snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return list(payload["observations"]), audit
    events = payload if isinstance(payload, list) else payload.get("events", [])
    fetched_at = (
        payload.get("fetched_at_utc")
        if isinstance(payload, dict)
        else None
    ) or datetime.now(timezone.utc).isoformat()
    rows = [
        row
        for event in events
        for row in flatten_event_odds(event, fetched_at_utc=fetched_at)
    ]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, {
        "provider": "fixture",
        "sport_key": SPORT_KEY,
        "fetched_at_utc": fetched_at,
        "events_discovered": len(events),
        "events_with_odds": len(events),
        "complete_two_sided_rows": len(rows),
        "markets": list(MARKET_KEYS),
        "regions": "fixture",
        "raw_source_sha256": digest,
        "quota": {},
    }


def write_complete_slate(path: Path, rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "audit": audit, "observations": rows}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
