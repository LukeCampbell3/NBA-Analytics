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
SPORTSGAMEODDS_API_URL = "https://api.sportsgameodds.com/v2/events"
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
# Real team-level markets (moneyline winner, game total) -- a genuinely
# different real shape from the player-prop markets above: h2h outcomes
# are named after the two real teams with no line/point at all, and
# totals outcomes carry no player identity. flatten_event_odds() above
# is built specifically for two-sided PLAYER over/under markets and
# structurally cannot parse either shape (requires a player name AND a
# numeric point on every outcome) -- see flatten_event_team_market_odds
# below instead. Spread is deliberately not included yet: the product
# ask was moneyline + total; spread can be added the same way later.
TEAM_MARKET_KEYS = ("h2h", "totals")
SPORTSGAMEODDS_MARKETS = {
    "passing_yards": "player_pass_yds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
}
SPORTSGAMEODDS_ODD_IDS = tuple(
    f"{stat}-PLAYER_ID-game-ou-over" for stat in SPORTSGAMEODDS_MARKETS
)
SPORTSGAMEODDS_BOOKMAKERS = (
    "bet365",
    "betmgm",
    "caesars",
    "draftkings",
    "fanduel",
    "fanatics",
)


def _request_json(
    session: requests.Session,
    path: str,
    params: dict[str, Any],
    *,
    max_retries: int = 4,
    base_url: str = "https://api.the-odds-api.com",
) -> tuple[Any, dict[str, str]]:
    for attempt in range(max_retries + 1):
        response = session.get(
            f"{base_url}{path}", params=params, timeout=45
        )
        if response.ok:
            return response.json(), {
                key.lower(): value for key, value in response.headers.items()
            }
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
            try:
                body = response.json()
                reason = (
                    body.get("error_code")
                    or body.get("error")
                    or body.get("message")
                    or "unknown"
                )
            except Exception:
                reason = "unknown"
            raise RuntimeError(
                f"Odds provider returned HTTP {response.status_code} ({reason})."
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


def flatten_event_team_market_odds(
    event: dict[str, Any], *, fetched_at_utc: str
) -> list[dict[str, Any]]:
    """Real team-level markets for one event: moneyline (h2h) and game
    total (totals). A genuinely different real shape from
    flatten_event_odds's player-prop rows -- h2h outcomes are the two
    real team names with no line, totals outcomes have no player
    identity -- so this is a separate function rather than a branch
    bolted onto the player-prop one, to keep the already-proven
    player-prop path untouched.

    Retains only a complete real observation per (market, book): h2h
    needs both real team prices; totals needs both real Over and Under
    prices at the same real line."""
    home_team = str(event.get("home_team") or "").strip()
    away_team = str(event.get("away_team") or "").strip()
    rows: list[dict[str, Any]] = []

    for bookmaker in event.get("bookmakers") or []:
        bookmaker_key = str(bookmaker.get("key") or "").strip().lower()
        if not bookmaker_key:
            continue
        for market in bookmaker.get("markets") or []:
            market_key = str(market.get("key") or "").strip()
            if market_key not in TEAM_MARKET_KEYS:
                continue
            snapshot_time = market.get("last_update") or fetched_at_utc
            base_row = {
                "event_id": str(event.get("id") or ""),
                "commence_time_utc": event.get("commence_time"),
                "home_team": home_team,
                "away_team": away_team,
                "market": market_key,
                "bookmaker": bookmaker_key,
                "bookmaker_title": bookmaker.get("title") or bookmaker_key,
                "snapshot_time_utc": snapshot_time,
                "fetched_at_utc": fetched_at_utc,
                "source": "the_odds_api_live",
            }

            if market_key == "h2h":
                prices: dict[str, float] = {}
                for outcome in market.get("outcomes") or []:
                    name = str(outcome.get("name") or "").strip()
                    try:
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if name and name in {home_team, away_team}:
                        prices[name] = price
                if home_team in prices and away_team in prices:
                    rows.append(
                        {
                            **base_row,
                            "target": "moneyline",
                            "line": None,
                            "home_moneyline": prices[home_team],
                            "away_moneyline": prices[away_team],
                        }
                    )
            elif market_key == "totals":
                grouped: dict[float, dict[str, float]] = {}
                for outcome in market.get("outcomes") or []:
                    side = str(outcome.get("name") or "").strip().lower()
                    point = outcome.get("point")
                    if side not in {"over", "under"} or point is None:
                        continue
                    try:
                        line = float(point)
                        price = float(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue
                    grouped.setdefault(line, {})[side] = price
                for line, sides in grouped.items():
                    if "over" in sides and "under" in sides:
                        rows.append(
                            {
                                **base_row,
                                "target": "game_total",
                                "line": line,
                                "over_price": sides["over"],
                                "under_price": sides["under"],
                            }
                        )
    return rows


def fetch_live_team_market_slate(
    *,
    api_key: str,
    commence_from_utc: str,
    commence_to_utc: str,
    regions: str = "us",
    markets: tuple[str, ...] = TEAM_MARKET_KEYS,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Real live moneyline/game-total capture -- a separate function from
    fetch_live_slate (not a parameterization of it) so the already-proven
    player-prop path above is never touched by this addition. Same
    events/odds request shape and audit contract as fetch_live_slate,
    just wired to TEAM_MARKET_KEYS and flatten_event_team_market_odds."""
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
        rows.extend(flatten_event_team_market_odds(payload, fetched_at_utc=fetched_at))

    raw_hash = hashlib.sha256(
        json.dumps(event_payloads, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "provider": "the_odds_api",
        "sport_key": SPORT_KEY,
        "fetched_at_utc": fetched_at,
        "events_discovered": len(events),
        "events_with_odds": len(event_payloads),
        "complete_team_market_rows": len(rows),
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


def flatten_sportsgameodds_event(
    event: dict[str, Any], *, fetched_at_utc: str
) -> list[dict[str, Any]]:
    """Normalize complete current two-sided SportsGameOdds book/line pairs."""

    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    if bool(status.get("live")) or bool(status.get("started")):
        return []
    commence = status.get("startsAt") or event.get("startsAt")
    players = event.get("players") if isinstance(event.get("players"), dict) else {}
    teams = event.get("teams") if isinstance(event.get("teams"), dict) else {}

    def team_name(side: str) -> str | None:
        team = teams.get(side) if isinstance(teams.get(side), dict) else {}
        names = team.get("names") if isinstance(team.get("names"), dict) else {}
        return names.get("long") or names.get("medium") or names.get("short") or team.get("name")

    grouped: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    updates: dict[tuple[str, str, str, float], list[str]] = {}
    odds = event.get("odds") if isinstance(event.get("odds"), dict) else {}
    for odd in odds.values():
        if not isinstance(odd, dict):
            continue
        stat_id = str(odd.get("statID") or "")
        target_market = SPORTSGAMEODDS_MARKETS.get(stat_id)
        side = str(odd.get("sideID") or "").lower()
        if (
            target_market is None
            or str(odd.get("periodID") or "") != "game"
            or str(odd.get("betTypeID") or "") != "ou"
            or side not in {"over", "under"}
            or bool(odd.get("started"))
        ):
            continue
        player_id = str(odd.get("playerID") or odd.get("statEntityID") or "")
        player = players.get(player_id) if isinstance(players.get(player_id), dict) else {}
        player_name = player.get("name") or player.get("display")
        if not player_name:
            continue
        by_book = odd.get("byBookmaker") if isinstance(odd.get("byBookmaker"), dict) else {}
        for bookmaker, offer in by_book.items():
            if not isinstance(offer, dict) or offer.get("available") is False:
                continue
            try:
                line = float(offer.get("overUnder"))
                price = float(offer.get("odds"))
            except (TypeError, ValueError):
                continue
            if price == 0:
                continue
            book = str(bookmaker).strip().lower()
            key = (player_id, target_market, book, line)
            row = grouped.setdefault(
                key,
                {
                    "event_id": str(event.get("eventID") or ""),
                    "commence_time_utc": commence,
                    "home_team": team_name("home"),
                    "away_team": team_name("away"),
                    "player": str(player_name).strip(),
                    "provider_player_id": player_id,
                    "market": target_market,
                    "target": TARGET_BY_MARKET[target_market],
                    "line": line,
                    "bookmaker": book,
                    "bookmaker_title": book,
                    "over_price": None,
                    "under_price": None,
                    "snapshot_time_utc": fetched_at_utc,
                    "fetched_at_utc": fetched_at_utc,
                    "source": "sportsgameodds_live",
                },
            )
            row[f"{side}_price"] = price
            updated = offer.get("lastUpdatedAt")
            if updated:
                updates.setdefault(key, []).append(str(updated))

    rows: list[dict[str, Any]] = []
    for key, row in grouped.items():
        if row["over_price"] is None or row["under_price"] is None:
            continue
        if updates.get(key):
            parsed = [
                value
                for value in (
                    datetime.fromisoformat(item.replace("Z", "+00:00"))
                    for item in updates[key]
                )
                if value.tzinfo is not None
            ]
            if parsed:
                row["snapshot_time_utc"] = min(parsed).astimezone(timezone.utc).isoformat()
        rows.append(row)
    return rows


def fetch_sportsgameodds_live_slate(
    *,
    api_key: str,
    commence_from_utc: str,
    commence_to_utc: str,
    bookmakers: tuple[str, ...] | None = None,
    session: requests.Session | None = None,
    max_retries: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key.strip():
        raise ValueError("A non-empty SportsGameOdds API key is required.")
    active_session = session or requests.Session()
    active_session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "NFL-Predictor/2.0",
            "x-api-key": api_key.strip(),
        }
    )
    base_params: dict[str, Any] = {
        "leagueID": "NFL",
        "oddsAvailable": "true",
        "started": "false",
        "live": "false",
        "startsAfter": commence_from_utc,
        "startsBefore": commence_to_utc,
        "oddID": ",".join(SPORTSGAMEODDS_ODD_IDS),
        "includeOpposingOdds": "true",
        "includeAltLines": "false",
        "limit": 50,
    }
    if bookmakers:
        base_params["bookmakerID"] = ",".join(bookmakers)
    payloads: list[dict[str, Any]] = []
    headers: dict[str, str] = {}
    cursor: str | None = None
    while True:
        params = dict(base_params)
        if cursor:
            params["cursor"] = cursor
        payload, headers = _request_json(
            active_session,
            "/v2/events",
            params,
            max_retries=max_retries,
            base_url="https://api.sportsgameodds.com",
        )
        if not isinstance(payload, dict) or payload.get("success") is False:
            raise RuntimeError("SportsGameOdds returned an unsuccessful payload.")
        payloads.append(payload)
        cursor_value = payload.get("nextCursor")
        cursor = str(cursor_value) if cursor_value else None
        if not cursor:
            break

    fetched_at = datetime.now(timezone.utc).isoformat()
    events = [event for payload in payloads for event in payload.get("data", [])]
    rows = [
        row
        for event in events
        for row in flatten_sportsgameodds_event(event, fetched_at_utc=fetched_at)
    ]
    raw_hash = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return rows, {
        "provider": "sportsgameodds",
        "status": "success" if rows else "no_props",
        "sport_key": "NFL",
        "fetched_at_utc": fetched_at,
        "pages_fetched": len(payloads),
        "events_discovered": len(events),
        "events_with_odds": sum(bool(event.get("odds")) for event in events),
        "complete_two_sided_rows": len(rows),
        "markets": list(SPORTSGAMEODDS_ODD_IDS),
        "bookmakers": list(bookmakers or []),
        "raw_source_sha256": raw_hash,
        "rate_limit_remaining": headers.get("x-ratelimit-remaining"),
    }


def fetch_available_live_slate(
    *,
    sportsgameodds_api_key: str | None,
    the_odds_api_key: str | None,
    commence_from_utc: str,
    commence_to_utc: str,
    regions: str = "us",
    provider_priority: tuple[str, ...] = ("sportsgameodds", "the_odds_api"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for provider in provider_priority:
        try:
            if provider == "sportsgameodds":
                if not sportsgameodds_api_key:
                    attempts.append({"provider": provider, "status": "missing_credentials"})
                    continue
                rows, audit = fetch_sportsgameodds_live_slate(
                    api_key=sportsgameodds_api_key,
                    commence_from_utc=commence_from_utc,
                    commence_to_utc=commence_to_utc,
                )
            elif provider == "the_odds_api":
                if not the_odds_api_key:
                    attempts.append({"provider": provider, "status": "missing_credentials"})
                    continue
                rows, audit = fetch_live_slate(
                    api_key=the_odds_api_key,
                    commence_from_utc=commence_from_utc,
                    commence_to_utc=commence_to_utc,
                    regions=regions,
                )
                audit["status"] = "success" if rows else "no_props"
            else:
                attempts.append({"provider": provider, "status": "unsupported"})
                continue
        except Exception as error:
            attempts.append(
                {"provider": provider, "status": "api_error", "message": str(error)[:300]}
            )
            continue
        attempts.append(
            {
                "provider": provider,
                "status": audit.get("status", "success"),
                "complete_two_sided_rows": len(rows),
                "events_discovered": audit.get("events_discovered"),
                "events_with_odds": audit.get("events_with_odds"),
            }
        )
        if rows:
            return rows, {**audit, "provider_attempts": attempts}

    statuses = {str(item.get("status")) for item in attempts}
    return [], {
        "provider": "provider_chain",
        "status": "missing_credentials" if statuses == {"missing_credentials"} else "no_props",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_two_sided_rows": 0,
        "provider_attempts": attempts,
        "raw_source_sha256": None,
    }


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
