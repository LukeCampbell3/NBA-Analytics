"""Credential-free Formula 1 race-winner market data.

Polymarket and Kalshi expose documented, unauthenticated read APIs. Their YES
asks are executable exchange prices rather than scraped display text, which is
more stable and easier to audit. Missing markets always fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any

import requests


POLYMARKET_BASE_URL = "https://gamma-api.polymarket.com"
KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


def _request_json(
    session: requests.Session,
    base_url: str,
    path: str,
    params: dict[str, Any],
    *,
    max_retries: int = 3,
) -> tuple[Any, dict[str, str]]:
    for attempt in range(max_retries + 1):
        response = session.get(f"{base_url}{path}", params=params, timeout=45)
        if response.ok:
            return response.json(), {key.lower(): value for key, value in response.headers.items()}
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
            raise RuntimeError(f"Free market provider returned HTTP {response.status_code} for {path}")
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2**attempt
        time.sleep(min(delay, 20.0))
    raise AssertionError("unreachable")


def normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_value.lower())


def american_price(decimal: float | None) -> int | None:
    if decimal is None or decimal <= 1:
        return None
    return round((decimal - 1) * 100) if decimal >= 2 else round(-100 / (decimal - 1))


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _winner_driver_from_question(question: str, group_title: str = "") -> str:
    text = str(group_title or question).strip()
    patterns = (
        r"^Will\s+(.+?)\s+win\b",
        r"^(.+?)\s+to win\b",
        r"^(.+?)\s+winner$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?-–—")
    return ""


def flatten_polymarket_event(event: dict[str, Any], fetched_at_utc: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_title = str(event.get("title") or "")
    event_text = f"{event_title} {event.get('description') or ''}".lower()
    if not ("formula 1" in event_text or "f1" in event_text or "grand prix" in event_text):
        return []
    for market in event.get("markets") or []:
        if not isinstance(market, dict) or market.get("closed") or market.get("active") is False:
            continue
        question = str(market.get("question") or "")
        group_title = str(market.get("groupItemTitle") or "")
        combined = f"{event_title} {question} {market.get('description') or ''}".lower()
        if "win" not in combined or any(blocked in combined for blocked in ("championship", "constructor", "qualifying", "sprint", "podium", "top 3", "top six", "top 6")):
            continue
        outcomes = [str(item) for item in _json_list(market.get("outcomes"))]
        prices = _json_list(market.get("outcomePrices"))
        try:
            yes_index = next(index for index, value in enumerate(outcomes) if value.lower() == "yes")
            probability = float(prices[yes_index])
        except (StopIteration, IndexError, TypeError, ValueError):
            continue
        # The current executable YES ask is preferred to a stale last trade.
        try:
            ask = float(market.get("bestAsk"))
        except (TypeError, ValueError):
            ask = probability
        if not 0 < ask < 1:
            continue
        driver = _winner_driver_from_question(question, group_title)
        if not driver:
            continue
        decimal = 1.0 / ask
        rows.append(
            {
                "event_id": str(event.get("id") or ""),
                "event_name": event_title,
                "commence_time_utc": event.get("startTime") or event.get("eventDate") or event.get("endDate"),
                "market": "race_winner",
                "driver": driver,
                "bookmaker": "polymarket",
                "bookmaker_title": "Polymarket",
                "market_probability": ask,
                "decimal_price": decimal,
                "american_price": american_price(decimal),
                "liquidity": float(market.get("liquidityNum") or market.get("liquidity") or 0),
                "snapshot_time_utc": market.get("updatedAt") or fetched_at_utc,
                "source": "polymarket_public_api",
            }
        )
    return rows


def fetch_polymarket(
    *, event_name: str, session: requests.Session | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active = session or requests.Session()
    active.headers.update({"Accept": "application/json", "User-Agent": "Prediction-Bounties-F1/1.0"})
    search, _ = _request_json(
        active,
        POLYMARKET_BASE_URL,
        "/public-search",
        {
            "q": f"Formula 1 {event_name} winner",
            "events_status": "active",
            "limit_per_type": 20,
            "keep_closed_markets": 0,
            "search_profiles": "false",
        },
    )
    candidates = search.get("events", []) if isinstance(search, dict) else []
    events: list[dict[str, Any]] = []
    for candidate in candidates:
        event_id = candidate.get("id") if isinstance(candidate, dict) else None
        if event_id is None:
            continue
        event, _ = _request_json(active, POLYMARKET_BASE_URL, f"/events/{event_id}", {})
        if isinstance(event, dict):
            events.append(event)
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [row for event in events for row in flatten_polymarket_event(event, fetched_at)]
    return rows, {
        "provider": "polymarket",
        "status": "success" if rows else "no_markets",
        "fetched_at_utc": fetched_at,
        "events_discovered": len(events),
        "observations": len(rows),
        "credential_required": False,
        "raw_source_sha256": hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest(),
    }


def _kalshi_is_f1(value: dict[str, Any]) -> bool:
    text = " ".join(
        [str(value.get("title") or ""), str(value.get("category") or ""), str(value.get("ticker") or "")]
        + [str(tag) for tag in value.get("tags") or []]
    ).lower()
    return "formula 1" in text or bool(re.search(r"\bf1\b", text))


def _kalshi_driver(market: dict[str, Any]) -> str:
    for value in (
        market.get("yes_sub_title"),
        market.get("subtitle"),
        market.get("primary_participant_key"),
        market.get("title"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        parsed = _winner_driver_from_question(text, text)
        if parsed:
            return parsed
        lower = text.lower()
        tokens = set(re.findall(r"[a-z]+", lower))
        if len(text.split()) in {2, 3} and "grand prix" not in lower and not tokens.intersection({"formula", "race", "yes", "no"}):
            return text
    return ""


def flatten_kalshi_event(event: dict[str, Any], fetched_at_utc: str) -> list[dict[str, Any]]:
    title = str(event.get("title") or "")
    if not _kalshi_is_f1(event) and not ("grand prix" in title.lower()):
        return []
    rows: list[dict[str, Any]] = []
    for market in event.get("markets") or []:
        if not isinstance(market, dict):
            continue
        text = " ".join(str(market.get(key) or "") for key in ("title", "subtitle", "yes_sub_title", "ticker")).lower()
        is_winner = "win" in text or "finish in first" in text or "winner" in title.lower()
        if not is_winner or any(blocked in text for blocked in ("championship", "constructor", "qualifying", "sprint", "podium", "top 3", "top 6")):
            continue
        driver = _kalshi_driver(market)
        try:
            ask = float(market.get("yes_ask_dollars"))
        except (TypeError, ValueError):
            try:
                ask = float(market.get("last_price_dollars"))
            except (TypeError, ValueError):
                continue
        if not driver or not 0 < ask < 1:
            continue
        decimal = 1.0 / ask
        rows.append(
            {
                "event_id": str(event.get("event_ticker") or ""),
                "event_name": title,
                "commence_time_utc": market.get("expected_expiration_time") or market.get("close_time"),
                "market": "race_winner",
                "driver": driver,
                "bookmaker": "kalshi",
                "bookmaker_title": "Kalshi",
                "market_probability": ask,
                "decimal_price": decimal,
                "american_price": american_price(decimal),
                "liquidity": float(market.get("liquidity_dollars") or 0),
                "snapshot_time_utc": market.get("updated_time") or fetched_at_utc,
                "source": "kalshi_public_api",
            }
        )
    return rows


def fetch_kalshi(
    *, event_name: str, session: requests.Session | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active = session or requests.Session()
    active.headers.update({"Accept": "application/json", "User-Agent": "Prediction-Bounties-F1/1.0"})
    series_payload, _ = _request_json(
        active, KALSHI_BASE_URL, "/series", {"category": "Sports", "include_product_metadata": "true"}
    )
    all_series = series_payload.get("series", []) if isinstance(series_payload, dict) else []
    f1_series = [series for series in all_series if isinstance(series, dict) and _kalshi_is_f1(series)]
    events: list[dict[str, Any]] = []
    target_tokens = {
        normalize_name(token)
        for token in re.findall(r"[A-Za-z0-9]+", event_name)
        if len(token) > 3 and token.lower() not in {"formula", "grand", "prix"}
    }
    for series in f1_series:
        payload, _ = _request_json(
            active,
            KALSHI_BASE_URL,
            "/events",
            {"series_ticker": series.get("ticker"), "status": "open", "with_nested_markets": "true", "limit": 200},
        )
        candidates = payload.get("events", []) if isinstance(payload, dict) else []
        for event in candidates:
            compact = normalize_name(event.get("title", ""))
            if not target_tokens or any(token in compact for token in target_tokens):
                events.append(event)
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [row for event in events for row in flatten_kalshi_event(event, fetched_at)]
    return rows, {
        "provider": "kalshi",
        "status": "success" if rows else "no_markets",
        "fetched_at_utc": fetched_at,
        "series_discovered": len(f1_series),
        "events_discovered": len(events),
        "observations": len(rows),
        "credential_required": False,
        "raw_source_sha256": hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest(),
    }


def fetch_available_odds(
    *,
    event_name: str,
    provider_priority: tuple[str, ...] = ("polymarket", "kalshi"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect every available free feed so consensus can use both exchanges."""

    attempts: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for provider in provider_priority:
        try:
            if provider == "polymarket":
                rows, audit = fetch_polymarket(event_name=event_name)
            elif provider == "kalshi":
                rows, audit = fetch_kalshi(event_name=event_name)
            else:
                attempts.append({"provider": provider, "status": "unsupported"})
                continue
        except Exception as error:
            attempts.append({"provider": provider, "status": "api_error", "message": str(error)[:300]})
            continue
        combined.extend(rows)
        audits.append(audit)
        attempts.append({"provider": provider, "status": audit.get("status"), "observations": len(rows)})
    return combined, {
        "provider": "free_exchange_consensus",
        "status": "success" if combined else "no_markets",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "observations": len(combined),
        "credential_required": False,
        "sources": audits,
        "provider_attempts": attempts,
    }


def attach_consensus_market(projections: list[dict[str, Any]], observations: list[dict[str, Any]]) -> None:
    projection_names = {normalize_name(row["driver"]): row for row in projections}
    last_names: dict[str, list[dict[str, Any]]] = {}
    for row in projections:
        last_names.setdefault(normalize_name(row["driver"].split()[-1]), []).append(row)

    offers_by_projection: dict[int, list[dict[str, Any]]] = {}
    for offer in observations:
        key = normalize_name(offer.get("driver", ""))
        projection = projection_names.get(key)
        if projection is None:
            candidates = [row for last, rows in last_names.items() if last and last in key for row in rows]
            if len(candidates) == 1:
                projection = candidates[0]
        if projection is not None:
            offers_by_projection.setdefault(id(projection), []).append(offer)

    for projection in projections:
        offers = offers_by_projection.get(id(projection), [])
        if not offers:
            projection.update({"market_probability": None, "edge": None, "best_price": None, "best_book": None, "book_count": 0})
            continue
        # YES asks are directly comparable executable probabilities. Mean one
        # observation per exchange, then use the lowest ask as the best price.
        by_exchange: dict[str, list[float]] = {}
        for offer in offers:
            by_exchange.setdefault(str(offer["bookmaker"]), []).append(float(offer["market_probability"]))
        market_probability = sum(sum(values) / len(values) for values in by_exchange.values()) / len(by_exchange)
        best = min(offers, key=lambda offer: float(offer["market_probability"]))
        projection.update(
            {
                "market_probability": market_probability,
                "edge": float(projection["win_probability"]) - market_probability,
                "best_price": best.get("american_price"),
                "best_decimal_price": best.get("decimal_price"),
                "best_book": best.get("bookmaker_title") or best.get("bookmaker"),
                "book_count": len(by_exchange),
            }
        )
