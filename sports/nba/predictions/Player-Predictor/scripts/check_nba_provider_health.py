#!/usr/bin/env python3
"""Check NBA SportsGameOdds provider health and export a JSON healthcheck summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_nba_market_props import normalize_sportsgameodds_events, resolve_api_key  # noqa: E402


def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def safe_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def request_json(base_url: str, params: dict[str, object]) -> tuple[object, dict[str, str]]:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{base_url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Player-Predictor/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = {key.lower(): value for key, value in response.headers.items()}
        return payload, headers


def _count_starts_at(events: list[dict]) -> int:
    count = 0
    for event in events:
        status = event.get("status") if isinstance(event.get("status"), dict) else {}
        starts_at = status.get("startsAt") if isinstance(status, dict) else None
        if starts_at and str(starts_at).strip():
            count += 1
    return count


def build_provider_healthcheck_report(
    *,
    api_key_visible: bool,
    request_success: bool,
    events_returned: int,
    odds_rows_returned: int,
    starts_at_available_count: int,
    side_specific_price_count: int,
    books_observed: list[str],
    failure_reason: str,
    output_path: Path,
    fetched_at_utc: str,
) -> dict[str, Any]:
    report = {
        "fetched_at_utc": fetched_at_utc,
        "api_key_visible": api_key_visible,
        "request_success": request_success,
        "events_returned": events_returned,
        "odds_rows_returned": odds_rows_returned,
        "startsAt_available_count": starts_at_available_count,
        "side_specific_price_count": side_specific_price_count,
        "books_observed": sorted(set(books_observed)),
        "failure_reason": failure_reason,
    }
    safe_write_json(output_path, report)
    return report


def fetch_sportsgameodds_events(api_key: str, event_limit: int = 50) -> tuple[list[dict], dict[str, Any]]:
    base_url = "https://api.sportsgameodds.com/v2/events"
    params: dict[str, object] = {
        "leagueID": "NBA",
        "oddsAvailable": "true",
        "limit": int(event_limit),
    }
    payload, headers = request_json(base_url, params | {"apiKey": api_key})
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected SportsGameOdds payload type: {type(payload)!r}")
    if payload.get("success") is False:
        raise RuntimeError(f"SportsGameOdds returned success=false: {payload}")
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected SportsGameOdds data payload type: {type(data)!r}")
    events = [event for event in data if isinstance(event, dict)]
    return events, headers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA SportsGameOdds provider healthcheck.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp") / f"current_games_refresh_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d')}",
        help="Directory where provider_healthcheck.json will be written.",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Override the SportsGameOdds API key.")
    parser.add_argument("--event-limit", type=int, default=50, help="Maximum number of NBA events to fetch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "provider_healthcheck.json"
    fetched_at_utc = utc_now_iso()
    env_key = os.getenv("SPORTSGAMEODDS_API_KEY")
    api_key_visible = bool(env_key and str(env_key).strip())
    failure_reason = ""
    request_success = False
    events_returned = 0
    odds_rows_returned = 0
    starts_at_available_count = 0
    side_specific_price_count = 0
    books_observed: list[str] = []

    try:
        api_key = resolve_api_key(args.api_key)
        events, _headers = fetch_sportsgameodds_events(api_key, event_limit=args.event_limit)
        events_returned = len(events)
        starts_at_available_count = _count_starts_at(events)
        long_df, _wide_df = normalize_sportsgameodds_events(events, fetched_at_utc)
        odds_rows_returned = len(long_df)
        if not long_df.empty:
            price_rows = long_df[long_df["over_price"].notna() | long_df["under_price"].notna()]
            side_specific_price_count = len(price_rows)
            books_observed = [str(book).strip() for book in long_df.get("bookmaker_key", []).tolist() if str(book).strip()]
        request_success = True
    except Exception as exc:
        failure_reason = str(exc)

    report = build_provider_healthcheck_report(
        api_key_visible=api_key_visible,
        request_success=request_success,
        events_returned=events_returned,
        odds_rows_returned=odds_rows_returned,
        starts_at_available_count=starts_at_available_count,
        side_specific_price_count=side_specific_price_count,
        books_observed=books_observed,
        failure_reason=failure_reason,
        output_path=output_path,
        fetched_at_utc=fetched_at_utc,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
