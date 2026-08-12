#!/usr/bin/env python3
"""Fetch NFL player-prop closing lines from SportsGameOdds history.

The command is a dry run unless --execute is supplied. SportsGameOdds history
requires a Pro-or-higher plan; the free key remains useful for schema probes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_sources import (  # noqa: E402
    flatten_sportsgameodds_closing_lines,
    flatten_sportsgameodds_consensus_closing_lines,
)
from sports.nfl.scripts.fetch_historical_nfl_props import load_schedule  # noqa: E402


API_URL = "https://api.sportsgameodds.com/v2/events"
ODD_IDS = (
    "passing_yards-PLAYER_ID-game-ou-over",
    "rushing_yards-PLAYER_ID-game-ou-over",
    "receiving_yards-PLAYER_ID-game-ou-over",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", default=None, help="Comma-separated regular-season weeks; default all.")
    parser.add_argument("--bookmakers", default="draftkings,fanduel,betmgm,caesars")
    parser.add_argument(
        "--allow-consensus-close",
        action="store_true",
        help="Use explicit provider consensus closes when named-book closes are unavailable.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sports/nfl/data/raw/sportsgameodds_closing_props.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("sports/nfl/data/raw/sportsgameodds_closing_props_manifest.json"),
    )
    return parser.parse_args()


def _request_page(
    session: requests.Session,
    *,
    params: dict[str, Any],
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    for attempt in range(max_retries + 1):
        response = session.get(API_URL, params=params, timeout=60)
        if response.ok:
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("success") is False:
                raise RuntimeError("SportsGameOdds returned an unsuccessful JSON payload.")
            return payload, {key.lower(): value for key, value in response.headers.items()}
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
            message = "unknown"
            try:
                body = response.json()
                message = str(body.get("message") or body.get("error") or message)
            except Exception:
                pass
            raise RuntimeError(f"SportsGameOdds returned HTTP {response.status_code} ({message}).")
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2**attempt
        time.sleep(min(delay, 30.0))
    raise AssertionError("unreachable")


def _season_window(schedule: pd.DataFrame) -> tuple[str, str]:
    starts = pd.to_datetime(schedule["commence_time_utc"], utc=True)
    after = (starts.min() - pd.Timedelta(hours=18)).isoformat().replace("+00:00", "Z")
    before = (starts.max() + pd.Timedelta(hours=18)).isoformat().replace("+00:00", "Z")
    return after, before


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)
    weeks = {int(value) for value in args.weeks.split(",")} if args.weeks else None
    schedule = load_schedule(args.season, weeks)
    starts_after, starts_before = _season_window(schedule)
    print(f"Regular-season schedule games: {len(schedule)}")
    print(f"Historical window: {starts_after} through {starts_before}")
    print(f"Estimated billable event objects: at most {len(schedule)}")
    print("Required access: SportsGameOdds Pro or higher historical data")
    if not args.execute:
        print("Dry run only. Re-run with --execute after confirming historical player-prop coverage.")
        return 0

    api_key = args.api_key or os.getenv("SPORTSGAMEODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set SPORTSGAMEODDS_API_KEY before using --execute.")

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "NFL-Predictor/1.0",
            "x-api-key": api_key.strip(),
        }
    )
    base_params: dict[str, Any] = {
        "leagueID": "NFL",
        "finalized": "true",
        "oddsPresent": "true",
        "includeOpenCloseOdds": "true",
        "includeOpposingOdds": "true",
        "includeAltLines": "false",
        "oddID": ",".join(ODD_IDS),
        "bookmakerID": args.bookmakers,
        "startsAfter": starts_after,
        "startsBefore": starts_before,
        "limit": args.limit,
    }
    payloads: list[dict[str, Any]] = []
    pages = 0
    headers: dict[str, str] = {}
    cursor: str | None = None
    while True:
        params = dict(base_params)
        if cursor:
            params["cursor"] = cursor
        payload, headers = _request_page(session, params=params, max_retries=args.max_retries)
        payloads.append(payload)
        pages += 1
        cursor_value = payload.get("nextCursor")
        cursor = str(cursor_value) if cursor_value else None
        if not cursor:
            break
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    events = [event for payload in payloads for event in payload.get("data", [])]
    output, audit = flatten_sportsgameodds_closing_lines(
        events,
        season=args.season,
        schedule=schedule,
    )
    source_scope = "named_book_close"
    if output.empty and args.allow_consensus_close:
        output, audit = flatten_sportsgameodds_consensus_closing_lines(
            events,
            season=args.season,
            schedule=schedule,
        )
        source_scope = "provider_consensus_close_non_executable_book"
    if weeks and not output.empty:
        output = output.loc[output["week"].isin(weeks)].copy()
    output = output.sort_values(
        ["season", "week", "event_id", "market", "player", "bookmaker", "line"]
    ) if not output.empty else output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "provider": "sportsgameodds_historical_close",
        "provider_endpoint": API_URL,
        "season": args.season,
        "weeks": sorted(weeks) if weeks else "all_regular_season",
        "bookmakers": [value for value in args.bookmakers.split(",") if value],
        "markets": list(ODD_IDS),
        "starts_after": starts_after,
        "starts_before": starts_before,
        "schedule_games": int(len(schedule)),
        "pages_fetched": pages,
        "events_fetched": len(events),
        "rows": int(len(output)),
        "target_row_counts": {
            str(key): int(value) for key, value in output["market"].value_counts().items()
        } if not output.empty else {},
        "bookmaker_row_counts": {
            str(key): int(value) for key, value in output["bookmaker"].value_counts().items()
        } if not output.empty else {},
        "season_weeks_returned": int(output[["season", "week"]].drop_duplicates().shape[0])
        if not output.empty else 0,
        "normalization_audit": audit,
        "source_scope": source_scope,
        "executable_book_verified": source_scope == "named_book_close",
        "rate_limit_remaining": headers.get("x-ratelimit-remaining"),
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    required_markets = {"player_pass_yds", "player_rush_yds", "player_reception_yds"}
    returned_markets = set(output["market"].astype(str)) if not output.empty else set()
    if output.empty or not required_markets.issubset(returned_markets):
        print(
            "Coverage probe failed: verified two-sided closing rows were not returned "
            "for all three yardage targets. Do not run market promotion on this file."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
