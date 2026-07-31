#!/usr/bin/env python3
"""Fetch auditable pregame NFL player props from The Odds API history.

Historical player props are paid, event-scoped calls.  The command is a dry
run unless --execute is supplied so a season request cannot consume quota by
accident.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SPORT = "americanfootball_nfl"
SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
MARKETS = ("player_pass_yds", "player_rush_yds", "player_reception_yds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", default=None, help="Comma-separated regular-season weeks; default all.")
    parser.add_argument("--minutes-before-kickoff", type=int, default=30)
    parser.add_argument("--regions", default="us")
    parser.add_argument("--bookmakers", default="draftkings,fanduel")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sports/nfl/data/raw/historical_player_props.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("sports/nfl/data/raw/historical_player_props_manifest.json"),
    )
    return parser.parse_args()


def _kickoff_utc(frame: pd.DataFrame) -> pd.Series:
    local = pd.to_datetime(
        frame["gameday"].astype(str) + " " + frame["gametime"].astype(str), errors="coerce"
    )
    # nflverse expresses the schedule clock in US Eastern, including London games.
    return local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def load_schedule(season: int, weeks: set[int] | None) -> pd.DataFrame:
    schedule = pd.read_parquet(SCHEDULE_URL)
    schedule = schedule.loc[
        schedule["season"].eq(season) & schedule["game_type"].eq("REG")
    ].copy()
    if weeks:
        schedule = schedule.loc[schedule["week"].isin(weeks)].copy()
    schedule["commence_time_utc"] = _kickoff_utc(schedule)
    return schedule.dropna(subset=["commence_time_utc"])


def _request(
    session: requests.Session,
    path: str,
    params: dict[str, Any],
    *,
    max_retries: int = 4,
) -> tuple[Any, dict[str, str]]:
    for attempt in range(max_retries + 1):
        response = session.get(f"https://api.the-odds-api.com{path}", params=params, timeout=45)
        if response.ok:
            return response.json(), {key.lower(): value for key, value in response.headers.items()}
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
            error_code = "unknown"
            try:
                error_code = str(response.json().get("error_code") or response.json().get("message") or "unknown")
            except Exception:
                pass
            raise RuntimeError(f"The Odds API returned HTTP {response.status_code} ({error_code}).")
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2**attempt
        time.sleep(min(delay, 30.0))
    raise AssertionError("unreachable")


def _flatten_event(
    event: dict[str, Any], *, season: int, week: int, requested_snapshot: str, actual_snapshot: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            grouped: dict[tuple[str, float], dict[str, Any]] = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description") or outcome.get("participant")
                point = outcome.get("point")
                if not player or point is None:
                    continue
                key = (str(player), float(point))
                row = grouped.setdefault(
                    key,
                    {
                        "season": season,
                        "week": week,
                        "player": str(player),
                        "market": market.get("key"),
                        "line": float(point),
                        "over_price": pd.NA,
                        "under_price": pd.NA,
                        "bookmaker": bookmaker.get("key"),
                        "bookmaker_title": bookmaker.get("title"),
                        "source": "the_odds_api_historical",
                        "event_id": event.get("id"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "snapshot_time_utc": actual_snapshot,
                        "requested_snapshot_utc": requested_snapshot,
                        "commence_time_utc": event.get("commence_time"),
                    },
                )
                side = str(outcome.get("name") or "").lower()
                if side == "over":
                    row["over_price"] = outcome.get("price")
                elif side == "under":
                    row["under_price"] = outcome.get("price")
            rows.extend(grouped.values())
    return rows


def main() -> int:
    args = parse_args()
    weeks = {int(value) for value in args.weeks.split(",")} if args.weeks else None
    schedule = load_schedule(args.season, weeks)
    games = schedule[["season", "week", "commence_time_utc"]].drop_duplicates()
    maximum_event_odds_credits = int(len(schedule) * len(MARKETS) * 10)
    maximum_discovery_credits = int(len(games))
    maximum_credits = maximum_event_odds_credits + maximum_discovery_credits
    print(f"Games: {len(schedule)}; kickoff slots: {len(games)}")
    print(
        "Maximum quota estimate: "
        f"{maximum_credits} credits ({maximum_event_odds_credits} event odds + "
        f"{maximum_discovery_credits} event discovery)"
    )
    if not args.execute:
        print("Dry run only. Re-run with --execute after confirming plan access and quota.")
        return 0

    api_key = args.api_key or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set THE_ODDS_API_KEY (or ODDS_API_KEY) before using --execute.")

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "NFL-Predictor/1.0"})
    rows: list[dict[str, Any]] = []
    event_ids_seen: set[str] = set()
    quota_headers: dict[str, str] = {}
    for slot in games.itertuples(index=False):
        requested = slot.commence_time_utc - timedelta(minutes=args.minutes_before_kickoff)
        requested_iso = requested.isoformat().replace("+00:00", "Z")
        events_payload, quota_headers = _request(
            session,
            f"/v4/historical/sports/{SPORT}/events",
            {"apiKey": api_key, "date": requested_iso, "dateFormat": "iso"},
            max_retries=args.max_retries,
        )
        events = events_payload.get("data", []) if isinstance(events_payload, dict) else []
        for event in events:
            commence = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
            if pd.isna(commence) or abs((commence - slot.commence_time_utc).total_seconds()) > 300:
                continue
            event_id = str(event.get("id") or "")
            if not event_id or event_id in event_ids_seen:
                continue
            odds_payload, quota_headers = _request(
                session,
                f"/v4/historical/sports/{SPORT}/events/{event_id}/odds",
                {
                    "apiKey": api_key,
                    "date": requested_iso,
                    "regions": args.regions,
                    "bookmakers": args.bookmakers,
                    "markets": ",".join(MARKETS),
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
                max_retries=args.max_retries,
            )
            actual_snapshot = str(odds_payload.get("timestamp") or requested_iso)
            event_payload = odds_payload.get("data", odds_payload)
            rows.extend(
                _flatten_event(
                    event_payload,
                    season=int(slot.season),
                    week=int(slot.week),
                    requested_snapshot=requested_iso,
                    actual_snapshot=actual_snapshot,
                )
            )
            event_ids_seen.add(event_id)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "provider": "the_odds_api_historical",
        "season": args.season,
        "weeks": sorted(weeks) if weeks else "all_regular_season",
        "minutes_before_kickoff": args.minutes_before_kickoff,
        "markets": list(MARKETS),
        "bookmakers": args.bookmakers.split(","),
        "schedule_games": int(len(schedule)),
        "events_fetched": int(len(event_ids_seen)),
        "rows": int(len(output)),
        "requests_remaining": quota_headers.get("x-requests-remaining"),
        "requests_used": quota_headers.get("x-requests-used"),
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
