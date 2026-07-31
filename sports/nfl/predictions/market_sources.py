"""Provider adapters for authentic historical NFL player-prop lines."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


SPORTSGAMEODDS_STAT_MARKETS = {
    "passing_yards": "player_pass_yds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
}


def _events(payload: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        values = payload.get("data", payload.get("events", []))
    else:
        values = payload
    return [value for value in values if isinstance(value, Mapping)]


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _week_from_event(event: Mapping[str, Any]) -> int | None:
    info = event.get("info") if isinstance(event.get("info"), Mapping) else {}
    label = str(info.get("seasonWeek") or info.get("week") or "")
    match = re.search(r"\bweek\s*(\d{1,2})\b", label, flags=re.IGNORECASE)
    if match is None and re.fullmatch(r"\s*\d{1,2}\s*", label):
        match = re.search(r"\d{1,2}", label)
    return int(match.group(1)) if match else None


def infer_schedule_week(commence_time: Any, schedule: pd.DataFrame) -> int | None:
    """Resolve a provider event to the nearest nflverse regular-season kickoff."""

    kickoff = pd.to_datetime(commence_time, utc=True, errors="coerce")
    if pd.isna(kickoff) or schedule.empty:
        return None
    deltas = (schedule["commence_time_utc"] - kickoff).abs()
    if deltas.empty or deltas.min() > pd.Timedelta(hours=18):
        return None
    return int(schedule.loc[deltas.idxmin(), "week"])


def flatten_sportsgameodds_closing_lines(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    season: int,
    schedule: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize per-book closing lines from a SportsGameOdds event response.

    Only the provider's explicit ``close*`` fields are accepted. Current/live
    ``odds`` and ``overUnder`` values are never used as historical lines.
    """

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    audit = {
        "events_received": 0,
        "events_not_finalized": 0,
        "events_without_week": 0,
        "target_odds_seen": 0,
        "book_sides_without_close": 0,
        "closing_sides_accepted": 0,
    }
    schedule_frame = schedule if schedule is not None else pd.DataFrame()

    for event in _events(payload):
        audit["events_received"] += 1
        status = event.get("status") if isinstance(event.get("status"), Mapping) else {}
        if not bool(status.get("finalized")):
            audit["events_not_finalized"] += 1
            continue
        commence = status.get("startsAt") or event.get("startsAt") or event.get("commence_time")
        week = _week_from_event(event)
        if week is None and not schedule_frame.empty:
            week = infer_schedule_week(commence, schedule_frame)
        if week is None:
            audit["events_without_week"] += 1
            continue

        players = event.get("players") if isinstance(event.get("players"), Mapping) else {}
        teams = event.get("teams") if isinstance(event.get("teams"), Mapping) else {}
        home = teams.get("home") if isinstance(teams.get("home"), Mapping) else {}
        away = teams.get("away") if isinstance(teams.get("away"), Mapping) else {}
        home_names = home.get("names") if isinstance(home.get("names"), Mapping) else {}
        away_names = away.get("names") if isinstance(away.get("names"), Mapping) else {}
        odds = event.get("odds") if isinstance(event.get("odds"), Mapping) else {}

        for odd in odds.values():
            if not isinstance(odd, Mapping):
                continue
            stat_id = str(odd.get("statID") or "")
            if stat_id not in SPORTSGAMEODDS_STAT_MARKETS:
                continue
            if str(odd.get("periodID") or "") != "game" or str(odd.get("betTypeID") or "") != "ou":
                continue
            side = str(odd.get("sideID") or "").lower()
            if side not in {"over", "under"}:
                continue
            audit["target_odds_seen"] += 1
            player_id = str(odd.get("playerID") or odd.get("statEntityID") or "")
            player = players.get(player_id) if isinstance(players.get(player_id), Mapping) else {}
            player_name = player.get("name") or player.get("display")
            if not player_name:
                market_name = str(odd.get("marketName") or "")
                suffix = re.search(r"\s+(?:Passing|Rushing|Receiving)\s+Yards", market_name, re.IGNORECASE)
                player_name = market_name[: suffix.start()].strip() if suffix else ""
            if not player_name:
                continue

            by_book = odd.get("byBookmaker") if isinstance(odd.get("byBookmaker"), Mapping) else {}
            for bookmaker, book_value in by_book.items():
                if not isinstance(book_value, Mapping):
                    continue
                line = _as_float(book_value.get("closeOverUnder"))
                price = _as_float(book_value.get("closeOdds"))
                if line is None or price is None:
                    audit["book_sides_without_close"] += 1
                    continue
                key = (event.get("eventID"), player_id, stat_id, str(bookmaker), line)
                row = grouped.setdefault(
                    key,
                    {
                        "season": int(season),
                        "week": int(week),
                        "player": str(player_name),
                        "player_id": player_id,
                        "market": SPORTSGAMEODDS_STAT_MARKETS[stat_id],
                        "line": line,
                        "over_price": pd.NA,
                        "under_price": pd.NA,
                        "bookmaker": str(bookmaker),
                        "source": "sportsgameodds_historical_close",
                        "event_id": event.get("eventID"),
                        "home_team": home_names.get("long") or home.get("name"),
                        "away_team": away_names.get("long") or away.get("name"),
                        "snapshot_time_utc": pd.NA,
                        "commence_time_utc": commence,
                        "line_phase": "closing_pregame",
                        "pregame_verified": True,
                        "verification_method": "provider_explicit_close_fields",
                    },
                )
                row[f"{side}_price"] = price
                audit["closing_sides_accepted"] += 1

    columns = [
        "season", "week", "player", "player_id", "market", "line",
        "over_price", "under_price", "bookmaker", "source", "event_id",
        "home_team", "away_team", "snapshot_time_utc", "commence_time_utc",
        "line_phase", "pregame_verified", "verification_method",
    ]
    frame = pd.DataFrame(grouped.values(), columns=columns)
    paired = frame["over_price"].notna() & frame["under_price"].notna() if not frame.empty else pd.Series(dtype=bool)
    audit["normalized_rows_before_price_pair_filter"] = int(len(frame))
    audit["dropped_one_sided_rows"] = int((~paired).sum()) if not frame.empty else 0
    frame = frame.loc[paired].reset_index(drop=True) if not frame.empty else frame
    audit["normalized_rows"] = int(len(frame))
    audit["two_sided_price_rows"] = int(len(frame))
    return frame, audit
