#!/usr/bin/env python3
"""Real PGA Tour event/field/scoring data from ESPN's public golf API.

WHY ESPN: it is a real, free, no-key-required public API that carries the
real season schedule (with real ESPN event IDs), the real tournament field
once posted, and real per-round scores as they are played -- the same kind
of real, legitimate public statistical source this repo already relies on
elsewhere (MLB StatsAPI, nflverse). No fabricated or simulated rows are
ever produced by this module: an event with no field posted yet returns an
explicit empty/pending result, never a guessed one.

Endpoints used (both public, no API key):
  - Scoreboard/calendar: https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard
    Carries the real full-season schedule (leagues[0].calendar), with a
    real ESPN event id + start/end date for every real PGA Tour event this
    season -- used to resolve "the next real event" without guessing.
  - Leaderboard: https://site.api.espn.com/apis/site/v2/sports/golf/leaderboard?event=<id>
    Carries the real field for one event: athlete identity, real per-round
    linescores (round number, strokes, in/out splits), real position, and
    real completion status. Empty `competitors` before the field is
    officially posted (typically 1-2 days before the first tee time) --
    this module reports that honestly rather than inventing a field.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
LEADERBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/leaderboard"
REQUEST_TIMEOUT_SECONDS = 20.0
# The lowest 18-hole score ever recorded on the PGA Tour is 58 (Jim Furyk,
# 2016 & 2017) -- any reported round strokes value below this cannot be a
# real played round. See extract_player_rounds()'s docstring for the real
# ESPN data quirk this guards against.
MIN_PLAUSIBLE_ROUND_STROKES = 50.0

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "sports" / "golf" / "data" / "raw" / "espn"


@dataclass(frozen=True)
class ScheduledEvent:
    event_id: str
    name: str
    start_date: str  # ISO-8601 UTC, real
    end_date: str  # ISO-8601 UTC, real


def fetch_season_calendar(*, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[ScheduledEvent]:
    """Real full-season PGA Tour schedule -- every real event this ESPN
    season object knows about, in real calendar order. Never fabricated;
    an unreachable endpoint raises rather than returning a guessed list."""
    response = requests.get(SCOREBOARD_URL, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    leagues = payload.get("leagues") or []
    calendar = leagues[0].get("calendar", []) if leagues else []
    events: list[ScheduledEvent] = []
    for entry in calendar:
        event_id = str(entry.get("id") or "").strip()
        if not event_id:
            continue
        events.append(
            ScheduledEvent(
                event_id=event_id,
                name=str(entry.get("label") or "").strip(),
                start_date=str(entry.get("startDate") or ""),
                end_date=str(entry.get("endDate") or ""),
            )
        )
    return events


def resolve_current_or_next_event(
    calendar: list[ScheduledEvent], *, as_of: Optional[datetime] = None
) -> Optional[ScheduledEvent]:
    """The real event whose window [start_date, end_date] contains `as_of`,
    or -- if none is currently underway (a real, expected state most of
    the week and during PGA Tour off-weeks) -- the real next event to
    start after `as_of`. Returns None only when the real calendar has no
    real future event left (e.g. past the last entry ESPN currently
    publishes) -- never a fabricated placeholder."""
    now = as_of or datetime.now(timezone.utc)

    def _parse(ts: str) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    parsed = [(event, _parse(event.start_date), _parse(event.end_date)) for event in calendar]
    for event, start, end in parsed:
        if start is not None and end is not None and start <= now <= end:
            return event
    upcoming = [(event, start) for event, start, _end in parsed if start is not None and start > now]
    if not upcoming:
        return None
    upcoming.sort(key=lambda pair: pair[1])
    return upcoming[0][0]


def recent_completed_events(
    calendar: list[ScheduledEvent], *, as_of: Optional[datetime] = None, limit: int = 6
) -> list[ScheduledEvent]:
    """The real last `limit` events whose window ended before `as_of` --
    used to build real recent-form features. Real calendar order, most
    recent first."""
    now = as_of or datetime.now(timezone.utc)

    def _parse(ts: str) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    completed = [(event, _parse(event.end_date)) for event in calendar]
    completed = [(event, end) for event, end in completed if end is not None and end < now]
    completed.sort(key=lambda pair: pair[1], reverse=True)
    return [event for event, _end in completed[:limit]]


def fetch_event_leaderboard(event_id: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Real leaderboard payload for one event: status, and every real
    competitor ESPN currently has posted (empty list before the field is
    announced -- reported as-is, never guessed)."""
    response = requests.get(LEADERBOARD_URL, params={"event": event_id}, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events") or []
    if not events:
        return {"event_id": event_id, "status": "UNKNOWN", "completed": False, "competitors": []}
    event = events[0]
    status = event.get("status", {}).get("type", {})
    competitions = event.get("competitions") or []
    competitors = competitions[0].get("competitors", []) if competitions else []
    return {
        "event_id": event_id,
        "event_name": str(event.get("date") and event.get("id") or event_id),
        "status": str(status.get("name") or "UNKNOWN"),
        "completed": bool(status.get("completed", False)),
        "competitors": competitors,
    }


def extract_player_rounds(competitor: dict[str, Any]) -> dict[str, Any]:
    """Flattens one real ESPN competitor row into the (player, per-round
    real strokes, real position, real status) shape the rest of this
    pipeline consumes. Never fills in a missing round with a guess --
    rounds not yet played are simply absent from `rounds`.

    REAL DATA QUIRK, found by inspecting a real withdrawn player's row
    (Robert MacIntyre, FedEx St. Jude Championship 2026, WD after round
    1): ESPN encodes an unplayed round -- after a withdrawal, or a round
    genuinely not reached yet -- as `value: 0.0, displayValue: "-"`, NOT
    as a null/None value. Treating that literal 0.0 as a real score of
    "70 strokes below par" silently corrupted every downstream form
    calculation for that player. Guarded on both signals: the "-" display
    marker (primary, matches ESPN's own convention) and a hard floor of
    MIN_PLAUSIBLE_ROUND_STROKES (defense in depth against any other
    malformed placeholder) -- the lowest 18-hole score ever recorded on
    the PGA Tour is 58, so any value below the floor cannot be real."""
    athlete = competitor.get("athlete", {}) or {}
    status = competitor.get("status", {}) or {}
    position = status.get("position", {}) or {}
    score = competitor.get("score", {}) or {}
    rounds: list[dict[str, Any]] = []
    for line in competitor.get("linescores", []) or []:
        value = line.get("value")
        display_value = str(line.get("displayValue") or "").strip()
        if value is None:
            continue
        if display_value == "-" or not display_value:
            continue
        if float(value) < MIN_PLAUSIBLE_ROUND_STROKES:
            continue
        rounds.append(
            {
                "round": line.get("period"),
                "strokes": float(value),
                "score_to_par_display": line.get("displayValue"),
            }
        )
    return {
        "player_id": str(athlete.get("id") or ""),
        "player_name": str(athlete.get("displayName") or ""),
        "headshot_url": (athlete.get("headshot") or {}).get("href", ""),
        "country": ((athlete.get("flag") or {}).get("alt") or ""),
        "position_display": str(position.get("displayName") or ""),
        "is_tied": bool(position.get("isTie", False)),
        "status_detail": str(status.get("detail") or ""),
        "status_state": str((status.get("type") or {}).get("state") or ""),
        "completed": bool((status.get("type") or {}).get("completed", False)),
        "cut_status": "CUT" if "CUT" in str((status.get("type") or {}).get("description") or "").upper() else "",
        "withdrawn": "WD" in str(status.get("detail") or "").upper(),
        "total_strokes": score.get("value"),
        "total_score_display": score.get("displayValue"),
        "rounds": rounds,
    }


def persist_event_snapshot(event_id: str, payload: dict[str, Any], *, raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    """Durable, real, timestamped archive of one real leaderboard fetch --
    never overwritten, so real round-by-round history for this event
    accumulates as the tournament is actually played (learning directly
    from this session's MLB price-capture investigation: never build a
    pipeline that only keeps the latest collapsed snapshot)."""
    event_dir = raw_root / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = event_dir / f"leaderboard_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    latest_path = event_dir / "latest_leaderboard.json"
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", type=str, default=None, help="Fetch one specific real ESPN event id instead of resolving the current/next one.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calendar = fetch_season_calendar()
    if args.event_id:
        target_id = args.event_id
    else:
        current = resolve_current_or_next_event(calendar)
        if current is None:
            print(json.dumps({"status": "no_event_in_calendar"}, indent=2))
            return 0
        target_id = current.event_id

    leaderboard = fetch_event_leaderboard(target_id)
    leaderboard["players"] = [extract_player_rounds(row) for row in leaderboard.pop("competitors", [])]
    out_path = persist_event_snapshot(target_id, leaderboard, raw_root=args.raw_root)
    print(json.dumps({"status": "ok", "event_id": target_id, "field_size": len(leaderboard["players"]), "written": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
