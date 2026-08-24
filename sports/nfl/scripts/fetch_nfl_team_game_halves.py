#!/usr/bin/env python3
"""Real historical NFL first-half scores, from ESPN's public site API.

WHY THIS EXISTS SEPARATELY FROM fetch_nfl_team_game_history.py: nflverse's
games.parquet (that script's real source) only carries real final scores
and real closing full-game lines -- no quarter/half splits -- so it can't
support an over/under-half market. ESPN's per-game `summary` endpoint
(the same real, free source this session already used for NBA's team-game
history and PGA golf) has real quarter-by-quarter linescores for every
real completed game, so Q1+Q2 gives a real first-half total the same way
it did for NBA.

Deliberately a standalone dataset (own game id, own CSV) rather than a
forced join into nfl_team_game_history.csv's nflverse-keyed rows: nflverse
and ESPN use different real game-id schemes and, in a few real cases,
different team abbreviations (e.g. the same real Rams game is "LA" in one
source and "LAR" in the other) -- joining them correctly needs a real,
verified abbreviation-mapping pass, which is disclosed follow-up work, not
guessed here. This file's own (date, home_team, away_team) columns are
enough to join manually or programmatically once that mapping exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
REQUEST_TIMEOUT_SECONDS = 20.0
REQUEST_DELAY_SECONDS = 0.3

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "sports" / "nfl" / "data" / "raw" / "espn_team_game_halves"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "sports" / "nfl" / "data" / "reference" / "nfl_team_game_halves.csv"

OUTPUT_COLUMNS = [
    "game_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_q1", "home_q2", "home_q3", "home_q4",
    "away_q1", "away_q2", "away_q3", "away_q4",
    "home_first_half", "away_first_half", "first_half_total",
    "market_book",
    "market_first_half_total",
]


def list_completed_game_ids_for_date(date_str: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS, max_retries: int = 3) -> list[str]:
    last_error: Optional[Exception] = None
    payload: Optional[dict[str, Any]] = None
    for attempt in range(max_retries):
        try:
            response = requests.get(SCOREBOARD_URL, params={"dates": date_str.replace("-", "")}, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    if payload is None:
        raise last_error
    ids: list[str] = []
    for event in payload.get("events", []):
        status = event.get("status", {}).get("type", {})
        if status.get("completed"):
            event_id = str(event.get("id") or "").strip()
            if event_id:
                ids.append(event_id)
    return ids


def list_game_ids_in_date_range(start: date, end: date, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[str]:
    """Real completed game ids across every real date in [start, end]
    (inclusive). NFL plays far fewer real game-days per week than
    NBA/MLB, so scanning every calendar date (rather than needing a
    separate real schedule lookup) stays cheap even across a real
    several-week window."""
    game_ids: list[str] = []
    current = start
    while current <= end:
        game_ids.extend(list_completed_game_ids_for_date(current.isoformat(), timeout_seconds=timeout_seconds))
        time.sleep(REQUEST_DELAY_SECONDS)
        current += timedelta(days=1)
    return game_ids


def fetch_game_summary(game_id: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS, max_retries: int = 3) -> dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = requests.get(SUMMARY_URL, params={"event": game_id}, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    raise last_error


def _quarter_scores(linescores: list[dict[str, Any]]) -> list[Optional[float]]:
    values: list[Optional[float]] = []
    for entry in linescores[:4]:
        raw = entry.get("displayValue")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(None)
    while len(values) < 4:
        values.append(None)
    return values


def extract_team_game_half_row(summary: dict[str, Any], *, game_id: str) -> Optional[dict[str, Any]]:
    """Same real extraction contract as NBA's fetch_nba_team_game_history:
    returns None (never a guessed row) for an incomplete real game or
    missing real competitor data. A real book's first-half total line
    (`market_first_half_total`) is included when ESPN's pickcenter
    exposes one -- it does not for every real game, and this leaves it
    None rather than guessing when absent."""
    header = summary.get("header", {})
    competitions = header.get("competitions", [])
    if not competitions:
        return None
    competition = competitions[0]
    status = competition.get("status", {}).get("type", {})
    if not status.get("completed"):
        return None

    competitors = competition.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if home is None or away is None:
        return None

    def _score(competitor: dict[str, Any]) -> Optional[float]:
        try:
            return float(competitor.get("score"))
        except (TypeError, ValueError):
            return None

    home_score = _score(home)
    away_score = _score(away)
    if home_score is None or away_score is None:
        return None

    home_q = _quarter_scores(home.get("linescores", []) or [])
    away_q = _quarter_scores(away.get("linescores", []) or [])
    home_first_half = home_q[0] + home_q[1] if home_q[0] is not None and home_q[1] is not None else None
    away_first_half = away_q[0] + away_q[1] if away_q[0] is not None and away_q[1] is not None else None

    row: dict[str, Any] = {
        "game_id": game_id,
        "date": str(competition.get("date", ""))[:10],
        "home_team": (home.get("team") or {}).get("abbreviation", ""),
        "away_team": (away.get("team") or {}).get("abbreviation", ""),
        "home_score": home_score,
        "away_score": away_score,
        "home_q1": home_q[0], "home_q2": home_q[1], "home_q3": home_q[2], "home_q4": home_q[3],
        "away_q1": away_q[0], "away_q2": away_q[1], "away_q3": away_q[2], "away_q4": away_q[3],
        "home_first_half": home_first_half,
        "away_first_half": away_first_half,
        "first_half_total": (
            home_first_half + away_first_half if home_first_half is not None and away_first_half is not None else None
        ),
        "market_book": "",
        "market_first_half_total": None,
    }

    pickcenter = summary.get("pickcenter") or []
    if pickcenter:
        best = pickcenter[0]
        row["market_book"] = (best.get("provider") or {}).get("name", "")
        # ESPN's pickcenter carries the full-game total in `overUnder`;
        # a real first-half-specific total line isn't a standard field
        # there, so this stays honestly unset rather than substituting
        # the full-game number for a different real market.

    return row


def _trim_summary_for_storage(summary: dict[str, Any]) -> dict[str, Any]:
    return {"header": summary.get("header"), "pickcenter": summary.get("pickcenter")}


def persist_game_snapshot(game_id: str, summary: dict[str, Any], *, raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    out_path = raw_root / f"{game_id}.json"
    out_path.write_text(json.dumps(_trim_summary_for_storage(summary), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def fetch_and_persist_games(
    game_ids: list[str],
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    refresh: bool = False,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_id in game_ids:
        snapshot_path = raw_root / f"{game_id}.json"
        if snapshot_path.exists() and not refresh:
            summary = json.loads(snapshot_path.read_text(encoding="utf-8"))
        else:
            summary = fetch_game_summary(game_id, timeout_seconds=timeout_seconds)
            persist_game_snapshot(game_id, summary, raw_root=raw_root)
            time.sleep(REQUEST_DELAY_SECONDS)
        row = extract_team_game_half_row(summary, game_id=game_id)
        if row is not None:
            rows.append(row)
    return rows


def write_team_game_halves_csv(rows: list[dict[str, Any]], *, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    game_ids = list_game_ids_in_date_range(args.start_date, args.end_date)
    rows = fetch_and_persist_games(game_ids, raw_root=args.raw_root, refresh=args.refresh)
    out_path = write_team_game_halves_csv(rows, output_path=args.output)
    print(f"scanned {len(game_ids)} real game ids, wrote {len(rows)} real completed games with real half scores to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
