#!/usr/bin/env python3
"""Real historical NBA team-game outcomes AND real closing market lines,
from ESPN's public site API (no key needed) -- the same real, free,
already-proven source this repo uses for PGA golf
(sports/golf/scripts/fetch_pga_event.py).

WHY ESPN, NOT nba_api: nba_api/stats.nba.com carries real box scores but
never market odds, and is unreachable from some environments (blocked/
rate-limited). There is also no NBA equivalent of nflverse's single
games.parquet file with real closing lines baked in. ESPN's per-game
`summary` endpoint fills both gaps at once for a real game once you have
its real ESPN event id: `header.competitions[].competitors[].score` (and
`.linescores`, real quarter-by-quarter scores -- Q1+Q2 gives a real first-
half total) for the real outcome, and `pickcenter` for the real closing
moneyline/spread/total line a real bookmaker (commonly DraftKings) had on
the game.

Two-stage, resumable pipeline (mirrors this repo's other backfill
scripts): `list_season_game_ids` walks the real season calendar to find
every real completed game's ESPN id (one scoreboard call per real game
date -- no bulk "give me every id" endpoint exists), then
`fetch_team_game_row` pulls one real game's full detail. A durable
per-game snapshot is written for every real fetch so a season backfill
never has to refetch a game it already has (see `--refresh` to force
one anyway).

Deliberately stops at "fetch and persist the real dataset" -- fitting a
model is real, separate, sequenced follow-up work (same posture as this
session's NFL team-game history fetcher).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
REQUEST_TIMEOUT_SECONDS = 20.0
REQUEST_DELAY_SECONDS = 0.3  # real, polite pacing against a free public API -- never hammer it

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "sports" / "nba" / "data" / "raw" / "espn_team_games"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "sports" / "nba" / "data" / "reference" / "nba_team_game_history.csv"

OUTPUT_COLUMNS = [
    "game_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_won",
    "total_points",
    "home_q1", "home_q2", "home_q3", "home_q4",
    "away_q1", "away_q2", "away_q3", "away_q4",
    "home_first_half", "away_first_half", "first_half_total",
    "market_book",
    "market_spread", "market_spread_favorite_home",
    "market_total",
    "market_home_moneyline", "market_away_moneyline",
]


def fetch_season_calendar_dates(*, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[str]:
    """Every real date the current NBA season's scoreboard calendar
    lists (YYYY-MM-DD strings) -- the closest thing to a bulk schedule
    ESPN's public API offers; there is no single endpoint that lists
    every real game id for a season directly."""
    response = requests.get(SCOREBOARD_URL, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    leagues = payload.get("leagues") or []
    calendar = leagues[0].get("calendar", []) if leagues else []
    return [str(entry)[:10] for entry in calendar if isinstance(entry, str)]


def list_completed_game_ids_for_date(date_str: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[str]:
    """Real, completed (STATUS_FINAL) game ids for one real calendar
    date. `date_str` is YYYY-MM-DD; ESPN's dates param wants YYYYMMDD."""
    response = requests.get(SCOREBOARD_URL, params={"dates": date_str.replace("-", "")}, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    ids: list[str] = []
    for event in payload.get("events", []):
        status = event.get("status", {}).get("type", {})
        if status.get("completed"):
            event_id = str(event.get("id") or "").strip()
            if event_id:
                ids.append(event_id)
    return ids


def list_season_game_ids(*, max_dates: Optional[int] = None, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[str]:
    """Every real completed game id across the real current season
    calendar. `max_dates` (if given) caps how many real calendar dates
    are scanned -- useful for a bounded verification run rather than a
    full-season backfill."""
    dates = fetch_season_calendar_dates(timeout_seconds=timeout_seconds)
    if max_dates is not None:
        dates = dates[:max_dates]
    game_ids: list[str] = []
    for date_str in dates:
        game_ids.extend(list_completed_game_ids_for_date(date_str, timeout_seconds=timeout_seconds))
        time.sleep(REQUEST_DELAY_SECONDS)
    return game_ids


def fetch_game_summary(game_id: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    response = requests.get(SUMMARY_URL, params={"event": game_id}, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


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


def extract_team_game_row(summary: dict[str, Any], *, game_id: str) -> Optional[dict[str, Any]]:
    """Flattens one real ESPN game summary into a single real team-game
    row. Returns None (never a guessed row) if the real game isn't
    actually completed yet, or real competitor data is missing."""
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
        "date": str(header.get("competitions", [{}])[0].get("date", ""))[:10],
        "home_team": (home.get("team") or {}).get("abbreviation", ""),
        "away_team": (away.get("team") or {}).get("abbreviation", ""),
        "home_score": home_score,
        "away_score": away_score,
        "home_won": int(home_score > away_score),
        "total_points": home_score + away_score,
        "home_q1": home_q[0], "home_q2": home_q[1], "home_q3": home_q[2], "home_q4": home_q[3],
        "away_q1": away_q[0], "away_q2": away_q[1], "away_q3": away_q[2], "away_q4": away_q[3],
        "home_first_half": home_first_half,
        "away_first_half": away_first_half,
        "first_half_total": (
            home_first_half + away_first_half if home_first_half is not None and away_first_half is not None else None
        ),
        "market_book": "",
        "market_spread": None,
        "market_spread_favorite_home": None,
        "market_total": None,
        "market_home_moneyline": None,
        "market_away_moneyline": None,
    }

    pickcenter = summary.get("pickcenter") or []
    if pickcenter:
        # Real, best-available closing line: ESPN lists one entry per
        # real book; take the first (highest-priority) real provider
        # rather than averaging across books -- matches this session's
        # own established discipline (a single real book's price, never
        # a blended/averaged one) from the MLB price-capture work.
        best = pickcenter[0]
        provider = (best.get("provider") or {}).get("name", "")
        home_odds = best.get("homeTeamOdds", {}) or {}
        away_odds = best.get("awayTeamOdds", {}) or {}
        row["market_book"] = provider
        row["market_spread"] = best.get("spread")
        row["market_spread_favorite_home"] = home_odds.get("favorite")
        row["market_total"] = best.get("overUnder")
        row["market_home_moneyline"] = home_odds.get("moneyLine")
        row["market_away_moneyline"] = away_odds.get("moneyLine")

    return row


def persist_game_snapshot(game_id: str, summary: dict[str, Any], *, raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    game_dir = raw_root
    game_dir.mkdir(parents=True, exist_ok=True)
    out_path = game_dir / f"{game_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def fetch_and_persist_games(
    game_ids: list[str],
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    refresh: bool = False,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Fetches (or reuses an already-persisted real snapshot for) every
    real game id given, and returns the flattened rows for real,
    completed games. Resumable by construction: a game already on disk
    is reused unless `refresh` is set, so a season backfill can be run
    incrementally across many turns without ever refetching real data it
    already has."""
    rows: list[dict[str, Any]] = []
    for game_id in game_ids:
        snapshot_path = raw_root / f"{game_id}.json"
        if snapshot_path.exists() and not refresh:
            summary = json.loads(snapshot_path.read_text(encoding="utf-8"))
        else:
            summary = fetch_game_summary(game_id, timeout_seconds=timeout_seconds)
            persist_game_snapshot(game_id, summary, raw_root=raw_root)
            time.sleep(REQUEST_DELAY_SECONDS)
        row = extract_team_game_row(summary, game_id=game_id)
        if row is not None:
            rows.append(row)
    return rows


def write_team_game_history_csv(rows: list[dict[str, Any]], *, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dates", type=int, default=None, help="Cap how many real calendar dates to scan (omit for the full season).")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--refresh", action="store_true", help="Refetch every game even if a real snapshot already exists on disk.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    game_ids = list_season_game_ids(max_dates=args.max_dates)
    rows = fetch_and_persist_games(game_ids, raw_root=args.raw_root, refresh=args.refresh)
    out_path = write_team_game_history_csv(rows, output_path=args.output)
    print(f"scanned {len(game_ids)} real game ids, wrote {len(rows)} real completed games to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
