#!/usr/bin/env python3
"""Real historical MLB team-game outcomes AND real closing market lines,
from ESPN's public site API (no key needed) -- same real, free, already-
proven source and pattern this session used for NBA and NFL's team-game
history.

WHY ESPN, NOT MLB StatsAPI: StatsAPI (already used elsewhere in this
repo) carries real box scores but never market odds. ESPN's per-game
`summary` endpoint carries both: `header.competitions[].competitors[]`
for the real final score and real per-inning linescores (innings 1-5
summed give a real "First 5 Innings" total -- baseball's standard
equivalent of a first-half total, the market this session's original ask
named "over/under half (or inning)"), and `pickcenter` for a real book's
closing moneyline and full-game run total.

Two-stage, resumable pipeline identical in shape to the NBA/NFL fetchers:
`list_completed_game_ids_for_date` walks real dates to discover real
completed game ids, `fetch_and_persist_games` pulls (or reuses an
already-persisted trimmed real snapshot for) each one. Deliberately
stops at "fetch and persist the real dataset" -- fitting a model is real,
separate, sequenced follow-up work.
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

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"
REQUEST_TIMEOUT_SECONDS = 20.0
REQUEST_DELAY_SECONDS = 0.3

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "espn_team_games"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_team_game_history.csv"

OUTPUT_COLUMNS = [
    "game_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_won",
    "total_runs",
    "home_innings_1_5", "away_innings_1_5", "first_5_innings_total",
    "market_book",
    "market_run_total",
    "market_home_moneyline", "market_away_moneyline",
]


def list_completed_game_ids_for_date(date_str: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[str]:
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


def list_game_ids_in_date_range(start: date, end: date, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[str]:
    game_ids: list[str] = []
    current = start
    while current <= end:
        game_ids.extend(list_completed_game_ids_for_date(current.isoformat(), timeout_seconds=timeout_seconds))
        time.sleep(REQUEST_DELAY_SECONDS)
        current += timedelta(days=1)
    return game_ids


def fetch_game_summary(game_id: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    response = requests.get(SUMMARY_URL, params={"event": game_id}, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _inning_runs(linescores: list[dict[str, Any]]) -> list[Optional[float]]:
    """Real runs scored per real inning. Unlike NBA/NFL's fixed 4
    quarters, a real MLB linescores list can be shorter than 9 (the home
    team skips the bottom of the last inning when already leading) or
    longer (extra innings) -- this returns exactly what ESPN reports,
    never padded or truncated to a fixed length."""
    values: list[Optional[float]] = []
    for entry in linescores:
        raw = entry.get("displayValue")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(None)
    return values


def _sum_first_n_innings(innings: list[Optional[float]], n: int) -> Optional[float]:
    first_n = innings[:n]
    if len(first_n) < n or any(value is None for value in first_n):
        return None
    return sum(first_n)


def extract_team_game_row(summary: dict[str, Any], *, game_id: str) -> Optional[dict[str, Any]]:
    """Same real extraction contract as the NBA/NFL fetchers: returns
    None (never a guessed row) for an incomplete real game or missing
    real competitor data. A real game with fewer than 5 real innings on
    either side (a real rain-shortened game) reports first_5_innings_total
    as None rather than a partial/guessed sum."""
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

    home_innings = _inning_runs(home.get("linescores", []) or [])
    away_innings = _inning_runs(away.get("linescores", []) or [])
    home_f5 = _sum_first_n_innings(home_innings, 5)
    away_f5 = _sum_first_n_innings(away_innings, 5)

    row: dict[str, Any] = {
        "game_id": game_id,
        "date": str(competition.get("date", ""))[:10],
        "home_team": (home.get("team") or {}).get("abbreviation", ""),
        "away_team": (away.get("team") or {}).get("abbreviation", ""),
        "home_score": home_score,
        "away_score": away_score,
        "home_won": int(home_score > away_score),
        "total_runs": home_score + away_score,
        "home_innings_1_5": home_f5,
        "away_innings_1_5": away_f5,
        "first_5_innings_total": (home_f5 + away_f5) if home_f5 is not None and away_f5 is not None else None,
        "market_book": "",
        "market_run_total": None,
        "market_home_moneyline": None,
        "market_away_moneyline": None,
    }

    pickcenter = summary.get("pickcenter") or []
    if pickcenter:
        best = pickcenter[0]
        home_odds = best.get("homeTeamOdds", {}) or {}
        away_odds = best.get("awayTeamOdds", {}) or {}
        row["market_book"] = (best.get("provider") or {}).get("name", "")
        row["market_run_total"] = best.get("overUnder")
        row["market_home_moneyline"] = home_odds.get("moneyLine")
        row["market_away_moneyline"] = away_odds.get("moneyLine")

    return row


def _trim_summary_for_storage(summary: dict[str, Any]) -> dict[str, Any]:
    """See this session's NBA fetcher for why: ESPN's full real summary
    payload runs several hundred KB per game (box score, play-by-play,
    at-bats, news, videos); extract_team_game_row only ever reads
    `header` and `pickcenter`."""
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
        row = extract_team_game_row(summary, game_id=game_id)
        if row is not None:
            rows.append(row)
    return rows


def write_team_game_history_csv(rows: list[dict[str, Any]], *, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
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
    out_path = write_team_game_history_csv(rows, output_path=args.output)
    print(f"scanned {len(game_ids)} real game ids, wrote {len(rows)} real completed games to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
