#!/usr/bin/env python3
"""Real per-game starting-pitcher identity and real starter/bullpen box
score pitching lines, from MLB StatsAPI's public schedule + boxscore
endpoints (no key needed) -- the same real source
generate_daily_prediction_pool.py already uses for probable-pitcher
lookups, extended here for real completed-game box scores.

WHY THIS EXISTS: team_win_model.py's backtest (this session, earlier)
showed a Pythagorean-only win-probability model is slightly WORSE than
the real market (51.3% accuracy vs. market's implied edge) -- an honest,
expected result for a model with no starting-pitcher or bullpen signal.
This is the real historical foundation for that enrichment: for every
real completed MLB game, the real starting pitcher on each side and a
real starter-vs-bullpen split of the box score's pitching line.

REAL EXTRACTION DETAIL: StatsAPI's boxscore reports each pitcher's real
outs recorded (`outs`, an integer -- e.g. 18 outs = 6.0 real innings)
rather than the "X.1 / X.2" innings-pitched NOTATION used elsewhere in
baseball (where the fractional part means outs, not tenths) -- using
`outs` sidesteps that notation entirely, so innings-pitched here is
always a real `outs / 3.0`, never a misparsed decimal.

The real starting pitcher is identified via StatsAPI's own
`gamesStarted == 1` flag on that pitcher's real box-score pitching
stats -- never assumed from list position. A game where no pitcher on a
side carries that flag (should not happen in a real completed game, but
is a real possible data gap) is skipped for that side rather than
guessed.

Deliberately keyed by (date, home_team, away_team) -- the same real join
key mlb_team_game_history.csv (ESPN-sourced) uses -- rather than
StatsAPI's own gamePk, so this dataset can be joined directly onto the
existing real team-game history without a separate id-mapping pass.
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

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
REQUEST_TIMEOUT_SECONDS = 20.0
REQUEST_DELAY_SECONDS = 0.3

# Real, VERIFIED mismatches between StatsAPI's and ESPN's team
# abbreviations for the same two real MLB teams -- found by actually
# diffing the two real datasets' full abbreviation sets (every other
# team matches exactly), not guessed upfront. Same kind of real,
# disclosed cross-source join gap as NFL's "LA" vs "LAR" case elsewhere
# in this repo. Normalizing to ESPN's convention here (rather than in
# the join) keeps this dataset directly joinable on (date, home_team,
# away_team) against mlb_team_game_history.csv without a separate
# mapping pass at read time.
STATSAPI_TO_ESPN_TEAM_ABBREVIATION = {
    "AZ": "ARI",   # Arizona Diamondbacks
    "CWS": "CHW",  # Chicago White Sox
}


def _to_espn_abbreviation(statsapi_abbreviation: str) -> str:
    return STATSAPI_TO_ESPN_TEAM_ABBREVIATION.get(statsapi_abbreviation, statsapi_abbreviation)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "statsapi_pitcher_boxscores"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_pitcher_game_data.csv"

OUTPUT_COLUMNS = [
    "game_pk",
    "date",
    "home_team",
    "away_team",
    "home_starter_id",
    "home_starter_name",
    "home_starter_outs",
    "home_starter_earned_runs",
    "home_bullpen_outs",
    "home_bullpen_earned_runs",
    "away_starter_id",
    "away_starter_name",
    "away_starter_outs",
    "away_starter_earned_runs",
    "away_bullpen_outs",
    "away_bullpen_earned_runs",
]


def list_completed_games_for_date(date_str: str, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS, max_retries: int = 3) -> list[dict[str, Any]]:
    """Real (gamePk, home_team, away_team) triples for every real completed
    game on a real date. Retries a real transient network failure
    (timeout / connection error) the same way this session's other MLB
    fetcher does; a real HTTP error status is never retried."""
    last_error: Optional[Exception] = None
    payload: Optional[dict[str, Any]] = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                SCHEDULE_URL,
                params={"sportId": 1, "date": date_str, "hydrate": "team"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    if payload is None:
        raise last_error

    games: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            status = game.get("status", {})
            if status.get("codedGameState") != "F":
                continue  # real completed games only, never a postponed/in-progress one
            game_pk = game.get("gamePk")
            teams = game.get("teams", {})
            home = teams.get("home", {}).get("team", {})
            away = teams.get("away", {}).get("team", {})
            if game_pk is None or not home.get("id") or not away.get("id"):
                continue
            games.append({"game_pk": game_pk, "date": date_str, "home_team_id": home["id"], "away_team_id": away["id"]})
    return games


def list_games_in_date_range(start: date, end: date, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    current = start
    while current <= end:
        games.extend(list_completed_games_for_date(current.isoformat(), timeout_seconds=timeout_seconds))
        time.sleep(REQUEST_DELAY_SECONDS)
        current += timedelta(days=1)
    return games


def fetch_boxscore(game_pk: int, *, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS, max_retries: int = 3) -> dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = requests.get(BOXSCORE_URL_TEMPLATE.format(game_pk=game_pk), timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    raise last_error


def _extract_side_pitching(team_box: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Real starter-vs-bullpen split for one real team's real box score.
    Returns None (never a guessed split) when no pitcher on this side
    carries StatsAPI's own real `gamesStarted == 1` flag."""
    pitcher_ids = team_box.get("pitchers", []) or []
    players = team_box.get("players", {}) or {}

    starter_id: Optional[int] = None
    starter_stats: Optional[dict[str, Any]] = None
    bullpen_outs = 0
    bullpen_earned_runs = 0

    lines: list[tuple[int, dict[str, Any]]] = []
    for pid in pitcher_ids:
        player = players.get(f"ID{pid}")
        if player is None:
            continue
        stats = (player.get("stats") or {}).get("pitching")
        if not stats:
            continue
        lines.append((pid, stats))

    for pid, stats in lines:
        if stats.get("gamesStarted") == 1:
            starter_id = pid
            starter_stats = stats
            break

    if starter_id is None or starter_stats is None:
        return None

    for pid, stats in lines:
        if pid == starter_id:
            continue
        bullpen_outs += int(stats.get("outs") or 0)
        bullpen_earned_runs += int(stats.get("earnedRuns") or 0)

    starter_player = players.get(f"ID{starter_id}", {})
    return {
        "starter_id": starter_id,
        "starter_name": (starter_player.get("person") or {}).get("fullName", ""),
        "starter_outs": int(starter_stats.get("outs") or 0),
        "starter_earned_runs": int(starter_stats.get("earnedRuns") or 0),
        "bullpen_outs": bullpen_outs,
        "bullpen_earned_runs": bullpen_earned_runs,
    }


def extract_pitcher_game_row(boxscore: dict[str, Any], *, game_pk: int, date_str: str) -> Optional[dict[str, Any]]:
    teams = boxscore.get("teams", {}) or {}
    home_box = teams.get("home", {}) or {}
    away_box = teams.get("away", {}) or {}
    home_team = _to_espn_abbreviation((home_box.get("team") or {}).get("abbreviation", ""))
    away_team = _to_espn_abbreviation((away_box.get("team") or {}).get("abbreviation", ""))
    if not home_team or not away_team:
        return None

    home_pitching = _extract_side_pitching(home_box)
    away_pitching = _extract_side_pitching(away_box)
    if home_pitching is None or away_pitching is None:
        return None

    return {
        "game_pk": game_pk,
        "date": date_str,
        "home_team": home_team,
        "away_team": away_team,
        "home_starter_id": home_pitching["starter_id"],
        "home_starter_name": home_pitching["starter_name"],
        "home_starter_outs": home_pitching["starter_outs"],
        "home_starter_earned_runs": home_pitching["starter_earned_runs"],
        "home_bullpen_outs": home_pitching["bullpen_outs"],
        "home_bullpen_earned_runs": home_pitching["bullpen_earned_runs"],
        "away_starter_id": away_pitching["starter_id"],
        "away_starter_name": away_pitching["starter_name"],
        "away_starter_outs": away_pitching["starter_outs"],
        "away_starter_earned_runs": away_pitching["starter_earned_runs"],
        "away_bullpen_outs": away_pitching["bullpen_outs"],
        "away_bullpen_earned_runs": away_pitching["bullpen_earned_runs"],
    }


def _trim_boxscore_for_storage(boxscore: dict[str, Any]) -> dict[str, Any]:
    """Real boxscore payloads carry full real batting lines, coaching
    staff, and bench/bullpen personnel not used here -- trim to just the
    two teams' abbreviation/pitchers/pitching-stats fields actually read
    by extract_pitcher_game_row, mirroring this session's NBA/NFL/MLB
    _trim_summary_for_storage pattern."""
    trimmed: dict[str, Any] = {"teams": {}}
    for side in ("home", "away"):
        team_box = boxscore.get("teams", {}).get(side, {}) or {}
        players = team_box.get("players", {}) or {}
        trimmed_players = {
            key: {
                "person": {"fullName": (player.get("person") or {}).get("fullName", "")},
                "stats": {"pitching": (player.get("stats") or {}).get("pitching", {})},
            }
            for key, player in players.items()
            if (player.get("stats") or {}).get("pitching")
        }
        trimmed["teams"][side] = {
            "team": {"abbreviation": (team_box.get("team") or {}).get("abbreviation", "")},
            "pitchers": team_box.get("pitchers", []),
            "players": trimmed_players,
        }
    return trimmed


def persist_boxscore_snapshot(game_pk: int, boxscore: dict[str, Any], *, raw_root: Path = DEFAULT_RAW_ROOT) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    out_path = raw_root / f"{game_pk}.json"
    out_path.write_text(json.dumps(_trim_boxscore_for_storage(boxscore), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


def fetch_and_persist_games(
    games: list[dict[str, Any]],
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    refresh: bool = False,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in games:
        game_pk = game["game_pk"]
        snapshot_path = raw_root / f"{game_pk}.json"
        if snapshot_path.exists() and not refresh:
            boxscore = json.loads(snapshot_path.read_text(encoding="utf-8"))
        else:
            boxscore = fetch_boxscore(game_pk, timeout_seconds=timeout_seconds)
            persist_boxscore_snapshot(game_pk, boxscore, raw_root=raw_root)
            time.sleep(REQUEST_DELAY_SECONDS)
        row = extract_pitcher_game_row(boxscore, game_pk=game_pk, date_str=game["date"])
        if row is not None:
            rows.append(row)
    return rows


def write_pitcher_game_data_csv(rows: list[dict[str, Any]], *, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
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
    games = list_games_in_date_range(args.start_date, args.end_date)
    rows = fetch_and_persist_games(games, raw_root=args.raw_root, refresh=args.refresh)
    out_path = write_pitcher_game_data_csv(rows, output_path=args.output)
    print(f"scanned {len(games)} real completed games, wrote {len(rows)} real starter/bullpen rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
