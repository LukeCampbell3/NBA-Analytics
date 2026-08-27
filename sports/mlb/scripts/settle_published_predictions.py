#!/usr/bin/env python3
"""
Settle published MLB predictions against real, final MLB Stats API results.

Runs against the actual frontend-facing payloads -- sports/mlb/web/data/
daily_predictions.json and every sports/mlb/web/data/history/<date>.json --
and writes real win/loss/push outcomes directly onto each play/leg dict so
the board (today's and every historical date a viewer browses to, since
history/<date>.json is exactly what predictions.js fetches for that date)
shows a real settled result once the underlying MLB game is final.

Design principles (mirrors settle_mlb_production_shadow.py's own stated
rule): appends outcome fields only, never modifies prediction-time fields.
The added keys are always a strict superset of what a row already carries:
    settlement_status        "won" | "lost" | "push" | "pending"
    settlement_actual_value  real stat value from the final boxscore (float)
    settlement_source        "mlb_statsapi_live_feed" (only once resolved)
    settlement_checked_at    ISO8601 UTC timestamp of the last attempt
    settlement_reason        short reason code (only while still "pending")

Idempotent and safe to run unboundedly often: a row already resolved to
won/lost/push is never re-fetched or re-graded. This is also what makes the
very first run a correct catch-up run with no special flag -- every row in
every history file starts with no settlement fields, so the first run
attempts every one of them; every later run only spends a real HTTP fetch
on a row that is still "pending" (game wasn't final yet, or a transient
fetch error) or has never been attempted.

Deliberately reuses the real MLB Stats API game-status/boxscore shape from
sports/mlb/predictions/scripts/settle_mlb_evidence_once.py (STAT_MAP,
_norm_name), but NOT that script's own game-matching path (_find_final_game
imports a module, audit_mlb_settlement_matching, that does not exist
anywhere in this repository -- every call into it raises ImportError,
silently caught and reported as "statsapi_schedule_error:ImportError". That
machinery is effectively dead code today). This script instead uses the
real MLB game_id every play/leg already carries (the exact MLB Stats API
gamePk -- verified against a real completed game) to fetch
`/api/v1.1/game/{game_id}/feed/live` directly: one real endpoint that
carries both the final game-status flag and the full boxscore in a single
call, with no team/date matching required at all.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "sports" / "mlb" / "web" / "data"
DEFAULT_REPORT_PATH = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "settlement" / "mlb_settlement_report.json"

RESOLVED_STATUSES = {"won", "lost", "push"}

# target short code -> {player_type -> (statsapi stat_group, statsapi stat key)}.
# Every code actually observed in sports/mlb/web/data/history/*.json today
# resolves unambiguously to a single role (K is always a pitcher's
# strikeouts in this app's real data; H/TB/R/RBI/HR are always a hitter's).
# The second role per code is kept only so a future market of that other
# shape settles correctly too, rather than being silently skipped.
TARGET_STAT_MAP: Dict[str, Dict[str, Tuple[str, str]]] = {
    "H": {"hitter": ("batting", "hits"), "pitcher": ("pitching", "hits")},
    "TB": {"hitter": ("batting", "totalBases")},
    "R": {"hitter": ("batting", "runs")},
    "RBI": {"hitter": ("batting", "rbi")},
    "HR": {"hitter": ("batting", "homeRuns")},
    "K": {"pitcher": ("pitching", "strikeOuts"), "hitter": ("batting", "strikeOuts")},
    "ER": {"pitcher": ("pitching", "earnedRuns")},
}
DEFAULT_ROLE_FOR_TARGET = {"H": "hitter", "TB": "hitter", "R": "hitter", "RBI": "hitter", "HR": "hitter", "K": "pitcher", "ER": "pitcher"}

FINAL_ABSTRACT_STATES = {"final"}
FINAL_DETAILED_STATES = {"final", "game over", "completed early"}


def _norm_name(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9 ]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def resolve_stat_spec(target: Any, player_type: Any) -> Optional[Tuple[str, str]]:
    code = str(target or "").upper()
    spec_by_role = TARGET_STAT_MAP.get(code)
    if not spec_by_role:
        return None
    role = str(player_type or "").lower()
    if role not in spec_by_role:
        role = DEFAULT_ROLE_FOR_TARGET.get(code, next(iter(spec_by_role)))
    return spec_by_role.get(role)


def _is_final_status(status: Dict[str, Any]) -> bool:
    abstract = str(status.get("abstractGameState", "")).lower()
    detailed = str(status.get("detailedState", "")).lower()
    return abstract in FINAL_ABSTRACT_STATES or detailed in FINAL_DETAILED_STATES


def _extract_stat(feed: Dict[str, Any], player_name: Any, stat_group: str, stat_key: str) -> Tuple[Optional[float], str]:
    boxscore = feed.get("liveData", {}).get("boxscore", {})
    wanted = _norm_name(player_name)
    if not wanted:
        return None, "missing_player_name"
    for side in ("home", "away"):
        players = boxscore.get("teams", {}).get(side, {}).get("players", {})
        for player in players.values():
            person = player.get("person", {})
            if _norm_name(person.get("fullName", "")) != wanted:
                continue
            stats = player.get("stats", {}).get(stat_group, {})
            if not stats:
                return None, "lineup_role_mismatch"
            value = stats.get(stat_key)
            if value is None:
                return None, "stat_not_available"
            try:
                return float(value), ""
            except (TypeError, ValueError):
                return None, "stat_not_numeric"
    return None, "player_not_found"


def grade_outcome(actual: float, line: float, direction: Any) -> str:
    """Mirrors sports/mlb/scripts/backtest_prediction_method.py's grade():
    OVER wins above the line, UNDER wins below it, either pushes exactly on
    it. Returns "won"/"lost"/"push" (this feature's own settlement
    vocabulary) rather than that function's "win"/"loss"."""
    is_under = str(direction or "").upper() == "UNDER"
    if actual == line:
        return "push"
    if is_under:
        return "won" if actual < line else "lost"
    return "won" if actual > line else "lost"


def make_live_feed_fetcher(session: Optional[requests.Session] = None, request_timeout: float = 20.0, sleep_between_requests: float = 0.0) -> Callable[[str], Dict[str, Any]]:
    """Returns a get_live_feed(game_id) -> dict callable backed by the real
    MLB Stats API, with a per-run in-memory cache so a game_id referenced by
    more than one row (e.g. a same-game combo's two legs) is fetched once."""
    http = session or requests.Session()
    cache: Dict[str, Dict[str, Any]] = {}

    def get_live_feed(game_id: str) -> Dict[str, Any]:
        if game_id not in cache:
            if sleep_between_requests > 0 and cache:
                time.sleep(sleep_between_requests)
            response = http.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live", timeout=request_timeout)
            response.raise_for_status()
            cache[game_id] = response.json()
        return cache[game_id]

    return get_live_feed


def _row_spec(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Any, Any, Any, Any]:
    game_id = row.get("game_id")
    player = row.get("player_display_name") or row.get("player")
    target = row.get("target")
    line = row.get("market_line", row.get("line"))
    direction = row.get("direction") or row.get("side")
    player_type = row.get("player_type")
    return (str(game_id) if game_id is not None else None), player, target, line, direction, player_type


def settle_row(row: Dict[str, Any], get_live_feed: Callable[[str], Dict[str, Any]], now_iso: str) -> bool:
    """Mutates `row` in place, adding settlement_* fields only. Returns True
    if the row was written to (resolved or a pending attempt was recorded),
    False if it was left untouched (already resolved, or not a settleable
    row at all -- e.g. missing a required field)."""
    if row.get("settlement_status") in RESOLVED_STATUSES:
        return False

    game_id, player, target, line, direction, player_type = _row_spec(row)
    if not game_id or not player or target is None or line is None or not direction:
        return False

    def _pending(reason: str) -> bool:
        row["settlement_status"] = "pending"
        row["settlement_reason"] = reason
        row["settlement_checked_at"] = now_iso
        return True

    stat_spec = resolve_stat_spec(target, player_type)
    if not stat_spec:
        return _pending("unsupported_target")

    try:
        feed = get_live_feed(game_id)
    except Exception as exc:  # real network/HTTP failures only -- never silently swallowed
        return _pending(f"fetch_error:{type(exc).__name__}")

    status = feed.get("gameData", {}).get("status", {})
    if not _is_final_status(status):
        return _pending("game_not_final")

    stat_group, stat_key = stat_spec
    value, reason = _extract_stat(feed, player, stat_group, stat_key)
    if value is None:
        return _pending(reason or "stat_unavailable")

    try:
        line_value = float(line)
    except (TypeError, ValueError):
        return _pending("invalid_market_line")

    row["settlement_status"] = grade_outcome(value, line_value, direction)
    row["settlement_actual_value"] = value
    row["settlement_source"] = "mlb_statsapi_live_feed"
    row["settlement_checked_at"] = now_iso
    row.pop("settlement_reason", None)
    return True


def iter_settleable_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every row actually rendered on the board or a historical date's
    board -- see predictions.js's mergeLegacySoloBets/renderParlayV2Legs.
    profit_boost_ticket and ticket_ladder are real payload fields but are
    never rendered anywhere on the frontend, so they are deliberately left
    out here."""
    rows: List[Dict[str, Any]] = []
    plays = payload.get("plays")
    if isinstance(plays, list):
        rows.extend(p for p in plays if isinstance(p, dict))

    ticket = ((payload.get("daily_parlay") or {}).get("selected_ticket") or {})
    legs = ticket.get("legs")
    if isinstance(legs, list):
        rows.extend(leg for leg in legs if isinstance(leg, dict))

    parlays = payload.get("parlays") or {}
    for key in ("selected_parlay", "shadow_candidate"):
        pair = parlays.get(key)
        if not isinstance(pair, dict):
            continue
        for leg_key in ("leg_1", "leg_2"):
            leg = pair.get(leg_key)
            if isinstance(leg, dict):
                rows.append(leg)

    return rows


def settle_payload(payload: Dict[str, Any], get_live_feed: Callable[[str], Dict[str, Any]], now_iso: str) -> Dict[str, int]:
    counts = {"won": 0, "lost": 0, "push": 0, "pending": 0, "touched": 0}
    for row in iter_settleable_rows(payload):
        if settle_row(row, get_live_feed, now_iso):
            counts["touched"] += 1
            status = row.get("settlement_status")
            if status in counts:
                counts[status] += 1
    return counts


def settle_file(path: Path, get_live_feed: Callable[[str], Dict[str, Any]], now_iso: str) -> Optional[Dict[str, int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    counts = settle_payload(payload, get_live_feed, now_iso)
    if counts["touched"] > 0:
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return counts


def _merge_counts(total: Dict[str, int], counts: Dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def run(data_dir: Path, only_date: Optional[str] = None, request_timeout: float = 20.0, sleep_between_requests: float = 0.1, report_path: Optional[Path] = None) -> Dict[str, Any]:
    get_live_feed = make_live_feed_fetcher(request_timeout=request_timeout, sleep_between_requests=sleep_between_requests)
    now_iso = datetime.now(timezone.utc).isoformat()

    per_file: Dict[str, Dict[str, int]] = {}
    total = {"won": 0, "lost": 0, "push": 0, "pending": 0, "touched": 0}

    if only_date is None:
        daily_path = data_dir / "daily_predictions.json"
        if daily_path.exists():
            counts = settle_file(daily_path, get_live_feed, now_iso)
            if counts is not None:
                per_file["daily_predictions.json"] = counts
                _merge_counts(total, counts)

    history_dir = data_dir / "history"
    if history_dir.exists():
        history_files = sorted(history_dir.glob("????-??-??.json"))
        if only_date is not None:
            history_files = [p for p in history_files if p.stem == only_date]
        for path in history_files:
            counts = settle_file(path, get_live_feed, now_iso)
            if counts is not None and counts["touched"] > 0:
                per_file[f"history/{path.name}"] = counts
                _merge_counts(total, counts)

    report = {
        "generated_at_utc": now_iso,
        "total": total,
        "files_touched": per_file,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing daily_predictions.json and history/ (default: sports/mlb/web/data)")
    parser.add_argument("--only-date", default=None, help="Settle a single history date (YYYY-MM-DD) instead of every date lacking settlement. Skips daily_predictions.json.")
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.1, help="Courtesy delay between distinct real MLB Stats API requests.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    report = run(
        data_dir=args.data_dir,
        only_date=args.only_date,
        request_timeout=args.request_timeout,
        sleep_between_requests=args.sleep_between_requests,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
