#!/usr/bin/env python3
"""Maintain an append-only lifecycle record for NFL picks shown publicly.

The live board is intentionally allowed to change before kickoff as prices,
roles, and market availability change. This ledger makes those changes
auditable instead of allowing a later JSON replacement to erase them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
NFL_ROOT = REPO_ROOT / "sports" / "nfl"
DAILY_PATH = NFL_ROOT / "web/data/daily_predictions.json"
MARKET_PATH = NFL_ROOT / "web/data/week_1_market_board.json"
OUTPUT_PATH = NFL_ROOT / "web/data/pick_history.json"


def parse_time(value: object) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def normalized_pick(row: dict, product: str) -> dict | None:
    player = str(row.get("player") or row.get("player_display_name") or "").strip()
    player_id = str(row.get("player_id") or "").strip()
    event_id = str(row.get("event_id") or row.get("game_id") or "").strip()
    market = str(row.get("market") or row.get("target") or "").strip()
    side = str(row.get("direction") or row.get("side") or "").strip().upper()
    line = row.get("line", row.get("market_line"))
    if not player or not market or side not in {"OVER", "UNDER"} or line is None:
        return None
    kickoff = row.get("game_start_utc") or row.get("kickoff_utc")
    book = row.get("selected_sportsbook_key") or row.get("bookmaker") or row.get("book")
    price = row.get("selected_side_price", row.get("price"))
    identity = "|".join(
        [product, event_id, player_id or player.lower(), market, side, str(line), str(book or "")]
    )
    return {
        "pick_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "product": product,
        "player": player,
        "player_id": player_id or None,
        "event_id": event_id or None,
        "team": row.get("team"),
        "opponent": row.get("opponent"),
        "market": market,
        "side": side,
        "line": line,
        "book": book,
        "price": price,
        "projection": row.get("projection"),
        "model_probability": row.get("model_hit_probability", row.get("raw_model_probability")),
        "kickoff_utc": kickoff,
        "selection_status": row.get("selection_status") or row.get("action_status"),
        "candidate_authorized": bool(row.get("candidate_authorized", False)),
    }


def published_picks(daily: dict, market: dict) -> list[dict]:
    result: list[dict] = []
    daily_plays = daily.get("plays") if isinstance(daily.get("plays"), list) else []
    weekly_singles = market.get("best_available_singles") if isinstance(market.get("best_available_singles"), list) else []
    singles = daily_plays if daily_plays else weekly_singles
    for row in singles:
        pick = normalized_pick(row, "qualified_single") if isinstance(row, dict) else None
        if pick:
            result.append(pick)
    pools = market.get("pools") if isinstance(market.get("pools"), dict) else {}
    for product, rows in pools.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            pick = normalized_pick(row, str(product)) if isinstance(row, dict) else None
            if pick:
                result.append(pick)
    return list({row["pick_id"]: row for row in result}.values())


def empty_ledger() -> dict:
    return {
        "schema_version": 1,
        "generated_at_utc": None,
        "summary": {},
        "snapshots": [],
        "picks": [],
    }


def observe(ledger: dict, daily: dict, market: dict, *, source: str) -> dict:
    picks = published_picks(daily, market)
    observed_candidates = [
        value
        for value in (
            parse_time(daily.get("generated_at_utc")),
            parse_time(market.get("generated_at_utc")),
        )
        if value is not None
    ]
    observed = max(observed_candidates) if observed_candidates else datetime.now(timezone.utc)
    observed_at = iso_time(observed)
    run_date = str(daily.get("run_date") or market.get("run_date") or "")
    pick_ids = sorted(row["pick_id"] for row in picks)
    publication_seed = json.dumps([observed_at, pick_ids], separators=(",", ":"))
    publication_id = hashlib.sha256(publication_seed.encode("utf-8")).hexdigest()[:20]

    snapshots = ledger.setdefault("snapshots", [])
    if any(row.get("publication_id") == publication_id for row in snapshots):
        return ledger

    records = {row.get("pick_id"): row for row in ledger.setdefault("picks", []) if row.get("pick_id")}
    current_ids = set(pick_ids)
    for pick_id, record in records.items():
        if record.get("status") != "ACTIVE" or pick_id in current_ids:
            continue
        kickoff = parse_time(record.get("pick", {}).get("kickoff_utc"))
        status = "LOCKED_AFTER_KICKOFF" if kickoff and observed >= kickoff else "REMOVED_BEFORE_KICKOFF"
        reason = "GAME_STARTED" if status == "LOCKED_AFTER_KICKOFF" else "NO_LONGER_SELECTED_ON_REFRESH"
        record["status"] = status
        record["status_changed_at_utc"] = observed_at
        record["removal_reason"] = reason
        record.setdefault("events", []).append({"at_utc": observed_at, "event": status, "source": source})

    for pick in picks:
        pick_id = pick["pick_id"]
        record = records.get(pick_id)
        if record is None:
            record = {
                "pick_id": pick_id,
                "first_published_at_utc": observed_at,
                "last_published_at_utc": observed_at,
                "status": "ACTIVE",
                "status_changed_at_utc": observed_at,
                "appearances": 1,
                "pick": pick,
                "events": [{"at_utc": observed_at, "event": "PUBLISHED", "source": source}],
            }
            records[pick_id] = record
        else:
            if record.get("status") != "ACTIVE":
                record.setdefault("events", []).append({"at_utc": observed_at, "event": "REPUBLISHED", "source": source})
            record["status"] = "ACTIVE"
            record["status_changed_at_utc"] = observed_at
            record["last_published_at_utc"] = observed_at
            record["appearances"] = int(record.get("appearances") or 0) + 1
            record["removal_reason"] = None
            record["pick"] = pick

    snapshots.append(
        {
            "publication_id": publication_id,
            "run_date": run_date,
            "generated_at_utc": observed_at,
            "source": source,
            "pick_ids": pick_ids,
            "pick_count": len(pick_ids),
        }
    )
    ledger["picks"] = sorted(records.values(), key=lambda row: row.get("first_published_at_utc", ""), reverse=True)
    counts: dict[str, int] = {}
    for record in ledger["picks"]:
        status = str(record.get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    ledger["generated_at_utc"] = observed_at
    ledger["summary"] = {"snapshots": len(snapshots), "total_picks": len(records), **counts}
    return ledger


def git_json(commit: str, path: Path) -> dict:
    relative = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    if result.returncode:
        return {}
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def historical_commits(paths: Iterable[Path]) -> list[str]:
    relative = [path.relative_to(REPO_ROOT).as_posix() for path in paths]
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--", *relative],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return list(dict.fromkeys(line.strip() for line in result.stdout.splitlines() if line.strip()))


def git_commit_time(commit: str) -> str:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%cI", commit],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return iso_time(parse_time(result.stdout.strip()) or datetime.now(timezone.utc))


def recover_history() -> dict:
    ledger = empty_ledger()
    for commit in historical_commits((DAILY_PATH, MARKET_PATH)):
        daily = git_json(commit, DAILY_PATH)
        market = git_json(commit, MARKET_PATH)
        if daily or market:
            committed_at = git_commit_time(commit)
            if daily and not daily.get("generated_at_utc"):
                daily["generated_at_utc"] = committed_at
            if market and not market.get("generated_at_utc"):
                market["generated_at_utc"] = committed_at
            observe(ledger, daily, market, source=f"git:{commit[:12]}")
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DAILY_PATH)
    parser.add_argument("--market", type=Path, default=MARKET_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--recover-git-history-if-empty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger = load_json(args.output)
    if not ledger and args.recover_git_history_if_empty:
        ledger = recover_history()
    if not ledger:
        ledger = empty_ledger()
    observe(ledger, load_json(args.daily), load_json(args.market), source="current_publication")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ledger.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
