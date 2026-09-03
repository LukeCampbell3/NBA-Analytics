#!/usr/bin/env python3
"""Apply the MLB candidate/issuance publication boundary.

The prediction pipeline may rewrite its candidate board on every inference run.
This module makes that mutable output non-authoritative: only PUBLISH and
LATE_ADD may alter the append-only issued-pick ledger, and no mode may mutate
the prediction-time fields of an existing issued pick.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOARD = REPO_ROOT / "sports/mlb/web/data/daily_predictions.json"
DEFAULT_CANDIDATES = REPO_ROOT / "sports/mlb/web/data/latest_candidates.json"
DEFAULT_ISSUED_ROOT = REPO_ROOT / "sports/mlb/web/data/history/issued"
MODES = ("DISCOVER", "PUBLISH", "REFRESH", "LATE_ADD")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _run_date(payload: dict[str, Any]) -> str:
    value = str(payload.get("run_date") or "")
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"invalid run_date: {value!r}")
    return value


def pick_key(play: dict[str, Any]) -> str:
    """Stable wager identity; model outputs and timestamps are deliberately excluded."""
    identity = {
        "player_id": play.get("player_id") or play.get("mlbam_player_id"),
        "player": play.get("player") or play.get("player_name"),
        "game_id": play.get("game_id") or play.get("event_id"),
        "market": play.get("market") or play.get("market_type") or play.get("target"),
        "side": play.get("side") or play.get("direction") or play.get("pick"),
        "line": play.get("line"),
        "sportsbook": play.get("sportsbook") or play.get("book"),
        "odds": play.get("odds") or play.get("american_odds") or play.get("price"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _issued_play(play: dict[str, Any], run_date: str, board: str, number: int, issued_at: str) -> dict[str, Any]:
    result = deepcopy(play)
    key = pick_key(result)
    result["issuance_id"] = f"MLB-{run_date.replace('-', '')}-{board}-{number:03d}"
    result["issuance_key"] = key
    result["issued_at_utc"] = issued_at
    result["issuance_board"] = {"1130": "OFFICIAL", "1730": "LATE_ADD", "LEGACY": "LEGACY_IMPORT"}[board]
    result["prediction_time_fields_immutable"] = True
    return result


def apply_protocol(
    board_path: Path,
    candidates_path: Path,
    issued_root: Path,
    mode: str,
    *,
    prior_board_path: Path | None = None,
) -> dict[str, Any]:
    mode = mode.upper()
    if mode not in MODES:
        raise ValueError(f"unsupported publication mode: {mode}")
    generated = _read(board_path)
    run_date = _run_date(generated)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text(json.dumps(generated, indent=2, sort_keys=True), encoding="utf-8")

    ledger_path = issued_root / f"{run_date}.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        ledger = _read(ledger_path)
    else:
        ledger = {"run_date": run_date, "sport": "mlb", "schema_version": 1, "issued_picks": [], "publication_events": []}

    # One-time migration: retain picks already shown before this protocol was
    # deployed. This prevents the deployment push itself from erasing a board.
    if not ledger["issued_picks"] and prior_board_path and prior_board_path.exists():
        prior = _read(prior_board_path)
        if prior.get("run_date") == run_date and prior.get("plays"):
            issued_at = str(prior.get("generated_at_utc") or datetime.now(timezone.utc).isoformat())
            for number, play in enumerate(prior["plays"], 1):
                ledger["issued_picks"].append(_issued_play(play, run_date, "LEGACY", number, issued_at))
            ledger["publication_events"].append({"mode": "LEGACY_IMPORT", "at_utc": issued_at, "added": len(prior["plays"])})

    existing = {row.get("issuance_key") or pick_key(row) for row in ledger["issued_picks"]}
    additions: list[dict[str, Any]] = []
    may_issue = mode == "LATE_ADD" or (mode == "PUBLISH" and not ledger["issued_picks"])
    if may_issue:
        board_code = "1130" if mode == "PUBLISH" else "1730"
        issued_at = datetime.now(timezone.utc).isoformat()
        for play in generated.get("plays") or []:
            key = pick_key(play)
            if key in existing:
                continue
            additions.append(_issued_play(play, run_date, board_code, len(ledger["issued_picks"]) + len(additions) + 1, issued_at))
            existing.add(key)
        ledger["issued_picks"].extend(additions)
        ledger["publication_events"].append({"mode": mode, "at_utc": issued_at, "added": len(additions)})
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    elif ledger_path.exists() or ledger["issued_picks"]:
        # Migration can create the ledger, but monitoring never rewrites one.
        if not ledger_path.exists():
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")

    public = deepcopy(generated)
    public["plays"] = deepcopy(ledger["issued_picks"])
    public["publication_protocol"] = {
        "mode": mode,
        "candidate_count": len(generated.get("plays") or []),
        "issued_count": len(ledger["issued_picks"]),
        "added_this_run": len(additions),
        "public_pick_retention": 1.0,
        "issued_picks_immutable": True,
        "candidate_source": candidates_path.name,
        "issued_source": str(ledger_path.relative_to(board_path.parent)),
    }
    public["publication_state"] = "PRELIMINARY_DISCOVERY" if not ledger["issued_picks"] else "ISSUED_BOARD"
    board_path.write_text(json.dumps(public, indent=2, sort_keys=True), encoding="utf-8")
    return public["publication_protocol"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--issued-root", type=Path, default=DEFAULT_ISSUED_ROOT)
    parser.add_argument("--prior-board", type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_protocol(args.board, args.candidates, args.issued_root, args.mode, prior_board_path=args.prior_board)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
