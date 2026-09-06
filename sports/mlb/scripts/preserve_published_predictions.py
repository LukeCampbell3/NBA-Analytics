#!/usr/bin/env python3
"""Preserve each MLB publication before a later run can replace it.

The dated board remains the user-facing daily archive.  This ledger adds one
immutable snapshot per actual publication, including same-day reruns, so picks
that disappear from a later board are still available for settlement/audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOARD = REPO_ROOT / "sports/mlb/web/data/daily_predictions.json"
DEFAULT_HISTORY = REPO_ROOT / "sports/mlb/web/data/history"
PRODUCT_FILENAMES = (
    "same_game_predictions.json",
    "pitcher_parlay_predictions.json",
    "high_hit_parlay_predictions.json",
    "exotic_market_predictions.json",
)


def _valid_date(value: Any) -> str:
    token = str(value or "").strip()
    if date.fromisoformat(token).isoformat() != token:
        raise ValueError(f"invalid run_date: {token!r}")
    return token


def _snapshot_id(payload: dict[str, Any], raw: bytes) -> str:
    generated = str(payload.get("generated_at_utc") or "").strip()
    if generated:
        try:
            stamp = datetime.fromisoformat(generated.replace("Z", "+00:00")).astimezone(timezone.utc)
            return stamp.strftime("%Y%m%dT%H%M%S.%fZ")
        except ValueError:
            pass
    return hashlib.sha256(raw).hexdigest()[:20]


def _selection_count(filename: str, payload: dict[str, Any]) -> int:
    if filename == "daily_predictions.json":
        return len(payload.get("plays") or [])
    if filename == "same_game_predictions.json":
        return sum(len(game.get("combo_candidates") or []) for game in payload.get("games") or [] if isinstance(game, dict))
    if filename == "pitcher_parlay_predictions.json":
        return 1 if payload.get("parlay") or payload.get("max_hit_control") else 0
    if filename == "high_hit_parlay_predictions.json":
        return len(payload.get("parlays") or []) + int(bool(payload.get("shadow_fallback")))
    if filename == "exotic_market_predictions.json":
        return len(payload.get("candidates") or [])
    return 0


def preserve(board_path: Path, history_dir: Path, *, product: bool = False) -> Path | None:
    if not board_path.exists():
        return None
    raw = board_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("MLB board must be a JSON object")
    run_date = _valid_date(payload.get("run_date"))
    snapshot_id = re.sub(r"[^A-Za-z0-9_.-]", "_", _snapshot_id(payload, raw))
    run_root = history_dir / "runs" / run_date / snapshot_id
    target = run_root / board_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Settlement may have appended outcome fields to the preserved copy.
        # Never replace that evidence with a later rendering of the same run.
        return target
    target.write_bytes(raw)

    # The date view exposes the richest preserved publication for each
    # product. A later empty/provider-failed run must never erase an earlier
    # slate that users actually saw.
    dated = (history_dir / "products" / run_date / board_path.name) if product else (history_dir / f"{run_date}.json")
    dated.parent.mkdir(parents=True, exist_ok=True)
    replace = not dated.exists()
    if dated.exists():
        try:
            existing = json.loads(dated.read_text(encoding="utf-8"))
            replace = _selection_count(board_path.name, payload) > _selection_count(board_path.name, existing)
        except (OSError, json.JSONDecodeError):
            replace = False
    if replace:
        dated.write_bytes(raw)
    return target


def refresh_history_index(history_dir: Path) -> Path:
    """Derive user-facing history navigation from the dated archive files."""
    history_dir.mkdir(parents=True, exist_ok=True)
    dates: list[str] = []
    for path in history_dir.glob("????-??-??.json"):
        try:
            dates.append(_valid_date(path.stem))
        except ValueError:
            continue
    dates = sorted(set(dates), reverse=True)
    target = history_dir / "index.json"
    payload = {
        "dates": dates,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def preserve_all(board_path: Path, history_dir: Path) -> list[Path]:
    preserved: list[Path] = []
    primary = preserve(board_path, history_dir)
    if primary:
        preserved.append(primary)
    for filename in PRODUCT_FILENAMES:
        target = preserve(board_path.parent / filename, history_dir, product=True)
        if target:
            preserved.append(target)
    refresh_history_index(history_dir)
    return preserved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    targets = preserve_all(args.board, args.history_dir)
    print(json.dumps({"snapshots": [str(target) for target in targets]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
