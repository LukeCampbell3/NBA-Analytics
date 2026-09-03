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


def preserve(board_path: Path, history_dir: Path) -> Path | None:
    if not board_path.exists():
        return None
    raw = board_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("MLB board must be a JSON object")
    run_date = _valid_date(payload.get("run_date"))
    snapshot_id = re.sub(r"[^A-Za-z0-9_.-]", "_", _snapshot_id(payload, raw))
    target = history_dir / "runs" / run_date / f"{snapshot_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Settlement may have appended outcome fields to the preserved copy.
        # Never replace that evidence with a later rendering of the same run.
        return target
    target.write_bytes(raw)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    target = preserve(args.board, args.history_dir)
    print(json.dumps({"snapshot": str(target) if target else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
