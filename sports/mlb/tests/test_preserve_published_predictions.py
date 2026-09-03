from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import preserve_published_predictions as preserve  # noqa: E402


def test_preserves_every_same_day_publication_without_overwriting(tmp_path: Path) -> None:
    board = tmp_path / "daily_predictions.json"
    history = tmp_path / "history"
    first = {
        "run_date": "2026-09-03",
        "generated_at_utc": "2026-09-03T12:00:00Z",
        "plays": [{"player": "First Pick", "game_id": "1"}],
    }
    board.write_text(json.dumps(first), encoding="utf-8")
    first_path = preserve.preserve(board, history)

    second = {
        "run_date": "2026-09-03",
        "generated_at_utc": "2026-09-03T18:30:00Z",
        "plays": [{"player": "Later Pick", "game_id": "2"}],
    }
    board.write_text(json.dumps(second), encoding="utf-8")
    second_path = preserve.preserve(board, history)

    assert first_path != second_path
    assert json.loads(first_path.read_text())["plays"][0]["player"] == "First Pick"
    assert json.loads(second_path.read_text())["plays"][0]["player"] == "Later Pick"


def test_existing_snapshot_is_never_replaced_after_settlement(tmp_path: Path) -> None:
    board = tmp_path / "daily_predictions.json"
    history = tmp_path / "history"
    payload = {
        "run_date": "2026-09-03",
        "generated_at_utc": "2026-09-03T12:00:00Z",
        "plays": [{"player": "Tracked Pick", "game_id": "1"}],
    }
    board.write_text(json.dumps(payload), encoding="utf-8")
    target = preserve.preserve(board, history)
    settled = json.loads(target.read_text())
    settled["plays"][0]["settlement_status"] = "won"
    target.write_text(json.dumps(settled), encoding="utf-8")

    assert preserve.preserve(board, history) == target
    assert json.loads(target.read_text())["plays"][0]["settlement_status"] == "won"
