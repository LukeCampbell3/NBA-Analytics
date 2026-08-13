from __future__ import annotations

import json

import sports.f1.scripts.run_f1_daily_predictions as runner
from sports.f1.scripts.validate_f1_publication import validate


def test_validator_rejects_staking_enabled(tmp_path) -> None:
    payload = {
        "schema_version": 1, "sport": "f1", "run_date": "2026-08-13", "mode": "live_shadow",
        "model": {"backtest": {"holdout_races": 4}}, "projections": [], "plays": [],
        "selection": {"staking_enabled": True},
    }
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "staking_enabled must remain false" in validate(path)


def test_archived_snapshot_is_graded_from_later_result(tmp_path, monkeypatch) -> None:
    f1_root = tmp_path / "f1"
    history_dir = f1_root / "web/data/history"
    history_dir.mkdir(parents=True)
    archived = {
        "event": {"season": 2026, "round": 1},
        "projections": [
            {"driver_id": "alpha", "driver": "Alpha", "win_probability": 0.7},
            {"driver_id": "bravo", "driver": "Bravo", "win_probability": 0.3},
        ],
        "plays": [{"driver_id": "alpha"}],
    }
    path = history_dir / "2026-03-01.json"
    path.write_text(json.dumps(archived), encoding="utf-8")
    monkeypatch.setattr(runner, "F1_ROOT", f1_root)
    report = runner.grade_archived_snapshots([
        {"season": 2026, "round": 1, "results": [
            {"driver_id": "alpha", "driver": "Alpha", "finish": 1},
            {"driver_id": "bravo", "driver": "Bravo", "finish": 2},
        ]}
    ])
    assert report["settled_snapshots"] == 1
    assert report["top_pick_accuracy"] == 1.0
    assert report["play_hit_rate"] == 1.0
    assert json.loads(path.read_text(encoding="utf-8"))["settlement"]["winner"] == "Alpha"
