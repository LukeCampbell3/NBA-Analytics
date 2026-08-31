import json
import subprocess

from sports.mlb.unified.promotion_validation import committed_daily_snapshots, run_promotion_validation


def test_committed_snapshot_recovery_is_deduplicated_and_never_labels_missing_fields_exact():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    rows = committed_daily_snapshots(root)
    keys = [(r.run_date, r.play.get("game_id"), r.play.get("player_id") or r.play.get("player"), r.play.get("target"), r.play.get("market_line")) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(not (row.fidelity == "EXACT" and row.reason.startswith("MISSING")) for row in rows)


def test_locked_validation_fails_closed_when_frozen_probability_history_is_missing():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    result = run_promotion_validation(root)
    cert = result["certification"]
    assert cert["status"] == "HISTORICAL_VALIDATION_FAIL"
    assert cert["selected_singles"] == 0
    assert "FROZEN_USABLE_PROBABILITY_HISTORY_UNAVAILABLE" in cert["failures"]
    status = json.loads((root / "artifacts/mlb_unified_production_status.json").read_text())
    assert status["active_engine"] == "legacy"
    assert status["production_authorized"] is False
