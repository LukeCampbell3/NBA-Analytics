import json
import subprocess

from sports.mlb.unified.promotion_validation import certification, committed_daily_snapshots, run_promotion_validation


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


def test_certification_is_not_hardcoded_to_fail_when_a_capability_earns_every_gate():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    policy = json.loads((root / "config/mlb_unified_promotion_policy.json").read_text())
    rows = []
    for slate in range(20):
        for pick in range(3):
            rows.append({
                "event_date": f"2026-07-{slate+1:02d}", "market": "H", "eligible": True,
                "usable_probability": .95, "quoted_odds": -110, "settlement": "won",
                "book": "fanduel" if (slate * 3 + pick) % 2 else "draftkings",
            })
        rows.append({
            "event_date": f"2026-07-{slate+1:02d}", "market": "H", "eligible": False,
            "usable_probability": .45, "quoted_odds": 110, "settlement": "lost", "book": "fanduel",
        })
    result = certification(root, policy, rows, [])
    assert result["status"] == "HISTORICAL_VALIDATION_PARTIAL"
    assert result["capabilities"]["batter_hits"] == "CERTIFIED"
    assert "CAPABILITY_LEVEL_SAMPLE_REQUIREMENTS_NOT_MET" not in result["failures"]
