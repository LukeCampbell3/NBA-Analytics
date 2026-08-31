import json
import subprocess

import pytest

from sports.mlb.unified.historical_settlements import validate_historical_settlement
from sports.mlb.unified.promotion_validation import (
    build_corpus, certification, committed_daily_snapshots,
    recovered_workflow_snapshots, run_promotion_validation,
)


def test_committed_snapshot_recovery_is_deduplicated_and_never_labels_missing_fields_exact():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    rows = committed_daily_snapshots(root)
    keys = [(r.run_date, r.play.get("game_id"), r.play.get("player_id") or r.play.get("player"), r.play.get("target"), r.play.get("market_line")) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(not (row.fidelity == "EXACT" and row.reason.startswith("MISSING")) for row in rows)


def test_recovered_workflow_outputs_preserve_exact_pregame_contract():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    rows = recovered_workflow_snapshots(root)
    assert len(rows) == 9
    assert {row.source_artifact_id for row in rows} == {9734123174, 9735656814, 9737132236}
    assert all(row.fidelity == "EXACT" for row in rows)
    assert all(row.source_kind == "GITHUB_ACTIONS_ARTIFACT" for row in rows)


def test_locked_validation_contains_eight_exact_observations_but_fails_sample_gate():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    eligible, _ = build_corpus(root)
    assert len(eligible) == 8
    pca = next(row for row in eligible if row["player_id"] == "pete_crow-armstrong")
    assert pca["settlement"] == "won"
    assert pca["actual_value"] == 8
    assert pca["usable_probability"] == pytest.approx(0.613848)
    assert pca["conservative_expected_value"] == pytest.approx(0.08603876923076914)
    assert pca["snapshot_candidate_sha256"] == "f6aa1bbaf4bbd8b2f51749a94ff2b5c046af04ca0e668b743ff5d0927555431f"
    assert pca["settlement_source_sha256"] == "3581bdb4884d0042ff1d346ffefa7378dcb7abe737aba163e7ad86232b428c1b"
    result = run_promotion_validation(root)
    cert = result["certification"]
    assert cert["status"] == "HISTORICAL_VALIDATION_FAIL"
    assert cert["selected_singles"] == 8
    assert cert["capability_results"]["batter_hits"]["selected_singles"] == 6
    assert cert["capability_results"]["batter_total_bases"]["selected_singles"] == 2
    assert "INDEPENDENT_SLATES:1<20" in cert["failures"]
    assert "SELECTED_SINGLES:8<50" in cert["failures"]
    status = json.loads((root / "artifacts/mlb_unified_production_status.json").read_text())
    assert status["active_engine"] == "legacy"
    assert status["production_authorized"] is False


def test_external_settlement_fails_closed_on_unapproved_source_or_wrong_result():
    record = {
        "source_commit": "a" * 40, "game_id": "1", "player_id": "p", "market": "TB",
        "side": "OVER", "line": 1.5, "actual_value": 8, "settlement": "won",
        "source_type": "MLB_STATSAPI_FINAL_FEED", "source_url": "https://statsapi.mlb.com/x",
        "source_sha256": "b" * 64, "game_finalized_at_utc": "2026-08-31T02:00:00Z",
        "retrieved_at_utc": "2026-08-31T03:00:00Z",
    }
    validate_historical_settlement(record, "2026-08-30T20:00:00Z")
    with pytest.raises(ValueError, match="not approved"):
        validate_historical_settlement({**record, "source_type": "UNVERIFIED"}, "2026-08-30T20:00:00Z")
    with pytest.raises(ValueError, match="disagrees"):
        validate_historical_settlement({**record, "settlement": "lost"}, "2026-08-30T20:00:00Z")
    with pytest.raises(ValueError, match="predate"):
        validate_historical_settlement({**record, "game_finalized_at_utc": "2026-08-30T19:00:00Z"}, "2026-08-30T20:00:00Z")


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


def test_fifty_mixed_market_selections_do_not_certify_either_capability():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    policy = json.loads((root / "config/mlb_unified_promotion_policy.json").read_text())
    rows = []
    for index in range(50):
        rows.append({
            "event_date": f"2026-07-{index % 20 + 1:02d}",
            "market": "H" if index < 25 else "TB", "eligible": True,
            "usable_probability": .95, "quoted_odds": -110,
            "settlement": "won", "book": "fanduel" if index % 2 else "draftkings",
        })
    result = certification(root, policy, rows, [])
    assert result["capabilities"]["batter_hits"] == "VALIDATION_ONLY"
    assert result["capabilities"]["batter_total_bases"] == "VALIDATION_ONLY"
    assert result["capability_results"]["batter_hits"]["selected_singles"] == 25
    assert result["capability_results"]["batter_total_bases"]["selected_singles"] == 25
