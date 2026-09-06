from __future__ import annotations

import pytest

from sports.mlb.scripts import run_unified_mlb_shadow as runner
from sports.mlb.unified.policy_manifest import FROZEN_POLICY_COMMIT


def _locked_manifest() -> dict:
    return {
        "active_engine": "legacy",
        "fallback_engine": "legacy",
        "production_state": "LOCKED_HISTORICAL_VALIDATION_FAILED",
        "certified_capabilities": [],
        "unified_policy_commit": FROZEN_POLICY_COMMIT,
        "policy_hash": "old-hash",
        "generated_at": "2026-09-02T00:00:00Z",
    }


def test_runtime_hash_drift_is_recorded_without_promoting_authority(monkeypatch):
    monkeypatch.setattr(runner, "_current_commit", lambda: "abc123")
    result = runner._runtime_manifest(_locked_manifest(), "new-runtime-hash")

    assert result["policy_hash"] == "new-runtime-hash"
    assert result["prior_policy_hash"] == "old-hash"
    assert result["governance_drift_detected"] is True
    assert result["active_engine"] == "legacy"
    assert result["fallback_engine"] == "legacy"
    assert result["certified_capabilities"] == []
    assert result["implementation_commit"] == "abc123"


def test_runtime_hash_reconciliation_refuses_authority_bearing_state():
    manifest = _locked_manifest()
    manifest["production_state"] = "PRODUCTION_ACTIVE"
    with pytest.raises(ValueError, match="authority-bearing"):
        runner._runtime_manifest(manifest, "new-runtime-hash")


def test_runtime_hash_reconciliation_refuses_unified_active_engine():
    manifest = _locked_manifest()
    manifest["active_engine"] = "unified"
    with pytest.raises(ValueError, match="legacy remains active"):
        runner._runtime_manifest(manifest, "new-runtime-hash")


def test_matching_hash_does_not_create_drift():
    manifest = _locked_manifest()
    manifest["policy_hash"] = "same"
    result = runner._runtime_manifest(manifest, "same")
    assert result["governance_drift_detected"] is False
    assert "prior_policy_hash" not in result
