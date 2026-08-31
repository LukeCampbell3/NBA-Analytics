import json

import pytest

from sports.mlb.unified.policy_manifest import build_policy_manifest, verify_policy_manifest
from sports.mlb.unified.production_state import (
    CapabilityAuthority,
    EngineState,
    assert_transition,
    atomic_write_json,
    build_engine_manifest,
    validate_manifest,
)


def test_frozen_policy_hash_is_canonical_and_tamper_evident():
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    manifest = build_policy_manifest(root)
    assert verify_policy_manifest(manifest)
    manifest["decision_policy"]["minimum_usable_probability"] = .59
    assert not verify_policy_manifest(manifest)


def test_state_machine_rejects_skipping_dark_deploy():
    assert_transition(EngineState.PRODUCTION_CANDIDATE, EngineState.PRODUCTION_DEPLOYED_DARK)
    with pytest.raises(ValueError):
        assert_transition(EngineState.PRODUCTION_CANDIDATE, EngineState.PRODUCTION_ACTIVE)


def test_atomic_manifest_and_authority_consistency(tmp_path):
    payload = build_engine_manifest(
        policy_hash="a" * 64,
        implementation_commit="b" * 40,
        state=EngineState.PRODUCTION_DEPLOYED_DARK,
        capabilities={"batter_hits": CapabilityAuthority.SHADOW.value},
    )
    validate_manifest(payload)
    path = tmp_path / "manifest.json"
    atomic_write_json(path, payload)
    assert json.loads(path.read_text()) == payload
    payload["active_engine"] = "unified"
    with pytest.raises(ValueError):
        validate_manifest(payload)
