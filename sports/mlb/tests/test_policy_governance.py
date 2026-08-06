from __future__ import annotations

import json
from pathlib import Path

import pytest

from sports.mlb.governance.policy_governance import (
    authorize_candidate,
    canonical_json_hash,
    load_policy_registry,
    parse_strict_bool,
    validate_evidence_partitions,
)


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "governance" / "policies" / "mlb_policy_family_v1.json"


def test_policy_registry_is_bounded_and_uncertified() -> None:
    registry = load_policy_registry(REGISTRY_PATH)

    assert len(registry["policies"]) == 2
    assert all(policy["policy_stage"] == "POLICY_DEVELOPMENT" for policy in registry["policies"])
    assert all(len(policy["policy_digest"]) == 64 for policy in registry["policies"])
    assert all(policy["staking_enabled"] is False for policy in registry["policies"])


def test_evidence_partitions_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_evidence_partitions(
            {
                "development_period": {"start": "2026-01-01", "end": "2026-04-01"},
                "locked_validation_period": {"start": "2026-04-01", "end": "2026-05-01"},
                "prospective_period": {"start": None, "end": None},
            }
        )


def test_candidate_is_denied_without_prospective_certificate() -> None:
    policy = load_policy_registry(REGISTRY_PATH)["policies"][0]
    candidate = {
        "market": "H",
        "side": "OVER",
        "book": "draftkings",
        "line": 0.5,
        "price_decimal": 1.8,
        "generated_by_exact_policy": True,
        "inside_support": True,
        "shift_status": "STABLE",
        "price_current": True,
        "price_executable": True,
        "lineup_confirmed": True,
        "identity_confirmed": True,
        "settlement_supported": True,
        "exposure_controls_passed": True,
    }

    result = authorize_candidate(candidate, policy=policy, certificate=None)

    assert result["candidate_authorized"] is False
    assert "NO_ACTIVE_PROSPECTIVE_CERTIFICATE" in result["reasons"]
    assert "POLICY_NOT_PRODUCTION_ELIGIBLE" in result["reasons"]


def test_certificate_digest_binds_exact_policy() -> None:
    policy = load_policy_registry(REGISTRY_PATH)["policies"][0]
    original_digest = policy["policy_digest"]
    policy_copy = json.loads(json.dumps(policy))
    policy_copy.pop("policy_digest")
    policy_copy["decision_rule"]["maximum_daily_selections"] = 4

    assert canonical_json_hash(policy_copy) != original_digest


def test_strict_boolean_parser_does_not_treat_arbitrary_text_as_true() -> None:
    assert parse_strict_bool("false") is False
    assert parse_strict_bool("true") is True
    with pytest.raises(ValueError, match="Invalid boolean"):
        parse_strict_bool("unknown")
