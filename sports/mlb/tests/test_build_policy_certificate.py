from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sports.mlb.governance.build_policy_certificate import build_certificate, anytime_hoeffding_interval
from sports.mlb.governance.policy_governance import authorize_candidate, canonical_json_hash, load_policy_registry


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "governance" / "policies" / "mlb_policy_family_v1.json"


def _production_policy() -> dict:
    policy = json.loads(json.dumps(load_policy_registry(REGISTRY_PATH)["policies"][0]))
    policy.pop("policy_digest")
    policy["policy_stage"] = "PRODUCTION_ELIGIBLE"
    policy["evidence"]["locked_validation_period"] = {"start": "2026-08-06", "end": "2026-08-31"}
    policy["evidence"]["prospective_period"] = {"start": "2026-09-01", "end": "2029-12-31"}
    policy["certificate_requirements"].update(
        {
            "minimum_eligible_slates": 100,
            "minimum_resolved_action_slates": 100,
            "minimum_resolved_selections": 100,
            "minimum_slate_coverage": 0.1,
            "minimum_candidate_coverage": 0.01,
            "maximum_losing_action_slate_rate": 0.2,
        }
    )
    policy["policy_digest"] = canonical_json_hash(policy)
    return policy


def _prospective_evidence(policy: dict, rows: int = 1000) -> pd.DataFrame:
    dates = pd.date_range("2026-09-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "slate_id": [f"MLB_{value:%Y%m%d}" for value in dates],
            "snapshot_id": [f"snapshot-{index}" for index in range(rows)],
            "slate_date": dates,
            "policy_version": policy["policy_version"],
            "policy_digest": policy["policy_digest"],
            "evidence_partition": "PROSPECTIVE_SHADOW",
            "capture_label": "FULL_SLATE_SNAPSHOT",
            "decision_frozen_at_utc": dates.tz_localize("UTC"),
            "eligible_slate": True,
            "action_taken": True,
            "resolved": True,
            "selection_count": 1,
            "eligible_candidate_count": 10,
            "selected_candidate_count": 1,
            "daily_unit_return": 0.5,
        }
    )


def test_anytime_interval_rejects_return_outside_declared_scope() -> None:
    with pytest.raises(ValueError, match="outside"):
        anytime_hoeffding_interval([3.0], lower_bound=-1.0, upper_bound=2.0, delta=0.05)


def test_prospective_certificate_can_authorize_only_exact_policy_candidate() -> None:
    policy = _production_policy()
    certificate = build_certificate(
        policy=policy,
        evidence=_prospective_evidence(policy),
        support_status="IN_SUPPORT",
        shift_status="STABLE",
        dependence_stress_status="PASSED",
    )
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

    authorization = authorize_candidate(candidate, policy=policy, certificate=certificate)

    assert certificate["certificate_status"] == "ACTIVE"
    assert certificate["evaluation"]["anytime_valid_return_lcb"] > certificate["evaluation"]["deployment_margin"]
    assert authorization["candidate_authorized"] is True
    assert authorization["staking_enabled"] is False


def test_policy_change_invalidates_certificate_digest() -> None:
    policy = _production_policy()
    certificate = build_certificate(
        policy=policy,
        evidence=_prospective_evidence(policy),
        support_status="IN_SUPPORT",
        shift_status="STABLE",
        dependence_stress_status="PASSED",
    )
    changed = json.loads(json.dumps(policy))
    changed["scope"]["maximum_decimal_odds"] = 4.0
    changed["policy_digest"] = canonical_json_hash({key: value for key, value in changed.items() if key != "policy_digest"})

    authorization = authorize_candidate(
        {
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
        },
        policy=changed,
        certificate=certificate,
    )

    assert authorization["candidate_authorized"] is False
    assert "POLICY_DIGEST_MISMATCH" in authorization["reasons"]


def test_prospective_certificate_rejects_evidence_from_different_policy_digest() -> None:
    policy = _production_policy()
    evidence = _prospective_evidence(policy)
    evidence["policy_digest"] = "0" * 64

    with pytest.raises(ValueError, match="exact policy digest"):
        build_certificate(
            policy=policy,
            evidence=evidence,
            support_status="IN_SUPPORT",
            shift_status="STABLE",
            dependence_stress_status="PASSED",
        )


def test_prospective_certificate_rejects_ambiguous_string_booleans() -> None:
    policy = _production_policy()
    evidence = _prospective_evidence(policy)
    evidence["eligible_slate"] = evidence["eligible_slate"].astype(object)
    evidence.loc[0, "eligible_slate"] = "not-a-boolean"

    with pytest.raises(ValueError, match="Invalid boolean evidence value"):
        build_certificate(
            policy=policy,
            evidence=evidence,
            support_status="IN_SUPPORT",
            shift_status="STABLE",
            dependence_stress_status="PASSED",
        )
