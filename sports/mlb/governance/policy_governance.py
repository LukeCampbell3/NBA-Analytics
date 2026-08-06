#!/usr/bin/env python3
"""Validation and authorization contracts for versioned MLB decision policies."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable


POLICY_STAGES = {
    "POLICY_DEVELOPMENT",
    "LOCKED_VALIDATION",
    "PROSPECTIVE_SHADOW",
    "PRODUCTION_ELIGIBLE",
    "DEMOTED",
    "REJECTED",
}
CERTIFICATE_STATUSES = {"ACTIVE", "DEGRADED", "EXPIRED", "REVOKED", "REJECTED"}
AUTHORIZATION_CONTRACT_VERSION = "MLB_POLICY_AUTHORIZATION_V1"


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_content_digest(policy: dict[str, Any]) -> str:
    return canonical_json_hash({key: value for key, value in policy.items() if key != "policy_digest"})


def parse_strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean evidence value: {value!r}.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _period_bounds(period: dict[str, Any], label: str) -> tuple[date | None, date | None]:
    start_raw = period.get("start")
    end_raw = period.get("end")
    start = date.fromisoformat(str(start_raw)) if start_raw else None
    end = date.fromisoformat(str(end_raw)) if end_raw else None
    if (start is None) != (end is None):
        raise ValueError(f"{label} must declare both start and end or neither.")
    if start is not None and end is not None and end < start:
        raise ValueError(f"{label} ends before it starts.")
    return start, end


def validate_evidence_partitions(evidence: dict[str, Any]) -> None:
    ordered = []
    for name in ("development_period", "locked_validation_period", "prospective_period"):
        period = evidence.get(name)
        if not isinstance(period, dict):
            raise ValueError(f"Missing evidence partition: {name}.")
        start, end = _period_bounds(period, name)
        if start is not None and end is not None:
            ordered.append((name, start, end))
    for left, right in zip(ordered, ordered[1:]):
        if right[1] <= left[2]:
            raise ValueError(f"Evidence partitions overlap or touch: {left[0]} and {right[0]}.")


def validate_policy(policy: dict[str, Any]) -> None:
    required = {
        "policy_version",
        "policy_stage",
        "policy_kind",
        "implementation",
        "scope",
        "decision_rule",
        "support_rule",
        "shift_rule",
        "exposure_controls",
        "settlement_policy",
        "evidence",
        "certificate_requirements",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Policy is missing required fields: {', '.join(missing)}.")
    if str(policy["policy_stage"]) not in POLICY_STAGES:
        raise ValueError(f"Invalid policy stage: {policy['policy_stage']}.")
    if bool(policy.get("staking_enabled", False)):
        raise ValueError("Policy registry cannot enable staking.")
    validate_evidence_partitions(policy["evidence"])
    periods = policy["evidence"]
    development_start, _ = _period_bounds(periods["development_period"], "development_period")
    locked_start, _ = _period_bounds(periods["locked_validation_period"], "locked_validation_period")
    prospective_start, _ = _period_bounds(periods["prospective_period"], "prospective_period")
    if development_start is None:
        raise ValueError("Every policy must declare a development period.")
    stage = str(policy["policy_stage"])
    if stage in {"LOCKED_VALIDATION", "PROSPECTIVE_SHADOW", "PRODUCTION_ELIGIBLE"} and locked_start is None:
        raise ValueError(f"{stage} requires a locked validation period.")
    if stage in {"PROSPECTIVE_SHADOW", "PRODUCTION_ELIGIBLE"} and prospective_start is None:
        raise ValueError(f"{stage} requires a prospective period.")


def load_policy_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("policies"), list):
        raise ValueError("Policy registry must be an object containing a policies list.")
    versions: set[str] = set()
    for policy in payload["policies"]:
        if not isinstance(policy, dict):
            raise ValueError("Each registered policy must be an object.")
        validate_policy(policy)
        version = str(policy["policy_version"])
        if version in versions:
            raise ValueError(f"Duplicate policy version: {version}.")
        versions.add(version)
        policy["policy_digest"] = policy_content_digest(policy)
    return payload


def validate_certificate(certificate: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(certificate.get("certificate_status")) not in CERTIFICATE_STATUSES:
        reasons.append("INVALID_CERTIFICATE_STATUS")
    if str(certificate.get("certificate_status")) != "ACTIVE":
        reasons.append("CERTIFICATE_NOT_ACTIVE")
    if str(certificate.get("policy_version")) != str(policy.get("policy_version")):
        reasons.append("POLICY_VERSION_MISMATCH")
    expected_digest = policy_content_digest(policy)
    if policy.get("policy_digest") and str(policy["policy_digest"]) != expected_digest:
        reasons.append("POLICY_DIGEST_INVALID")
    if str(certificate.get("policy_digest")) != expected_digest:
        reasons.append("POLICY_DIGEST_MISMATCH")
    certificate_digest = certificate.get("certificate_digest")
    unsigned_certificate = {key: value for key, value in certificate.items() if key != "certificate_digest"}
    if not certificate_digest or str(certificate_digest) != canonical_json_hash(unsigned_certificate):
        reasons.append("CERTIFICATE_DIGEST_INVALID")
    if certificate.get("scope") != policy.get("scope"):
        reasons.append("CERTIFICATE_SCOPE_MISMATCH")
    evaluation = certificate.get("evaluation") if isinstance(certificate.get("evaluation"), dict) else {}
    margin = float(evaluation.get("deployment_margin") or 0.0)
    lcb = evaluation.get("anytime_valid_return_lcb")
    if lcb is None or float(lcb) <= margin:
        reasons.append("RETURN_LCB_DOES_NOT_CLEAR_MARGIN")
    gates = certificate.get("authorization_gates") if isinstance(certificate.get("authorization_gates"), dict) else {}
    for metric, minimum in (
        ("resolved_action_slates", "minimum_resolved_action_slates"),
        ("resolved_selections", "minimum_resolved_selections"),
        ("eligible_slates", "minimum_eligible_slates"),
    ):
        if int(evaluation.get(metric) or 0) < int(gates.get(minimum) or 0):
            reasons.append(f"{metric.upper()}_BELOW_MINIMUM")
    if float(evaluation.get("slate_coverage") or 0.0) < float(gates.get("minimum_slate_coverage") or 0.0):
        reasons.append("SLATE_COVERAGE_BELOW_MINIMUM")
    support = certificate.get("support") if isinstance(certificate.get("support"), dict) else {}
    shift = certificate.get("shift") if isinstance(certificate.get("shift"), dict) else {}
    if str(support.get("current_status")) != "IN_SUPPORT":
        reasons.append("CERTIFICATE_SUPPORT_NOT_ACTIVE")
    if str(shift.get("current_status")) not in {"STABLE", "TOLERABLE"}:
        reasons.append("CERTIFICATE_SHIFT_NOT_TOLERABLE")
    if str(evaluation.get("evidence_partition")) != "PROSPECTIVE_SHADOW":
        reasons.append("CERTIFICATE_IS_NOT_PROSPECTIVE")
    if bool(certificate.get("staking_enabled", False)):
        reasons.append("CERTIFICATE_CANNOT_ENABLE_STAKING")
    authorization = certificate.get("authorization") if isinstance(certificate.get("authorization"), dict) else {}
    if not bool(authorization.get("eligible_for_candidate_authorization", False)):
        reasons.append("CERTIFICATE_AUTHORIZATION_GATE_FAILED")
    return sorted(set(reasons))


def authorize_candidate(
    candidate: dict[str, Any],
    *,
    policy: dict[str, Any],
    certificate: dict[str, Any] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if str(policy.get("policy_stage")) != "PRODUCTION_ELIGIBLE":
        reasons.append("POLICY_NOT_PRODUCTION_ELIGIBLE")
    if certificate is None:
        reasons.append("NO_ACTIVE_PROSPECTIVE_CERTIFICATE")
    else:
        reasons.extend(validate_certificate(certificate, policy))

    scope = policy.get("scope") if isinstance(policy.get("scope"), dict) else {}
    market = str(candidate.get("market") or candidate.get("target") or "").upper()
    side = str(candidate.get("side") or candidate.get("direction") or "").upper()
    book = str(candidate.get("book") or candidate.get("selected_sportsbook_key") or "").lower()
    line = candidate.get("line", candidate.get("market_line"))
    decimal_odds = candidate.get("price_decimal")
    if market not in {str(value).upper() for value in scope.get("markets", [])}:
        reasons.append("MARKET_OUTSIDE_CERTIFICATE_SCOPE")
    if side not in {str(value).upper() for value in scope.get("sides", [])}:
        reasons.append("SIDE_OUTSIDE_CERTIFICATE_SCOPE")
    if book not in {str(value).lower() for value in scope.get("books", [])}:
        reasons.append("BOOK_OUTSIDE_CERTIFICATE_SCOPE")
    if line is None or float(line) not in {float(value) for value in scope.get("lines", [])}:
        reasons.append("LINE_OUTSIDE_CERTIFICATE_SCOPE")
    if decimal_odds is None or not (
        float(scope.get("minimum_decimal_odds", 0.0))
        <= float(decimal_odds)
        <= float(scope.get("maximum_decimal_odds", float("inf")))
    ):
        reasons.append("ODDS_OUTSIDE_CERTIFICATE_SCOPE")
    if not bool(candidate.get("generated_by_exact_policy", False)):
        reasons.append("NOT_GENERATED_BY_EXACT_POLICY_VERSION")
    if not bool(candidate.get("inside_support", False)):
        reasons.append("CANDIDATE_OUTSIDE_SUPPORT")
    if str(candidate.get("shift_status", "")).upper() not in {"STABLE", "TOLERABLE"}:
        reasons.append("CANDIDATE_SHIFT_NOT_TOLERABLE")
    for field, reason in (
        ("price_current", "PRICE_NOT_CURRENT"),
        ("price_executable", "PRICE_NOT_EXECUTABLE"),
        ("lineup_confirmed", "LINEUP_NOT_CONFIRMED"),
        ("identity_confirmed", "IDENTITY_NOT_CONFIRMED"),
        ("settlement_supported", "SETTLEMENT_NOT_SUPPORTED"),
        ("exposure_controls_passed", "EXPOSURE_CONTROLS_FAILED"),
    ):
        if not bool(candidate.get(field, False)):
            reasons.append(reason)
    return {
        "authorization_contract_version": AUTHORIZATION_CONTRACT_VERSION,
        "policy_version": policy.get("policy_version"),
        "certificate_id": certificate.get("certificate_id") if certificate else None,
        "candidate_authorized": not reasons,
        "staking_enabled": False,
        "reasons": sorted(set(reasons)),
    }


def active_certificate_for_policy(certificates: Iterable[dict[str, Any]], policy_version: str) -> dict[str, Any] | None:
    active = [
        item for item in certificates
        if str(item.get("policy_version")) == policy_version and str(item.get("certificate_status")) == "ACTIVE"
    ]
    if len(active) > 1:
        raise ValueError(f"Multiple active certificates found for {policy_version}.")
    return active[0] if active else None
