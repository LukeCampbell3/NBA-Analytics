#!/usr/bin/env python3
"""Build a deny-by-default prospective MLB policy certificate from daily returns."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.governance.policy_governance import (
    canonical_json_hash,
    load_policy_registry,
    parse_strict_bool,
    policy_content_digest,
)


REQUIRED_EVIDENCE_COLUMNS = {
    "slate_id",
    "snapshot_id",
    "slate_date",
    "policy_version",
    "policy_digest",
    "evidence_partition",
    "capture_label",
    "decision_frozen_at_utc",
    "eligible_slate",
    "action_taken",
    "resolved",
    "selection_count",
    "eligible_candidate_count",
    "selected_candidate_count",
    "daily_unit_return",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a prospective MLB policy certificate.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-status", choices=["IN_SUPPORT", "OUT_OF_SUPPORT"], required=True)
    parser.add_argument("--shift-status", choices=["STABLE", "TOLERABLE", "BLOCKING"], required=True)
    parser.add_argument("--dependence-stress-status", choices=["PASSED", "FAILED", "NOT_RUN"], required=True)
    return parser.parse_args()


def anytime_hoeffding_interval(
    values: Iterable[float],
    *,
    lower_bound: float,
    upper_bound: float,
    delta: float,
) -> tuple[float | None, float | None]:
    observations = [float(value) for value in values]
    if not observations:
        return None, None
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one.")
    if upper_bound <= lower_bound:
        raise ValueError("Return upper bound must exceed its lower bound.")
    if any(value < lower_bound - 1e-12 or value > upper_bound + 1e-12 for value in observations):
        raise ValueError("Observed return lies outside the policy's declared bounded-return scope.")
    count = len(observations)
    mean = sum(observations) / count
    # Sum_n delta/[n(n+1)] = delta, so a union bound makes this interval time-uniform.
    point_delta = delta / (count * (count + 1))
    radius = (upper_bound - lower_bound) * math.sqrt(math.log(1.0 / point_delta) / (2.0 * count))
    return max(lower_bound, mean - radius), min(upper_bound, mean + radius)


def _policy_by_version(registry: dict[str, Any], version: str) -> dict[str, Any]:
    matches = [policy for policy in registry["policies"] if str(policy["policy_version"]) == version]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registered policy named {version}.")
    return matches[0]


def build_certificate(
    *,
    policy: dict[str, Any],
    evidence: pd.DataFrame,
    support_status: str,
    shift_status: str,
    dependence_stress_status: str,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_EVIDENCE_COLUMNS - set(evidence.columns))
    if missing:
        raise ValueError(f"Prospective evidence is missing columns: {', '.join(missing)}")
    frame = evidence.copy()
    frame["slate_date"] = pd.to_datetime(frame["slate_date"], errors="raise").dt.date
    if frame.empty:
        raise ValueError("Prospective evidence is empty.")
    if frame["slate_id"].astype(str).str.strip().eq("").any() or frame["snapshot_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every prospective evidence row must identify its slate and immutable snapshot.")
    if frame["slate_id"].duplicated().any() or frame["slate_date"].duplicated().any():
        raise ValueError("Prospective evidence must contain exactly one row per daily slate.")
    expected_digest = policy_content_digest(policy)
    if str(policy.get("policy_digest")) != expected_digest:
        raise ValueError("Policy digest does not match the exact policy contents.")
    if set(frame["policy_version"].astype(str)) != {str(policy["policy_version"])}:
        raise ValueError("Prospective evidence contains a different policy version.")
    if set(frame["policy_digest"].astype(str)) != {expected_digest}:
        raise ValueError("Prospective evidence is not bound to the exact policy digest.")
    if set(frame["evidence_partition"].astype(str).str.upper()) != {"PROSPECTIVE_SHADOW"}:
        raise ValueError("Certificate evidence must be labeled PROSPECTIVE_SHADOW.")
    if set(frame["capture_label"].astype(str).str.upper()) != {"FULL_SLATE_SNAPSHOT"}:
        raise ValueError("Certificate evidence must derive from complete-slate snapshots.")
    frozen_at = pd.to_datetime(frame["decision_frozen_at_utc"], utc=True, errors="coerce")
    if frozen_at.isna().any():
        raise ValueError("Every prospective decision must have a valid pre-event freeze timestamp.")
    evidence_period = policy["evidence"]["prospective_period"]
    if not evidence_period.get("start") or not evidence_period.get("end"):
        raise ValueError("Policy has no frozen prospective evidence period.")
    period_start = pd.Timestamp(evidence_period["start"]).date()
    period_end = pd.Timestamp(evidence_period["end"]).date()
    if frame["slate_date"].min() < period_start or frame["slate_date"].max() > period_end:
        raise ValueError("Prospective evidence falls outside the policy's frozen prospective period.")

    for column in ("eligible_slate", "action_taken", "resolved"):
        frame[column] = frame[column].map(parse_strict_bool)
    for column in ("selection_count", "eligible_candidate_count", "selected_candidate_count", "daily_unit_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    eligible = frame.loc[frame["eligible_slate"]].copy()
    action = eligible.loc[eligible["action_taken"]].copy()
    resolved_action = action.loc[action["resolved"] & action["daily_unit_return"].notna()].copy()
    unresolved_action = action.loc[~action["resolved"] | action["daily_unit_return"].isna()].copy()
    resolved_calendar = eligible.loc[(~eligible["action_taken"]) | eligible["resolved"]].copy()
    resolved_calendar["calendar_unit_return"] = resolved_calendar["daily_unit_return"].where(
        resolved_calendar["action_taken"], 0.0
    )

    scope = policy["scope"]
    return_lower = -1.0
    return_upper = float(scope["maximum_decimal_odds"]) - 1.0
    requirements = policy["certificate_requirements"]
    delta = float(requirements["delta"])
    calendar_lcb, calendar_ucb = anytime_hoeffding_interval(
        resolved_calendar["calendar_unit_return"].dropna(),
        lower_bound=return_lower,
        upper_bound=return_upper,
        delta=delta / 2.0,
    )
    action_lcb, action_ucb = anytime_hoeffding_interval(
        resolved_action["daily_unit_return"].dropna(),
        lower_bound=return_lower,
        upper_bound=return_upper,
        delta=delta,
    )
    losing = (resolved_action["daily_unit_return"] < 0.0).astype(float)
    losing_lcb, losing_ucb = anytime_hoeffding_interval(
        losing,
        lower_bound=0.0,
        upper_bound=1.0,
        delta=delta / 2.0,
    )

    eligible_slates = int(len(eligible))
    action_slates = int(len(action))
    resolved_action_slates = int(len(resolved_action))
    resolved_selections = int(resolved_action["selection_count"].fillna(0).sum())
    eligible_candidates = int(eligible["eligible_candidate_count"].fillna(0).sum())
    selected_candidates = int(eligible["selected_candidate_count"].fillna(0).sum())
    slate_coverage = action_slates / eligible_slates if eligible_slates else 0.0
    candidate_coverage = selected_candidates / eligible_candidates if eligible_candidates else 0.0
    deployment_margin = float(requirements["deployment_margin"])

    failures: list[str] = []
    if policy["policy_stage"] != "PRODUCTION_ELIGIBLE":
        failures.append("POLICY_NOT_PRODUCTION_ELIGIBLE")
    if calendar_lcb is None or calendar_lcb <= deployment_margin:
        failures.append("CALENDAR_RETURN_LCB_DOES_NOT_CLEAR_MARGIN")
    if len(unresolved_action):
        failures.append("UNRESOLVED_ACTION_SLATES_PRESENT")
    for actual, key, reason in (
        (eligible_slates, "minimum_eligible_slates", "INSUFFICIENT_ELIGIBLE_SLATES"),
        (resolved_action_slates, "minimum_resolved_action_slates", "INSUFFICIENT_RESOLVED_ACTION_SLATES"),
        (resolved_selections, "minimum_resolved_selections", "INSUFFICIENT_RESOLVED_SELECTIONS"),
    ):
        if actual < int(requirements[key]):
            failures.append(reason)
    if slate_coverage < float(requirements["minimum_slate_coverage"]):
        failures.append("SLATE_COVERAGE_BELOW_MINIMUM")
    if candidate_coverage < float(requirements["minimum_candidate_coverage"]):
        failures.append("CANDIDATE_COVERAGE_BELOW_MINIMUM")
    if losing_ucb is None or losing_ucb > float(requirements["maximum_losing_action_slate_rate"]):
        failures.append("LOSING_SLATE_RATE_UCB_EXCEEDS_MAXIMUM")
    if support_status != "IN_SUPPORT":
        failures.append("SUPPORT_GATE_FAILED")
    if shift_status not in {"STABLE", "TOLERABLE"}:
        failures.append("SHIFT_GATE_FAILED")
    if bool(requirements.get("dependence_stress_required")) and dependence_stress_status != "PASSED":
        failures.append("DEPENDENCE_STRESS_NOT_PASSED")
    cvar_status = str(requirements.get("cvar_certificate_status") or "NOT_REQUIRED")
    if cvar_status.startswith("BLOCKING"):
        failures.append("CVAR_METHOD_NOT_VALIDATED")

    generated_at = datetime.now(timezone.utc)
    certificate = {
        "schema_version": "MLB_POLICY_CERTIFICATE_V1",
        "certificate_id": f"{policy['policy_version']}_PROSPECTIVE_{generated_at.strftime('%Y%m%dT%H%M%SZ')}",
        "policy_version": policy["policy_version"],
        "policy_digest": policy["policy_digest"],
        "certificate_status": "ACTIVE" if not failures else "REJECTED",
        "generated_at_utc": generated_at.isoformat(),
        "scope": scope,
        "evidence": policy["evidence"],
        "evaluation": {
            "unit": "DAILY_SLATE",
            "primary_return_estimand": requirements["primary_return_estimand"],
            "evidence_partition": "PROSPECTIVE_SHADOW",
            "eligible_slates": eligible_slates,
            "action_slates": action_slates,
            "resolved_action_slates": resolved_action_slates,
            "unresolved_action_slates": int(len(unresolved_action)),
            "resolved_selections": resolved_selections,
            "eligible_candidates": eligible_candidates,
            "selected_candidates": selected_candidates,
            "slate_coverage": slate_coverage,
            "candidate_coverage": candidate_coverage,
            "mean_action_day_unit_return": float(resolved_action["daily_unit_return"].mean()) if resolved_action_slates else None,
            "mean_calendar_slate_unit_return": float(resolved_calendar["calendar_unit_return"].mean()) if len(resolved_calendar) else None,
            "anytime_valid_return_lcb": calendar_lcb,
            "anytime_valid_return_ucb": calendar_ucb,
            "action_day_return_lcb": action_lcb,
            "action_day_return_ucb": action_ucb,
            "losing_action_slate_rate": float(losing.mean()) if len(losing) else None,
            "losing_action_slate_rate_lcb": losing_lcb,
            "losing_action_slate_rate_ucb": losing_ucb,
            "deployment_margin": deployment_margin,
            "return_bounds": [return_lower, return_upper],
            "confidence_sequence_method": "HOEFFDING_UNION_BOUND_V1",
            "confidence_sequence_delta": delta,
            "authorization_error_allocation": {
                "calendar_return_lcb": delta / 2.0,
                "losing_action_slate_rate_ucb": delta / 2.0,
                "action_day_return_interval_diagnostic": delta,
            },
            "confidence_sequence_assumptions": [
                "daily returns remain inside the declared payout bounds",
                "daily observations satisfy the predeclared conditional-mean martingale assumptions",
                "policy decisions are frozen before outcomes",
                "dependence stress analysis passes"
            ],
            "cvar_certificate_status": cvar_status,
        },
        "authorization_gates": requirements,
        "support": {
            "method_version": policy["support_rule"]["method_version"],
            "current_status": support_status,
        },
        "shift": {
            "detector_version": policy["shift_rule"]["detector_version"],
            "current_status": shift_status,
        },
        "dependence_stress_status": dependence_stress_status,
        "authorization": {
            "eligible_for_candidate_authorization": not failures,
            "staking_enabled": False,
            "blocking_reasons": sorted(set(failures)),
        },
        "staking_enabled": False,
    }
    certificate["certificate_digest"] = canonical_json_hash(certificate)
    return certificate


def main() -> None:
    args = parse_args()
    registry = load_policy_registry(args.registry.resolve())
    policy = _policy_by_version(registry, args.policy_version)
    evidence = pd.read_csv(args.evidence_csv.resolve())
    certificate = build_certificate(
        policy=policy,
        evidence=evidence,
        support_status=args.support_status,
        shift_status=args.shift_status,
        dependence_stress_status=args.dependence_stress_status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(f"Certificate {certificate['certificate_id']}: {certificate['certificate_status']}")
    for reason in certificate["authorization"]["blocking_reasons"]:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
