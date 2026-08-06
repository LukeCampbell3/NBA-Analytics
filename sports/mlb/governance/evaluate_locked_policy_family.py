#!/usr/bin/env python3
"""Multiplicity-controlled locked validation for a bounded MLB policy family."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.governance.policy_governance import load_policy_registry, parse_strict_bool, policy_content_digest


REQUIRED_COLUMNS = {
    "slate_id", "snapshot_id", "policy_version", "policy_digest", "evidence_partition", "capture_label",
    "decision_frozen_at_utc", "slate_date", "eligible_slate", "action_taken", "resolved",
    "selection_count", "eligible_candidate_count", "selected_candidate_count", "daily_unit_return",
}


def fixed_hoeffding_lcb(values: pd.Series, *, lower: float, upper: float, alpha: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    if (clean < lower).any() or (clean > upper).any():
        raise ValueError("Locked-validation return exceeds declared policy bounds.")
    radius = (upper - lower) * math.sqrt(math.log(1.0 / alpha) / (2.0 * len(clean)))
    return max(lower, float(clean.mean()) - radius)


def evaluate_family(registry: dict[str, Any], evidence: pd.DataFrame, *, family_delta: float = 0.05) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS - set(evidence.columns))
    if missing:
        raise ValueError(f"Locked evidence is missing columns: {', '.join(missing)}")
    policies = registry["policies"]
    if not policies:
        raise ValueError("Policy family is empty.")
    alpha = family_delta / len(policies)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        version = str(policy["policy_version"])
        period = policy["evidence"]["locked_validation_period"]
        blockers: list[str] = []
        if policy["policy_stage"] != "LOCKED_VALIDATION":
            blockers.append("POLICY_NOT_IN_LOCKED_VALIDATION")
        if not bool(policy["decision_rule"].get("family_is_frozen")):
            blockers.append("POLICY_FAMILY_NOT_FROZEN")
        if not period.get("start") or not period.get("end"):
            blockers.append("LOCKED_PERIOD_NOT_DECLARED")
        frame = evidence.loc[evidence["policy_version"].astype(str) == version].copy()
        if frame.empty:
            blockers.append("NO_LOCKED_EVIDENCE")
            rows.append({"policy_version": version, "status": "REJECTED", "blocking_reasons": blockers})
            continue
        frame["slate_date"] = pd.to_datetime(frame["slate_date"], errors="raise").dt.date
        if frame["slate_id"].duplicated().any() or frame["slate_date"].duplicated().any():
            blockers.append("DUPLICATE_DAILY_SLATE_EVIDENCE")
        if set(frame["policy_digest"].astype(str)) != {policy_content_digest(policy)}:
            blockers.append("POLICY_DIGEST_MISMATCH")
        if set(frame["evidence_partition"].astype(str).str.upper()) != {"LOCKED_VALIDATION"}:
            blockers.append("EVIDENCE_PARTITION_MISMATCH")
        if set(frame["capture_label"].astype(str).str.upper()) != {"FULL_SLATE_SNAPSHOT"}:
            blockers.append("FULL_SLATE_REPLAY_UNAVAILABLE")
        if pd.to_datetime(frame["decision_frozen_at_utc"], utc=True, errors="coerce").isna().any():
            blockers.append("INVALID_DECISION_FREEZE_TIMESTAMP")
        for column in ("eligible_slate", "action_taken", "resolved"):
            frame[column] = frame[column].map(parse_strict_bool)
        if period.get("start") and period.get("end"):
            start = pd.Timestamp(period["start"]).date()
            end = pd.Timestamp(period["end"]).date()
            if frame["slate_date"].min() < start or frame["slate_date"].max() > end:
                blockers.append("EVIDENCE_OUTSIDE_LOCKED_PERIOD")
        eligible = frame.loc[frame["eligible_slate"]].copy()
        unresolved_actions = eligible.loc[eligible["action_taken"] & ~eligible["resolved"]]
        if len(unresolved_actions):
            blockers.append("UNRESOLVED_ACTION_SLATES_PRESENT")
        resolved = eligible.loc[(~eligible["action_taken"]) | eligible["resolved"]].copy()
        resolved["calendar_return"] = pd.to_numeric(resolved["daily_unit_return"], errors="coerce").where(
            resolved["action_taken"], 0.0
        )
        scope = policy["scope"]
        lcb = fixed_hoeffding_lcb(
            resolved["calendar_return"],
            lower=-1.0,
            upper=float(scope["maximum_decimal_odds"]) - 1.0,
            alpha=alpha,
        )
        margin = float(policy["certificate_requirements"]["deployment_margin"])
        if lcb is None or lcb <= margin:
            blockers.append("LOCKED_RETURN_LCB_DOES_NOT_CLEAR_MARGIN")
        action_slates = int(eligible["action_taken"].sum())
        coverage = action_slates / len(eligible) if len(eligible) else 0.0
        if coverage < float(policy["certificate_requirements"]["minimum_slate_coverage"]):
            blockers.append("LOCKED_SLATE_COVERAGE_BELOW_MINIMUM")
        rows.append(
            {
                "policy_version": version,
                "policy_digest": policy["policy_digest"],
                "status": "PASSED_FOR_PROSPECTIVE_SHADOW" if not blockers else "REJECTED",
                "eligible_slates": int(len(eligible)),
                "action_slates": action_slates,
                "slate_coverage": coverage,
                "mean_calendar_return": float(resolved["calendar_return"].mean()) if len(resolved) else None,
                "multiplicity_adjusted_return_lcb": lcb,
                "deployment_margin": margin,
                "per_policy_alpha": alpha,
                "blocking_reasons": sorted(set(blockers)),
            }
        )
    return {
        "schema_version": "MLB_LOCKED_POLICY_VALIDATION_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "LEARN_THEN_TEST_BONFERRONI_HOEFFDING_V1",
        "family_size": len(policies),
        "family_delta": family_delta,
        "policies": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen MLB policy family on locked daily returns.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-delta", type=float, default=0.05)
    args = parser.parse_args()
    report = evaluate_family(
        load_policy_registry(args.registry.resolve()),
        pd.read_csv(args.evidence_csv.resolve()),
        family_delta=args.family_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for policy in report["policies"]:
        print(f"{policy['policy_version']}: {policy['status']}")


if __name__ == "__main__":
    main()
