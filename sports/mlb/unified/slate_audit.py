from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable


ACTIONABLE_STATUSES = {"ACTIONABLE_SHADOW", "CERTIFIED"}
RESEARCH_STATUSES = {"RESEARCH_ONLY", "BLOCKED_DATA", "REJECTED_VALUE", "REJECTED_SUPPORT", "REJECTED_IDENTITY"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def authoritative_status(candidate: dict[str, Any]) -> str:
    reasons = set(candidate.get("rejection_reasons") or [])
    if candidate.get("identity_status") != "CONFIRMED" or "IDENTITY_INVALID" in reasons:
        return "REJECTED_IDENTITY"
    if reasons & {"LINEUP_INVALID", "PLAYER_STATUS_INVALID", "QUOTE_STALE", "QUOTE_FRESHNESS_UNPROVABLE", "EXACT_SELECTION_UNAVAILABLE"}:
        return "BLOCKED_DATA"
    if reasons & {"SUPPORT_INVALID", "OUT_OF_SUPPORT", "CAPABILITY_NOT_SUPPORTED", "OOD_UNMEASURED"}:
        return "REJECTED_SUPPORT"
    if reasons & {"NON_POSITIVE_CONSERVATIVE_EV", "NON_POSITIVE_LCB_EV", "EDGE_LCB_NOT_POSITIVE", "PROBABILITY_EDGE_BELOW_FLOOR"}:
        return "REJECTED_VALUE"
    if candidate.get("final_selection_decision"):
        return "ACTIONABLE_SHADOW"
    return "RESEARCH_ONLY"


def audit_candidates(candidates: Iterable[dict[str, Any]], *, run_date: str) -> dict[str, Any]:
    rows, issues, plateaus = [], [], defaultdict(list)
    counts = Counter()
    for candidate in candidates:
        row = _json_safe(dict(candidate))
        status = authoritative_status(row)
        row["authoritative_candidate_status"] = status
        row_issues: list[str] = []
        artifact_date = row.get("artifact_run_date") or row.get("run_date") or run_date
        if artifact_date != run_date:
            row_issues.append("STALE_ARTIFACT")
        if row.get("quote_timestamp") is None and row.get("odds_snapshot_time") is None:
            row_issues.append("QUOTE_FRESHNESS_UNPROVABLE")
        if row.get("identity_status") != "CONFIRMED":
            row_issues.append("IDENTITY_MISMATCH")
        is_hitter = str(row.get("market_type") or row.get("capability") or "").startswith("batter_")
        if is_hitter and str(row.get("lineup_status") or "").upper() != "CONFIRMED":
            row_issues.append("LINEUP_ROLE_INVALID")
        has_action = bool(row.get("betslip_url") or row.get("sportsbook_deeplink") or row.get("show_betslip_action"))
        if has_action and status not in ACTIONABLE_STATUSES:
            row_issues.append("NON_ACTIONABLE_BETSLIP_CTA")
        if row.get("conservative_expected_value") is not None and float(row["conservative_expected_value"]) <= 0 and has_action:
            row_issues.append("NEGATIVE_EV_BETSLIP_CTA")
        probability = row.get("calibrated_probability")
        if probability is not None:
            plateaus[round(float(probability), 12)].append(str(row.get("player_id") or row.get("candidate_id")))
        row["audit_issues"] = sorted(set(row_issues))
        issues.extend(row["audit_issues"])
        counts[status] += 1
        rows.append(row)
    repeated = [{"probability": probability, "candidate_ids": ids, "count": len(ids),
                 "issue": "CALIBRATION_PLATEAU_REVIEW"}
                for probability, ids in sorted(plateaus.items()) if len(set(ids)) >= 3]
    fatal = {"STALE_ARTIFACT", "IDENTITY_MISMATCH", "LINEUP_ROLE_INVALID", "NON_ACTIONABLE_BETSLIP_CTA", "NEGATIVE_EV_BETSLIP_CTA"}
    return {
        "schema_version": 1, "run_date": run_date, "candidates": rows,
        "candidate_status_counts": dict(sorted(counts.items())),
        "issue_counts": dict(sorted(Counter(issues).items())), "probability_plateaus": repeated,
        "publication_integrity": "FAIL" if fatal.intersection(issues) else "PASS",
        "fatal_issues": sorted(fatal.intersection(issues)),
    }
