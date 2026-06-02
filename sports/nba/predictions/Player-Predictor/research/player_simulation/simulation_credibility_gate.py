from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def evaluate_simulation_credibility(
    *,
    backtest_report_path: Path | None,
    output_dir: Path,
    leakage_audit_path: Path | None = None,
    min_p10_p90_coverage: float = 0.72,
    min_joined_rows: int = 100,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _read_json(backtest_report_path) if backtest_report_path else {}
    leakage = _read_json(leakage_audit_path) if leakage_audit_path else {}
    blocked_reasons: list[str] = []
    status = "BACKTEST_REQUIRED"

    if leakage and leakage.get("status") == "BACKTEST_FAILED_LEAKAGE":
        status = "BACKTEST_FAILED_LEAKAGE"
        blocked_reasons.extend(str(reason) for reason in leakage.get("failures", []))
    elif not report:
        blocked_reasons.append("frozen_preseason_backtest_missing")
    elif report.get("status") != "BACKTEST_EVALUATED":
        report_status = str(report.get("status") or "backtest_not_evaluated")
        if report_status == "BACKTEST_FAILED_LEAKAGE":
            status = "BACKTEST_FAILED_LEAKAGE"
        else:
            status = "BACKTEST_REQUIRED"
        blocked_reasons.append(report_status)
        blockers = report.get("missing_data_impact", {}).get("blockers", [])
        blocked_reasons.extend(str(blocker) for blocker in blockers)
    else:
        joined_rows = int(report.get("joined_rows") or 0)
        if joined_rows < int(min_joined_rows):
            status = "BACKTEST_FAILED_INSUFFICIENT_SAMPLE"
            blocked_reasons.append("joined_rows_below_threshold")
        else:
            coverage = report.get("actual_within_p10_p90_rate")
            if coverage is None:
                status = "BACKTEST_REQUIRED"
                blocked_reasons.append("coverage_metric_missing")
            elif float(coverage) < float(min_p10_p90_coverage):
                status = "BACKTEST_FAILED_CALIBRATION"
                blocked_reasons.append("p10_p90_coverage_below_threshold")
            elif not _confidence_tiers_monotonic(report.get("confidence_tier_reliability", {})):
                status = "PUBLISH_WITH_WARNINGS"
                blocked_reasons.append("confidence_tiers_not_reliably_monotonic")
            else:
                status = "PUBLISH_CALIBRATED_RANGES"

    if status == "BACKTEST_REQUIRED" and report:
        status = "PUBLISH_RESEARCH_ONLY"
    if status in {"BACKTEST_REQUIRED", "PUBLISH_RESEARCH_ONLY", "BACKTEST_FAILED_INSUFFICIENT_SAMPLE"}:
        frontend_label = "research projection / uncalibrated"
    elif status == "BACKTEST_FAILED_LEAKAGE":
        frontend_label = "research projection / leakage audit failed"
    elif status == "BACKTEST_FAILED_CALIBRATION":
        frontend_label = "research projection / calibration failed"
    elif status == "PUBLISH_WITH_WARNINGS":
        frontend_label = "Backtested, but calibration warnings remain."
    else:
        frontend_label = "Backtested range projection"

    gate = {
        "status": status,
        "labels": {
            "pipeline": "SIMULATION_PIPELINE_READY",
            "credibility": status,
            "frontend_label": frontend_label,
        },
        "publish_as_calibrated": status == "PUBLISH_CALIBRATED_RANGES",
        "blocked_reasons": sorted(set(reason for reason in blocked_reasons if reason)),
        "backtest_report_path": str(backtest_report_path) if backtest_report_path else "",
        "minimum_requirements": {
            "frozen_preseason_backtest_exists": True,
            "min_p10_p90_coverage": float(min_p10_p90_coverage),
            "min_joined_rows": int(min_joined_rows),
            "confidence_tiers_reliable": True,
            "leakage_audit_passes": True,
        },
        "current_evidence": {
            "backtest_status": report.get("status", "MISSING"),
            "leakage_status": leakage.get("status", "MISSING"),
            "actual_within_p10_p90_rate": report.get("actual_within_p10_p90_rate"),
            "actual_within_p25_p75_rate": report.get("actual_within_p25_p75_rate"),
            "joined_rows": report.get("joined_rows"),
        },
        "production_behavior_changed": False,
        "promotion_ready": False,
    }
    (output_dir / "simulation_credibility_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (output_dir / "simulation_credibility_gate.md").write_text(_format_markdown(gate), encoding="utf-8")
    return gate


def _confidence_tiers_monotonic(reliability: dict[str, Any]) -> bool:
    if not reliability:
        return False
    order = ["INSUFFICIENT_DATA", "LOW_CONFIDENCE", "MEDIUM_CONFIDENCE", "HIGH_CONFIDENCE"]
    observed: list[tuple[int, float]] = []
    for idx, tier in enumerate(order):
        payload = reliability.get(tier)
        if not isinstance(payload, dict) or payload.get("p10_p90_coverage") is None:
            continue
        observed.append((idx, float(payload["p10_p90_coverage"])))
    if len(observed) < 2:
        return False
    return all(observed[i][1] <= observed[i + 1][1] + 0.10 for i in range(len(observed) - 1))


def _format_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# Simulation Credibility Gate",
        "",
        f"- status: {gate.get('status')}",
        f"- frontend_label: {gate.get('labels', {}).get('frontend_label')}",
        f"- publish_as_calibrated: {gate.get('publish_as_calibrated')}",
        f"- promotion_ready: {gate.get('promotion_ready')}",
        "",
        "## Blocked Reasons",
        "",
    ]
    reasons = gate.get("blocked_reasons", [])
    lines.extend(f"- {reason}" for reason in reasons) if reasons else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simulation publication credibility from frozen preseason backtest artifacts.")
    parser.add_argument("--backtest-report", type=Path)
    parser.add_argument("--leakage-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = evaluate_simulation_credibility(
        backtest_report_path=args.backtest_report,
        leakage_audit_path=args.leakage_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
