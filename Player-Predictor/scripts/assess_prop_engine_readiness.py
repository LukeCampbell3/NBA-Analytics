#!/usr/bin/env python3
"""
Assess prop-engine production readiness from manifest + validation report.

This is deliberately conservative. A model can be useful in controlled shadow
production while still being blocked from live betting if true market odds,
closing lines, or CLV evidence are unavailable.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "min_snapshot_match_rate": 0.95,
    "max_all_brier": 0.238,
    "max_gated_brier": 0.232,
    "min_gated_bss": 0.07,
    "min_gated_clv_edge_correlation": 0.10,
    "min_positive_clv_rate": 0.53,
    "max_gated_ece": 0.03,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def assess(manifest: dict[str, Any], validation: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    attachment = manifest.get("market_attachment", {})
    schema_status = _get(attachment, "schema_validation.status")
    match_rate = attachment.get("match_rate")
    market_source = _get(validation, "market_validation.market_source", {})
    clv_gated = _get(validation, "market_validation.clv.gated", {})
    all_model = _get(validation, "comparison.v9_calibrated", {})
    gated_model = _get(validation, "comparison.v9_calibrated_gate", {})
    market = _get(validation, "comparison.current_market_no_vig", {})

    checks = {
        "schema_validation_pass": schema_status == "pass",
        "snapshot_match_rate_pass": _is_number(match_rate) and match_rate >= thresholds["min_snapshot_match_rate"],
        "real_market_probability_available": bool(market_source.get("real_market_probability_available")),
        "closing_odds_available": bool(market_source.get("closing_odds_available")),
        "clv_available": bool(clv_gated.get("available")),
        "all_brier_target": _is_number(all_model.get("brier")) and all_model["brier"] <= thresholds["max_all_brier"],
        "gated_brier_target": _is_number(gated_model.get("brier")) and gated_model["brier"] <= thresholds["max_gated_brier"],
        "gated_bss_target": _is_number(gated_model.get("brier_skill_score")) and gated_model["brier_skill_score"] >= thresholds["min_gated_bss"],
        "gated_ece_target": _is_number(gated_model.get("ece")) and gated_model["ece"] <= thresholds["max_gated_ece"],
        "beats_true_market_brier": (
            _is_number(all_model.get("brier"))
            and _is_number(market.get("brier"))
            and bool(market_source.get("real_market_probability_available"))
            and all_model["brier"] < market["brier"]
        ),
        "clv_edge_correlation_target": (
            _is_number(clv_gated.get("clv_edge_correlation"))
            and clv_gated["clv_edge_correlation"] >= thresholds["min_gated_clv_edge_correlation"]
        ),
        "positive_clv_rate_target": (
            _is_number(clv_gated.get("positive_clv_rate"))
            and clv_gated["positive_clv_rate"] >= thresholds["min_positive_clv_rate"]
        ),
    }

    controlled_shadow_required = [
        "schema_validation_pass",
        "gated_brier_target",
        "gated_bss_target",
        "gated_ece_target",
    ]
    live_required = [
        *controlled_shadow_required,
        "snapshot_match_rate_pass",
        "real_market_probability_available",
        "closing_odds_available",
        "clv_available",
        "beats_true_market_brier",
        "clv_edge_correlation_target",
        "positive_clv_rate_target",
    ]

    controlled_shadow_ready = all(checks[name] for name in controlled_shadow_required)
    live_ready = all(checks[name] for name in live_required)
    blockers = [name for name in live_required if not checks[name]]

    return {
        "assessed_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest.get("model_version"),
        "manifest_status": manifest.get("status"),
        "controlled_shadow_ready": controlled_shadow_ready,
        "live_bankroll_ready": live_ready,
        "status": "live_bankroll_ready" if live_ready else "controlled_shadow_ready" if controlled_shadow_ready else "not_ready",
        "checks": checks,
        "live_blockers": blockers,
        "thresholds": thresholds,
        "key_metrics": {
            "snapshot_match_rate": match_rate,
            "all_brier": all_model.get("brier"),
            "gated_brier": gated_model.get("brier"),
            "gated_bss": gated_model.get("brier_skill_score"),
            "gated_ece": gated_model.get("ece"),
            "market_brier": market.get("brier"),
            "clv_edge_correlation": clv_gated.get("clv_edge_correlation"),
            "positive_clv_rate": clv_gated.get("positive_clv_rate"),
        },
    }


def _merge_optimized_policy(readiness: dict[str, Any], optimized_report: dict[str, Any] | None, thresholds: dict[str, float]) -> dict[str, Any]:
    if not optimized_report:
        return readiness
    aggregate = optimized_report.get("aggregate_gated", {})
    optimized_checks = {
        "optimized_gated_brier_target": _is_number(aggregate.get("brier")) and aggregate["brier"] <= thresholds["max_gated_brier"],
        "optimized_gated_bss_target": _is_number(aggregate.get("brier_skill_score")) and aggregate["brier_skill_score"] >= thresholds["min_gated_bss"],
        "optimized_gated_ece_target": _is_number(aggregate.get("ece")) and aggregate["ece"] <= thresholds["max_gated_ece"],
        "optimized_min_gated_rows": _is_number(aggregate.get("n")) and aggregate["n"] >= 1500,
    }
    readiness["optimized_policy"] = {
        "available": True,
        "aggregate_gated": aggregate,
        "recommended_next_policy": optimized_report.get("recommended_next_policy"),
        "checks": optimized_checks,
    }
    readiness["controlled_shadow_ready"] = bool(readiness["controlled_shadow_ready"] or all(optimized_checks.values()))
    if readiness["controlled_shadow_ready"] and not readiness["live_bankroll_ready"]:
        readiness["status"] = "controlled_shadow_ready"
    readiness["key_metrics"]["optimized_gated_brier"] = aggregate.get("brier")
    readiness["key_metrics"]["optimized_gated_bss"] = aggregate.get("brier_skill_score")
    readiness["key_metrics"]["optimized_gated_ece"] = aggregate.get("ece")
    readiness["key_metrics"]["optimized_gated_rows"] = aggregate.get("n")
    return readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess prop engine readiness gates")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--optimized-policy-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.manifest)
    validation = _load_json(args.validation_report)
    report = assess(manifest, validation, dict(DEFAULT_THRESHOLDS))
    optimized = _load_json(args.optimized_policy_report) if args.optimized_policy_report and args.optimized_policy_report.exists() else None
    report = _merge_optimized_policy(report, optimized, dict(DEFAULT_THRESHOLDS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
