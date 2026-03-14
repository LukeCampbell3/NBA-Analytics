"""
validate_college_metric_parity.py

Validates whether college data can support the same valuation + breakout
pillars used in NBA player evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from analyze_players import analyze_player
from value_college_players import build_card_from_row, read_csv_non_empty, resolve_column_map
from value_players import PlayerValuator


MetricExtractor = Callable[[Dict[str, Any]], Any]


def utc_now_iso() -> str:
    """UTC timestamp in ISO8601 with timezone."""
    return datetime.now(timezone.utc).isoformat()


def is_valid_numeric(value: Any) -> bool:
    """True for finite numeric values."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_non_empty_text(value: Any) -> bool:
    """True for non-empty strings."""
    return str(value).strip() != ""


def is_non_empty_mapping(value: Any) -> bool:
    """True for dict-like values with at least one key."""
    return isinstance(value, dict) and len(value) > 0


def build_build_summary_checks(build_summary_path: Path) -> List[Dict[str, Any]]:
    """Read ingestion summary and capture compliance checks."""
    checks: List[Dict[str, Any]] = []
    if not build_summary_path.exists():
        checks.append(
            {
                "name": "build_summary_exists",
                "passed": False,
                "severity": "warning",
                "detail": f"Missing: {build_summary_path}",
            }
        )
        return checks

    with open(build_summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    checks.append(
        {
            "name": "robots_enforced",
            "passed": bool(summary.get("strict_robots", False)),
            "severity": "critical",
            "detail": f"strict_robots={summary.get('strict_robots', False)}",
        }
    )
    checks.append(
        {
            "name": "user_agent_set",
            "passed": is_non_empty_text(summary.get("user_agent", "")),
            "severity": "warning",
            "detail": f"user_agent='{summary.get('user_agent', '')}'",
        }
    )
    checks.append(
        {
            "name": "source_ingestion_errors",
            "passed": len(summary.get("errors", [])) == 0,
            "severity": "warning",
            "detail": f"errors={len(summary.get('errors', []))}",
        }
    )
    return checks


def evaluate_metric_coverage(
    records: List[Dict[str, Any]],
    metric_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate per-metric coverage across evaluated players."""
    coverage: Dict[str, Dict[str, Any]] = {}
    total = len(records)
    for metric_name, spec in metric_specs.items():
        extractor: MetricExtractor = spec["extractor"]
        validator: Callable[[Any], bool] = spec["validator"]
        valid_count = 0
        for record in records:
            if validator(extractor(record)):
                valid_count += 1
        coverage_rate = (valid_count / total) if total else 0.0
        coverage[metric_name] = {
            "valid_count": valid_count,
            "total_count": total,
            "coverage_rate": round(coverage_rate, 4),
        }
    return coverage


def pass_rate(coverage: Dict[str, Dict[str, Any]], threshold: float) -> bool:
    """Return True if every metric coverage_rate >= threshold."""
    return all(metric["coverage_rate"] >= threshold for metric in coverage.values())


def run_parity_validation(
    input_path: Path,
    build_summary_path: Path,
    max_players: Optional[int],
    coverage_threshold: float,
) -> Dict[str, Any]:
    """Run metric parity validation against college rows."""
    summary_checks = build_build_summary_checks(build_summary_path)

    try:
        df = read_csv_non_empty(input_path)
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "run_at_utc": utc_now_iso(),
            "input_path": str(input_path),
            "rows_evaluated": 0,
            "overall_status": "fail",
            "blocking_issues": [str(exc)],
            "build_summary_checks": summary_checks,
            "pillars": {},
        }

    if max_players is not None:
        df = df.head(max_players).copy()

    col_map = resolve_column_map(list(df.columns))
    valuator = PlayerValuator()

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        card = build_card_from_row(row=row, col_map=col_map)
        valuation = valuator.generate_report(valuator.valuate_player(card))
        breakout = analyze_player(card)
        records.append(
            {
                "row": row.to_dict(),
                "card": card,
                "valuation": valuation,
                "breakout": breakout,
            }
        )

    valuation_specs = {
        "wins_added": {
            "extractor": lambda r: r["valuation"].get("impact", {}).get("wins_added"),
            "validator": is_valid_numeric,
        },
        "market_value_curve": {
            "extractor": lambda r: r["valuation"].get("market_value", {}).get("by_year"),
            "validator": is_non_empty_mapping,
        },
        "npv_surplus": {
            "extractor": lambda r: r["valuation"].get("contract", {}).get("npv_surplus"),
            "validator": is_valid_numeric,
        },
        "trade_value_base": {
            "extractor": lambda r: r["valuation"].get("trade_value", {}).get("base"),
            "validator": is_valid_numeric,
        },
        "aging_phase": {
            "extractor": lambda r: r["valuation"].get("aging", {}).get("current_phase"),
            "validator": is_non_empty_text,
        },
    }

    breakout_specs = {
        "breakout_opportunity_score": {
            "extractor": lambda r: r["breakout"].get("breakout_potential", {}).get("opportunity_score"),
            "validator": is_valid_numeric,
        },
        "breakout_signal_strength": {
            "extractor": lambda r: r["breakout"].get("breakout_potential", {}).get("signal_strength"),
            "validator": is_valid_numeric,
        },
        "breakout_confidence": {
            "extractor": lambda r: r["breakout"].get("breakout_potential", {}).get("confidence"),
            "validator": is_valid_numeric,
        },
        "defense_portability_score": {
            "extractor": lambda r: r["breakout"].get("defense_portability", {}).get("portability", {}).get("score"),
            "validator": is_valid_numeric,
        },
        "impact_sanity_level": {
            "extractor": lambda r: r["breakout"].get("impact_sanity", {}).get("sanity_level"),
            "validator": is_non_empty_text,
        },
    }

    isolation_specs = {
        "trust_score": {
            "extractor": lambda r: r["card"].get("trust", {}).get("score"),
            "validator": is_valid_numeric,
        },
        "uncertainty_score": {
            "extractor": lambda r: r["card"].get("uncertainty", {}).get("overall"),
            "validator": is_valid_numeric,
        },
        "usage_rate_input": {
            "extractor": lambda r: r["card"].get("performance", {}).get("advanced", {}).get("usage_rate"),
            "validator": is_valid_numeric,
        },
        "plus_minus_input": {
            "extractor": lambda r: r["card"].get("performance", {}).get("advanced", {}).get("plus_minus"),
            "validator": is_valid_numeric,
        },
        "source_url_provenance": {
            "extractor": lambda r: r["row"].get(col_map.get("source_url", ""), ""),
            "validator": is_non_empty_text,
        },
    }

    valuation_coverage = evaluate_metric_coverage(records=records, metric_specs=valuation_specs)
    breakout_coverage = evaluate_metric_coverage(records=records, metric_specs=breakout_specs)
    isolation_coverage = evaluate_metric_coverage(records=records, metric_specs=isolation_specs)

    pillars = {
        "valuation_pillars": {
            "coverage_threshold": coverage_threshold,
            "passed": pass_rate(valuation_coverage, coverage_threshold),
            "metrics": valuation_coverage,
        },
        "breakout_pillars": {
            "coverage_threshold": coverage_threshold,
            "passed": pass_rate(breakout_coverage, coverage_threshold),
            "metrics": breakout_coverage,
        },
        "isolation_pillars": {
            "coverage_threshold": coverage_threshold,
            "passed": pass_rate(isolation_coverage, coverage_threshold),
            "metrics": isolation_coverage,
        },
    }

    blocking_issues: List[str] = []
    if len(records) == 0:
        blocking_issues.append("No rows available for parity evaluation.")

    overall_pass = (
        len(records) > 0
        and pillars["valuation_pillars"]["passed"]
        and pillars["breakout_pillars"]["passed"]
        and pillars["isolation_pillars"]["passed"]
    )
    overall_status = "pass" if overall_pass else "fail"

    return {
        "run_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "rows_evaluated": len(records),
        "overall_status": overall_status,
        "blocking_issues": blocking_issues,
        "build_summary_checks": summary_checks,
        "pillars": pillars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate college metric parity with NBA valuation/breakout pillars."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/college/players_season.csv"),
        help="Canonical college player-season CSV input.",
    )
    parser.add_argument(
        "--build-summary",
        type=Path,
        default=Path("data/processed/college/build_summary.json"),
        help="College ingestion build summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/college/metric_parity_report.json"),
        help="Output parity report JSON.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help="Minimum per-metric coverage rate required for pass.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Optional max number of rows to evaluate.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Always exit 0 even if parity fails.",
    )
    args = parser.parse_args()

    report = run_parity_validation(
        input_path=args.input,
        build_summary_path=args.build_summary,
        max_players=args.max_players,
        coverage_threshold=args.coverage_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[INFO] Rows evaluated: {report['rows_evaluated']}")
    print(f"[INFO] Overall parity status: {report['overall_status']}")
    print(f"[INFO] Report: {args.output}")

    if report["overall_status"] != "pass" and not args.non_strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
