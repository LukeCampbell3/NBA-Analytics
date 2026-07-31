#!/usr/bin/env python3
"""Export committed NFL selector/replay evidence for the static frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector-report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_selector_report.json",
    )
    parser.add_argument(
        "--replay-report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "production_replay_report.json",
    )
    parser.add_argument(
        "--weekly-ledger",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "production_replay_weekly.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NFL_ROOT / "web" / "data" / "market_validation_summary.json",
    )
    return parser.parse_args()


def build_payload(selector: dict, replay: dict, weekly: pd.DataFrame) -> dict:
    effectiveness = replay["effectiveness"]
    result = effectiveness["result"]
    bootstrap = effectiveness["week_cluster_bootstrap"]
    baselines = replay["baseline_comparison"]
    cap = selector["weekly_cap_policy"]
    validated_targets = replay["locked_policy"]["targets"]
    if len(validated_targets) != 1:
        raise ValueError(
            "The static validation summary currently requires one locked target."
        )
    validated_target = validated_targets[0]
    target_report = next(
        item for item in selector["targets"] if item["target"] == validated_target
    )
    return {
        "schema_version": 1,
        "generated_at_utc": replay.get("generated_at_utc")
        or selector.get("generated_at_utc"),
        "publication_status": "research_only_source_blocked",
        "status": replay["status"],
        "validated_targets": selector["validated_targets"],
        "locked_policy": replay["locked_policy"],
        "development": cap["development_result"],
        "final_test": result,
        "statistical_evidence": {
            "wilson_hit_rate_95": result["hit_rate_wilson_95"],
            "week_cluster_hit_rate_95": bootstrap["hit_rate_95"],
            "week_cluster_roi_95": bootstrap["roi_95"],
            "one_sided_exact_p_value_vs_50_percent": effectiveness[
                "one_sided_exact_p_value_vs_50_percent"
            ],
        },
        "baselines": {
            "always_under": baselines["always_under_same_cohort"],
            "point_projection_side": baselines["point_projection_side_same_cohort"],
            "model_vs_always_under": baselines["model_vs_always_under_paired"],
            "model_vs_point_projection": baselines[
                "model_vs_point_projection_paired"
            ],
        },
        "gates": {
            "contract": replay["contract_gate"],
            "operational": replay["operational_gate"],
            "effectiveness": {"status": effectiveness["status"]},
            "stability": {"status": replay["stability_gate"]["status"]},
            "source_provenance": replay["source_provenance_gate"],
            "deployment": replay["deployment_gate"],
        },
        "stability": {
            "halves": replay["stability_gate"]["halves"],
            "drawdown": replay["stability_gate"]["drawdown"],
            "weekly_cap_sensitivity": replay["stability_gate"][
                "weekly_cap_sensitivity"
            ],
        },
        "weekly": [
            {
                "season": int(row["season"]),
                "week": int(row["week"]),
                "picks": int(row["bets"]),
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "hit_rate": float(row["hit_rate"]),
                "roi": float(row["roi"]),
                "profit_units": float(row["profit_units"]),
            }
            for row in weekly.to_dict(orient="records")
        ],
        "methodology": {
            "development_season": selector["design"]["development_season"],
            "final_test_season": selector["design"]["final_test_season"],
            "architecture_selection": selector["design"][
                "architecture_selection_metric"
            ],
            "selected_architecture": target_report["selected_architecture"],
            "minimum_side_probability": selector["design"][
                "minimum_side_probability"
            ],
            "minimum_no_vig_advantage": selector["design"][
                "minimum_no_vig_advantage"
            ],
            "weekly_top_n": cap["selected_top_n"],
            "line_contract": selector["design"]["line_contract"],
            "line_movement_used": False,
            "closing_line_used": False,
        },
        "limitations": [
            "Only passing yards cleared both development and final target gates.",
            "The free Bovada archive has no capture timestamps, so line timing cannot be independently authenticated.",
            "The selector is directionally better than always-under on the matched cohort, but that paired uplift is not statistically significant.",
            "The board remains research-only until a prospective or timestamp-authenticated replay passes.",
        ],
    }


def main() -> int:
    args = parse_args()
    selector = json.loads(args.selector_report.read_text(encoding="utf-8"))
    replay = json.loads(args.replay_report.read_text(encoding="utf-8"))
    weekly = pd.read_csv(args.weekly_ledger)
    payload = build_payload(selector, replay, weekly)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"NFL market validation payload: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
