#!/usr/bin/env python3
"""Select a market edge on one season and test it once on a later season."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_backtest import (  # noqa: E402
    evaluate_market_backtest,
    load_market_archive,
)


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--development-predictions", type=Path, required=True)
    parser.add_argument("--final-predictions", type=Path, required=True)
    parser.add_argument(
        "--edge-candidates",
        default="0,2.5,5,7.5,10,12.5,15,20,25,30,40,50",
        help="Ascending fixed candidate grid; the smallest development-pass edge is selected.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_holdout_report.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_holdout_rows.csv",
    )
    return parser.parse_args()


def _single_season(frame: pd.DataFrame, label: str) -> int:
    values = sorted(int(value) for value in frame["season"].dropna().unique())
    if len(values) != 1:
        raise ValueError(f"{label} predictions must contain exactly one season; found {values}.")
    return values[0]


def main() -> int:
    args = parse_args()
    candidates = sorted(
        {float(value.strip()) for value in args.edge_candidates.split(",") if value.strip()}
    )
    if not candidates or candidates[0] < 0:
        raise ValueError("Edge candidates must be a non-empty set of non-negative values.")
    development = pd.read_csv(args.development_predictions, low_memory=False)
    final = pd.read_csv(args.final_predictions, low_memory=False)
    development_season = _single_season(development, "Development")
    final_season = _single_season(final, "Final")
    if development_season >= final_season:
        raise ValueError("Development season must be earlier than the final test season.")

    markets = load_market_archive(args.lines)
    grid: list[dict] = []
    selected: float | None = None
    for edge in candidates:
        report, _ = evaluate_market_backtest(
            development, markets, minimum_edge_yards=edge
        )
        grid.append(
            {
                "minimum_edge_yards": edge,
                "overall": report["overall"],
                "by_target": report["by_target"],
                "distinct_season_weeks": report["distinct_season_weeks"],
                "performance_gate": report["performance_gate"]["status"],
            }
        )
        if selected is None and report["performance_gate"]["status"] == "passed":
            selected = edge

    diagnostic_edge = selected if selected is not None else candidates[0]
    final_report, final_rows = evaluate_market_backtest(
        final, markets, minimum_edge_yards=diagnostic_edge
    )
    final_performance_passed = final_report["performance_gate"]["status"] == "passed"
    validated = selected is not None and final_performance_passed
    output = {
        "status": "validated" if validated else "failed",
        "design": {
            "development_season": development_season,
            "final_untouched_season": final_season,
            "selection_rule": (
                "Choose the smallest fixed edge candidate whose development performance gate passes; "
                "apply it once to the later season."
            ),
            "edge_candidates_yards": candidates,
            "selected_edge_yards": selected,
            "diagnostic_final_edge_yards": diagnostic_edge,
            "final_season_was_not_used_for_threshold_selection": True,
        },
        "development_grid": grid,
        "final_test": final_report,
        "promotion_gate": {
            "status": (
                "passed"
                if validated and final_report["promotion_gate"]["status"] == "passed"
                else "failed"
            ),
            "reason": (
                "Final performance and strict source-provenance gates passed."
                if validated and final_report["promotion_gate"]["status"] == "passed"
                else "Development selection, final performance, or strict source provenance failed."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    final_rows.to_csv(args.rows, index=False)
    print(json.dumps({"status": output["status"], **output["design"]}, indent=2))
    print(json.dumps(final_report["overall"], indent=2))
    print(f"Report: {args.report}")
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
