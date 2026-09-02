#!/usr/bin/env python3
"""Evaluate the frozen live selection rule by NFL player-prop capability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_selector import summarize_market_rows  # noqa: E402
from sports.nfl.predictions.pick_meta import apply_meta_policy  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports/nfl"
CAPABILITIES = ("passing", "rushing", "receiving")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-report",
        type=Path,
        default=NFL_ROOT / "data/evaluation/pick_meta_backtest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NFL_ROOT / "data/evaluation/week_market_policy_backtest.json",
    )
    return parser.parse_args()


def _apply(frame: pd.DataFrame, capability: str, policy: dict[str, Any]) -> pd.DataFrame:
    # apply_meta_policy intentionally validates passing only. Replaying the exact
    # numerical contract here measures transfer without granting it authority.
    eligible = frame.loc[
        frame["target"].eq(capability)
        & frame["estimated_side_probability"].ge(policy["minimum_side_probability"])
        & frame["probability_advantage"].ge(policy["minimum_no_vig_advantage"])
        & frame["selected_price"].between(
            policy["minimum_price"], policy["maximum_price"], inclusive="both"
        )
    ].copy()
    return (
        eligible.sort_values(
            ["season", "week", "estimated_side_probability", "probability_advantage", "player_display_name"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["season", "week"], group_keys=False, sort=True)
        .head(int(policy["weekly_cap"]))
        .reset_index(drop=True)
    )


def build_report(policy_report: dict[str, Any], pools: dict[str, pd.DataFrame]) -> dict[str, Any]:
    policy = policy_report["selected_policy"]
    results: dict[str, Any] = {}
    for capability in CAPABILITIES:
        periods = {
            name: summarize_market_rows(_apply(frame, capability, policy))
            for name, frame in pools.items()
        }
        positive_all = all(
            metrics.get("roi") is not None and metrics["roi"] > 0
            for metrics in periods.values()
        )
        enough_recent = periods["confirmation_2025"]["graded_decisions"] >= 50
        authority = capability == "passing" and positive_all and enough_recent
        results[capability] = {
            "state": "BACKTEST_VALIDATED_SHADOW" if authority else "NO_RELIABLE_EDGE_FOUND",
            "selection_authority": authority,
            "periods": periods,
        }
    return {
        "schema_version": 1,
        "artifact_type": "nfl_week_market_policy_backtest",
        "policy_version": policy_report["model_version"],
        "evidence_as_of_utc": policy_report["evidence_as_of_utc"],
        "policy": policy,
        "ranking_rule": "Within each week: estimated side probability, then no-vig advantage.",
        "design": {
            "policy_selected_on": "2025 weeks 1-12 development only",
            "locked_period": "2025 weeks 13-18",
            "stress_periods": [2021, 2022],
            "capability_transfer_rule": "A capability receives authority only when every evaluated period has positive ROI and recent evidence has at least 50 decisions.",
            "parlays": "No authority; locked 2022 replay was 2-16.",
        },
        "capabilities": results,
    }


def main() -> int:
    args = parse_args()
    policy_report = json.loads(args.policy_report.read_text(encoding="utf-8"))
    evaluation = NFL_ROOT / "data/evaluation"
    pools = {
        "stress_2021": pd.read_csv(evaluation / "market_selector_pool_2021.csv", low_memory=False),
        "stress_2022": pd.read_csv(evaluation / "market_selector_pool_2022.csv", low_memory=False),
        "confirmation_2025": pd.read_csv(evaluation / "recent_selector_pool_2025.csv", low_memory=False),
    }
    report = build_report(policy_report, pools)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["state"] for key, value in report["capabilities"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
