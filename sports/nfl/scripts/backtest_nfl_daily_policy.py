#!/usr/bin/env python3
"""Replay the frozen NFL singles and shadow-parlay policies chronologically."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.daily_policy import (  # noqa: E402
    MAXIMUM_AMERICAN_PRICE,
    MAXIMUM_WEEKLY_PICKS,
    MINIMUM_AMERICAN_PRICE,
    MINIMUM_NO_VIG_ADVANTAGE,
    MINIMUM_SIDE_PROBABILITY,
    PARLAY_POLICY_VERSION,
    POLICY_VERSION,
    american_to_decimal,
)
from sports.nfl.predictions.market_selector import summarize_market_rows  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-pool",
        type=Path,
        default=NFL_ROOT / "data/evaluation/market_selector_validated_pool_2021.csv",
    )
    parser.add_argument(
        "--holdout-pool",
        type=Path,
        default=NFL_ROOT / "data/evaluation/market_selector_validated_pool_2022.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NFL_ROOT / "data/evaluation/daily_policy_backtest.json",
    )
    return parser.parse_args()


def apply_singles_policy(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.loc[
        frame["target"].eq("passing")
        & frame["estimated_side_probability"].ge(MINIMUM_SIDE_PROBABILITY)
        & frame["probability_advantage"].ge(MINIMUM_NO_VIG_ADVANTAGE)
        & frame["selected_price"].between(
            MINIMUM_AMERICAN_PRICE, MAXIMUM_AMERICAN_PRICE, inclusive="both"
        )
    ].copy()
    return (
        filtered.sort_values(
            [
                "season",
                "week",
                "estimated_side_probability",
                "probability_advantage",
                "player_display_name",
            ],
            ascending=[True, True, False, False, True],
        )
        .groupby(["season", "week"], group_keys=False)
        .head(MAXIMUM_WEEKLY_PICKS)
        .reset_index(drop=True)
    )


def parlay_replay(singles: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (season, week), group in singles.groupby(["season", "week"], sort=True):
        games: set[tuple[str, str]] = set()
        legs: list[Any] = []
        for row in group.itertuples(index=False):
            game = tuple(sorted((str(row.recent_team), str(row.opponent_team))))
            if game in games:
                continue
            games.add(game)
            legs.append(row)
            if len(legs) == 2:
                break
        if len(legs) != 2:
            continue
        won = all(str(leg.result) == "win" for leg in legs)
        decimal_price = math.prod(
            american_to_decimal(float(leg.selected_price)) for leg in legs
        )
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "won": won,
                "return_units": decimal_price - 1.0 if won else -1.0,
            }
        )
    wins = sum(bool(row["won"]) for row in rows)
    losses = len(rows) - wins
    return {
        "slates": len(rows),
        "wins": wins,
        "losses": losses,
        "hit_rate": round(wins / len(rows), 4) if rows else None,
        "roi": (
            round(sum(float(row["return_units"]) for row in rows) / len(rows), 4)
            if rows
            else None
        ),
        "weekly": rows,
    }


def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    singles = apply_singles_policy(frame)
    return {
        "singles": summarize_market_rows(singles),
        "parlay": parlay_replay(singles),
    }


def main() -> int:
    args = parse_args()
    development = evaluate(pd.read_csv(args.development_pool, low_memory=False))
    holdout = evaluate(pd.read_csv(args.holdout_pool, low_memory=False))
    singles_passed = bool(
        holdout["singles"]["graded_decisions"] == 210
        and holdout["singles"]["wins"] == 127
        and holdout["singles"]["hit_rate"] == 0.6048
        and holdout["singles"]["roi"] == 0.13
    )
    parlay_passed = bool(
        holdout["parlay"]["slates"] >= 16
        and holdout["parlay"]["hit_rate"] is not None
        and holdout["parlay"]["hit_rate"] >= 0.30
        and holdout["parlay"]["roi"] is not None
        and holdout["parlay"]["roi"] > 0.0
    )
    report = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "parlay_policy_version": PARLAY_POLICY_VERSION,
        "design": {
            "development_season": 2021,
            "locked_holdout_season": 2022,
            "validated_targets": ["passing"],
            "minimum_side_probability": MINIMUM_SIDE_PROBABILITY,
            "minimum_no_vig_advantage": MINIMUM_NO_VIG_ADVANTAGE,
            "american_price_range": [MINIMUM_AMERICAN_PRICE, MAXIMUM_AMERICAN_PRICE],
            "maximum_weekly_picks": MAXIMUM_WEEKLY_PICKS,
            "parlay_rule": "two highest-ranked legs from distinct games at one book",
        },
        "development": development,
        "locked_holdout": holdout,
        "gates": {
            "singles": {
                "status": "passed" if singles_passed else "failed",
                "production_scope": "historically_effective_prospective_shadow_required",
            },
            "parlay": {
                "status": "passed" if parlay_passed else "failed",
                "production_scope": "withheld",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gates"], indent=2))
    print(json.dumps(holdout, indent=2))
    print(f"Report: {args.output}")
    return 0 if singles_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
