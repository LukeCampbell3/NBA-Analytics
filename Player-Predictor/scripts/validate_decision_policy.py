#!/usr/bin/env python3
"""
Walk-forward validation for the full market decision policy.

This replays historical decisions through the stages that determine profit:
prediction -> gating -> selection -> sizing -> outcome -> validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine.gating import StrategyConfig
from decision_engine.policy_tuning import build_default_shadow_strategies, build_grid_strategies, run_policy_tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and tune the market decision policy with historical replay.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "historical_market_decisions_long.csv",
        help="Long-form historical decisions CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "decision_policy",
        help="Directory for simulation outputs.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="ending_bankroll",
        choices=["ending_bankroll", "total_profit", "profit_per_opportunity", "log_bankroll_growth", "roi"],
        help="Objective used to rank strategies.",
    )
    parser.add_argument(
        "--starting-bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll used for simulation.",
    )
    parser.add_argument(
        "--strategies-json",
        type=Path,
        default=None,
        help="Optional JSON file with explicit strategy configs.",
    )
    parser.add_argument("--grid-search", action="store_true", help="Generate counterfactual grid-search strategies.")
    parser.add_argument("--min-final-confidence-grid", type=str, default="0.02,0.03,0.04", help="Comma-separated min final confidence values.")
    parser.add_argument("--min-ev-grid", type=str, default="-0.01,0.0,0.01", help="Comma-separated EV cutoffs.")
    parser.add_argument("--max-total-plays-grid", type=str, default="8,12,16", help="Comma-separated max total plays values.")
    parser.add_argument("--non-pts-gap-grid", type=str, default="0.85,0.90,0.93", help="Comma-separated non-PTS percentile cutoffs.")
    parser.add_argument("--kelly-fraction-grid", type=str, default="0.15,0.25,0.35", help="Comma-separated Kelly fractions.")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_strategy_configs(args: argparse.Namespace) -> list[StrategyConfig]:
    if args.strategies_json:
        payload = json.loads(args.strategies_json.read_text(encoding="utf-8"))
        return [StrategyConfig(**item) for item in payload]

    if args.grid_search:
        base = StrategyConfig(name="grid_base", starting_bankroll=float(args.starting_bankroll))
        return build_grid_strategies(
            min_final_confidences=parse_float_list(args.min_final_confidence_grid),
            min_evs=parse_float_list(args.min_ev_grid),
            max_total_plays=parse_int_list(args.max_total_plays_grid),
            non_pts_percentiles=parse_float_list(args.non_pts_gap_grid),
            kelly_fractions=parse_float_list(args.kelly_fraction_grid),
            base_config=base,
        )

    configs = build_default_shadow_strategies()
    for config in configs:
        config.starting_bankroll = float(args.starting_bankroll)
    return configs


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Historical decisions CSV not found: {args.csv}")

    configs = load_strategy_configs(args)
    results, summary_df = run_policy_tuning(args.csv, configs, objective=args.objective)
    if summary_df.empty:
        raise RuntimeError("No policy simulation results were produced.")

    best_strategy_name = str(summary_df.iloc[0]["strategy"])
    best_result = next(result for result in results if result.config.name == best_strategy_name)

    args.outdir.mkdir(parents=True, exist_ok=True)
    strategies_csv = args.outdir / "strategy_results.csv"
    all_decisions_csv = args.outdir / "all_strategy_decisions.csv"
    all_daily_csv = args.outdir / "all_strategy_daily.csv"
    best_decisions_csv = args.outdir / "best_strategy_decisions.csv"
    best_daily_csv = args.outdir / "best_strategy_daily.csv"
    summary_json = args.outdir / "decision_policy_summary.json"

    summary_df.to_csv(strategies_csv, index=False)
    pd.concat([result.decisions for result in results], ignore_index=True).to_csv(all_decisions_csv, index=False)
    pd.concat([result.daily for result in results], ignore_index=True).to_csv(all_daily_csv, index=False)
    best_result.decisions.to_csv(best_decisions_csv, index=False)
    best_result.daily.to_csv(best_daily_csv, index=False)

    payload = {
        "source_csv": str(args.csv.resolve()),
        "objective": args.objective,
        "strategies_tested": len(configs),
        "best_strategy": best_strategy_name,
        "best_summary": best_result.summary,
        "strategy_results": summary_df.to_dict(orient="records"),
        "artifacts": {
            "strategy_results_csv": str(strategies_csv.resolve()),
            "all_decisions_csv": str(all_decisions_csv.resolve()),
            "all_daily_csv": str(all_daily_csv.resolve()),
            "best_strategy_decisions_csv": str(best_decisions_csv.resolve()),
            "best_strategy_daily_csv": str(best_daily_csv.resolve()),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    print("DECISION POLICY VALIDATION COMPLETE")
    print("=" * 100)
    print(f"Historical CSV:      {args.csv.resolve()}")
    print(f"Strategies tested:   {len(configs)}")
    print(f"Objective:           {args.objective}")
    print(f"Best strategy:       {best_strategy_name}")
    print(f"Strategy results:    {strategies_csv}")
    print(f"Best decisions CSV:  {best_decisions_csv}")
    print(f"Summary JSON:        {summary_json}")
    print("\nTop strategies:")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
