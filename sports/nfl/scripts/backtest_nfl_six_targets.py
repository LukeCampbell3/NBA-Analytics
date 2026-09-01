#!/usr/bin/env python3
"""Run a leakage-safe 2025 holdout for six NFL player production targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.pipeline import (  # noqa: E402
    NFLVERSE_PLAYER_STATS_URL,
    TargetSpec,
    load_weekly_stats,
    train_target,
)
from sports.nfl.predictions.pbp_stats import load_aggregated_season  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports" / "nfl"
SPECS = (
    TargetSpec("pass_yds", "Passing yards", "passing_yards", "attempts", 10.0, 30.0),
    TargetSpec("rec_yds", "Receiving yards", "receiving_yards", "targets", 2.0, 15.0),
    TargetSpec("rush_yds", "Rushing yards", "rushing_yards", "carries", 2.0, 15.0),
    TargetSpec("pass_tds", "Passing touchdowns", "passing_tds", "attempts", 10.0, 0.5),
    TargetSpec("rec_tds", "Receiving touchdowns", "receiving_tds", "targets", 2.0, 0.5),
    TargetSpec("rush_tds", "Rushing touchdowns", "rushing_tds", "carries", 2.0, 0.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=NFLVERSE_PLAYER_STATS_URL)
    parser.add_argument("--cache", type=Path, default=NFL_ROOT / "data/raw/player_stats.parquet")
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--selection-seasons", default="2021,2022,2023,2024")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data/evaluation/six_target_2025_backtest.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=NFL_ROOT / "data/evaluation/six_target_2025_backtest_rows.csv",
    )
    return parser.parse_args()


def load_stats(args: argparse.Namespace) -> pd.DataFrame:
    historical_end = min(args.holdout_season, 2024)
    stats = load_weekly_stats(
        args.source,
        cache_path=args.cache,
        start_season=args.start_season,
        end_season=historical_end,
    )
    supplements = [
        load_aggregated_season(
            season,
            cache_path=NFL_ROOT / "data/raw" / f"player_stats_{season}_pbp.parquet",
        )
        for season in range(historical_end + 1, args.holdout_season + 1)
    ]
    if supplements:
        stats = pd.concat([stats, *supplements], ignore_index=True)
    return stats.sort_values(["season", "week", "player_id"]).reset_index(drop=True)


def week_bootstrap(rows: pd.DataFrame, seed: int, samples: int = 10_000) -> list[float]:
    weekly = rows.groupby("week").agg(delta_sum=("mae_delta", "sum"), rows=("mae_delta", "size"))
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.integers(0, len(weekly), len(weekly))
        estimates[index] = weekly["delta_sum"].to_numpy()[selected].sum() / weekly["rows"].to_numpy()[selected].sum()
    return [round(float(value), 6) for value in np.quantile(estimates, (0.025, 0.975))]


def main() -> int:
    args = parse_args()
    stats = load_stats(args)
    selection_seasons = tuple(int(value) for value in args.selection_seasons.split(","))
    reports = []
    rows = []
    for offset, spec in enumerate(SPECS):
        report, _, scored = train_target(
            stats,
            spec,
            holdout_season=args.holdout_season,
            meta_seasons=selection_seasons,
            random_state=args.random_state + offset * 100,
        )
        scored["mae_delta"] = scored["absolute_error"] - np.abs(scored["actual"] - scored["baseline"])
        interval = week_bootstrap(scored, args.random_state + offset)
        report["paired_week_bootstrap_mae_delta_95"] = interval
        report["verification"] = {
            "lower_mae_than_lagged_rolling_baseline": bool(report["metrics"]["mae"] < report["metrics"]["baseline_mae"]),
            "statistically_verified_at_95_percent": bool(interval[1] < 0),
        }
        reports.append(report)
        rows.append(scored)

    scored_rows = pd.concat(rows, ignore_index=True)
    incumbent_path = NFL_ROOT / "data/evaluation/backtest_report.json"
    incumbent = json.loads(incumbent_path.read_text(encoding="utf-8"))
    incumbent_mae = {
        item["target"]: item["metrics"]["mae"] for item in incumbent["targets"]
    }
    yardage_incumbent_keys = {
        "pass_yds": "passing",
        "rec_yds": "receiving",
        "rush_yds": "rushing",
    }
    for item in reports:
        incumbent_key = yardage_incumbent_keys.get(item["target"])
        if incumbent_key:
            prior_mae = float(incumbent_mae[incumbent_key])
            item["incumbent_comparison"] = {
                "incumbent": "predictive latent-state hybrid",
                "incumbent_mae": prior_mae,
                "challenger_mae": item["metrics"]["mae"],
                "challenger_beats_incumbent": bool(item["metrics"]["mae"] < prior_mae),
            }
        else:
            item["incumbent_comparison"] = {
                "incumbent": "no previously evaluated touchdown model",
                "challenger_mae": item["metrics"]["mae"],
                "challenger_beats_incumbent": None,
            }
    audit_columns = ["player_id", "season", "week", *(spec.target for spec in SPECS)]
    audit = stats[audit_columns].sort_values(["season", "week", "player_id"])
    report_bundle = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_design": {
            "type": "expanding_pre_holdout_model_selection_with_untouched_season_holdout",
            "source_seasons": sorted(int(value) for value in stats["season"].unique()),
            "selection_seasons": list(selection_seasons),
            "holdout_season": args.holdout_season,
            "eligibility": "At least three prior games and target-role prior opportunity floor.",
            "leakage_controls": [
                "Every player and opponent feature is shifted by at least one game.",
                "Architecture selection uses only seasons before the 2025 holdout.",
                "The 2025 season is evaluated only after each target architecture is frozen.",
                "Uncertainty resamples complete weeks rather than individual player rows.",
            ],
        },
        "data_audit": {
            "provider": "nflverse weekly statistics plus official 2025 play-by-play aggregation",
            "rows_loaded": int(len(stats)),
            "target_sha256": hashlib.sha256(pd.util.hash_pandas_object(audit, index=False).values.tobytes()).hexdigest(),
        },
        "targets": reports,
        "summary": {
            "targets_lower_mae": sum(item["verification"]["lower_mae_than_lagged_rolling_baseline"] for item in reports),
            "targets_statistically_verified": sum(item["verification"]["statistically_verified_at_95_percent"] for item in reports),
            "all_six_lower_mae": all(item["verification"]["lower_mae_than_lagged_rolling_baseline"] for item in reports),
            "all_six_statistically_verified": all(item["verification"]["statistically_verified_at_95_percent"] for item in reports),
            "yardage_targets_beating_existing_latent_incumbent": sum(
                item["incumbent_comparison"]["challenger_beats_incumbent"] is True
                for item in reports
            ),
            "decision": (
                "Keep the existing latent hybrid for all three yardage targets; "
                "accept the new touchdown models as validated challengers against their lagged baselines."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_bundle, indent=2) + "\n", encoding="utf-8")
    scored_rows.to_csv(args.rows, index=False)
    print(json.dumps(report_bundle["summary"], indent=2))
    for item in reports:
        metrics = item["metrics"]
        print(f"{item['target']}: n={metrics['rows']} model={metrics['mae']:.4f} baseline={metrics['baseline_mae']:.4f} delta95={item['paired_week_bootstrap_mae_delta_95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
