#!/usr/bin/env python3
"""Train and backtest the NFL player yardage stack."""

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

from sports.nfl.predictions.pipeline import (  # noqa: E402
    NFLVERSE_PLAYER_STATS_URL,
    load_weekly_stats,
    write_training_outputs,
)
from sports.nfl.predictions.latent_pipeline import train_and_backtest_latent  # noqa: E402
from sports.nfl.predictions.pbp_stats import load_aggregated_season  # noqa: E402
from sports.nfl.predictions.market_backtest import (  # noqa: E402
    evaluate_market_backtest,
    load_market_archive,
)


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=NFLVERSE_PLAYER_STATS_URL)
    parser.add_argument("--cache", type=Path, default=NFL_ROOT / "data" / "raw" / "player_stats.parquet")
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument(
        "--selection-seasons",
        default=None,
        help="Optional comma-separated pre-holdout architecture-selection seasons.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report", type=Path, default=NFL_ROOT / "data" / "evaluation" / "backtest_report.json")
    parser.add_argument("--rows", type=Path, default=NFL_ROOT / "data" / "evaluation" / "backtest_rows.csv")
    parser.add_argument("--artifact", type=Path, default=NFL_ROOT / "model" / "nfl_yardage_latent_hybrid.joblib")
    parser.add_argument(
        "--web-payload",
        type=Path,
        default=NFL_ROOT / "web" / "data" / "daily_predictions.json",
        help="Static-site backtest payload (historical holdout rows, not live picks).",
    )
    parser.add_argument(
        "--market-lines",
        type=Path,
        default=None,
        help="Optional authentic historical NFL prop CSV/parquet for hit-rate validation.",
    )
    parser.add_argument("--market-minimum-edge", type=float, default=0.0)
    parser.add_argument(
        "--market-report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_backtest_report.json",
    )
    parser.add_argument(
        "--market-rows",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_backtest_rows.csv",
    )
    return parser.parse_args()


def build_web_payload(report: dict, rows) -> dict:
    latest_season = int(rows["season"].max())
    latest_week = int(rows.loc[rows["season"].eq(latest_season), "week"].max())
    showcase = (
        rows.loc[rows["season"].eq(latest_season) & rows["week"].eq(latest_week)]
        .sort_values(["target", "player_display_name"])
        .groupby("target", group_keys=False)
        .head(8)
    )
    records = []
    for row in showcase.to_dict(orient="records"):
        records.append(
            {
                "player": row["player_display_name"],
                "team": row["recent_team"],
                "opponent": row["opponent_team"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "target": row["target"],
                "prediction": round(float(row["prediction"]), 1),
                "actual": round(float(row["actual"]), 1),
                "absolute_error": round(float(row["absolute_error"]), 1),
            }
        )
    return {
        "schema_version": 1,
        "run_date": report["generated_at_utc"][:10],
        "publication_status": (
            "validated_backtest"
            if report.get("promotion_gate", {}).get("status") == "passed"
            else "research_only"
        ),
        "mode": "historical_holdout",
        "holdout_season": report["evaluation_design"]["holdout_season"],
        "architecture": report["architecture"],
        "overall": report["overall"],
        "targets": report["targets"],
        "promotion_gate": report["promotion_gate"],
        "market_validation": report.get("market_validation", {}),
        "methodology": report["evaluation_design"],
        "plays": records,
    }


def main() -> int:
    args = parse_args()
    historical_end = min(args.holdout_season, 2024)
    stats = load_weekly_stats(
        args.source,
        cache_path=args.cache,
        start_season=args.start_season,
        end_season=historical_end,
    )
    if args.holdout_season > historical_end:
        supplements = [
            load_aggregated_season(
                season,
                cache_path=NFL_ROOT / "data" / "raw" / f"player_stats_{season}_pbp.parquet",
            )
            for season in range(historical_end + 1, args.holdout_season + 1)
        ]
        stats = pd.concat([stats, *supplements], ignore_index=True).sort_values(
            ["season", "week", "player_id"]
        )
    selection_seasons = (
        tuple(int(value.strip()) for value in args.selection_seasons.split(",") if value.strip())
        if args.selection_seasons
        else None
    )
    report, artifact, rows = train_and_backtest_latent(
        stats,
        holdout_season=args.holdout_season,
        selection_seasons=selection_seasons,
        random_state=args.random_state,
    )
    if args.market_lines is not None:
        market_report, market_rows = evaluate_market_backtest(
            rows,
            load_market_archive(args.market_lines),
            minimum_edge_yards=args.market_minimum_edge,
        )
        report["market_validation"] = market_report
        projection_passed = report["promotion_gate"].get("projection_status") == "passed"
        market_passed = market_report["promotion_gate"]["status"] == "passed"
        report["promotion_gate"]["status"] = (
            "passed" if projection_passed and market_passed else "failed"
        )
        report["promotion_gate"]["reason"] = (
            "Projection and authentic historical-market gates passed."
            if report["promotion_gate"]["status"] == "passed"
            else "Projection or authentic historical-market gate did not pass."
        )
        args.market_report.parent.mkdir(parents=True, exist_ok=True)
        args.market_rows.parent.mkdir(parents=True, exist_ok=True)
        args.market_report.write_text(
            json.dumps(market_report, indent=2) + "\n", encoding="utf-8"
        )
        market_rows.to_csv(args.market_rows, index=False)
    write_training_outputs(
        report,
        artifact,
        rows,
        report_path=args.report,
        artifact_path=args.artifact,
        rows_path=args.rows,
    )
    web_payload = build_web_payload(report, rows)
    args.web_payload.parent.mkdir(parents=True, exist_ok=True)
    args.web_payload.write_text(json.dumps(web_payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["overall"], indent=2))
    for target in report["targets"]:
        metrics = target["metrics"]
        print(
            f"{target['label']}: n={metrics['rows']}, MAE={metrics['mae']:.2f}, "
            f"within {metrics['tolerance_yards']:.0f} yd={metrics['within_tolerance_accuracy']:.1%}, "
            f"vs baseline={metrics['mae_improvement_vs_rolling_baseline']:.1%}"
        )
    print(f"Report: {args.report}")
    print(f"Model artifact: {args.artifact}")
    print(f"Static payload: {args.web_payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
