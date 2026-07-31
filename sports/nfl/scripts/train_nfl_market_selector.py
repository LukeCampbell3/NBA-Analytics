#!/usr/bin/env python3
"""Train a selective NFL prop-side model and test it on a later season."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_selector import (  # noqa: E402
    build_frozen_latent_features,
    build_learning_frames,
    build_prediction_pool,
    build_weekly_validation,
    candidate_feature_sets,
    expanding_oof_probabilities,
    fit_selected_model,
    probability_metrics,
    score_probabilities,
    summarize_market_rows,
    target_promotion_gate,
)
from sports.nfl.predictions.pipeline import load_weekly_stats  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        type=Path,
        default=NFL_ROOT / "data" / "raw" / "player_stats.parquet",
    )
    parser.add_argument("--development-market-rows", type=Path, required=True)
    parser.add_argument("--final-market-rows", type=Path, required=True)
    parser.add_argument("--minimum-side-probability", type=float, default=0.56)
    parser.add_argument("--minimum-no-vig-advantage", type=float, default=0.025)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_selector_report.json",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=NFL_ROOT / "model" / "nfl_market_selector.joblib",
    )
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument(
        "--development-pool",
        type=Path,
        default=None,
        help="Eligible leakage-safe development picks; defaults beside the report.",
    )
    parser.add_argument(
        "--final-pool",
        type=Path,
        default=None,
        help="Eligible final-season picks; defaults beside the report.",
    )
    parser.add_argument(
        "--weekly-validation",
        type=Path,
        default=None,
        help="Week/target validation table; defaults beside the report.",
    )
    return parser.parse_args()


def _single_season(frame: pd.DataFrame, label: str) -> int:
    seasons = sorted(int(value) for value in frame["season"].dropna().unique())
    if len(seasons) != 1:
        raise ValueError(f"{label} rows must contain exactly one season; found {seasons}.")
    return seasons[0]


def main() -> int:
    args = parse_args()
    development_market = pd.read_csv(args.development_market_rows, low_memory=False)
    final_market = pd.read_csv(args.final_market_rows, low_memory=False)
    development_season = _single_season(development_market, "Development")
    final_season = _single_season(final_market, "Final")
    if development_season >= final_season:
        raise ValueError("Development season must be earlier than the final season.")

    stats = load_weekly_stats(
        args.stats,
        start_season=2018,
        end_season=final_season,
    )
    latent_encoder, latent, latent_audit = build_frozen_latent_features(
        stats,
        development_season=development_season,
        random_state=args.random_state + 378,
    )
    combined_market = pd.concat([development_market, final_market], ignore_index=True)
    frames, raw_features, latent_columns = build_learning_frames(
        stats,
        combined_market,
        latent=latent,
    )

    target_reports: list[dict] = []
    fitted_models: dict[str, dict] = {}
    development_parts: list[pd.DataFrame] = []
    final_parts: list[pd.DataFrame] = []
    validated_parts: list[pd.DataFrame] = []
    for target, combined in frames.items():
        development = combined.loc[combined["season"].eq(development_season)].copy()
        final = combined.loc[combined["season"].eq(final_season)].copy()
        feature_sets = candidate_feature_sets(raw_features[target], latent_columns)
        candidate_rows: dict[str, pd.DataFrame] = {}
        candidate_reports: list[dict] = []
        for architecture, features in feature_sets.items():
            oof = expanding_oof_probabilities(
                development,
                features,
                architecture=architecture,
                random_state=args.random_state,
            )
            candidate_rows[architecture] = oof
            selected = score_probabilities(
                oof,
                oof["over_probability"].to_numpy(),
                minimum_side_probability=args.minimum_side_probability,
                minimum_no_vig_advantage=args.minimum_no_vig_advantage,
            )
            candidate_reports.append(
                {
                    "architecture": architecture,
                    "features": len(features),
                    "probability_metrics": probability_metrics(oof),
                    "selective_market_metrics": summarize_market_rows(selected),
                }
            )
        candidate_reports.sort(
            key=lambda row: (
                row["probability_metrics"]["brier_score"],
                row["probability_metrics"]["log_loss"],
                len(feature_sets[row["architecture"]]),
            )
        )
        selected_architecture = candidate_reports[0]["architecture"]
        selected_features = feature_sets[selected_architecture]
        selected_oof = score_probabilities(
            candidate_rows[selected_architecture],
            candidate_rows[selected_architecture]["over_probability"].to_numpy(),
            minimum_side_probability=args.minimum_side_probability,
            minimum_no_vig_advantage=args.minimum_no_vig_advantage,
        )
        development_parts.append(selected_oof)
        model = fit_selected_model(
            development,
            selected_features,
            architecture=selected_architecture,
            random_state=args.random_state,
        )
        final_probability = model.predict_proba(final[selected_features])[:, 1]
        final_scored = score_probabilities(
            final,
            final_probability,
            minimum_side_probability=args.minimum_side_probability,
            minimum_no_vig_advantage=args.minimum_no_vig_advantage,
        )
        final_summary = summarize_market_rows(final_scored)
        promotion_gate = target_promotion_gate(final_summary)
        final_scored["target_promotion_status"] = promotion_gate["status"]
        final_parts.append(final_scored)
        if promotion_gate["status"] == "passed":
            validated_parts.append(final_scored)
        deployment_model = fit_selected_model(
            combined,
            selected_features,
            architecture=selected_architecture,
            random_state=args.random_state,
        )
        fitted_models[target] = {
            "architecture": selected_architecture,
            "features": selected_features,
            "model": deployment_model,
            "promotion_status": promotion_gate["status"],
        }
        target_reports.append(
            {
                "target": target,
                "selected_architecture": selected_architecture,
                "architecture_selection_metric": "lowest expanding-2021 Brier score",
                "development_walk_forward": summarize_market_rows(selected_oof),
                "final_test": final_summary,
                "promotion_gate": promotion_gate,
                "candidates": candidate_reports,
            }
        )

    final_rows = pd.concat(final_parts, ignore_index=True)
    development_rows = pd.concat(development_parts, ignore_index=True)
    validated_rows = (
        pd.concat(validated_parts, ignore_index=True)
        if validated_parts
        else final_rows.iloc[0:0].copy()
    )
    validated_summary = summarize_market_rows(validated_rows)
    validated_targets = [
        item["target"]
        for item in target_reports
        if item["promotion_gate"]["status"] == "passed"
    ]
    architecture_by_target = {
        item["target"]: item["selected_architecture"] for item in target_reports
    }
    promotion_by_target = {
        item["target"]: item["promotion_gate"]["status"] for item in target_reports
    }
    development_pool = build_prediction_pool(
        development_rows,
        evaluation_split="development_walk_forward",
        architecture_by_target=architecture_by_target,
        promotion_by_target=promotion_by_target,
    )
    final_pool = build_prediction_pool(
        final_rows,
        evaluation_split="final_test",
        architecture_by_target=architecture_by_target,
        promotion_by_target=promotion_by_target,
    )
    season_weeks = {
        development_season: sorted(
            int(value) for value in development_market["week"].dropna().unique()
        ),
        final_season: sorted(int(value) for value in final_market["week"].dropna().unique()),
    }
    weekly_validation = build_weekly_validation(
        [development_pool, final_pool],
        season_weeks=season_weeks,
        promotion_by_target=promotion_by_target,
        development_season=development_season,
    )
    development_pool_path = args.development_pool or args.report.with_name(
        f"market_selector_pool_{development_season}.csv"
    )
    final_pool_path = args.final_pool or args.report.with_name(
        f"market_selector_pool_{final_season}.csv"
    )
    weekly_validation_path = args.weekly_validation or args.report.with_name(
        "market_selector_weekly_validation.csv"
    )
    validated_development_pool = development_pool.loc[
        development_pool["target_final_validation_status"].eq("passed")
    ].copy()
    validated_final_pool = final_pool.loc[
        final_pool["target_final_validation_status"].eq("passed")
    ].copy()
    validated_development_pool_path = args.report.with_name(
        f"market_selector_validated_pool_{development_season}.csv"
    )
    validated_final_pool_path = args.report.with_name(
        f"market_selector_validated_pool_{final_season}.csv"
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partially_validated" if validated_targets else "failed",
        "design": {
            "development_season": development_season,
            "final_test_season": final_season,
            "development_method": "expanding two-week folds; weeks 11-18 scored only from prior weeks",
            "architecture_selection_metric": "Brier score, then log loss, then fewer features",
            "minimum_side_probability": args.minimum_side_probability,
            "minimum_no_vig_advantage": args.minimum_no_vig_advantage,
            "line_contract": "one earliest available posted pregame line per player/stat/book",
            "line_movement_used": False,
            "closing_line_used": False,
            "final_season_used_for_architecture_or_threshold_selection": False,
            "target_markets_promoted_independently": True,
            "artifact_refit_through_final_season_after_evaluation": True,
        },
        "latent_encoder": latent_audit,
        "targets": target_reports,
        "validated_targets": validated_targets,
        "validated_board_final_test": validated_summary,
        "prediction_pool_exports": {
            "development_pool": str(development_pool_path),
            "development_pool_rows": int(len(development_pool)),
            "development_unscored_warmup_weeks": list(range(1, 11)),
            "validated_development_pool": str(validated_development_pool_path),
            "validated_development_pool_rows": int(len(validated_development_pool)),
            "final_pool": str(final_pool_path),
            "final_pool_rows": int(len(final_pool)),
            "validated_final_pool": str(validated_final_pool_path),
            "validated_final_pool_rows": int(len(validated_final_pool)),
            "weekly_validation": str(weekly_validation_path),
        },
        "source_provenance_gate": {
            "status": "failed",
            "reason": (
                "The free Bovada archive identifies authentic posted lines but has no capture "
                "timestamps, so performance can be measured but opening/pregame timing cannot "
                "be independently authenticated."
            ),
        },
        "static_deployment_gate": {
            "status": "blocked",
            "reason": "Source provenance is not sufficient for static betting-board promotion.",
        },
    }
    artifact = {
        "schema_version": 1,
        "trained_at_utc": report["generated_at_utc"],
        "development_season": development_season,
        "minimum_side_probability": args.minimum_side_probability,
        "minimum_no_vig_advantage": args.minimum_no_vig_advantage,
        "validated_targets": validated_targets,
        "latent_encoder": latent_encoder,
        "models": fitted_models,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)
    development_pool_path.parent.mkdir(parents=True, exist_ok=True)
    final_pool_path.parent.mkdir(parents=True, exist_ok=True)
    weekly_validation_path.parent.mkdir(parents=True, exist_ok=True)
    development_pool.to_csv(development_pool_path, index=False)
    final_pool.to_csv(final_pool_path, index=False)
    weekly_validation.to_csv(weekly_validation_path, index=False)
    validated_development_pool.to_csv(validated_development_pool_path, index=False)
    validated_final_pool.to_csv(validated_final_pool_path, index=False)
    if args.rows is not None:
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        final_rows.to_csv(args.rows, index=False)
    print(json.dumps({"validated_targets": validated_targets, "final": validated_summary}, indent=2))
    print(f"Report: {args.report}")
    return 0 if validated_targets else 2


if __name__ == "__main__":
    raise SystemExit(main())
