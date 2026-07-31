#!/usr/bin/env python3
"""Compare predictive latent-state hybrids with the current NFL champions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.latent import PredictiveLatentEncoder, build_sequence_table  # noqa: E402
from sports.nfl.predictions.pbp_stats import load_aggregated_season  # noqa: E402
from sports.nfl.predictions.pipeline import (  # noqa: E402
    ARCHITECTURES,
    TARGET_SPECS,
    _architecture_prediction,
    _component_predictions,
    _fit_components,
    build_features,
    load_weekly_stats,
)


NFL_ROOT = REPO_ROOT / "sports" / "nfl"
LATENT_COMPONENTS = ("xgboost", "catboost")
LATENT_CONTEXT = [
    "season_progress",
    "early_season",
    "games_played_prior",
    "opponent_allowed_roll3",
    "opponent_allowed_roll8",
    "position_QB",
    "position_RB",
    "position_FB",
    "position_WR",
    "position_TE",
    "position_OTHER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--selection-seasons", default="2021,2022,2023,2024")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--latent-dimensions", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "latent_challenger_report.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "latent_challenger_rows.csv",
    )
    return parser.parse_args()


def _load_stats(holdout_season: int) -> pd.DataFrame:
    historical = load_weekly_stats(
        NFL_ROOT / "data" / "raw" / "player_stats.parquet",
        start_season=2018,
        end_season=holdout_season - 1,
    )
    holdout = load_aggregated_season(
        holdout_season,
        cache_path=NFL_ROOT / "data" / "raw" / f"player_stats_{holdout_season}_pbp.parquet",
    )
    return pd.concat([historical, holdout], ignore_index=True).sort_values(
        ["season", "week", "player_id"]
    ).reset_index(drop=True)


def _augment(frame: pd.DataFrame, latent: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.merge(latent, on=["player_id", "season", "week"], how="left", validate="one_to_one")
    latent_columns = [column for column in latent.columns if column.startswith("latent_")]
    if output[latent_columns].isna().any().any():
        raise ValueError("Latent-state join left unmatched target rows.")
    return output, latent_columns


def _latent_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    raw_features: list[str],
    latent_features: list[str],
    target: str,
    random_state: int,
) -> dict[str, np.ndarray]:
    raw_latent_features = raw_features + latent_features
    context_features = [column for column in LATENT_CONTEXT if column in train.columns] + latent_features
    raw_models = _fit_components(
        train, raw_latent_features, target, random_state, LATENT_COMPONENTS
    )
    context_models = _fit_components(
        train, context_features, target, random_state + 1000, LATENT_COMPONENTS
    )
    raw_values = _component_predictions(raw_models, validation, raw_latent_features)
    context_values = _component_predictions(context_models, validation, context_features)
    return {
        "raw_latent_xgboost": raw_values["xgboost"],
        "raw_latent_catboost": raw_values["catboost"],
        "raw_latent_boost_blend": 0.5 * raw_values["xgboost"] + 0.5 * raw_values["catboost"],
        "latent_context_xgboost": context_values["xgboost"],
        "latent_context_catboost": context_values["catboost"],
        "latent_context_boost_blend": 0.5 * context_values["xgboost"] + 0.5 * context_values["catboost"],
    }


def _candidate_mae(actual: pd.Series, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(mean_absolute_error(actual, values)) for name, values in predictions.items()}


def _select_challenger(
    actual: pd.Series,
    current_predictions: dict[str, np.ndarray],
    latent_predictions: dict[str, np.ndarray],
) -> tuple[str, str, dict[str, float], dict[str, np.ndarray]]:
    current_scores = _candidate_mae(actual, current_predictions)
    current_name = min(current_scores, key=current_scores.get)
    expanded = dict(latent_predictions)
    for name, values in latent_predictions.items():
        expanded[f"current_plus_{name}"] = 0.5 * current_predictions[current_name] + 0.5 * values
    latent_scores = _candidate_mae(actual, expanded)
    challenger_name = min(latent_scores, key=latent_scores.get)
    all_scores = {**current_scores, **latent_scores}
    return current_name, challenger_name, all_scores, expanded


def _resolve_challenger_prediction(
    name: str,
    current_prediction: np.ndarray,
    latent_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    if name.startswith("current_plus_"):
        base_name = name.removeprefix("current_plus_")
        return 0.5 * current_prediction + 0.5 * latent_predictions[base_name]
    return latent_predictions[name]


def _bootstrap_delta(rows: pd.DataFrame, random_state: int, samples: int = 4000) -> list[float]:
    grouped = rows.groupby("week", sort=True).agg(
        delta_sum=("absolute_error_delta", "sum"), rows=("absolute_error_delta", "size")
    )
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(grouped))
    estimates = np.empty(samples, dtype=float)
    delta = grouped["delta_sum"].to_numpy()
    counts = grouped["rows"].to_numpy()
    for index in range(samples):
        selected = rng.choice(indices, size=len(indices), replace=True)
        estimates[index] = delta[selected].sum() / counts[selected].sum()
    return [round(float(value), 4) for value in np.quantile(estimates, [0.025, 0.975])]


def main() -> int:
    args = parse_args()
    selection_seasons = tuple(int(value) for value in args.selection_seasons.split(","))
    stats = _load_stats(args.holdout_season)
    sequence_table, sequence_features, sequence_targets = build_sequence_table(
        stats, sequence_length=args.sequence_length
    )
    target_frames = {spec.key: build_features(stats, spec) for spec in TARGET_SPECS}
    oof: dict[str, list[pd.DataFrame]] = {spec.key: [] for spec in TARGET_SPECS}
    encoder_audit: list[dict[str, object]] = []

    for season in selection_seasons:
        encoder_train = sequence_table.loc[sequence_table["season"].lt(season)]
        encoder = PredictiveLatentEncoder(
            sequence_length=args.sequence_length,
            latent_dimensions=args.latent_dimensions,
            random_state=args.random_state + season,
        ).fit(encoder_train, sequence_features, sequence_targets)
        latent = encoder.transform_frame(sequence_table.loc[sequence_table["season"].le(season)])
        encoder_audit.append(
            {
                "validation_season": season,
                "training_rows": encoder.training_rows_,
                "iterations": encoder.iterations_,
                "pretraining_validation_score": round(encoder.pretraining_validation_score_, 6),
            }
        )
        print(f"latent fold {season}: n={encoder.training_rows_}, iterations={encoder.iterations_}", flush=True)
        for spec in TARGET_SPECS:
            frame, raw_features = target_frames[spec.key]
            augmented, latent_features = _augment(frame.loc[frame["season"].le(season)], latent)
            train = augmented.loc[augmented["season"].lt(season)]
            validation = augmented.loc[augmented["season"].eq(season)]
            current_models = _fit_components(
                train, raw_features, spec.target, args.random_state + season
            )
            current_values = _component_predictions(current_models, validation, raw_features)
            latent_values = _latent_predictions(
                train,
                validation,
                raw_features,
                latent_features,
                spec.target,
                args.random_state + season,
            )
            part = validation[["season", "week", "player_id", spec.target]].rename(
                columns={spec.target: "actual"}
            )
            for architecture in ARCHITECTURES:
                part[f"current::{architecture}"] = _architecture_prediction(
                    architecture, validation, current_values
                )
            for name, values in latent_values.items():
                part[f"latent::{name}"] = values
            oof[spec.key].append(part)

    selections: dict[str, dict[str, object]] = {}
    for spec in TARGET_SPECS:
        pooled = pd.concat(oof[spec.key], ignore_index=True)
        current_values = {
            column.removeprefix("current::"): pooled[column].to_numpy()
            for column in pooled.columns
            if column.startswith("current::")
        }
        latent_values = {
            column.removeprefix("latent::"): pooled[column].to_numpy()
            for column in pooled.columns
            if column.startswith("latent::")
        }
        current_name, challenger_name, scores, _ = _select_challenger(
            pooled["actual"], current_values, latent_values
        )
        selections[spec.key] = {
            "current": current_name,
            "challenger": challenger_name,
            "selection_rows": int(len(pooled)),
            "candidate_mae": {
                name: round(value, 4) for name, value in sorted(scores.items(), key=lambda item: item[1])
            },
        }
        print(
            f"{spec.key} selection: current={current_name}, latent={challenger_name}", flush=True
        )

    encoder_train = sequence_table.loc[sequence_table["season"].lt(args.holdout_season)]
    final_encoder = PredictiveLatentEncoder(
        sequence_length=args.sequence_length,
        latent_dimensions=args.latent_dimensions,
        random_state=args.random_state,
    ).fit(encoder_train, sequence_features, sequence_targets)
    final_latent = final_encoder.transform_frame(sequence_table)
    scored_parts: list[pd.DataFrame] = []
    target_results: list[dict[str, object]] = []
    for spec in TARGET_SPECS:
        frame, raw_features = target_frames[spec.key]
        augmented, latent_features = _augment(frame, final_latent)
        train = augmented.loc[augmented["season"].lt(args.holdout_season)]
        holdout = augmented.loc[augmented["season"].eq(args.holdout_season)].copy()
        current_name = str(selections[spec.key]["current"])
        challenger_name = str(selections[spec.key]["challenger"])
        required_current = [name for name, _ in ARCHITECTURES[current_name]]
        current_models = _fit_components(
            train, raw_features, spec.target, args.random_state, required_current
        )
        current_components = _component_predictions(current_models, holdout, raw_features)
        current_prediction = _architecture_prediction(current_name, holdout, current_components)
        latent_values = _latent_predictions(
            train,
            holdout,
            raw_features,
            latent_features,
            spec.target,
            args.random_state,
        )
        challenger_prediction = _resolve_challenger_prediction(
            challenger_name, current_prediction, latent_values
        )
        actual = holdout[spec.target].astype(float).to_numpy()
        scored = holdout[
            ["season", "week", "player_id", "player_display_name", "position", "recent_team", "opponent_team"]
        ].copy()
        scored["target"] = spec.key
        scored["actual"] = actual
        scored["current_prediction"] = current_prediction
        scored["challenger_prediction"] = challenger_prediction
        scored["current_absolute_error"] = np.abs(actual - current_prediction)
        scored["challenger_absolute_error"] = np.abs(actual - challenger_prediction)
        scored["absolute_error_delta"] = (
            scored["challenger_absolute_error"] - scored["current_absolute_error"]
        )
        scored_parts.append(scored)
        current_mae = float(scored["current_absolute_error"].mean())
        challenger_mae = float(scored["challenger_absolute_error"].mean())
        target_results.append(
            {
                "target": spec.key,
                "rows": int(len(scored)),
                "current_architecture": current_name,
                "challenger_architecture": challenger_name,
                "current_mae": round(current_mae, 4),
                "challenger_mae": round(challenger_mae, 4),
                "challenger_improvement": round((current_mae - challenger_mae) / current_mae, 4),
            }
        )

    scored_rows = pd.concat(scored_parts, ignore_index=True)
    current_weighted = float(scored_rows["current_absolute_error"].mean())
    challenger_weighted = float(scored_rows["challenger_absolute_error"].mean())
    improvement = (current_weighted - challenger_weighted) / current_weighted
    confidence_interval = _bootstrap_delta(scored_rows, args.random_state)
    selection_wins = all(
        item["candidate_mae"][str(item["challenger"])]
        < item["candidate_mae"][str(item["current"])]
        for item in selections.values()
    )
    no_material_target_regression = all(
        float(item["challenger_improvement"]) >= -0.01 for item in target_results
    )
    statistically_supported = confidence_interval[1] < 0
    replace = bool(
        selection_wins
        and improvement >= 0.005
        and no_material_target_regression
        and statistically_supported
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "replace_current" if replace else "keep_current",
        "holdout_season": args.holdout_season,
        "selection_seasons": list(selection_seasons),
        "latent_architecture": {
            "type": "predictive multi-output MLP bottleneck",
            "sequence_length": args.sequence_length,
            "latent_dimensions": args.latent_dimensions,
            "pretraining_targets": [column.removeprefix("next_") for column in sequence_targets],
            "downstream_models": list(LATENT_COMPONENTS),
        },
        "leakage_controls": [
            "Every latent input contains only prior player games.",
            "The encoder is refit using only seasons earlier than each validation fold.",
            "Downstream models are refit using only seasons earlier than the scored season.",
            "The 2025 holdout is not used for architecture or latent-state selection.",
        ],
        "encoder_folds": encoder_audit,
        "selections": selections,
        "holdout": {
            "rows": int(len(scored_rows)),
            "current_weighted_mae": round(current_weighted, 4),
            "challenger_weighted_mae": round(challenger_weighted, 4),
            "challenger_improvement": round(improvement, 4),
            "paired_week_bootstrap_mae_delta_95": confidence_interval,
            "targets": target_results,
        },
        "promotion_rule": {
            "challenger_wins_pre_holdout_selection_for_every_target": selection_wins,
            "minimum_holdout_weighted_mae_improvement": 0.005,
            "no_target_regression_beyond": -0.01,
            "paired_week_bootstrap_upper_bound_below_zero": statistically_supported,
            "passed": replace,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    scored_rows.to_csv(args.rows, index=False)
    print(json.dumps(report["holdout"], indent=2))
    print(f"Decision: {report['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
