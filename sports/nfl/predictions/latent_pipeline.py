"""Production training for the validated predictive-latent NFL hybrid."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .latent import PredictiveLatentEncoder, build_sequence_table
from .pipeline import (
    ARCHITECTURES,
    IDENTITY_COLUMNS,
    TARGET_SPECS,
    TargetSpec,
    _aggregate_metrics,
    _architecture_prediction,
    _component_predictions,
    _fit_components,
    _metrics,
    _position_metrics,
    _promotion_gate,
    build_features,
)


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


def _augment(frame: pd.DataFrame, latent: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.merge(latent, on=["player_id", "season", "week"], how="left", validate="one_to_one")
    latent_columns = [column for column in latent.columns if column.startswith("latent_")]
    if output[latent_columns].isna().any().any():
        raise ValueError("Latent-state join left unmatched player-week rows.")
    return output, latent_columns


def _latent_feature_sets(
    raw_features: list[str], latent_features: list[str], frame: pd.DataFrame
) -> dict[str, list[str]]:
    return {
        "raw_latent": raw_features + latent_features,
        "latent_context": [column for column in LATENT_CONTEXT if column in frame.columns]
        + latent_features,
    }


def _all_latent_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    raw_features: list[str],
    latent_features: list[str],
    target: str,
    random_state: int,
) -> dict[str, np.ndarray]:
    feature_sets = _latent_feature_sets(raw_features, latent_features, train)
    output: dict[str, np.ndarray] = {}
    for offset, (prefix, features) in enumerate(feature_sets.items()):
        models = _fit_components(
            train, features, target, random_state + offset * 1000, LATENT_COMPONENTS
        )
        values = _component_predictions(models, validation, features)
        output[f"{prefix}_xgboost"] = values["xgboost"]
        output[f"{prefix}_catboost"] = values["catboost"]
        output[f"{prefix}_boost_blend"] = 0.5 * values["xgboost"] + 0.5 * values["catboost"]
    return output


def _select(
    actual: pd.Series,
    current: dict[str, np.ndarray],
    latent: dict[str, np.ndarray],
) -> tuple[str, str, dict[str, float]]:
    current_mae = {
        name: float(mean_absolute_error(actual, values)) for name, values in current.items()
    }
    current_name = min(current_mae, key=current_mae.get)
    expanded = dict(latent)
    for name, values in latent.items():
        expanded[f"current_plus_{name}"] = 0.5 * current[current_name] + 0.5 * values
    latent_mae = {
        name: float(mean_absolute_error(actual, values)) for name, values in expanded.items()
    }
    challenger_name = min(latent_mae, key=latent_mae.get)
    return current_name, challenger_name, {**current_mae, **latent_mae}


def _selected_latent_configuration(
    challenger_name: str,
    raw_features: list[str],
    latent_features: list[str],
    frame: pd.DataFrame,
) -> tuple[str, list[str], tuple[str, ...]]:
    base_name = challenger_name.removeprefix("current_plus_")
    prefix = "raw_latent" if base_name.startswith("raw_latent_") else "latent_context"
    suffix = base_name.removeprefix(prefix + "_")
    components = {
        "xgboost": ("xgboost",),
        "catboost": ("catboost",),
        "boost_blend": ("xgboost", "catboost"),
    }[suffix]
    features = _latent_feature_sets(raw_features, latent_features, frame)[prefix]
    return base_name, features, components


def _selected_latent_prediction(
    base_name: str,
    models: dict[str, Any],
    frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    values = _component_predictions(models, frame, features)
    if base_name.endswith("_xgboost"):
        return values["xgboost"]
    if base_name.endswith("_catboost"):
        return values["catboost"]
    return 0.5 * values["xgboost"] + 0.5 * values["catboost"]


def _combine_challenger(
    challenger_name: str, current: np.ndarray, latent: np.ndarray
) -> np.ndarray:
    if challenger_name.startswith("current_plus_"):
        return 0.5 * current + 0.5 * latent
    return latent


def _week_bootstrap_delta(
    scored: pd.DataFrame, random_state: int, samples: int = 4000
) -> list[float]:
    grouped = scored.groupby("week", sort=True).agg(
        delta_sum=("absolute_error_delta", "sum"), rows=("absolute_error_delta", "size")
    )
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(grouped))
    deltas = grouped["delta_sum"].to_numpy()
    counts = grouped["rows"].to_numpy()
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(indices, size=len(indices), replace=True)
        estimates[index] = deltas[selected].sum() / counts[selected].sum()
    return [round(float(value), 4) for value in np.quantile(estimates, [0.025, 0.975])]


def train_and_backtest_latent(
    stats: pd.DataFrame,
    *,
    holdout_season: int,
    selection_seasons: Iterable[int] | None = None,
    sequence_length: int = 8,
    latent_dimensions: int = 16,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Select the hybrid pre-holdout, test it once, and build a deployable bundle."""

    selection_values = tuple(
        selection_seasons or range(holdout_season - 4, holdout_season)
    )
    sequence_table, sequence_features, sequence_targets = build_sequence_table(
        stats, sequence_length=sequence_length
    )
    target_frames = {spec.key: build_features(stats, spec) for spec in TARGET_SPECS}
    oof: dict[str, list[pd.DataFrame]] = {spec.key: [] for spec in TARGET_SPECS}
    encoder_folds: list[dict[str, Any]] = []

    for season in selection_values:
        encoder = PredictiveLatentEncoder(
            sequence_length=sequence_length,
            latent_dimensions=latent_dimensions,
            random_state=random_state + season,
        ).fit(
            sequence_table.loc[sequence_table["season"].lt(season)],
            sequence_features,
            sequence_targets,
        )
        latent = encoder.transform_frame(
            sequence_table.loc[sequence_table["season"].le(season)]
        )
        encoder_folds.append(
            {
                "validation_season": season,
                "training_rows": encoder.training_rows_,
                "iterations": encoder.iterations_,
                "pretraining_validation_score": round(
                    encoder.pretraining_validation_score_, 6
                ),
            }
        )
        for spec in TARGET_SPECS:
            frame, raw_features = target_frames[spec.key]
            augmented, latent_features = _augment(
                frame.loc[frame["season"].le(season)], latent
            )
            train = augmented.loc[augmented["season"].lt(season)]
            validation = augmented.loc[augmented["season"].eq(season)]
            current_models = _fit_components(
                train, raw_features, spec.target, random_state + season
            )
            current_components = _component_predictions(
                current_models, validation, raw_features
            )
            latent_values = _all_latent_predictions(
                train,
                validation,
                raw_features,
                latent_features,
                spec.target,
                random_state + season,
            )
            part = pd.DataFrame({"actual": validation[spec.target].astype(float).to_numpy()})
            for architecture in ARCHITECTURES:
                part[f"current::{architecture}"] = _architecture_prediction(
                    architecture, validation, current_components
                )
            for name, values in latent_values.items():
                part[f"latent::{name}"] = values
            oof[spec.key].append(part)

    selections: dict[str, dict[str, Any]] = {}
    for spec in TARGET_SPECS:
        pooled = pd.concat(oof[spec.key], ignore_index=True)
        current_values = {
            column.removeprefix("current::"): pooled[column].to_numpy()
            for column in pooled
            if column.startswith("current::")
        }
        latent_values = {
            column.removeprefix("latent::"): pooled[column].to_numpy()
            for column in pooled
            if column.startswith("latent::")
        }
        current_name, challenger_name, candidate_mae = _select(
            pooled["actual"], current_values, latent_values
        )
        selections[spec.key] = {
            "current_architecture": current_name,
            "challenger_architecture": challenger_name,
            "selection_rows": int(len(pooled)),
            "candidate_mae": {
                name: round(value, 4)
                for name, value in sorted(candidate_mae.items(), key=lambda item: item[1])
            },
        }

    final_encoder = PredictiveLatentEncoder(
        sequence_length=sequence_length,
        latent_dimensions=latent_dimensions,
        random_state=random_state,
    ).fit(
        sequence_table.loc[sequence_table["season"].lt(holdout_season)],
        sequence_features,
        sequence_targets,
    )
    final_latent = final_encoder.transform_frame(sequence_table)
    comparison_parts: list[pd.DataFrame] = []
    evaluation_models: dict[str, dict[str, Any]] = {}
    for spec in TARGET_SPECS:
        frame, raw_features = target_frames[spec.key]
        augmented, latent_features = _augment(frame, final_latent)
        train = augmented.loc[augmented["season"].lt(holdout_season)]
        holdout = augmented.loc[augmented["season"].eq(holdout_season)].copy()
        selection = selections[spec.key]
        current_name = selection["current_architecture"]
        challenger_name = selection["challenger_architecture"]
        current_components_needed = [name for name, _ in ARCHITECTURES[current_name]]
        current_models = _fit_components(
            train, raw_features, spec.target, random_state, current_components_needed
        )
        current_components = _component_predictions(current_models, holdout, raw_features)
        current_prediction = _architecture_prediction(
            current_name, holdout, current_components
        )
        base_name, latent_model_features, latent_components_needed = _selected_latent_configuration(
            challenger_name, raw_features, latent_features, train
        )
        latent_models = _fit_components(
            train,
            latent_model_features,
            spec.target,
            random_state,
            latent_components_needed,
        )
        latent_prediction = _selected_latent_prediction(
            base_name, latent_models, holdout, latent_model_features
        )
        challenger_prediction = _combine_challenger(
            challenger_name, current_prediction, latent_prediction
        )
        actual = holdout[spec.target].astype(float).to_numpy()
        comparison = holdout[IDENTITY_COLUMNS + ["baseline_prediction"]].copy()
        comparison["target"] = spec.key
        comparison["actual"] = actual
        comparison["current_prediction"] = current_prediction
        comparison["challenger_prediction"] = challenger_prediction
        comparison["current_absolute_error"] = np.abs(actual - current_prediction)
        comparison["challenger_absolute_error"] = np.abs(actual - challenger_prediction)
        comparison["absolute_error_delta"] = (
            comparison["challenger_absolute_error"]
            - comparison["current_absolute_error"]
        )
        comparison_parts.append(comparison)
        evaluation_models[spec.key] = {
            "spec": spec,
            "raw_features": raw_features,
            "latent_features": latent_features,
            "current_architecture": current_name,
            "challenger_architecture": challenger_name,
            "base_latent_architecture": base_name,
            "latent_model_features": latent_model_features,
        }

    comparison_rows = pd.concat(comparison_parts, ignore_index=True)
    current_weighted_mae = float(comparison_rows["current_absolute_error"].mean())
    challenger_weighted_mae = float(comparison_rows["challenger_absolute_error"].mean())
    challenger_improvement = (
        current_weighted_mae - challenger_weighted_mae
    ) / current_weighted_mae
    confidence_interval = _week_bootstrap_delta(comparison_rows, random_state)
    target_comparison: list[dict[str, Any]] = []
    for spec in TARGET_SPECS:
        group = comparison_rows.loc[comparison_rows["target"].eq(spec.key)]
        current_mae = float(group["current_absolute_error"].mean())
        challenger_mae = float(group["challenger_absolute_error"].mean())
        target_comparison.append(
            {
                "target": spec.key,
                "rows": int(len(group)),
                "current_mae": round(current_mae, 4),
                "challenger_mae": round(challenger_mae, 4),
                "challenger_improvement": round(
                    (current_mae - challenger_mae) / current_mae, 4
                ),
            }
        )
    selection_wins = all(
        item["candidate_mae"][item["challenger_architecture"]]
        < item["candidate_mae"][item["current_architecture"]]
        for item in selections.values()
    )
    no_target_regression = all(
        item["challenger_improvement"] >= -0.01 for item in target_comparison
    )
    use_latent = bool(
        selection_wins
        and challenger_improvement >= 0.005
        and no_target_regression
        and confidence_interval[1] < 0
    )

    target_reports: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    for spec in TARGET_SPECS:
        comparison = comparison_rows.loc[comparison_rows["target"].eq(spec.key)].copy()
        comparison["prediction"] = (
            comparison["challenger_prediction"]
            if use_latent
            else comparison["current_prediction"]
        )
        comparison["baseline"] = comparison["baseline_prediction"]
        comparison["absolute_error"] = np.abs(
            comparison["actual"] - comparison["prediction"]
        )
        scored = comparison[
            IDENTITY_COLUMNS
            + [
                "target",
                "actual",
                "baseline",
                "prediction",
                "absolute_error",
                "current_prediction",
                "challenger_prediction",
            ]
        ].copy()
        scored_parts.append(scored)
        actual = scored["actual"].to_numpy()
        prediction = scored["prediction"].to_numpy()
        baseline = scored["baseline"].to_numpy()
        selected_name = (
            selections[spec.key]["challenger_architecture"]
            if use_latent
            else selections[spec.key]["current_architecture"]
        )
        target_reports.append(
            {
                "target": spec.key,
                "label": spec.label,
                "train_rows": int(
                    len(target_frames[spec.key][0].loc[target_frames[spec.key][0]["season"].lt(holdout_season)])
                ),
                "selection_rows": selections[spec.key]["selection_rows"],
                "holdout_season": holdout_season,
                "metrics": _metrics(
                    actual, prediction, baseline, spec.tolerance_yards
                ),
                "position_metrics": _position_metrics(scored),
                "model_selection": {
                    **selections[spec.key],
                    "selected_architecture": selected_name,
                    "selection_metric": "pooled expanding-window MAE",
                },
            }
        )

    # Refit the deployable encoder and downstream models through the holdout.
    deployment_encoder = PredictiveLatentEncoder(
        sequence_length=sequence_length,
        latent_dimensions=latent_dimensions,
        random_state=random_state,
    ).fit(
        sequence_table.loc[sequence_table["season"].le(holdout_season)],
        sequence_features,
        sequence_targets,
    )
    deployment_latent = deployment_encoder.transform_frame(sequence_table)
    deployment_models: dict[str, dict[str, Any]] = {}
    for spec in TARGET_SPECS:
        frame, raw_features = target_frames[spec.key]
        augmented, latent_features = _augment(frame, deployment_latent)
        train = augmented.loc[augmented["season"].le(holdout_season)]
        configuration = evaluation_models[spec.key]
        current_components_needed = [
            name for name, _ in ARCHITECTURES[configuration["current_architecture"]]
        ]
        current_models = _fit_components(
            train,
            raw_features,
            spec.target,
            random_state,
            current_components_needed,
        )
        _, latent_model_features, latent_components_needed = _selected_latent_configuration(
            configuration["challenger_architecture"], raw_features, latent_features, train
        )
        latent_models = _fit_components(
            train,
            latent_model_features,
            spec.target,
            random_state,
            latent_components_needed,
        )
        deployment_models[spec.key] = {
            "spec": spec,
            "raw_features": raw_features,
            "current_architecture": configuration["current_architecture"],
            "current_models": current_models,
            "challenger_architecture": configuration["challenger_architecture"],
            "base_latent_architecture": configuration["base_latent_architecture"],
            "latent_model_features": latent_model_features,
            "latent_models": latent_models,
            "use_latent": use_latent,
        }

    audit_columns = IDENTITY_COLUMNS + [spec.target for spec in TARGET_SPECS]
    audit_frame = stats[audit_columns].sort_values(
        ["season", "week", "player_id"]
    ).reset_index(drop=True)
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(audit_frame, index=False).values.tobytes()
    ).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 2,
        "generated_at_utc": generated_at,
        "evaluation_design": {
            "type": "expanding_selection_with_untouched_season_holdout",
            "source_seasons": sorted(int(value) for value in stats["season"].unique()),
            "architecture_selection_seasons": list(selection_values),
            "holdout_season": holdout_season,
            "leakage_controls": [
                "All raw and latent inputs contain only games before the prediction.",
                "The latent encoder is refit only on seasons earlier than each validation fold.",
                "Downstream estimators are refit only on seasons earlier than each scored season.",
                "The holdout is used once for the predeclared replacement gate.",
            ],
            "scope_note": (
                "Accuracy describes eligible player yardage projections. Sportsbook hit rate "
                "still requires authentic archived pregame lines."
            ),
        },
        "architecture": {
            "name": "predictive latent-state hybrid" if use_latent else "per-target chronological champion",
            "sequence_length": sequence_length,
            "latent_dimensions": latent_dimensions,
            "selected_by_target": {
                item["target"]: item["model_selection"]["selected_architecture"]
                for item in target_reports
            },
        },
        "latent_challenger_evidence": {
            "decision": "replace_current" if use_latent else "keep_current",
            "encoder_folds": encoder_folds,
            "current_weighted_mae": round(current_weighted_mae, 4),
            "challenger_weighted_mae": round(challenger_weighted_mae, 4),
            "challenger_improvement": round(challenger_improvement, 4),
            "paired_week_bootstrap_mae_delta_95": confidence_interval,
            "targets": target_comparison,
            "promotion_rule_passed": use_latent,
        },
        "data_audit": {
            "provider": "nflverse player stats plus play-by-play aggregation",
            "rows_loaded": int(len(stats)),
            "feature_target_sha256": source_hash,
        },
        "overall": _aggregate_metrics(target_reports),
        "targets": target_reports,
    }
    report["promotion_gate"] = _promotion_gate(target_reports)
    report["market_validation"] = {
        "status": "not_evaluated",
        "reason": "No authentic timestamped historical player-prop archive was supplied.",
        "required_for_static_promotion": True,
    }
    artifact = {
        "schema_version": 2,
        "trained_at_utc": generated_at,
        "holdout_season": holdout_season,
        "sequence_features": sequence_features,
        "sequence_targets": sequence_targets,
        "latent_encoder": deployment_encoder,
        "models": deployment_models,
    }
    scored_rows = pd.concat(scored_parts, ignore_index=True)
    return report, artifact, scored_rows


def predict_week_latent(
    stats: pd.DataFrame,
    artifact_bundle: dict[str, Any],
    *,
    season: int,
    week: int,
) -> pd.DataFrame:
    sequence_table, _, _ = build_sequence_table(stats)
    latent = artifact_bundle["latent_encoder"].transform_frame(sequence_table)
    parts: list[pd.DataFrame] = []
    for key, artifact in artifact_bundle["models"].items():
        spec: TargetSpec = artifact["spec"]
        frame, _ = build_features(stats, spec)
        augmented, _ = _augment(frame, latent)
        current = augmented.loc[
            augmented["season"].eq(season) & augmented["week"].eq(week)
        ].copy()
        if current.empty:
            continue
        current_components = _component_predictions(
            artifact["current_models"], current, artifact["raw_features"]
        )
        current_prediction = _architecture_prediction(
            artifact["current_architecture"], current, current_components
        )
        latent_prediction = _selected_latent_prediction(
            artifact["base_latent_architecture"],
            artifact["latent_models"],
            current,
            artifact["latent_model_features"],
        )
        current["prediction"] = (
            _combine_challenger(
                artifact["challenger_architecture"],
                current_prediction,
                latent_prediction,
            )
            if artifact["use_latent"]
            else current_prediction
        )
        current["target"] = key
        parts.append(current[IDENTITY_COLUMNS + ["target", "prediction"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
