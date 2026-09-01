"""Leakage-aware NFL player yardage prediction pipeline.

Lagged sequence summaries feed a compact set of regularized tabular learners.
Each yardage target selects its architecture on expanding pre-holdout seasons;
the final season is scored once and is never used for architecture selection.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


NFLVERSE_PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats.parquet"
)


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    target: str
    volume: str
    minimum_prior_volume: float
    tolerance_yards: float


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("passing", "Passing yards", "passing_yards", "attempts", 10.0, 30.0),
    TargetSpec("rushing", "Rushing yards", "rushing_yards", "carries", 2.0, 15.0),
    TargetSpec("receiving", "Receiving yards", "receiving_yards", "targets", 2.0, 15.0),
)

IDENTITY_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "opponent_team",
    "season",
    "week",
]

HISTORY_COLUMNS = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "attempts",
    "completions",
    "carries",
    "targets",
    "receptions",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "interceptions",
    "passing_epa",
    "rushing_epa",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
]


def load_weekly_stats(
    source: str | Path = NFLVERSE_PLAYER_STATS_URL,
    *,
    cache_path: Path | None = None,
    start_season: int = 2018,
    end_season: int | None = None,
) -> pd.DataFrame:
    """Load regular-season player stats from a local file or nflverse."""

    source_value = str(source)
    read_path: str | Path = source_value
    if cache_path is not None and cache_path.is_file():
        read_path = cache_path

    if str(read_path).lower().endswith((".parquet", ".pq")):
        frame = pd.read_parquet(read_path)
    else:
        frame = pd.read_csv(read_path, low_memory=False)

    if cache_path is not None and not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)

    required = set(IDENTITY_COLUMNS + HISTORY_COLUMNS + ["season_type"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"NFL stats source is missing required columns: {', '.join(missing)}")

    frame = frame.loc[frame["season_type"].eq("REG")].copy()
    frame = frame.loc[frame["season"].ge(start_season)]
    if end_season is not None:
        frame = frame.loc[frame["season"].le(end_season)]
    frame = frame.dropna(subset=["player_id", "season", "week"])
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    return frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True)


def _lagged_rolling(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).mean()


def _lagged_std(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=2).std()


def _opponent_history(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    allowed = (
        frame.groupby(["opponent_team", "season", "week"], as_index=False)[target]
        .sum()
        .rename(columns={"opponent_team": "defense", target: "yards_allowed"})
        .sort_values(["defense", "season", "week"])
    )
    grouped = allowed.groupby("defense", sort=False)["yards_allowed"]
    allowed["opponent_allowed_roll3"] = grouped.transform(lambda s: _lagged_rolling(s, 3))
    allowed["opponent_allowed_roll8"] = grouped.transform(lambda s: _lagged_rolling(s, 8))
    return allowed[["defense", "season", "week", "opponent_allowed_roll3", "opponent_allowed_roll8"]]


def build_features(stats: pd.DataFrame, spec: TargetSpec) -> tuple[pd.DataFrame, list[str]]:
    """Build strictly pregame features for one yardage target."""

    frame = stats.copy().sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    numeric_history = [column for column in HISTORY_COLUMNS if column in frame.columns]
    player_groups = frame.groupby("player_id", sort=False)

    frame["games_played_prior"] = player_groups.cumcount()
    for column in numeric_history:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame[column] = values
        grouped = frame.groupby("player_id", sort=False)[column]
        frame[f"{column}_roll3"] = grouped.transform(lambda s: _lagged_rolling(s, 3))
        frame[f"{column}_roll5"] = grouped.transform(lambda s: _lagged_rolling(s, 5))

    target_grouped = frame.groupby("player_id", sort=False)[spec.target]
    frame[f"{spec.target}_std5"] = target_grouped.transform(lambda s: _lagged_std(s, 5))
    frame[f"{spec.target}_career"] = target_grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=2).mean()
    )

    defense = _opponent_history(stats, spec.target)
    frame = frame.merge(
        defense,
        how="left",
        left_on=["opponent_team", "season", "week"],
        right_on=["defense", "season", "week"],
    ).drop(columns=["defense"])

    position = frame["position"].fillna("UNK").astype(str).str.upper()
    for value in ("QB", "RB", "FB", "WR", "TE"):
        frame[f"position_{value}"] = position.eq(value).astype(float)
    frame["position_OTHER"] = (~position.isin({"QB", "RB", "FB", "WR", "TE"})).astype(float)
    frame["season_progress"] = frame["week"].clip(upper=18) / 18.0
    frame["early_season"] = frame["week"].le(4).astype(float)

    volume_feature = f"{spec.volume}_roll3"
    eligible = frame["games_played_prior"].ge(3) & frame[volume_feature].ge(spec.minimum_prior_volume)
    frame = frame.loc[eligible].copy()

    features = [
        "season_progress",
        "early_season",
        "games_played_prior",
        "opponent_allowed_roll3",
        "opponent_allowed_roll8",
        f"{spec.target}_roll3",
        f"{spec.target}_roll5",
        f"{spec.target}_std5",
        f"{spec.target}_career",
        f"{spec.volume}_roll3",
        f"{spec.volume}_roll5",
        "position_QB",
        "position_RB",
        "position_FB",
        "position_WR",
        "position_TE",
        "position_OTHER",
    ]
    # Target keys are report identities, not feature-routing controls.  Infer
    # the football role from the actual outcome column so yardage and touchdown
    # variants use the same strictly lagged context without special-casing every
    # report key.
    role = spec.target.split("_", 1)[0]
    context_candidates = {
        "passing": ["completions", "passing_tds", "interceptions", "passing_epa"],
        "rushing": ["rushing_tds", "rushing_epa"],
        "receiving": ["receptions", "receiving_tds", "receiving_epa", "target_share", "air_yards_share", "wopr"],
    }[role]
    for column in context_candidates:
        for window in (3, 5):
            feature = f"{column}_roll{window}"
            if feature in frame.columns and feature not in features:
                features.append(feature)

    frame["baseline_prediction"] = frame[f"{spec.target}_roll5"]
    return frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True), features


ARCHITECTURES: dict[str, tuple[tuple[str, float], ...]] = {
    "rolling_baseline": (),
    "ridge": (("ridge", 1.0),),
    "hist_gradient_boosting": (("hist_gradient_boosting", 1.0),),
    "extra_trees": (("extra_trees", 1.0),),
    "xgboost": (("xgboost", 1.0),),
    "catboost": (("catboost", 1.0),),
    "ridge_extra_trees_blend": (("ridge", 0.5), ("extra_trees", 0.5)),
    "xgboost_catboost_blend": (("xgboost", 0.5), ("catboost", 0.5)),
    "hist_catboost_blend": (("hist_gradient_boosting", 0.5), ("catboost", 0.5)),
}


def _candidate_estimators(random_state: int) -> dict[str, Any]:
    """Return fixed, small-tabular candidates for chronological selection.

    Hyperparameters are intentionally regularized and shared across targets.
    The model family, not a target-specific holdout tweak, is selected using
    expanding pre-holdout folds.
    """

    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=12.0)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=0.055,
                        max_iter=220,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=2.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=8,
                        max_features=0.8,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgboost": XGBRegressor(
            n_estimators=350,
            learning_rate=0.025,
            max_depth=3,
            min_child_weight=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            reg_alpha=0.5,
            objective="reg:absoluteerror",
            n_jobs=-1,
            random_state=random_state,
        ),
        "catboost": CatBoostRegressor(
            iterations=350,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=8.0,
            loss_function="MAE",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            thread_count=-1,
        ),
    }


def _clean_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].replace([np.inf, -np.inf], np.nan)


def _fit_components(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
    component_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    candidates = _candidate_estimators(random_state)
    selected = set(candidates if component_names is None else component_names)
    x_train = _clean_matrix(train, features)
    y_train = train[target].astype(float)
    fitted: dict[str, Any] = {}
    for name, model in candidates.items():
        if name in selected:
            model.fit(x_train, y_train)
            fitted[name] = model
    return fitted


def _component_predictions(
    models: dict[str, Any], frame: pd.DataFrame, features: list[str]
) -> dict[str, np.ndarray]:
    x_value = _clean_matrix(frame, features)
    return {
        name: np.maximum(0.0, np.asarray(model.predict(x_value), dtype=float))
        for name, model in models.items()
    }


def _architecture_prediction(
    architecture: str,
    frame: pd.DataFrame,
    component_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    if architecture == "rolling_baseline":
        return np.maximum(0.0, frame["baseline_prediction"].astype(float).to_numpy())
    weighted = ARCHITECTURES[architecture]
    prediction = np.zeros(len(frame), dtype=float)
    for component, weight in weighted:
        prediction += weight * component_predictions[component]
    return np.maximum(0.0, prediction)


def _position_metrics(scored: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for position, group in scored.groupby("position", dropna=False):
        output.append(
            {
                "position": str(position or "UNK"),
                "rows": int(len(group)),
                "mae": round(float(mean_absolute_error(group["actual"], group["prediction"])), 4),
                "zero_actual_rate": round(float(group["actual"].eq(0).mean()), 4),
            }
        )
    return output


def _metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray, tolerance: float) -> dict[str, Any]:
    errors = np.abs(actual - predicted)
    baseline_mae = float(mean_absolute_error(actual, baseline))
    model_mae = float(mean_absolute_error(actual, predicted))
    direction_mask = np.abs(actual - baseline) >= 1.0
    direction_accuracy = (
        float(np.mean(np.sign(predicted[direction_mask] - baseline[direction_mask]) == np.sign(actual[direction_mask] - baseline[direction_mask])))
        if direction_mask.any()
        else None
    )
    return {
        "rows": int(len(actual)),
        "mae": round(model_mae, 4),
        "rmse": round(float(math.sqrt(mean_squared_error(actual, predicted))), 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "median_absolute_error": round(float(np.median(errors)), 4),
        "within_tolerance_accuracy": round(float(np.mean(errors <= tolerance)), 4),
        "tolerance_yards": tolerance,
        "residual_direction_accuracy": round(direction_accuracy, 4) if direction_accuracy is not None else None,
        "baseline_mae": round(baseline_mae, 4),
        "mae_improvement_vs_rolling_baseline": round((baseline_mae - model_mae) / baseline_mae, 4) if baseline_mae else None,
    }


def train_target(
    stats: pd.DataFrame,
    spec: TargetSpec,
    *,
    holdout_season: int,
    meta_seasons: Iterable[int],
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Select one target architecture and score a never-seen holdout season."""

    frame, features = build_features(stats, spec)
    holdout = frame.loc[frame["season"].eq(holdout_season)].copy()
    if holdout.empty:
        raise ValueError(f"No eligible {spec.key} rows found for holdout season {holdout_season}.")

    validation_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for season in sorted(set(int(value) for value in meta_seasons)):
        fold_train = frame.loc[frame["season"].lt(season)]
        validation_fold = frame.loc[frame["season"].eq(season)].copy()
        if len(fold_train) < 100 or validation_fold.empty:
            continue
        models = _fit_components(fold_train, features, spec.target, random_state + season)
        component_values = _component_predictions(models, validation_fold, features)
        part = pd.DataFrame(index=validation_fold.index)
        part["actual"] = validation_fold[spec.target].astype(float)
        for architecture in ARCHITECTURES:
            part[architecture] = _architecture_prediction(
                architecture, validation_fold, component_values
            )
        validation_parts.append(part)
        fold_metrics.append(
            {
                "season": season,
                "train_rows": int(len(fold_train)),
                "validation_rows": int(len(validation_fold)),
                "candidate_mae": {
                    architecture: round(
                        float(mean_absolute_error(part["actual"], part[architecture])), 4
                    )
                    for architecture in ARCHITECTURES
                },
            }
        )

    if not validation_parts:
        raise ValueError(f"No chronological selection folds were available for {spec.key}.")
    validation = pd.concat(validation_parts).sort_index()
    candidate_mae = {
        architecture: float(mean_absolute_error(validation["actual"], validation[architecture]))
        for architecture in ARCHITECTURES
    }
    selected_architecture = min(candidate_mae, key=candidate_mae.get)
    selected_components = [name for name, _ in ARCHITECTURES[selected_architecture]]

    final_train = frame.loc[frame["season"].lt(holdout_season)]
    final_models = _fit_components(
        final_train,
        features,
        spec.target,
        random_state,
        selected_components,
    )
    component_values = _component_predictions(final_models, holdout, features)
    predictions = _architecture_prediction(selected_architecture, holdout, component_values)
    actual = holdout[spec.target].astype(float).to_numpy()
    baseline = holdout["baseline_prediction"].astype(float).to_numpy()

    scored = holdout[IDENTITY_COLUMNS + [spec.target, "baseline_prediction"]].copy()
    scored = scored.rename(columns={spec.target: "actual", "baseline_prediction": "baseline"})
    scored["prediction"] = predictions
    scored["absolute_error"] = np.abs(actual - predictions)
    scored["target"] = spec.key

    report = {
        "target": spec.key,
        "label": spec.label,
        "train_rows": int(len(final_train)),
        "selection_rows": int(len(validation)),
        "holdout_season": holdout_season,
        "metrics": _metrics(actual, predictions, baseline, spec.tolerance_yards),
        "position_metrics": _position_metrics(scored),
        "model_selection": {
            "selected_architecture": selected_architecture,
            "selection_metric": "pooled expanding-window MAE",
            "candidate_mae": {
                name: round(value, 4)
                for name, value in sorted(candidate_mae.items(), key=lambda item: item[1])
            },
            "folds": fold_metrics,
        },
    }
    # The holdout result above is immutable evidence.  Only after scoring it do
    # we refit the deployable base estimators through the validated season.
    deployment_models = _fit_components(
        frame.loc[frame["season"].le(holdout_season)],
        features,
        spec.target,
        random_state,
        selected_components,
    )
    artifact = {
        "spec": spec,
        "features": features,
        "architecture": selected_architecture,
        "component_models": deployment_models,
        "trained_through_season": holdout_season,
    }
    return report, artifact, scored


def _aggregate_metrics(target_reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(item["metrics"]["rows"] for item in target_reports)
    if not total:
        return {"rows": 0}

    def weighted(name: str) -> float:
        return sum(item["metrics"][name] * item["metrics"]["rows"] for item in target_reports) / total

    return {
        "rows": total,
        "weighted_mae": round(weighted("mae"), 4),
        "weighted_baseline_mae": round(weighted("baseline_mae"), 4),
        "weighted_within_tolerance_accuracy": round(weighted("within_tolerance_accuracy"), 4),
        "weighted_mae_improvement_vs_rolling_baseline": round(
            1.0 - weighted("mae") / weighted("baseline_mae"), 4
        ),
    }


def _promotion_gate(target_reports: list[dict[str, Any]]) -> dict[str, Any]:
    criteria = {
        "minimum_rows_per_target": 500,
        "minimum_r2_per_target": 0.15,
        "minimum_mae_improvement_vs_baseline": 0.02,
        "minimum_residual_direction_accuracy": 0.53,
    }
    checks: list[dict[str, Any]] = []
    for target in target_reports:
        metrics = target["metrics"]
        passed = bool(
            metrics["rows"] >= criteria["minimum_rows_per_target"]
            and metrics["r2"] >= criteria["minimum_r2_per_target"]
            and metrics["mae_improvement_vs_rolling_baseline"]
            >= criteria["minimum_mae_improvement_vs_baseline"]
            and metrics["residual_direction_accuracy"]
            >= criteria["minimum_residual_direction_accuracy"]
        )
        checks.append({"target": target["target"], "passed": passed})
    projection_passed = all(item["passed"] for item in checks)
    return {
        # Static promotion additionally requires authentic historical market
        # validation, which the training stats alone cannot provide.
        "status": "failed",
        "projection_status": "passed" if projection_passed else "failed",
        "reason": "Authentic historical player-prop hit rate has not been evaluated.",
        "criteria": criteria,
        "target_checks": checks,
    }


def predict_week(
    stats: pd.DataFrame,
    artifact_bundle: dict[str, Any],
    *,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Score pre-built rows for a week using the post-validation artifact.

    The input should include historical rows plus one placeholder row per
    scheduled player. Placeholder outcome fields may be zero because all
    engineered inputs are shifted and therefore ignore the current row.
    """

    parts: list[pd.DataFrame] = []
    for key, artifact in artifact_bundle["models"].items():
        spec: TargetSpec = artifact["spec"]
        frame, _ = build_features(stats, spec)
        current = frame.loc[frame["season"].eq(season) & frame["week"].eq(week)].copy()
        if current.empty:
            continue
        component_values = _component_predictions(
            artifact["component_models"], current, artifact["features"]
        )
        current["prediction"] = _architecture_prediction(
            artifact["architecture"], current, component_values
        )
        current["target"] = key
        parts.append(current[IDENTITY_COLUMNS + ["target", "prediction"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def train_and_backtest(
    stats: pd.DataFrame,
    *,
    holdout_season: int,
    meta_seasons: Iterable[int] | None = None,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Train all yardage models and return report, artifact bundle and rows."""

    meta_values = tuple(meta_seasons or range(holdout_season - 4, holdout_season))
    reports: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    scored_parts: list[pd.DataFrame] = []
    for spec in TARGET_SPECS:
        report, artifact, scored = train_target(
            stats,
            spec,
            holdout_season=holdout_season,
            meta_seasons=meta_values,
            random_state=random_state,
        )
        reports.append(report)
        artifacts[spec.key] = artifact
        scored_parts.append(scored)

    source_seasons = sorted(int(value) for value in stats["season"].unique())
    audit_columns = IDENTITY_COLUMNS + [spec.target for spec in TARGET_SPECS]
    audit_frame = stats[audit_columns].sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(audit_frame, index=False).values.tobytes()
    ).hexdigest()
    report_bundle = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_design": {
            "type": "chronological_season_holdout",
            "source_seasons": source_seasons,
            "architecture_selection_seasons": list(meta_values),
            "holdout_season": holdout_season,
            "leakage_controls": [
                "All player and opponent features are shifted by at least one game.",
                "Base estimators for a scored season are trained only on earlier seasons.",
                "Architectures are ranked only on expanding chronological pre-holdout folds.",
                "The final holdout season is never used for architecture or threshold selection.",
            ],
            "scope_note": (
                "Accuracy describes player yardage projections for eligible active-player rows. "
                "It is not sportsbook win rate because archived market lines are not present."
            ),
        },
        "architecture": {
            "name": "per-target chronological champion",
            "selection_metric": "pooled expanding-window MAE",
            "candidate_architectures": list(ARCHITECTURES),
            "selected_by_target": {
                item["target"]: item["model_selection"]["selected_architecture"]
                for item in reports
            },
            "design_reason": (
                "The sample is medium-sized tabular weekly data, so regularized linear and "
                "tree-boosting families are tested directly instead of assuming a deep network."
            ),
        },
        "data_audit": {
            "provider": "nflverse",
            "source_url": NFLVERSE_PLAYER_STATS_URL,
            "rows_loaded": int(len(stats)),
            "feature_target_sha256": source_hash,
        },
        "overall": _aggregate_metrics(reports),
        "targets": reports,
    }
    report_bundle["promotion_gate"] = _promotion_gate(reports)
    report_bundle["market_validation"] = {
        "status": "not_evaluated",
        "reason": "No authentic timestamped historical player-prop archive was supplied.",
        "required_for_static_promotion": True,
    }
    artifact_bundle = {
        "schema_version": 1,
        "trained_at_utc": report_bundle["generated_at_utc"],
        "holdout_season": holdout_season,
        "models": artifacts,
    }
    scored_rows = pd.concat(scored_parts, ignore_index=True)
    return report_bundle, artifact_bundle, scored_rows


def write_training_outputs(
    report: dict[str, Any],
    artifact: dict[str, Any],
    scored_rows: pd.DataFrame,
    *,
    report_path: Path,
    artifact_path: Path,
    rows_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    joblib.dump(artifact, artifact_path, compress=3)
    scored_rows.to_csv(rows_path, index=False)
