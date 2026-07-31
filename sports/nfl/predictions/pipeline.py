"""Leakage-aware NFL player yardage prediction pipeline.

The implementation adapts the repository's stacked NBA approach to the much
smaller weekly NFL sample: lagged sequence summaries feed a gradient-boosted
tree and a regularized linear model, then a Ridge meta-learner combines those
estimates with the rolling baseline.  Every feature is available before the
game being predicted and every reported score is from a later season than the
rows used to fit the underlying estimators.
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    ]
    context_candidates = {
        "passing": ["completions", "passing_tds", "interceptions", "passing_epa"],
        "rushing": ["rushing_tds", "rushing_epa"],
        "receiving": ["receptions", "receiving_tds", "receiving_epa", "target_share", "air_yards_share", "wopr"],
    }[spec.key]
    for column in context_candidates:
        for window in (3, 5):
            feature = f"{column}_roll{window}"
            if feature in frame.columns:
                features.append(feature)

    frame["baseline_prediction"] = frame[f"{spec.target}_roll5"]
    return frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True), features


def _base_estimators(random_state: int) -> tuple[Pipeline, GradientBoostingRegressor]:
    linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=12.0)),
        ]
    )
    boosted = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.035,
        max_depth=2,
        min_samples_leaf=18,
        loss="huber",
        random_state=random_state,
    )
    return linear, boosted


def _clean_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].replace([np.inf, -np.inf], np.nan)


def _fit_base(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
) -> tuple[Pipeline, Pipeline]:
    linear, boosted_model = _base_estimators(random_state)
    boosted = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("gbm", boosted_model)]
    )
    x_train = _clean_matrix(train, features)
    y_train = train[target].astype(float)
    linear.fit(x_train, y_train)
    boosted.fit(x_train, y_train)
    return linear, boosted


def _base_predictions(models: tuple[Pipeline, Pipeline], frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    x_value = _clean_matrix(frame, features)
    linear, boosted = models
    return np.column_stack(
        [
            frame["baseline_prediction"].astype(float).to_numpy(),
            linear.predict(x_value),
            boosted.predict(x_value),
        ]
    )


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
    """Train one stacked target model and score a never-seen holdout season."""

    frame, features = build_features(stats, spec)
    holdout = frame.loc[frame["season"].eq(holdout_season)].copy()
    if holdout.empty:
        raise ValueError(f"No eligible {spec.key} rows found for holdout season {holdout_season}.")

    meta_parts: list[pd.DataFrame] = []
    for season in sorted(set(int(value) for value in meta_seasons)):
        base_train = frame.loc[frame["season"].lt(season)]
        meta_fold = frame.loc[frame["season"].eq(season)].copy()
        if len(base_train) < 100 or meta_fold.empty:
            continue
        models = _fit_base(base_train, features, spec.target, random_state + season)
        stack_values = _base_predictions(models, meta_fold, features)
        part = pd.DataFrame(stack_values, columns=["baseline", "ridge", "gbm"], index=meta_fold.index)
        part["actual"] = meta_fold[spec.target].astype(float)
        meta_parts.append(part)

    if not meta_parts:
        raise ValueError(f"No chronological meta-training folds were available for {spec.key}.")
    meta_train = pd.concat(meta_parts).sort_index()
    meta_model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=4.0, positive=True))])
    meta_model.fit(meta_train[["baseline", "ridge", "gbm"]].to_numpy(), meta_train["actual"])

    final_train = frame.loc[frame["season"].lt(holdout_season)]
    base_models = _fit_base(final_train, features, spec.target, random_state)
    holdout_stack = _base_predictions(base_models, holdout, features)
    predictions = np.maximum(0.0, meta_model.predict(holdout_stack))
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
        "meta_rows": int(len(meta_train)),
        "holdout_season": holdout_season,
        "metrics": _metrics(actual, predictions, baseline, spec.tolerance_yards),
    }
    # The holdout result above is immutable evidence.  Only after scoring it do
    # we refit the deployable base estimators through the validated season.
    deployment_models = _fit_base(
        frame.loc[frame["season"].le(holdout_season)],
        features,
        spec.target,
        random_state,
    )
    artifact = {
        "spec": spec,
        "features": features,
        "base_models": deployment_models,
        "meta_model": meta_model,
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
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
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
        stack = _base_predictions(artifact["base_models"], current, artifact["features"])
        current["prediction"] = np.maximum(0.0, artifact["meta_model"].predict(stack))
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

    meta_values = tuple(meta_seasons or (holdout_season - 2, holdout_season - 1))
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
            "meta_seasons": list(meta_values),
            "holdout_season": holdout_season,
            "leakage_controls": [
                "All player and opponent features are shifted by at least one game.",
                "Base estimators for a scored season are trained only on earlier seasons.",
                "The stacking model is fit on chronological pre-holdout folds.",
                "The final holdout season is never used for fitting or threshold selection.",
            ],
            "scope_note": (
                "Accuracy describes player yardage projections for eligible active-player rows. "
                "It is not sportsbook win rate because archived market lines are not present."
            ),
        },
        "architecture": {
            "name": "lagged sequence stack",
            "base_models": ["five-game rolling baseline", "Ridge regression", "GradientBoostingRegressor"],
            "meta_model": "positive Ridge regression",
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
