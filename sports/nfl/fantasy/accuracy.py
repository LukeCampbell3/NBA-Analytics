"""Forward-only fantasy point model selection and confidence calibration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


POSITIONS = ("QB", "RB", "WR", "TE")
ROLLING_WINDOWS = (3, 5, 8, 17)
STD_WINDOWS = (3, 8)
BASE_FEATURES = (
    "fantasy_points_ppr_model",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "target_share",
    "air_yards_share",
    "wopr",
)
CATEGORICAL_FEATURES = ("position", "recent_team", "opponent_team")
CANDIDATE_DEPTHS = (4, 5, 6)


def _feature_names() -> list[str]:
    names = [
        f"{column}_{operation}{window}"
        for column in BASE_FEATURES
        for operation, windows in (("mean", ROLLING_WINDOWS), ("std", STD_WINDOWS))
        for window in windows
    ]
    return [*names, "history_games", "week", *CATEGORICAL_FEATURES]


def build_accuracy_features(history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create leakage-safe features; every rolling value is shifted one game."""

    frame = history.sort_values(["player_id", "season", "week"]).copy()
    for column in BASE_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    grouped = frame.groupby("player_id", sort=False)
    parts: list[pd.Series] = []
    for column in BASE_FEATURES:
        for window in ROLLING_WINDOWS:
            parts.append(
                grouped[column]
                .transform(lambda values, size=window: values.shift(1).rolling(size, min_periods=1).mean())
                .rename(f"{column}_mean{window}")
            )
        for window in STD_WINDOWS:
            parts.append(
                grouped[column]
                .transform(lambda values, size=window: values.shift(1).rolling(size, min_periods=2).std())
                .rename(f"{column}_std{window}")
            )
    engineered = pd.concat(parts, axis=1)
    engineered["history_games"] = grouped.cumcount()
    engineered["week"] = pd.to_numeric(frame["week"], errors="coerce").fillna(0).astype(int)
    for column in CATEGORICAL_FEATURES:
        engineered[column] = frame[column].fillna("UNK").astype(str)
    output = pd.concat(
        [frame[["player_id", "season", "fantasy_points_ppr_model"]], engineered],
        axis=1,
    )
    return output, _feature_names()


def _model(depth: int, random_seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        depth=depth,
        learning_rate=0.05 if depth <= 5 else 0.04,
        l2_leaf_reg=15.0,
        iterations=500,
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    error = rows["actual"] - rows["prediction"]
    correlations = []
    for _, week in rows.groupby("week"):
        if len(week) >= 20:
            correlations.append(week["actual"].corr(week["prediction"], method="spearman"))
    return {
        "rows": int(len(rows)),
        "mae_fantasy_points": round(float(error.abs().mean()), 4),
        "rmse_fantasy_points": round(float(np.sqrt(np.mean(np.square(error)))), 4),
        "mean_bias": round(float((-error).mean()), 4),
        "mean_weekly_spearman": round(float(np.nanmean(correlations)), 4),
    }


def _baseline_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[["player_id", "season", "week", "position", "fantasy_points_ppr_model"]].copy()
    grouped = frame.groupby("player_id", sort=False)["fantasy_points_ppr_model"]
    output["last_game"] = grouped.shift(1)
    output["recency"] = grouped.transform(
        lambda values: values.shift(1).ewm(alpha=0.20, adjust=False).mean()
    )
    return output


def train_accuracy_model(
    history: pd.DataFrame,
    *,
    holdout_season: int = 2025,
    minimum_prior_games: int = 4,
    random_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune on holdout-1, evaluate once on holdout, and return deployable models."""

    validation_season = holdout_season - 1
    selection_end = validation_season - 1
    features, feature_names = build_accuracy_features(history)
    eligible = features.loc[features["history_games"].ge(minimum_prior_games)].copy()
    baselines = _baseline_predictions(history)
    eligible = eligible.merge(
        baselines[["player_id", "season", "week", "last_game", "recency"]],
        on=["player_id", "season", "week"],
        how="left",
        validate="one_to_one",
    )
    target = "fantasy_points_ppr_model"
    categorical = list(CATEGORICAL_FEATURES)
    candidate_rows: list[dict[str, Any]] = []
    validation_parts: list[pd.DataFrame] = []
    selected_depths: dict[str, int] = {}
    final_models: dict[str, CatBoostRegressor] = {}

    # The global family shares signal across positions. It competes honestly
    # against the position specialists and their blends on the validation year.
    global_selection = eligible["season"].le(selection_end)
    global_validation = eligible["season"].eq(validation_season)
    global_predictions: dict[int, np.ndarray] = {}
    for depth in CANDIDATE_DEPTHS:
        candidate = _model(depth, random_seed)
        candidate.fit(
            eligible.loc[global_selection, feature_names],
            eligible.loc[global_selection, target],
            cat_features=categorical,
        )
        prediction = candidate.predict(eligible.loc[global_validation, feature_names])
        global_predictions[depth] = prediction
        candidate_rows.append(
            {
                "position": "ALL",
                "architecture": f"catboost_mae_depth_{depth}",
                "validation_season": validation_season,
                "rows": int(global_validation.sum()),
                "mae": round(float(np.mean(np.abs(eligible.loc[global_validation, target] - prediction))), 4),
            }
        )
    global_champion = min(
        (row for row in candidate_rows if row["position"] == "ALL"),
        key=lambda row: (row["mae"], row["architecture"]),
    )
    global_depth = int(global_champion["architecture"].rsplit("_", 1)[-1])
    global_validation_rows = eligible.loc[
        global_validation, ["player_id", "season", "week", "position"]
    ].copy()
    global_validation_rows["global_prediction"] = global_predictions[global_depth]
    global_model = _model(global_depth, random_seed)
    global_final_train = eligible["season"].lt(holdout_season)
    global_model.fit(
        eligible.loc[global_final_train, feature_names],
        eligible.loc[global_final_train, target],
        cat_features=categorical,
    )

    for position in POSITIONS:
        position_rows = eligible.loc[eligible["position"].eq(position)]
        selection = position_rows["season"].le(selection_end)
        validation = position_rows["season"].eq(validation_season)
        if not selection.any() or not validation.any():
            continue
        depth_predictions: dict[int, np.ndarray] = {}
        for depth in CANDIDATE_DEPTHS:
            candidate = _model(depth, random_seed)
            candidate.fit(
                position_rows.loc[selection, feature_names],
                position_rows.loc[selection, target],
                cat_features=categorical,
            )
            prediction = candidate.predict(position_rows.loc[validation, feature_names])
            depth_predictions[depth] = prediction
            mae = float(np.mean(np.abs(position_rows.loc[validation, target] - prediction)))
            candidate_rows.append(
                {
                    "position": position,
                    "architecture": f"catboost_mae_depth_{depth}",
                    "validation_season": validation_season,
                    "rows": int(validation.sum()),
                    "mae": round(mae, 4),
                }
            )
        position_candidates = [row for row in candidate_rows if row["position"] == position]
        champion = min(position_candidates, key=lambda row: (row["mae"], row["architecture"]))
        depth = int(champion["architecture"].rsplit("_", 1)[-1])
        selected_depths[position] = depth
        validation_rows = position_rows.loc[
            validation, ["player_id", "season", "week", "position", target, "fantasy_points_ppr_model_std8"]
        ].copy()
        validation_rows = validation_rows.rename(columns={target: "actual"})
        validation_rows["prediction"] = depth_predictions[depth]
        validation_parts.append(validation_rows)

        final_train = position_rows["season"].lt(holdout_season)
        final_model = _model(depth, random_seed)
        final_model.fit(
            position_rows.loc[final_train, feature_names],
            position_rows.loc[final_train, target],
            cat_features=categorical,
        )
        final_models[position] = final_model

    calibration = pd.concat(validation_parts, ignore_index=True)
    calibration = calibration.merge(
        global_validation_rows,
        on=["player_id", "season", "week", "position"],
        how="inner",
        validate="one_to_one",
    )
    blend_candidates = []
    for global_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        prediction = (
            global_weight * calibration["global_prediction"]
            + (1.0 - global_weight) * calibration["prediction"]
        )
        blend_candidates.append(
            {
                "global_weight": global_weight,
                "position_weight": 1.0 - global_weight,
                "validation_mae": round(float((calibration["actual"] - prediction).abs().mean()), 4),
            }
        )
    selected_blend = min(blend_candidates, key=lambda row: (row["validation_mae"], row["global_weight"]))
    global_weight = float(selected_blend["global_weight"])
    calibration["prediction"] = (
        global_weight * calibration["global_prediction"]
        + (1.0 - global_weight) * calibration["prediction"]
    )
    offsets = calibration.groupby("position").apply(
        lambda rows: float(np.median(rows["actual"] - rows["prediction"])),
        include_groups=False,
    ).to_dict()
    position_scales = calibration.groupby("position")["fantasy_points_ppr_model_std8"].median().to_dict()
    calibration["scale"] = calibration.apply(
        lambda row: max(float(row["fantasy_points_ppr_model_std8"] or 0), float(position_scales[row["position"]]) * 0.5, 2.0),
        axis=1,
    )
    calibration["prediction"] += calibration["position"].map(offsets)
    calibration["normalized_error"] = (
        (calibration["actual"] - calibration["prediction"]).abs() / calibration["scale"]
    )
    conformal_quantiles = calibration.groupby("position")["normalized_error"].quantile(0.80).to_dict()

    seen_parts: list[pd.DataFrame] = []
    unseen_parts: list[pd.DataFrame] = []
    for position, model in final_models.items():
        position_rows = eligible.loc[eligible["position"].eq(position)]
        for split, mask, collector in (
            ("seen", position_rows["season"].lt(holdout_season), seen_parts),
            ("unseen", position_rows["season"].eq(holdout_season), unseen_parts),
        ):
            scored = position_rows.loc[
                mask,
                ["player_id", "season", "week", "position", target, "last_game", "recency", "fantasy_points_ppr_model_std8"],
            ].copy()
            if scored.empty:
                continue
            scored = scored.rename(columns={target: "actual"})
            position_prediction = model.predict(position_rows.loc[mask, feature_names])
            global_prediction = global_model.predict(position_rows.loc[mask, feature_names])
            scored["prediction"] = (
                global_weight * global_prediction
                + (1.0 - global_weight) * position_prediction
                + offsets[position]
            )
            scored["split"] = split
            collector.append(scored)
    seen = pd.concat(seen_parts, ignore_index=True)
    unseen = pd.concat(unseen_parts, ignore_index=True)
    unseen["scale"] = unseen.apply(
        lambda row: max(float(row["fantasy_points_ppr_model_std8"] or 0), float(position_scales[row["position"]]) * 0.5, 2.0),
        axis=1,
    )
    unseen["interval_half_width"] = unseen["scale"] * unseen["position"].map(conformal_quantiles)
    unseen["covered"] = (
        unseen["actual"].sub(unseen["prediction"]).abs() <= unseen["interval_half_width"]
    )
    cutoffs = unseen["interval_half_width"].quantile([1 / 3, 2 / 3]).to_list()
    unseen["confidence"] = np.select(
        [unseen["interval_half_width"].le(cutoffs[0]), unseen["interval_half_width"].le(cutoffs[1])],
        ["high", "medium"],
        default="low",
    )
    confidence = {}
    for label, rows in unseen.groupby("confidence"):
        confidence[str(label)] = {
            "rows": int(len(rows)),
            "mae": round(float((rows["actual"] - rows["prediction"]).abs().mean()), 4),
            "coverage": round(float(rows["covered"].mean()), 4),
            "mean_interval_half_width": round(float(rows["interval_half_width"].mean()), 4),
        }

    seen_metrics = _metrics(seen)
    unseen_metrics = _metrics(unseen)
    unseen_metrics["last_game_mae"] = round(float((unseen["actual"] - unseen["last_game"]).abs().mean()), 4)
    unseen_metrics["recency_baseline_mae"] = round(float((unseen["actual"] - unseen["recency"]).abs().mean()), 4)
    unseen_metrics["mae_improvement_vs_recency"] = round(
        (unseen_metrics["recency_baseline_mae"] - unseen_metrics["mae_fantasy_points"])
        / unseen_metrics["recency_baseline_mae"],
        4,
    )
    unseen_metrics["central_80_interval_coverage"] = round(float(unseen["covered"].mean()), 4)
    overfit_gap = (unseen_metrics["mae_fantasy_points"] - seen_metrics["mae_fantasy_points"]) / seen_metrics["mae_fantasy_points"]
    weekly = []
    for week, rows in unseen.groupby("week"):
        week_metrics = _metrics(rows)
        weekly.append({"week": int(week), **week_metrics})
    checks = {
        "minimum_unseen_rows": len(unseen) >= 2_000,
        "beats_recency_baseline": unseen_metrics["mae_fantasy_points"] < unseen_metrics["recency_baseline_mae"],
        "unseen_weekly_rank_correlation": unseen_metrics["mean_weekly_spearman"] >= 0.50,
        "central_80_interval_calibrated": 0.72 <= unseen_metrics["central_80_interval_coverage"] <= 0.88,
        "overfit_gap_below_10_percent": overfit_gap <= 0.10,
        "confidence_orders_error": confidence.get("high", {}).get("mae", float("inf")) < confidence.get("low", {}).get("mae", 0),
    }
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "design": (
            f"Architectures selected on {validation_season}; final models refit through {validation_season}; "
            f"all {holdout_season} weeks remain chronologically unseen."
        ),
        "selection_seasons": [int(eligible["season"].min()), selection_end],
        "validation_season": validation_season,
        "holdout_season": holdout_season,
        "selected_architecture_by_position": {
            position: f"catboost_mae_depth_{depth}" for position, depth in selected_depths.items()
        },
        "selected_global_architecture": f"catboost_mae_depth_{global_depth}",
        "blend_selection": {
            "candidates": blend_candidates,
            "selected": selected_blend,
        },
        "candidate_validation": sorted(candidate_rows, key=lambda row: (row["position"], row["mae"])),
        "seen_weeks": seen_metrics,
        "unseen_weeks": unseen_metrics,
        "overfit_gap": round(float(overfit_gap), 4),
        "confidence_calibration": {
            "method": "position-specific normalized split conformal",
            "target_coverage": 0.80,
            "tiers": confidence,
        },
        "weekly_unseen": weekly,
        # Compatibility aliases for existing frontend consumers.
        "rows": int(len(unseen)),
        "metrics": unseen_metrics,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
        "limitations": [
            "Seen-week error is diagnostic only; model selection uses forward validation and promotion uses unseen weeks.",
            "The holdout validates players who logged a game, not preseason availability or injury news.",
            "Rookies and players with fewer than four prior games retain conservative simulation priors.",
        ],
    }
    bundle = {
        "feature_names": feature_names,
        "categorical_features": categorical,
        "models": final_models,
        "global_model": global_model,
        "global_depth": global_depth,
        "global_weight": global_weight,
        "selected_depths": selected_depths,
        "median_offsets": offsets,
        "position_scales": position_scales,
        "conformal_quantiles": conformal_quantiles,
        "minimum_prior_games": minimum_prior_games,
        "trained_through_season": validation_season,
    }
    return report, bundle


def upcoming_accuracy_forecasts(
    history: pd.DataFrame,
    roster: pd.DataFrame,
    schedule: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    season: int,
) -> dict[tuple[str, int, str], dict[str, float]]:
    """Score scheduled games from fixed pre-season history features."""

    feature_names = list(bundle["feature_names"])
    rows: list[dict[str, Any]] = []
    for player in roster.itertuples(index=False):
        player_id = str(player.player_id)
        logs = history.loc[history["player_id"].astype(str).eq(player_id)].sort_values(["season", "week"])
        if len(logs) < int(bundle["minimum_prior_games"]):
            continue
        base: dict[str, Any] = {
            "player_id": player_id,
            "position": str(player.position),
            "recent_team": str(player.recent_team),
            "history_games": len(logs),
        }
        for column in BASE_FEATURES:
            values = pd.to_numeric(logs[column], errors="coerce").fillna(0.0)
            for window in ROLLING_WINDOWS:
                base[f"{column}_mean{window}"] = float(values.tail(window).mean())
            for window in STD_WINDOWS:
                base[f"{column}_std{window}"] = float(values.tail(window).std()) if len(values.tail(window)) >= 2 else np.nan
        games = schedule.loc[
            schedule["season"].eq(season)
            & schedule.get("game_type", pd.Series("REG", index=schedule.index)).eq("REG")
            & (schedule["home_team"].eq(player.recent_team) | schedule["away_team"].eq(player.recent_team))
        ]
        for game in games.itertuples(index=False):
            row = dict(base)
            row["week"] = int(game.week)
            row["opponent_team"] = str(game.away_team if game.home_team == player.recent_team else game.home_team)
            rows.append(row)
    if not rows:
        return {}
    scoring = pd.DataFrame(rows)
    output: dict[tuple[str, int, str], dict[str, float]] = {}
    for position, model in bundle["models"].items():
        mask = scoring["position"].eq(position)
        if not mask.any():
            continue
        position_rows = scoring.loc[mask].copy()
        position_predictions = model.predict(position_rows[feature_names])
        global_predictions = bundle["global_model"].predict(position_rows[feature_names])
        global_weight = float(bundle["global_weight"])
        predictions = (
            global_weight * global_predictions
            + (1.0 - global_weight) * position_predictions
            + float(bundle["median_offsets"][position])
        )
        scale = position_rows["fantasy_points_ppr_model_std8"].fillna(bundle["position_scales"][position]).clip(lower=2.0)
        half_width = scale * float(bundle["conformal_quantiles"][position])
        for row, prediction, width in zip(position_rows.itertuples(index=False), predictions, half_width):
            output[(str(row.player_id), int(row.week), str(row.opponent_team))] = {
                "mean": max(0.0, float(prediction)),
                "interval_half_width": float(width),
            }
    return output
