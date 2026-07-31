"""Leakage-aware NFL player-prop side probability models.

The yardage stack remains responsible for point projections.  This module adds
the smaller market-facing layer: it estimates the probability that a player
finishes over a single posted pregame line, then abstains unless the estimated
side probability is high enough.  Target markets are promoted independently so
a weak rushing or receiving market cannot hide behind a strong passing market.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .latent import PredictiveLatentEncoder, build_sequence_table
from .pipeline import TARGET_SPECS, build_features


TARGET_SCALES = {"passing": 60.0, "rushing": 20.0, "receiving": 20.0}
MARKET_SIGNAL_COLUMNS = [
    "line_scaled",
    "current_edge_scaled",
    "challenger_edge_scaled",
    "baseline_edge_scaled",
    "model_disagreement_scaled",
]


def _model_factory(name: str, random_state: int) -> Any:
    if name.startswith("regularized_logistic"):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(C=0.05, max_iter=2_000, random_state=random_state),
                ),
            ]
        )
    if name.startswith("catboost"):
        return CatBoostClassifier(
            iterations=250,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=12.0,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            thread_count=-1,
        )
    raise KeyError(f"Unknown NFL market architecture: {name}")


def _implied_probability(price: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(price, errors="coerce")
    return pd.Series(
        np.where(
            numeric.gt(0),
            100.0 / (numeric + 100.0),
            numeric.abs() / (numeric.abs() + 100.0),
        ),
        index=price.index,
    )


def _american_profit(price: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(price, errors="coerce")
    return pd.Series(
        np.where(numeric.gt(0), numeric / 100.0, 100.0 / numeric.abs()),
        index=price.index,
    )


def _wilson_interval(wins: int, losses: int, z: float = 1.96) -> list[float] | None:
    total = wins + losses
    if total == 0:
        return None
    rate = wins / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def score_probabilities(
    frame: pd.DataFrame,
    over_probability: np.ndarray,
    *,
    minimum_side_probability: float = 0.56,
    minimum_no_vig_advantage: float = 0.025,
) -> pd.DataFrame:
    """Choose one side of one posted line and apply a price-aware abstention gate."""

    scored = frame.copy()
    scored["over_probability"] = np.asarray(over_probability, dtype=float)
    scored["side"] = np.where(scored["over_probability"].ge(0.5), "over", "under")
    scored["estimated_side_probability"] = np.maximum(
        scored["over_probability"], 1.0 - scored["over_probability"]
    )
    over_implied = _implied_probability(scored["over_price"])
    under_implied = _implied_probability(scored["under_price"])
    probability_sum = over_implied + under_implied
    no_vig_over = over_implied / probability_sum
    scored["no_vig_side_probability"] = np.where(
        scored["side"].eq("over"), no_vig_over, 1.0 - no_vig_over
    )
    scored["probability_advantage"] = (
        scored["estimated_side_probability"] - scored["no_vig_side_probability"]
    )
    scored["eligible"] = (
        scored["estimated_side_probability"].ge(float(minimum_side_probability))
        & scored["probability_advantage"].ge(float(minimum_no_vig_advantage))
        & scored["over_price"].notna()
        & scored["under_price"].notna()
    )
    over = scored["side"].eq("over")
    push = scored["actual"].eq(scored["line"])
    win = np.where(over, scored["actual"].gt(scored["line"]), scored["actual"].lt(scored["line"]))
    scored["result"] = np.where(push, "push", np.where(win, "win", "loss"))
    scored["selected_price"] = np.where(over, scored["over_price"], scored["under_price"])
    win_profit = _american_profit(scored["selected_price"])
    scored["profit_units"] = np.where(
        scored["result"].eq("push"),
        0.0,
        np.where(scored["result"].eq("win"), win_profit, -1.0),
    )
    return scored


def summarize_market_rows(rows: pd.DataFrame) -> dict[str, Any]:
    eligible = rows.loc[rows["eligible"]].copy() if "eligible" in rows else rows.copy()
    wins = int(eligible["result"].eq("win").sum())
    losses = int(eligible["result"].eq("loss").sum())
    pushes = int(eligible["result"].eq("push").sum())
    decisions = wins + losses
    return {
        "bets": int(len(eligible)),
        "graded_decisions": decisions,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(wins / decisions, 4) if decisions else None,
        "hit_rate_wilson_95": _wilson_interval(wins, losses),
        "roi": round(float(eligible["profit_units"].mean()), 4) if len(eligible) else None,
        "profit_units": round(float(eligible["profit_units"].sum()), 4) if len(eligible) else None,
        "distinct_weeks": int(eligible[["season", "week"]].drop_duplicates().shape[0]),
        "over_bets": int(eligible["side"].eq("over").sum()),
        "under_bets": int(eligible["side"].eq("under").sum()),
    }


def prune_weekly_pool(pool: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Keep the highest-confidence fixed-size board within each season/week."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if pool.empty:
        return pool.copy()
    ranked = pool.sort_values(
        [
            "season",
            "week",
            "estimated_side_probability",
            "probability_advantage",
            "player_display_name",
        ],
        ascending=[True, True, False, False, True],
    )
    return (
        ranked.groupby(["season", "week"], group_keys=False, sort=True)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def select_weekly_cap(
    development_pool: pd.DataFrame,
    *,
    candidates: tuple[int, ...] = (6, 8, 10, 12),
    minimum_decisions: int = 60,
    minimum_weeks: int = 8,
) -> tuple[int, list[dict[str, Any]]]:
    """Select one cap on development evidence without reading final outcomes."""

    leaderboard: list[dict[str, Any]] = []
    for top_n in sorted(set(int(value) for value in candidates)):
        pruned = prune_weekly_pool(development_pool, top_n=top_n)
        summary = summarize_market_rows(pruned)
        interval = summary.get("hit_rate_wilson_95")
        eligible = bool(
            summary["graded_decisions"] >= int(minimum_decisions)
            and summary["distinct_weeks"] >= int(minimum_weeks)
            and interval
            and interval[0] > 0.5
            and summary.get("roi") is not None
            and summary["roi"] > 0.0
        )
        leaderboard.append({"top_n": top_n, "eligible": eligible, **summary})
    eligible_rows = [row for row in leaderboard if row["eligible"]]
    if not eligible_rows:
        raise ValueError("No weekly-cap candidate passed the development evidence gate.")
    selected = max(
        eligible_rows,
        key=lambda row: (
            row["hit_rate_wilson_95"][0],
            row["hit_rate"],
            row["roi"],
            -row["top_n"],
        ),
    )
    return int(selected["top_n"]), leaderboard


def target_promotion_gate(summary: dict[str, Any]) -> dict[str, Any]:
    interval = summary.get("hit_rate_wilson_95")
    passed = bool(
        summary.get("graded_decisions", 0) >= 150
        and summary.get("distinct_weeks", 0) >= 8
        and summary.get("hit_rate") is not None
        and summary["hit_rate"] >= 0.58
        and interval
        and interval[0] > 0.50
        and summary.get("roi") is not None
        and summary["roi"] > 0.0
    )
    return {
        "status": "passed" if passed else "failed",
        "criteria": {
            "minimum_graded_decisions": 150,
            "minimum_distinct_weeks": 8,
            "minimum_hit_rate": 0.58,
            "wilson_95_lower_bound_above": 0.50,
            "positive_roi": True,
        },
    }


def build_prediction_pool(
    rows: pd.DataFrame,
    *,
    evaluation_split: str,
    architecture_by_target: dict[str, str],
    promotion_by_target: dict[str, str],
) -> pd.DataFrame:
    """Return only policy-eligible picks with transparent settled validation."""

    pool = rows.loc[rows["eligible"]].copy()
    pool["evaluation_split"] = evaluation_split
    pool["selected_architecture"] = pool["target"].map(architecture_by_target)
    pool["target_final_validation_status"] = pool["target"].map(promotion_by_target)
    pool["pick_validation"] = pool["result"].map(
        {"win": "pass", "loss": "fail", "push": "push"}
    )
    preferred = [
        "season",
        "week",
        "evaluation_split",
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "opponent_team",
        "target",
        "side",
        "line",
        "over_price",
        "under_price",
        "selected_price",
        "estimated_side_probability",
        "no_vig_side_probability",
        "probability_advantage",
        "actual",
        "result",
        "pick_validation",
        "profit_units",
        "current_prediction",
        "challenger_prediction",
        "baseline",
        "selected_architecture",
        "target_final_validation_status",
        "bookmaker",
        "source",
        "line_phase",
        "snapshot_time_utc",
        "commence_time_utc",
    ]
    columns = [column for column in preferred if column in pool.columns]
    sort_columns = [
        column
        for column in ["season", "week", "target", "player_display_name", "player_id"]
        if column in columns
    ]
    return pool[columns].sort_values(sort_columns, na_position="last").reset_index(drop=True)


def build_weekly_validation(
    pools: list[pd.DataFrame],
    *,
    season_weeks: dict[int, list[int]],
    promotion_by_target: dict[str, str],
    development_season: int,
    development_warmup_through_week: int = 10,
) -> pd.DataFrame:
    """Build week/target validation, retaining empty warm-up weeks explicitly."""

    combined = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    targets = sorted(promotion_by_target)
    output: list[dict[str, Any]] = []
    for season, weeks in sorted(season_weeks.items()):
        season_split = (
            "development_walk_forward" if season == development_season else "final_test"
        )
        for week in sorted(set(int(value) for value in weeks)):
            for target in ["overall", *targets]:
                if combined.empty:
                    part = combined
                else:
                    mask = combined["season"].eq(season) & combined["week"].eq(week)
                    if target != "overall":
                        mask &= combined["target"].eq(target)
                    part = combined.loc[mask]
                wins = int(part["result"].eq("win").sum()) if not part.empty else 0
                losses = int(part["result"].eq("loss").sum()) if not part.empty else 0
                pushes = int(part["result"].eq("push").sum()) if not part.empty else 0
                decisions = wins + losses
                if season == development_season and week <= development_warmup_through_week:
                    pool_status = "calibration_warmup"
                elif part.empty:
                    pool_status = "no_eligible_picks"
                else:
                    pool_status = "scored"
                output.append(
                    {
                        "season": season,
                        "week": week,
                        "evaluation_split": season_split,
                        "target": target,
                        "pool_status": pool_status,
                        "target_final_validation_status": (
                            "mixed" if target == "overall" else promotion_by_target[target]
                        ),
                        "picks": int(len(part)),
                        "wins": wins,
                        "losses": losses,
                        "pushes": pushes,
                        "hit_rate": round(wins / decisions, 4) if decisions else None,
                        "roi": (
                            round(float(part["profit_units"].mean()), 4)
                            if not part.empty
                            else None
                        ),
                        "profit_units": (
                            round(float(part["profit_units"].sum()), 4)
                            if not part.empty
                            else None
                        ),
                    }
                )
    return pd.DataFrame(output)


def build_learning_frames(
    stats: pd.DataFrame,
    market_rows: pd.DataFrame,
    *,
    latent: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], list[str]]:
    """Join posted lines to pregame raw/latent player features."""

    latent_columns = [column for column in latent.columns if column.startswith("latent_")]
    frames: dict[str, pd.DataFrame] = {}
    raw_features: dict[str, list[str]] = {}
    required_market = {
        "player_id",
        "season",
        "week",
        "target",
        "line",
        "actual",
        "over_price",
        "under_price",
        "current_prediction",
        "challenger_prediction",
        "baseline",
    }
    missing = sorted(required_market.difference(market_rows.columns))
    if missing:
        raise ValueError(f"Market rows are missing required columns: {', '.join(missing)}")

    for spec in TARGET_SPECS:
        feature_frame, features = build_features(stats, spec)
        feature_frame = feature_frame.merge(
            latent,
            on=["player_id", "season", "week"],
            how="left",
            validate="one_to_one",
        )
        selected_market = market_rows.loc[market_rows["target"].eq(spec.key)].copy()
        market_columns = [
            "player_id",
            "season",
            "week",
            "line",
            "actual",
            "over_price",
            "under_price",
            "current_prediction",
            "challenger_prediction",
            "baseline",
        ]
        identity_optional = [
            column
            for column in [
                "player_display_name",
                "position",
                "recent_team",
                "opponent_team",
                "bookmaker",
                "source",
                "line_phase",
                "snapshot_time_utc",
                "commence_time_utc",
            ]
            if column in selected_market.columns
        ]
        joined = selected_market[market_columns + identity_optional].merge(
            feature_frame[["player_id", "season", "week"] + features + latent_columns],
            on=["player_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        scale = TARGET_SCALES[spec.key]
        joined["line_scaled"] = joined["line"] / scale
        joined["current_edge_scaled"] = (joined["current_prediction"] - joined["line"]) / scale
        joined["challenger_edge_scaled"] = (
            joined["challenger_prediction"] - joined["line"]
        ) / scale
        joined["baseline_edge_scaled"] = (joined["baseline"] - joined["line"]) / scale
        joined["model_disagreement_scaled"] = (
            joined["current_prediction"] - joined["challenger_prediction"]
        ).abs() / scale
        joined["over_result"] = joined["actual"].gt(joined["line"]).astype(int)
        joined["target"] = spec.key
        frames[spec.key] = joined.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
        raw_features[spec.key] = features + MARKET_SIGNAL_COLUMNS
    return frames, raw_features, latent_columns


def expanding_oof_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    *,
    architecture: str,
    random_state: int,
) -> pd.DataFrame:
    """Score weeks 11-18 from models that only read earlier weeks."""

    parts: list[pd.DataFrame] = []
    maximum_week = int(frame["week"].max())
    for validation_start in range(11, maximum_week + 1, 2):
        validation_weeks = [validation_start, validation_start + 1]
        train = frame.loc[frame["week"].lt(validation_start)]
        validation = frame.loc[frame["week"].isin(validation_weeks)].copy()
        if len(train) < 100 or validation.empty:
            continue
        model = _model_factory(architecture, random_state + validation_start)
        model.fit(train[features], train["over_result"])
        validation["over_probability"] = model.predict_proba(validation[features])[:, 1]
        parts.append(validation)
    if not parts:
        raise ValueError("No expanding market-classifier folds could be constructed.")
    return pd.concat(parts, ignore_index=True)


def probability_metrics(rows: pd.DataFrame) -> dict[str, float | int | None]:
    actual = rows["over_result"].astype(int)
    probability = rows["over_probability"].astype(float)
    return {
        "rows": int(len(rows)),
        "brier_score": round(float(brier_score_loss(actual, probability)), 6),
        "log_loss": round(float(log_loss(actual, probability)), 6),
        "roc_auc": (
            round(float(roc_auc_score(actual, probability)), 6)
            if actual.nunique() == 2
            else None
        ),
    }


def candidate_feature_sets(
    raw_features: list[str], latent_columns: list[str]
) -> dict[str, list[str]]:
    return {
        "regularized_logistic_raw": raw_features,
        "regularized_logistic_raw_latent": raw_features + latent_columns,
        "catboost_raw": raw_features,
        "catboost_raw_latent": raw_features + latent_columns,
    }


def fit_selected_model(
    frame: pd.DataFrame,
    features: list[str],
    *,
    architecture: str,
    random_state: int,
) -> Any:
    model = _model_factory(architecture, random_state)
    model.fit(frame[features], frame["over_result"])
    return model


def build_frozen_latent_features(
    stats: pd.DataFrame,
    *,
    development_season: int,
    random_state: int = 42,
) -> tuple[PredictiveLatentEncoder, pd.DataFrame, dict[str, Any]]:
    """Fit one stable latent coordinate system before any line-labeled season."""

    sequence_table, sequence_features, sequence_targets = build_sequence_table(stats)
    encoder = PredictiveLatentEncoder(random_state=random_state).fit(
        sequence_table.loc[sequence_table["season"].lt(development_season)],
        sequence_features,
        sequence_targets,
    )
    latent = encoder.transform_frame(sequence_table)
    audit = {
        "training_seasons": sorted(
            int(value)
            for value in sequence_table.loc[
                sequence_table["season"].lt(development_season), "season"
            ].unique()
        ),
        "training_rows": int(encoder.training_rows_),
        "latent_dimensions": int(encoder.latent_dimensions),
        "pretraining_validation_score": round(float(encoder.pretraining_validation_score_), 6),
    }
    return encoder, latent, audit
