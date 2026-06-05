#!/usr/bin/env python3
"""
NBA v10 probability stack.

v10 treats the market as the prior and learns a correction, then blends:
  - distribution/CDF probability from v9
  - direct line-crossing classifier
  - market-residual probability
  - shrinkage side priors

The goal is higher probability resolution without sacrificing calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:  # pragma: no cover
    ColumnTransformer = None


EPS = 1e-4


def clip_prob(values) -> np.ndarray:
    return np.asarray(values, dtype=float).clip(EPS, 1.0 - EPS)


def logit(values) -> np.ndarray:
    p = clip_prob(values)
    return np.log(p / (1.0 - p))


def sigmoid(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "market_no_vig_over",
        "line",
        "model_mean",
        "sigma",
        "belief_uncertainty",
        "line_distance",
        "line_distance_abs",
        "model_minus_line",
        "model_minus_line_z",
        "player_market_games_prior",
        "player_market_over_rate_prior",
        "market_line_over_rate_prior",
        "market_line_games_prior",
        "market_side_prior_over",
        "side_prior_over",
        "dist_logit",
        "market_logit",
    ]
    return [c for c in candidates if c in frame.columns]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["market", "line_bucket", "player_volume_bucket"] if c in frame.columns]


def add_v10_features(rows: pd.DataFrame, prior_source: pd.DataFrame | None = None, shrink_k: int = 300) -> pd.DataFrame:
    """Add leakage-safe v10 features. Priors are fit from prior_source if supplied."""
    rows = rows.copy()
    prior_cols = [
        "player_market_games_prior",
        "player_market_over_rate_raw",
        "player_market_residual_mean_prior",
        "market_line_games_prior",
        "market_line_over_rate_raw",
        "market_side_games_prior",
        "market_side_over_rate_raw",
        "player_market_over_rate_prior",
        "market_line_over_rate_prior",
        "market_side_prior_over",
        "side_prior_over",
        "player_volume_bucket",
    ]
    rows = rows.drop(columns=[c for c in prior_cols if c in rows.columns], errors="ignore")
    if "market_no_vig_over" not in rows.columns:
        rows["market_no_vig_over"] = 0.5
    if "market_no_vig_under" not in rows.columns:
        rows["market_no_vig_under"] = 1.0 - rows["market_no_vig_over"]
    rows["p_over_raw"] = clip_prob(rows["p_over_raw"])
    rows["market_no_vig_over"] = clip_prob(rows["market_no_vig_over"])
    rows["dist_logit"] = logit(rows["p_over_raw"])
    rows["market_logit"] = logit(rows["market_no_vig_over"])
    rows["model_minus_line"] = rows.get("model_mean", 0.0) - rows.get("line", 0.0)
    rows["model_minus_line_z"] = rows["model_minus_line"] / pd.Series(rows.get("sigma", 1.0)).replace(0, np.nan).fillna(1.0)
    rows["line_distance"] = rows["line"] - rows.get("model_mean", rows["line"])
    rows["line_distance_abs"] = rows["line_distance"].abs()
    rows["line_bucket"] = pd.cut(
        rows["line"],
        bins=[-0.001, 3.5, 7.5, 12.5, 20.5, 99],
        labels=["tiny", "small", "medium", "large", "star"],
    ).astype(str)

    source = rows if prior_source is None or prior_source.empty else prior_source.copy()
    source["line_bucket"] = pd.cut(
        source["line"],
        bins=[-0.001, 3.5, 7.5, 12.5, 20.5, 99],
        labels=["tiny", "small", "medium", "large", "star"],
    ).astype(str)

    player_market = source.groupby(["player", "market"]).agg(
        n=("result_over", "size"),
        over_rate=("result_over", "mean"),
        residual_mean=("residual", "mean"),
    )
    market_line = source.groupby(["market", "line_bucket"]).agg(
        n=("result_over", "size"),
        over_rate=("result_over", "mean"),
    )
    market_side = source.groupby("market").agg(
        n=("result_over", "size"),
        over_rate=("result_over", "mean"),
    )

    rows = rows.merge(
        player_market.rename(
            columns={
                "n": "player_market_games_prior",
                "over_rate": "player_market_over_rate_raw",
                "residual_mean": "player_market_residual_mean_prior",
            }
        ),
        left_on=["player", "market"],
        right_index=True,
        how="left",
    )
    rows = rows.merge(
        market_line.rename(columns={"n": "market_line_games_prior", "over_rate": "market_line_over_rate_raw"}),
        left_on=["market", "line_bucket"],
        right_index=True,
        how="left",
    )
    rows = rows.merge(
        market_side.rename(columns={"n": "market_side_games_prior", "over_rate": "market_side_over_rate_raw"}),
        left_on="market",
        right_index=True,
        how="left",
    )

    for n_col, raw_col, out_col in [
        ("player_market_games_prior", "player_market_over_rate_raw", "player_market_over_rate_prior"),
        ("market_line_games_prior", "market_line_over_rate_raw", "market_line_over_rate_prior"),
        ("market_side_games_prior", "market_side_over_rate_raw", "market_side_prior_over"),
    ]:
        n = rows[n_col].fillna(0.0)
        raw = rows[raw_col].fillna(0.5)
        rows[out_col] = 0.5 + (raw - 0.5) * n / (n + shrink_k)

    rows["side_prior_over"] = (
        0.50 * rows["market_side_prior_over"].fillna(0.5)
        + 0.30 * rows["market_line_over_rate_prior"].fillna(0.5)
        + 0.20 * rows["player_market_over_rate_prior"].fillna(0.5)
    ).clip(0.01, 0.99)
    rows["player_volume_bucket"] = pd.cut(
        rows["player_market_games_prior"].fillna(0.0),
        bins=[-0.001, 20, 60, 120, 100000],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)

    numeric = rows.select_dtypes(include=[np.number]).columns
    rows[numeric] = rows[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return rows


def _preprocessor(frame: pd.DataFrame):
    numeric = _numeric_columns(frame)
    categorical = _categorical_columns(frame)
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers), numeric, categorical


@dataclass
class V10ProbabilityStack:
    direct_model: object
    residual_model: object
    blender_model: object
    risk_model: object
    feature_columns: list[str]
    categorical_columns: list[str]

    def predict_components(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = add_v10_features(rows)
        out = rows.copy()
        out["p_distribution"] = clip_prob(out["p_over_raw"])
        out["p_direct"] = self.direct_model.predict_proba(out)[:, 1].clip(0.01, 0.99)
        residual_prob = self.residual_model.predict_proba(out)[:, 1].clip(0.01, 0.99)
        out["p_market_residual"] = sigmoid(logit(out["market_no_vig_over"]) + (logit(residual_prob) - logit(0.5)))
        out["p_side_prior"] = clip_prob(out["side_prior_over"])
        blend_features = build_blender_frame(out)
        out["p_v10_raw"] = self.blender_model.predict_proba(blend_features)[:, 1].clip(0.01, 0.99)
        out["brier_risk"] = np.asarray(self.risk_model.predict(build_risk_frame(out)), dtype=float).clip(0.0, 1.0)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "V10ProbabilityStack":
        return joblib.load(str(path))


def build_blender_frame(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "logit_distribution": logit(rows["p_distribution"]),
            "logit_direct": logit(rows["p_direct"]),
            "logit_market_residual": logit(rows["p_market_residual"]),
            "logit_side_prior": logit(rows["p_side_prior"]),
            "uncertainty": rows.get("uncertainty", rows.get("belief_uncertainty", 0.5)),
            "line_distance_abs": rows.get("line_distance_abs", 0.0),
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_risk_frame(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "abs_edge_distribution": (rows["p_distribution"] - rows["market_no_vig_over"]).abs(),
            "branch_disagreement": rows[["p_distribution", "p_direct", "p_market_residual", "p_side_prior"]].std(axis=1),
            "belief_uncertainty": rows.get("belief_uncertainty", 0.5),
            "line_distance_abs": rows.get("line_distance_abs", 0.0),
            "player_market_games_prior": rows.get("player_market_games_prior", 0.0),
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_v10_probability_stack(train_rows: pd.DataFrame, random_state: int = 42) -> V10ProbabilityStack:
    if ColumnTransformer is None:
        raise ImportError("scikit-learn is required for v10 probability stack")
    rows = add_v10_features(train_rows)
    y = rows["result_over"].astype(int).to_numpy()
    pre_direct, numeric, categorical = _preprocessor(rows)
    pre_residual, _, _ = _preprocessor(rows)

    direct_model = Pipeline(
        [
            ("pre", pre_direct),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.035,
                    max_leaf_nodes=18,
                    l2_regularization=0.08,
                    random_state=random_state,
                ),
            ),
        ]
    )
    residual_model = Pipeline(
        [
            ("pre", pre_residual),
            (
                "clf",
                LogisticRegression(
                    C=0.45,
                    max_iter=1000,
                    class_weight=None,
                    random_state=random_state,
                ),
            ),
        ]
    )
    direct_oof = np.full(len(rows), 0.5, dtype=float)
    residual_oof = np.full(len(rows), 0.5, dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_idx, valid_idx in cv.split(rows, y):
        direct_fold = clone(direct_model)
        residual_fold = clone(residual_model)
        direct_fold.fit(rows.iloc[train_idx], y[train_idx])
        residual_fold.fit(rows.iloc[train_idx], y[train_idx])
        direct_oof[valid_idx] = direct_fold.predict_proba(rows.iloc[valid_idx])[:, 1]
        residual_oof[valid_idx] = residual_fold.predict_proba(rows.iloc[valid_idx])[:, 1]

    direct_model.fit(rows, y)
    residual_model.fit(rows, y)

    component_rows = rows.copy()
    component_rows["p_distribution"] = clip_prob(component_rows["p_over_raw"])
    component_rows["p_direct"] = np.clip(direct_oof, 0.01, 0.99)
    residual_prob = np.clip(residual_oof, 0.01, 0.99)
    component_rows["p_market_residual"] = sigmoid(
        logit(component_rows["market_no_vig_over"]) + (logit(residual_prob) - logit(0.5))
    )
    component_rows["p_side_prior"] = clip_prob(component_rows["side_prior_over"])

    blender_model = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.35, max_iter=1000, random_state=random_state)),
        ]
    )
    blender_model.fit(build_blender_frame(component_rows), y)

    component_rows["p_v10_raw"] = blender_model.predict_proba(build_blender_frame(component_rows))[:, 1].clip(0.01, 0.99)
    component_rows["probability_abs_error"] = (component_rows["result_over"] - component_rows["p_v10_raw"]).abs()
    risk_model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=16,
        l2_regularization=0.10,
        random_state=random_state,
    )
    risk_model.fit(build_risk_frame(component_rows), component_rows["probability_abs_error"].to_numpy())

    return V10ProbabilityStack(
        direct_model=direct_model,
        residual_model=residual_model,
        blender_model=blender_model,
        risk_model=risk_model,
        feature_columns=numeric,
        categorical_columns=categorical,
    )


def fit_predict_v10(train_rows: pd.DataFrame, score_rows: pd.DataFrame) -> tuple[V10ProbabilityStack, pd.DataFrame]:
    stack = fit_v10_probability_stack(train_rows)
    score = add_v10_features(score_rows, prior_source=train_rows)
    # The stack's predict_components calls add_v10_features again, so preserve
    # leakage-safe priors by predicting manually with the prepared score frame.
    out = score.copy()
    out["p_distribution"] = clip_prob(out["p_over_raw"])
    out["p_direct"] = stack.direct_model.predict_proba(out)[:, 1].clip(0.01, 0.99)
    residual_prob = stack.residual_model.predict_proba(out)[:, 1].clip(0.01, 0.99)
    out["p_market_residual"] = sigmoid(logit(out["market_no_vig_over"]) + (logit(residual_prob) - logit(0.5)))
    out["p_side_prior"] = clip_prob(out["side_prior_over"])
    out["p_v10_raw"] = stack.blender_model.predict_proba(build_blender_frame(out))[:, 1].clip(0.01, 0.99)
    out["brier_risk"] = np.asarray(stack.risk_model.predict(build_risk_frame(out)), dtype=float).clip(0.0, 1.0)
    return stack, out
