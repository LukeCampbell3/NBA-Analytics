from __future__ import annotations

"""H_OVER_RANKER_V1 candidate: walk-forward-fit logistic regression.

Fit strategy follows "prefer simple, stable, interpretable ranking rules
unless a more complex model shows robust chronological improvement": a
single logistic regression, refit on each fold's TRAIN dates only and
applied to that fold's VAL date (true walk-forward -- the model that scores
date D never saw date D or later). Feature set is chosen by
`edge_shape.py`'s disciplined out-of-fold comparison, not hand-picked.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .chronological_cv import Fold, split

# Frozen feature set for H_OVER_RANKER_V1. `corrected_edge_sq` was tried
# (the edge-shape test in edge_shape.py only marginally preferred a
# quadratic shape -- mean OOF Brier 0.24315 vs 0.24329 linear, not a robust
# margin) and dropped after an ablation showed it changed nothing (OOF
# Brier, top1, and top2 identical to 4 decimals with or without it): dead
# weight, removed per "prefer simple ... unless a more complex model shows
# robust chronological improvement." See reports/ for the ablation.
# market_books and market_line_std were tried and dropped: both are
# constant at exactly 0.0 for every H-target row in development data (not a
# code bug -- verified against the raw Market_Books/Market_Line_Std columns
# directly), so they carry zero information and got coefficient 0.0 in every
# fit. If the upstream pool CSV ever starts populating these for the H
# target, the ranker should be re-evaluated with them; until then they are
# dead weight and correctly excluded.
FEATURE_COLUMNS = (
    "corrected_edge",
    "rmse",
    "q_edge_over_rmse",
    "log1p_history_rows",
    "is_real_market",
)

# Frozen regularization strength. Chosen by minimizing mean out-of-fold
# Brier score (a secondary diagnostic, not the primary top-1/top-2 endpoint)
# across a logarithmic C sweep from 0.001 to 1.0; Brier was flat across
# C=1.0..0.01 (0.2430-0.2431) and only degraded below that, so C was picked
# from the middle of that plateau rather than the least-regularized point
# that happened to show the single best top-1 number (C=1.0 gave a
# suspiciously perfect 8/8 top-1 across all folds -- picking it because it
# looked best would be exactly the kind of post-hoc tuning against the
# primary endpoint this development protocol is required to avoid).
FROZEN_C = 0.1


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["q_edge_over_rmse"] = out["corrected_edge"] / out["rmse"].clip(lower=1e-6)
    out["log1p_history_rows"] = np.log1p(out["history_rows"].clip(lower=0))
    out["is_real_market"] = (out["market_source"] == "real").astype(float)
    return out


def fit_predict_walkforward(
    frame: pd.DataFrame,
    folds: list[Fold],
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    c: float = FROZEN_C,
) -> pd.DataFrame:
    """Walk-forward logistic regression: fit on fold.train_dates, score fold.val_date only.

    Returns a frame restricted to rows that appear as a val_date in some
    fold (i.e. exactly the rows evaluate.py's harness will score), with a
    `score_ranker_v1` column and, for provenance, the fold index and the
    fitted coefficients used for that row's prediction.
    """
    featured = build_features(frame)
    scored_parts = []
    fold_models: list[dict] = []
    for fold in folds:
        train, val = split(featured, fold)
        if len(train) < 20 or val.empty:
            continue
        train_x = train[list(feature_columns)].to_numpy(dtype=float)
        val_x = val[list(feature_columns)].to_numpy(dtype=float)
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std[std < 1e-9] = 1.0
        train_x_std = (train_x - mean) / std
        val_x_std = (val_x - mean) / std

        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(train_x_std, train["win"])
        val = val.copy()
        val["score_ranker_v1"] = model.predict_proba(val_x_std)[:, 1]
        val["fold_index"] = fold.index
        scored_parts.append(val)
        fold_models.append(
            {
                "fold_index": fold.index,
                "val_date": fold.val_date,
                "n_train": len(train),
                "coef": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
                "standardize_mean": mean.tolist(),
                "standardize_std": std.tolist(),
            }
        )
    result = pd.concat(scored_parts, ignore_index=True) if scored_parts else featured.iloc[0:0].copy()
    result.attrs["fold_models"] = fold_models
    result.attrs["feature_columns"] = list(feature_columns)
    return result


def fit_final_model(
    frame: pd.DataFrame, feature_columns: tuple[str, ...] = FEATURE_COLUMNS, c: float = FROZEN_C
) -> dict:
    """Fit on ALL rows passed in (intended: every DEVELOPMENT_STAMPS-eligible
    row) for deployment -- this is the model whose coefficients get frozen
    into manifest.py, not any single fold's walk-forward model above."""
    featured = build_features(frame)
    x = featured[list(feature_columns)].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    x_std = (x - mean) / std
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(x_std, featured["win"])
    return {
        "feature_columns": list(feature_columns),
        "c": c,
        "n_rows": int(len(featured)),
        "n_dates": int(featured["date"].nunique()),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "standardize_mean": mean.tolist(),
        "standardize_std": std.tolist(),
    }


def score_with_frozen_model(rows: pd.DataFrame, frozen: dict) -> np.ndarray:
    """Apply a fit_final_model()-style frozen config to new eligible rows."""
    featured = build_features(rows)
    x = featured[frozen["feature_columns"]].to_numpy(dtype=float)
    mean = np.array(frozen["standardize_mean"])
    std = np.array(frozen["standardize_std"])
    x_std = (x - mean) / std
    coef = np.array(frozen["coef"])
    logits = x_std @ coef + frozen["intercept"]
    return 1.0 / (1.0 + np.exp(-logits))
