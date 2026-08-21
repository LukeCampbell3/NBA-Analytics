from __future__ import annotations

"""Disciplined, out-of-fold test of the edge-vs-hit-rate shape.

Does NOT hand-pick a sweet spot from eyeballing bucket win rates (that was
the earlier session's mistake). Instead fits three simple logistic-regression
shape families on TRAIN-fold data only, and compares them purely on
out-of-fold log-loss/Brier on the held-out day:

  linear:      win ~ a + b*edge                        (monotonic)
  quadratic:   win ~ a + b*edge + c*edge^2              (saturating or inverted-U,
                                                          sign of c decides which)
  log:         win ~ a + b*log1p(edge)                  (saturating, monotonic)

All three also include rmse as a control feature, fit on standardized
(train-only mean/std) inputs. The winner is whichever has the lowest mean
out-of-fold Brier score across folds; ties go to the simpler family.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .chronological_cv import Fold, split


def _standardize(train_x: np.ndarray, apply_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    return (apply_x - mean) / std


@dataclass
class ShapeFoldResult:
    shape: str
    date: str
    brier: float
    quadratic_coef: float | None


def _features(frame: pd.DataFrame, shape: str) -> np.ndarray:
    edge = frame["corrected_edge"].to_numpy(dtype=float)
    rmse = frame["rmse"].to_numpy(dtype=float)
    if shape == "linear":
        return np.column_stack([edge, rmse])
    if shape == "quadratic":
        return np.column_stack([edge, edge**2, rmse])
    if shape == "log":
        return np.column_stack([np.log1p(edge), rmse])
    raise ValueError(shape)


def evaluate_edge_shapes(frame: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    results: list[ShapeFoldResult] = []
    for shape in ("linear", "quadratic", "log"):
        for fold in folds:
            train, val = split(frame, fold)
            if len(train) < 20 or val.empty:
                continue
            train_x_raw = _features(train, shape)
            val_x_raw = _features(val, shape)
            train_x = _standardize(train_x_raw, train_x_raw)
            val_x = _standardize(train_x_raw, val_x_raw)
            model = LogisticRegression(C=1.0, max_iter=1000)
            model.fit(train_x, train["win"])
            probs = model.predict_proba(val_x)[:, 1]
            brier = float(np.mean((probs - val["win"].to_numpy(dtype=float)) ** 2))
            quad_coef = float(model.coef_[0][1]) if shape == "quadratic" else None
            results.append(ShapeFoldResult(shape=shape, date=fold.val_date, brier=brier, quadratic_coef=quad_coef))
    return pd.DataFrame([r.__dict__ for r in results])


def summarize_shape_comparison(shape_results: pd.DataFrame) -> pd.DataFrame:
    summary = shape_results.groupby("shape")["brier"].agg(["mean", "std", "count"])
    summary = summary.rename(columns={"mean": "mean_oof_brier", "std": "std_oof_brier", "count": "n_folds"})
    quad = shape_results[shape_results["shape"] == "quadratic"]["quadratic_coef"]
    if not quad.empty:
        summary.loc["quadratic", "quadratic_coef_sign_consistency"] = float((quad < 0).mean())
        summary.loc["quadratic", "mean_quadratic_coef"] = float(quad.mean())
    return summary.sort_values("mean_oof_brier")
