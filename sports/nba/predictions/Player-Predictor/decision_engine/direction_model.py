"""Direction-specific binary classification model.

Trains a gradient-boosted classifier that directly predicts P(pick wins)
given features like edge, direction, target, uncertainty, spike probability,
and market context.  This replaces the heuristic gap-percentile calibration
with a learned model that captures non-linear interactions.

The model is trained walk-forward: for each prediction day, it uses only
data from prior days.  This prevents look-ahead bias.

Key insight from empirical analysis:
  - UNDER wins at 68% overall, OVER at 50%
  - Edge magnitude is the strongest single predictor
  - TRB/AST OVERs with edge >= 1.0 win at 67-70%
  - Low-edge OVERs (< 0.5) lose at 58%
  - Spike probability and uncertainty interact with direction
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_predict
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


FEATURE_COLUMNS = [
    "abs_edge",
    "edge_to_sigma",
    "uncertainty_sigma",
    "spike_probability",
    "estimated_win_rate",
    "selection_confidence",
    "history_rows",
    "robust_pool_score",
    "direction_is_under",
    "target_is_pts",
    "target_is_trb",
    "target_is_ast",
    "edge_x_direction",
    "spike_x_direction",
    "edge_squared",
    "market_line_normalized",
]


@dataclass
class DirectionModelConfig:
    enabled: bool = True
    min_train_rows: int = 200
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.08
    subsample: float = 0.8
    min_samples_leaf: int = 20
    model_path: str = ""
    blend_weight: float = 0.40  # how much to trust the model vs the base rate


def _sf(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from raw data."""
    out = pd.DataFrame(index=df.index)

    out["abs_edge"] = pd.to_numeric(df.get("abs_edge"), errors="coerce").fillna(0.0)
    out["edge_to_sigma"] = pd.to_numeric(df.get("edge_to_sigma"), errors="coerce").fillna(0.0)
    out["uncertainty_sigma"] = pd.to_numeric(df.get("uncertainty_sigma"), errors="coerce").fillna(3.0)
    out["spike_probability"] = pd.to_numeric(df.get("spike_probability"), errors="coerce").fillna(0.5)

    # Handle column name differences between validation history and selector CSVs
    if "estimated_win_rate" in df.columns:
        out["estimated_win_rate"] = pd.to_numeric(df["estimated_win_rate"], errors="coerce").fillna(0.5)
    elif "expected_win_rate" in df.columns:
        out["estimated_win_rate"] = pd.to_numeric(df["expected_win_rate"], errors="coerce").fillna(0.5)
    else:
        out["estimated_win_rate"] = 0.5

    if "selection_confidence" in df.columns:
        out["selection_confidence"] = pd.to_numeric(df["selection_confidence"], errors="coerce").fillna(0.0)
    elif "final_confidence" in df.columns:
        out["selection_confidence"] = pd.to_numeric(df["final_confidence"], errors="coerce").fillna(0.0)
    else:
        out["selection_confidence"] = 0.0

    out["history_rows"] = pd.to_numeric(df.get("history_rows"), errors="coerce").fillna(30.0).clip(upper=200.0)
    out["robust_pool_score"] = pd.to_numeric(df.get("robust_pool_score"), errors="coerce").fillna(0.0)

    direction = df.get("direction", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
    target = df.get("target", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()

    out["direction_is_under"] = (direction == "UNDER").astype(float)
    out["target_is_pts"] = (target == "PTS").astype(float)
    out["target_is_trb"] = (target == "TRB").astype(float)
    out["target_is_ast"] = (target == "AST").astype(float)

    # Interactions
    out["edge_x_direction"] = out["abs_edge"] * out["direction_is_under"]
    out["spike_x_direction"] = out["spike_probability"] * (1.0 - out["direction_is_under"])
    out["edge_squared"] = out["abs_edge"] ** 2

    # Normalized market line (different scale per target)
    market_line = pd.to_numeric(df.get("market_line"), errors="coerce").fillna(10.0)
    target_medians = {"PTS": 20.0, "TRB": 6.0, "AST": 5.0}
    median_line = target.map(target_medians).fillna(10.0).astype(float)
    out["market_line_normalized"] = (market_line / median_line).clip(0.1, 5.0)

    return out[FEATURE_COLUMNS]


def train_direction_model(
    history_df: pd.DataFrame,
    *,
    config: DirectionModelConfig | None = None,
) -> dict[str, Any]:
    """Train the direction classifier from historical validation data.

    Parameters
    ----------
    history_df : pd.DataFrame
        Must have columns: result (win/loss/push), direction, target,
        abs_edge, uncertainty_sigma, spike_probability, etc.
    config : DirectionModelConfig

    Returns
    -------
    Dict with 'model', 'feature_columns', 'train_rows', 'metrics'.
    """
    if not HAS_SKLEARN:
        return {"model": None, "error": "sklearn not installed"}

    cfg = config or DirectionModelConfig()

    # Filter to graded rows only
    graded = history_df[history_df["result"].isin(["win", "loss"])].copy()
    if len(graded) < cfg.min_train_rows:
        return {"model": None, "error": f"insufficient_rows ({len(graded)} < {cfg.min_train_rows})"}

    y = (graded["result"] == "win").astype(int)
    X = _prepare_features(graded)

    # Drop rows with NaN features
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < cfg.min_train_rows:
        return {"model": None, "error": f"insufficient_valid_rows ({len(X)} < {cfg.min_train_rows})"}

    # Train
    base_model = GradientBoostingClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=42,
    )
    base_model.fit(X, y)

    # Calibrate with isotonic regression
    calibrated = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    calibrated.fit(X, y)

    # Cross-validated metrics
    cv_probs = cross_val_predict(base_model, X, y, cv=5, method="predict_proba")[:, 1]
    cv_preds = (cv_probs >= 0.5).astype(int)
    accuracy = float((cv_preds == y).mean())
    brier = float(((cv_probs - y) ** 2).mean())

    # Feature importance
    importances = dict(zip(FEATURE_COLUMNS, base_model.feature_importances_.tolist()))

    return {
        "model": calibrated,
        "base_model": base_model,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(X)),
        "metrics": {
            "accuracy": accuracy,
            "brier_score": brier,
            "mean_predicted_prob": float(cv_probs.mean()),
            "actual_win_rate": float(y.mean()),
            "calibration_gap": float(abs(cv_probs.mean() - y.mean())),
        },
        "feature_importance": importances,
    }


def predict_win_probability(
    candidates: pd.DataFrame,
    model_payload: dict[str, Any],
    *,
    config: DirectionModelConfig | None = None,
) -> pd.Series:
    """Predict win probability for candidates using the trained model.

    Returns a Series of probabilities aligned with candidates.index.
    """
    cfg = config or DirectionModelConfig()
    model = model_payload.get("model")

    if model is None or not cfg.enabled:
        return pd.to_numeric(
            candidates.get("estimated_win_rate", candidates.get("expected_win_rate")),
            errors="coerce",
        ).fillna(0.5)

    X = _prepare_features(candidates)
    valid_mask = X.notna().all(axis=1)

    probs = pd.Series(0.5, index=candidates.index, dtype="float64")
    if valid_mask.any():
        X_valid = X[valid_mask]
        model_probs = model.predict_proba(X_valid)[:, 1]
        probs.loc[valid_mask] = pd.Series(model_probs, index=X_valid.index, dtype="float64").values

    # Blend with base rate
    if "estimated_win_rate" in candidates.columns:
        base_rate = pd.to_numeric(candidates["estimated_win_rate"], errors="coerce").fillna(0.5)
    elif "expected_win_rate" in candidates.columns:
        base_rate = pd.to_numeric(candidates["expected_win_rate"], errors="coerce").fillna(0.5)
    else:
        base_rate = pd.Series(0.5, index=candidates.index, dtype="float64")

    blended = (cfg.blend_weight * probs + (1.0 - cfg.blend_weight) * base_rate).astype("float64")
    return blended.clip(0.01, 0.99)


def walk_forward_backtest(
    history_df: pd.DataFrame,
    *,
    min_train_days: int = 7,
    retrain_every_days: int = 3,
    config: DirectionModelConfig | None = None,
) -> pd.DataFrame:
    """Run walk-forward backtest: train on past, predict next day, grade.

    Returns a DataFrame with columns: market_date, predicted_prob,
    actual_result, win, direction, target, abs_edge.
    """
    cfg = config or DirectionModelConfig()
    if not HAS_SKLEARN:
        return pd.DataFrame()

    graded = history_df[history_df["result"].isin(["win", "loss"])].copy()
    graded["_date"] = pd.to_datetime(graded["market_date"], errors="coerce")
    graded = graded.dropna(subset=["_date"]).sort_values("_date")

    dates = sorted(graded["_date"].dt.date.unique())
    if len(dates) < min_train_days + 1:
        return pd.DataFrame()

    results = []
    current_model = None
    last_train_date_idx = -1

    for i, test_date in enumerate(dates):
        if i < min_train_days:
            continue

        # Retrain periodically
        if current_model is None or (i - last_train_date_idx) >= retrain_every_days:
            train_data = graded[graded["_date"].dt.date < test_date]
            payload = train_direction_model(train_data, config=cfg)
            if payload.get("model") is not None:
                current_model = payload
                last_train_date_idx = i

        if current_model is None:
            continue

        # Predict for test date
        test_data = graded[graded["_date"].dt.date == test_date].copy()
        if test_data.empty:
            continue

        probs = predict_win_probability(test_data, current_model, config=cfg)
        test_data["predicted_prob"] = probs.values
        test_data["win"] = (test_data["result"] == "win").astype(int)

        for _, row in test_data.iterrows():
            results.append({
                "market_date": str(test_date),
                "predicted_prob": float(row["predicted_prob"]),
                "win": int(row["win"]),
                "direction": str(row.get("direction", "")),
                "target": str(row.get("target", "")),
                "abs_edge": float(row.get("abs_edge", 0)),
                "base_win_rate": float(row.get("estimated_win_rate", 0.5)),
            })

    return pd.DataFrame(results)
