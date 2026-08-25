"""Primary validation metrics (spec section 43)."""
from __future__ import annotations

import numpy as np


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def auc(probs: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    if len(probs) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "auc": None, "ece": None}
    return {
        "n": int(len(probs)),
        "brier": brier_score(probs, labels),
        "log_loss": log_loss(probs, labels),
        "auc": auc(probs, labels),
        "ece": expected_calibration_error(probs, labels),
    }


def regression_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    if len(pred) == 0:
        return {"n": 0, "mae": None, "rmse": None}
    return {"n": int(len(pred)), "mae": mae(pred, actual), "rmse": rmse(pred, actual)}
