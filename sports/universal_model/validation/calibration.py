"""Calibration reporting (spec section 43): reliability curve + ECE,
broken out by sport and by predicted-probability bucket. "By odds bucket"
is reported only for rows with a real market price (MLB, ~2.5% of MLB
rows per reports/INVENTORY.md) -- disclosed rather than silently
extrapolated to the ~97.5% of rows with no real quoted price.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.model.universal_model import UniversalModel
from sports.universal_model.validation.metrics import expected_calibration_error


@torch.no_grad()
def reliability_curve(model: UniversalModel, dataset: UniversalDataset, n_bins: int = 10, batch_size: int = 256) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs_all, y_all, has_price_all = [], [], []
    idx = 0
    has_price = (dataset.frame["american_price"].notna()).values
    for batch in loader:
        out = model(batch)
        mask = batch["y_over_mask"].numpy() > 0
        bs = mask.shape[0]
        probs_all.append(out["prob_over"].numpy()[mask])
        y_all.append(batch["y_over"].numpy()[mask])
        has_price_all.append(has_price[idx: idx + bs][mask])
        idx += bs
    model.train()
    probs = np.concatenate(probs_all) if probs_all else np.array([])
    y = np.concatenate(y_all) if y_all else np.array([])
    has_price_flat = np.concatenate(has_price_all) if has_price_all else np.array([])

    bins = np.linspace(0, 1, n_bins + 1)
    curve = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            curve.append({"bin": [float(lo), float(hi)], "n": 0, "mean_predicted": None, "empirical_rate": None})
            continue
        curve.append(
            {
                "bin": [float(lo), float(hi)],
                "n": int(mask.sum()),
                "mean_predicted": float(probs[mask].mean()),
                "empirical_rate": float(y[mask].mean()),
            }
        )

    priced_mask = has_price_flat
    return {
        "n": int(len(probs)),
        "ece_overall": expected_calibration_error(probs, y) if len(probs) else None,
        "reliability_curve": curve,
        "priced_subset": {
            "n": int(priced_mask.sum()),
            "note": "rows with a real quoted market price (american_price not null); ~2.5% of MLB rows per reports/INVENTORY.md",
            "ece": expected_calibration_error(probs[priced_mask], y[priced_mask]) if priced_mask.sum() > 20 else None,
        },
    }
