"""Temperature-based sport sampling (spec section 12): P(sport=s) ∝ N_s^α.

Documented default alpha=0.5, never tuned on TEST (only ever constructed
from DERIVE row counts in practice).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from sports.universal_model.data.dataset import UniversalDataset


def build_temperature_sampler(dataset: UniversalDataset, alpha: float = 0.5, num_samples: int | None = None) -> WeightedRandomSampler:
    sports = dataset.frame["sport"].values
    counts = {s: int((sports == s).sum()) for s in set(sports)}
    # per-row weight so that summed sport probability âˆ N_s^alpha:
    # P(row in sport s) = N_s^alpha / N_s = N_s^(alpha-1), per row.
    weights = np.array([counts[s] ** (alpha - 1.0) for s in sports], dtype=np.float64)
    weights = weights / weights.sum()
    effective_contribution = {
        s: float(weights[sports == s].sum()) for s in counts
    }
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples or len(dataset),
        replacement=True,
    )
    sampler.effective_contribution = effective_contribution  # type: ignore[attr-defined]
    sampler.raw_counts = counts  # type: ignore[attr-defined]
    return sampler
