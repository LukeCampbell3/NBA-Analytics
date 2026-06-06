"""
Percentile computation engine.

Computes raw, position, and role percentiles for all capability dimensions.
Uses existing player data from Data-Proc and player_cards.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_percentile(value: float, population: np.ndarray) -> float:
    """Compute percentile rank of a value within a population.

    Returns 0-100 percentile (100 = best).
    """
    if len(population) == 0 or np.isnan(value):
        return 0.0
    below = np.sum(population < value)
    equal = np.sum(population == value)
    pct = (below + 0.5 * equal) / len(population) * 100.0
    return round(float(np.clip(pct, 0, 100)), 1)


def compute_percentiles_for_column(
    values: pd.Series,
    groups: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute raw and group-based percentiles for a series.

    Args:
        values: The stat values
        groups: Optional grouping (e.g. position) for position percentiles

    Returns:
        DataFrame with raw_percentile and optionally group_percentile columns
    """
    population = values.dropna().values
    raw_pcts = values.apply(lambda v: compute_percentile(v, population) if pd.notna(v) else None)

    result = pd.DataFrame({"raw_percentile": raw_pcts})

    if groups is not None:
        group_pcts = []
        for idx in values.index:
            val = values[idx]
            grp = groups[idx]
            if pd.isna(val) or pd.isna(grp):
                group_pcts.append(None)
                continue
            group_pop = values[groups == grp].dropna().values
            group_pcts.append(compute_percentile(val, group_pop))
        result["group_percentile"] = group_pcts

    return result


def reliability_shrinkage(
    raw_percentile: float,
    sample_size: int,
    min_reliable_sample: int = 30,
    league_mean: float = 50.0,
) -> float:
    """Apply Bayesian-style shrinkage toward league mean for small samples.

    Small samples get pulled toward 50th percentile.
    Large samples retain their raw percentile.
    """
    if sample_size >= min_reliable_sample:
        weight = 1.0
    else:
        weight = sample_size / min_reliable_sample

    adjusted = (weight * raw_percentile) + ((1 - weight) * league_mean)
    return round(float(np.clip(adjusted, 0, 100)), 1)
