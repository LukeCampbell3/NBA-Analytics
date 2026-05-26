from __future__ import annotations

import numpy as np
import pandas as pd


BELIEF_UNCERTAINTY_LOWER = 0.75
BELIEF_UNCERTAINTY_UPPER = 1.15


def _resolved_bounds(lower: float, upper: float) -> tuple[float, float]:
    low = float(lower)
    high = float(upper)
    if not np.isfinite(low):
        low = float(BELIEF_UNCERTAINTY_LOWER)
    if not np.isfinite(high):
        high = float(BELIEF_UNCERTAINTY_UPPER)
    if high <= low:
        high = low + 1e-9
    return low, high


def normalize_belief_uncertainty(
    value,
    default: float = 1.0,
    lower: float = BELIEF_UNCERTAINTY_LOWER,
    upper: float = BELIEF_UNCERTAINTY_UPPER,
):
    low, high = _resolved_bounds(lower, upper)
    span = max(high - low, 1e-9)
    if isinstance(value, pd.Series):
        numeric = pd.to_numeric(value, errors="coerce").fillna(float(default))
        return ((numeric - low) / span).clip(lower=0.0, upper=1.0)
    try:
        numeric = float(value)
        if np.isnan(numeric):
            numeric = float(default)
    except Exception:
        numeric = float(default)
    return float(np.clip((numeric - low) / span, 0.0, 1.0))


def belief_confidence_factor(
    value,
    default: float = 1.0,
    lower: float = BELIEF_UNCERTAINTY_LOWER,
    upper: float = BELIEF_UNCERTAINTY_UPPER,
):
    normalized = normalize_belief_uncertainty(value, default=default, lower=lower, upper=upper)
    if isinstance(normalized, pd.Series):
        return (1.0 - normalized).clip(lower=0.0, upper=1.0)
    return float(np.clip(1.0 - float(normalized), 0.0, 1.0))
