from __future__ import annotations

import numpy as np
import pandas as pd


# Latent belief uncertainty is emitted on a std-like scale that often clusters
# around ~1.0, not as a probability. We map it into a softer [0, 1] penalty band
# so slightly-above-1 values do not collapse candidate confidence to zero.
BELIEF_UNCERTAINTY_LOWER = 0.75
BELIEF_UNCERTAINTY_UPPER = 1.15


def _numeric_or_default(values, default: float):
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").fillna(default).astype(float)
    numeric = pd.to_numeric(pd.Series([values]), errors="coerce").fillna(default).iloc[0]
    return float(numeric)


def normalize_belief_uncertainty(
    values,
    default: float = 1.0,
    lower: float = BELIEF_UNCERTAINTY_LOWER,
    upper: float = BELIEF_UNCERTAINTY_UPPER,
):
    numeric = _numeric_or_default(values, default=default)
    span = max(float(upper) - float(lower), 1e-6)
    if isinstance(numeric, pd.Series):
        return ((numeric - float(lower)) / span).clip(lower=0.0, upper=1.0)
    return float(np.clip((float(numeric) - float(lower)) / span, 0.0, 1.0))


def belief_confidence_factor(
    values,
    default: float = 1.0,
    lower: float = BELIEF_UNCERTAINTY_LOWER,
    upper: float = BELIEF_UNCERTAINTY_UPPER,
):
    normalized = normalize_belief_uncertainty(values, default=default, lower=lower, upper=upper)
    if isinstance(normalized, pd.Series):
        return (1.0 - normalized).clip(lower=0.0, upper=1.0)
    return float(np.clip(1.0 - float(normalized), 0.0, 1.0))
