from __future__ import annotations

"""Ranking baselines audited against the fitted H_OVER_RANKER_V1 candidate.

`probability_score` is the same Poisson-tail-marginalized-over-model-error
formula from the session's calibration-fix backtest (see
sports/mlb/scripts/select_high_precision_predictions.py's
estimate_count_hit_probabilities for the original uncorrected version this
extends). It is a *generic* count-market probability estimator, not
H-OVER-specific, and its job here is only to serve as the incumbent
ranking baseline to beat.
"""

import math

import numpy as np
import pandas as pd

_Z_GRID = np.linspace(-3.5, 3.5, 15)
_Z_WEIGHTS = np.exp(-0.5 * _Z_GRID**2)
_Z_WEIGHTS = _Z_WEIGHTS / _Z_WEIGHTS.sum()


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    lam = max(0.0, float(lam))
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp((-lam) + (k * math.log(lam)) - math.lgamma(k + 1))


def _poisson_cdf(k: int, lam: float) -> float:
    return min(1.0, sum(_poisson_pmf(i, lam) for i in range(k + 1)))


def _over_probability_given_lambda(lam: float, market_line: float) -> float:
    rounded = round(market_line)
    if abs(market_line - rounded) < 1e-9:
        return max(0.0, min(1.0, 1.0 - _poisson_cdf(int(rounded), lam)))
    floor_line = math.floor(market_line)
    return max(0.0, min(1.0, 1.0 - _poisson_cdf(int(floor_line), lam)))


def probability_score(corrected_prediction: float, market_line: float, rmse: float) -> float:
    """Poisson-OVER-tail probability, marginalized over lambda ~ N(pred, rmse)."""
    rmse = max(float(rmse), 1e-6)
    lams = np.clip(corrected_prediction + _Z_GRID * rmse, 0.0, None)
    probs = np.array([_over_probability_given_lambda(float(lam), market_line) for lam in lams])
    return float(np.clip(np.sum(probs * _Z_WEIGHTS), 0.0, 1.0))


def add_baseline_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the three audited baseline scores as new columns (no fitting)."""
    out = rows.copy()
    out["score_probability"] = [
        probability_score(p, m, r) for p, m, r in zip(out["corrected_prediction"], out["market_line"], out["rmse"])
    ]
    out["score_raw_edge"] = out["corrected_edge"]
    out["score_q_edge_over_rmse"] = out["corrected_edge"] / out["rmse"].clip(lower=1e-6)
    return out
