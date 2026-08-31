from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConditioningResult:
    probability: float
    effective_sample_size: float
    authority: bool
    level: int
    reason: str


def condition_mask(
    event_mask: np.ndarray,
    structural_state_pmf: np.ndarray,
    market_state_pmf: np.ndarray | None,
    *,
    identification_level: int,
    clip: tuple[float, float] = (0.25, 4.0),
    minimum_ess_fraction: float = 0.5,
) -> ConditioningResult:
    base = float(event_mask.astype(bool).mean())
    if identification_level < 2 or market_state_pmf is None:
        return ConditioningResult(base, float(len(event_mask)), False, identification_level, "MARKET_RESIDUAL_DIAGNOSTIC_ONLY")
    structural = np.asarray(structural_state_pmf, dtype=float)
    market = np.asarray(market_state_pmf, dtype=float)
    if structural.shape != market.shape or structural.shape != event_mask.shape:
        raise ValueError("market and structural state vectors must align with trajectories")
    raw = np.divide(market, structural, out=np.ones_like(market), where=structural > 1e-12)
    weights = np.clip(raw, clip[0], clip[1])
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    probability = float(np.average(event_mask.astype(float), weights=weights))
    authority = ess >= minimum_ess_fraction * len(weights)
    return ConditioningResult(probability, ess, authority, identification_level, "SUPPORTED" if authority else "EFFECTIVE_SAMPLE_SIZE_TOO_LOW")
