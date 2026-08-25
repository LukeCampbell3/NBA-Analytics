"""DRM COMMIT decision function (spec section 28): a mutation becomes
permanent only if it improves a complexity-aware objective evaluated on
SELECT -- never TEST."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComplexityWeights:
    lambda_complexity: float = 1e-8  # per active parameter
    lambda_compute: float = 0.01  # per normalized compute unit (active_params / baseline_active_params)


def compute_J(validation_loss: float, active_params: int, baseline_active_params: int, weights: ComplexityWeights) -> float:
    normalized_compute = active_params / max(baseline_active_params, 1)
    return validation_loss + weights.lambda_complexity * active_params + weights.lambda_compute * normalized_compute


def decide_commit(j_before: float, j_after: float, min_improvement: float = 1e-4) -> tuple[bool, str]:
    delta = j_before - j_after
    if delta > min_improvement:
        return True, f"J improved by {delta:.6f} (> min_improvement={min_improvement})"
    return False, f"J did not improve enough (delta={delta:.6f}, min_improvement={min_improvement})"
