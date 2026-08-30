from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import simulate_local_node_selector_v2 as simulation  # noqa: E402


def test_boundary_world_is_not_treated_as_power() -> None:
    result = simulation.simulate_world(
        rng=np.random.default_rng(7),
        trials=20_000,
        slates=15,
        rows_per_slate=20,
        true_residual=0.02,
        slate_shock_sd=0.08,
        confidence=0.975,
    )
    assert result.acceptance_rate < 0.04
    assert result.lcb_coverage > 0.96


def test_more_rows_and_slates_reduce_required_sample() -> None:
    sparse = simulation.approximate_required_slates(
        true_residual=0.10,
        practical_residual_floor=0.02,
        rows_per_slate=5,
        slate_shock_sd=0.08,
        confidence=0.975,
    )
    dense = simulation.approximate_required_slates(
        true_residual=0.10,
        practical_residual_floor=0.02,
        rows_per_slate=20,
        slate_shock_sd=0.08,
        confidence=0.975,
    )
    assert sparse is not None and dense is not None
    assert dense < sparse


def test_no_finite_sample_claim_at_or_below_practical_floor() -> None:
    assert simulation.approximate_required_slates(
        true_residual=0.02,
        practical_residual_floor=0.02,
        rows_per_slate=20,
        slate_shock_sd=0.08,
        confidence=0.975,
    ) is None

