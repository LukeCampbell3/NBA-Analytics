#!/usr/bin/env python3
"""
v12 Phase 2: the miss-budget constrained SafeEV optimizer
(premium_safe_ev_v12_shadow).

Replaces v11's top-N-by-selection-score board with a real, exact 0/1
selection that maximizes total expected value subject to a real "miss
budget" -- the expected number of losses the slate is allowed to carry
(sum of (1 - p_safe) across selected picks) -- plus the same
diversification caps (max per market bucket, max per team) v11's own
select_top_candidates() already enforces. Solved exactly via
scipy.optimize.milp (HiGHS branch-and-bound) -- a real integer-programming
solve, not a greedy top-EV-per-unit-risk approximation.

SHADOW ONLY: nothing here mutates a Candidate or touches v11's real
selection. This is a separate, disclosed research product compared
against v11 by compare_v11_v12_slates.py's real backtest harness under
the asymmetric promotion gate defined there.

Every candidate's probability/EV inputs default to its own real
safe_probability / safe_expected_value (v12's veto-adjusted numbers, see
pick_survival_model.apply_winner_signature_model) but fall back to v11's
own calibrated_hit_probability / expected_value_per_unit whenever the
winner-signature model is inactive for that row (disabled, cutoff
violation, insufficient segment support, or the model itself is still
insufficient_support -- its real status as of this writing). That makes
this optimizer real and runnable today even before the winner-signature
model has enough evidence to activate, and it starts using real veto
signal automatically the moment that model clears its own promotion bar
-- no separate wiring needed later.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

PRODUCT_VERSION = "premium_safe_ev_v12_shadow"
DEFAULT_MISS_BUDGET = 2.0
DEFAULT_MAX_PICKS = 10
DEFAULT_MAX_PER_MARKET_BUCKET = 2
DEFAULT_MAX_PER_TEAM = 2


def effective_probability(candidate: Any) -> float | None:
    """v12's own safe_probability when real for this row, else a real
    fallback to v11's own calibrated probability. Never fabricated, and
    never higher than v11's own number -- safe_probability is already
    min-clamped at its source (apply_winner_signature_model)."""
    safe = getattr(candidate, "safe_probability", None)
    if safe is not None:
        return float(safe)
    calibrated = getattr(candidate, "calibrated_hit_probability", None)
    return float(calibrated) if calibrated is not None else None


def effective_expected_value(candidate: Any) -> float | None:
    safe_ev = getattr(candidate, "safe_expected_value", None)
    if safe_ev is not None:
        return float(safe_ev)
    ev = getattr(candidate, "expected_value_per_unit", None)
    return float(ev) if ev is not None else None


def optimize_slate(
    candidates: Sequence[Any],
    *,
    miss_budget: float = DEFAULT_MISS_BUDGET,
    max_picks: int = DEFAULT_MAX_PICKS,
    max_per_market_bucket: int = DEFAULT_MAX_PER_MARKET_BUCKET,
    max_per_team: int = DEFAULT_MAX_PER_TEAM,
) -> dict[str, Any]:
    """Exact 0/1 selection maximizing sum(EV_safe) subject to
    sum(1 - p_safe) <= miss_budget, a total-picks cap, and the same
    per-market-bucket / per-team diversification caps v11 enforces.
    Candidates missing a real probability or EV (either the safe_* fields
    or their v11 fallback) are excluded from consideration -- never
    guessed. Returns the chosen Candidate objects, never mutates them."""
    usable: list[tuple[Any, float, float]] = []
    for candidate in candidates:
        probability = effective_probability(candidate)
        expected_value = effective_expected_value(candidate)
        if probability is None or expected_value is None:
            continue
        usable.append((candidate, probability, expected_value))

    if not usable:
        return {
            "status": "no_usable_candidates",
            "selected": [],
            "candidates_considered": len(candidates),
            "candidates_usable": 0,
        }

    n = len(usable)
    objective = np.array([-expected_value for _, _, expected_value in usable])  # milp minimizes
    miss_weights = np.array([1.0 - probability for _, probability, _ in usable])

    constraints = [LinearConstraint(miss_weights.reshape(1, -1), -np.inf, float(miss_budget))]
    if max_picks > 0:
        constraints.append(LinearConstraint(np.ones((1, n)), -np.inf, float(max_picks)))

    buckets: dict[str, list[int]] = {}
    teams: dict[str, list[int]] = {}
    for index, (candidate, _, _) in enumerate(usable):
        buckets.setdefault(str(getattr(candidate, "market_bucket", "")), []).append(index)
        teams.setdefault(str(getattr(candidate, "team", "")), []).append(index)

    if max_per_market_bucket > 0:
        for indices in buckets.values():
            if len(indices) <= max_per_market_bucket:
                continue
            row = np.zeros(n)
            row[indices] = 1.0
            constraints.append(LinearConstraint(row.reshape(1, -1), -np.inf, float(max_per_market_bucket)))
    if max_per_team > 0:
        for indices in teams.values():
            if len(indices) <= max_per_team:
                continue
            row = np.zeros(n)
            row[indices] = 1.0
            constraints.append(LinearConstraint(row.reshape(1, -1), -np.inf, float(max_per_team)))

    result = milp(objective, constraints=constraints, integrality=np.ones(n), bounds=Bounds(0, 1))

    if not result.success:
        return {
            "status": f"solver_failed:{result.message}",
            "selected": [],
            "candidates_considered": len(candidates),
            "candidates_usable": n,
        }

    chosen = [usable[index][0] for index, value in enumerate(result.x) if value > 0.5]
    return {
        "status": "optimal",
        "selected": chosen,
        "candidates_considered": len(candidates),
        "candidates_usable": n,
        "miss_budget": float(miss_budget),
        "miss_budget_used": float(sum(1.0 - effective_probability(candidate) for candidate in chosen)),
        "expected_value_total": float(sum(effective_expected_value(candidate) for candidate in chosen)),
    }
