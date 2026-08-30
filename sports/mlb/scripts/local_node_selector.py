#!/usr/bin/env python3
"""Leakage-safe shadow selector for recovering nodes when balanced P is weak.

This module is deliberately not wired into publication.  It evaluates the
hypothesis that a candidate can be rescued by a repeatable local error surface:
settled *prior* nodes with similar pregame balanced/market probabilities should
outperform the balanced probability estimate by a stable amount.

Selection order is conservative:
1. require enough prior local support;
2. require a positive local correction and market disagreement;
3. require the Wilson lower confidence bound to clear a configured floor;
4. among survivors, maximize price EV, then the conservative probability.

Candidate outcomes are never inputs to scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class LocalNodeScore:
    candidate_id: str
    neighbor_count: int
    local_hit_rate: float | None
    local_lcb: float | None
    local_correction: float | None
    recovered_probability: float | None
    expected_value: float | None
    eligible: bool
    reasons: tuple[str, ...]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def american_to_decimal(price: Any) -> float | None:
    number = _finite(price)
    if number is None or (-100.0 < number < 100.0):
        return None
    return 1.0 + (number / 100.0 if number > 0 else 100.0 / abs(number))


def wilson_lower_bound(wins: int, rows: int, z: float = 1.6448536269514722) -> float | None:
    """One-sided 95% Wilson lower bound."""
    if rows <= 0:
        return None
    p = wins / rows
    z2 = z * z
    denominator = 1.0 + z2 / rows
    center = p + z2 / (2.0 * rows)
    radius = z * math.sqrt(p * (1.0 - p) / rows + z2 / (4.0 * rows * rows))
    return max(0.0, (center - radius) / denominator)


def _pregame_vector(row: dict[str, Any]) -> tuple[float, float, float] | None:
    balanced = _finite(row.get("balanced_probability"))
    market = _finite(row.get("market_probability"))
    if balanced is None or market is None:
        return None
    return balanced, market, market - balanced


def _distance(candidate: dict[str, Any], history: dict[str, Any]) -> float | None:
    left = _pregame_vector(candidate)
    right = _pregame_vector(history)
    if left is None or right is None:
        return None
    # Fixed scales keep the metric frozen and interpretable.  A 5pp change in
    # either probability is one unit; disagreement receives the same scale.
    scales = (0.05, 0.05, 0.05)
    return math.sqrt(sum(((a - b) / scale) ** 2 for a, b, scale in zip(left, right, scales)))


def score_candidate(
    candidate: dict[str, Any],
    prior_history: Iterable[dict[str, Any]],
    *,
    k_neighbors: int = 40,
    min_neighbors: int = 20,
    min_market_disagreement: float = 0.03,
    min_local_correction: float = 0.02,
    min_lcb: float = 0.55,
) -> LocalNodeScore:
    candidate_id = str(candidate.get("candidate_id") or candidate.get("play_key") or "")
    balanced = _finite(candidate.get("balanced_probability"))
    market = _finite(candidate.get("market_probability"))
    reasons: list[str] = []
    if balanced is None or market is None:
        return LocalNodeScore(candidate_id, 0, None, None, None, None, None, False, ("probability_unavailable",))

    comparable: list[tuple[float, dict[str, Any]]] = []
    candidate_date = str(candidate.get("slate_date") or "")
    candidate_game = str(candidate.get("game_id") or "")
    for row in prior_history:
        # Hard leakage barrier: only strictly earlier settled slates.
        row_date = str(row.get("slate_date") or "")
        if not row_date or not candidate_date or row_date >= candidate_date:
            continue
        if row.get("win") not in (0, 1, False, True):
            continue
        # Avoid treating another proposition from the same game as independent
        # local evidence if malformed dates ever enter the caller's history.
        if candidate_game and row_date == candidate_date and str(row.get("game_id") or "") == candidate_game:
            continue
        distance = _distance(candidate, row)
        if distance is not None:
            comparable.append((distance, row))

    comparable.sort(key=lambda item: item[0])
    neighbors = comparable[: max(0, int(k_neighbors))]
    n = len(neighbors)
    if n < min_neighbors:
        reasons.append("insufficient_local_support")

    if n:
        wins = sum(int(bool(row.get("win"))) for _, row in neighbors)
        local_hit_rate = wins / n
        local_lcb = wilson_lower_bound(wins, n)
        # Local correction is measured against what balanced P predicted for
        # those same neighbors, not against the candidate's probability.
        neighbor_balanced = [float(row["balanced_probability"]) for _, row in neighbors]
        mean_balanced = sum(neighbor_balanced) / n
        local_correction = local_hit_rate - mean_balanced
        recovered = max(0.0, min(1.0, balanced + local_correction))
    else:
        local_hit_rate = local_lcb = local_correction = recovered = None

    disagreement = market - balanced
    if disagreement < min_market_disagreement:
        reasons.append("market_disagreement_too_small")
    if local_correction is None or local_correction < min_local_correction:
        reasons.append("local_correction_not_positive_enough")
    if local_lcb is None or local_lcb < min_lcb:
        reasons.append("local_lcb_below_floor")

    decimal = american_to_decimal(candidate.get("price"))
    expected_value = recovered * decimal - 1.0 if recovered is not None and decimal is not None else None
    if expected_value is None:
        reasons.append("price_unavailable")
    elif expected_value < 0.0:
        reasons.append("recovered_ev_negative")

    return LocalNodeScore(
        candidate_id=candidate_id,
        neighbor_count=n,
        local_hit_rate=local_hit_rate,
        local_lcb=local_lcb,
        local_correction=local_correction,
        recovered_probability=recovered,
        expected_value=expected_value,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def select_node(
    candidates: Iterable[dict[str, Any]],
    prior_history: Iterable[dict[str, Any]],
    **score_kwargs: Any,
) -> tuple[dict[str, Any] | None, list[LocalNodeScore]]:
    history = list(prior_history)
    pairs = [(candidate, score_candidate(candidate, history, **score_kwargs)) for candidate in candidates]
    eligible = [(candidate, score) for candidate, score in pairs if score.eligible]
    if not eligible:
        return None, [score for _, score in pairs]
    eligible.sort(
        key=lambda pair: (
            pair[1].expected_value if pair[1].expected_value is not None else -999.0,
            pair[1].local_lcb if pair[1].local_lcb is not None else -999.0,
            pair[1].recovered_probability if pair[1].recovered_probability is not None else -999.0,
        ),
        reverse=True,
    )
    return eligible[0][0], [score for _, score in pairs]
