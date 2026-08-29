from __future__ import annotations

"""Quality-first shadow selection for MLB same-game combinations.

The joint simulator, market provider, calibration store, and authorization
rules remain unchanged. This module only decides which already-built shadow
candidate can become the public headline.

A headline candidate must now clear all three distinct tests before ranking:

    joint hit probability >= 50%
    model edge vs no-vig joint market >= 3 percentage points
    synthetic-price model EV >= 5%

The last value is explicitly synthetic until a real combined FanDuel SGP quote
is captured. These are deliberately conservative SHADOW presentation gates,
not a retrospectively certified production rule.
"""

from typing import Iterable

from select_mlb_same_game_bets import MIN_COMBO_JOINT_PROBABILITY, SameGameComboCandidate


DEFAULT_MAX_HEADLINE_CANDIDATES = 10
DEFAULT_MAX_EXPLORATORY_CANDIDATES = 10
MIN_HEADLINE_PROBABILITY_EDGE = 0.03
MIN_HEADLINE_EXPECTED_VALUE = 0.05


def _finite_value(value, default: float = float("-inf")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and number not in {float("inf"), float("-inf")} else default


def quality_safe_candidates(
    candidates: Iterable[SameGameComboCandidate],
    *,
    min_joint_probability: float = MIN_COMBO_JOINT_PROBABILITY,
    min_probability_edge: float = MIN_HEADLINE_PROBABILITY_EDGE,
    min_expected_value: float = MIN_HEADLINE_EXPECTED_VALUE,
    max_candidates: int = DEFAULT_MAX_HEADLINE_CANDIDATES,
) -> list[SameGameComboCandidate]:
    """Return only combinations suitable for the tighter headline pool."""

    survivors: list[SameGameComboCandidate] = []
    for candidate in candidates:
        joint = _finite_value(candidate.real_joint_model_probability)
        edge = _finite_value(candidate.probability_edge)
        ev = _finite_value(candidate.expected_value_per_unit)
        if joint < min_joint_probability:
            continue
        if edge < min_probability_edge:
            continue
        if ev < min_expected_value:
            continue
        survivors.append(candidate)

    # Probability safety is a gate, not the ranking objective. Once all three
    # thresholds pass, use price efficiency first, then reliability and edge.
    survivors.sort(
        key=lambda candidate: (
            _finite_value(candidate.expected_value_per_unit),
            _finite_value(candidate.real_joint_model_probability),
            _finite_value(candidate.probability_edge),
        ),
        reverse=True,
    )
    return survivors[:max_candidates]


def exploratory_ev_candidates(
    candidates: Iterable[SameGameComboCandidate],
    *,
    min_joint_probability: float = MIN_COMBO_JOINT_PROBABILITY,
    max_candidates: int = DEFAULT_MAX_EXPLORATORY_CANDIDATES,
) -> list[SameGameComboCandidate]:
    """Keep rejected positive-EV combinations for research, never headline."""

    exploratory = [
        candidate
        for candidate in candidates
        if candidate.expected_value_per_unit is not None
        and _finite_value(candidate.expected_value_per_unit) > 0.0
        and (
            _finite_value(candidate.real_joint_model_probability) < min_joint_probability
            or _finite_value(candidate.probability_edge) < MIN_HEADLINE_PROBABILITY_EDGE
            or _finite_value(candidate.expected_value_per_unit) < MIN_HEADLINE_EXPECTED_VALUE
        )
    ]
    exploratory.sort(
        key=lambda candidate: (
            _finite_value(candidate.expected_value_per_unit),
            _finite_value(candidate.real_joint_model_probability),
        ),
        reverse=True,
    )
    return exploratory[:max_candidates]
