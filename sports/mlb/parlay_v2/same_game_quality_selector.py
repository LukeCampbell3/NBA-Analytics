from __future__ import annotations

"""Quality-first shadow selection for MLB same-game combinations.

This module deliberately does not change the joint game simulator, market
pricing, calibration store, or authorization rules in
``select_mlb_same_game_bets``.  It only controls which already-built shadow
candidate is promoted to the public headline.

The policy is selective-prediction shaped:

    joint hit probability floor -> positive edge/EV -> maximize EV

A low-probability/high-EV combination remains useful research evidence, but it
must not displace a probability-safe combination on the headline card.  If no
combination clears the joint floor, the headline pool is empty (ABSTAIN) and
the EV-only candidates remain available as exploratory diagnostics.
"""

from typing import Iterable

from select_mlb_same_game_bets import MIN_COMBO_JOINT_PROBABILITY, SameGameComboCandidate


DEFAULT_MAX_HEADLINE_CANDIDATES = 10
DEFAULT_MAX_EXPLORATORY_CANDIDATES = 10


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
    require_positive_edge: bool = True,
    require_positive_ev: bool = True,
    max_candidates: int = DEFAULT_MAX_HEADLINE_CANDIDATES,
) -> list[SameGameComboCandidate]:
    """Return only combinations suitable for the headline shadow pool.

    The probability constraint is applied *before* EV ranking.  This is the
    key distinction from the legacy display selector, which ranked every
    priced combination by EV and could headline a candidate that the policy's
    own 50% joint-probability gate would never authorize.
    """

    survivors: list[SameGameComboCandidate] = []
    for candidate in candidates:
        joint = _finite_value(candidate.real_joint_model_probability)
        edge = _finite_value(candidate.probability_edge)
        ev = _finite_value(candidate.expected_value_per_unit)
        if joint < min_joint_probability:
            continue
        if require_positive_edge and edge <= 0.0:
            continue
        if require_positive_ev and ev <= 0.0:
            continue
        survivors.append(candidate)

    # Once the hit-rate floor is satisfied, optimize price efficiency.
    # Joint probability is the first tie-break so two equal-EV choices prefer
    # the more reliable one, followed by model-vs-market probability edge.
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
    """Keep low-joint/high-EV combinations for research, never headline.

    This preserves the information value of combinations such as a 25% joint
    event with a large modeled edge without presenting that event as the
    system's best same-game parlay.
    """

    exploratory = [
        candidate
        for candidate in candidates
        if candidate.expected_value_per_unit is not None
        and _finite_value(candidate.real_joint_model_probability) < min_joint_probability
    ]
    exploratory.sort(
        key=lambda candidate: (
            _finite_value(candidate.expected_value_per_unit),
            _finite_value(candidate.real_joint_model_probability),
        ),
        reverse=True,
    )
    return exploratory[:max_candidates]
