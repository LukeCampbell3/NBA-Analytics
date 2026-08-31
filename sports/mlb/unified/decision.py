from __future__ import annotations

from dataclasses import dataclass

from .schemas import BetCandidate


def american_to_decimal(price: float | int | None) -> float | None:
    if price is None:
        return None
    price = float(price)
    if -100.0 < price < 100.0:
        return None
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


@dataclass(frozen=True)
class DecisionPolicy:
    minimum_usable_probability: float = 0.60
    minimum_probability_edge: float = 0.01
    minimum_conservative_ev: float = 0.0
    uncertainty_multiplier: float = 1.0
    valid_support_states: frozenset[str] = frozenset({"SUPPORTED", "IN_SUPPORT"})
    valid_lineup_states: frozenset[str] = frozenset({"CONFIRMED", "NOT_APPLICABLE"})
    valid_role_states: frozenset[str] = frozenset({"CONFIRMED", "NOT_APPLICABLE"})
    valid_identity_states: frozenset[str] = frozenset({"CONFIRMED"})
    require_exact_selection_ids: bool = False


def decide(candidate: BetCandidate, policy: DecisionPolicy) -> BetCandidate:
    reasons: list[str] = []
    decimal_price = candidate.decimal_price or american_to_decimal(candidate.american_price)
    base_probability = (
        candidate.calibrated_probability
        if candidate.calibrated_probability is not None
        else candidate.raw_probability
    )
    if base_probability is None:
        reasons.append("PROBABILITY_UNAVAILABLE")
    if decimal_price is None:
        reasons.append("PRICE_UNAVAILABLE")
    uncertainty = candidate.uncertainty
    if uncertainty is None:
        reasons.append("UNCERTAINTY_UNAVAILABLE")
    usable = None if base_probability is None or uncertainty is None else max(
        0.0, min(1.0, base_probability - policy.uncertainty_multiplier * max(0.0, uncertainty))
    )
    break_even = None if decimal_price is None else 1.0 / decimal_price
    edge = None if usable is None or break_even is None else usable - break_even
    ev = None if usable is None or decimal_price is None else usable * decimal_price - 1.0

    if usable is not None and usable < policy.minimum_usable_probability:
        reasons.append("USABLE_PROBABILITY_BELOW_FLOOR")
    if edge is not None and edge < policy.minimum_probability_edge:
        reasons.append("PROBABILITY_EDGE_BELOW_FLOOR")
    if ev is not None and ev <= policy.minimum_conservative_ev:
        reasons.append("NON_POSITIVE_CONSERVATIVE_EV")
    if candidate.support_status not in policy.valid_support_states:
        reasons.append("SUPPORT_INVALID")
    if candidate.lineup_status not in policy.valid_lineup_states:
        reasons.append("LINEUP_INVALID")
    if candidate.role_status not in policy.valid_role_states:
        reasons.append("ROLE_INVALID")
    if candidate.identity_status not in policy.valid_identity_states:
        reasons.append("IDENTITY_INVALID")
    if policy.require_exact_selection_ids and (not candidate.sportsbook_market_id or not candidate.sportsbook_selection_id):
        reasons.append("EXACT_SELECTION_UNAVAILABLE")

    candidate.decimal_price = decimal_price
    candidate.usable_probability = usable
    candidate.market_break_even_probability = break_even
    candidate.probability_edge = edge
    candidate.expected_value = ev
    candidate.conservative_expected_value = ev
    candidate.rejection_reasons = sorted(set(reasons))
    candidate.publication_authority = False
    return candidate


def select(candidates: list[BetCandidate], policy: DecisionPolicy) -> tuple[list[BetCandidate], list[BetCandidate]]:
    evaluated = [decide(candidate, policy) for candidate in candidates]
    accepted = [candidate for candidate in evaluated if not candidate.rejection_reasons]
    rejected = [candidate for candidate in evaluated if candidate.rejection_reasons]
    accepted.sort(
        key=lambda candidate: (
            candidate.conservative_expected_value or float("-inf"),
            candidate.usable_probability or float("-inf"),
        ),
        reverse=True,
    )
    return accepted, rejected
