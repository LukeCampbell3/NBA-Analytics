from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

from .schemas import BetCandidate, EvidenceState, Ticket
from .trajectory import TrajectoryBatch


@dataclass(frozen=True)
class TicketPolicy:
    leg_count: int
    minimum_leg_decimal_price: float = 1.20
    minimum_leg_probability: float = 0.60
    minimum_leg_ev: float = 0.0
    minimum_joint_probability: float = 0.45
    minimum_combined_decimal_price: float = 1.5
    minimum_ticket_ev: float = 0.0
    top_k: int = 20


DEFAULT_TICKET_POLICIES = {
    2: TicketPolicy(2, minimum_joint_probability=0.45),
    3: TicketPolicy(3, minimum_joint_probability=0.30),
    4: TicketPolicy(4, minimum_joint_probability=0.20),
}


def _semantic_key(candidate: BetCandidate) -> tuple:
    return (candidate.game_id, candidate.subject_id, candidate.market_type, candidate.period, candidate.side, candidate.line)


def prune_safe_candidates(candidates: list[BetCandidate], policy: TicketPolicy) -> tuple[list[BetCandidate], dict[str, int]]:
    counts = {"input": len(candidates), "invalid": 0, "price_trap": 0, "duplicate": 0, "top_k": 0}
    best: dict[tuple, BetCandidate] = {}
    for candidate in candidates:
        if candidate.rejection_reasons or candidate.usable_probability is None or candidate.conservative_expected_value is None:
            counts["invalid"] += 1
            continue
        if candidate.decimal_price is None or candidate.decimal_price < policy.minimum_leg_decimal_price:
            counts["price_trap"] += 1
            continue
        if candidate.usable_probability < policy.minimum_leg_probability or candidate.conservative_expected_value <= policy.minimum_leg_ev:
            counts["invalid"] += 1
            continue
        key = _semantic_key(candidate)
        prior = best.get(key)
        if prior is not None:
            counts["duplicate"] += 1
        if prior is None or (candidate.conservative_expected_value, candidate.usable_probability) > (prior.conservative_expected_value, prior.usable_probability):
            best[key] = candidate
    safe = sorted(best.values(), key=lambda c: (c.conservative_expected_value, c.usable_probability), reverse=True)
    if len(safe) > policy.top_k:
        counts["top_k"] = len(safe) - policy.top_k
        safe = safe[: policy.top_k]
    counts["retained"] = len(safe)
    return safe, counts


def _contradictory(legs: tuple[BetCandidate, ...]) -> bool:
    keys = {(leg.game_id, leg.subject_id, leg.market_type, leg.period, leg.line) for leg in legs}
    if len(keys) != len(legs):
        return True
    directions: dict[tuple, set[str]] = {}
    for leg in legs:
        base = (leg.game_id, leg.subject_id, leg.market_type, leg.period, leg.line)
        directions.setdefault(base, set()).add(leg.side.lower())
    return any(len(sides) > 1 for sides in directions.values())


def _ticket_id(legs: tuple[BetCandidate, ...]) -> str:
    return hashlib.sha256("|".join(sorted(leg.candidate_id for leg in legs)).encode()).hexdigest()[:24]


def _build_ticket(legs: tuple[BetCandidate, ...], policy: TicketPolicy, trajectories: dict[str, TrajectoryBatch] | None) -> Ticket:
    independent = math.prod(float(leg.usable_probability) for leg in legs)
    games = {leg.game_id for leg in legs}
    reasons: list[str] = []
    joint = independent
    ticket_type = "cross_game"
    evidence = EvidenceState.PROSPECTIVE_SHADOW
    if len(games) == 1:
        ticket_type = "same_game"
        game_id = next(iter(games))
        refs = [leg.trajectory_mask_reference for leg in legs]
        if not trajectories or game_id not in trajectories or any(not ref for ref in refs):
            reasons.append("COMMON_WORLD_MASK_REQUIRED")
        else:
            joint = trajectories[game_id].joint_probability([str(ref) for ref in refs])
    joint = min(joint, *(float(leg.usable_probability) for leg in legs))
    combined = math.prod(float(leg.decimal_price) for leg in legs)
    break_even = 1.0 / combined
    edge = joint - break_even
    ev = joint * combined - 1.0
    if joint < policy.minimum_joint_probability:
        reasons.append("JOINT_PROBABILITY_BELOW_FLOOR")
    if combined < policy.minimum_combined_decimal_price:
        reasons.append("COMBINED_PRICE_BELOW_FLOOR")
    if ev <= policy.minimum_ticket_ev:
        reasons.append("NON_POSITIVE_TICKET_EV")
    exact = all(leg.sportsbook_market_id and leg.sportsbook_selection_id and leg.sportsbook for leg in legs)
    if not exact:
        reasons.append("EXACT_COMBINED_BETSLIP_UNAVAILABLE")
    return Ticket(
        ticket_id=_ticket_id(legs), ticket_type=ticket_type, leg_count=len(legs), legs=list(legs),
        combined_decimal_price=combined, independent_product_probability=independent,
        joint_probability=joint, dependency_delta=joint - independent,
        break_even_probability=break_even, probability_edge=edge, conservative_expected_value=ev,
        evidence_state=evidence, publication_authority=False,
        sportsbook=legs[0].sportsbook if len({leg.sportsbook for leg in legs}) == 1 else None,
        betslip_url=None, rejection_reasons=sorted(set(reasons)),
    )


def construct_ticket_class(
    candidates: list[BetCandidate], policy: TicketPolicy, *, trajectories: dict[str, TrajectoryBatch] | None = None
) -> tuple[list[Ticket], dict[str, int]]:
    safe, counts = prune_safe_candidates(candidates, policy)
    tickets = []
    enumerated = 0
    for legs in itertools.combinations(safe, policy.leg_count):
        enumerated += 1
        if _contradictory(legs):
            continue
        ticket = _build_ticket(legs, policy, trajectories)
        if not ticket.rejection_reasons:
            tickets.append(ticket)
    tickets.sort(key=lambda t: (t.conservative_expected_value or -math.inf, t.joint_probability), reverse=True)
    counts["enumerated"] = enumerated
    counts["qualified"] = len(tickets)
    return tickets, counts


def construct_all_ticket_classes(candidates: list[BetCandidate], *, trajectories: dict[str, TrajectoryBatch] | None = None):
    return {leg_count: construct_ticket_class(candidates, policy, trajectories=trajectories) for leg_count, policy in DEFAULT_TICKET_POLICIES.items()}
