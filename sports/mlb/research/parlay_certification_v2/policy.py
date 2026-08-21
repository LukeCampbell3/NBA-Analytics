from __future__ import annotations

"""DECISION POLICY (mission section 1/12/13) -- the policy acts or
abstains; it is the ONLY thing that sets A_t. Predictive/world-layer code
(e.g. joint_position_builder_v2.pairs.CandidatePair) may rank candidates,
estimate joint probabilities, build compatible worlds, and propose
wagers -- it never sets `action` or any authorization flag itself; those
live exclusively here and in state_machine.py, driven only by
anytime_monitor.py's simultaneous certificate.

V2 initial deployment (section 13, frozen): max_actions_per_eligible_slate
= 1, two-leg parlays only. A day with no certified candidate is an
ABSTENTION (A_t=0), not a change to E_t -- eligibility.py never sees this
module's output.
"""

from dataclasses import dataclass

import numpy as np

from .eligibility import EligibilityDecision
from .evidence_store import DecisionRecord
from .settlement import reject_if_price_exceeds_bound
from .world_certificate import NonvacuousWorldCertificate, build_nonvacuous_world_certificate

POLICY_VERSION = "PARLAY_POLICY_V2_TWO_LEG_SINGLE_ACTION"
MAX_ACTIONS_PER_ELIGIBLE_SLATE = 1


@dataclass(frozen=True)
class CandidateWager:
    """A single proposed wager (e.g. a 2-leg pair) from the predictive/
    world layer. Carries only what the policy needs to certify and price
    it -- no probability/EV field is read directly by the policy; only the
    world-certificate machinery (built from retained_world_ids /
    world_probabilities / losing_world_ids) determines certification."""

    wager_id: str
    decimal_price: float | None
    retained_world_ids: np.ndarray
    world_probabilities: np.ndarray
    losing_world_ids: np.ndarray
    book: str | None = None


@dataclass(frozen=True)
class ActionSelection:
    action: int  # A_t
    selected: CandidateWager | None
    certificate: NonvacuousWorldCertificate | None
    reason: str


def select_action_for_day(
    eligibility: EligibilityDecision,
    candidates: list[CandidateWager],
    *,
    r_max: float,
) -> ActionSelection:
    """Only ever called for E_t=1 days -- E_t=0 days take no action by
    definition and are not a policy decision at all."""
    if not eligibility.eligible:
        raise ValueError("select_action_for_day must only be called for E_t=1 days")

    certified: list[tuple[CandidateWager, NonvacuousWorldCertificate]] = []
    for cand in candidates:
        if cand.decimal_price is None:
            # No real quote for this candidate: it is simply not actionable.
            # This does NOT touch eligibility (section: missing quote != E=0).
            continue
        try:
            reject_if_price_exceeds_bound(cand.decimal_price, r_max=r_max)
        except ValueError:
            continue  # price-bound violation rejects the CANDIDATE, not the day
        cert = build_nonvacuous_world_certificate(cand.retained_world_ids, cand.world_probabilities, cand.losing_world_ids)
        if cert.certified:
            certified.append((cand, cert))

    if not certified:
        return ActionSelection(action=0, selected=None, certificate=None, reason="no_certified_candidate")

    # Frozen tie-breaker (section 14): most retained probability mass
    # (most information retained under the calibrated world model), then
    # lexicographic wager_id for full determinism.
    certified.sort(key=lambda pair: (-pair[1].retained_probability_mass, pair[0].wager_id))
    selected, cert = certified[0]
    return ActionSelection(action=1, selected=selected, certificate=cert, reason="certified_candidate_selected")


def build_decision_record(
    *,
    date: str,
    eligibility: EligibilityDecision,
    decision_timestamp_utc: str,
    predictive_model_version: str,
    candidate_universe_size: int,
    action_selection: ActionSelection,
    c: float,
    r: float,
    delta: float,
    r_max: float,
) -> DecisionRecord:
    selected = action_selection.selected
    cert = action_selection.certificate
    return DecisionRecord(
        date=date,
        eligible=eligibility.eligible,
        eligibility_reason=eligibility.reason,
        eligibility_version=eligibility.eligibility_version,
        decision_timestamp_utc=decision_timestamp_utc,
        policy_version=POLICY_VERSION,
        predictive_model_version=predictive_model_version,
        candidate_universe_size=candidate_universe_size,
        action=action_selection.action,
        selected_wager=selected.wager_id if selected else None,
        accepted_decimal_price=selected.decimal_price if selected else None,
        accepted_book=selected.book if selected else None,
        c=c,
        r=r,
        delta=delta,
        r_max=r_max,
        world_certificate_diagnostics=(
            {
                "retained_world_count": cert.retained_world_count,
                "retained_probability_mass": cert.retained_probability_mass,
                "counterexample_count": cert.counterexample_count,
                "counterexample_mass": cert.counterexample_mass,
                "certified": cert.certified,
                "version": cert.version,
            }
            if cert is not None
            else None
        ),
    )
