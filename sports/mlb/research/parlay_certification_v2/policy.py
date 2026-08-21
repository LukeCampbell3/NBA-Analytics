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

WORLD_GATE_MODE (mission: "Resolve the remaining PARLAY_V2 APS /
counterexample admission bottleneck"). `select_action_for_day` decides how
world/counterexample information participates in ADMISSION -- never how
G_C/G_L/G_V or the outer prospective anytime certificate work (those live
in anytime_monitor.py/state_machine.py, untouched by this parameter):

    REQUIRED (default -- byte-identical to this function's pre-existing
        behavior, and what PARLAY_POLICY_V2_PROSPECTIVE_002 uses):
        only NONVACUOUS_WORLD_CERTIFICATE candidates (B_S(C)=empty) are
        admitted. Empirically degenerate at the frozen APS_THRESHOLD=1.0
        (see world_gate_research.py: 0% of real DEVELOPMENT pairs achieve
        this) -- kept only for REQUIRED-mode callers/replay compatibility.
    BOUNDED_RISK: admits a candidate when its world_risk_rho (the
        outside-mass-protected quantity -- see world_certificate.py) is
        <= world_risk_threshold (required for this mode). Implemented and
        tested per mission section 22.D, but NOT what any currently
        frozen policy version activates -- world_gate_research.py's
        DEVELOPMENT evidence did not support a frozen threshold (see its
        report).
    OBSERVE_ONLY: world/counterexample information can never block
        admission. Candidates are ranked by ascending world_risk_rho (a
        DEVELOPMENT-validated, chronologically-replicated ranking
        diagnostic -- see world_gate_research.py -- NOT
        retained_probability_mass, which is a constant 1.0 at the frozen
        APS_THRESHOLD and carries no ranking information), then
        lexicographic wager_id. This is what
        PARLAY_POLICY_V2_PROSPECTIVE_003 activates.

Whichever mode is used, `action=1` NEVER implies real-money staking is
authorized -- that determination is made entirely outside this module
(run_parlay_v2.py's staking_authorized, always False for selection alone)
and is untouched by this parameter.
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


_VALID_WORLD_GATE_MODES = ("REQUIRED", "BOUNDED_RISK", "OBSERVE_ONLY")


def select_action_for_day(
    eligibility: EligibilityDecision,
    candidates: list[CandidateWager],
    *,
    r_max: float,
    world_gate_mode: str = "REQUIRED",
    world_risk_threshold: float | None = None,
) -> ActionSelection:
    """Only ever called for E_t=1 days -- E_t=0 days take no action by
    definition and are not a policy decision at all.

    world_gate_mode defaults to "REQUIRED", reproducing this function's
    original behavior exactly for any existing caller that does not pass
    it -- see module docstring's WORLD_GATE_MODE section."""
    if not eligibility.eligible:
        raise ValueError("select_action_for_day must only be called for E_t=1 days")
    if world_gate_mode not in _VALID_WORLD_GATE_MODES:
        raise ValueError(f"world_gate_mode must be one of {_VALID_WORLD_GATE_MODES}, got {world_gate_mode!r}")
    if world_gate_mode == "BOUNDED_RISK" and world_risk_threshold is None:
        raise ValueError("world_gate_mode='BOUNDED_RISK' requires world_risk_threshold")

    admitted: list[tuple[CandidateWager, NonvacuousWorldCertificate]] = []
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
        if world_gate_mode == "REQUIRED":
            if cert.certified:
                admitted.append((cand, cert))
        elif world_gate_mode == "BOUNDED_RISK":
            if cert.nonempty and cert.positive_mass and cert.world_risk_rho <= world_risk_threshold + 1e-12:
                admitted.append((cand, cert))
        else:  # OBSERVE_ONLY -- world/counterexample information can never block admission
            if cert.nonempty and cert.positive_mass:
                admitted.append((cand, cert))

    if not admitted:
        reason = "no_certified_candidate" if world_gate_mode == "REQUIRED" else "no_admissible_candidate"
        return ActionSelection(action=0, selected=None, certificate=None, reason=reason)

    if world_gate_mode == "REQUIRED":
        # Frozen tie-breaker (section 14, unchanged): most retained
        # probability mass, then lexicographic wager_id.
        admitted.sort(key=lambda pair: (-pair[1].retained_probability_mass, pair[0].wager_id))
        reason = "certified_candidate_selected"
    else:
        # Ranking diagnostic validated by world_gate_research.py
        # (chronological DERIVE->SELECT replication, day-clustered):
        # ascending world_risk_rho, i.e. lowest predicted joint-failure
        # risk first, then lexicographic wager_id for full determinism.
        admitted.sort(key=lambda pair: (pair[1].world_risk_rho, pair[0].wager_id))
        reason = "admitted_candidate_selected"
    selected, cert = admitted[0]
    return ActionSelection(action=1, selected=selected, certificate=cert, reason=reason)


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
    world_gate_mode: str = "REQUIRED",
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
        world_gate_mode=world_gate_mode,
        world_certificate_diagnostics=(
            {
                "retained_world_count": cert.retained_world_count,
                "retained_probability_mass": cert.retained_probability_mass,
                "counterexample_count": cert.counterexample_count,
                "counterexample_mass": cert.counterexample_mass,
                "outside_probability_mass": cert.outside_probability_mass,
                "world_risk_rho": cert.world_risk_rho,
                "certified": cert.certified,
                "version": cert.version,
            }
            if cert is not None
            else None
        ),
    )
