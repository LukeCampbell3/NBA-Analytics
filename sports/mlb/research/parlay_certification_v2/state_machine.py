from __future__ import annotations

"""Reversible support state machine (mission section 9):

    DEVELOPMENT
        -> FROZEN_PROSPECTIVE_INCONCLUSIVE
        -> FROZEN_POLICY_PROSPECTIVELY_SUPPORTED
        -> SUPPORTED_CURRENT
        -> PRODUCTION_DEMOTED  (if bounds later fail)

Certification is NOT permanent: the monitor keeps updating after first
support, and a later failure of the simultaneous bounds demotes. A
historical certificate remains an audit fact about the horizon at which it
held -- it does not guarantee future distribution stability, and this
module never automatically re-promotes a demoted or development policy by
re-tuning the same version (a parameter change requires a NEW policy
version and a new prospective stream -- see manifest.py).
"""

from dataclasses import dataclass
from enum import Enum

STATE_MACHINE_VERSION = "PARLAY_CERTIFICATION_STATE_V1"


class PolicyStatus(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    FROZEN_PROSPECTIVE_INCONCLUSIVE = "FROZEN_PROSPECTIVE_INCONCLUSIVE"
    FROZEN_POLICY_PROSPECTIVELY_SUPPORTED = "FROZEN_POLICY_PROSPECTIVELY_SUPPORTED"
    SUPPORTED_CURRENT = "SUPPORTED_CURRENT"
    PRODUCTION_DEMOTED = "PRODUCTION_DEMOTED"


@dataclass(frozen=True)
class StateTransition:
    previous: PolicyStatus
    next: PolicyStatus
    reason: str
    t: int


_ALLOWED_TRANSITIONS: dict[PolicyStatus, set[PolicyStatus]] = {
    PolicyStatus.DEVELOPMENT: {PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE},
    PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE: {
        PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE,
        PolicyStatus.FROZEN_POLICY_PROSPECTIVELY_SUPPORTED,
    },
    PolicyStatus.FROZEN_POLICY_PROSPECTIVELY_SUPPORTED: {PolicyStatus.SUPPORTED_CURRENT},
    PolicyStatus.SUPPORTED_CURRENT: {PolicyStatus.SUPPORTED_CURRENT, PolicyStatus.PRODUCTION_DEMOTED},
    # Terminal for THIS policy version. A fix requires a new policy version.
    PolicyStatus.PRODUCTION_DEMOTED: {PolicyStatus.PRODUCTION_DEMOTED},
}


def next_status(current: PolicyStatus, *, fully_supported: bool, t: int) -> StateTransition:
    """One monotone step of the reversible state machine, driven only by
    this horizon's SimultaneousCertificate.fully_supported. Raises if an
    illegal transition would result (defensive; the branches below only
    ever propose allowed transitions)."""
    if current == PolicyStatus.DEVELOPMENT:
        nxt, reason = PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE, "policy frozen for prospective evaluation"
    elif current == PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE:
        if fully_supported:
            nxt, reason = PolicyStatus.FROZEN_POLICY_PROSPECTIVELY_SUPPORTED, "all three simultaneous bounds satisfied at this horizon"
        else:
            nxt, reason = PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE, "bounds not yet simultaneously satisfied"
    elif current == PolicyStatus.FROZEN_POLICY_PROSPECTIVELY_SUPPORTED:
        nxt, reason = PolicyStatus.SUPPORTED_CURRENT, "first support recorded as an audit fact; continuing to monitor as SUPPORTED_CURRENT"
    elif current == PolicyStatus.SUPPORTED_CURRENT:
        if fully_supported:
            nxt, reason = PolicyStatus.SUPPORTED_CURRENT, "bounds remain simultaneously satisfied"
        else:
            nxt, reason = PolicyStatus.PRODUCTION_DEMOTED, "one or more bounds no longer simultaneously satisfied"
    else:
        nxt, reason = PolicyStatus.PRODUCTION_DEMOTED, "policy demoted; a new policy version is required to re-attempt support"

    if nxt not in _ALLOWED_TRANSITIONS[current]:
        raise RuntimeError(f"illegal transition {current} -> {nxt}")
    return StateTransition(previous=current, next=nxt, reason=reason, t=t)
