from __future__ import annotations

"""Deterministic replay for the POLICY EVIDENCE stream (mission section
11B). Given a frozen policy's immutable evidence rows (from
evidence_store.EvidenceStore), reproduce the cumulative G_C/G_L/G_V
processes, their anytime bounds, PolicyStatus transitions, and the
first-support / demotion horizons if any.

Never refits the predictive system, never alters existing evidence rows,
and never reads a calibration observation admitted on/after the slate
being replayed (that invariant is enforced upstream by
parlay_v2.calibration.store.CalibrationStore -- this module only consumes
already-frozen decision/evidence records, which by construction already
respected it when they were written).
"""

from dataclasses import dataclass

from .anytime_monitor import (
    AlphaAllocation,
    SimultaneousCertificate,
    default_equal_split,
    evaluate_simultaneous_certificate,
    g_c_value,
    g_l_value,
    g_v_value,
)
from .state_machine import PolicyStatus, StateTransition, next_status

REPLAY_VERSION = "PARLAY_CERTIFICATION_REPLAY_V1"


@dataclass(frozen=True)
class PolicyReplayResult:
    n: int
    g_c_values: list[float]
    g_l_values: list[float]
    g_v_values: list[float]
    certificates: list[SimultaneousCertificate]  # one per horizon t=1..n
    status_transitions: list[StateTransition]
    final_status: PolicyStatus
    first_support_t: int | None
    demotion_t: int | None


def replay_policy_evidence(
    evidence_rows: list[dict],
    *,
    c: float,
    r: float,
    delta: float,
    r_max: float,
    alpha_allocation: AlphaAllocation | None = None,
    starting_status: PolicyStatus = PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE,
) -> PolicyReplayResult:
    """evidence_rows: FinalEvidenceRecord dicts (as loaded from
    EvidenceStore.load_all()), IN THE ORDER THEY WERE APPENDED -- replay
    does not re-sort them, since append order is itself part of what must
    reproduce identically (mission section 11: "Replaying immutable
    evidence from disk must reproduce exactly the same ... status
    transitions"). Each row must carry `action`, `loss`, `realized_return`.
    """
    alloc = alpha_allocation or default_equal_split(0.05)
    g_c_values: list[float] = []
    g_l_values: list[float] = []
    g_v_values: list[float] = []
    certificates: list[SimultaneousCertificate] = []
    transitions: list[StateTransition] = []

    status = starting_status
    first_support_t: int | None = None
    demotion_t: int | None = None

    for t, row in enumerate(evidence_rows, start=1):
        g_c_values.append(g_c_value(row["action"], c))
        g_l_values.append(g_l_value(row["action"], row["loss"], r))
        g_v_values.append(g_v_value(row["action"], row["realized_return"], delta))
        cert = evaluate_simultaneous_certificate(
            g_c_values, g_l_values, g_v_values, c=c, r=r, delta=delta, r_max=r_max, alpha_allocation=alloc
        )
        certificates.append(cert)

        transition = next_status(status, fully_supported=cert.fully_supported, t=t)
        status = transition.next
        transitions.append(transition)
        if status == PolicyStatus.FROZEN_POLICY_PROSPECTIVELY_SUPPORTED and first_support_t is None:
            first_support_t = t
            promote = next_status(status, fully_supported=cert.fully_supported, t=t)
            status = promote.next
            transitions.append(promote)
        if status == PolicyStatus.PRODUCTION_DEMOTED and demotion_t is None:
            demotion_t = t

    return PolicyReplayResult(
        n=len(evidence_rows),
        g_c_values=g_c_values,
        g_l_values=g_l_values,
        g_v_values=g_v_values,
        certificates=certificates,
        status_transitions=transitions,
        final_status=status,
        first_support_t=first_support_t,
        demotion_t=demotion_t,
    )
