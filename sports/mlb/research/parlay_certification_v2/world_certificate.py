from __future__ import annotations

"""NONVACUOUS_WORLD_CERTIFICATE (mission section 5/6) -- fixes the
vacuous-set bug in the logical world-certificate theory.

For settlement-relevant world omega, define:
    lambda_S(omega) = 1{settlement return for S in omega < 0}
    B_S(C) = {omega in C : lambda_S(omega) = 1}

A valid logical loss-free certificate requires ALL THREE:
    1. C is nonempty
    2. retained probability mass > 0
    3. B_S(C) is empty

C = empty makes B_S(C) = empty VACUOUSLY -- that must never certify.
`build_nonvacuous_world_certificate` enforces all three conditions
explicitly and independently. `naive_vacuous_rule_certified` reproduces
the OLD/NAIVE rule (checking only condition 3) ONLY so the required
regression test can show it wrongly certifies an empty world set -- it
must never be used for a real decision.

The logical property this module preserves exactly, for NONEMPTY C:
    B_S(C) = empty  iff  lambda_S(omega) = 0 for every retained omega.
World-set contraction is never itself the final statistical guarantee --
see anytime_monitor.py, which remains authoritative (section 6).
"""

from dataclasses import dataclass

import numpy as np

WORLD_CERTIFICATE_VERSION = "NONVACUOUS_WORLD_CERTIFICATE_V1"


@dataclass(frozen=True)
class NonvacuousWorldCertificate:
    retained_world_count: int
    retained_probability_mass: float
    counterexample_count: int
    counterexample_mass: float
    nonempty: bool
    positive_mass: bool
    zero_loss_counterexamples: bool
    certified: bool  # ALL THREE of the above, simultaneously -- UNCHANGED definition
    # Additive fields (mission: "Resolve the remaining PARLAY_V2 APS /
    # counterexample admission bottleneck") -- never read by `certified`
    # above, never change REQUIRED-mode behavior. outside_probability_mass
    # = 1 - retained_probability_mass. world_risk_rho = counterexample_mass
    # + outside_probability_mass: the world_gate_research.py-derived
    # "outside-mass-protected" risk quantity -- provably
    # world_risk_rho >= counterexample_mass computed at full retention
    # (APS_THRESHOLD=1.0) always, so shrinking the retained set can never
    # make a candidate look safer under this quantity than it would look
    # with no world-set shrinkage at all (see world_gate_research.py's
    # module docstring for the exact identity and its proof).
    outside_probability_mass: float = 0.0
    world_risk_rho: float = 0.0
    version: str = WORLD_CERTIFICATE_VERSION


def build_nonvacuous_world_certificate(
    retained_world_ids: np.ndarray,
    world_probabilities: np.ndarray,
    losing_world_ids: np.ndarray,
    *,
    mass_epsilon: float = 1e-12,
) -> NonvacuousWorldCertificate:
    """retained_world_ids: C, the calibration/APS-retained world-id set.
    world_probabilities: the full distribution, indexed by world id.
    losing_world_ids: the settlement-relevant lambda_S=1 set -- worlds in
    which the candidate action's settlement return would be < 0."""
    retained_ids = np.asarray(retained_world_ids, dtype=int)
    retained_count = int(len(retained_ids))
    retained_mass = float(world_probabilities[retained_ids].sum()) if retained_count > 0 else 0.0

    losing_set = set(np.asarray(losing_world_ids, dtype=int).tolist())
    counterexample_ids = np.array([w for w in retained_ids.tolist() if w in losing_set], dtype=int)
    counterexample_count = int(len(counterexample_ids))
    counterexample_mass = float(world_probabilities[counterexample_ids].sum()) if counterexample_count > 0 else 0.0

    nonempty = retained_count > 0
    positive_mass = retained_mass > mass_epsilon
    zero_loss_counterexamples = counterexample_count == 0

    # THE FIX: certification requires nonempty AND positive_mass AND
    # zero_loss_counterexamples, all three. A naive rule checking only the
    # third would wrongly certify when C is empty (or has zero mass) --
    # see naive_vacuous_rule_certified and
    # test_naive_rule_certifies_empty_set_but_v2_refuses.
    certified = bool(nonempty and positive_mass and zero_loss_counterexamples)

    outside_probability_mass = float(1.0 - retained_mass)
    world_risk_rho = float(counterexample_mass + outside_probability_mass)

    return NonvacuousWorldCertificate(
        retained_world_count=retained_count,
        retained_probability_mass=retained_mass,
        counterexample_count=counterexample_count,
        counterexample_mass=counterexample_mass,
        nonempty=nonempty,
        positive_mass=positive_mass,
        zero_loss_counterexamples=zero_loss_counterexamples,
        certified=certified,
        outside_probability_mass=outside_probability_mass,
        world_risk_rho=world_risk_rho,
    )


def naive_vacuous_rule_certified(retained_world_ids: np.ndarray, losing_world_ids: np.ndarray) -> bool:
    """THE OLD/NAIVE RULE -- reproduced ONLY for the mandatory regression
    test proving it is wrong. Checks only B_S(C)=empty, ignoring whether C
    itself is empty. NEVER call this to make a real decision."""
    retained_ids = np.asarray(retained_world_ids, dtype=int)
    losing_set = set(np.asarray(losing_world_ids, dtype=int).tolist())
    counterexamples = [w for w in retained_ids.tolist() if w in losing_set]
    return len(counterexamples) == 0  # vacuously True when retained_ids is empty


def world_coverage_loss_bound(alpha_world: float, c: float) -> float:
    """Mechanical bridge theorem (section 6): for a policy acting only
    with valid nonvacuous certificates, {loss AND action} is a subset of
    {realized world not in C}. So if, conditional on E=1,
    P(realized world not in C | E=1) <= alpha_world and P(A=1|E=1) >= c,
    then L <= min(1, alpha_world/c).

    This is a mechanical bound on world-set coverage -- it does NOT
    substitute for the outer prospective anytime certificate
    (anytime_monitor.py), which remains authoritative per section 6."""
    if c <= 0:
        raise ValueError("c must be > 0 to evaluate the world-coverage bridge bound")
    return float(min(1.0, alpha_world / c))
