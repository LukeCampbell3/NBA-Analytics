from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .outcome_worlds import (
    BinaryOutcomeSet,
    WorldDistribution,
    apply_joint_world_evidence_path,
    build_binary_outcome_set,
    build_world_distribution,
    enumerate_binary_worlds,
    guaranteed_winner_indices,
    search_parlay_proof_frontier,
)
from .protocol import BINARY_OUTCOME_SET_PROTOCOL, BinaryOutcomeSetProtocol


def exhaustive_intersection_theorem_audit(candidate_count: int = 3) -> dict[str, Any]:
    """Exhaustively verify the finite-state perfect-parlay existence theorem."""

    worlds = enumerate_binary_worlds(candidate_count)
    checks = 0
    failures = 0
    for retained_mask in range(1, 1 << len(worlds)):
        retained_ids = np.flatnonzero(
            [retained_mask & (1 << world_id) for world_id in range(len(worlds))]
        )
        retained = worlds[retained_ids]
        guaranteed_count = int((retained == 1).all(axis=0).sum())
        for leg_count in range(1, candidate_count + 1):
            direct_exists = any(
                bool(retained[:, indices].all())
                for indices in combinations(range(candidate_count), leg_count)
            )
            intersection_exists = guaranteed_count >= leg_count
            checks += 1
            failures += int(direct_exists != intersection_exists)
    return {
        "candidate_count": candidate_count,
        "nonempty_outcome_sets": (1 << len(worlds)) - 1,
        "theorem_checks": checks,
        "failures": failures,
        "passed": failures == 0,
        "theorem": (
            "An n-leg perfect parlay exists inside a retained binary outcome set "
            "if and only if at least n candidate coordinates equal one in every "
            "retained world."
        ),
    }


def _scenario_summary(
    distribution: WorldDistribution,
    outcome_set: BinaryOutcomeSet,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    guaranteed = guaranteed_winner_indices(outcome_set)
    pair_frontier = search_parlay_proof_frontier(
        candidates,
        outcome_set,
        requested_leg_count=2,
    )
    return {
        "retained_worlds": outcome_set.world_count,
        "retained_world_fraction": outcome_set.world_count / distribution.world_count,
        "guaranteed_candidate_ids": [
            distribution.candidate_ids[index] for index in guaranteed
        ],
        "guaranteed_winner_count": len(guaranteed),
        "pair_logically_proven": pair_frontier.logically_proven,
        "pair_selected_candidate_ids": list(pair_frontier.selected_candidate_ids),
        "pair_counterexample_worlds": pair_frontier.counterexample_world_count,
        "pair_counterexample_mass_within_set": (
            pair_frontier.counterexample_mass_within_set
        ),
        "entropy": distribution.entropy,
        "effective_worlds": distribution.effective_worlds,
    }


def run_binary_path_sensitivity_audit(
    *,
    aps_threshold: float = 0.90,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> dict[str, Any]:
    """Test that coherent joint evidence can create, and reversal removes, a proof."""

    candidate_ids = ("candidate_a", "candidate_b", "candidate_c", "candidate_d")
    candidates = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "player": ("Alpha", "Beta", "Gamma", "Delta"),
            "survival_probability": (0.61, 0.57, 0.53, 0.49),
        }
    )
    prior = build_world_distribution(
        candidate_ids,
        candidates["survival_probability"],
        protocol=protocol,
    )
    shared_pair_state = prior.outcomes[:, 0].astype(bool) & prior.outcomes[:, 1].astype(
        bool
    )
    coherent_checkpoint = np.where(shared_pair_state, 5.0, 0.0)
    coherent_path = apply_joint_world_evidence_path(
        prior,
        np.vstack([coherent_checkpoint, coherent_checkpoint]),
        checkpoint_labels=("T-30", "T-5"),
    )
    reversal_path = apply_joint_world_evidence_path(
        prior,
        np.vstack(
            [coherent_checkpoint, coherent_checkpoint, -2.0 * coherent_checkpoint]
        ),
        checkpoint_labels=("T-30", "T-5", "REVERSAL"),
    )

    distributions = {
        "endpoint_only": prior,
        "coherent_joint_path": coherent_path.distributions[-1],
        "fully_reversed_path": reversal_path.distributions[-1],
    }
    scenarios: dict[str, dict[str, Any]] = {}
    for name, distribution in distributions.items():
        outcome_set = build_binary_outcome_set(
            distribution,
            aps_threshold=aps_threshold,
            calibration_slates=protocol.minimum_calibration_slates,
            protocol=protocol,
        )
        scenarios[name] = _scenario_summary(distribution, outcome_set, candidates)

    theorem_audit = exhaustive_intersection_theorem_audit(candidate_count=3)
    mechanism_passed = bool(
        theorem_audit["passed"]
        and not scenarios["endpoint_only"]["pair_logically_proven"]
        and scenarios["coherent_joint_path"]["pair_logically_proven"]
        and not scenarios["fully_reversed_path"]["pair_logically_proven"]
        and np.allclose(
            reversal_path.distributions[-1].probabilities,
            prior.probabilities,
        )
    )
    return {
        "audit": "binary_joint_path_proof_sensitivity",
        "outcome_set_version": protocol.version,
        "aps_threshold": aps_threshold,
        "evidence_status": "SYNTHETIC_MECHANISM_AUDIT_NOT_ACCURACY_EVIDENCE",
        "mechanism_passed": mechanism_passed,
        "theorem_audit": theorem_audit,
        "scenarios": scenarios,
        "coherent_path_diagnostics": coherent_path.diagnostics.to_dict(
            orient="records"
        ),
        "reversal_path_diagnostics": reversal_path.diagnostics.to_dict(
            orient="records"
        ),
        "supported_claim": (
            "A shared-state evidence path can produce an n-leg logical intersection "
            "when it excludes every retained counterexample world; exact reversal "
            "removes that intersection."
        ),
        "unsupported_claim": (
            "This synthetic audit does not show that observed NBA market paths carry "
            "the required evidence or that a future parlay will win."
        ),
    }
