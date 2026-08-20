from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .outcome_worlds import (
    BinaryOutcomeSet,
    WorldPath,
    build_binary_outcome_set,
    guaranteed_winner_indices,
    search_parlay_proof_frontier,
)
from .protocol import BINARY_OUTCOME_SET_PROTOCOL, BinaryOutcomeSetProtocol


@dataclass(frozen=True)
class ProofTrajectory:
    diagnostics: pd.DataFrame
    fixed_targets: dict[int, tuple[str, ...]]
    target_origin: str
    threshold_mode: str


def certificate_world_ceiling(candidate_count: int, leg_count: int) -> int:
    """Maximum binary worlds compatible with `leg_count` fixed winners."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if leg_count <= 0 or leg_count > candidate_count:
        raise ValueError("leg_count must lie in [1, candidate_count]")
    return 1 << (candidate_count - leg_count)


def minimum_support_contraction_bits(
    retained_world_count: float, certificate_ceiling: int
) -> float:
    """Cardinality-only contraction needed to reach a certificate's ceiling.

    This is not mutual information or Shannon information gain. It is only the
    base-two log ratio between the current support size and the necessary
    certificate support ceiling.
    """

    if retained_world_count < 0:
        raise ValueError("retained_world_count cannot be negative")
    if certificate_ceiling <= 0:
        raise ValueError("certificate_ceiling must be positive")
    if retained_world_count == 0:
        return 0.0
    return float(max(math.log2(retained_world_count / certificate_ceiling), 0.0))


def _expand_thresholds(
    aps_thresholds: float | Iterable[float], checkpoint_count: int
) -> tuple[tuple[float, ...], str]:
    if np.isscalar(aps_thresholds):
        values = (float(aps_thresholds),) * checkpoint_count
        mode = "FIXED_MECHANISM_THRESHOLD"
    else:
        values = tuple(float(value) for value in aps_thresholds)
        mode = "CHECKPOINT_SPECIFIC_THRESHOLDS"
    if len(values) != checkpoint_count:
        raise ValueError("aps_thresholds must contain one value per path distribution")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("all APS thresholds must be finite and lie in [0, 1]")
    return values, mode


def _checkpoint_labels(world_path: WorldPath) -> tuple[str, ...]:
    if len(world_path.diagnostics) != len(world_path.distributions):
        raise ValueError("world-path diagnostics must align with its distributions")
    if "checkpoint" not in world_path.diagnostics.columns:
        raise ValueError("world-path diagnostics are missing checkpoint labels")
    return tuple(world_path.diagnostics["checkpoint"].astype(str))


def _conditional_entropy_bits(outcome_set: BinaryOutcomeSet) -> float | None:
    if outcome_set.world_count == 0:
        return None
    retained = outcome_set.distribution.probabilities[outcome_set.world_ids]
    retained_mass = float(retained.sum())
    if retained_mass <= 0.0:
        return None
    conditional = retained / retained_mass
    positive = conditional[conditional > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def _fixed_target_metrics(
    outcome_set: BinaryOutcomeSet, target_candidate_ids: tuple[str, ...]
) -> tuple[int | None, float | None, bool]:
    if outcome_set.world_count == 0:
        return None, None, False
    distribution = outcome_set.distribution
    index_by_id = {
        candidate_id: index
        for index, candidate_id in enumerate(distribution.candidate_ids)
    }
    try:
        indices = np.asarray(
            [index_by_id[candidate_id] for candidate_id in target_candidate_ids],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(f"unknown fixed target candidate: {exc.args[0]}") from exc
    retained_ids = outcome_set.world_ids
    retained = distribution.outcomes[retained_ids]
    counterexample_mask = ~retained[:, indices].all(axis=1)
    counterexample_ids = retained_ids[counterexample_mask]
    retained_mass = float(distribution.probabilities[retained_ids].sum())
    counterexample_mass = float(distribution.probabilities[counterexample_ids].sum())
    return (
        int(len(counterexample_ids)),
        counterexample_mass / retained_mass if retained_mass > 0.0 else None,
        len(counterexample_ids) == 0,
    )


def build_proof_trajectory(
    candidates: pd.DataFrame,
    world_path: WorldPath,
    *,
    aps_thresholds: float | Iterable[float],
    calibration_slates: int,
    fixed_targets: Mapping[int, Iterable[str]] | None = None,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> ProofTrajectory:
    """Measure support and counterexample elimination through a world path.

    When explicit targets are omitted, each target is frozen from the prior
    checkpoint's exact frontier. The adaptive frontier is also reported at every
    checkpoint, but it is not treated as the same proposed parlay through time.
    """

    required = {"candidate_id", "player"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidates are missing columns: {missing}")
    frame = candidates.copy().reset_index(drop=True)
    candidate_ids = tuple(frame["candidate_id"].astype(str))
    if not world_path.distributions:
        raise ValueError("world_path must contain at least one distribution")
    if any(
        distribution.candidate_ids != candidate_ids
        for distribution in world_path.distributions
    ):
        raise ValueError("candidate order must match every world-path distribution")

    labels = _checkpoint_labels(world_path)
    thresholds, threshold_mode = _expand_thresholds(
        aps_thresholds, len(world_path.distributions)
    )
    outcome_sets = tuple(
        build_binary_outcome_set(
            distribution,
            aps_threshold=threshold,
            calibration_slates=calibration_slates,
            protocol=protocol,
        )
        for distribution, threshold in zip(world_path.distributions, thresholds)
    )

    if fixed_targets is None:
        if outcome_sets[0].world_count == 0:
            targets: dict[int, tuple[str, ...]] = {}
        else:
            targets = {
                leg_count: search_parlay_proof_frontier(
                    frame,
                    outcome_sets[0],
                    requested_leg_count=leg_count,
                    protocol=protocol,
                ).selected_candidate_ids
                for leg_count in protocol.requested_leg_counts
                if leg_count <= len(frame)
            }
        target_origin = "PRIOR_CHECKPOINT_FRONTIER_FROZEN"
    else:
        targets = {
            int(leg_count): tuple(str(value) for value in values)
            for leg_count, values in fixed_targets.items()
        }
        target_origin = "CALLER_SUPPLIED_FROZEN_TARGET"
    for leg_count, target in targets.items():
        if leg_count not in protocol.requested_leg_counts:
            raise ValueError(
                "fixed target leg count is outside the frozen policy family"
            )
        if len(target) != leg_count or len(set(target)) != leg_count:
            raise ValueError("fixed targets must contain exactly leg_count unique IDs")
        if not set(target).issubset(candidate_ids):
            raise ValueError("fixed targets contain an unknown candidate ID")

    rows: list[dict[str, Any]] = []
    previous_fixed: dict[int, tuple[int | None, float | None]] = {}
    initial_world_count: int | None = None
    for step, (label, threshold, distribution, outcome_set) in enumerate(
        zip(labels, thresholds, world_path.distributions, outcome_sets)
    ):
        retained_count = outcome_set.world_count
        if step == 0:
            initial_world_count = retained_count
        retained_mass = float(distribution.probabilities[outcome_set.world_ids].sum())
        guaranteed = guaranteed_winner_indices(outcome_set)
        row: dict[str, Any] = {
            "step": step,
            "checkpoint": label,
            "aps_threshold": threshold,
            "outcome_set_status": (
                "VALID" if retained_count else "EMPTY_OUTCOME_SET_ABSTAIN"
            ),
            "retained_world_count": retained_count,
            "retained_world_fraction": retained_count / distribution.world_count,
            "outcome_set_posterior_mass": retained_mass,
            "support_cardinality_bits": (
                float(math.log2(retained_count)) if retained_count else None
            ),
            "support_contraction_bits_from_initial": (
                float(math.log2(initial_world_count / retained_count))
                if initial_world_count and retained_count
                else None
            ),
            "distribution_entropy_bits": float(distribution.entropy / math.log(2.0)),
            "outcome_set_conditional_entropy_bits": _conditional_entropy_bits(
                outcome_set
            ),
            "guaranteed_winner_count": len(guaranteed),
            "guaranteed_candidate_ids": "|".join(
                distribution.candidate_ids[index] for index in guaranteed
            ),
        }
        for leg_count in protocol.requested_leg_counts:
            if leg_count > distribution.candidate_count:
                continue
            prefix = f"{leg_count}_leg"
            ceiling = certificate_world_ceiling(distribution.candidate_count, leg_count)
            row[f"{prefix}_certificate_world_ceiling"] = ceiling
            row[f"{prefix}_cardinality_feasible"] = bool(
                retained_count > 0 and retained_count <= ceiling
            )
            row[f"{prefix}_minimum_additional_world_removals"] = max(
                retained_count - ceiling, 0
            )
            row[f"{prefix}_minimum_support_contraction_bits"] = (
                minimum_support_contraction_bits(retained_count, ceiling)
            )

            if retained_count:
                frontier = search_parlay_proof_frontier(
                    frame,
                    outcome_set,
                    requested_leg_count=leg_count,
                    protocol=protocol,
                )
                count_minimum = int(
                    frontier.ranking["counterexample_world_count"].min()
                )
                count_row = frontier.ranking.loc[
                    frontier.ranking["counterexample_world_count"].eq(count_minimum)
                ].iloc[0]
                mass_minimum = float(
                    frontier.ranking["counterexample_mass_within_set"].min()
                )
                mass_row = frontier.ranking.loc[
                    np.isclose(
                        frontier.ranking["counterexample_mass_within_set"],
                        mass_minimum,
                        rtol=0.0,
                        atol=protocol.score_epsilon,
                    )
                ].iloc[0]
                row[f"{prefix}_adaptive_candidate_ids"] = "|".join(
                    frontier.selected_candidate_ids
                )
                row[f"{prefix}_adaptive_counterexample_world_count"] = (
                    frontier.counterexample_world_count
                )
                row[f"{prefix}_adaptive_counterexample_mass"] = (
                    frontier.counterexample_mass_within_set
                )
                row[f"{prefix}_adaptive_logical_certificate"] = (
                    frontier.logically_proven
                )
                row[f"{prefix}_minimum_counterexample_world_count"] = count_minimum
                row[f"{prefix}_count_minimizer_candidate_ids"] = "|".join(
                    count_row["candidate_ids"]
                )
                row[f"{prefix}_minimum_counterexample_mass"] = mass_minimum
                row[f"{prefix}_mass_minimizer_candidate_ids"] = "|".join(
                    mass_row["candidate_ids"]
                )
            else:
                row[f"{prefix}_adaptive_candidate_ids"] = ""
                row[f"{prefix}_adaptive_counterexample_world_count"] = None
                row[f"{prefix}_adaptive_counterexample_mass"] = None
                row[f"{prefix}_adaptive_logical_certificate"] = False
                row[f"{prefix}_minimum_counterexample_world_count"] = None
                row[f"{prefix}_count_minimizer_candidate_ids"] = ""
                row[f"{prefix}_minimum_counterexample_mass"] = None
                row[f"{prefix}_mass_minimizer_candidate_ids"] = ""

            target = targets.get(leg_count)
            row[f"{prefix}_fixed_candidate_ids"] = (
                "|".join(target) if target is not None else ""
            )
            if target is None:
                fixed_count, fixed_mass, fixed_certificate = None, None, False
            else:
                fixed_count, fixed_mass, fixed_certificate = _fixed_target_metrics(
                    outcome_set, target
                )
            row[f"{prefix}_fixed_counterexample_world_count"] = fixed_count
            row[f"{prefix}_fixed_counterexample_mass"] = fixed_mass
            row[f"{prefix}_fixed_logical_certificate"] = fixed_certificate
            previous_count, previous_mass = previous_fixed.get(leg_count, (None, None))
            row[f"{prefix}_fixed_counterexamples_eliminated_since_prior"] = (
                previous_count - fixed_count
                if previous_count is not None and fixed_count is not None
                else None
            )
            row[f"{prefix}_fixed_counterexample_mass_reduction_since_prior"] = (
                previous_mass - fixed_mass
                if previous_mass is not None and fixed_mass is not None
                else None
            )
            previous_fixed[leg_count] = (fixed_count, fixed_mass)
        rows.append(row)

    return ProofTrajectory(
        diagnostics=pd.DataFrame(rows),
        fixed_targets=targets,
        target_origin=target_origin,
        threshold_mode=threshold_mode,
    )
