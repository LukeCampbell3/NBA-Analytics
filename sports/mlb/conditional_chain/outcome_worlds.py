from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .protocol import BINARY_OUTCOME_SET_PROTOCOL, BinaryOutcomeSetProtocol


@dataclass(frozen=True)
class WorldDistribution:
    candidate_ids: tuple[str, ...]
    outcomes: np.ndarray
    probabilities: np.ndarray
    representation_version: str

    @property
    def world_count(self) -> int:
        return int(len(self.probabilities))

    @property
    def candidate_count(self) -> int:
        return int(len(self.candidate_ids))

    @property
    def marginals(self) -> np.ndarray:
        return self.probabilities @ self.outcomes

    @property
    def entropy(self) -> float:
        positive = self.probabilities[self.probabilities > 0.0]
        return float(-np.sum(positive * np.log(positive)))

    @property
    def effective_worlds(self) -> float:
        return float(np.exp(self.entropy))


@dataclass(frozen=True)
class BinaryOutcomeSet:
    distribution: WorldDistribution
    world_ids: np.ndarray
    aps_threshold: float
    calibration_slates: int
    target_miscoverage: float
    calibration_method: str

    @property
    def world_count(self) -> int:
        return int(len(self.world_ids))


@dataclass(frozen=True)
class WorldPath:
    distributions: tuple[WorldDistribution, ...]
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class PerfectParlayCertificate:
    status: str
    requested_leg_count: int
    selected_candidate_ids: tuple[str, ...]
    selected_players: tuple[str, ...]
    guaranteed_winner_count: int
    retained_world_count: int
    logical_implication_proven: bool
    target_marginal_outcome_set_coverage: float | None
    conditional_win_probability_claim: float | None
    path_certificate_valid: bool
    production_authorized: bool
    proof_statement: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "requested_leg_count": self.requested_leg_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_players": list(self.selected_players),
            "guaranteed_winner_count": self.guaranteed_winner_count,
            "retained_world_count": self.retained_world_count,
            "logical_implication_proven": self.logical_implication_proven,
            "target_marginal_outcome_set_coverage": (
                self.target_marginal_outcome_set_coverage
            ),
            "conditional_win_probability_claim": self.conditional_win_probability_claim,
            "path_certificate_valid": self.path_certificate_valid,
            "production_authorized": self.production_authorized,
            "proof_statement": self.proof_statement,
        }


@dataclass(frozen=True)
class ParlayProofFrontier:
    requested_leg_count: int
    selected_candidate_ids: tuple[str, ...]
    selected_players: tuple[str, ...]
    logically_proven: bool
    retained_world_count: int
    counterexample_world_count: int
    counterexample_world_sha256: str
    counterexample_mass_within_set: float
    posterior_all_win_probability: float
    combinations_evaluated: int
    ranking: pd.DataFrame

    @property
    def worlds_remaining_to_exclude(self) -> int:
        return self.counterexample_world_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_leg_count": self.requested_leg_count,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_players": list(self.selected_players),
            "logically_proven": self.logically_proven,
            "retained_world_count": self.retained_world_count,
            "counterexample_world_count": self.counterexample_world_count,
            "counterexample_world_sha256": self.counterexample_world_sha256,
            "counterexample_mass_within_set": self.counterexample_mass_within_set,
            "posterior_all_win_probability": self.posterior_all_win_probability,
            "combinations_evaluated": self.combinations_evaluated,
            "worlds_remaining_to_exclude": self.worlds_remaining_to_exclude,
        }


def enumerate_binary_worlds(candidate_count: int) -> np.ndarray:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if candidate_count > BINARY_OUTCOME_SET_PROTOCOL.maximum_candidates:
        raise ValueError(
            "candidate_count exceeds the frozen exact-enumeration maximum of "
            f"{BINARY_OUTCOME_SET_PROTOCOL.maximum_candidates}"
        )
    world_ids = np.arange(1 << candidate_count, dtype=np.uint32)
    bit_positions = np.arange(candidate_count, dtype=np.uint32)
    return ((world_ids[:, None] >> bit_positions) & 1).astype(np.int8)


def _validate_interactions(
    interactions: np.ndarray, candidate_count: int
) -> np.ndarray:
    matrix = np.asarray(interactions, dtype=float)
    if matrix.shape != (candidate_count, candidate_count):
        raise ValueError("interactions must be a square candidate_count matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("interactions must be finite")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("interactions must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("interaction diagonal must be zero")
    return matrix


def build_world_distribution(
    candidate_ids: Iterable[str],
    marginal_probabilities: Iterable[float],
    *,
    interactions: np.ndarray | None = None,
    admissible_world_mask: np.ndarray | None = None,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> WorldDistribution:
    ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
    probabilities = np.asarray(list(marginal_probabilities), dtype=float)
    if len(ids) != len(probabilities) or not ids:
        raise ValueError(
            "candidate IDs and marginal probabilities must have equal positive length"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    if len(ids) > protocol.maximum_candidates:
        raise ValueError("candidate count exceeds the frozen exact-enumeration maximum")
    if not np.isfinite(probabilities).all() or bool(
        ((probabilities <= 0.0) | (probabilities >= 1.0)).any()
    ):
        raise ValueError(
            "marginal probabilities must be finite and strictly between zero and one"
        )

    outcomes = enumerate_binary_worlds(len(ids))
    clipped = np.clip(
        probabilities, protocol.score_epsilon, 1.0 - protocol.score_epsilon
    )
    log_weights = outcomes @ np.log(clipped) + (1 - outcomes) @ np.log1p(-clipped)
    if interactions is not None:
        matrix = _validate_interactions(interactions, len(ids))
        signed = 2.0 * outcomes - 1.0
        log_weights = log_weights + 0.5 * np.einsum(
            "bi,ij,bj->b", signed, matrix, signed
        )
    if admissible_world_mask is not None:
        mask = np.asarray(admissible_world_mask, dtype=bool)
        if mask.shape != (len(outcomes),):
            raise ValueError("admissible_world_mask has the wrong shape")
        if not bool(mask.any()):
            raise ValueError("at least one world must remain admissible")
        log_weights = np.where(mask, log_weights, -np.inf)
    normalized = np.exp(log_weights - logsumexp(log_weights))
    return WorldDistribution(
        candidate_ids=ids,
        outcomes=outcomes,
        probabilities=normalized,
        representation_version=protocol.version,
    )


def update_world_distribution(
    distribution: WorldDistribution,
    world_log_evidence: Iterable[float],
) -> WorldDistribution:
    evidence = np.asarray(list(world_log_evidence), dtype=float)
    if evidence.shape != (distribution.world_count,) or not np.isfinite(evidence).all():
        raise ValueError("world_log_evidence must contain one finite value per world")
    current_log = np.full(distribution.world_count, -np.inf, dtype=float)
    positive = distribution.probabilities > 0.0
    current_log[positive] = np.log(distribution.probabilities[positive])
    updated_log = current_log + evidence
    if not np.isfinite(logsumexp(updated_log)):
        raise ValueError("evidence eliminated every world")
    updated = np.exp(updated_log - logsumexp(updated_log))
    return WorldDistribution(
        candidate_ids=distribution.candidate_ids,
        outcomes=distribution.outcomes,
        probabilities=updated,
        representation_version=distribution.representation_version,
    )


def apply_candidate_evidence_path(
    prior: WorldDistribution,
    win_likelihood_ratios: np.ndarray,
    *,
    checkpoint_labels: Iterable[str] | None = None,
) -> WorldPath:
    ratios = np.asarray(win_likelihood_ratios, dtype=float)
    if ratios.ndim != 2 or ratios.shape[1] != prior.candidate_count:
        raise ValueError("win_likelihood_ratios must be checkpoints by candidates")
    if not np.isfinite(ratios).all() or bool((ratios <= 0.0).any()):
        raise ValueError("likelihood ratios must be finite and positive")
    labels = (
        list(checkpoint_labels)
        if checkpoint_labels is not None
        else [f"checkpoint_{index + 1}" for index in range(len(ratios))]
    )
    if len(labels) != len(ratios):
        raise ValueError("checkpoint_labels must match the number of checkpoints")

    distributions = [prior]
    rows = [
        {
            "step": 0,
            "checkpoint": "prior",
            "entropy": prior.entropy,
            "effective_worlds": prior.effective_worlds,
            "maximum_world_probability": float(prior.probabilities.max()),
        }
    ]
    current = prior
    for step, (label, checkpoint_ratios) in enumerate(zip(labels, ratios), start=1):
        log_evidence = current.outcomes @ np.log(checkpoint_ratios)
        current = update_world_distribution(current, log_evidence)
        distributions.append(current)
        rows.append(
            {
                "step": step,
                "checkpoint": str(label),
                "entropy": current.entropy,
                "effective_worlds": current.effective_worlds,
                "maximum_world_probability": float(current.probabilities.max()),
            }
        )
    return WorldPath(
        distributions=tuple(distributions),
        diagnostics=pd.DataFrame(rows),
    )


def apply_joint_world_evidence_path(
    prior: WorldDistribution,
    checkpoint_world_log_evidence: np.ndarray,
    *,
    checkpoint_labels: Iterable[str] | None = None,
) -> WorldPath:
    """Apply shared-state evidence directly to joint outcome worlds."""

    evidence = np.asarray(checkpoint_world_log_evidence, dtype=float)
    if evidence.ndim != 2 or evidence.shape[1] != prior.world_count:
        raise ValueError(
            "checkpoint_world_log_evidence must be checkpoints by outcome worlds"
        )
    if not np.isfinite(evidence).all():
        raise ValueError("checkpoint_world_log_evidence must be finite")
    labels = (
        list(checkpoint_labels)
        if checkpoint_labels is not None
        else [f"checkpoint_{index + 1}" for index in range(len(evidence))]
    )
    if len(labels) != len(evidence):
        raise ValueError("checkpoint_labels must match the number of checkpoints")

    distributions = [prior]
    rows = [
        {
            "step": 0,
            "checkpoint": "prior",
            "entropy": prior.entropy,
            "effective_worlds": prior.effective_worlds,
            "maximum_world_probability": float(prior.probabilities.max()),
        }
    ]
    current = prior
    for step, (label, world_evidence) in enumerate(zip(labels, evidence), start=1):
        current = update_world_distribution(current, world_evidence)
        distributions.append(current)
        rows.append(
            {
                "step": step,
                "checkpoint": str(label),
                "entropy": current.entropy,
                "effective_worlds": current.effective_worlds,
                "maximum_world_probability": float(current.probabilities.max()),
            }
        )
    return WorldPath(
        distributions=tuple(distributions),
        diagnostics=pd.DataFrame(rows),
    )


def aps_world_scores(distribution: WorldDistribution) -> np.ndarray:
    """Return tie-aware cumulative-mass scores for every joint outcome."""

    scores = np.empty(distribution.world_count, dtype=float)
    order = np.argsort(-distribution.probabilities, kind="mergesort")
    cumulative = 0.0
    start = 0
    while start < len(order):
        probability = distribution.probabilities[order[start]]
        end = start + 1
        while end < len(order) and np.isclose(
            distribution.probabilities[order[end]], probability, rtol=0.0, atol=1e-15
        ):
            end += 1
        cumulative += float(distribution.probabilities[order[start:end]].sum())
        scores[order[start:end]] = cumulative
        start = end
    return scores


def world_id_from_outcomes(outcomes: Iterable[int | bool]) -> int:
    values = np.asarray(list(outcomes), dtype=int)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("outcomes must be a one-dimensional binary vector")
    return int(
        np.sum(values.astype(np.uint64) << np.arange(len(values), dtype=np.uint64))
    )


def conformal_aps_threshold(
    calibration_scores: Iterable[float],
    *,
    target_miscoverage: float = BINARY_OUTCOME_SET_PROTOCOL.target_miscoverage,
) -> float:
    scores = np.asarray(list(calibration_scores), dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("calibration_scores must be non-empty and finite")
    if not 0.0 < target_miscoverage < 1.0:
        raise ValueError("target_miscoverage must lie strictly between zero and one")
    rank = int(np.ceil((len(scores) + 1) * (1.0 - target_miscoverage)))
    if rank > len(scores):
        return 1.0
    return float(np.sort(scores)[rank - 1])


def build_binary_outcome_set(
    distribution: WorldDistribution,
    *,
    aps_threshold: float,
    calibration_slates: int,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> BinaryOutcomeSet:
    if not 0.0 <= aps_threshold <= 1.0 + protocol.score_epsilon:
        raise ValueError("aps_threshold must lie in [0, 1]")
    scores = aps_world_scores(distribution)
    world_ids = np.flatnonzero(
        (distribution.probabilities > 0.0)
        & (scores <= aps_threshold + protocol.score_epsilon)
    )
    return BinaryOutcomeSet(
        distribution=distribution,
        world_ids=world_ids.astype(np.int64),
        aps_threshold=float(min(aps_threshold, 1.0)),
        calibration_slates=int(calibration_slates),
        target_miscoverage=protocol.target_miscoverage,
        calibration_method=protocol.calibration_method,
    )


def guaranteed_winner_indices(outcome_set: BinaryOutcomeSet) -> tuple[int, ...]:
    if outcome_set.world_count == 0:
        return ()
    retained = outcome_set.distribution.outcomes[outcome_set.world_ids]
    return tuple(np.flatnonzero((retained == 1).all(axis=0)).astype(int))


def search_parlay_proof_frontier(
    candidates: pd.DataFrame,
    outcome_set: BinaryOutcomeSet,
    *,
    requested_leg_count: int,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> ParlayProofFrontier:
    """Exhaustively find the parlay closest to a universal outcome-set proof.

    A combination is proven only when it has no counterexample in the retained
    outcome set. The probabilistic fields rank shadow candidates and never
    substitute for that universal condition.
    """

    required = {"candidate_id", "player"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidates are missing columns: {missing}")
    if requested_leg_count not in protocol.requested_leg_counts:
        raise ValueError("requested_leg_count is outside the frozen policy family")
    frame = candidates.copy().reset_index(drop=True)
    candidate_ids = tuple(frame["candidate_id"].astype(str))
    if candidate_ids != outcome_set.distribution.candidate_ids:
        raise ValueError(
            "candidate order does not match the outcome-set representation"
        )
    if requested_leg_count > len(frame):
        raise ValueError("requested_leg_count exceeds the candidate count")
    if outcome_set.world_count == 0:
        raise ValueError("cannot search an empty outcome set")

    distribution = outcome_set.distribution
    retained_ids = outcome_set.world_ids
    retained_probabilities = distribution.probabilities[retained_ids]
    retained_mass = float(retained_probabilities.sum())
    score_column = (
        "survival_probability"
        if "survival_probability" in frame.columns
        else "robust_score" if "robust_score" in frame.columns else None
    )
    rows: list[dict[str, Any]] = []
    for indices in combinations(range(len(frame)), requested_leg_count):
        index_array = np.asarray(indices, dtype=int)
        all_world_wins = distribution.outcomes[:, index_array].all(axis=1)
        retained_wins = all_world_wins[retained_ids]
        counterexample_ids = retained_ids[~retained_wins]
        counterexample_mass = float(
            distribution.probabilities[counterexample_ids].sum()
        )
        encoded_counterexamples = np.asarray(counterexample_ids, dtype="<u4").tobytes()
        rows.append(
            {
                "candidate_indices": tuple(int(index) for index in indices),
                "candidate_ids": tuple(
                    frame.loc[index_array, "candidate_id"].astype(str)
                ),
                "players": tuple(frame.loc[index_array, "player"].astype(str)),
                "logically_proven": len(counterexample_ids) == 0,
                "counterexample_world_count": int(len(counterexample_ids)),
                "counterexample_world_sha256": hashlib.sha256(
                    encoded_counterexamples
                ).hexdigest(),
                "counterexample_mass_within_set": (
                    counterexample_mass / retained_mass if retained_mass > 0.0 else 1.0
                ),
                "posterior_all_win_probability": float(
                    distribution.probabilities[all_world_wins].sum()
                ),
                "candidate_score_sum": (
                    float(frame.loc[index_array, score_column].sum())
                    if score_column is not None
                    else 0.0
                ),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "logically_proven",
            "counterexample_mass_within_set",
            "counterexample_world_count",
            "posterior_all_win_probability",
            "candidate_score_sum",
            "candidate_ids",
        ],
        ascending=[False, True, True, False, False, True],
        kind="mergesort",
    )
    ranking = ranking.reset_index(drop=True)
    best = ranking.iloc[0]
    return ParlayProofFrontier(
        requested_leg_count=requested_leg_count,
        selected_candidate_ids=tuple(best["candidate_ids"]),
        selected_players=tuple(best["players"]),
        logically_proven=bool(best["logically_proven"]),
        retained_world_count=outcome_set.world_count,
        counterexample_world_count=int(best["counterexample_world_count"]),
        counterexample_world_sha256=str(best["counterexample_world_sha256"]),
        counterexample_mass_within_set=float(best["counterexample_mass_within_set"]),
        posterior_all_win_probability=float(best["posterior_all_win_probability"]),
        combinations_evaluated=len(ranking),
        ranking=ranking,
    )


def certify_perfect_parlay(
    candidates: pd.DataFrame,
    outcome_set: BinaryOutcomeSet,
    *,
    requested_leg_count: int,
    path_certificate: dict[str, Any] | None = None,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> PerfectParlayCertificate:
    required = {"candidate_id", "player"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidates are missing columns: {missing}")
    if requested_leg_count not in protocol.requested_leg_counts:
        raise ValueError("requested_leg_count is outside the frozen policy family")
    frame = candidates.copy().reset_index(drop=True)
    candidate_ids = tuple(frame["candidate_id"].astype(str))
    if candidate_ids != outcome_set.distribution.candidate_ids:
        raise ValueError(
            "candidate order does not match the outcome-set representation"
        )

    guaranteed = list(guaranteed_winner_indices(outcome_set))
    score_column = (
        "survival_probability"
        if "survival_probability" in frame.columns
        else "robust_score" if "robust_score" in frame.columns else None
    )
    if score_column is not None:
        guaranteed.sort(
            key=lambda index: (float(frame.loc[index, score_column]), -index),
            reverse=True,
        )
    selected_indices = guaranteed[:requested_leg_count]
    logical_proof = bool(
        outcome_set.world_count > 0 and len(selected_indices) == requested_leg_count
    )
    calibrated = bool(
        outcome_set.calibration_slates >= protocol.minimum_calibration_slates
    )
    path_valid = bool(
        path_certificate
        and path_certificate.get("status") == "PATH_INCREMENTAL_VALUE_SUPPORTED"
        and path_certificate.get("path_authorized", False)
    )

    if not logical_proof:
        status = "NO_ROBUST_WINNER_INTERSECTION"
    elif not calibrated:
        status = "OUTCOME_SET_NOT_CALIBRATED"
    elif protocol.require_path_certificate and not path_valid:
        status = "PATH_INCREMENTAL_VALUE_NOT_CERTIFIED"
    else:
        status = "LOGICALLY_ROBUST_SHADOW_AWAITING_SELECTIVE_RISK_CERTIFICATE"
    coverage = 1.0 - outcome_set.target_miscoverage if calibrated else None
    selected_ids = tuple(frame.loc[selected_indices, "candidate_id"].astype(str))
    selected_players = tuple(frame.loc[selected_indices, "player"].astype(str))
    return PerfectParlayCertificate(
        status=status,
        requested_leg_count=requested_leg_count,
        selected_candidate_ids=selected_ids,
        selected_players=selected_players,
        guaranteed_winner_count=len(guaranteed),
        retained_world_count=outcome_set.world_count,
        logical_implication_proven=logical_proof,
        target_marginal_outcome_set_coverage=coverage,
        conditional_win_probability_claim=None,
        path_certificate_valid=path_valid,
        production_authorized=False,
        proof_statement=(
            "Every selected leg is a win in every retained binary outcome world. "
            "This is a logical implication conditional on the outcome set containing the "
            "realized world; it is not an unconditional or action-conditional guarantee."
        ),
    )
