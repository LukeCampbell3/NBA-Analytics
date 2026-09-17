from __future__ import annotations

"""Exact-identity bridge from PARLAY_V2 candidates into robust valuation.

The bridge does not create evidence. It only joins already-existing PCA/A*
leg evidence, uncertainty samples, dependence evidence, and executable pair
prices to a descriptive PairCandidate. Missing evidence stays missing and
therefore cannot become CERTIFIED_PAIR downstream.
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .candidate_adapter import Leg, PairCandidate, exact_event_key
from .robust_parlay_search import JointDependencyEvidence, LegProbabilityEvidence


@dataclass(frozen=True)
class PcaLegRiskEvidence:
    probability_samples: Sequence[float] = field(default_factory=tuple)
    effective_support: int = 0
    pca_edge_certified: bool = False
    sampling_source: str = "UNKNOWN"


def exact_leg_key(leg: Leg) -> tuple[str, str, str, str, float]:
    return exact_event_key(leg.player_id, leg.game_id, leg.target, leg.side, leg.line)


def pair_key(candidate: PairCandidate) -> str:
    return str(candidate.candidate_id)


def leg_probability_evidence(
    leg: Leg,
    pca_evidence: Mapping[tuple[str, str, str, str, float], PcaLegRiskEvidence],
) -> LegProbabilityEvidence:
    ev = pca_evidence.get(exact_leg_key(leg), PcaLegRiskEvidence())
    return LegProbabilityEvidence(
        leg_id="|".join(map(str, exact_leg_key(leg))),
        game_id=str(leg.game_id),
        mean_probability=float(leg.model_probability_estimate),
        decimal_price=leg.decimal_price,
        effective_support=int(ev.effective_support),
        probability_samples=tuple(float(x) for x in ev.probability_samples),
        pca_edge_certified=bool(ev.pca_edge_certified),
        market=str(leg.target),
        book=leg.book,
        sampling_source=str(ev.sampling_source),
    )


def robust_pair_input(
    candidate: PairCandidate,
    *,
    pca_evidence: Mapping[tuple[str, str, str, str, float], PcaLegRiskEvidence],
    dependency_evidence: Mapping[str, JointDependencyEvidence] | None = None,
    combined_prices: Mapping[str, float] | None = None,
) -> tuple[str, LegProbabilityEvidence, LegProbabilityEvidence, JointDependencyEvidence, float | None]:
    pid = pair_key(candidate)
    dependency_evidence = dependency_evidence or {}
    combined_prices = combined_prices or {}

    dep = dependency_evidence.get(pid)
    if dep is None:
        if candidate.leg_1.game_id != candidate.leg_2.game_id:
            dep = JointDependencyEvidence(
                mode="INDEPENDENT",
                support_count=0,
                note="cross-game structural default; marginal uncertainty still required",
            )
        else:
            dep = JointDependencyEvidence(
                mode="UNSUPPORTED",
                note="same-game dependence evidence missing",
            )

    price = combined_prices.get(pid)
    if price is None and candidate.leg_1.game_id != candidate.leg_2.game_id:
        if candidate.leg_1.decimal_price is not None and candidate.leg_2.decimal_price is not None:
            price = float(candidate.leg_1.decimal_price) * float(candidate.leg_2.decimal_price)

    return (
        pid,
        leg_probability_evidence(candidate.leg_1, pca_evidence),
        leg_probability_evidence(candidate.leg_2, pca_evidence),
        dep,
        price,
    )


def build_robust_inputs(
    candidates: Sequence[PairCandidate],
    *,
    pca_evidence: Mapping[tuple[str, str, str, str, float], PcaLegRiskEvidence],
    dependency_evidence: Mapping[str, JointDependencyEvidence] | None = None,
    combined_prices: Mapping[str, float] | None = None,
) -> list[tuple[str, LegProbabilityEvidence, LegProbabilityEvidence, JointDependencyEvidence, float | None]]:
    return [
        robust_pair_input(
            c,
            pca_evidence=pca_evidence,
            dependency_evidence=dependency_evidence,
            combined_prices=combined_prices,
        )
        for c in candidates
    ]
