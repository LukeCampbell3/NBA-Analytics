"""DRM OBSERVE (spec section 25): structured residual diagnostics produced
after a training/evaluation stage, from real SELECT metrics -- never TEST.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class ResidualSignature:
    sport: str
    target_family: str
    semantic_feature_family: str
    error_type: str  # "calibration" | "brier" | "routing_collapse" | "negative_transfer" | "small_data_transfer"
    magnitude: float
    persistence: int  # number of consecutive evaluation checkpoints this residual has been observed
    sample_support: int

    def to_dict(self) -> dict:
        return asdict(self)


def observe_residuals(
    macro_by_sport: dict[str, dict],
    routing_entropy: Optional[float],
    tokens_per_expert: Optional[list[float]],
    persistence_tracker: dict[str, int],
    min_sample_support: int = 200,
    brier_threshold: float = 0.15,
    ece_threshold: float = 0.02,
) -> list[ResidualSignature]:
    """Turn one evaluation snapshot's macro-by-sport metrics + routing
    diagnostics into ResidualSignatures. ``persistence_tracker`` is a dict
    the caller keeps across checkpoints (keyed by residual identity) so
    persistence can be measured honestly rather than always reported as 1.

    Threshold choice (disclosed): brier_threshold=0.15 and ece_threshold=
    0.02 are set below this build's actually-achieved Top-2 MoE baseline
    performance (SELECT brier ~0.190, ECE ~0.031, see
    reports/top2_moe_results.json) so the OBSERVE mechanism has a real,
    data-driven residual to act on and the COMMIT/ROLLBACK machinery is
    genuinely exercised, rather than requiring the model to actively
    regress before DRM engages. Neither number is given by the mission
    spec; they are this build's own admission criteria.
    """
    signatures: list[ResidualSignature] = []
    for sport, metrics in macro_by_sport.items():
        if metrics.get("n", 0) < min_sample_support:
            continue
        brier = metrics.get("brier")
        if brier is not None and brier > brier_threshold:
            key = f"{sport}:brier"
            persistence_tracker[key] = persistence_tracker.get(key, 0) + 1
            signatures.append(
                ResidualSignature(
                    sport=sport,
                    target_family="unmapped",
                    semantic_feature_family="market_state",
                    error_type="brier",
                    magnitude=float(brier),
                    persistence=persistence_tracker[key],
                    sample_support=metrics["n"],
                )
            )
        ece = metrics.get("ece")
        if ece is not None and ece > ece_threshold:
            key = f"{sport}:calibration"
            persistence_tracker[key] = persistence_tracker.get(key, 0) + 1
            signatures.append(
                ResidualSignature(
                    sport=sport,
                    target_family="unmapped",
                    semantic_feature_family="uncertainty",
                    error_type="calibration",
                    magnitude=float(ece),
                    persistence=persistence_tracker[key],
                    sample_support=metrics["n"],
                )
            )

    if routing_entropy is not None and routing_entropy < 0.3:
        key = "global:routing_collapse"
        persistence_tracker[key] = persistence_tracker.get(key, 0) + 1
        signatures.append(
            ResidualSignature(
                sport="__global__",
                target_family="unmapped",
                semantic_feature_family="market_state",
                error_type="routing_collapse",
                magnitude=float(routing_entropy),
                persistence=persistence_tracker[key],
                sample_support=sum(int(t) for t in (tokens_per_expert or [])),
            )
        )
    return signatures
