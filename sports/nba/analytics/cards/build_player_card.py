"""
Player Card Builder

Outputs transparent player cards with:
  vector, role distribution, role envelope, spacing profile,
  confidence, warnings, sources.

Uses exact percentiles, not vague labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..schema import PlayerCapabilityVector, CAPABILITY_DIMENSIONS
from ..roles.role_distribution import generate_role_distribution
from ..features.spacing_features import compute_player_spacing


def build_player_card(
    vector: PlayerCapabilityVector,
    spacing_profile: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build a full transparent player card.

    Includes: vector summary, role distribution, spacing, confidence, warnings.
    All values use exact percentiles first, interpretation second.
    """
    role_dist = generate_role_distribution(vector)
    conf = vector.confidence_summary()

    # Top dimensions by percentile
    top_dims = []
    for dim in CAPABILITY_DIMENSIONS:
        d = vector.get(dim)
        if d.raw_percentile is not None:
            top_dims.append({
                "dimension": dim,
                "raw_value": d.raw_value,
                "percentile": d.raw_percentile,
                "position_percentile": d.position_percentile,
                "adjusted_percentile": d.reliability_adjusted_percentile,
                "confidence": d.confidence,
                "status": d.observation_status.value,
            })
    top_dims.sort(key=lambda x: x["percentile"] or 0, reverse=True)

    # Warnings
    warnings = []
    if conf["unavailable_dimensions"] > 5:
        warnings.append(f"high_missing_data: {conf['unavailable_dimensions']}/22 dimensions unavailable")
    if conf["average_confidence"] < 0.4:
        warnings.append(f"low_confidence: avg={conf['average_confidence']:.2f}")
    stale = sum(1 for d in vector.dimensions.values() if d.stale_flag)
    if stale > 0:
        warnings.append(f"stale_data: {stale} dimensions flagged stale")

    card = {
        "player_id": vector.player_id,
        "player_name": vector.player_name,
        "team": vector.team,
        "position": vector.position,
        "season": vector.season,

        "capability_summary": {
            "top_5_dimensions": top_dims[:5],
            "bottom_3_dimensions": top_dims[-3:] if len(top_dims) >= 3 else [],
            "total_dimensions": len(CAPABILITY_DIMENSIONS),
            "observed": conf["observed_dimensions"],
            "inferred": conf["inferred_dimensions"],
            "unavailable": conf["unavailable_dimensions"],
        },

        "role_distribution": role_dist.to_dict(),

        "spacing_profile": spacing_profile or {},

        "confidence": {
            "average": conf["average_confidence"],
            "data_coverage": conf["data_coverage"],
            "sources": vector.metadata.get("data_sources", []),
        },

        "warnings": warnings,

        "full_vector": {dim: vector.get(dim).to_dict() for dim in CAPABILITY_DIMENSIONS},
    }

    return card


def build_team_card(
    team_name: str,
    vectors: List[PlayerCapabilityVector],
    roster_score_result: Dict[str, Any] = None,
    spacing_ecology: Dict[str, Any] = None,
    salary_total: float = 0.0,
) -> Dict[str, Any]:
    """Build a team card with area, density, spacing, holes, conflicts, salary."""
    player_summaries = []
    for v in vectors:
        dims = [d.raw_value for d in v.dimensions.values() if d.raw_value is not None]
        avg = float(sum(dims) / len(dims)) if dims else 0
        player_summaries.append({
            "player": v.player_name,
            "avg_capability": round(avg, 1),
            "top_dimension": max(
                [(d.raw_percentile or 0, name) for name, d in v.dimensions.items()],
                default=(0, "none")
            )[1],
        })

    card = {
        "team": team_name,
        "roster_size": len(vectors),
        "players": player_summaries,
        "roster_score": roster_score_result or {},
        "spacing_ecology": spacing_ecology or {},
        "salary_total": salary_total,
        "warnings": [],
    }

    # Add team-level warnings from roster score
    if roster_score_result:
        card["warnings"] = roster_score_result.get("fatal_holes", []) + roster_score_result.get("warnings", [])

    return card
