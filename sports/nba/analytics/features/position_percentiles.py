"""
Position-Specific and Role-Specific Percentiles

Computes percentiles within position groups and role clusters,
not just league-wide. A center's rebounding should be compared
to other centers, not to guards.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..schema import CAPABILITY_DIMENSIONS, PlayerCapabilityVector
from .percentiles import compute_percentile, reliability_shrinkage


# Position groups for percentile computation
POSITION_GROUPS = {
    "guard": ["G", "PG", "SG", "Guard"],
    "wing": ["F", "SF", "GF", "Wing", "Forward"],
    "big": ["C", "PF", "FC", "Big", "Center"],
}


def _classify_position(position: str) -> str:
    """Classify a player position into guard/wing/big."""
    pos = position.upper().strip()
    for group, patterns in POSITION_GROUPS.items():
        if any(p.upper() in pos for p in patterns):
            return group
    # Fallback heuristics
    if "G" in pos:
        return "guard"
    if "C" in pos:
        return "big"
    return "wing"


def compute_position_percentiles(
    vectors: List[PlayerCapabilityVector],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute position-specific percentiles for all players.

    Returns: {player_name: {dimension: {"position_pct": X, "position_group": Y}}}
    """
    # Group players by position
    groups: Dict[str, List[PlayerCapabilityVector]] = {"guard": [], "wing": [], "big": []}
    for v in vectors:
        grp = _classify_position(v.position)
        groups[grp].append(v)

    # Build population arrays per group per dimension
    group_pops: Dict[str, Dict[str, np.ndarray]] = {}
    for grp, members in groups.items():
        group_pops[grp] = {}
        for dim in CAPABILITY_DIMENSIONS:
            vals = [m.get(dim).raw_value for m in members if m.get(dim).raw_value is not None]
            group_pops[grp][dim] = np.array(vals) if vals else np.array([])

    # Assign position percentiles
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for v in vectors:
        grp = _classify_position(v.position)
        player_result = {}
        for dim in CAPABILITY_DIMENSIONS:
            d = v.get(dim)
            if d.raw_value is not None and len(group_pops[grp][dim]) > 3:
                pos_pct = compute_percentile(d.raw_value, group_pops[grp][dim])
                player_result[dim] = {
                    "position_percentile": pos_pct,
                    "position_group": grp,
                    "group_size": len(group_pops[grp][dim]),
                }
            else:
                player_result[dim] = {
                    "position_percentile": None,
                    "position_group": grp,
                    "group_size": len(group_pops.get(grp, {}).get(dim, [])),
                }
        results[v.player_name] = player_result

    return results


def assign_position_percentiles_to_vectors(
    vectors: List[PlayerCapabilityVector],
) -> None:
    """Assign position percentiles directly to vector dimensions (in-place)."""
    pos_pcts = compute_position_percentiles(vectors)
    for v in vectors:
        player_pcts = pos_pcts.get(v.player_name, {})
        for dim, info in player_pcts.items():
            pct = info.get("position_percentile")
            if pct is not None:
                v.set_dimension(dim, position_percentile=pct)
