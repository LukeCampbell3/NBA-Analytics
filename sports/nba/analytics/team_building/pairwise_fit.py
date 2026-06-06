"""
Pairwise Fit Matrix

Computes pairwise fit between all player pairs.
Fit = complementarity - conflict.

Positive fit: creator + spacer, rim protector + perimeter defender
Negative fit: two high-usage non-shooters, two rim-only bigs
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..schema import PlayerCapabilityVector


def compute_pairwise_fit(a: PlayerCapabilityVector, b: PlayerCapabilityVector) -> Dict[str, Any]:
    """Compute fit score between two players.

    Returns dict with fit_score, complementarity, conflict, and reasons.
    """
    complementarity = 0.0
    conflict = 0.0
    reasons_pos = []
    reasons_neg = []

    def _v(vec, dim):
        d = vec.get(dim)
        return d.raw_value if d.raw_value is not None else 0.0

    # Creator + Spacer = positive
    a_creation = _v(a, "on_ball_creation")
    b_spacing = _v(b, "spacing_gravity")
    b_creation = _v(b, "on_ball_creation")
    a_spacing = _v(a, "spacing_gravity")

    if a_creation > 45 and b_spacing > 50:
        bonus = min(a_creation, b_spacing) * 0.02
        complementarity += bonus
        reasons_pos.append(f"creator({a.player_name})+spacer({b.player_name})")
    if b_creation > 45 and a_spacing > 50:
        bonus = min(b_creation, a_spacing) * 0.02
        complementarity += bonus
        reasons_pos.append(f"creator({b.player_name})+spacer({a.player_name})")

    # Rim pressure + corner spacing = positive
    a_rim = _v(a, "rim_pressure")
    b_corner = _v(b, "corner_spacing_value")
    b_rim = _v(b, "rim_pressure")
    a_corner = _v(a, "corner_spacing_value")

    if a_rim > 50 and b_corner > 30:
        complementarity += 0.5
        reasons_pos.append("rim_pressure+corner_spacing")
    if b_rim > 50 and a_corner > 30:
        complementarity += 0.5

    # Rim protector + perimeter defender = positive
    a_rim_prot = _v(a, "rim_protection")
    b_disrupt = _v(b, "defensive_disruption")
    b_rim_prot = _v(b, "rim_protection")
    a_disrupt = _v(a, "defensive_disruption")

    if a_rim_prot > 30 and b_disrupt > 30:
        complementarity += 0.4
        reasons_pos.append("rim_protector+perimeter_defender")
    if b_rim_prot > 30 and a_disrupt > 30:
        complementarity += 0.4

    # Two high-usage non-shooters = conflict
    if a_creation > 40 and b_creation > 40 and a_spacing < 35 and b_spacing < 35:
        conflict += 1.5
        reasons_neg.append("dual_high_usage_non_shooters")

    # Two rim-only players = conflict
    if a_rim > 50 and b_rim > 50 and a_spacing < 30 and b_spacing < 30:
        conflict += 1.0
        reasons_neg.append("dual_rim_only_non_shooters")

    # Same resource competition (both need ball, neither spaces)
    if a_creation > 50 and b_creation > 50:
        overlap_penalty = min(a_creation, b_creation) * 0.005
        conflict += overlap_penalty
        reasons_neg.append("ball_handler_overlap")

    fit_score = complementarity - conflict

    return {
        "player_a": a.player_name,
        "player_b": b.player_name,
        "fit_score": round(fit_score, 2),
        "complementarity": round(complementarity, 2),
        "conflict": round(conflict, 2),
        "positive_reasons": reasons_pos,
        "negative_reasons": reasons_neg,
    }


def build_pairwise_matrix(vectors: List[PlayerCapabilityVector]) -> List[Dict[str, Any]]:
    """Build pairwise fit matrix for a set of players.

    Returns list of pairwise fit results (only non-zero fits to save space).
    """
    results = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            fit = compute_pairwise_fit(vectors[i], vectors[j])
            if abs(fit["fit_score"]) > 0.1:  # Only track meaningful fits
                results.append(fit)

    results.sort(key=lambda x: -x["fit_score"])
    return results
