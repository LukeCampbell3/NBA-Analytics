"""
Role Distribution Engine

Generates probabilistic role distributions from capability vectors.
Roles are summaries of the vector, not fixed labels.

Role regions:
  Primary Creator, Secondary Creator, Combo Guard, Movement Shooter,
  Spot-Up Spacer, Off-Ball Gravity Wing, Slashing Wing, Connector Wing,
  Defensive Event Wing, Point-of-Attack Guard, Rim-Running Big,
  Stretch Big, Passing Big, Defensive Event Big, Interior Anchor, Hybrid Forward
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..schema import PlayerCapabilityVector, ObservationStatus


ROLE_REGIONS = [
    "primary_creator",
    "secondary_creator",
    "combo_guard",
    "movement_shooter",
    "spot_up_spacer",
    "off_ball_gravity_wing",
    "slashing_wing",
    "connector_wing",
    "defensive_event_wing",
    "point_of_attack_guard",
    "rim_running_big",
    "stretch_big",
    "passing_big",
    "defensive_event_big",
    "interior_anchor",
    "hybrid_forward",
]


@dataclass
class RoleScore:
    """Score for a single role region."""
    role: str
    probability: float = 0.0
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class RoleDistribution:
    """Full role distribution for a player."""
    player_name: str = ""
    roles: List[RoleScore] = field(default_factory=list)
    primary_role: str = ""
    secondary_role: str = ""
    envelope: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_name": self.player_name,
            "primary_role": self.primary_role,
            "secondary_role": self.secondary_role,
            "role_probabilities": {r.role: round(r.probability, 3) for r in self.roles},
            "role_evidence": {r.role: r.evidence for r in self.roles if r.probability > 0.05},
            "envelope": self.envelope,
        }


def _dim_val(vector: PlayerCapabilityVector, name: str) -> float:
    """Get raw value of a dimension, 0 if unavailable."""
    d = vector.get(name)
    return d.raw_value if d.raw_value is not None else 0.0


def generate_role_distribution(vector: PlayerCapabilityVector) -> RoleDistribution:
    """Generate role distribution from capability vector.

    Each role gets a score based on how well the vector matches that role's requirements.
    Probabilities are normalized to sum to 1.
    """
    dist = RoleDistribution(player_name=vector.player_name)
    scores: Dict[str, float] = {}
    evidence: Dict[str, List[str]] = {}

    # Helper
    v = lambda name: _dim_val(vector, name)

    # Primary Creator: high on-ball + passing + decision quality
    s = v("on_ball_creation") * 0.35 + v("passing_creation") * 0.25 + v("decision_quality") * 0.2 + v("pull_up_spacing_pressure") * 0.2
    scores["primary_creator"] = s
    evidence["primary_creator"] = [f"on_ball_creation: {v('on_ball_creation'):.0f}", f"passing: {v('passing_creation'):.0f}"]

    # Secondary Creator
    s = v("on_ball_creation") * 0.25 + v("self_scoring_efficiency") * 0.3 + v("off_ball_scalability") * 0.25 + v("passing_creation") * 0.2
    scores["secondary_creator"] = s
    evidence["secondary_creator"] = [f"scoring_eff: {v('self_scoring_efficiency'):.0f}", f"off_ball: {v('off_ball_scalability'):.0f}"]

    # Combo Guard
    s = v("on_ball_creation") * 0.25 + v("shooting_gravity") * 0.25 + v("defensive_disruption") * 0.2 + v("ball_security") * 0.15 + v("transition_value") * 0.15
    scores["combo_guard"] = s
    evidence["combo_guard"] = [f"shooting: {v('shooting_gravity'):.0f}", f"defense: {v('defensive_disruption'):.0f}"]

    # Movement Shooter
    s = v("spacing_gravity") * 0.3 + v("catch_and_shoot_gravity") * 0.3 + v("off_ball_scalability") * 0.25 + v("shooting_gravity") * 0.15
    scores["movement_shooter"] = s
    evidence["movement_shooter"] = [f"spacing: {v('spacing_gravity'):.0f}", f"c&s: {v('catch_and_shoot_gravity'):.0f}"]

    # Spot-Up Spacer
    s = v("spacing_gravity") * 0.35 + v("corner_spacing_value") * 0.3 + v("off_ball_scalability") * 0.2 + v("catch_and_shoot_gravity") * 0.15
    scores["spot_up_spacer"] = s
    evidence["spot_up_spacer"] = [f"spacing: {v('spacing_gravity'):.0f}", f"corner: {v('corner_spacing_value'):.0f}"]

    # Off-Ball Gravity Wing
    s = v("off_ball_scalability") * 0.3 + v("spacing_gravity") * 0.25 + v("self_scoring_efficiency") * 0.25 + v("transition_value") * 0.2
    scores["off_ball_gravity_wing"] = s
    evidence["off_ball_gravity_wing"] = [f"off_ball: {v('off_ball_scalability'):.0f}", f"scoring: {v('self_scoring_efficiency'):.0f}"]

    # Slashing Wing
    s = v("rim_pressure") * 0.35 + v("self_scoring_efficiency") * 0.25 + v("transition_value") * 0.2 + v("physical_translation") * 0.2
    scores["slashing_wing"] = s
    evidence["slashing_wing"] = [f"rim_pressure: {v('rim_pressure'):.0f}", f"transition: {v('transition_value'):.0f}"]

    # Connector Wing
    s = v("passing_creation") * 0.25 + v("decision_quality") * 0.25 + v("defensive_coverage_range") * 0.2 + v("off_ball_scalability") * 0.15 + v("ball_security") * 0.15
    scores["connector_wing"] = s
    evidence["connector_wing"] = [f"passing: {v('passing_creation'):.0f}", f"decisions: {v('decision_quality'):.0f}"]

    # Defensive Event Wing
    s = v("defensive_disruption") * 0.35 + v("defensive_coverage_range") * 0.3 + v("physical_translation") * 0.2 + v("transition_value") * 0.15
    scores["defensive_event_wing"] = s
    evidence["defensive_event_wing"] = [f"disruption: {v('defensive_disruption'):.0f}", f"coverage: {v('defensive_coverage_range'):.0f}"]

    # Point-of-Attack Guard
    s = v("defensive_disruption") * 0.3 + v("defensive_coverage_range") * 0.25 + v("ball_security") * 0.2 + v("on_ball_creation") * 0.15 + v("transition_value") * 0.1
    scores["point_of_attack_guard"] = s
    evidence["point_of_attack_guard"] = [f"disruption: {v('defensive_disruption'):.0f}", f"on_ball: {v('on_ball_creation'):.0f}"]

    # Rim-Running Big
    s = v("rim_pressure") * 0.35 + v("rebounding_value") * 0.25 + v("physical_translation") * 0.2 + v("rim_protection") * 0.2
    scores["rim_running_big"] = s
    evidence["rim_running_big"] = [f"rim_pressure: {v('rim_pressure'):.0f}", f"rebounding: {v('rebounding_value'):.0f}"]

    # Stretch Big
    s = v("spacing_gravity") * 0.35 + v("rebounding_value") * 0.25 + v("rim_protection") * 0.2 + v("above_break_spacing_value") * 0.2
    scores["stretch_big"] = s
    evidence["stretch_big"] = [f"spacing: {v('spacing_gravity'):.0f}", f"rebounding: {v('rebounding_value'):.0f}"]

    # Passing Big
    s = v("passing_creation") * 0.3 + v("decision_quality") * 0.25 + v("rebounding_value") * 0.2 + v("rim_protection") * 0.15 + v("off_ball_scalability") * 0.1
    scores["passing_big"] = s
    evidence["passing_big"] = [f"passing: {v('passing_creation'):.0f}", f"decisions: {v('decision_quality'):.0f}"]

    # Defensive Event Big
    s = v("rim_protection") * 0.35 + v("defensive_disruption") * 0.25 + v("rebounding_value") * 0.2 + v("physical_translation") * 0.2
    scores["defensive_event_big"] = s
    evidence["defensive_event_big"] = [f"rim_protect: {v('rim_protection'):.0f}", f"rebounding: {v('rebounding_value'):.0f}"]

    # Interior Anchor
    s = v("rim_protection") * 0.3 + v("rebounding_value") * 0.3 + v("physical_translation") * 0.2 + v("defensive_coverage_range") * 0.2
    scores["interior_anchor"] = s
    evidence["interior_anchor"] = [f"rim_protect: {v('rim_protection'):.0f}", f"physical: {v('physical_translation'):.0f}"]

    # Hybrid Forward
    s = (v("rim_pressure") + v("spacing_gravity") + v("defensive_disruption") + v("rebounding_value") + v("passing_creation")) / 5.0
    scores["hybrid_forward"] = s
    evidence["hybrid_forward"] = [f"balanced across {sum(1 for d in ['rim_pressure','spacing_gravity','defensive_disruption'] if v(d) > 30)} dimensions"]

    # Normalize to probabilities
    total = sum(max(0, s) for s in scores.values())
    if total <= 0:
        total = 1.0

    role_scores = []
    for role in ROLE_REGIONS:
        prob = max(0, scores.get(role, 0)) / total
        conf = vector.confidence_summary()["average_confidence"]
        role_scores.append(RoleScore(
            role=role,
            probability=prob,
            confidence=conf,
            evidence=evidence.get(role, []),
        ))

    # Sort by probability
    role_scores.sort(key=lambda r: r.probability, reverse=True)
    dist.roles = role_scores
    dist.primary_role = role_scores[0].role if role_scores else ""
    dist.secondary_role = role_scores[1].role if len(role_scores) > 1 else ""

    # Build envelope
    dist.envelope = {
        "proven_actions": [r.role for r in role_scores if r.probability > 0.15],
        "viable_actions": [r.role for r in role_scores if 0.08 < r.probability <= 0.15],
        "risky_actions": [r.role for r in role_scores if 0.04 < r.probability <= 0.08],
        "unproven_actions": [r.role for r in role_scores if r.probability <= 0.04],
    }

    return dist
