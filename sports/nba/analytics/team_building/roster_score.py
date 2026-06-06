"""
Roster Score

RosterScore = CoverageArea + UsefulDensity + SpacingDensity + LaneCreationContext
            + Complementarity + PeakTalent + DepthResilience
            - FatalHolePenalty - ConflictDensityPenalty - PaintCongestionPenalty
            - SpacingFragilityPenalty - NonShooterConflictPenalty - SalaryPenalty - RiskPenalty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..schema import PlayerCapabilityVector, CAPABILITY_DIMENSIONS
from ..features.spacing_features import PlayerSpacingProfile, compute_player_spacing
from .spacing_ecology import SpacingEcology, evaluate_spacing_ecology


@dataclass
class RosterScoreResult:
    """Full roster evaluation result."""
    roster_score: float = 0.0

    # Positive components
    coverage_area: float = 0.0
    useful_density: float = 0.0
    spacing_density: float = 0.0
    lane_creation_context: float = 0.0
    complementarity: float = 0.0
    peak_talent: float = 0.0
    depth_resilience: float = 0.0

    # Penalties
    fatal_hole_penalty: float = 0.0
    conflict_density_penalty: float = 0.0
    paint_congestion_penalty: float = 0.0
    spacing_fragility_penalty: float = 0.0
    non_shooter_conflict_penalty: float = 0.0
    salary_penalty: float = 0.0
    risk_penalty: float = 0.0

    # Details
    fatal_holes: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    spacing_ecology: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def compute_total(self) -> float:
        self.roster_score = (
            self.coverage_area
            + self.useful_density
            + self.spacing_density
            + self.lane_creation_context
            + self.complementarity
            + self.peak_talent
            + self.depth_resilience
            - self.fatal_hole_penalty
            - self.conflict_density_penalty
            - self.paint_congestion_penalty
            - self.spacing_fragility_penalty
            - self.non_shooter_conflict_penalty
            - self.salary_penalty
            - self.risk_penalty
        )
        return self.roster_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roster_score": round(self.roster_score, 1),
            "components": {
                "coverage_area": round(self.coverage_area, 1),
                "useful_density": round(self.useful_density, 1),
                "spacing_density": round(self.spacing_density, 1),
                "lane_creation_context": round(self.lane_creation_context, 1),
                "complementarity": round(self.complementarity, 1),
                "peak_talent": round(self.peak_talent, 1),
                "depth_resilience": round(self.depth_resilience, 1),
            },
            "penalties": {
                "fatal_hole": round(self.fatal_hole_penalty, 1),
                "conflict_density": round(self.conflict_density_penalty, 1),
                "paint_congestion": round(self.paint_congestion_penalty, 1),
                "spacing_fragility": round(self.spacing_fragility_penalty, 1),
                "non_shooter_conflict": round(self.non_shooter_conflict_penalty, 1),
                "salary": round(self.salary_penalty, 1),
                "risk": round(self.risk_penalty, 1),
            },
            "fatal_holes": self.fatal_holes,
            "conflicts": self.conflicts,
            "spacing_ecology": self.spacing_ecology,
            "warnings": self.warnings,
        }


def score_roster(
    vectors: List[PlayerCapabilityVector],
    spacing_profiles: List[PlayerSpacingProfile] = None,
    salary_total: float = 0.0,
    cap_limit: float = 141.0,
) -> RosterScoreResult:
    """Score a roster of players.

    Args:
        vectors: List of PlayerCapabilityVectors
        spacing_profiles: Optional pre-computed spacing profiles
        salary_total: Total salary of roster
        cap_limit: Cap limit for salary penalty
    """
    result = RosterScoreResult()
    n = len(vectors)
    if n == 0:
        return result

    # 1. Coverage Area: how much capability space is covered
    all_vals = []
    for v in vectors:
        row = []
        for dim in CAPABILITY_DIMENSIONS:
            d = v.get(dim)
            row.append(d.raw_value if d.raw_value is not None else 0.0)
        all_vals.append(row)
    arr = np.array(all_vals)
    # Area = range covered per dimension, averaged
    ranges = np.ptp(arr, axis=0)
    result.coverage_area = float(np.mean(ranges))

    # 2. Useful Density: average capability level
    means = np.mean(arr, axis=0)
    result.useful_density = float(np.mean(means))

    # 3. Peak Talent: best player's average dimension
    player_means = np.mean(arr, axis=1)
    result.peak_talent = float(np.max(player_means)) if len(player_means) > 0 else 0

    # 4. Spacing ecology
    if spacing_profiles and len(spacing_profiles) >= min(n, 5):
        eco = evaluate_spacing_ecology(spacing_profiles[:5])
        result.spacing_density = eco.spacing_density
        result.lane_creation_context = eco.lane_creation_context
        result.spacing_ecology = eco.to_dict()

        if eco.paint_congestion_warning:
            result.paint_congestion_penalty = eco.paint_congestion * 0.5
        if eco.spacing_fragility_warning:
            result.spacing_fragility_penalty = eco.spacing_fragility * 0.4

        result.warnings.extend(eco.warnings)
    else:
        # Estimate from vectors
        spacing_vals = [v.get("spacing_gravity").raw_value or 0 for v in vectors]
        result.spacing_density = float(np.mean(spacing_vals))

    # 5. Fatal holes
    def _check_hole(dim: str, threshold: float, label: str):
        best = max((v.get(dim).raw_value or 0) for v in vectors)
        if best < threshold:
            result.fatal_holes.append(f"{label} (best={best:.0f}, need>{threshold:.0f})")

    _check_hole("on_ball_creation", 40, "no_creator")
    _check_hole("rim_protection", 15, "no_rim_protector")
    _check_hole("spacing_gravity", 30, "insufficient_spacing")
    _check_hole("defensive_disruption", 15, "no_defensive_disruption")

    result.fatal_hole_penalty = len(result.fatal_holes) * 15

    # 6. Conflict density
    # Too many high-usage non-shooters
    high_usage_non_shooters = sum(
        1 for v in vectors
        if (v.get("on_ball_creation").raw_value or 0) > 40 and (v.get("spacing_gravity").raw_value or 0) < 30
    )
    if high_usage_non_shooters >= 2:
        result.conflicts.append(f"high_usage_non_shooters: {high_usage_non_shooters}")
        result.conflict_density_penalty += high_usage_non_shooters * 8

    # Non-shooter conflict
    non_shooters = sum(1 for v in vectors if (v.get("spacing_gravity").raw_value or 0) < 30)
    if non_shooters >= 3:
        result.non_shooter_conflict_penalty = (non_shooters - 2) * 10
        result.conflicts.append(f"non_shooter_overload: {non_shooters}")

    # 7. Complementarity: creator + spacer pairs
    creators = [v for v in vectors if (v.get("on_ball_creation").raw_value or 0) > 45]
    spacers = [v for v in vectors if (v.get("spacing_gravity").raw_value or 0) > 55]
    result.complementarity = min(len(creators), len(spacers)) * 8

    # 8. Depth resilience: how good is the drop-off
    if n >= 8:
        top5 = sorted(player_means, reverse=True)[:5]
        rest = sorted(player_means, reverse=True)[5:]
        if rest:
            drop = np.mean(top5) - np.mean(rest)
            result.depth_resilience = max(0, 30 - drop * 2)

    # 9. Salary penalty
    if salary_total > cap_limit:
        over = salary_total - cap_limit
        result.salary_penalty = over * 0.5

    # 10. Risk penalty
    risk_vals = [(v.get("risk").raw_value or 0) for v in vectors]
    result.risk_penalty = float(np.mean(risk_vals)) * 0.3

    result.compute_total()
    return result
