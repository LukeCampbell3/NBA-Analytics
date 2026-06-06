"""
Spacing Ecology System

Evaluates how a lineup's geometry opens or compresses the floor.

Computes:
  spacing_density, spacing_area, paint_congestion,
  lane_creation_context, spacing_fragility, spacing_redundancy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..features.spacing_features import PlayerSpacingProfile


@dataclass
class SpacingEcology:
    """Spacing ecology evaluation for a lineup or roster."""
    lineup_size: int = 0

    # Spacing density
    spacing_density: float = 0.0
    count_above_80: int = 0
    count_above_70: int = 0
    count_above_60: int = 0
    count_below_30: int = 0

    # Spacing area (spread across dimensions)
    spacing_area: float = 0.0

    # Paint congestion
    paint_congestion: float = 0.0
    paint_congestion_warning: bool = False

    # Lane creation context
    lane_creation_context: float = 0.0

    # Spacing fragility
    spacing_fragility: float = 0.0
    spacing_fragility_warning: bool = False

    # Spacing redundancy
    spacing_redundancy: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineup_size": self.lineup_size,
            "spacing_density": round(self.spacing_density, 1),
            "count_above_80": self.count_above_80,
            "count_above_70": self.count_above_70,
            "count_above_60": self.count_above_60,
            "count_below_30": self.count_below_30,
            "spacing_area": round(self.spacing_area, 1),
            "paint_congestion": round(self.paint_congestion, 1),
            "paint_congestion_warning": self.paint_congestion_warning,
            "lane_creation_context": round(self.lane_creation_context, 1),
            "spacing_fragility": round(self.spacing_fragility, 1),
            "spacing_fragility_warning": self.spacing_fragility_warning,
            "spacing_redundancy": round(self.spacing_redundancy, 1),
            "warnings": self.warnings,
        }


def evaluate_spacing_ecology(
    players: List[PlayerSpacingProfile],
    creator_indices: List[int] = None,
) -> SpacingEcology:
    """Evaluate spacing ecology for a lineup.

    Args:
        players: List of PlayerSpacingProfile for the lineup (typically 5)
        creator_indices: Indices of primary creators in the lineup

    Returns:
        SpacingEcology with all sub-scores computed
    """
    eco = SpacingEcology(lineup_size=len(players))
    if not players:
        return eco

    gravities = [p.spacing_gravity for p in players]
    n = len(gravities)

    # 1. Spacing density
    eco.count_above_80 = sum(1 for g in gravities if g >= 80)
    eco.count_above_70 = sum(1 for g in gravities if g >= 70)
    eco.count_above_60 = sum(1 for g in gravities if g >= 60)
    eco.count_below_30 = sum(1 for g in gravities if g < 30)
    eco.spacing_density = float(np.mean(gravities))

    # Bonus for 3+ reliable spacers
    if eco.count_above_60 >= 3:
        eco.spacing_density = min(100, eco.spacing_density + 8)

    # 2. Spacing area (spread/variance of spacing types)
    corner_vals = [p.corner_spacing_value for p in players]
    ab_vals = [p.above_break_spacing_value for p in players]
    pull_vals = [p.pull_up_spacing_pressure for p in players]
    # Area = how spread the spacing coverage is across dimensions
    eco.spacing_area = float(np.std(gravities) * 2 + np.mean([
        np.std(corner_vals), np.std(ab_vals), np.std(pull_vals)
    ]) * 3)
    eco.spacing_area = float(np.clip(eco.spacing_area, 0, 100))

    # 3. Paint congestion
    non_shooters_with_rim = sum(
        1 for p in players if p.is_non_shooter() and p.rim_frequency > 0.35
    )
    eco.paint_congestion = float(np.clip(non_shooters_with_rim * 25, 0, 100))
    if non_shooters_with_rim >= 2:
        eco.paint_congestion_warning = True
        eco.warnings.append(
            f"paint_congestion: {non_shooters_with_rim} non-shooters with high rim frequency"
        )

    # 4. Lane creation context
    if creator_indices is None:
        # Detect creators: players with high pull-up or high volume
        creator_indices = [i for i, p in enumerate(players) if p.pull_up_spacing_pressure > 40]

    if creator_indices:
        # How much spacing surrounds the creators
        non_creator_spacing = [gravities[i] for i in range(n) if i not in creator_indices]
        if non_creator_spacing:
            eco.lane_creation_context = float(np.mean(non_creator_spacing))
        else:
            eco.lane_creation_context = eco.spacing_density
    else:
        eco.lane_creation_context = eco.spacing_density * 0.7

    # 5. Spacing fragility
    # If removing the best spacer drops density significantly
    if n >= 2:
        sorted_g = sorted(gravities, reverse=True)
        without_best = sorted_g[1:]
        drop = np.mean(gravities) - np.mean(without_best)
        eco.spacing_fragility = float(np.clip(drop * 3, 0, 100))
        if eco.count_above_60 <= 1 and eco.spacing_density > 45:
            eco.spacing_fragility_warning = True
            eco.warnings.append("spacing_fragility: spacing depends heavily on one player")

    # 6. Spacing redundancy (repeated spacers = additive, repeated non-shooters = conflict)
    if eco.count_above_60 >= 3:
        eco.spacing_redundancy = 20.0  # Positive: redundant shooting is good
    if eco.count_below_30 >= 2:
        eco.spacing_redundancy -= 15.0  # Negative: redundant non-shooters is bad
        eco.warnings.append("non_shooter_conflict: 2+ players below 30 spacing gravity")

    return eco
