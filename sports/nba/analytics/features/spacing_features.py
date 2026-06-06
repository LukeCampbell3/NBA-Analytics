"""
Spacing Features

Player-level spacing metrics derived from shooting profile.
Spacing is a first-class subsystem, not just 3P%.

Player spacing features:
  spacing_gravity_percentile, shooting_volume_gravity, shooting_accuracy_gravity,
  corner_spacing_value, above_break_spacing_value, catch_and_shoot_gravity,
  pull_up_spacing_pressure, off_ball_spacing_value, floor_spacer_reliability,
  non_shooter_risk, paint_congestion_risk, lane_opener_value
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PlayerSpacingProfile:
    """All spacing-related metrics for a single player."""
    player_id: str = ""
    player_name: str = ""

    # Raw shooting inputs
    three_pa_rate: float = 0.0
    three_pct: float = 0.0
    ft_pct: float = 0.0
    three_pa_per_game: float = 0.0
    assisted_three_rate: float = 0.5
    unassisted_three_rate: float = 0.5
    rim_frequency: float = 0.3

    # Derived spacing metrics (0-100 scale)
    spacing_gravity: float = 0.0
    shooting_volume_gravity: float = 0.0
    shooting_accuracy_gravity: float = 0.0
    corner_spacing_value: float = 0.0
    above_break_spacing_value: float = 0.0
    catch_and_shoot_gravity: float = 0.0
    pull_up_spacing_pressure: float = 0.0
    off_ball_spacing_value: float = 0.0
    floor_spacer_reliability: float = 0.0
    non_shooter_risk: float = 0.0
    paint_congestion_risk: float = 0.0
    lane_opener_value: float = 0.0

    # Confidence
    confidence: float = 0.0
    sample_size: int = 0

    def is_reliable_spacer(self, threshold: float = 60.0) -> bool:
        return self.spacing_gravity >= threshold and self.confidence >= 0.5

    def is_non_shooter(self, threshold: float = 30.0) -> bool:
        return self.spacing_gravity < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "spacing_gravity": round(self.spacing_gravity, 1),
            "shooting_volume_gravity": round(self.shooting_volume_gravity, 1),
            "shooting_accuracy_gravity": round(self.shooting_accuracy_gravity, 1),
            "corner_spacing_value": round(self.corner_spacing_value, 1),
            "above_break_spacing_value": round(self.above_break_spacing_value, 1),
            "catch_and_shoot_gravity": round(self.catch_and_shoot_gravity, 1),
            "pull_up_spacing_pressure": round(self.pull_up_spacing_pressure, 1),
            "off_ball_spacing_value": round(self.off_ball_spacing_value, 1),
            "floor_spacer_reliability": round(self.floor_spacer_reliability, 1),
            "non_shooter_risk": round(self.non_shooter_risk, 1),
            "paint_congestion_risk": round(self.paint_congestion_risk, 1),
            "lane_opener_value": round(self.lane_opener_value, 1),
            "is_reliable_spacer": self.is_reliable_spacer(),
            "is_non_shooter": self.is_non_shooter(),
            "confidence": round(self.confidence, 3),
            "sample_size": self.sample_size,
        }


def compute_player_spacing(
    three_pct: float,
    three_pa_rate: float,
    three_pa_per_game: float,
    ft_pct: float,
    assisted_rate: float = 0.5,
    rim_frequency: float = 0.3,
    usage_rate: float = 0.2,
    games_played: int = 0,
    player_id: str = "",
    player_name: str = "",
) -> PlayerSpacingProfile:
    """Compute full spacing profile from available shooting data."""
    prof = PlayerSpacingProfile(player_id=player_id, player_name=player_name)
    prof.three_pct = three_pct
    prof.three_pa_rate = three_pa_rate
    prof.three_pa_per_game = three_pa_per_game
    prof.ft_pct = ft_pct
    prof.assisted_three_rate = assisted_rate
    prof.unassisted_three_rate = 1.0 - assisted_rate
    prof.rim_frequency = rim_frequency
    prof.sample_size = games_played
    prof.confidence = min(1.0, games_played / 50) * 0.85

    # Volume gravity: how much 3P volume the player creates
    prof.shooting_volume_gravity = float(np.clip(three_pa_per_game * 12, 0, 100))

    # Accuracy gravity: 3P% scaled
    prof.shooting_accuracy_gravity = float(np.clip((three_pct - 0.28) / 0.14 * 100, 0, 100))

    # Overall spacing gravity: combination of volume + accuracy + FT%
    prof.spacing_gravity = float(np.clip(
        0.45 * prof.shooting_accuracy_gravity +
        0.35 * prof.shooting_volume_gravity +
        0.20 * (ft_pct * 100),
        0, 100
    ))

    # Corner spacing: high assisted rate + good accuracy = corner threat
    prof.corner_spacing_value = float(np.clip(
        assisted_rate * prof.shooting_accuracy_gravity * 1.2, 0, 100
    ))

    # Above-break: unassisted + pull-up capability
    prof.above_break_spacing_value = float(np.clip(
        (1 - assisted_rate) * prof.shooting_accuracy_gravity * 1.1, 0, 100
    ))

    # Catch-and-shoot: assisted * accuracy * volume
    prof.catch_and_shoot_gravity = float(np.clip(
        assisted_rate * three_pct * three_pa_per_game * 20, 0, 100
    ))

    # Pull-up spacing: unassisted shooting + usage
    prof.pull_up_spacing_pressure = float(np.clip(
        (1 - assisted_rate) * three_pct * usage_rate * 300, 0, 100
    ))

    # Off-ball value: low usage + good shooting = scalable off-ball
    prof.off_ball_spacing_value = float(np.clip(
        (1 - usage_rate) * prof.spacing_gravity * 1.5, 0, 100
    ))

    # Floor spacer reliability: consistency proxy (accuracy + volume together)
    prof.floor_spacer_reliability = float(np.clip(
        min(prof.shooting_accuracy_gravity, prof.shooting_volume_gravity) * 1.2, 0, 100
    ))

    # Non-shooter risk: inverse of spacing
    prof.non_shooter_risk = float(np.clip(100 - prof.spacing_gravity, 0, 100))

    # Paint congestion risk: high rim freq + low spacing = congestion risk
    prof.paint_congestion_risk = float(np.clip(
        rim_frequency * 100 + (100 - prof.spacing_gravity) * 0.5 - 30, 0, 100
    ))

    # Lane opener: how much this player opens driving lanes for teammates
    prof.lane_opener_value = float(np.clip(
        prof.spacing_gravity * 0.7 + prof.corner_spacing_value * 0.3, 0, 100
    ))

    return prof
