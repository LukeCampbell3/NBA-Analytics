"""
PlayerCapabilityVector Schema

Each player is a multidimensional capability vector.
The vector is the truth. Roles are summaries.

Every dimension includes:
  raw_value, raw_percentile, position_percentile, role_percentile,
  context_adjusted_percentile, reliability_adjusted_percentile,
  sample_size, confidence, source, observation_status, stale_flag
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNAVAILABLE = "unavailable"


@dataclass
class VectorDimension:
    """A single dimension of the capability vector."""
    name: str
    raw_value: Optional[float] = None
    raw_percentile: Optional[float] = None
    position_percentile: Optional[float] = None
    role_percentile: Optional[float] = None
    context_adjusted_percentile: Optional[float] = None
    reliability_adjusted_percentile: Optional[float] = None
    sample_size: int = 0
    confidence: float = 0.0
    source: str = ""
    observation_status: ObservationStatus = ObservationStatus.UNAVAILABLE
    stale_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_value": self.raw_value,
            "raw_percentile": self.raw_percentile,
            "position_percentile": self.position_percentile,
            "role_percentile": self.role_percentile,
            "context_adjusted_percentile": self.context_adjusted_percentile,
            "reliability_adjusted_percentile": self.reliability_adjusted_percentile,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "source": self.source,
            "observation_status": self.observation_status.value,
            "stale_flag": self.stale_flag,
        }


# All 22 capability vector dimensions
CAPABILITY_DIMENSIONS = [
    "on_ball_creation",
    "self_scoring_efficiency",
    "rim_pressure",
    "shooting_gravity",
    "spacing_gravity",
    "corner_spacing_value",
    "above_break_spacing_value",
    "catch_and_shoot_gravity",
    "pull_up_spacing_pressure",
    "off_ball_scalability",
    "passing_creation",
    "decision_quality",
    "ball_security",
    "transition_value",
    "defensive_disruption",
    "defensive_coverage_range",
    "rim_protection",
    "rebounding_value",
    "physical_translation",
    "competition_translation",
    "upside",
    "risk",
]


@dataclass
class PlayerCapabilityVector:
    """Full capability vector for a player."""
    player_id: str = ""
    player_name: str = ""
    team: str = ""
    position: str = ""
    season: int = 2026
    dimensions: Dict[str, VectorDimension] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure all dimensions exist
        for dim_name in CAPABILITY_DIMENSIONS:
            if dim_name not in self.dimensions:
                self.dimensions[dim_name] = VectorDimension(name=dim_name)

    def get(self, dim_name: str) -> VectorDimension:
        return self.dimensions.get(dim_name, VectorDimension(name=dim_name))

    def set_dimension(self, dim_name: str, **kwargs) -> None:
        if dim_name not in self.dimensions:
            self.dimensions[dim_name] = VectorDimension(name=dim_name)
        for k, v in kwargs.items():
            if hasattr(self.dimensions[dim_name], k):
                setattr(self.dimensions[dim_name], k, v)

    def confidence_summary(self) -> Dict[str, float]:
        observed = sum(1 for d in self.dimensions.values() if d.observation_status == ObservationStatus.OBSERVED)
        inferred = sum(1 for d in self.dimensions.values() if d.observation_status == ObservationStatus.INFERRED)
        unavail = sum(1 for d in self.dimensions.values() if d.observation_status == ObservationStatus.UNAVAILABLE)
        total = len(self.dimensions)
        avg_confidence = sum(d.confidence for d in self.dimensions.values()) / max(total, 1)
        return {
            "observed_dimensions": observed,
            "inferred_dimensions": inferred,
            "unavailable_dimensions": unavail,
            "total_dimensions": total,
            "average_confidence": round(avg_confidence, 3),
            "data_coverage": round(observed / max(total, 1), 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "position": self.position,
            "season": self.season,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "confidence_summary": self.confidence_summary(),
            "metadata": self.metadata,
        }
