"""Shared advantage-routing pipeline (spec section 2/33): the one place
that assembles a player's baseline usage, recipient network, gravity
profile, and advantage-pass metrics, so drive.py and post.py can both
call into it rather than duplicating logic.

ADVANTAGE_PASS RATE (spec section 12): defined as
advantage passes / eligible passes, where an "advantage pass" is one
landing in a specific ORIGIN-ROUTING state (CUT_RIM, RIM_FEED,
SECONDARY_DRIVE, ...). Computing it requires exactly the touch/origin
classification that routing/states.classify_routing_state documents as
UNAVAILABLE in this data environment. `compute_advantage_pass_metrics`
below is therefore a real, honest, always-UNAVAILABLE result right now
-- not a placeholder quietly computing something else under the same
name. Once a real touch/tracking source is wired into sources/, this
is the one function to update; everything downstream (build/build_player.py,
the frontend) already reads its Metric.status field rather than assuming
a value exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.schemas import EvidenceStatus, Metric
from .states import ROUTING_STATE_UNAVAILABLE_REASON


@dataclass
class AdvantagePassMetrics:
    mode: str  # "drive" | "post"
    advantage_pass_rate: Metric
    advantage_creation_rate: Metric
    advantage_conversion: Metric
    high_value_shot_rate: Metric

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "advantage_pass_rate": self.advantage_pass_rate.as_dict(),
            "advantage_creation_rate": self.advantage_creation_rate.as_dict(),
            "advantage_conversion": self.advantage_conversion.as_dict(),
            "high_value_shot_rate": self.high_value_shot_rate.as_dict(),
        }


def compute_advantage_pass_metrics(mode: str) -> AdvantagePassMetrics:
    """Honest placeholder -- see module docstring. Always returns
    UNAVAILABLE Metrics with the exact reason, for both "drive" and
    "post" modes, until a real touch/tracking source is reachable."""
    reason = f"{ROUTING_STATE_UNAVAILABLE_REASON} (mode={mode})"
    return AdvantagePassMetrics(
        mode=mode,
        advantage_pass_rate=Metric.unavailable(f"{mode}_advantage_pass_rate", reason=reason),
        advantage_creation_rate=Metric.unavailable(f"{mode}_advantage_creation_rate", reason=reason),
        advantage_conversion=Metric.unavailable(f"{mode}_advantage_conversion", reason=reason),
        high_value_shot_rate=Metric.unavailable(f"{mode}_high_value_shot_rate", reason=reason),
    )


@dataclass
class RoutingVector:
    """Section 6's routing tendency vector. Always UNAVAILABLE right now
    (see states.py) -- kept as a real, typed object (not silently
    omitted) so the frontend has a stable contract to render an honest
    'no real touch-tracking source' state from, per section 39's 'no
    NaN/undefined under any state' requirement."""

    mode: str
    states: list[str]
    probabilities: dict[str, Metric]

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "states": self.states,
            "probabilities": {s: m.as_dict() for s, m in self.probabilities.items()},
            "status": EvidenceStatus.UNAVAILABLE.value,
            "reason": ROUTING_STATE_UNAVAILABLE_REASON,
        }


def build_routing_vector(mode: str, states: list[str]) -> RoutingVector:
    probabilities = {s: Metric.unavailable(f"{mode}_{s.lower()}_probability", reason=ROUTING_STATE_UNAVAILABLE_REASON) for s in states}
    return RoutingVector(mode=mode, states=states, probabilities=probabilities)
