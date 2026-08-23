"""Post / interior-pass analysis (spec sections 4-5). Two related
analyses sharing routing/common.py's machinery:

  STRICT POST HUB  -- legitimate low-post touches only.
  INTERIOR HUB     -- post touches + elbow touches + nail/high-post
                       touches + short-roll catches + delay actions +
                       selected DHO initiation states (a Jokic-style hub
                       cannot be represented by low-post touches alone).

Both require the same touch-level origin signal drive.py's module
docstring already documents as unreachable in this environment
(stats.nba.com touch tracking). Both modes are therefore built with the
same real UNAVAILABLE-Metric discipline -- distinguished from each other
only by `mode` and their touch-type coverage description, so the
frontend can render both tabs honestly rather than only one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.schemas import Metric, PostState
from .common import AdvantagePassMetrics, RoutingVector, build_routing_vector, compute_advantage_pass_metrics

POST_DATA_UNAVAILABLE_REASON = (
    "Post-touch detection (low-post touches for STRICT_POST_HUB; post + "
    "elbow + nail/high-post + short-roll + delay + selected DHO touches "
    "for INTERIOR_HUB) requires stats.nba.com touch-tracking data, which "
    "is unreachable from this environment. No proxy is invented."
)

TOUCH_COVERAGE = {
    "STRICT_POST_HUB": "legitimate low-post touches only",
    "INTERIOR_HUB": "post + elbow + nail/high-post + short-roll catches + delay actions + selected DHO initiation states",
}


@dataclass
class PostProfile:
    hub_type: str  # "STRICT_POST_HUB" | "INTERIOR_HUB"
    touch_coverage: str
    post_touches: Metric
    post_passes: Metric
    post_assists: Metric
    post_turnovers: Metric
    pass_pct_of_touches: Metric
    ast_pct_of_post_pass: Metric
    routing_vector: RoutingVector
    advantage_metrics: AdvantagePassMetrics

    def as_dict(self) -> dict:
        return {
            "hub_type": self.hub_type,
            "touch_coverage": self.touch_coverage,
            "post_touches": self.post_touches.as_dict(),
            "post_passes": self.post_passes.as_dict(),
            "post_assists": self.post_assists.as_dict(),
            "post_turnovers": self.post_turnovers.as_dict(),
            "pass_pct_of_touches": self.pass_pct_of_touches.as_dict(),
            "ast_pct_of_post_pass": self.ast_pct_of_post_pass.as_dict(),
            "routing_vector": self.routing_vector.as_dict(),
            "advantage_metrics": self.advantage_metrics.as_dict(),
        }


def build_post_profile(hub_type: str = "INTERIOR_HUB") -> PostProfile:
    if hub_type not in TOUCH_COVERAGE:
        raise ValueError(f"hub_type must be one of {list(TOUCH_COVERAGE)}, got {hub_type!r}")
    unavailable = lambda name: Metric.unavailable(name, reason=POST_DATA_UNAVAILABLE_REASON)
    states = [s.value for s in PostState]
    return PostProfile(
        hub_type=hub_type,
        touch_coverage=TOUCH_COVERAGE[hub_type],
        post_touches=unavailable("post_touches"),
        post_passes=unavailable("post_passes"),
        post_assists=unavailable("post_assists"),
        post_turnovers=unavailable("post_turnovers"),
        pass_pct_of_touches=unavailable("pass_pct_of_touches"),
        ast_pct_of_post_pass=unavailable("ast_pct_of_post_pass"),
        routing_vector=build_routing_vector(hub_type.lower(), states),
        advantage_metrics=compute_advantage_pass_metrics(hub_type.lower()),
    )
