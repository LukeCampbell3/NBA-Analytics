"""Drive-pass analysis (spec section 3). Thin, mode-specific entry point
over routing/common.py's shared advantage-routing pipeline -- drive and
post share the same underlying machinery (state vector shape, advantage-
pass metrics), differing only in their state vocabulary and the eligible-
touch definition.

DRIVES / DRIVE_PASSES / DRIVE_ASSISTS / DRIVE_TOV / PASS % OF DRIVES /
AST % OF DRIVES: all require a real drive-detection signal (a touch that
began with dribble penetration into the paint/a closeout). That is a
stats.nba.com PlayerDashPtShotLog/Drives tracking endpoint -- unreachable
in this environment (see sources/bball_ref.py). This module therefore
returns real UNAVAILABLE Metrics for every drive-specific count and
RESERVES the shared, real recipient-network/gravity/pass-value pipeline
(built from real assist play-by-play, which is drive-agnostic) as the
substantive real signal available today for a drive-oriented player.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.schemas import DriveState, Metric
from .common import AdvantagePassMetrics, RoutingVector, build_routing_vector, compute_advantage_pass_metrics

DRIVE_DATA_UNAVAILABLE_REASON = (
    "Drive detection (dribble penetration into the paint/a closeout) "
    "requires stats.nba.com PlayerDashPtShotLog/Drives tracking data, "
    "which is unreachable from this environment. No proxy is invented."
)


@dataclass
class DriveProfile:
    drives: Metric
    drive_passes: Metric
    drive_assists: Metric
    drive_turnovers: Metric
    pass_pct_of_drives: Metric
    ast_pct_of_drives: Metric
    ast_per_drive_pass: Metric
    tov_per_drive: Metric
    tov_per_drive_pass: Metric
    routing_vector: RoutingVector
    advantage_metrics: AdvantagePassMetrics

    def as_dict(self) -> dict:
        return {
            "drives": self.drives.as_dict(),
            "drive_passes": self.drive_passes.as_dict(),
            "drive_assists": self.drive_assists.as_dict(),
            "drive_turnovers": self.drive_turnovers.as_dict(),
            "pass_pct_of_drives": self.pass_pct_of_drives.as_dict(),
            "ast_pct_of_drives": self.ast_pct_of_drives.as_dict(),
            "ast_per_drive_pass": self.ast_per_drive_pass.as_dict(),
            "tov_per_drive": self.tov_per_drive.as_dict(),
            "tov_per_drive_pass": self.tov_per_drive_pass.as_dict(),
            "routing_vector": self.routing_vector.as_dict(),
            "advantage_metrics": self.advantage_metrics.as_dict(),
        }


def build_drive_profile() -> DriveProfile:
    unavailable = lambda name: Metric.unavailable(name, reason=DRIVE_DATA_UNAVAILABLE_REASON)
    states = [s.value for s in DriveState]
    return DriveProfile(
        drives=unavailable("drives"),
        drive_passes=unavailable("drive_passes"),
        drive_assists=unavailable("drive_assists"),
        drive_turnovers=unavailable("drive_turnovers"),
        pass_pct_of_drives=unavailable("pass_pct_of_drives"),
        ast_pct_of_drives=unavailable("ast_pct_of_drives"),
        ast_per_drive_pass=unavailable("ast_per_drive_pass"),
        tov_per_drive=unavailable("tov_per_drive"),
        tov_per_drive_pass=unavailable("tov_per_drive_pass"),
        routing_vector=build_routing_vector("drive", states),
        advantage_metrics=compute_advantage_pass_metrics("drive"),
    )
