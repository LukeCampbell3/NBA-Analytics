"""Shot-zone classification and the drive/post routing-STATE
classification honesty boundary.

TWO DIFFERENT THINGS, DELIBERATELY KEPT SEPARATE:

1. SHOT-ZONE classification (RIM / SHORT_PAINT / MIDRANGE / CORNER_3 /
   ABOVE_BREAK_3 / ...) describes the RESULTING SHOT. This IS derivable
   from Basketball-Reference's real play-by-play text (a real reported
   distance in feet and a real 2pt/3pt flag) -- see
   ``classify_shot_zone_from_text``. It is DERIVED, not OBSERVED (we are
   computing a zone from a reported distance, not reading an x/y
   coordinate), and it CANNOT distinguish CORNER_3 from ABOVE_BREAK_3
   (bball-ref's play-by-play text does not report corner-vs-arc; that
   is a geometry-only distinction). Every 3-point shot this module
   classifies is therefore returned as ShotZone.ABOVE_BREAK_3 with an
   explicit caveat recorded in the classification result -- never
   silently guessed as one or the other.

2. ROUTING-STATE classification (SPRAY_3, RIM_FEED, CUT_RIM,
   WEAKSIDE_SKIP, ...) describes the ORIGIN ACTION that produced the
   pass -- was this off a drive? A post touch? A short roll? THIS
   REQUIRES KNOWING THE ORIGIN TOUCH, which no reachable source in this
   environment provides (see sources/bball_ref.py's module docstring --
   stats.nba.com's touch/tracking endpoints are unreachable, and
   play-by-play text alone never says "off a drive" or "off a post
   touch"). ``classify_routing_state`` therefore ALWAYS returns
   UNAVAILABLE -- this is not a placeholder to be filled in later with
   guesswork, it is the honest ceiling of what this data source can
   support. See docs/advantage-routing.md "Observed vs reconstructed
   fields" for the full accounting, and routing/common.py for how a
   real touch/tracking source would be wired in once one is reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models.schemas import EvidenceStatus, ShotZone

ROUTING_STATE_UNAVAILABLE_REASON = (
    "Origin-touch classification (drive vs. post touch vs. short roll vs. "
    "other) requires possession/touch-level tracking data. stats.nba.com "
    "(the only reachable-in-principle source for this) is unreachable from "
    "this environment (verified: every live endpoint call times out with "
    "zero bytes), and no other real source publishes it. This field is "
    "left UNAVAILABLE rather than guessed."
)


@dataclass(frozen=True)
class ShotZoneClassification:
    zone: str  # ShotZone value
    status: str  # EvidenceStatus value
    method: str
    caveat: Optional[str] = None


def classify_shot_zone_from_text(shot_description: str, distance_ft: Optional[float], is_three: bool) -> ShotZoneClassification:
    """Real-text-based shot-zone classification -- the coarser cousin of
    geometry-based classification (section 9's preferred method, which
    needs real x/y coordinates this pipeline does not have)."""
    description = shot_description.lower()

    if is_three:
        return ShotZoneClassification(
            zone=ShotZone.ABOVE_BREAK_3.value,
            status=EvidenceStatus.DERIVED.value,
            method="bball_ref_pbp_text:3pt_flag",
            caveat="Basketball-Reference play-by-play text does not report corner-vs-above-break; every real 3PA is classified ABOVE_BREAK_3 by convention, not because the corner was ruled out.",
        )

    if distance_ft is None:
        return ShotZoneClassification(zone=ShotZone.MIDRANGE.value, status=EvidenceStatus.DERIVED.value, method="bball_ref_pbp_text:no_distance_reported_default_midrange")

    if "dunk" in description or distance_ft <= 2:
        return ShotZoneClassification(zone=ShotZone.RIM.value, status=EvidenceStatus.DERIVED.value, method="bball_ref_pbp_text:distance_le_2ft_or_dunk")
    if distance_ft <= 3:
        return ShotZoneClassification(zone=ShotZone.RIM.value, status=EvidenceStatus.DERIVED.value, method="bball_ref_pbp_text:distance_le_3ft")
    if distance_ft <= 10:
        return ShotZoneClassification(zone=ShotZone.SHORT_PAINT.value, status=EvidenceStatus.DERIVED.value, method="bball_ref_pbp_text:distance_3_to_10ft")
    return ShotZoneClassification(zone=ShotZone.MIDRANGE.value, status=EvidenceStatus.DERIVED.value, method="bball_ref_pbp_text:distance_gt_10ft")


def classify_routing_state(*, event_type: Optional[str] = None) -> ShotZoneClassification:
    """Always UNAVAILABLE in the current data environment -- see module
    docstring. Kept as a real function (not a bare constant) so that
    once a real touch/tracking source is wired into
    sources/, this is the one place to implement the actual
    classification without changing every caller's contract."""
    return ShotZoneClassification(
        zone="UNAVAILABLE", status=EvidenceStatus.UNAVAILABLE.value,
        method="none", caveat=ROUTING_STATE_UNAVAILABLE_REASON,
    )
