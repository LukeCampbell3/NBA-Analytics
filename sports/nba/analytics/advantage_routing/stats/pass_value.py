"""Expected pass value (spec section 13).

    PassValue = E[points | resulting state] - E[points | baseline state]

This pipeline cannot support true possession-level EPV (that needs
play-type/possession tracking this environment cannot reach -- see
sources/bball_ref.py). What IS real and reachable: Basketball-
Reference's real, single-request league-average FG% AND real league
shot-selection frequency by distance zone
(sources.bball_ref.fetch_league_shooting_baseline). This module uses
that real league reference table to build:

    E[points | zone]      = league_fg_pct[zone] * points_value[zone]
    E[points | baseline]  = sum_z( league_freq[z] * E[points | z] )
                             -- i.e. the real league-average points per
                             shot attempt, zone-selection-weighted.

    AddedPassValue(zone)  = E[points | zone] - E[points | baseline]

This is an "empirical state expectations" model exactly as section 13
calls a "practical first version" -- built entirely from real,
OBSERVED league shooting data, never a possession-level EPV claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models.schemas import EvidenceStatus, Metric, ShotZone
from ..sources.bball_ref import LeagueShootingBaseline

ZONE_POINT_VALUE = {
    ShotZone.RIM.value: 2.0,
    ShotZone.SHORT_PAINT.value: 2.0,
    ShotZone.MIDRANGE.value: 2.0,
    ShotZone.CORNER_3.value: 3.0,
    ShotZone.ABOVE_BREAK_3.value: 3.0,
    ShotZone.PULLUP_3.value: 3.0,
}

_ZONE_TO_BASELINE_FIELDS = {
    ShotZone.RIM.value: ("fg_pct_rim", "freq_rim"),
    ShotZone.SHORT_PAINT.value: ("fg_pct_short_paint", "freq_short_paint"),
    ShotZone.MIDRANGE.value: ("fg_pct_midrange", "freq_midrange"),
    ShotZone.ABOVE_BREAK_3.value: ("fg_pct_three", "freq_three"),
    ShotZone.CORNER_3.value: ("fg_pct_three", "freq_three"),
    ShotZone.PULLUP_3.value: ("fg_pct_three", "freq_three"),
}


@dataclass
class PassValueModel:
    season: str
    expected_points_by_zone: dict[str, Metric]
    baseline_expected_points: Metric
    added_pass_value_by_zone: dict[str, Metric]

    def as_dict(self) -> dict:
        return {
            "season": self.season,
            "expected_points_by_zone": {z: m.as_dict() for z, m in self.expected_points_by_zone.items()},
            "baseline_expected_points": self.baseline_expected_points.as_dict(),
            "added_pass_value_by_zone": {z: m.as_dict() for z, m in self.added_pass_value_by_zone.items()},
        }


def build_pass_value_model(baseline: Optional[LeagueShootingBaseline], season: str) -> PassValueModel:
    if baseline is None:
        reason = "No real Basketball-Reference league shooting baseline was reachable for this season."
        return PassValueModel(
            season=season,
            expected_points_by_zone={z: Metric.unavailable(f"expected_points_{z}", reason=reason) for z in ZONE_POINT_VALUE},
            baseline_expected_points=Metric.unavailable("baseline_expected_points", reason=reason),
            added_pass_value_by_zone={z: Metric.unavailable(f"added_pass_value_{z}", reason=reason) for z in ZONE_POINT_VALUE},
        )

    field_map = {
        "fg_pct_rim": baseline.fg_pct_rim, "fg_pct_short_paint": baseline.fg_pct_short_paint,
        "fg_pct_midrange": baseline.fg_pct_midrange, "fg_pct_three": baseline.fg_pct_three,
    }
    freq_map = {
        "freq_rim": baseline.freq_rim, "freq_short_paint": baseline.freq_short_paint,
        "freq_midrange": baseline.freq_midrange, "freq_three": baseline.freq_three,
    }

    expected_points_by_zone: dict[str, Metric] = {}
    ep_values: dict[str, float] = {}
    for zone, points_value in ZONE_POINT_VALUE.items():
        fg_field, _ = _ZONE_TO_BASELINE_FIELDS[zone]
        fg_pct = field_map.get(fg_field)
        ep = (fg_pct * points_value) if fg_pct is not None else None
        expected_points_by_zone[zone] = Metric.derived(
            f"expected_points_{zone}", ep,
            method=f"real league FG% at this zone * shot value ({points_value:g} pts)", season=season,
        )
        if ep is not None:
            ep_values[zone] = ep

    # Real, frequency-weighted league-average points per shot attempt.
    weighted_sum = 0.0
    weight_total = 0.0
    for zone in (ShotZone.RIM.value, ShotZone.SHORT_PAINT.value, ShotZone.MIDRANGE.value, ShotZone.ABOVE_BREAK_3.value):
        _, freq_field = _ZONE_TO_BASELINE_FIELDS[zone]
        freq = freq_map.get(freq_field)
        ep = ep_values.get(zone)
        if freq is not None and ep is not None:
            weighted_sum += freq * ep
            weight_total += freq
    baseline_ep = (weighted_sum / weight_total) if weight_total else None
    baseline_metric = Metric.derived(
        "baseline_expected_points", baseline_ep,
        method="sum(real league zone-selection frequency * expected points at zone) / sum(frequency) -- real league-average points per shot attempt", season=season,
    )

    added_pass_value_by_zone: dict[str, Metric] = {}
    for zone, ep_metric in expected_points_by_zone.items():
        added = (ep_metric.value - baseline_ep) if (ep_metric.value is not None and baseline_ep is not None) else None
        added_pass_value_by_zone[zone] = Metric.derived(
            f"added_pass_value_{zone}", added,
            method="E[points | zone] - E[points | league-average shot attempt], both from real league shooting data", season=season,
        )

    return PassValueModel(
        season=season, expected_points_by_zone=expected_points_by_zone,
        baseline_expected_points=baseline_metric, added_pass_value_by_zone=added_pass_value_by_zone,
    )
