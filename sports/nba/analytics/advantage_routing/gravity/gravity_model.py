"""Gravity model (spec section 10) -- NOT raw FG%. Separates scoring
gravity into mechanisms, each built from real, zone/shot-type-level
Basketball-Reference data (real season shooting splits) plus real
box-score rates, never from touch/roll-man tracking (unreachable -- see
sources/bball_ref.py).

Honest mapping from the spec's six mechanisms to what real data
actually supports:

  PAINT_FACEUP_GRAVITY  -- real (zone FGA/FG% at 0-3ft + 3-10ft, real
                            FTA/FGA rate as a foul-drawing proxy). DERIVED.
  VERTICAL_GRAVITY      -- real (real Dunk FGA/FG% + real 0-3ft rim
                            FGA/FG%). DERIVED, strong signal (dunks are
                            an almost-unambiguous vertical-gravity tell).
  POP_GRAVITY           -- real (real season 3PA, real 3P%). DERIVED,
                            but cannot isolate catch-and-shoot from
                            off-the-dribble 3s -- that split is a
                            tracking-only quantity. Caveat recorded.
  PERIMETER_GRAVITY     -- real (real share of FGA taken from 10ft+,
                            i.e. all non-rim jump shooting, not just
                            3s). DERIVED, broader than POP_GRAVITY on
                            purpose -- see docstring below.
  POST_SCORING_GRAVITY  -- RECONSTRUCTED, moderate-low confidence: real
                            Hook Shot volume/efficiency (a real, if
                            imperfect, post-move signature) plus the
                            real complement of the short-paint assisted
                            rate (unassisted makes near the rim/short
                            paint skew toward self-created post scoring
                            rather than a set play). Method and
                            confidence recorded explicitly -- this is
                            the one mechanism where real proxy data
                            genuinely substitutes for the ideal
                            (touch-tracked) signal, and it is labeled
                            as such rather than presented as equal in
                            kind to VERTICAL_GRAVITY.
  SHORT_ROLL_GRAVITY    -- UNAVAILABLE. No real proxy exists for
                            roll-man frequency without ball-screen/
                            touch tracking; this pipeline does not
                            invent one. Explicit null, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models.schemas import GravityMechanism, Metric
from ..sources.bball_ref import SeasonShootingTable

SHORT_ROLL_UNAVAILABLE_REASON = (
    "Requires roll-man/ball-screen touch tracking (stats.nba.com SportVU/"
    "Synergy). No reachable source publishes this; no proxy is invented."
)


@dataclass
class GravityProfile:
    player_name: str
    season: str
    components: dict[str, dict]  # mechanism -> {metric_name: Metric}
    mechanisms_present: list[str]  # mechanisms with at least one real/derived value

    def as_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "season": self.season,
            "components": {
                mech: {name: metric.as_dict() for name, metric in metrics.items()}
                for mech, metrics in self.components.items()
            },
            "mechanisms_present": self.mechanisms_present,
        }


def _zone(table: Optional[SeasonShootingTable], label: str):
    if table is None:
        return None
    for z in table.zones:
        if z.zone_label == label:
            return z
    return None


def _shot_type(table: Optional[SeasonShootingTable], key: str):
    if table is None:
        return None
    for s in table.shot_types:
        if s.shot_type == key:
            return s
    return None


def build_gravity_profile(
    player_name: str,
    season: str,
    *,
    shooting_table: Optional[SeasonShootingTable],
    mean_fga_per_game: Optional[float],
    mean_fta_per_game: Optional[float],
    games_played: int,
) -> GravityProfile:
    components: dict[str, dict] = {}

    if shooting_table is None:
        for mech in GravityMechanism:
            components[mech.value] = {
                "summary": Metric.unavailable(f"{mech.value.lower()}_summary", reason="No real Basketball-Reference shooting table was reachable/parsed for this player/season.")
            }
        return GravityProfile(player_name=player_name, season=season, components=components, mechanisms_present=[])

    rim = _zone(shooting_table, "At Rim")
    short_paint = _zone(shooting_table, "3 to <10 ft")
    long_mid = _zone(shooting_table, "16 ft to <3-pt")
    mid = _zone(shooting_table, "10 to <16 ft")
    three = _zone(shooting_table, "3-pt")
    dunk = _shot_type(shooting_table, "DUNK")
    hook = _shot_type(shooting_table, "HOOK_SHOT")

    # --- PAINT_FACEUP_GRAVITY -------------------------------------
    paint_fga = (rim.fga if rim else 0) + (short_paint.fga if short_paint else 0)
    paint_fg = (rim.fg if rim else 0) + (short_paint.fg if short_paint else 0)
    paint_fg_pct = (paint_fg / paint_fga) if paint_fga else None
    ft_rate = (mean_fta_per_game / mean_fga_per_game) if (mean_fta_per_game and mean_fga_per_game) else None
    components[GravityMechanism.PAINT_FACEUP_GRAVITY.value] = {
        "paint_attempts_season": Metric.observed("paint_attempts_season", paint_fga, source="Basketball-Reference season shooting table", season=season),
        "paint_fg_pct": Metric.derived("paint_fg_pct", paint_fg_pct, method="(rim_fg + short_paint_fg) / (rim_fga + short_paint_fga), real zone totals", season=season),
        "rim_fg_pct_0_3ft": Metric.observed("rim_fg_pct_0_3ft", rim.fg_pct if rim else None, source="Basketball-Reference season shooting table", season=season),
        "ft_rate_proxy": Metric.derived("ft_rate_proxy", ft_rate, method="mean(FTA)/mean(FGA), real box-score per-game rates -- a foul-drawing proxy, not a play-type-specific foul rate", season=season, sample_size=games_played),
    }

    # --- VERTICAL_GRAVITY -------------------------------------------
    components[GravityMechanism.VERTICAL_GRAVITY.value] = {
        "dunk_attempts_season": Metric.observed("dunk_attempts_season", dunk.fga if dunk else None, source="Basketball-Reference season shooting table", season=season),
        "dunk_fg_pct": Metric.observed("dunk_fg_pct", dunk.fg_pct if dunk else None, source="Basketball-Reference season shooting table", season=season),
        "rim_attempts_season": Metric.observed("rim_attempts_season", rim.fga if rim else None, source="Basketball-Reference season shooting table", season=season),
        "rim_fg_pct": Metric.observed("rim_fg_pct", rim.fg_pct if rim else None, source="Basketball-Reference season shooting table", season=season),
    }

    # --- SHORT_ROLL_GRAVITY -- always unavailable ---------------------
    components[GravityMechanism.SHORT_ROLL_GRAVITY.value] = {
        "summary": Metric.unavailable("short_roll_gravity_summary", reason=SHORT_ROLL_UNAVAILABLE_REASON),
    }

    # --- POP_GRAVITY --------------------------------------------------
    three_pa_season = three.fga if three else None
    components[GravityMechanism.POP_GRAVITY.value] = {
        "three_pa_season": Metric.observed("three_pa_season", three_pa_season, source="Basketball-Reference season shooting table", season=season),
        "three_p_pct": Metric.observed("three_p_pct", three.fg_pct if three else None, source="Basketball-Reference season shooting table", season=season),
        "three_pa_share_of_fga": Metric.derived(
            "three_pa_share_of_fga", (three.fga / shooting_table.season_fga) if (three and shooting_table.season_fga) else None,
            method="season 3PA / season total FGA, real zone totals", season=season,
        ),
    }

    # --- PERIMETER_GRAVITY (broader: all 10ft+ jump shooting) --------
    perimeter_fga = (mid.fga if mid else 0) + (long_mid.fga if long_mid else 0) + (three.fga if three else 0)
    components[GravityMechanism.PERIMETER_GRAVITY.value] = {
        "perimeter_attempts_share": Metric.derived(
            "perimeter_attempts_share", (perimeter_fga / shooting_table.season_fga) if shooting_table.season_fga else None,
            method="(mid + long_mid + three FGA) / season total FGA, real zone totals -- broader than POP_GRAVITY (includes mid-range jump shooting)", season=season,
        ),
    }

    # --- POST_SCORING_GRAVITY (RECONSTRUCTED, moderate-low confidence)
    unassisted_short_paint_rate = (1 - short_paint.fg_assisted_pct) if (short_paint and short_paint.fg_assisted_pct is not None) else None
    post_gravity_score = None
    confidence = 0.45
    if hook is not None and unassisted_short_paint_rate is not None and hook.fga:
        # A simple, transparent, documented combination -- not fit to
        # any target, just an average of two real, directionally
        # consistent proxies. Never presented as tracking-grade.
        hook_efficiency_component = hook.fg_pct or 0.0
        post_gravity_score = 0.5 * (hook.fga / max(1, shooting_table.season_fga)) * 10 + 0.5 * unassisted_short_paint_rate
    components[GravityMechanism.POST_SCORING_GRAVITY.value] = {
        "hook_shot_attempts_season": Metric.observed("hook_shot_attempts_season", hook.fga if hook else None, source="Basketball-Reference season shooting table", season=season),
        "hook_shot_fg_pct": Metric.observed("hook_shot_fg_pct", hook.fg_pct if hook else None, source="Basketball-Reference season shooting table", season=season),
        "unassisted_short_paint_rate": Metric.derived(
            "unassisted_short_paint_rate", unassisted_short_paint_rate,
            method="1 - (real %Ast'd at 3-10ft zone) -- self-created short-paint makes skew toward post scoring rather than a set play", season=season,
        ),
        "post_scoring_gravity_index": Metric.reconstructed(
            "post_scoring_gravity_index", post_gravity_score,
            method="0.5*(hook_shot_FGA / season_FGA)*10 + 0.5*unassisted_short_paint_rate -- a transparent, documented combination of two real proxies (hook-shot volume, unassisted short-paint rate), NOT a tracking-grade post-touch measurement. See module docstring.",
            confidence=confidence, season=season,
        ),
    }

    mechanisms_present = [
        mech for mech, metrics in components.items()
        if any(m.status != "UNAVAILABLE" and m.value is not None for m in metrics.values())
    ]
    return GravityProfile(player_name=player_name, season=season, components=components, mechanisms_present=mechanisms_present)
