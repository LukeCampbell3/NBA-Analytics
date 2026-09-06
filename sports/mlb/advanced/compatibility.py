from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .schema import BatterProcessProfile, PitcherProcessProfile


@dataclass(frozen=True)
class PitchCompatibilitySignal:
    support: float
    matched_usage: float
    k_probability_delta: float
    contact_hit_probability_delta: float
    xslg_delta: float
    expected_xwoba_contact: float | None


def _finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def build_pitch_compatibility_signal(
    batter: BatterProcessProfile,
    pitcher: PitcherProcessProfile,
) -> PitchCompatibilitySignal:
    """Estimate bounded batter-vs-arsenal compatibility from observed pitch-type process data.

    The signal is deliberately residual: it only adjusts the already-estimated
    batter/pitcher matchup for pitch types where both sides have usable evidence.
    Missing pitch-type history produces a neutral result, never a fabricated
    preference.  Usage is taken from the pitcher's current arsenal.
    """
    if not pitcher.arsenal:
        return PitchCompatibilitySignal(0.0, 0.0, 0.0, 0.0, 0.0, None)

    weighted_batter_xwoba = 0.0
    weighted_batter_whiff = 0.0
    weighted_pitcher_contact_xwoba = 0.0
    xwoba_weight = 0.0
    whiff_weight = 0.0
    pitcher_xwoba_weight = 0.0
    matched_usage = 0.0

    for pitch_type, arsenal_row in pitcher.arsenal.items():
        usage = _finite((arsenal_row or {}).get("usage"))
        if usage is None or usage <= 0:
            continue
        usage = _clip(usage, 0.0, 1.0)
        bx = _finite(batter.pitch_type_xwoba.get(pitch_type))
        bw = _finite(batter.pitch_type_whiff_rate.get(pitch_type))
        px = _finite((arsenal_row or {}).get("xwoba_allowed_contact"))
        if bx is not None:
            weighted_batter_xwoba += usage * bx
            xwoba_weight += usage
            matched_usage += usage
        if bw is not None:
            weighted_batter_whiff += usage * bw
            whiff_weight += usage
        if px is not None:
            weighted_pitcher_contact_xwoba += usage * px
            pitcher_xwoba_weight += usage

    matched_usage = _clip(matched_usage, 0.0, 1.0)
    sample_support = min(float(batter.support), float(pitcher.support))
    support = _clip(matched_usage * sample_support, 0.0, 1.0)
    if support <= 0.0:
        return PitchCompatibilitySignal(0.0, matched_usage, 0.0, 0.0, 0.0, None)

    batter_pitch_xwoba = weighted_batter_xwoba / xwoba_weight if xwoba_weight > 0 else None
    batter_pitch_whiff = weighted_batter_whiff / whiff_weight if whiff_weight > 0 else None
    pitcher_pitch_xwoba = weighted_pitcher_contact_xwoba / pitcher_xwoba_weight if pitcher_xwoba_weight > 0 else None

    batter_baseline_xwoba = _finite(batter.xwoba)
    pitcher_baseline_xwoba = _finite(pitcher.xwoba_allowed)
    batter_baseline_whiff = _finite(batter.whiff_rate) or 0.225

    xwoba_components: list[float] = []
    if batter_pitch_xwoba is not None and batter_baseline_xwoba is not None:
        xwoba_components.append(batter_pitch_xwoba - batter_baseline_xwoba)
    if pitcher_pitch_xwoba is not None and pitcher_baseline_xwoba is not None:
        xwoba_components.append(pitcher_pitch_xwoba - pitcher_baseline_xwoba)
    xwoba_residual = sum(xwoba_components) / len(xwoba_components) if xwoba_components else 0.0

    whiff_residual = (
        batter_pitch_whiff - batter_baseline_whiff
        if batter_pitch_whiff is not None
        else 0.0
    )

    # Residuals are intentionally tightly bounded. They modify a matchup but
    # cannot overpower the underlying K/contact-quality profiles.
    k_delta = _clip(0.32 * whiff_residual * support, -0.035, 0.035)
    contact_hit_delta = _clip(0.22 * xwoba_residual * support, -0.025, 0.025)
    xslg_delta = _clip(0.60 * xwoba_residual * support, -0.08, 0.08)

    expected_xwoba_contact = None
    if batter_pitch_xwoba is not None and pitcher_pitch_xwoba is not None:
        expected_xwoba_contact = 0.58 * batter_pitch_xwoba + 0.42 * pitcher_pitch_xwoba
    elif batter_pitch_xwoba is not None:
        expected_xwoba_contact = batter_pitch_xwoba
    elif pitcher_pitch_xwoba is not None:
        expected_xwoba_contact = pitcher_pitch_xwoba

    return PitchCompatibilitySignal(
        support=support,
        matched_usage=matched_usage,
        k_probability_delta=k_delta,
        contact_hit_probability_delta=contact_hit_delta,
        xslg_delta=xslg_delta,
        expected_xwoba_contact=expected_xwoba_contact,
    )
