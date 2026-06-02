from __future__ import annotations

from dataclasses import dataclass


FORECASTABILITY_GAP_COLUMNS = [
    "forecastability_gap_primary",
    "forecastability_gap_secondary",
    "forecastability_gap_count",
    "forecastability_gap_reasons",
    "forecastability_gap_missing_features",
    "forecastability_gap_fixability",
    "forecastability_gap_blocks_safe_state_flag",
    "forecastability_gap_severity",
]

FORECASTABILITY_GAP_TYPES = {
    "FORECASTABILITY_GAP_MINUTES_STATE": "Missing or unstable minutes band, low floor, wide range, or high minutes CV.",
    "FORECASTABILITY_GAP_USAGE_STATE": "Missing or unstable FGA, usage, touch, assist, or rebound opportunity proxy.",
    "FORECASTABILITY_GAP_ROLE_STATE": "Starter/bench, rotation, role-shift, or teammate-context role uncertainty.",
    "FORECASTABILITY_GAP_DISTRIBUTION_WIDTH": "Stat distribution is wide or line sits in a fragile outcome region.",
    "FORECASTABILITY_GAP_SIMILAR_STATE_SAMPLE": "Not enough comparable pre-event rows.",
    "FORECASTABILITY_GAP_SIMILAR_STATE_SCATTER": "Comparable rows exist, but outcomes are scattered.",
    "FORECASTABILITY_GAP_TEAMMATE_CONTEXT": "Teammate availability or role dependency is missing or unstable.",
    "FORECASTABILITY_GAP_OPPONENT_CONTEXT": "Opponent scheme, pace, or defensive-class context is missing or materially different.",
    "FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA": "Key pre-event evidence is missing, but unsafe state is not proven.",
    "FORECASTABILITY_GAP_TRUE_UNSTABLE_STATE": "Evidence exists and shows the player state is genuinely unstable.",
}

FIXABILITY_VALUES = {
    "FIXABLE_WITH_EXISTING_LOGS",
    "FIXABLE_WITH_NEW_PIPELINE_DATA",
    "NEEDS_MORE_SAMPLE",
    "TRUE_UNSTABLE_STATE",
    "UNKNOWN",
}

SEVERITY_ORDER = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


@dataclass(frozen=True)
class ForecastabilityGap:
    gap_type: str
    reason: str
    missing_features: tuple[str, ...] = ()
    fixability: str = "UNKNOWN"
    severity: str = "MEDIUM"


def max_severity(values: list[str]) -> str:
    if not values:
        return "NONE"
    return max(values, key=lambda item: SEVERITY_ORDER.get(str(item), 0))


def merge_fixability(values: list[str]) -> str:
    ordered = [
        "TRUE_UNSTABLE_STATE",
        "FIXABLE_WITH_NEW_PIPELINE_DATA",
        "NEEDS_MORE_SAMPLE",
        "FIXABLE_WITH_EXISTING_LOGS",
        "UNKNOWN",
    ]
    value_set = {str(value) for value in values if str(value)}
    for value in ordered:
        if value in value_set:
            return value
    return "UNKNOWN"

