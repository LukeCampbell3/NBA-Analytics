from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from research.common import (
    brier_score,
    calibration_gap,
    expected_calibration_error,
    safe_bool,
    safe_float,
    series_numeric,
    series_text,
)


@dataclass(frozen=True)
class FailureSubregionDefinition:
    subregion_id: str
    parent_failure_family: str
    description: str
    target_markets: tuple[str, ...]
    required_features: tuple[str, ...]
    proxy_features: tuple[str, ...]
    detector: Callable[[pd.Series], bool]
    intervention_type: str
    trigger_condition: str
    postgame_only: bool = False


def _market_type(row: pd.Series) -> str:
    market_id = str(row.get("market_id", "")).strip().upper()
    if market_id:
        return market_id
    target = str(row.get("target", "")).strip().upper()
    direction = str(row.get("direction", "")).strip().upper()
    return f"{target}_{direction}".strip("_")


def _market_family(row: pd.Series) -> str:
    return _market_type(row).split("_")[0]


def _is_over(row: pd.Series) -> bool:
    return str(row.get("direction", "")).strip().upper() == "OVER"


def _projected_team_fg_low(row: pd.Series, threshold: float = 0.468) -> bool:
    return safe_float(row.get("projected_team_fg_pct"), default=np.nan) <= threshold


def _fragility_high(row: pd.Series, threshold: float = 0.55) -> bool:
    return safe_float(row.get("line_decision_fragility_score"), default=np.nan) >= threshold


def _instability_high(row: pd.Series, threshold: float = 0.45) -> bool:
    return safe_float(row.get("line_decision_instability_score"), default=np.nan) >= threshold


def _same_team_stack(row: pd.Series, threshold: float = 2.0) -> bool:
    return safe_float(row.get("same_team_selected_over_count"), default=0.0) >= threshold


def _stress_prob(row: pd.Series) -> float:
    return safe_float(
        row.get("stress_probability"),
        default=safe_float(row.get("p_side_stress"), default=safe_float(row.get("expected_win_rate"), default=np.nan)),
    )


def _break_even(row: pd.Series) -> float:
    return safe_float(row.get("market_side_break_even"), default=safe_float(row.get("break_even_prob"), default=np.nan))


def _edge_small(row: pd.Series, threshold: float = 1.0) -> bool:
    prediction = safe_float(row.get("prediction"), default=np.nan)
    market_line = safe_float(row.get("market_line"), default=np.nan)
    if np.isnan(prediction) or np.isnan(market_line):
        return False
    return abs(prediction - market_line) <= threshold


def _expected_minutes_stable(row: pd.Series) -> bool:
    return (
        safe_float(row.get("expected_minutes_band_width"), default=np.nan) <= 6.0
        and safe_float(row.get("minutes_floor_recent"), default=np.nan) >= 28.0
    )


def _expected_win_high(row: pd.Series, threshold: float = 0.60) -> bool:
    return safe_float(row.get("expected_win_rate"), default=np.nan) >= threshold


def _belief_uncertain(row: pd.Series, threshold: float = 0.85) -> bool:
    return safe_float(row.get("belief_uncertainty"), default=np.nan) >= threshold


def _posterior_variance_high(row: pd.Series, threshold: float = 0.08) -> bool:
    return safe_float(row.get("posterior_variance"), default=np.nan) >= threshold


def _calibration_low_sample(row: pd.Series, threshold: float = 40.0) -> bool:
    return safe_float(row.get("calibration_bucket_rows"), default=np.nan) < threshold


def _market_books_thin(row: pd.Series, threshold: float = 2.0) -> bool:
    return safe_float(row.get("market_books"), default=np.nan) <= threshold


def _role_shift_high(row: pd.Series, threshold: float = 0.55) -> bool:
    return (
        safe_float(row.get("role_pathway_shift_score"), default=np.nan) >= threshold
        or safe_float(row.get("role_shift_risk"), default=np.nan) >= threshold
    )


def _volatility_high(row: pd.Series, threshold: float = 0.58) -> bool:
    return safe_float(row.get("volatility_score"), default=np.nan) >= threshold


def _forecastability_low(row: pd.Series) -> bool:
    feasibility = safe_float(row.get("feasibility"), default=np.nan)
    return _belief_uncertain(row) or _volatility_high(row) or (not np.isnan(feasibility) and feasibility < 0.80)


def _minutes_floor_low(row: pd.Series, threshold: float = 18.0) -> bool:
    return safe_float(row.get("minutes_floor_recent"), default=np.nan) < threshold


def _minutes_band_wide(row: pd.Series, threshold: float = 8.0) -> bool:
    return safe_float(row.get("expected_minutes_band_width"), default=np.nan) > threshold


def _bench_role(row: pd.Series) -> bool:
    return safe_bool(row.get("bench_role_flag"), default=False)


def _rotation_volatile(row: pd.Series, threshold: float = 0.60) -> bool:
    return safe_float(row.get("rotation_volatility_score"), default=np.nan) >= threshold


def _foul_risk(row: pd.Series, threshold: float = 0.55) -> bool:
    return safe_float(row.get("foul_rate_minutes_loss_risk"), default=np.nan) >= threshold


def _blowout_risk(row: pd.Series, threshold: float = 0.55) -> bool:
    return safe_float(row.get("blowout_minutes_sensitivity"), default=np.nan) >= threshold


def _price_bad(row: pd.Series, margin: float = 0.015) -> bool:
    break_even = _break_even(row)
    stress = _stress_prob(row)
    return not np.isnan(break_even) and not np.isnan(stress) and break_even > stress + margin


def _ev_low(row: pd.Series, threshold: float = 0.03) -> bool:
    ev = safe_float(row.get("ev_final"), default=safe_float(row.get("ev"), default=np.nan))
    return not np.isnan(ev) and ev <= threshold


ACTIONABLE_STALE_PRICE_SUBREGIONS = {
    "STALE_PRICE_DEPENDENCY",
    "PRICE_MOVED_EDGE_DECAY",
    "LINE_MOVED_EDGE_DECAY",
}

DIAGNOSTIC_STALE_PRICE_SUBREGIONS = {
    "MISSING_PRICE_EDGE_UNTRUSTED",
    "INVALID_PRICE_EDGE_UNTRUSTED",
    "CLOSE_ONLY_PRICE_CONTAMINATION",
    "PRICE_SOURCE_UNKNOWN_DIAGNOSTIC_ONLY",
}


def _stale_price_actionable(row: pd.Series) -> bool:
    subregion = str(row.get("stale_price_subregion", "")).strip()
    if subregion not in ACTIONABLE_STALE_PRICE_SUBREGIONS:
        return False
    if not safe_bool(row.get("would_change_decision"), default=False):
        return False
    corrected_break_even = safe_float(row.get("corrected_break_even"), default=np.nan)
    return not np.isnan(corrected_break_even)


SUBREGION_DEFINITIONS: tuple[FailureSubregionDefinition, ...] = (
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__same_team_over_stack",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Same-team over stack depends on a healthy team offense.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("same_team_selected_over_count",),
        proxy_features=("same_team_selected_over_count",),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and _same_team_stack(row),
        intervention_type="parlay_shared_failure_veto",
        trigger_condition="same_team_selected_over_count>=2",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__low_team_total_or_low_implied_total",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Team total or implied total is low before tip.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("team_total",),
        proxy_features=(),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and safe_float(row.get("team_total"), default=np.nan) <= 108.0,
        intervention_type="scenario_stress_increase",
        trigger_condition="team_total<=108",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__fragile_high_usage_player_over",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="A player over is relying on a fragile high-usage role.",
        target_markets=("PTS_OVER", "PRA_OVER"),
        required_features=("usage_proxy", "volatility_score"),
        proxy_features=("volatility_score", "role_pathway_shift_score"),
        detector=lambda row: _market_type(row) in {"PTS_OVER", "PRA_OVER"} and _volatility_high(row) and _role_shift_high(row, threshold=0.45),
        intervention_type="soft_downgrade",
        trigger_condition="fragile_high_usage_role",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__poor_projected_team_fg_environment",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Projected team shooting environment is weak.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("projected_team_fg_pct",),
        proxy_features=("projected_team_fg_pct",),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and _projected_team_fg_low(row),
        intervention_type="scenario_stress_increase",
        trigger_condition="projected_team_fg_pct<=0.468",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__high_line_decision_fragility",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Line-decision fragility is already elevated pre-event.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("line_decision_fragility_score",),
        proxy_features=("line_decision_fragility_score",),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and _fragility_high(row),
        intervention_type="scenario_stress_increase",
        trigger_condition="line_decision_fragility_score>=0.55",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__high_line_decision_instability",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Line-decision instability is already elevated pre-event.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("line_decision_instability_score",),
        proxy_features=("line_decision_instability_score",),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and _instability_high(row),
        intervention_type="scenario_stress_increase",
        trigger_condition="line_decision_instability_score>=0.45",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__team_offense_dependency_plus_bad_price",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Team-offense dependency combines with a price that no longer clears stress.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("projected_team_fg_pct", "market_side_break_even", "stress_probability"),
        proxy_features=("projected_team_fg_pct", "expected_win_rate"),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and (_projected_team_fg_low(row) or _same_team_stack(row)) and _price_bad(row),
        intervention_type="price_dependent_tier",
        trigger_condition="team_offense_dependency_and_bad_price",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__team_offense_dependency_plus_pace_risk",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Team-offense dependency plus pace risk.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("pace_proxy",),
        proxy_features=(),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and safe_float(row.get("pace_proxy"), default=np.nan) < 97.0,
        intervention_type="scenario_stress_increase",
        trigger_condition="team_offense_dependency_and_pace_risk",
    ),
    FailureSubregionDefinition(
        subregion_id="TEAM_OFFENSE_COLLAPSE__team_offense_dependency_plus_teammate_return_risk",
        parent_failure_family="TEAM_OFFENSE_COLLAPSE",
        description="Team-offense dependency plus teammate return risk.",
        target_markets=("AST_OVER", "PTS_OVER", "PRA_OVER"),
        required_features=("teammate_return_risk",),
        proxy_features=(),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PTS_OVER", "PRA_OVER"} and safe_float(row.get("teammate_return_risk"), default=np.nan) >= 0.50,
        intervention_type="scenario_stress_increase",
        trigger_condition="team_offense_dependency_and_teammate_return_risk",
    ),
    FailureSubregionDefinition(
        subregion_id="LOW_TEAM_ASSIST_ENVIRONMENT__AST_OVER_with_low_projected_fg_support",
        parent_failure_family="LOW_TEAM_ASSIST_ENVIRONMENT",
        description="Assist over with weak projected team FG support.",
        target_markets=("AST_OVER",),
        required_features=("projected_team_fg_pct",),
        proxy_features=("projected_team_fg_pct",),
        detector=lambda row: _market_type(row) == "AST_OVER" and _projected_team_fg_low(row),
        intervention_type="scenario_stress_increase",
        trigger_condition="AST_OVER_and_projected_team_fg_pct<=0.468",
    ),
    FailureSubregionDefinition(
        subregion_id="LOW_TEAM_ASSIST_ENVIRONMENT__AST_OVER_with_low_team_total",
        parent_failure_family="LOW_TEAM_ASSIST_ENVIRONMENT",
        description="Assist over with low team total.",
        target_markets=("AST_OVER",),
        required_features=("team_total",),
        proxy_features=(),
        detector=lambda row: _market_type(row) == "AST_OVER" and safe_float(row.get("team_total"), default=np.nan) <= 108.0,
        intervention_type="scenario_stress_increase",
        trigger_condition="AST_OVER_and_team_total<=108",
    ),
    FailureSubregionDefinition(
        subregion_id="LOW_TEAM_ASSIST_ENVIRONMENT__AST_OVER_with_high_teammate_conversion_dependency",
        parent_failure_family="LOW_TEAM_ASSIST_ENVIRONMENT",
        description="Assist over depends heavily on teammate conversion.",
        target_markets=("AST_OVER",),
        required_features=("projected_assist_conversion_proxy", "teammate_shooting_support"),
        proxy_features=(),
        detector=lambda row: _market_type(row) == "AST_OVER" and (
            safe_float(row.get("projected_assist_conversion_proxy"), default=np.nan) <= 0.46
            or safe_float(row.get("teammate_shooting_support"), default=np.nan) <= 0.46
        ),
        intervention_type="scenario_stress_increase",
        trigger_condition="AST_OVER_and_low_conversion_support",
    ),
    FailureSubregionDefinition(
        subregion_id="LOW_TEAM_ASSIST_ENVIRONMENT__AST_OVER_with_minutes_stable_but_team_shooting_fragile",
        parent_failure_family="LOW_TEAM_ASSIST_ENVIRONMENT",
        description="Minutes look stable, but the shooting environment is fragile.",
        target_markets=("AST_OVER",),
        required_features=("projected_team_fg_pct", "expected_minutes_band_width", "minutes_floor_recent"),
        proxy_features=("projected_team_fg_pct", "expected_minutes_band_width", "minutes_floor_recent"),
        detector=lambda row: _market_type(row) == "AST_OVER" and _expected_minutes_stable(row) and _projected_team_fg_low(row),
        intervention_type="price_dependent_tier",
        trigger_condition="AST_OVER_stable_minutes_but_fragile_shooting",
    ),
    FailureSubregionDefinition(
        subregion_id="LOW_TEAM_ASSIST_ENVIRONMENT__PRA_OVER_with_assist_component_fragile",
        parent_failure_family="LOW_TEAM_ASSIST_ENVIRONMENT",
        description="PRA over depends on an assist component in a fragile shooting environment.",
        target_markets=("PRA_OVER",),
        required_features=("projected_team_fg_pct",),
        proxy_features=("projected_team_fg_pct",),
        detector=lambda row: _market_type(row) == "PRA_OVER" and _projected_team_fg_low(row),
        intervention_type="scenario_stress_increase",
        trigger_condition="PRA_OVER_and_projected_team_fg_pct<=0.468",
    ),
    FailureSubregionDefinition(
        subregion_id="LOW_TEAM_ASSIST_ENVIRONMENT__parlay_assist_leg_shared_team_failure",
        parent_failure_family="LOW_TEAM_ASSIST_ENVIRONMENT",
        description="Assist exposure is clustered with other same-team overs.",
        target_markets=("AST_OVER", "PRA_OVER"),
        required_features=("same_team_selected_over_count",),
        proxy_features=("same_team_selected_over_count",),
        detector=lambda row: _market_type(row) in {"AST_OVER", "PRA_OVER"} and _same_team_stack(row),
        intervention_type="parlay_shared_failure_veto",
        trigger_condition="assist_leg_same_team_stack",
    ),
    FailureSubregionDefinition(
        subregion_id="USAGE_SUPPRESSION__points_over_after_teammate_return",
        parent_failure_family="USAGE_SUPPRESSION",
        description="Points over after teammate return risk is elevated.",
        target_markets=("PTS_OVER",),
        required_features=("teammate_return_risk",),
        proxy_features=(),
        detector=lambda row: _market_type(row) == "PTS_OVER" and safe_float(row.get("teammate_return_risk"), default=np.nan) >= 0.50,
        intervention_type="soft_downgrade",
        trigger_condition="PTS_OVER_and_teammate_return_risk>=0.50",
    ),
    FailureSubregionDefinition(
        subregion_id="USAGE_SUPPRESSION__PRA_over_with_usage_band_wide",
        parent_failure_family="USAGE_SUPPRESSION",
        description="PRA over relies on a wide usage band.",
        target_markets=("PRA_OVER",),
        required_features=("usage_proxy", "volatility_score"),
        proxy_features=("volatility_score", "expected_minutes_band_width"),
        detector=lambda row: _market_type(row) == "PRA_OVER" and (_volatility_high(row) or _minutes_band_wide(row)),
        intervention_type="soft_downgrade",
        trigger_condition="PRA_OVER_and_usage_band_wide",
    ),
    FailureSubregionDefinition(
        subregion_id="USAGE_SUPPRESSION__high_usage_projection_with_role_shift_risk",
        parent_failure_family="USAGE_SUPPRESSION",
        description="Role-shift risk is high for a usage-dependent over.",
        target_markets=("PTS_OVER", "PRA_OVER"),
        required_features=("role_pathway_shift_score",),
        proxy_features=("role_pathway_shift_score", "role_shift_risk"),
        detector=lambda row: _market_type(row) in {"PTS_OVER", "PRA_OVER"} and _role_shift_high(row),
        intervention_type="soft_downgrade",
        trigger_condition="usage_dependent_over_and_role_shift_risk",
    ),
    FailureSubregionDefinition(
        subregion_id="USAGE_SUPPRESSION__player_minutes_hold_but_fga_under_trailing",
        parent_failure_family="USAGE_SUPPRESSION",
        description="Minutes held, but actual FGA landed well below trailing usage.",
        target_markets=("PTS_OVER", "PRA_OVER"),
        required_features=("actual_minutes", "expected_minutes_band_low", "player_actual_fga_vs_trailing"),
        proxy_features=("actual_minutes", "player_actual_fga_vs_trailing"),
        detector=lambda row: _market_type(row) in {"PTS_OVER", "PRA_OVER"} and safe_float(row.get("actual_minutes"), default=np.nan) + 1.0 >= safe_float(row.get("expected_minutes_band_low"), default=np.nan) and safe_float(row.get("player_actual_fga_vs_trailing"), default=np.nan) <= -0.18,
        intervention_type="forecastability_penalty",
        trigger_condition="minutes_hold_but_fga_under_trailing",
        postgame_only=True,
    ),
    FailureSubregionDefinition(
        subregion_id="USAGE_SUPPRESSION__line_raised_after_recent_usage_spike",
        parent_failure_family="USAGE_SUPPRESSION",
        description="The line appears to have chased a recent usage spike.",
        target_markets=("PTS_OVER", "PRA_OVER"),
        required_features=("usage_proxy",),
        proxy_features=("volatility_score", "prediction_shrink_lambda"),
        detector=lambda row: _market_type(row) in {"PTS_OVER", "PRA_OVER"} and _volatility_high(row) and safe_float(row.get("prediction_shrink_lambda"), default=np.nan) <= 0.20,
        intervention_type="alternate_line_discovery",
        trigger_condition="usage_spike_line_chase",
    ),
    FailureSubregionDefinition(
        subregion_id="USAGE_SUPPRESSION__same_team_usage_competition",
        parent_failure_family="USAGE_SUPPRESSION",
        description="Usage-dependent overs are stacked on the same team.",
        target_markets=("PTS_OVER", "PRA_OVER"),
        required_features=("same_team_selected_over_count",),
        proxy_features=("same_team_selected_over_count",),
        detector=lambda row: _market_type(row) in {"PTS_OVER", "PRA_OVER"} and _same_team_stack(row),
        intervention_type="parlay_shared_failure_veto",
        trigger_condition="same_team_usage_competition",
    ),
    FailureSubregionDefinition(
        subregion_id="MINUTES_BAND_FAILURE__low_minutes_floor",
        parent_failure_family="MINUTES_BAND_FAILURE",
        description="Recent minutes floor is structurally low.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER", "TRB_OVER"),
        required_features=("minutes_floor_recent",),
        proxy_features=("minutes_floor_recent",),
        detector=lambda row: _is_over(row) and _minutes_floor_low(row),
        intervention_type="forecastability_penalty",
        trigger_condition="minutes_floor_recent<18",
    ),
    FailureSubregionDefinition(
        subregion_id="MINUTES_BAND_FAILURE__wide_minutes_band",
        parent_failure_family="MINUTES_BAND_FAILURE",
        description="Expected minutes band is too wide.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER", "TRB_OVER"),
        required_features=("expected_minutes_band_width",),
        proxy_features=("expected_minutes_band_width",),
        detector=lambda row: _is_over(row) and _minutes_band_wide(row),
        intervention_type="forecastability_penalty",
        trigger_condition="expected_minutes_band_width>8",
    ),
    FailureSubregionDefinition(
        subregion_id="MINUTES_BAND_FAILURE__bench_role_exposure",
        parent_failure_family="MINUTES_BAND_FAILURE",
        description="Bench role introduces structural minutes uncertainty.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER", "TRB_OVER"),
        required_features=("bench_role_flag",),
        proxy_features=("bench_role_flag",),
        detector=lambda row: _is_over(row) and _bench_role(row),
        intervention_type="forecastability_penalty",
        trigger_condition="bench_role_flag",
    ),
    FailureSubregionDefinition(
        subregion_id="MINUTES_BAND_FAILURE__foul_trouble_minutes_loss",
        parent_failure_family="MINUTES_BAND_FAILURE",
        description="Pre-event foul-risk context threatens minutes.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER", "TRB_OVER"),
        required_features=("foul_rate_minutes_loss_risk",),
        proxy_features=("foul_rate_minutes_loss_risk",),
        detector=lambda row: _is_over(row) and _foul_risk(row),
        intervention_type="forecastability_penalty",
        trigger_condition="foul_rate_minutes_loss_risk>=0.55",
    ),
    FailureSubregionDefinition(
        subregion_id="MINUTES_BAND_FAILURE__blowout_pull_minutes_loss",
        parent_failure_family="MINUTES_BAND_FAILURE",
        description="Blowout risk threatens late minutes.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER", "TRB_OVER"),
        required_features=("blowout_minutes_sensitivity",),
        proxy_features=("blowout_minutes_sensitivity",),
        detector=lambda row: _is_over(row) and _blowout_risk(row),
        intervention_type="forecastability_penalty",
        trigger_condition="blowout_minutes_sensitivity>=0.55",
    ),
    FailureSubregionDefinition(
        subregion_id="MINUTES_BAND_FAILURE__narrow_margin_over_plus_minutes_risk",
        parent_failure_family="MINUTES_BAND_FAILURE",
        description="A narrow-margin over is being asked to survive minutes risk.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER", "TRB_OVER"),
        required_features=("market_line", "prediction", "minutes_floor_recent"),
        proxy_features=("market_line", "prediction", "expected_minutes_band_width"),
        detector=lambda row: _is_over(row) and _edge_small(row) and (_minutes_floor_low(row, threshold=22.0) or _minutes_band_wide(row, threshold=6.0)),
        intervention_type="soft_downgrade",
        trigger_condition="narrow_margin_over_and_minutes_risk",
    ),
    FailureSubregionDefinition(
        subregion_id="MINUTES_BAND_FAILURE__role_change_minutes_uncertainty",
        parent_failure_family="MINUTES_BAND_FAILURE",
        description="Rotation or role changes widen the minutes distribution.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER", "TRB_OVER"),
        required_features=("rotation_volatility_score", "starter_status_change_count"),
        proxy_features=("rotation_volatility_score",),
        detector=lambda row: _is_over(row) and (_rotation_volatile(row) or safe_float(row.get("starter_status_change_count"), default=0.0) >= 1.0),
        intervention_type="forecastability_penalty",
        trigger_condition="role_change_minutes_uncertainty",
    ),
    FailureSubregionDefinition(
        subregion_id="MARKET_PRICE_MISPLACEMENT__raw_edge_positive_but_stress_edge_negative",
        parent_failure_family="MARKET_PRICE_MISPLACEMENT",
        description="Raw edge looks positive, but the stressed edge is already gone.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER"),
        required_features=("edge", "market_side_break_even", "stress_probability"),
        proxy_features=("edge", "expected_win_rate"),
        detector=lambda row: safe_float(row.get("edge"), default=np.nan) > 0.0 and _price_bad(row),
        intervention_type="price_dependent_tier",
        trigger_condition="edge>0_and_stress_prob<break_even",
    ),
    FailureSubregionDefinition(
        subregion_id="MARKET_PRICE_MISPLACEMENT__odds_do_not_clear_break_even",
        parent_failure_family="MARKET_PRICE_MISPLACEMENT",
        description="Odds no longer clear break-even after stress.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER"),
        required_features=("market_side_break_even", "stress_probability"),
        proxy_features=("expected_win_rate",),
        detector=lambda row: _price_bad(row),
        intervention_type="price_dependent_tier",
        trigger_condition="market_side_break_even>stress_probability+0.015",
    ),
    FailureSubregionDefinition(
        subregion_id="MARKET_PRICE_MISPLACEMENT__stale_price_dependency",
        parent_failure_family="MARKET_PRICE_MISPLACEMENT",
        description="Timestamp-safe price correction removes or materially weakens the edge.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER", "TRB_OVER", "TRB_UNDER"),
        required_features=("stale_price_subregion", "proposed_decision_after_price_fix", "would_change_decision"),
        proxy_features=("price_validity_status", "corrected_edge", "edge_decay"),
        detector=_stale_price_actionable,
        intervention_type="price_dependent_tier",
        trigger_condition="timestamp_safe_price_removes_or_weakens_edge",
    ),
    FailureSubregionDefinition(
        subregion_id="MARKET_PRICE_MISPLACEMENT__alternate_line_better_than_main_line",
        parent_failure_family="MARKET_PRICE_MISPLACEMENT",
        description="Main line framing is weak relative to the projected edge.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER"),
        required_features=("expected_push_rate", "market_line", "prediction"),
        proxy_features=("expected_push_rate", "market_line", "prediction"),
        detector=lambda row: safe_float(row.get("expected_push_rate"), default=np.nan) >= 0.10 and _edge_small(row, threshold=0.75),
        intervention_type="alternate_line_discovery",
        trigger_condition="expected_push_rate>=0.10_and_edge_small",
    ),
    FailureSubregionDefinition(
        subregion_id="MARKET_PRICE_MISPLACEMENT__opposite_side_price_better",
        parent_failure_family="MARKET_PRICE_MISPLACEMENT",
        description="The opposite side appears cheaper and better aligned with stress.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER"),
        required_features=("opposite_side_break_even", "opposite_side_stress_prob"),
        proxy_features=(),
        detector=lambda row: safe_float(row.get("opposite_side_stress_prob"), default=np.nan) > safe_float(row.get("opposite_side_break_even"), default=np.nan),
        intervention_type="opposite_side_discovery",
        trigger_condition="opposite_side_stress_prob>opposite_side_break_even",
    ),
    FailureSubregionDefinition(
        subregion_id="MARKET_PRICE_MISPLACEMENT__high_probability_but_bad_price",
        parent_failure_family="MARKET_PRICE_MISPLACEMENT",
        description="High apparent confidence is erased by the posted price.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER"),
        required_features=("expected_win_rate", "market_side_break_even", "stress_probability"),
        proxy_features=("expected_win_rate",),
        detector=lambda row: _expected_win_high(row, threshold=0.60) and _price_bad(row),
        intervention_type="price_dependent_tier",
        trigger_condition="expected_win_rate>=0.60_and_bad_price",
    ),
    FailureSubregionDefinition(
        subregion_id="MARKET_PRICE_MISPLACEMENT__low_probability_plus_money_value_candidate",
        parent_failure_family="MARKET_PRICE_MISPLACEMENT",
        description="Plus-money framing depends on a thin probability edge.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER"),
        required_features=("market_side_price", "expected_win_rate"),
        proxy_features=("market_side_price", "expected_win_rate"),
        detector=lambda row: safe_float(row.get("market_side_price"), default=np.nan) > 0.0 and safe_float(row.get("expected_win_rate"), default=np.nan) < 0.55,
        intervention_type="price_dependent_tier",
        trigger_condition="plus_money_and_expected_win_rate<0.55",
    ),
    FailureSubregionDefinition(
        subregion_id="CALIBRATION_OVERCONFIDENCE__high_expected_win_rate_bucket_underperforms",
        parent_failure_family="CALIBRATION_OVERCONFIDENCE",
        description="A high expected-win-rate bucket is carrying too much confidence.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER", "TRB_OVER", "TRB_UNDER"),
        required_features=("expected_win_rate",),
        proxy_features=("expected_win_rate",),
        detector=lambda row: _expected_win_high(row, threshold=0.62),
        intervention_type="calibration_shrink",
        trigger_condition="expected_win_rate>=0.62",
    ),
    FailureSubregionDefinition(
        subregion_id="CALIBRATION_OVERCONFIDENCE__high_belief_low_sample",
        parent_failure_family="CALIBRATION_OVERCONFIDENCE",
        description="Confidence is high despite a thin calibration bucket.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER", "TRB_OVER", "TRB_UNDER"),
        required_features=("expected_win_rate", "calibration_bucket_rows"),
        proxy_features=("expected_win_rate", "calibration_bucket_rows", "belief_uncertainty"),
        detector=lambda row: _expected_win_high(row, threshold=0.60) and _calibration_low_sample(row),
        intervention_type="calibration_shrink",
        trigger_condition="expected_win_rate>=0.60_and_calibration_bucket_rows<40",
    ),
    FailureSubregionDefinition(
        subregion_id="CALIBRATION_OVERCONFIDENCE__high_probability_high_variance_market",
        parent_failure_family="CALIBRATION_OVERCONFIDENCE",
        description="Confidence is high, but posterior variance is elevated.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER", "TRB_OVER", "TRB_UNDER"),
        required_features=("expected_win_rate", "posterior_variance"),
        proxy_features=("expected_win_rate", "posterior_variance"),
        detector=lambda row: _expected_win_high(row, threshold=0.60) and _posterior_variance_high(row),
        intervention_type="calibration_shrink",
        trigger_condition="expected_win_rate>=0.60_and_posterior_variance>=0.08",
    ),
    FailureSubregionDefinition(
        subregion_id="CALIBRATION_OVERCONFIDENCE__model_confidence_high_but_forecastability_low",
        parent_failure_family="CALIBRATION_OVERCONFIDENCE",
        description="Confidence is high while forecastability signals remain weak.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER", "TRB_OVER", "TRB_UNDER"),
        required_features=("expected_win_rate", "belief_uncertainty"),
        proxy_features=("expected_win_rate", "belief_uncertainty", "volatility_score", "feasibility"),
        detector=lambda row: _expected_win_high(row, threshold=0.60) and _forecastability_low(row),
        intervention_type="forecastability_penalty",
        trigger_condition="expected_win_rate>=0.60_and_forecastability_low",
    ),
    FailureSubregionDefinition(
        subregion_id="CALIBRATION_OVERCONFIDENCE__high_confidence_same_team_cluster",
        parent_failure_family="CALIBRATION_OVERCONFIDENCE",
        description="Confidence is high and same-team concentration is elevated.",
        target_markets=("PTS_OVER", "AST_OVER", "PRA_OVER"),
        required_features=("expected_win_rate", "same_team_selected_over_count"),
        proxy_features=("expected_win_rate", "same_team_selected_over_count"),
        detector=lambda row: _market_type(row) in {"PTS_OVER", "AST_OVER", "PRA_OVER"} and _expected_win_high(row, threshold=0.60) and _same_team_stack(row),
        intervention_type="board_objective_penalty",
        trigger_condition="expected_win_rate>=0.60_and_same_team_stack",
    ),
    FailureSubregionDefinition(
        subregion_id="CALIBRATION_OVERCONFIDENCE__high_confidence_low_LCB_edge",
        parent_failure_family="CALIBRATION_OVERCONFIDENCE",
        description="Confidence is high, but lower-confidence edge support is thin.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER", "TRB_OVER", "TRB_UNDER"),
        required_features=("expected_win_rate", "ev"),
        proxy_features=("expected_win_rate", "ev", "ev_final"),
        detector=lambda row: _expected_win_high(row, threshold=0.60) and _ev_low(row),
        intervention_type="calibration_shrink",
        trigger_condition="expected_win_rate>=0.60_and_ev<=0.03",
    ),
    FailureSubregionDefinition(
        subregion_id="CALIBRATION_OVERCONFIDENCE__high_confidence_market_disagreement",
        parent_failure_family="CALIBRATION_OVERCONFIDENCE",
        description="Confidence is high, but market disagreement or book thinness persists.",
        target_markets=("PTS_OVER", "PTS_UNDER", "AST_OVER", "AST_UNDER", "PRA_OVER", "PRA_UNDER", "TRB_OVER", "TRB_UNDER"),
        required_features=("expected_win_rate", "market_books"),
        proxy_features=("expected_win_rate", "market_books", "board_shadow_disagreement"),
        detector=lambda row: _expected_win_high(row, threshold=0.60) and (_market_books_thin(row) or safe_float(row.get("board_shadow_disagreement"), default=np.nan) >= 0.20),
        intervention_type="board_objective_penalty",
        trigger_condition="expected_win_rate>=0.60_and_market_disagreement",
    ),
)

SUBREGION_ACTION_RANK = {
    "VALIDATE_SHADOW": 0,
    "NEEDS_MORE_SAMPLE": 1,
    "FEATURE_GAP_BLOCKED": 2,
    "REJECT_RANDOM": 3,
}


def subregion_definitions_for_families(families: list[str] | set[str] | None = None) -> list[FailureSubregionDefinition]:
    family_set = {str(item).strip() for item in (families or []) if str(item).strip()}
    if not family_set:
        return list(SUBREGION_DEFINITIONS)
    return [definition for definition in SUBREGION_DEFINITIONS if definition.parent_failure_family in family_set]


def detect_failure_subregions(row: pd.Series, *, families: list[str] | set[str] | None = None) -> list[str]:
    detected: list[str] = []
    for definition in subregion_definitions_for_families(families):
        try:
            if bool(definition.detector(row)):
                detected.append(definition.subregion_id)
        except Exception:
            continue
    return detected


def build_failure_subregion_scoreboard(
    selected_rows: pd.DataFrame,
    *,
    candidate_pool_rows: pd.DataFrame | None = None,
    target_failure_modes: list[str] | set[str] | None = None,
    min_loss_count: int = 3,
    min_resolved_count: int = 8,
    max_coverage_cost: float = 0.25,
    max_non_target_damage: float = 0.15,
    min_pre_event_detectability: float = 0.60,
    max_win_removal_rate: float = 0.35,
) -> pd.DataFrame:
    definitions = subregion_definitions_for_families(target_failure_modes)
    total_selected = max(int(len(selected_rows)), 1)
    total_losses = max(int(series_text(selected_rows, "result").str.lower().eq("loss").sum()), 1)
    total_wins = max(int(series_text(selected_rows, "result").str.lower().eq("win").sum()), 1)
    selected = selected_rows.copy()
    selected["result"] = series_text(selected, "result").str.lower()
    selected["resolved_label"] = series_numeric(selected, "resolved_label", default=np.nan)
    missing_label = selected["resolved_label"].isna()
    if bool(missing_label.any()):
        selected.loc[missing_label, "resolved_label"] = np.where(
            selected.loc[missing_label, "result"].eq("win"),
            1.0,
            np.where(selected.loc[missing_label, "result"].eq("loss"), 0.0, np.nan),
        )
    selected["predicted_probability"] = series_numeric(
        selected,
        "predicted_probability",
        default=np.nan,
    ).fillna(series_numeric(selected, "stress_probability", default=np.nan)).fillna(series_numeric(selected, "expected_win_rate", default=np.nan))
    selected["profit_units"] = series_numeric(selected, "units", default=np.nan).fillna(series_numeric(selected, "profit_units", default=np.nan))

    candidate_pool = candidate_pool_rows.copy() if candidate_pool_rows is not None and not candidate_pool_rows.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    available_columns = set(selected.columns)
    if not candidate_pool.empty:
        available_columns.update(candidate_pool.columns)

    for definition in definitions:
        selected_mask = selected.apply(definition.detector, axis=1)
        group = selected.loc[selected_mask].copy()
        resolved_group = group.loc[group["result"].isin(["win", "loss"])].copy()
        losses = int(resolved_group["result"].eq("loss").sum())
        wins = int(resolved_group["result"].eq("win").sum())
        resolved_count = int(len(resolved_group))
        candidate_count = 0
        if not candidate_pool.empty:
            candidate_count = int(candidate_pool.apply(definition.detector, axis=1).sum())
        loss_concentration = float(losses / total_losses) if total_losses > 0 else 0.0
        win_concentration = float(wins / total_wins) if total_wins > 0 else 0.0
        coverage_cost = float(len(group) / total_selected)
        detectability_rate = 0.0 if definition.postgame_only else (1.0 if losses > 0 else 0.0)
        estimated_loss_removal_rate = float(detectability_rate * (losses / max(1, resolved_count))) if resolved_count > 0 else 0.0
        estimated_win_removal_rate = float(detectability_rate * (wins / max(1, resolved_count))) if resolved_count > 0 else 0.0
        non_target_damage_risk = float(np.clip(coverage_cost * estimated_win_removal_rate * max(1.0, win_concentration), 0.0, 1.0))
        sample_reliability_weight = float(np.clip(resolved_count / 20.0, 0.0, 1.0))
        favorable_margin = max(0.0, estimated_loss_removal_rate - estimated_win_removal_rate)
        subregion_priority_score = float(
            loss_concentration
            * max(detectability_rate, 0.0)
            * favorable_margin
            * max(0.0, 1.0 - non_target_damage_risk)
            * sample_reliability_weight
        )
        run_date_count = int(series_text(group, "run_date").replace("", np.nan).nunique(dropna=True)) if not group.empty else 0
        window_count = int(series_text(group, "source_selected_board_csv").replace("", np.nan).nunique(dropna=True)) if not group.empty else 0
        player_count = int(series_text(group, "player").replace("", np.nan).nunique(dropna=True)) if not group.empty else 0
        team_count = int(series_text(group, "actual_team").replace("", np.nan).nunique(dropna=True)) if not group.empty else 0
        market_count = int(series_text(group, "market_type").replace("", np.nan).nunique(dropna=True)) if not group.empty else 0
        player_share = float(series_text(group, "player").value_counts(normalize=True, dropna=True).max()) if not group.empty else 0.0
        team_share = float(series_text(group, "actual_team").value_counts(normalize=True, dropna=True).max()) if not group.empty else 0.0
        date_share = float(series_text(group, "run_date").value_counts(normalize=True, dropna=True).max()) if not group.empty else 0.0
        missing_features = [column for column in definition.required_features if column not in available_columns]
        proxy_available = [column for column in definition.proxy_features if column in available_columns]
        if definition.subregion_id == "MARKET_PRICE_MISPLACEMENT__stale_price_dependency":
            selected_diagnostic_count = int(series_text(selected, "stale_price_subregion").isin(DIAGNOSTIC_STALE_PRICE_SUBREGIONS).sum())
            if selected_diagnostic_count > 0 and (resolved_count < int(min_resolved_count) or losses < int(min_loss_count)):
                missing_features = sorted(
                    set(missing_features).union(
                        {"market_side_price", "market_side_break_even", "odds_snapshot_time", "price_source"}
                    )
                )
                recommended_status = "FEATURE_GAP_BLOCKED"
            elif missing_features and not proxy_available:
                recommended_status = "FEATURE_GAP_BLOCKED"
            elif definition.postgame_only:
                recommended_status = "REJECT_RANDOM"
            elif resolved_count < int(min_resolved_count) or losses < int(min_loss_count):
                recommended_status = "NEEDS_MORE_SAMPLE"
            elif estimated_loss_removal_rate <= estimated_win_removal_rate:
                recommended_status = "REJECT_RANDOM"
            elif coverage_cost > float(max_coverage_cost) or non_target_damage_risk > float(max_non_target_damage) or estimated_win_removal_rate > float(max_win_removal_rate):
                recommended_status = "REJECT_RANDOM"
            elif detectability_rate < float(min_pre_event_detectability):
                recommended_status = "REJECT_RANDOM"
            elif run_date_count <= 1 or window_count <= 1 or player_share > 0.75 or team_share > 0.80 or date_share > 0.80:
                recommended_status = "NEEDS_MORE_SAMPLE"
            else:
                recommended_status = "VALIDATE_SHADOW"
        elif missing_features and not proxy_available:
            recommended_status = "FEATURE_GAP_BLOCKED"
        elif definition.postgame_only:
            recommended_status = "REJECT_RANDOM"
        elif resolved_count < int(min_resolved_count) or losses < int(min_loss_count):
            recommended_status = "NEEDS_MORE_SAMPLE"
        elif estimated_loss_removal_rate <= estimated_win_removal_rate:
            recommended_status = "REJECT_RANDOM"
        elif coverage_cost > float(max_coverage_cost) or non_target_damage_risk > float(max_non_target_damage) or estimated_win_removal_rate > float(max_win_removal_rate):
            recommended_status = "REJECT_RANDOM"
        elif detectability_rate < float(min_pre_event_detectability):
            recommended_status = "REJECT_RANDOM"
        elif run_date_count <= 1 or window_count <= 1 or player_share > 0.75 or team_share > 0.80 or date_share > 0.80:
            recommended_status = "NEEDS_MORE_SAMPLE"
        else:
            recommended_status = "VALIDATE_SHADOW"
        rows.append(
            {
                "parent_failure_family": definition.parent_failure_family,
                "subregion_id": definition.subregion_id,
                "description": definition.description,
                "target_markets": "|".join(definition.target_markets),
                "required_features": "|".join(definition.required_features),
                "missing_features": "|".join(missing_features),
                "proxy_features": "|".join(definition.proxy_features),
                "candidate_count": candidate_count,
                "selected_count": int(len(group)),
                "resolved_count": resolved_count,
                "wins": wins,
                "losses": losses,
                "hit_rate": float(wins / max(1, wins + losses)) if (wins + losses) > 0 else np.nan,
                "profit_units": float(pd.to_numeric(resolved_group["profit_units"], errors="coerce").fillna(0.0).sum()),
                "ROI": float(pd.to_numeric(resolved_group["profit_units"], errors="coerce").fillna(0.0).sum() / max(1, resolved_count)) if resolved_count > 0 else np.nan,
                "Brier": brier_score(resolved_group["predicted_probability"], resolved_group["resolved_label"]),
                "ECE": expected_calibration_error(resolved_group["predicted_probability"], resolved_group["resolved_label"]),
                "calibration_gap": calibration_gap(resolved_group["predicted_probability"], resolved_group["resolved_label"]),
                "loss_concentration": loss_concentration,
                "win_concentration": win_concentration,
                "loss_to_win_ratio": float(losses / max(1, wins)) if losses > 0 else 0.0,
                "estimated_loss_removal_rate": estimated_loss_removal_rate,
                "estimated_win_removal_rate": estimated_win_removal_rate,
                "coverage_cost": coverage_cost,
                "non_target_damage_risk": non_target_damage_risk,
                "pre_event_detectability_rate": detectability_rate,
                "sample_reliability_weight": sample_reliability_weight,
                "subregion_priority_score": subregion_priority_score,
                "run_date_count": run_date_count,
                "window_count": window_count,
                "player_count": player_count,
                "team_count": team_count,
                "market_count": market_count,
                "max_player_share": player_share,
                "max_team_share": team_share,
                "max_date_share": date_share,
                "postgame_only": definition.postgame_only,
                "recommended_next_action": recommended_status,
                "intervention_type": definition.intervention_type,
                "trigger_condition": definition.trigger_condition,
            }
        )
    out = pd.DataFrame(rows)
    out["_action_rank"] = out["recommended_next_action"].map(lambda value: SUBREGION_ACTION_RANK.get(str(value), 99))
    out = out.sort_values(
        ["_action_rank", "subregion_priority_score", "losses", "resolved_count"],
        ascending=[True, False, False, False],
    ).drop(columns=["_action_rank"]).reset_index(drop=True)
    return out


def summarize_failure_subregion_scoreboard(scoreboard: pd.DataFrame) -> dict[str, Any]:
    if scoreboard.empty:
        return {
            "subregion_count": 0,
            "validate_shadow_count": 0,
            "broad_signal_unsafe_families": [],
        }
    family_summary: list[dict[str, Any]] = []
    for family, group in scoreboard.groupby("parent_failure_family", dropna=False):
        observed = group.loc[group["resolved_count"] > 0].copy()
        broad_signal = bool(not observed.empty and observed["losses"].sum() > 0)
        validate_shadow = bool(group["recommended_next_action"].astype(str).eq("VALIDATE_SHADOW").any())
        if broad_signal and not validate_shadow:
            family_summary.append({"failure_family": str(family), "status": "BROAD_SIGNAL_UNSAFE_TO_ACT"})
    return {
        "subregion_count": int(len(scoreboard)),
        "validate_shadow_count": int(scoreboard["recommended_next_action"].astype(str).eq("VALIDATE_SHADOW").sum()),
        "broad_signal_unsafe_families": family_summary,
    }
