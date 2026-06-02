from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import (
    as_string_list,
    build_candidate_id,
    coerce_market_family,
    coerce_market_type,
    safe_bool,
    safe_float,
    series_numeric,
    series_text,
    utc_now_iso,
    write_json,
)
from research.failure_modes.failure_mode_registry import get_failure_mode, load_failure_mode_registry


RECOVERABILITY_CLASSES = {
    "RECOVERABLE_PRE_EVENT",
    "PARTIALLY_RECOVERABLE",
    "ALEATORIC_OR_RANDOM",
    "DATA_MISSING",
    "MARKET_PRICE_FAILURE",
    "MODEL_CALIBRATION_FAILURE",
    "SELECTION_FAILURE",
}

MODE_PRIORITY = [
    "TEAM_OFFENSE_COLLAPSE",
    "LOW_TEAM_ASSIST_ENVIRONMENT",
    "USAGE_SUPPRESSION",
    "REBOUND_LOW_LINE_ROLE_VOLATILITY",
    "REBOUND_UPPER_BAND_SUPPLY_RISK",
    "REBOUND_SHARE_COMPETITION",
    "REBOUND_SUPPLY_COLLAPSE",
    "MINUTES_BAND_FAILURE",
    "BLOWOUT_PULL_RISK",
    "FOUL_TROUBLE_RISK",
    "MARKET_PRICE_MISPLACEMENT",
    "OPPOSITE_SIDE_SIGNAL",
    "CALIBRATION_OVERCONFIDENCE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute settled pick failures to reusable failure modes.")
    parser.add_argument("--selected-board-csv", type=Path, required=True, help="Settled selected-board rows.")
    parser.add_argument("--candidate-pool-csv", type=Path, default=None, help="Optional candidate-pool rows for feature backfill.")
    parser.add_argument("--failure-attribution-csv-out", type=Path, required=True)
    parser.add_argument("--failure-attribution-summary-json-out", type=Path, required=True)
    return parser.parse_args()


def _market_type_from_row(row: pd.Series) -> str:
    market_id = str(row.get("market_id", "")).strip().upper()
    if market_id:
        return market_id
    target = str(row.get("target", "")).strip().upper()
    direction = str(row.get("direction", "")).strip().upper()
    return f"{target}_{direction}".strip("_")


def _mode_is_enabled(
    failure_mode_id: str,
    *,
    allowed_failure_modes: set[str] | None = None,
    excluded_failure_modes: set[str] | None = None,
) -> bool:
    mode = str(failure_mode_id).strip()
    if excluded_failure_modes and mode in excluded_failure_modes:
        return False
    if allowed_failure_modes is None:
        return True
    return mode in allowed_failure_modes


def _minutes_held_flag(row: pd.Series) -> bool:
    actual_minutes = _actual_minutes(row)
    expected_low = safe_float(row.get("expected_minutes_band_low"), default=np.nan)
    minutes_median = safe_float(row.get("minutes_median_recent"), default=np.nan)
    if np.isnan(actual_minutes):
        return False
    if not np.isnan(expected_low):
        return bool(actual_minutes + 1.0 >= expected_low)
    if not np.isnan(minutes_median):
        return bool(actual_minutes >= 0.85 * minutes_median)
    return False


def _actual_stat_value(row: pd.Series) -> float:
    target = str(row.get("target", "")).strip().upper()
    explicit = safe_float(row.get("actual_result"), default=np.nan)
    if not np.isnan(explicit):
        return explicit
    generic_actual = safe_float(row.get("actual"), default=np.nan)
    if not np.isnan(generic_actual):
        return generic_actual
    if target:
        keyed = safe_float(row.get(f"actual_{target}"), default=np.nan)
        if not np.isnan(keyed):
            return keyed
    return safe_float(row.get("actual_stat_value"), default=np.nan)


def _actual_minutes(row: pd.Series) -> float:
    return safe_float(row.get("actual_minutes"), default=safe_float(row.get("minutes"), default=np.nan))


def _resolved_result(row: pd.Series) -> str:
    text = str(row.get("result", row.get("actual_result_label", ""))).strip().lower()
    if text in {"win", "loss", "push"}:
        return text
    miss_distance = _miss_distance(row)
    if np.isnan(miss_distance):
        return ""
    if abs(miss_distance) <= 1e-9:
        return "push"
    return "win" if miss_distance > 0 else "loss"


def _miss_distance(row: pd.Series) -> float:
    line = safe_float(row.get("market_line"), default=np.nan)
    actual = _actual_stat_value(row)
    direction = str(row.get("direction", "")).strip().upper()
    if np.isnan(line) or np.isnan(actual):
        return np.nan
    if direction == "UNDER":
        return float(line - actual)
    return float(actual - line)


def _pre_event_warning_features(row: pd.Series, failure_modes: list[str]) -> list[str]:
    warnings: list[str] = []
    column_map = {
        "TEAM_OFFENSE_COLLAPSE": ["projected_team_fg_pct", "line_decision_fragility_score", "same_team_selected_over_count"],
        "LOW_TEAM_ASSIST_ENVIRONMENT": ["projected_team_fg_pct", "line_decision_fragility_score", "expected_minutes_band_low"],
        "USAGE_SUPPRESSION": ["role_pathway_shift_score", "role_shift_risk", "volatility_score"],
        "REBOUND_UPPER_BAND_SUPPLY_RISK": ["upper_band_line_penalty", "trb_q75_recent", "projected_missed_fga_total"],
        "REBOUND_LOW_LINE_ROLE_VOLATILITY": ["low_line_role_volatility_penalty", "expected_minutes_band_width", "minutes_floor_recent", "bench_role_flag"],
        "REBOUND_SHARE_COMPETITION": ["rebound_share_competition_penalty", "teammate_rebound_competition_score", "player_rebound_share_std"],
        "REBOUND_SUPPLY_COLLAPSE": ["rebound_supply_penalty", "projected_missed_fga_total", "team_shooting_efficiency_stress", "opponent_shooting_efficiency_stress"],
        "MINUTES_BAND_FAILURE": ["expected_minutes_band_low", "expected_minutes_band_high", "minutes_floor_recent", "rotation_volatility_score"],
        "BLOWOUT_PULL_RISK": ["blowout_minutes_sensitivity", "spread_proxy"],
        "FOUL_TROUBLE_RISK": ["foul_rate_minutes_loss_risk", "matchup_foul_proxy"],
        "MARKET_PRICE_MISPLACEMENT": ["market_side_break_even", "stress_probability", "expected_win_rate"],
        "OPPOSITE_SIDE_SIGNAL": ["opposite_side_break_even", "opposite_side_stress_prob", "opposite_side_lcb_edge"],
        "CALIBRATION_OVERCONFIDENCE": ["predicted_probability", "stress_probability", "expected_win_rate"],
    }
    for failure_mode in failure_modes:
        for column in column_map.get(failure_mode, []):
            if column not in warnings and column in row.index and str(row.get(column, "")).strip() not in {"", "nan"}:
                warnings.append(column)
    return warnings


def _detect_pre_event_failure_modes(
    row: pd.Series,
    *,
    allowed_failure_modes: set[str] | None = None,
    excluded_failure_modes: set[str] | None = None,
) -> list[str]:
    market_type = _market_type_from_row(row)
    direction = str(row.get("direction", "")).strip().upper()
    detected: list[str] = []

    projected_team_fg_pct = safe_float(row.get("projected_team_fg_pct"), default=np.nan)
    line_fragility = safe_float(row.get("line_decision_fragility_score"), default=np.nan)
    line_instability = safe_float(row.get("line_decision_instability_score"), default=np.nan)
    same_team_over_count = safe_float(row.get("same_team_selected_over_count"), default=0.0)
    role_shift_score = safe_float(row.get("role_pathway_shift_score"), default=np.nan)
    role_shift_risk = safe_float(row.get("role_shift_risk"), default=np.nan)
    volatility_score = safe_float(row.get("volatility_score"), default=np.nan)
    expected_band_width = safe_float(row.get("expected_minutes_band_width"), default=np.nan)
    minutes_floor_recent = safe_float(row.get("minutes_floor_recent"), default=np.nan)
    break_even = safe_float(row.get("market_side_break_even"), default=safe_float(row.get("break_even_prob"), default=np.nan))
    stress_prob = safe_float(row.get("stress_probability"), default=safe_float(row.get("p_side_stress"), default=safe_float(row.get("expected_win_rate"), default=np.nan)))
    predicted_probability = safe_float(
        row.get("predicted_probability"),
        default=safe_float(row.get("stress_probability"), default=safe_float(row.get("expected_win_rate"), default=np.nan)),
    )
    bucket_rows = safe_float(row.get("calibration_bucket_rows"), default=np.nan)
    uncertainty = safe_float(row.get("belief_uncertainty"), default=np.nan)
    posterior_variance = safe_float(row.get("posterior_variance"), default=np.nan)

    if (
        market_type in {"AST_OVER", "PTS_OVER"}
        and _mode_is_enabled("TEAM_OFFENSE_COLLAPSE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and (
            (not np.isnan(projected_team_fg_pct) and projected_team_fg_pct <= 0.462)
            or (not np.isnan(line_fragility) and line_fragility >= 0.56 and not np.isnan(line_instability) and line_instability >= 0.45)
            or same_team_over_count >= 2.0
        )
    ):
        detected.append("TEAM_OFFENSE_COLLAPSE")

    if (
        market_type == "AST_OVER"
        and _mode_is_enabled("LOW_TEAM_ASSIST_ENVIRONMENT", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and (
            (not np.isnan(projected_team_fg_pct) and projected_team_fg_pct <= 0.468)
            or (not np.isnan(line_fragility) and line_fragility >= 0.54)
        )
    ):
        detected.append("LOW_TEAM_ASSIST_ENVIRONMENT")

    if (
        market_type == "PTS_OVER"
        and _mode_is_enabled("USAGE_SUPPRESSION", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and (
            (not np.isnan(role_shift_score) and role_shift_score >= 0.55)
            or (not np.isnan(role_shift_risk) and role_shift_risk >= 0.55)
            or (not np.isnan(volatility_score) and volatility_score >= 0.58)
        )
    ):
        detected.append("USAGE_SUPPRESSION")

    if market_type == "TRB_OVER":
        if (
            _mode_is_enabled("REBOUND_UPPER_BAND_SUPPLY_RISK", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and safe_float(row.get("upper_band_line_penalty"), default=0.0) > 0.0
        ):
            detected.append("REBOUND_UPPER_BAND_SUPPLY_RISK")
        if (
            _mode_is_enabled("REBOUND_LOW_LINE_ROLE_VOLATILITY", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and (
                safe_float(row.get("low_line_role_volatility_penalty"), default=0.0) > 0.0
                or safe_bool(row.get("low_line_role_volatility_flag"), default=False)
            )
        ):
            detected.append("REBOUND_LOW_LINE_ROLE_VOLATILITY")
        if (
            _mode_is_enabled("REBOUND_SHARE_COMPETITION", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and safe_float(row.get("rebound_share_competition_penalty"), default=0.0) > 0.0
        ):
            detected.append("REBOUND_SHARE_COMPETITION")
        if (
            _mode_is_enabled("REBOUND_SUPPLY_COLLAPSE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and safe_float(row.get("rebound_supply_penalty"), default=0.0) > 0.0
        ):
            detected.append("REBOUND_SUPPLY_COLLAPSE")

    if (
        direction == "OVER"
        and _mode_is_enabled("MINUTES_BAND_FAILURE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and (
            (not np.isnan(expected_band_width) and expected_band_width >= 8.0)
            or (not np.isnan(minutes_floor_recent) and minutes_floor_recent < 18.0)
            or safe_bool(row.get("bench_role_flag"), default=False)
            or safe_float(row.get("rotation_volatility_score"), default=0.0) >= 0.55
            or safe_float(row.get("blowout_minutes_sensitivity"), default=0.0) >= 0.55
            or safe_float(row.get("foul_rate_minutes_loss_risk"), default=0.0) >= 0.55
        )
    ):
        detected.append("MINUTES_BAND_FAILURE")

    if (
        _mode_is_enabled("MARKET_PRICE_MISPLACEMENT", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and not np.isnan(break_even)
        and not np.isnan(stress_prob)
        and break_even > stress_prob + 0.015
    ):
        detected.append("MARKET_PRICE_MISPLACEMENT")

    if (
        _mode_is_enabled("CALIBRATION_OVERCONFIDENCE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and not np.isnan(predicted_probability)
        and predicted_probability >= 0.64
        and (
            (not np.isnan(bucket_rows) and bucket_rows < 40.0)
            or (not np.isnan(uncertainty) and uncertainty >= 0.85)
            or (not np.isnan(posterior_variance) and posterior_variance >= 0.08)
            or predicted_probability >= 0.80
        )
    ):
        detected.append("CALIBRATION_OVERCONFIDENCE")
    return detected


def _detect_postgame_failure_modes(
    row: pd.Series,
    *,
    allowed_failure_modes: set[str] | None = None,
    excluded_failure_modes: set[str] | None = None,
) -> list[str]:
    market_type = _market_type_from_row(row)
    result = _resolved_result(row)
    detected: list[str] = []
    actual_minutes = _actual_minutes(row)
    expected_low = safe_float(row.get("expected_minutes_band_low"), default=np.nan)

    if result == "loss" and not np.isnan(actual_minutes) and not np.isnan(expected_low) and actual_minutes + 0.5 < expected_low:
        if _mode_is_enabled("MINUTES_BAND_FAILURE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes):
            detected.append("MINUTES_BAND_FAILURE")

    blowout_sensitivity = safe_float(row.get("blowout_minutes_sensitivity"), default=0.0)
    blowout_state = safe_bool(row.get("blowout_state"), default=False) or safe_bool(row.get("actual_blowout"), default=False)
    if (
        result == "loss"
        and _mode_is_enabled("BLOWOUT_PULL_RISK", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and blowout_sensitivity >= 0.45
        and blowout_state
        and not np.isnan(actual_minutes)
        and not np.isnan(expected_low)
        and actual_minutes + 0.5 < expected_low
    ):
        detected.append("BLOWOUT_PULL_RISK")

    foul_risk = safe_float(row.get("foul_rate_minutes_loss_risk"), default=0.0)
    actual_fouls = safe_float(row.get("actual_fouls"), default=np.nan)
    if (
        result == "loss"
        and _mode_is_enabled("FOUL_TROUBLE_RISK", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and foul_risk >= 0.45
        and not np.isnan(actual_fouls)
        and actual_fouls >= 4.0
        and not np.isnan(actual_minutes)
        and not np.isnan(expected_low)
        and actual_minutes + 0.5 < expected_low
    ):
        detected.append("FOUL_TROUBLE_RISK")

    team_points_vs_trailing = safe_float(row.get("team_actual_points_vs_trailing"), default=np.nan)
    team_ast_vs_trailing = safe_float(row.get("team_actual_ast_vs_trailing"), default=np.nan)
    team_fg_pct_delta = safe_float(row.get("team_actual_fg_pct_delta"), default=np.nan)
    same_team_over_loss_count = safe_float(row.get("same_team_selected_over_loss_count"), default=0.0)
    same_team_over_count = safe_float(row.get("same_team_selected_over_count"), default=0.0)
    if (
        result == "loss"
        and market_type in {"AST_OVER", "PTS_OVER"}
        and _mode_is_enabled("TEAM_OFFENSE_COLLAPSE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and (
            (not np.isnan(team_points_vs_trailing) and team_points_vs_trailing <= -0.12)
            or (not np.isnan(team_ast_vs_trailing) and team_ast_vs_trailing <= -0.18)
            or (not np.isnan(team_fg_pct_delta) and team_fg_pct_delta <= -0.05)
        )
        and (market_type == "AST_OVER" or same_team_over_loss_count >= 2.0 or same_team_over_count >= 2.0)
    ):
        detected.append("TEAM_OFFENSE_COLLAPSE")

    if (
        result == "loss"
        and market_type == "AST_OVER"
        and _minutes_held_flag(row)
        and _mode_is_enabled("LOW_TEAM_ASSIST_ENVIRONMENT", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and (
            (not np.isnan(team_ast_vs_trailing) and team_ast_vs_trailing <= -0.15)
            or (not np.isnan(team_points_vs_trailing) and team_points_vs_trailing <= -0.12)
        )
    ):
        detected.append("LOW_TEAM_ASSIST_ENVIRONMENT")

    player_fga_vs_trailing = safe_float(row.get("player_actual_fga_vs_trailing"), default=np.nan)
    player_usg_delta = safe_float(row.get("player_actual_usg_delta"), default=np.nan)
    if (
        result == "loss"
        and market_type == "PTS_OVER"
        and _minutes_held_flag(row)
        and _mode_is_enabled("USAGE_SUPPRESSION", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and (
            (not np.isnan(player_fga_vs_trailing) and player_fga_vs_trailing <= -0.18)
            or (not np.isnan(player_usg_delta) and player_usg_delta <= -3.0)
        )
    ):
        detected.append("USAGE_SUPPRESSION")

    if market_type == "TRB_OVER":
        if (
            result == "loss"
            and _mode_is_enabled("REBOUND_UPPER_BAND_SUPPLY_RISK", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and safe_float(row.get("upper_band_line_penalty"), default=0.0) > 0.0
        ):
            detected.append("REBOUND_UPPER_BAND_SUPPLY_RISK")
        if (
            result == "loss"
            and _mode_is_enabled("REBOUND_LOW_LINE_ROLE_VOLATILITY", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and (
                safe_float(row.get("low_line_role_volatility_penalty"), default=0.0) > 0.0
                or safe_bool(row.get("low_line_role_volatility_flag"), default=False)
            )
        ):
            detected.append("REBOUND_LOW_LINE_ROLE_VOLATILITY")
        if (
            result == "loss"
            and _mode_is_enabled("REBOUND_SHARE_COMPETITION", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and safe_float(row.get("rebound_share_competition_penalty"), default=0.0) > 0.0
        ):
            detected.append("REBOUND_SHARE_COMPETITION")
        if (
            result == "loss"
            and _mode_is_enabled("REBOUND_SUPPLY_COLLAPSE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
            and safe_float(row.get("rebound_supply_penalty"), default=0.0) > 0.0
        ):
            detected.append("REBOUND_SUPPLY_COLLAPSE")

    break_even = safe_float(row.get("market_side_break_even"), default=safe_float(row.get("break_even_prob"), default=np.nan))
    stress_prob = safe_float(row.get("stress_probability"), default=safe_float(row.get("p_side_stress"), default=safe_float(row.get("expected_win_rate"), default=np.nan)))
    if (
        result == "loss"
        and _mode_is_enabled("MARKET_PRICE_MISPLACEMENT", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and not np.isnan(break_even)
        and not np.isnan(stress_prob)
        and break_even > stress_prob + 0.02
    ):
        detected.append("MARKET_PRICE_MISPLACEMENT")

    opposite_lcb = safe_float(row.get("opposite_side_lcb_edge"), default=np.nan)
    opposite_stress = safe_float(row.get("opposite_side_stress_prob"), default=np.nan)
    opposite_break_even = safe_float(row.get("opposite_side_break_even"), default=np.nan)
    if (
        result == "loss"
        and _mode_is_enabled("OPPOSITE_SIDE_SIGNAL", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and not np.isnan(opposite_lcb)
        and opposite_lcb > 0.0
        and not np.isnan(opposite_stress)
        and not np.isnan(opposite_break_even)
        and opposite_stress > opposite_break_even
    ):
        detected.append("OPPOSITE_SIDE_SIGNAL")

    predicted_probability = safe_float(
        row.get("predicted_probability"),
        default=safe_float(row.get("stress_probability"), default=safe_float(row.get("expected_win_rate"), default=np.nan)),
    )
    if (
        result == "loss"
        and _mode_is_enabled("CALIBRATION_OVERCONFIDENCE", allowed_failure_modes=allowed_failure_modes, excluded_failure_modes=excluded_failure_modes)
        and not np.isnan(predicted_probability)
        and predicted_probability >= 0.64
    ):
        detected.append("CALIBRATION_OVERCONFIDENCE")
    return detected


def _detect_failure_modes(
    row: pd.Series,
    *,
    registry: dict[str, Any],
    allowed_failure_modes: set[str] | None = None,
    excluded_failure_modes: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    pre_event = _detect_pre_event_failure_modes(
        row,
        allowed_failure_modes=allowed_failure_modes,
        excluded_failure_modes=excluded_failure_modes,
    )
    postgame = _detect_postgame_failure_modes(
        row,
        allowed_failure_modes=allowed_failure_modes,
        excluded_failure_modes=excluded_failure_modes,
    )
    detected = [mode for mode in MODE_PRIORITY + sorted(set(pre_event + postgame) - set(MODE_PRIORITY)) if mode in set(pre_event) | set(postgame)]

    unique_detected: list[str] = []
    for failure_mode in detected:
        if failure_mode not in unique_detected and get_failure_mode(failure_mode, registry) is not None:
            unique_detected.append(failure_mode)
    return unique_detected, pre_event, postgame


def _recoverability_class(row: pd.Series, failure_modes: list[str]) -> str:
    result = _resolved_result(row)
    if result != "loss":
        return ""
    if np.isnan(_actual_stat_value(row)) and np.isnan(_actual_minutes(row)):
        return "DATA_MISSING"
    if "MARKET_PRICE_MISPLACEMENT" in failure_modes:
        return "MARKET_PRICE_FAILURE"
    if failure_modes and any(
        mode.startswith("REBOUND_")
        or mode in {"MINUTES_BAND_FAILURE", "OPPOSITE_SIDE_SIGNAL", "LOW_TEAM_ASSIST_ENVIRONMENT", "TEAM_OFFENSE_COLLAPSE", "USAGE_SUPPRESSION"}
        for mode in failure_modes
    ):
        return "RECOVERABLE_PRE_EVENT"
    if failure_modes and any(mode in {"BLOWOUT_PULL_RISK", "FOUL_TROUBLE_RISK"} for mode in failure_modes):
        return "PARTIALLY_RECOVERABLE"
    if failure_modes == ["CALIBRATION_OVERCONFIDENCE"] or ("CALIBRATION_OVERCONFIDENCE" in failure_modes and len(failure_modes) == 1):
        return "MODEL_CALIBRATION_FAILURE"
    if safe_float(row.get("stress_probability"), default=safe_float(row.get("expected_win_rate"), default=np.nan)) < safe_float(row.get("break_even_prob"), default=safe_float(row.get("market_side_break_even"), default=np.nan)):
        return "SELECTION_FAILURE"
    return "ALEATORIC_OR_RANDOM"


def _recommended_intervention_type(failure_modes: list[str], *, registry: dict[str, Any]) -> str:
    for failure_mode in failure_modes:
        definition = get_failure_mode(failure_mode, registry)
        if definition and definition.candidate_interventions:
            return str(definition.candidate_interventions[0].get("intervention_type", "")).strip()
    return ""


def _coalesce_selected_with_candidate_pool(
    selected_rows: pd.DataFrame,
    candidate_pool_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    selected = selected_rows.copy()
    selected["candidate_id"] = build_candidate_id(selected)
    if candidate_pool_rows is None or candidate_pool_rows.empty:
        return selected
    pool = candidate_pool_rows.copy()
    pool["candidate_id"] = build_candidate_id(pool)
    extra_columns = [column for column in pool.columns if column not in selected.columns or column == "candidate_id"]
    merged = selected.merge(
        pool.loc[:, extra_columns].drop_duplicates(subset=["candidate_id"]),
        on="candidate_id",
        how="left",
        suffixes=("", "_candidate"),
    )
    for column in list(merged.columns):
        if not column.endswith("_candidate"):
            continue
        base = column[:-10]
        if base not in merged.columns:
            merged = merged.rename(columns={column: base})
            continue
        merged[base] = merged[base].where(merged[base].notna(), merged[column])
        merged = merged.drop(columns=[column])
    return merged


def attribute_pick_failures(
    selected_rows: pd.DataFrame,
    candidate_pool_rows: pd.DataFrame | None = None,
    *,
    registry: dict[str, Any] | None = None,
    allowed_failure_modes: set[str] | None = None,
    excluded_failure_modes: set[str] | None = None,
) -> pd.DataFrame:
    active_registry = registry or load_failure_mode_registry()
    merged = _coalesce_selected_with_candidate_pool(selected_rows, candidate_pool_rows)
    if merged.empty:
        return merged.iloc[0:0].copy()

    merged["market_type"] = coerce_market_type(merged)
    merged["market_family"] = coerce_market_family(merged)
    merged["predicted_probability"] = series_numeric(
        merged,
        "predicted_probability",
        default=np.nan,
    ).fillna(series_numeric(merged, "stress_probability", default=np.nan)).fillna(series_numeric(merged, "expected_win_rate", default=np.nan))
    merged["stress_probability"] = series_numeric(merged, "stress_probability", default=np.nan).fillna(series_numeric(merged, "p_side_stress", default=np.nan))
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        detected, pre_event_detected, postgame_detected = _detect_failure_modes(
            row,
            registry=active_registry,
            allowed_failure_modes=allowed_failure_modes,
            excluded_failure_modes=excluded_failure_modes,
        )
        warning_features = _pre_event_warning_features(row, detected)
        recoverability = _recoverability_class(row, detected)
        primary = detected[0] if detected else ""
        secondary = detected[1:] if len(detected) > 1 else []
        result = _resolved_result(row)
        miss_distance = _miss_distance(row)
        actual_stat = _actual_stat_value(row)
        game_date = str(row.get("actual_matched_date", row.get("market_date", row.get("run_date", "")))).strip()
        output = dict(row)
        output.update(
            {
                "pick_id": str(row.get("candidate_id", "")),
                "game_date": game_date,
                "player": str(row.get("player", row.get("market_player_raw", ""))).strip(),
                "team": str(row.get("team", row.get("actual_team", row.get("team_abbrev", "")))).strip(),
                "opponent": str(row.get("opponent", row.get("actual_opponent", row.get("market_away_team", "")))).strip(),
                "market_type": str(row.get("market_type", _market_type_from_row(row))).strip(),
                "side": str(row.get("direction", "")).strip().upper(),
                "line": safe_float(row.get("market_line"), default=np.nan),
                "odds": safe_float(row.get("market_side_price"), default=safe_float(row.get("odds"), default=np.nan)),
                "predicted_probability": safe_float(row.get("predicted_probability"), default=np.nan),
                "stress_probability": safe_float(row.get("stress_probability"), default=np.nan),
                "break_even_probability": safe_float(row.get("market_side_break_even"), default=safe_float(row.get("break_even_prob"), default=np.nan)),
                "actual_stat": actual_stat,
                "actual_result_value": actual_stat,
                "actual_result": result,
                "result": result,
                "miss_distance": miss_distance,
                "failure_modes": detected,
                "pre_event_failure_modes": pre_event_detected,
                "postgame_failure_modes": postgame_detected,
                "primary_failure_mode": primary,
                "secondary_failure_modes": secondary,
                "was_failure_pre_event_detectable": bool(result == "loss" and bool(primary) and primary in pre_event_detected),
                "was_pre_event_detectable": bool(result == "loss" and bool(primary) and primary in pre_event_detected),
                "pre_event_warning_features": warning_features,
                "recommended_intervention_type": _recommended_intervention_type(detected, registry=active_registry),
                "recoverability_class": recoverability,
                "failure_recoverability": recoverability,
            }
        )
        rows.append(output)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["failure_modes"] = out["failure_modes"].apply(as_string_list)
        out["pre_event_failure_modes"] = out["pre_event_failure_modes"].apply(as_string_list)
        out["postgame_failure_modes"] = out["postgame_failure_modes"].apply(as_string_list)
        out["secondary_failure_modes"] = out["secondary_failure_modes"].apply(as_string_list)
        out["was_failure_pre_event_detectable"] = out["was_failure_pre_event_detectable"].astype(bool)
        out["was_pre_event_detectable"] = out["was_pre_event_detectable"].astype(bool)
    return out


def summarize_failure_attribution(attributed_rows: pd.DataFrame) -> dict[str, Any]:
    if attributed_rows.empty:
        return {
            "generated_at_utc": utc_now_iso(),
            "row_count": 0,
            "loss_count": 0,
            "recoverability_class_counts": {},
            "primary_failure_mode_counts": {},
            "detectable_loss_rate": np.nan,
        }
    result_series = series_text(attributed_rows, "result").str.lower()
    loss_mask = result_series.eq("loss")
    loss_count = int(loss_mask.sum())
    detectable_loss_rate = float(
        attributed_rows.loc[loss_mask, "was_failure_pre_event_detectable"].astype(bool).mean()
    ) if loss_count > 0 else np.nan
    return {
        "generated_at_utc": utc_now_iso(),
        "row_count": int(len(attributed_rows)),
        "loss_count": loss_count,
        "recoverability_class_counts": attributed_rows.get("recoverability_class", pd.Series(dtype="object")).value_counts(dropna=False).to_dict(),
        "primary_failure_mode_counts": attributed_rows.get("primary_failure_mode", pd.Series(dtype="object")).replace("", "UNATTRIBUTED").value_counts(dropna=False).to_dict(),
        "detectable_loss_rate": detectable_loss_rate,
    }


def main() -> None:
    args = parse_args()
    selected_rows = pd.read_csv(args.selected_board_csv)
    candidate_pool_rows = pd.read_csv(args.candidate_pool_csv) if args.candidate_pool_csv and args.candidate_pool_csv.exists() else None
    attributed = attribute_pick_failures(selected_rows, candidate_pool_rows)
    args.failure_attribution_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    attributed.to_csv(args.failure_attribution_csv_out, index=False)
    write_json(args.failure_attribution_summary_json_out, summarize_failure_attribution(attributed))


if __name__ == "__main__":
    main()
