from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.build_forecastability_gap import annotate_forecastability_gaps


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _clip01(values: pd.Series | float) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    return pd.Series(float(values)).clip(lower=0.0, upper=1.0)


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            out = out.where(out.notna(), values)
    return out.fillna(default)


def _volatility_to_stability(values: pd.Series, scale: float = 10.0) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    normalized = values.where(values.le(1.0), values / float(scale))
    return (1.0 - normalized.fillna(0.50)).clip(lower=0.0, upper=1.0)


def _truthy(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "bench"})


def _score_minutes(frame: pd.DataFrame) -> pd.Series:
    width = _coalesce_numeric(
        frame,
        ["expected_minutes_band_width", "minutes_band_width", "minutes_range_recent"],
        default=np.nan,
    )
    low = _coalesce_numeric(frame, ["expected_minutes_band_low", "minutes_floor_recent", "minutes_p25_recent"], default=np.nan)
    high = _coalesce_numeric(frame, ["expected_minutes_band_high"], default=np.nan)
    width = width.where(width.notna(), high - low)
    floor = _coalesce_numeric(frame, ["minutes_floor_recent", "expected_minutes_band_low", "minutes_p25_recent"], default=np.nan)
    p25 = _coalesce_numeric(frame, ["minutes_p25_recent", "expected_minutes_band_low"], default=np.nan)

    band_score = (1.0 - (width.fillna(10.0) / 16.0)).clip(lower=0.0, upper=1.0)
    floor_score = ((floor.fillna(18.0) - 12.0) / 24.0).clip(lower=0.0, upper=1.0)
    p25_score = ((p25.fillna(18.0) - 14.0) / 22.0).clip(lower=0.0, upper=1.0)
    rotation_stability = 1.0 - _clip01(_numeric(frame, "rotation_volatility_score", 0.35).fillna(0.35))
    blowout_stability = 1.0 - _clip01(_numeric(frame, "blowout_minutes_sensitivity", 0.20).fillna(0.20))
    foul_stability = 1.0 - _clip01(_numeric(frame, "foul_rate_minutes_loss_risk", 0.15).fillna(0.15))
    bench_role = _truthy(_text(frame, "bench_role_flag", "false")) | _text(frame, "starter_status_recent", "").str.lower().str.contains("bench")
    bench_penalty = pd.Series(0.10, index=frame.index).mask(bench_role, 0.22)

    score = (
        0.25 * band_score
        + 0.25 * floor_score
        + 0.15 * p25_score
        + 0.15 * rotation_stability
        + 0.10 * blowout_stability
        + 0.10 * foul_stability
        - bench_penalty
    )
    return score.clip(lower=0.0, upper=1.0)


def _score_usage(frame: pd.DataFrame) -> pd.Series:
    explicit = _numeric(frame, "usage_forecastability_score", np.nan)
    usage_stability = _volatility_to_stability(
        _coalesce_numeric(
            frame,
            ["usage_volatility", "usage_proxy_volatility", "production_band_width", "player_recent_volatility", "volatility_score"],
            default=0.45,
        ),
        scale=12.0,
    )
    fga_stability = _volatility_to_stability(_coalesce_numeric(frame, ["fga_volatility", "FGA_volatility"], default=0.45), scale=8.0)
    assist_stability = _volatility_to_stability(
        _coalesce_numeric(frame, ["assist_opportunity_volatility", "potential_assist_volatility"], default=0.45),
        scale=8.0,
    )
    rebound_stability = _volatility_to_stability(_coalesce_numeric(frame, ["rebound_chance_volatility"], default=0.45), scale=8.0)
    teammate_stability = 1.0 - _clip01(
        _coalesce_numeric(frame, ["teammate_return_risk", "teammate_availability_uncertainty", "role_shift_risk"], default=0.25)
    )
    score = (
        0.30 * usage_stability
        + 0.20 * fga_stability
        + 0.15 * assist_stability
        + 0.15 * rebound_stability
        + 0.20 * teammate_stability
    ).clip(lower=0.0, upper=1.0)
    return explicit.where(explicit.notna(), score)


def _score_role(frame: pd.DataFrame) -> pd.Series:
    explicit = _numeric(frame, "role_forecastability_score", np.nan)
    status_changes = _numeric(frame, "starter_status_change_count", 0.0).fillna(0.0)
    change_stability = (1.0 - (status_changes / 3.0)).clip(lower=0.0, upper=1.0)
    rotation_stability = 1.0 - _clip01(_numeric(frame, "rotation_volatility_score", 0.35).fillna(0.35))
    coach_trust = _clip01(_numeric(frame, "coach_trust_score", 0.60).fillna(0.60))
    starter_status = _text(frame, "starter_status_recent", "").str.lower()
    starter_stability = pd.Series(0.70, index=frame.index)
    starter_stability = starter_stability.mask(starter_status.str.contains("stable|starter|locked"), 0.90)
    starter_stability = starter_stability.mask(starter_status.str.contains("bench|changed|uncertain"), 0.45)
    score = (0.35 * change_stability + 0.30 * rotation_stability + 0.20 * coach_trust + 0.15 * starter_stability)
    return explicit.where(explicit.notna(), score.clip(lower=0.0, upper=1.0))


def _score_opponent(frame: pd.DataFrame) -> pd.Series:
    explicit = _numeric(frame, "opponent_adjusted_forecastability_score", np.nan)
    context_similarity = _clip01(_coalesce_numeric(frame, ["opponent_context_similarity", "opponent_defensive_context_similarity"], default=0.60))
    scenario_agreement = _clip01(_numeric(frame, "scenario_agreement", 0.60).fillna(0.60))
    pace_similarity = 1.0 - _clip01(_coalesce_numeric(frame, ["pace_mismatch_score", "opponent_scheme_disruption_score"], default=0.25))
    matchup_stability = 1.0 - _clip01(_coalesce_numeric(frame, ["matchup_role_shift_risk", "opponent_adjustment_risk"], default=0.25))
    score = (0.30 * context_similarity + 0.30 * scenario_agreement + 0.20 * pace_similarity + 0.20 * matchup_stability)
    return explicit.where(explicit.notna(), score.clip(lower=0.0, upper=1.0))


def _score_similar_state_support(frame: pd.DataFrame) -> pd.Series:
    explicit = _numeric(frame, "similar_state_reliability_score", np.nan)
    tightness = _numeric(frame, "similar_state_tightness_score", np.nan)
    count = _numeric(frame, "similar_state_count", np.nan)
    sample_weight = (count.fillna(0.0) / 8.0).clip(lower=0.0, upper=1.0)
    fallback = (tightness.fillna(0.50) * (0.55 + 0.45 * sample_weight)).clip(lower=0.0, upper=1.0)
    return explicit.where(explicit.notna(), fallback)


def annotate_player_state_forecastability(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    if "forecastability_gap_primary" not in out.columns:
        out = annotate_forecastability_gaps(out)
    minutes_score = _score_minutes(out)
    usage_score = _score_usage(out)
    role_score = _score_role(out)
    distribution_score = _clip01(_numeric(out, "distribution_stability_score", 0.50).fillna(0.50))
    opponent_score = _score_opponent(out)
    similar_score = _score_similar_state_support(out)
    teammate_score = _clip01(_numeric(out, "teammate_context_score", 0.50).fillna(0.50))
    gap_severity = _text(out, "forecastability_gap_severity", "NONE").str.upper()
    gap_fixability = _text(out, "forecastability_gap_fixability", "UNKNOWN").str.upper()
    similar_tier = _text(out, "similar_state_reliability_tier", "").str.upper()
    minutes_gap = (
        _text(out, "forecastability_gap_primary", "")
        + ";"
        + _text(out, "forecastability_gap_secondary", "")
    ).str.upper().str.contains("FORECASTABILITY_GAP_MINUTES_STATE|FORECASTABILITY_GAP_ROLE_STATE", regex=True)
    scenario_agreement = _clip01(_numeric(out, "scenario_agreement", 0.60).fillna(0.60))
    chaos = _clip01(_numeric(out, "chaos_score", 0.35).fillna(0.35))
    chaos_penalty = 0.15 * chaos

    overall = (
        0.25 * minutes_score
        + 0.15 * usage_score
        + 0.10 * role_score
        + 0.15 * distribution_score
        + 0.20 * similar_score
        + 0.05 * teammate_score
        + 0.05 * opponent_score
        + 0.05 * scenario_agreement
        - chaos_penalty
    ).clip(lower=0.0, upper=1.0)

    minute_columns = [
        "expected_minutes_band_low",
        "expected_minutes_band_high",
        "expected_minutes_band_width",
        "minutes_floor_recent",
        "minutes_p25_recent",
        "minutes_median_recent",
        "minutes_range_recent",
    ]
    missing_minutes_context = ~pd.Series(False, index=out.index)
    missing_minutes_context = ~pd.concat(
        [pd.to_numeric(out[col], errors="coerce").notna() for col in minute_columns if col in out.columns],
        axis=1,
    ).any(axis=1) if any(col in out.columns for col in minute_columns) else pd.Series(True, index=out.index)

    tier = pd.Series("UNFORECASTABLE", index=out.index, dtype="object")
    tier = tier.mask(overall.ge(0.50), "LOW_FORECASTABILITY")
    no_critical_gap = ~gap_severity.isin({"HIGH", "CRITICAL"})
    tier = tier.mask(overall.ge(0.62) & no_critical_gap, "MEDIUM_FORECASTABILITY")
    high_mask = (
        overall.ge(0.75)
        & no_critical_gap
        & ~similar_tier.eq("INSUFFICIENT_SAMPLE")
        & ~minutes_gap
        & minutes_score.ge(0.55)
        & chaos.le(0.35)
        & scenario_agreement.ge(0.65)
    )
    tier = tier.mask(high_mask, "HIGH_FORECASTABILITY")
    tier = tier.mask(missing_minutes_context & _numeric(out, "forecastability_score", np.nan).isna(), "UNFORECASTABLE")
    tier = tier.mask(gap_fixability.eq("TRUE_UNSTABLE_STATE") | gap_severity.eq("CRITICAL"), "UNFORECASTABLE")

    reasons: list[str] = []
    failure_modes: list[str] = []
    for idx in out.index:
        row_reasons: list[str] = []
        row_failures: list[str] = []
        if minutes_score.loc[idx] >= 0.70:
            row_reasons.append("minutes_band_stable")
        elif minutes_score.loc[idx] < 0.45:
            row_failures.append("MINUTES_BAND_FAILURE")
            row_reasons.append("minutes_band_unstable")
        if usage_score.loc[idx] >= 0.65:
            row_reasons.append("usage_context_stable")
        elif usage_score.loc[idx] < 0.45:
            row_failures.append("USAGE_SUPPRESSION_OR_ROLE_VOLATILITY")
            row_reasons.append("usage_context_unstable")
        if role_score.loc[idx] < 0.50:
            row_failures.append("ROLE_SHIFT_LOW_USAGE")
            row_reasons.append("role_state_uncertain")
        if opponent_score.loc[idx] < 0.45:
            row_failures.append("OPPONENT_SCHEME_DISRUPTION")
            row_reasons.append("opponent_context_uncertain")
        if distribution_score.loc[idx] < 0.45:
            row_reasons.append("distribution_width_unstable")
        if similar_score.loc[idx] < 0.45:
            row_reasons.append("similar_states_scattered_or_sparse")
        if chaos.loc[idx] > 0.60:
            row_failures.append("PLAYER_STATE_UNFORECASTABLE")
            row_reasons.append("chaos_score_high")
        if not row_reasons:
            row_reasons.append("forecastability_mixed")
        reasons.append(";".join(row_reasons))
        failure_modes.append(";".join(sorted(set(row_failures))))

    out["minutes_forecastability_score"] = minutes_score
    out["usage_forecastability_score"] = usage_score
    out["role_forecastability_score"] = role_score
    out["distribution_stability_score"] = distribution_score
    out["opponent_adjusted_forecastability_score"] = opponent_score
    out["similar_state_reliability_score"] = similar_score
    out["teammate_context_score"] = teammate_score
    out["opponent_context_score"] = opponent_score
    out["overall_player_forecastability_score"] = overall
    out["forecastability_tier"] = tier
    out["forecastability_reasons"] = reasons
    out["forecastability_failure_modes"] = failure_modes
    out["forecastability_blocks_safe_state_flag"] = tier.isin({"LOW_FORECASTABILITY", "UNFORECASTABLE"})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate player-state forecastability for candidate rows.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    annotated = annotate_player_state_forecastability(frame)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_csv, index=False)
    if args.summary_json:
        payload: dict[str, Any] = {
            "rows": int(len(annotated)),
            "forecastability_tier_counts": annotated["forecastability_tier"].value_counts(dropna=False).to_dict(),
            "shadow_only": True,
            "production_behavior_changed": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
