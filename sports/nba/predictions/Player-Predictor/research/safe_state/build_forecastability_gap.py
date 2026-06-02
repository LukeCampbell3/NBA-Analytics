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

from research.safe_state.forecastability_gap_schema import merge_fixability, max_severity


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _has_any(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [col for col in columns if col in frame.columns]
    if not present:
        return pd.Series(False, index=frame.index)
    checks = []
    for col in present:
        values = frame[col]
        checks.append(values.notna() & values.astype(str).str.strip().ne(""))
    return pd.concat(checks, axis=1).any(axis=1)


def _clean_gap_value(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _score_from_volatility(frame: pd.DataFrame, columns: list[str], default: float = 0.50) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in columns:
        if col in frame.columns:
            values = values.where(values.notna(), pd.to_numeric(frame[col], errors="coerce"))
    values = values.fillna(default)
    normalized = values.where(values.le(1.0), values / 12.0)
    return (1.0 - normalized).clip(lower=0.0, upper=1.0)


def annotate_forecastability_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()

    minutes_score = _num(out, "minutes_forecastability_score", np.nan).fillna(_num(out, "minutes_band_stability_score", np.nan))
    if minutes_score.isna().any():
        width = _num(out, "expected_minutes_band_width", np.nan)
        width = width.where(width.notna(), _num(out, "expected_minutes_band_high", np.nan) - _num(out, "expected_minutes_band_low", np.nan))
        floor = _num(out, "minutes_floor_recent", np.nan).fillna(_num(out, "expected_minutes_band_low", np.nan))
        p25 = _num(out, "minutes_p25_recent", np.nan).fillna(_num(out, "expected_minutes_band_low", np.nan))
        band_score = (1.0 - (width.fillna(10.0) / 16.0)).clip(0.0, 1.0)
        floor_score = ((floor.fillna(18.0) - 12.0) / 24.0).clip(0.0, 1.0)
        p25_score = ((p25.fillna(18.0) - 14.0) / 22.0).clip(0.0, 1.0)
        inferred_minutes_score = (0.40 * band_score + 0.40 * floor_score + 0.20 * p25_score).clip(0.0, 1.0)
        minutes_score = minutes_score.where(minutes_score.notna(), inferred_minutes_score)
    minutes_score = minutes_score.fillna(0.0)
    usage_has = _has_any(out, ["usage_forecastability_score", "usage_volatility", "usage_proxy", "FGA_volatility", "fga_volatility", "USG%", "FGA"])
    usage_score = _num(out, "usage_forecastability_score", np.nan).fillna(
        _score_from_volatility(out, ["usage_volatility", "usage_proxy_volatility", "FGA_volatility", "fga_volatility"], default=0.50)
    )
    role_score = _num(out, "role_forecastability_score", np.nan).fillna(
        (1.0 - _num(out, "rotation_volatility_score", 0.35).fillna(0.35)).clip(0.0, 1.0)
    )
    distribution_score = _num(out, "distribution_stability_score", np.nan)
    if distribution_score.isna().any():
        width = _num(out, "distribution_width", np.nan)
        width = width.where(width.notna(), _num(out, "q75", np.nan) - _num(out, "q25", np.nan))
        line = _num(out, "line", np.nan).fillna(_num(out, "market_line", np.nan))
        scale = pd.Series(np.maximum(2.5, line.abs().fillna(12.0) * 0.35), index=out.index)
        inferred_distribution_score = (1.0 - (width / scale)).clip(lower=0.0, upper=1.0)
        distribution_score = distribution_score.where(distribution_score.notna(), inferred_distribution_score)
    distribution_score = distribution_score.fillna(0.0)
    similar_score = _num(out, "similar_state_tightness_score", np.nan).fillna(0.0)
    teammate_has = _has_any(out, ["teammate_context_score", "teammate_availability_flags", "teammate_return_risk", "teammate_availability_uncertainty"])
    teammate_score = _num(out, "teammate_context_score", np.nan).fillna(
        (1.0 - _num(out, "teammate_return_risk", 0.50).fillna(0.50)).clip(0.0, 1.0)
    )
    opponent_has = _has_any(out, ["opponent_context_score", "opponent_context_similarity", "opponent_defensive_context_similarity", "opponent_scheme_disruption_score"])
    opponent_score = _num(out, "opponent_context_score", np.nan).fillna(
        _num(out, "opponent_context_similarity", np.nan).fillna(
            (1.0 - _num(out, "opponent_scheme_disruption_score", 0.50).fillna(0.50)).clip(0.0, 1.0)
        )
    )

    rows: list[dict[str, Any]] = []
    for idx in out.index:
        gaps: list[str] = []
        reasons: list[str] = []
        missing: list[str] = []
        fixabilities: list[str] = []
        severities: list[str] = []

        minutes_gap = _clean_gap_value(out.at[idx, "minutes_state_gap_type"]) if "minutes_state_gap_type" in out.columns else ""
        if minutes_gap:
            gaps.append(minutes_gap)
            reasons.append(str(out.at[idx, "minutes_state_gap_reason"]) if "minutes_state_gap_reason" in out.columns else "minutes_state_gap")
            fixabilities.append(str(out.at[idx, "minutes_state_fixability"]) if "minutes_state_fixability" in out.columns else "UNKNOWN")
            severities.append("HIGH" if minutes_gap == "FORECASTABILITY_GAP_MINUTES_STATE" else "MEDIUM")
        elif minutes_score.loc[idx] < 0.45:
            gaps.append("FORECASTABILITY_GAP_MINUTES_STATE")
            reasons.append(f"minutes_score={minutes_score.loc[idx]:.2f}")
            fixabilities.append("TRUE_UNSTABLE_STATE")
            severities.append("HIGH")

        if not bool(usage_has.loc[idx]):
            gaps.append("FORECASTABILITY_GAP_USAGE_STATE")
            reasons.append("usage_proxy_missing")
            missing.append("usage_proxy")
            fixabilities.append("FIXABLE_WITH_NEW_PIPELINE_DATA")
            severities.append("MEDIUM")
        elif usage_score.loc[idx] < 0.45:
            gaps.append("FORECASTABILITY_GAP_USAGE_STATE")
            reasons.append(f"usage_score={usage_score.loc[idx]:.2f}")
            fixabilities.append("TRUE_UNSTABLE_STATE")
            severities.append("HIGH")

        if role_score.loc[idx] < 0.45:
            gaps.append("FORECASTABILITY_GAP_ROLE_STATE")
            reasons.append(f"role_score={role_score.loc[idx]:.2f}")
            fixabilities.append("TRUE_UNSTABLE_STATE")
            severities.append("HIGH")

        distribution_gap = _clean_gap_value(out.at[idx, "distribution_gap_type"]) if "distribution_gap_type" in out.columns else ""
        if distribution_gap:
            gaps.append(distribution_gap)
            reasons.append(str(out.at[idx, "distribution_gap_reason"]) if "distribution_gap_reason" in out.columns else "distribution_gap")
            fixabilities.append(str(out.at[idx, "distribution_gap_fixability"]) if "distribution_gap_fixability" in out.columns else "UNKNOWN")
            severities.append("HIGH" if distribution_gap == "FORECASTABILITY_GAP_DISTRIBUTION_WIDTH" else "MEDIUM")
        elif distribution_score.loc[idx] < 0.45:
            gaps.append("FORECASTABILITY_GAP_DISTRIBUTION_WIDTH")
            reasons.append(f"distribution_stability_score={distribution_score.loc[idx]:.2f}")
            fixabilities.append("TRUE_UNSTABLE_STATE")
            severities.append("HIGH")

        similar_gap = _clean_gap_value(out.at[idx, "similar_state_gap_type"]) if "similar_state_gap_type" in out.columns else ""
        similar_tier = _clean_gap_value(out.at[idx, "similar_state_reliability_tier"]) if "similar_state_reliability_tier" in out.columns else ""
        if similar_gap:
            gaps.append(similar_gap)
            reasons.append(str(out.at[idx, "similar_state_gap_reason"]) if "similar_state_gap_reason" in out.columns else "similar_state_gap")
            fixabilities.append(str(out.at[idx, "similar_state_gap_fixability"]) if "similar_state_gap_fixability" in out.columns else "NEEDS_MORE_SAMPLE")
            severities.append("HIGH" if similar_gap == "FORECASTABILITY_GAP_SIMILAR_STATE_SCATTER" else "MEDIUM")
        elif similar_tier.upper() == "INSUFFICIENT_SAMPLE":
            gaps.append("FORECASTABILITY_GAP_SIMILAR_STATE_SAMPLE")
            reasons.append("similar_state_sample_insufficient")
            fixabilities.append("NEEDS_MORE_SAMPLE")
            severities.append("MEDIUM")
        elif similar_tier.upper() == "SCATTERED":
            gaps.append("FORECASTABILITY_GAP_SIMILAR_STATE_SCATTER")
            reasons.append("similar_states_scattered")
            fixabilities.append("TRUE_UNSTABLE_STATE")
            severities.append("HIGH")

        if not bool(teammate_has.loc[idx]):
            gaps.append("FORECASTABILITY_GAP_TEAMMATE_CONTEXT")
            reasons.append("teammate_context_missing")
            missing.append("teammate_availability")
            fixabilities.append("FIXABLE_WITH_NEW_PIPELINE_DATA")
            severities.append("LOW")
        elif teammate_score.loc[idx] < 0.40:
            gaps.append("FORECASTABILITY_GAP_TEAMMATE_CONTEXT")
            reasons.append(f"teammate_context_score={teammate_score.loc[idx]:.2f}")
            fixabilities.append("TRUE_UNSTABLE_STATE")
            severities.append("MEDIUM")

        if not bool(opponent_has.loc[idx]):
            gaps.append("FORECASTABILITY_GAP_OPPONENT_CONTEXT")
            reasons.append("opponent_context_missing")
            missing.append("opponent_context")
            fixabilities.append("FIXABLE_WITH_NEW_PIPELINE_DATA")
            severities.append("LOW")
        elif opponent_score.loc[idx] < 0.40:
            gaps.append("FORECASTABILITY_GAP_OPPONENT_CONTEXT")
            reasons.append(f"opponent_context_score={opponent_score.loc[idx]:.2f}")
            fixabilities.append("TRUE_UNSTABLE_STATE")
            severities.append("MEDIUM")

        if missing and not gaps:
            gaps.append("FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA")
            reasons.append("missing_pre_event_forecastability_features")
            fixabilities.append("FIXABLE_WITH_NEW_PIPELINE_DATA")
            severities.append("MEDIUM")

        primary = gaps[0] if gaps else ""
        secondary = ";".join(gaps[1:])
        severity = max_severity(severities)
        fixability = merge_fixability(fixabilities)
        true_unstable = fixability == "TRUE_UNSTABLE_STATE"
        if true_unstable and "FORECASTABILITY_GAP_TRUE_UNSTABLE_STATE" not in gaps:
            secondary = ";".join([part for part in [secondary, "FORECASTABILITY_GAP_TRUE_UNSTABLE_STATE"] if part])

        rows.append(
            {
                "forecastability_gap_primary": primary,
                "forecastability_gap_secondary": secondary,
                "forecastability_gap_count": int(len(gaps)),
                "forecastability_gap_reasons": ";".join(reasons),
                "forecastability_gap_missing_features": ";".join(sorted(set(missing))),
                "forecastability_gap_fixability": fixability,
                "forecastability_gap_blocks_safe_state_flag": bool(gaps),
                "forecastability_gap_severity": severity,
                "minutes_forecastability_score": minutes_score.loc[idx],
                "usage_forecastability_score": usage_score.loc[idx],
                "role_forecastability_score": role_score.loc[idx],
                "distribution_stability_score": distribution_score.loc[idx],
                "similar_state_tightness_score": similar_score.loc[idx],
                "teammate_context_score": teammate_score.loc[idx],
                "opponent_context_score": opponent_score.loc[idx],
            }
        )

    gap_frame = pd.DataFrame(rows, index=out.index)
    for col in gap_frame.columns:
        out[col] = gap_frame[col]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build specific forecastability gap diagnostics.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    out = annotate_forecastability_gaps(candidates)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    summary = (
        out.groupby(["forecastability_gap_primary", "forecastability_gap_fixability", "forecastability_gap_severity"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values("candidate_count", ascending=False)
    )
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
    if args.summary_json:
        payload = {
            "rows": int(len(out)),
            "primary_gap_counts": out["forecastability_gap_primary"].fillna("").astype(str).value_counts().to_dict(),
            "fixability_counts": out["forecastability_gap_fixability"].fillna("").astype(str).value_counts().to_dict(),
            "severity_counts": out["forecastability_gap_severity"].fillna("").astype(str).value_counts().to_dict(),
            "production_behavior_changed": False,
            "promotion_claim": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
