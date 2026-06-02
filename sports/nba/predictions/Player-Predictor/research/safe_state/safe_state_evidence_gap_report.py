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

from research.market_quality.common import candidate_identity_columns
from research.safe_state.analyze_minutes_forecastability_gap import annotate_minutes_gap_decomposition
from research.safe_state.analyze_usage_forecastability_gap import annotate_usage_gap_decomposition
from research.safe_state.safe_state_classifier import annotate_safe_state_stack


CORE_FORECASTABILITY_TIERS = {"HIGH_FORECASTABILITY", "MEDIUM_FORECASTABILITY"}
CORE_SIMILAR_TIERS = {"TIGHT", "ACCEPTABLE"}
CORE_STRUCTURAL_TIERS = {"STRUCTURAL_MISPRICE_STRONG", "STRUCTURAL_MISPRICE_ACCEPTABLE"}
SEVERE_FAILURE_PATTERNS = ["INJURY", "NEWS_UNRESOLVED", "PLAYER_STATE_UNFORECASTABLE", "DATA_MISSING"]


FEATURE_GROUPS: dict[str, dict[str, Any]] = {
    "timestamp_safe_price": {
        "columns": ["market_side_price", "market_side_break_even", "price_validity_status", "odds_snapshot_time", "price_source"],
        "gap_type": "PRICE_GAP",
        "fix": "Persist timestamp-safe side-specific entry odds, break-even, timestamp, and source provenance.",
    },
    "minutes_state": {
        "columns": ["expected_minutes_band_low", "expected_minutes_band_high", "expected_minutes_band_width", "minutes_floor_recent"],
        "gap_type": "FORECASTABILITY_GAP",
        "fix": "Backfill expected minutes bands and recent minutes floor/p25 from refreshed player logs.",
    },
    "usage_proxy": {
        "columns": ["usage_volatility", "usage_proxy", "fga_volatility", "FGA_volatility"],
        "gap_type": "FORECASTABILITY_GAP",
        "fix": "Add pre-event usage/FGA/touch proxy history for player-state stability.",
    },
    "teammate_availability": {
        "columns": ["teammate_availability_flags", "teammate_return_risk", "teammate_availability_uncertainty"],
        "gap_type": "FEATURE_MISSING_GAP",
        "fix": "Persist teammate in/out and return-risk context before slate selection.",
    },
    "opponent_context": {
        "columns": ["opponent_context_similarity", "opponent_defensive_context_similarity", "opponent_scheme_disruption_score"],
        "gap_type": "SCENARIO_GAP",
        "fix": "Add opponent scheme/pace/defensive-class context to comparable-state features.",
    },
    "distribution_quantiles": {
        "columns": ["model_mean", "prediction", "raw_prediction", "q25", "q50", "q75", "q90", "line_percentile"],
        "gap_type": "STRUCTURAL_MISPRICING_GAP",
        "fix": "Export conservative stat distribution quantiles and line percentile for every candidate.",
    },
    "similar_state_sample": {
        "columns": ["similar_state_count", "similar_state_tightness_score", "similar_state_reliability_tier"],
        "gap_type": "SIMILAR_STATE_GAP",
        "fix": "Build a larger pre-event comparable-state store by player/archetype/target/line zone.",
    },
    "structural_pathway": {
        "columns": ["structural_pathway_score", "rebound_supply_score", "team_assist_environment_score", "usage_forecastability_score"],
        "gap_type": "STRUCTURAL_MISPRICING_GAP",
        "fix": "Persist target-specific basketball pathway diagnostics for points, assists, rebounds, and combos.",
    },
    "event_start_verification": {
        "columns": ["market_commence_time_utc", "timestamp_safety_basis"],
        "gap_type": "FEATURE_MISSING_GAP",
        "fix": "Resolve provider or schedule event start time so price safety can be EVENT_START_VERIFIED.",
    },
}


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _identity(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    return candidate_identity_columns(frame.copy()).get("candidate_id", pd.Series("", index=frame.index)).fillna("").astype(str)


def _join_key(row: pd.Series) -> str:
    parts = [
        row.get("game_id", ""),
        row.get("market_date", row.get("game_date", "")),
        row.get("player", row.get("player_name", "")),
        row.get("target", ""),
        row.get("side", row.get("direction", "")),
        row.get("line", row.get("market_line", "")),
    ]
    return "::".join("" if pd.isna(part) else str(part) for part in parts)


def _is_missing_feature(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [col for col in columns if col in frame.columns]
    if not present:
        return pd.Series(True, index=frame.index)
    checks = []
    for col in present:
        values = frame[col]
        if pd.api.types.is_numeric_dtype(values):
            checks.append(pd.to_numeric(values, errors="coerce").notna())
        else:
            checks.append(values.fillna("").astype(str).str.strip().ne(""))
    return ~pd.concat(checks, axis=1).any(axis=1)


def _missing_features_for_row(row: pd.Series) -> list[str]:
    missing: list[str] = []
    for group, config in FEATURE_GROUPS.items():
        cols = [col for col in config["columns"] if col in row.index]
        if not cols:
            missing.append(group)
            continue
        has_any = False
        for col in cols:
            value = row.get(col)
            if pd.notna(value) and str(value).strip() != "":
                has_any = True
                break
        if not has_any:
            missing.append(group)
    return missing


def _safe_price_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        _text(frame, "edge_defendability_tier").str.upper().eq("EDGE_DEFENDABLE")
        & _text(frame, "price_validity_status").str.upper().eq("PRICE_VALID")
        & _num(frame, "stress_edge", 0.0).gt(0.0)
        & _num(frame, "lcb_edge", 0.0).gt(0.0)
    )


def _failure_mode_mask(frame: pd.DataFrame) -> pd.Series:
    failure_text = (
        _text(frame, "known_failure_modes")
        + ";"
        + _text(frame, "forecastability_failure_modes")
        + ";"
        + _text(frame, "failure_modes")
    ).str.upper()
    mask = pd.Series(False, index=frame.index)
    for pattern in SEVERE_FAILURE_PATTERNS:
        mask = mask | failure_text.str.contains(pattern, regex=False)
    return mask


def _blocker_groups(frame: pd.DataFrame) -> pd.DataFrame:
    price_clear = _safe_price_mask(frame)
    forecast_gap_severity = _text(frame, "forecastability_gap_severity", "NONE").str.upper().str.strip()
    forecast_gap_primary = _text(frame, "forecastability_gap_primary").str.upper().str.strip()
    forecast_ok = (
        _text(frame, "forecastability_tier").str.upper().isin(CORE_FORECASTABILITY_TIERS)
        & ~forecast_gap_severity.isin({"HIGH", "CRITICAL"})
        & forecast_gap_primary.eq("")
    )
    similar_ok = _text(frame, "similar_state_reliability_tier").str.upper().isin(CORE_SIMILAR_TIERS)
    structural_ok = _text(frame, "structural_mispricing_tier").str.upper().isin(CORE_STRUCTURAL_TIERS)
    scenario_ok = _num(frame, "chaos_score", 1.0).le(0.35) & _num(frame, "scenario_agreement", 0.0).ge(0.65)
    failure_ok = ~_failure_mode_mask(frame)
    return pd.DataFrame(
        {
            "PRICE_GAP": ~price_clear,
            "FORECASTABILITY_GAP": ~forecast_ok,
            "SIMILAR_STATE_GAP": ~similar_ok,
            "STRUCTURAL_MISPRICING_GAP": ~structural_ok,
            "SCENARIO_GAP": ~scenario_ok,
            "FAILURE_MODE_GAP": ~failure_ok,
        },
        index=frame.index,
    )


def _classify_row_blockers(row: pd.Series, blocker_row: pd.Series) -> tuple[str, list[str], list[str], str, list[str]]:
    blockers = [name for name, value in blocker_row.items() if bool(value)]
    missing = _missing_features_for_row(row)
    forecast_missing = str(row.get("forecastability_gap_missing_features", "") or "")
    missing.extend([part for part in forecast_missing.split(";") if part.strip()])
    evidence_gaps = list(blockers)
    forecast_gap_primary = _clean_text(row.get("forecastability_gap_primary", ""))
    forecast_gap_secondary = _clean_text(row.get("forecastability_gap_secondary", ""))
    forecast_sub_gaps = [part for part in [forecast_gap_primary] + [part.strip() for part in forecast_gap_secondary.split(";")] if _clean_text(part)]
    if "FORECASTABILITY_GAP" in evidence_gaps and forecast_sub_gaps and forecast_gap_primary:
        evidence_gaps = [gap for gap in evidence_gaps if gap != "FORECASTABILITY_GAP"] + forecast_sub_gaps

    if str(row.get("similar_state_reliability_tier", "")).upper() == "INSUFFICIENT_SAMPLE" or pd.to_numeric(pd.Series([row.get("similar_state_count")]), errors="coerce").fillna(0).iloc[0] < 5:
        if "SAMPLE_SIZE_GAP" not in evidence_gaps:
            evidence_gaps.append("SAMPLE_SIZE_GAP")
    if missing:
        if "FEATURE_MISSING_GAP" not in evidence_gaps:
            evidence_gaps.append("FEATURE_MISSING_GAP")

    priority = [
        "PRICE_GAP",
        "FAILURE_MODE_GAP",
        "FORECASTABILITY_GAP_MINUTES_STATE",
        "FORECASTABILITY_GAP_ROLE_STATE",
        "FORECASTABILITY_GAP_USAGE_STATE",
        "FORECASTABILITY_GAP_DISTRIBUTION_WIDTH",
        "FORECASTABILITY_GAP_SIMILAR_STATE_SCATTER",
        "FORECASTABILITY_GAP_TRUE_UNSTABLE_STATE",
        "FORECASTABILITY_GAP_SIMILAR_STATE_SAMPLE",
        "FORECASTABILITY_GAP_TEAMMATE_CONTEXT",
        "FORECASTABILITY_GAP_OPPONENT_CONTEXT",
        "FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA",
        "SIMILAR_STATE_GAP",
        "SAMPLE_SIZE_GAP",
        "STRUCTURAL_MISPRICING_GAP",
        "SCENARIO_GAP",
        "FEATURE_MISSING_GAP",
    ]
    primary = next((gap for gap in priority if gap in evidence_gaps), "")
    if not primary:
        primary = forecast_gap_primary if forecast_gap_primary else next((gap for gap in evidence_gaps if gap), "NONE")
    if primary == "FORECASTABILITY_GAP" and forecast_gap_primary:
        primary = forecast_gap_primary
    secondary = [gap for gap in evidence_gaps if gap != primary]
    return primary, secondary, missing, ";".join(evidence_gaps) if evidence_gaps else "NONE", blockers


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _gap_subtype_for_row(row: pd.Series, primary: str) -> tuple[str, str, str, str]:
    if primary == "FORECASTABILITY_GAP_MINUTES_STATE":
        return (
            _clean_text(row.get("minutes_gap_subtype", "")),
            _clean_text(row.get("minutes_gap_fixability", "")),
            _clean_text(row.get("minutes_gap_severity", "")),
            _clean_text(row.get("minutes_gap_recommended_fix", "")),
        )
    if primary == "FORECASTABILITY_GAP_USAGE_STATE":
        return (
            _clean_text(row.get("usage_gap_subtype", "")),
            _clean_text(row.get("usage_gap_fixability", "")),
            _clean_text(row.get("usage_gap_severity", "")),
            _clean_text(row.get("usage_gap_recommended_fix", "")),
        )
    return "", "", "", ""


def _fixability_for_blocker(primary: str, missing_features: list[str], row: pd.Series | None = None) -> str:
    if row is not None:
        subtype, subtype_fixability, _, _ = _gap_subtype_for_row(row, primary)
        if subtype_fixability:
            return subtype_fixability
        forecast_fixability = _clean_text(row.get("forecastability_gap_fixability", ""))
        if primary.startswith("FORECASTABILITY_GAP_") and forecast_fixability:
            return forecast_fixability
        if subtype:
            return "fixable_with_more_data"
    if primary in {"SIMILAR_STATE_GAP", "SAMPLE_SIZE_GAP", "FEATURE_MISSING_GAP"}:
        return "fixable_with_more_data"
    if primary == "STRUCTURAL_MISPRICING_GAP" and (
        "distribution_quantiles" in missing_features or "structural_pathway" in missing_features
    ):
        return "fixable_with_more_data"
    if primary == "FORECASTABILITY_GAP" and (
        "minutes_state" in missing_features or "usage_proxy" in missing_features or "teammate_availability" in missing_features
    ):
        return "fixable_with_more_data"
    if primary == "PRICE_GAP":
        return "price_pipeline_blocked"
    if primary in {"FAILURE_MODE_GAP", "SCENARIO_GAP", "FORECASTABILITY_GAP", "STRUCTURAL_MISPRICING_GAP"}:
        return "potentially_truly_unsafe"
    return "not_blocked"


def _build_candidate_blockers(annotated: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_identity_columns(annotated.copy())
    blocker_flags = _blocker_groups(frame)
    primary_values: list[str] = []
    secondary_values: list[str] = []
    missing_values: list[str] = []
    gap_type_values: list[str] = []
    raw_blocker_values: list[str] = []
    fixability_values: list[str] = []

    for idx, row in frame.iterrows():
        primary, secondary, missing, gap_types, blockers = _classify_row_blockers(row, blocker_flags.loc[idx])
        primary_values.append(primary)
        secondary_values.append(";".join(secondary))
        missing_values.append(";".join(missing))
        gap_type_values.append(gap_types)
        raw_blocker_values.append(";".join(blockers))
        fixability_values.append(_fixability_for_blocker(primary, missing, row))

    price_clear = _safe_price_mask(frame)
    major_blocker_counts = blocker_flags.drop(columns=["PRICE_GAP"]).sum(axis=1)
    classifier_near_core = _text(frame, "safe_state_tier").eq("SAFE_STATE_NEAR_CORE")
    near_core = classifier_near_core | (price_clear & major_blocker_counts.eq(1))
    gap_tier = _text(frame, "safe_state_tier").where(~near_core, "SAFE_STATE_NEAR_CORE")

    out = pd.DataFrame(
        {
            "candidate_id": frame.get("candidate_id", pd.Series("", index=frame.index)),
            "game_id": frame.get("game_id", pd.Series("", index=frame.index)),
            "market_date": frame.get("market_date", frame.get("game_date", pd.Series("", index=frame.index))),
            "player": frame.get("player", frame.get("player_name", pd.Series("", index=frame.index))),
            "target": frame.get("target", pd.Series("", index=frame.index)),
            "market_type": frame.get("market_type", pd.Series("", index=frame.index)),
            "side": frame.get("side", frame.get("direction", pd.Series("", index=frame.index))),
            "line": frame.get("line", frame.get("market_line", pd.Series(np.nan, index=frame.index))),
            "edge_defendability_tier": _text(frame, "edge_defendability_tier"),
            "price_validity_status": _text(frame, "price_validity_status"),
            "forecastability_tier": _text(frame, "forecastability_tier"),
            "similar_state_reliability_tier": _text(frame, "similar_state_reliability_tier"),
            "structural_mispricing_tier": _text(frame, "structural_mispricing_tier"),
            "safe_state_tier": _text(frame, "safe_state_tier"),
            "safe_state_gap_tier": gap_tier,
            "safe_state_score": _num(frame, "safe_state_score"),
            "stress_edge": _num(frame, "stress_edge"),
            "lcb_edge": _num(frame, "lcb_edge"),
            "overall_player_forecastability_score": _num(frame, "overall_player_forecastability_score"),
            "overall_structural_mispricing_score": _num(frame, "overall_structural_mispricing_score"),
            "similar_state_tightness_score": _num(frame, "similar_state_tightness_score"),
            "similar_state_count": _num(frame, "similar_state_count"),
            "safe_state_blockers": _text(frame, "safe_state_blockers").where(_text(frame, "safe_state_blockers").str.strip().ne(""), pd.Series(raw_blocker_values, index=frame.index)),
            "primary_blocker": primary_values,
            "secondary_blockers": secondary_values,
            "missing_features": missing_values,
            "evidence_gap_type": gap_type_values,
            "minutes_gap_subtype": _text(frame, "minutes_gap_subtype"),
            "usage_gap_subtype": _text(frame, "usage_gap_subtype"),
            "gap_subtype": [_gap_subtype_for_row(row, primary)[0] for primary, (_, row) in zip(primary_values, frame.iterrows())],
            "gap_fixability": [_gap_subtype_for_row(row, primary)[1] or fixability for primary, fixability, (_, row) in zip(primary_values, fixability_values, frame.iterrows())],
            "gap_severity": [_gap_subtype_for_row(row, primary)[2] or _text(pd.DataFrame([row]), "forecastability_gap_severity").iloc[0] for primary, (_, row) in zip(primary_values, frame.iterrows())],
            "gap_recommended_fix": [_gap_subtype_for_row(row, primary)[3] for primary, (_, row) in zip(primary_values, frame.iterrows())],
            "safe_state_near_core_flag": near_core,
            "near_core_blocker_fixability": fixability_values,
            "settlement_join_key": frame.apply(_join_key, axis=1),
        }
    )
    return out


def _feature_gap_rankings(blockers: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    edge_defendable = blockers["edge_defendability_tier"].astype(str).eq("EDGE_DEFENDABLE")
    for group, config in FEATURE_GROUPS.items():
        has_group = blockers["missing_features"].fillna("").astype(str).str.split(";").apply(lambda values: group in values)
        weak_signal = pd.Series(False, index=blockers.index)
        if group == "similar_state_sample":
            weak_signal = blockers["evidence_gap_type"].astype(str).str.contains("SIMILAR_STATE_GAP|SAMPLE_SIZE_GAP", regex=True)
        elif group == "distribution_quantiles":
            weak_signal = blockers["evidence_gap_type"].astype(str).str.contains("FORECASTABILITY_GAP_DISTRIBUTION_WIDTH", regex=True)
        elif group == "structural_pathway":
            weak_signal = blockers["evidence_gap_type"].astype(str).str.contains("STRUCTURAL_MISPRICING_GAP", regex=True)
        elif group == "minutes_state":
            weak_signal = blockers["evidence_gap_type"].astype(str).str.contains("FORECASTABILITY_GAP_MINUTES_STATE", regex=True)
        elif group == "usage_proxy":
            weak_signal = blockers["evidence_gap_type"].astype(str).str.contains("FORECASTABILITY_GAP_USAGE_STATE", regex=True)
        elif group == "teammate_availability":
            weak_signal = blockers["evidence_gap_type"].astype(str).str.contains("FORECASTABILITY_GAP_TEAMMATE_CONTEXT", regex=True)
        elif group == "opponent_context":
            weak_signal = blockers["evidence_gap_type"].astype(str).str.contains("FORECASTABILITY_GAP_OPPONENT_CONTEXT|SCENARIO_GAP", regex=True)
        mask = has_group | weak_signal
        if not mask.any():
            continue
        detail = blockers.loc[mask].copy()
        if "gap_subtype" not in detail.columns:
            detail["gap_subtype"] = ""
        if "gap_fixability" not in detail.columns:
            detail["gap_fixability"] = detail.get("near_core_blocker_fixability", pd.Series("", index=detail.index))
        group_missing = has_group.reindex(detail.index).fillna(False).astype(bool)
        group_weak = weak_signal.reindex(detail.index).fillna(False).astype(bool)
        raw_subtype = detail["gap_subtype"].fillna("").astype(str)
        raw_fixability = detail["gap_fixability"].fillna("").astype(str)
        detail["blocker_subtype"] = np.where(
            group_weak & raw_subtype.str.strip().ne(""),
            raw_subtype,
            f"{group.upper()}_MISSING",
        )
        detail["fixability"] = np.where(
            group_missing & ~group_weak,
            "FEATURE_MISSING",
            raw_fixability.where(raw_fixability.str.strip().ne(""), "UNKNOWN"),
        )
        for (subtype, fixability), grouped in detail.groupby(["blocker_subtype", "fixability"], dropna=False):
            grouped_mask = detail.index.isin(grouped.index)
            blocked_edge = int((mask & edge_defendable & blockers.index.isin(grouped.index)).sum())
            blocked_total = int(grouped_mask.sum())
            priority = "HIGH" if blocked_edge >= 2 or blocked_total >= 5 else "MEDIUM" if blocked_edge >= 1 or blocked_total >= 2 else "LOW"
            likely_fix = config["fix"]
            if str(fixability).upper() == "TRUE_UNSTABLE_STATE":
                priority = "REJECT_UNSAFE"
                likely_fix = "No pipeline fix recommended. More data may confirm instability, but current state remains unsafe."
            records.append(
                {
                    "feature_or_signal": group,
                    "blocked_candidate_count": blocked_total,
                    "blocked_edge_defendable_count": blocked_edge,
                    "blocker_subtype": subtype,
                    "fixability": fixability,
                    "affected_markets": ";".join(sorted(set(grouped["market_type"].fillna("").astype(str).tolist()))),
                    "likely_pipeline_fix": likely_fix,
                    "priority": priority,
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "feature_or_signal",
                "blocked_candidate_count",
                "blocked_edge_defendable_count",
                "blocker_subtype",
                "fixability",
                "affected_markets",
                "likely_pipeline_fix",
                "priority",
            ]
        )
    out = pd.DataFrame.from_records(records)
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    out["_priority_order"] = out["priority"].map(priority_order).fillna(9)
    return out.sort_values(["_priority_order", "blocked_edge_defendable_count", "blocked_candidate_count"], ascending=[True, False, False]).drop(columns=["_priority_order"]).reset_index(drop=True)


def _split_feature_rankings(feature_rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_rankings.empty:
        return feature_rankings.copy(), feature_rankings.copy()
    fix = feature_rankings.get("fixability", pd.Series("", index=feature_rankings.index)).fillna("").astype(str).str.upper()
    subtype = feature_rankings.get("blocker_subtype", pd.Series("", index=feature_rankings.index)).fillna("").astype(str).str.upper()
    likely = feature_rankings.get("likely_pipeline_fix", pd.Series("", index=feature_rankings.index)).fillna("").astype(str).str.upper()
    non_actionable_mask = (
        fix.eq("TRUE_UNSTABLE_STATE")
        | subtype.str.contains("REAL_MINUTES_VOLATILITY|REAL_USAGE_VOLATILITY|TRUE_UNSTABLE|MINUTES_LOW_FLOOR|MINUTES_WIDE_BAND|MINUTES_HIGH_VOLATILITY|MINUTES_ROLE_UNSTABLE", regex=True)
        | likely.str.contains("CURRENT STATE REMAINS UNSAFE", regex=False)
    )
    non_actionable = feature_rankings.loc[non_actionable_mask].copy()
    if not non_actionable.empty:
        non_actionable["priority"] = "REJECT_UNSAFE"
        non_actionable["recommended_action"] = "KEEP_UNSAFE_TRUE_VOLATILITY"
        non_actionable["likely_pipeline_fix"] = "No pipeline fix recommended. More data may confirm instability, but current state remains unsafe."
    actionable = feature_rankings.loc[~non_actionable_mask].copy()
    if not actionable.empty:
        actionable["recommended_action"] = actionable.get("fixability", pd.Series("", index=actionable.index)).fillna("").astype(str).map(
            {
                "NEEDS_MORE_SAMPLE": "BUILD_SAMPLE_AND_RECHECK",
                "FIXABLE_WITH_EXISTING_LOGS": "BACKFILL_EXISTING_LOG_FEATURES",
                "FIXABLE_WITH_NEW_PIPELINE_DATA": "ADD_NEW_PIPELINE_DATA",
                "FEATURE_MISSING": "ADD_NEW_PIPELINE_DATA",
            }
        ).fillna("WATCH")
    return actionable.reset_index(drop=True), non_actionable.reset_index(drop=True)


def _variant_board(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    price_clear = _safe_price_mask(frame)
    forecast_ok = _text(frame, "forecastability_tier").str.upper().isin(CORE_FORECASTABILITY_TIERS)
    similar_ok = _text(frame, "similar_state_reliability_tier").str.upper().isin(CORE_SIMILAR_TIERS)
    structural_ok = _text(frame, "structural_mispricing_tier").str.upper().isin(CORE_STRUCTURAL_TIERS)
    scenario_ok = _num(frame, "chaos_score", 1.0).le(0.35) & _num(frame, "scenario_agreement", 0.0).ge(0.65)
    failure_ok = ~_failure_mode_mask(frame)
    blockers = _blocker_groups(frame)
    near_core = price_clear & blockers.drop(columns=["PRICE_GAP"]).sum(axis=1).eq(1)

    if variant == "strict_current":
        mask = _text(frame, "safe_state_tier").eq("SAFE_STATE_CORE")
    elif variant == "similar_state_relaxed_only":
        mask = price_clear & forecast_ok & structural_ok & scenario_ok & failure_ok
    elif variant == "forecastability_relaxed_only":
        mask = price_clear & similar_ok & structural_ok & scenario_ok & failure_ok
    elif variant == "structural_relaxed_only":
        mask = price_clear & forecast_ok & similar_ok & scenario_ok & failure_ok
    elif variant == "near_core_allowed":
        mask = _text(frame, "safe_state_tier").eq("SAFE_STATE_CORE") | near_core
    else:
        mask = pd.Series(False, index=frame.index)

    sort_cols = [c for c in ["safe_state_score", "lcb_edge", "stress_edge"] if c in frame.columns]
    board = frame.loc[mask].copy()
    for col in sort_cols:
        board[col] = pd.to_numeric(board[col], errors="coerce")
    if sort_cols:
        board = board.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return board


def _threshold_sensitivity(frame: pd.DataFrame, production: pd.DataFrame, blockers: pd.DataFrame) -> list[dict[str, Any]]:
    production_ids = set(_identity(production).tolist())
    blocker_by_id = blockers.set_index("candidate_id", drop=False) if "candidate_id" in blockers.columns else pd.DataFrame()
    records: list[dict[str, Any]] = []
    for variant in [
        "strict_current",
        "similar_state_relaxed_only",
        "forecastability_relaxed_only",
        "structural_relaxed_only",
        "near_core_allowed",
    ]:
        board = _variant_board(frame, variant)
        board_ids = set(_identity(board).tolist())
        blocker_rows = blocker_by_id.reindex(list(board_ids)).dropna(how="all") if not blocker_by_id.empty and board_ids else pd.DataFrame()
        records.append(
            {
                "variant": variant,
                "board_size": int(len(board)),
                "SAFE_STATE_CORE_count": int(_text(board, "safe_state_tier").eq("SAFE_STATE_CORE").sum()) if not board.empty else 0,
                "SAFE_STATE_NEAR_CORE_count": int(blocker_rows.get("safe_state_near_core_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not blocker_rows.empty else 0,
                "avg_lcb_edge": None if board.empty else _float_or_none(_num(board, "lcb_edge").mean()),
                "avg_forecastability": None if board.empty else _float_or_none(_num(board, "overall_player_forecastability_score").mean()),
                "avg_structural_mispricing": None if board.empty else _float_or_none(_num(board, "overall_structural_mispricing_score").mean()),
                "avg_similar_state_tightness": None if board.empty else _float_or_none(_num(board, "similar_state_tightness_score").mean()),
                "overlap_with_production": int(len(board_ids & production_ids)),
                "missing_evidence_warnings": ";".join(sorted(set(";".join(blocker_rows.get("missing_features", pd.Series(dtype=str)).fillna("").astype(str)).split(";")) - {""})) if not blocker_rows.empty else "",
            }
        )
    return records


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_safe_state_evidence_gap_report(
    *,
    output_dir: Path,
    candidate_pool_csv: Path | None = None,
    production_board_csv: Path | None = None,
    safe_state_dir: Path | None = None,
    annotated_candidates_csv: Path | None = None,
    historical_csv: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_csv(candidate_pool_csv)
    production = _read_csv(production_board_csv)

    annotated_path = annotated_candidates_csv
    if annotated_path is None and safe_state_dir is not None:
        annotated_path = safe_state_dir / "safe_state_annotated_candidates.csv"
    annotated = _read_csv(annotated_path)
    price_defense_board = _read_csv(safe_state_dir / "price_defense_only_board.csv") if safe_state_dir is not None else pd.DataFrame()
    if annotated.empty:
        history = _read_csv(historical_csv)
        annotated = annotate_safe_state_stack(candidates, history)

    annotated = candidate_identity_columns(annotated)
    annotated = annotate_minutes_gap_decomposition(annotated)
    annotated = annotate_usage_gap_decomposition(annotated)
    production = candidate_identity_columns(production) if not production.empty else production
    blockers = _build_candidate_blockers(annotated)
    feature_rankings = _feature_gap_rankings(blockers)
    actionable_gaps, non_actionable_instability = _split_feature_rankings(feature_rankings)
    sensitivity = _threshold_sensitivity(annotated, production, blockers)

    candidate_blockers_path = output_dir / "safe_state_candidate_blockers.csv"
    feature_rankings_path = output_dir / "safe_state_feature_gap_rankings.csv"
    actionable_gaps_path = output_dir / "safe_state_actionable_evidence_gaps.csv"
    non_actionable_path = output_dir / "safe_state_non_actionable_instability.csv"
    sensitivity_path = output_dir / "safe_state_threshold_sensitivity.csv"
    blockers.to_csv(candidate_blockers_path, index=False)
    feature_rankings.to_csv(feature_rankings_path, index=False)
    actionable_gaps.to_csv(actionable_gaps_path, index=False)
    non_actionable_instability.to_csv(non_actionable_path, index=False)
    pd.DataFrame(sensitivity).to_csv(sensitivity_path, index=False)

    near_core = blockers.loc[blockers["safe_state_near_core_flag"].fillna(False).astype(bool)].copy()
    price_defense_board_ids = set(_identity(price_defense_board).tolist()) if not price_defense_board.empty else set()
    price_defense_board_blockers = (
        blockers.loc[blockers["candidate_id"].astype(str).isin(price_defense_board_ids)].copy()
        if price_defense_board_ids
        else pd.DataFrame()
    )
    near_core_candidates = near_core[
        [
            "candidate_id",
            "player",
            "market_type",
            "side",
            "line",
            "primary_blocker",
            "secondary_blockers",
            "near_core_blocker_fixability",
            "missing_features",
            "settlement_join_key",
        ]
    ].to_dict(orient="records")

    gap_counts = blockers["primary_blocker"].fillna("NONE").astype(str).value_counts().to_dict()
    edge_defendable = blockers["edge_defendability_tier"].astype(str).eq("EDGE_DEFENDABLE")
    report = {
        "input_paths": {
            "candidate_pool_csv": str(candidate_pool_csv) if candidate_pool_csv else "",
            "production_board_csv": str(production_board_csv) if production_board_csv else "",
            "safe_state_dir": str(safe_state_dir) if safe_state_dir else "",
            "annotated_candidates_csv": str(annotated_path) if annotated_path else "",
        },
        "output_paths": {
            "json": str(output_dir / "safe_state_evidence_gap_report.json"),
            "markdown": str(output_dir / "safe_state_evidence_gap_report.md"),
            "candidate_blockers_csv": str(candidate_blockers_path),
            "feature_gap_rankings_csv": str(feature_rankings_path),
            "actionable_evidence_gaps_csv": str(actionable_gaps_path),
            "non_actionable_instability_csv": str(non_actionable_path),
            "threshold_sensitivity_csv": str(sensitivity_path),
        },
        "total_candidates": int(len(blockers)),
        "production_rows": int(len(production)),
        "safe_state_core_count": int(blockers["safe_state_tier"].astype(str).eq("SAFE_STATE_CORE").sum()),
        "price_defense_candidate_count": int(edge_defendable.sum()),
        "price_defense_only_board_rows": int(len(price_defense_board)),
        "price_defense_only_board_primary_blocker_counts": {
            str(k): int(v)
            for k, v in price_defense_board_blockers.get("primary_blocker", pd.Series(dtype=str)).fillna("NONE").astype(str).value_counts().to_dict().items()
        },
        "near_core_count": int(len(near_core)),
        "near_core_candidates": near_core_candidates,
        "primary_blocker_counts": {str(k): int(v) for k, v in gap_counts.items()},
        "edge_defendable_primary_blocker_counts": {
            str(k): int(v)
            for k, v in blockers.loc[edge_defendable, "primary_blocker"].fillna("NONE").astype(str).value_counts().to_dict().items()
        },
        "threshold_sensitivity": sensitivity,
        "feature_gap_rankings": feature_rankings.to_dict(orient="records"),
        "actionable_evidence_gaps": actionable_gaps.to_dict(orient="records"),
        "non_actionable_true_instability": non_actionable_instability.to_dict(orient="records"),
        "settlement_join_fields": ["game_id", "market_date", "player", "target", "side", "line"],
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }

    _write_json(output_dir / "safe_state_evidence_gap_report.json", report)
    _write_markdown_report(output_dir / "safe_state_evidence_gap_report.md", report)
    return report


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Safe-State Evidence Gap Report",
        "",
        "## Executive Summary",
        f"- Total candidates: {report['total_candidates']}",
        f"- Production rows: {report['production_rows']}",
        f"- Price-defense candidates: {report['price_defense_candidate_count']}",
        f"- Price-defense-only board rows: {report.get('price_defense_only_board_rows', 0)}",
        f"- SAFE_STATE_CORE rows: {report['safe_state_core_count']}",
        f"- SAFE_STATE_NEAR_CORE rows: {report['near_core_count']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Primary Blockers",
    ]
    for blocker, count in report["primary_blocker_counts"].items():
        lines.append(f"- {blocker}: {count}")
    lines.extend(["", "## Near-Core Candidates"])
    if report["near_core_candidates"]:
        for row in report["near_core_candidates"]:
            lines.append(
                f"- {row.get('player', '')} {row.get('market_type', '')} {row.get('side', '')} {row.get('line', '')}: "
                f"{row.get('primary_blocker', '')} ({row.get('near_core_blocker_fixability', '')})"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Feature Gap Ranking"])
    for row in report["feature_gap_rankings"][:10]:
        lines.append(
            f"- {row.get('feature_or_signal')}: blocks {row.get('blocked_candidate_count')} candidates "
            f"({row.get('blocked_edge_defendable_count')} EDGE_DEFENDABLE); priority {row.get('priority')}"
        )
    lines.extend(["", "## Actionable Evidence Gaps"])
    if report.get("actionable_evidence_gaps"):
        for row in report["actionable_evidence_gaps"][:10]:
            lines.append(
                f"- {row.get('feature_or_signal')}: {row.get('blocker_subtype', '')} "
                f"({row.get('fixability', '')}) -> {row.get('recommended_action', '')}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Non-Actionable True Instability"])
    if report.get("non_actionable_true_instability"):
        for row in report["non_actionable_true_instability"][:10]:
            lines.append(
                f"- {row.get('feature_or_signal')}: {row.get('blocker_subtype', '')} "
                f"-> KEEP_UNSAFE_TRUE_VOLATILITY"
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Guardrails",
            "- Diagnostic only.",
            "- No production gate or threshold was changed.",
            "- No sidecar was materialized.",
            "- No promotion claim is made.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain why candidates fail SAFE_STATE_CORE.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool-csv", type=Path, required=True)
    parser.add_argument("--production-board-csv", type=Path, required=True)
    parser.add_argument("--safe-state-dir", type=Path)
    parser.add_argument("--annotated-candidates-csv", type=Path)
    parser.add_argument("--historical-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_safe_state_evidence_gap_report(
        output_dir=args.output_dir,
        candidate_pool_csv=args.candidate_pool_csv,
        production_board_csv=args.production_board_csv,
        safe_state_dir=args.safe_state_dir,
        annotated_candidates_csv=args.annotated_candidates_csv,
        historical_csv=args.historical_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
