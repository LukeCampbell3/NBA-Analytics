from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .uncertainty import (
    BELIEF_UNCERTAINTY_LOWER,
    BELIEF_UNCERTAINTY_UPPER,
    normalize_belief_uncertainty,
)


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _string_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(default).astype(bool)
        text = series.fillna(default).astype(str).str.strip().str.lower()
        return text.isin({"1", "true", "t", "yes", "y"})
    return pd.Series(default, index=frame.index, dtype=bool)


def _build_game_key(frame: pd.DataFrame) -> pd.Series:
    existing = _string_series(frame, "game_key").str.strip()
    event = _string_series(frame, "market_event_id").str.strip()
    home = _string_series(frame, "market_home_team").str.strip()
    away = _string_series(frame, "market_away_team").str.strip()
    teams = (home + "::" + away).str.strip(":")
    fallback = event.where(event != "", teams)
    return existing.where(existing != "", fallback)


def _normalize_script_cluster(values: pd.Series) -> pd.Series:
    tokens = values.fillna("").astype(str).str.strip()
    return tokens.where(~tokens.str.lower().isin({"", "script=unknown"}), "")


def _risk_status(score: float) -> str:
    numeric = float(np.clip(score, 0.0, 1.0))
    if numeric >= 0.65:
        return "BLOCKER"
    if numeric >= 0.45:
        return "FRAGILE"
    if numeric >= 0.25:
        return "MONITOR"
    return "STABLE"


def _join_reasons(reason_frame: pd.DataFrame) -> pd.Series:
    ordered = list(reason_frame.columns)

    def _collapse(row: pd.Series) -> str:
        reasons = [name for name in ordered if bool(row.get(name, False))]
        return "|".join(reasons)

    return reason_frame.apply(_collapse, axis=1)


def _price_components(frame: pd.DataFrame) -> dict[str, pd.Series]:
    validity = _string_series(frame, "price_validity_status").str.upper().str.strip()
    source_type = _string_series(frame, "price_source_type").str.upper().str.strip()
    timestamp_safe = _bool_series(frame, "timestamp_safe_flag", default=False)
    diagnostic_flag = _bool_series(frame, "diagnostic_only_flag", default=False)
    edge_untrusted = _bool_series(frame, "edge_price_untrusted_flag", default=False)
    market_side_price = pd.to_numeric(frame.get("market_side_price"), errors="coerce") if "market_side_price" in frame.columns else pd.Series(np.nan, index=frame.index, dtype="float64")
    market_side_break_even = pd.to_numeric(frame.get("market_side_break_even"), errors="coerce") if "market_side_break_even" in frame.columns else pd.Series(np.nan, index=frame.index, dtype="float64")

    explicit_valid = validity.eq("PRICE_VALID") & timestamp_safe
    missing = validity.eq("MISSING_PRICE") | market_side_price.isna() | market_side_break_even.isna()
    invalid = validity.eq("INVALID_PRICE")
    stale = validity.eq("STALE_PRICE")
    diagnostic = (
        diagnostic_flag
        | validity.str.startswith("DIAGNOSTIC_ONLY")
        | source_type.eq("CLOSE_ONLY_DIAGNOSTIC")
    )
    unknown = validity.eq("PRICE_SOURCE_UNKNOWN") | source_type.eq("UNKNOWN") | (validity.eq("") & ~explicit_valid & ~missing)
    source_unknown = unknown | validity.eq("")
    untrusted = edge_untrusted | ~explicit_valid

    component = pd.Series(0.0, index=frame.index, dtype="float64")
    component = component.where(explicit_valid, 0.90)
    component = component.mask(missing, 0.85)
    component = component.mask(unknown, 0.90)
    component = component.mask(invalid | stale, 1.00)
    component = component.mask(diagnostic, 1.00)
    component = component.mask(edge_untrusted & explicit_valid, 0.40)

    return {
        "price_validity_status": validity,
        "price_source_type": source_type,
        "timestamp_safe_flag": timestamp_safe,
        "diagnostic_only_flag": diagnostic,
        "price_explicit_valid": explicit_valid,
        "price_missing_flag": missing,
        "price_invalid_flag": invalid,
        "price_stale_flag": stale,
        "price_unknown_flag": source_unknown,
        "price_untrusted_flag": untrusted,
        "price_component": component.clip(lower=0.0, upper=1.0),
    }


def annotate_board_readiness(
    frame: pd.DataFrame,
    *,
    belief_uncertainty_lower: float = BELIEF_UNCERTAINTY_LOWER,
    belief_uncertainty_upper: float = BELIEF_UNCERTAINTY_UPPER,
    low_quality_threshold: float = 0.58,
    low_recency_threshold: float = 0.70,
    high_uncertainty_threshold: float = 0.65,
    fragility_threshold: float = 0.55,
    instability_threshold: float = 0.45,
    same_game_share_threshold: float = 0.50,
    same_script_share_threshold: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        summary = {
            "row_count": 0,
            "board_readiness_score": 0.0,
            "board_readiness_status": "EMPTY",
            "production_readiness_clear": False,
            "blocked_reasons": ["empty_board"],
            "recommended_action": "no_board_rows",
        }
        return frame.copy(), summary

    out = frame.copy()
    row_count = max(1, int(len(out)))

    out["board_readiness_game_key"] = _build_game_key(out).fillna("").astype(str).str.strip()
    out["board_readiness_script_cluster"] = _normalize_script_cluster(_string_series(out, "script_cluster_id"))

    uncertainty_raw = _numeric_series(out, "belief_uncertainty", 1.0).clip(lower=0.0)
    if "belief_uncertainty_normalized" in out.columns:
        uncertainty = _numeric_series(out, "belief_uncertainty_normalized", np.nan)
        uncertainty = uncertainty.fillna(
            normalize_belief_uncertainty(
                uncertainty_raw,
                default=1.0,
                lower=float(belief_uncertainty_lower),
                upper=float(belief_uncertainty_upper),
            )
        ).clip(lower=0.0, upper=1.0)
    else:
        uncertainty = normalize_belief_uncertainty(
            uncertainty_raw,
            default=1.0,
            lower=float(belief_uncertainty_lower),
            upper=float(belief_uncertainty_upper),
        ).clip(lower=0.0, upper=1.0)

    fragility = _numeric_series(out, "line_decision_fragility_score", 0.0).clip(lower=0.0, upper=1.0)
    instability = _numeric_series(out, "line_decision_instability_score", 0.0).clip(lower=0.0, upper=1.0)
    quality = _numeric_series(out, "final_pool_quality_score", 0.50).clip(lower=0.0, upper=1.0)
    recency = _numeric_series(out, "recency_factor", 1.0).clip(lower=0.0, upper=1.0)
    noise = _numeric_series(out, "noise_score", 0.0).clip(lower=0.0, upper=1.0)
    contradiction = _numeric_series(out, "contradiction_score", 0.0).clip(lower=0.0, upper=1.0)
    gate_keep_prob = _numeric_series(out, "accepted_pick_gate_keep_prob", 1.0).clip(lower=0.0, upper=1.0)

    game_counts = out["board_readiness_game_key"].replace("", np.nan).value_counts(dropna=True)
    game_count_series = out["board_readiness_game_key"].map(game_counts).fillna(1.0)
    same_game_share = (game_count_series / float(row_count)).clip(lower=0.0, upper=1.0)

    cluster_counts = out["board_readiness_script_cluster"].replace("", np.nan).value_counts(dropna=True)
    cluster_count_series = out["board_readiness_script_cluster"].map(cluster_counts).fillna(1.0)
    same_script_share = (cluster_count_series / float(row_count)).clip(lower=0.0, upper=1.0)

    price_parts = _price_components(out)
    for column, series in price_parts.items():
        out[f"board_readiness_{column}"] = series

    quality_component = ((float(low_quality_threshold) - quality) / max(float(low_quality_threshold), 1e-9)).clip(lower=0.0, upper=1.0)
    recency_component = ((float(low_recency_threshold) - recency) / max(float(low_recency_threshold), 1e-9)).clip(lower=0.0, upper=1.0)
    gate_component = ((0.55 - gate_keep_prob) / 0.55).clip(lower=0.0, upper=1.0)
    same_game_component = ((same_game_share - 0.25) / 0.75).clip(lower=0.0, upper=1.0)
    same_script_component = ((same_script_share - 0.25) / 0.75).clip(lower=0.0, upper=1.0)
    dependency_component = pd.concat([same_game_component, same_script_component], axis=1).max(axis=1)

    out["board_readiness_uncertainty_component"] = uncertainty
    out["board_readiness_fragility_component"] = fragility
    out["board_readiness_instability_component"] = instability
    out["board_readiness_quality_component"] = quality_component
    out["board_readiness_recency_component"] = recency_component
    out["board_readiness_noise_component"] = noise
    out["board_readiness_contradiction_component"] = contradiction
    out["board_readiness_gate_component"] = gate_component
    out["board_readiness_same_game_count"] = game_count_series.astype("int64")
    out["board_readiness_same_game_share"] = same_game_share
    out["board_readiness_same_script_count"] = cluster_count_series.astype("int64")
    out["board_readiness_same_script_share"] = same_script_share
    out["board_readiness_dependency_component"] = dependency_component

    out["board_readiness_high_uncertainty_flag"] = uncertainty >= float(high_uncertainty_threshold)
    out["board_readiness_line_fragility_flag"] = fragility >= float(fragility_threshold)
    out["board_readiness_line_instability_flag"] = instability >= float(instability_threshold)
    out["board_readiness_low_quality_flag"] = quality < float(low_quality_threshold)
    out["board_readiness_low_recency_flag"] = recency < float(low_recency_threshold)
    out["board_readiness_noise_flag"] = noise >= 0.35
    out["board_readiness_contradiction_flag"] = contradiction >= 0.30
    out["board_readiness_gate_risk_flag"] = gate_keep_prob < 0.55
    out["board_readiness_same_game_concentration_flag"] = (game_count_series > 1.0) & (same_game_share >= float(same_game_share_threshold))
    out["board_readiness_same_script_concentration_flag"] = (
        out["board_readiness_script_cluster"].ne("")
        & (cluster_count_series > 1.0)
        & (same_script_share >= float(same_script_share_threshold))
    )
    out["board_readiness_price_untrusted_flag"] = price_parts["price_untrusted_flag"]
    out["board_readiness_stale_price_dependency_candidate_flag"] = (
        price_parts["price_stale_flag"]
        | price_parts["price_missing_flag"]
        | price_parts["price_invalid_flag"]
        | price_parts["price_unknown_flag"]
    )

    out["board_readiness_risk_score"] = (
        0.15 * out["board_readiness_uncertainty_component"]
        + 0.10 * out["board_readiness_fragility_component"]
        + 0.08 * out["board_readiness_instability_component"]
        + 0.12 * out["board_readiness_quality_component"]
        + 0.08 * out["board_readiness_recency_component"]
        + 0.08 * out["board_readiness_noise_component"]
        + 0.06 * out["board_readiness_contradiction_component"]
        + 0.06 * out["board_readiness_gate_component"]
        + 0.17 * out["board_readiness_price_component"]
        + 0.10 * out["board_readiness_dependency_component"]
    ).clip(lower=0.0, upper=1.0)

    reason_frame = pd.DataFrame(
        {
            "price_untrusted": out["board_readiness_price_untrusted_flag"],
            "high_uncertainty": out["board_readiness_high_uncertainty_flag"],
            "line_fragility": out["board_readiness_line_fragility_flag"],
            "line_instability": out["board_readiness_line_instability_flag"],
            "low_quality": out["board_readiness_low_quality_flag"],
            "low_recency": out["board_readiness_low_recency_flag"],
            "noise": out["board_readiness_noise_flag"],
            "contradiction": out["board_readiness_contradiction_flag"],
            "gate_risk": out["board_readiness_gate_risk_flag"],
            "same_game_concentration": out["board_readiness_same_game_concentration_flag"],
            "same_script_concentration": out["board_readiness_same_script_concentration_flag"],
        },
        index=out.index,
    )
    out["board_readiness_reasons"] = _join_reasons(reason_frame)
    out["board_readiness_warning_count"] = reason_frame.sum(axis=1).astype("int64")
    out["board_readiness_status"] = out["board_readiness_risk_score"].map(_risk_status)
    out["board_readiness_review_required"] = out["board_readiness_status"].isin(["FRAGILE", "BLOCKER"])

    repeated_game_mask = game_count_series > 1.0
    repeated_script_mask = out["board_readiness_script_cluster"].ne("") & (cluster_count_series > 1.0)
    same_game_max_share = float(same_game_share.loc[repeated_game_mask].max()) if repeated_game_mask.any() else 0.0
    same_script_max_share = float(same_script_share.loc[repeated_script_mask].max()) if repeated_script_mask.any() else 0.0
    mean_risk = float(pd.to_numeric(out["board_readiness_risk_score"], errors="coerce").fillna(0.0).mean())
    board_score = float(
        np.clip(
            mean_risk
            + 0.10 * max(0.0, same_game_max_share - 0.35)
            + 0.08 * max(0.0, same_script_max_share - 0.35),
            0.0,
            1.0,
        )
    )

    blocked_reasons: list[str] = []
    if int(price_parts["price_explicit_valid"].sum()) < row_count:
        blocked_reasons.append("timestamp_safe_price_evidence_incomplete")
    if repeated_game_mask.any() and same_game_max_share >= float(same_game_share_threshold):
        blocked_reasons.append("same_game_concentration_elevated")
    if repeated_script_mask.any() and same_script_max_share >= float(same_script_share_threshold):
        blocked_reasons.append("script_cluster_concentration_elevated")
    if float(out["board_readiness_high_uncertainty_flag"].mean()) >= 0.34:
        blocked_reasons.append("belief_uncertainty_elevated")
    if float(out["board_readiness_low_quality_flag"].mean()) >= 0.34:
        blocked_reasons.append("final_pool_quality_risk_elevated")
    if float((out["board_readiness_line_fragility_flag"] | out["board_readiness_line_instability_flag"]).mean()) >= 0.34:
        blocked_reasons.append("line_decision_fragility_elevated")
    if float(out["board_readiness_low_recency_flag"].mean()) >= 0.25:
        blocked_reasons.append("recency_support_thin")

    summary = {
        "row_count": int(row_count),
        "board_readiness_score": board_score,
        "board_readiness_status": "BLOCKER" if blocked_reasons and _risk_status(board_score) == "FRAGILE" else _risk_status(max(board_score, 0.65 if blocked_reasons else board_score)),
        "production_readiness_clear": bool(not blocked_reasons and board_score < 0.45),
        "blocked_reasons": blocked_reasons,
        "recommended_action": "shadow_only_review" if blocked_reasons else ("monitor" if board_score >= 0.25 else "auditable"),
        "high_uncertainty_rows": int(out["board_readiness_high_uncertainty_flag"].sum()),
        "line_fragility_rows": int(out["board_readiness_line_fragility_flag"].sum()),
        "line_instability_rows": int(out["board_readiness_line_instability_flag"].sum()),
        "low_quality_rows": int(out["board_readiness_low_quality_flag"].sum()),
        "low_recency_rows": int(out["board_readiness_low_recency_flag"].sum()),
        "price_untrusted_rows": int(out["board_readiness_price_untrusted_flag"].sum()),
        "timestamp_safe_price_rows": int(price_parts["price_explicit_valid"].sum()),
        "missing_price_rows": int(price_parts["price_missing_flag"].sum()),
        "invalid_price_rows": int(price_parts["price_invalid_flag"].sum()),
        "stale_price_rows": int(price_parts["price_stale_flag"].sum()),
        "diagnostic_only_price_rows": int(price_parts["diagnostic_only_flag"].sum()),
        "unknown_price_rows": int(price_parts["price_unknown_flag"].sum()),
        "same_game_max_share": same_game_max_share,
        "same_script_cluster_max_share": same_script_max_share,
        "fragile_or_blocker_rows": int(out["board_readiness_review_required"].sum()),
        "status_counts": {
            str(key): int(value)
            for key, value in out["board_readiness_status"].value_counts(dropna=False).to_dict().items()
        },
    }
    return out, summary
