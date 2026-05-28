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

from research.market_quality.edge_defendability import annotate_edge_defendability
from research.safe_state.player_state_forecastability import annotate_player_state_forecastability
from research.safe_state.similar_state_reliability import annotate_similar_state_reliability
from research.safe_state.structural_line_mispricing import annotate_structural_line_mispricing


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _clip01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def _coalesce(frame: pd.DataFrame, columns: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            out = out.where(out.notna(), values)
    return out.fillna(default)


def annotate_safe_state_stack(
    candidates: pd.DataFrame,
    historical_rows: pd.DataFrame | None = None,
    *,
    similar_state_min_count: int = 5,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    out = candidates.copy()
    if "edge_defendability_tier" not in out.columns:
        out = annotate_edge_defendability(out)
    if "similar_state_reliability_tier" not in out.columns:
        out = annotate_similar_state_reliability(out, historical_rows, min_count=int(similar_state_min_count))
    out = annotate_player_state_forecastability(out)
    out = annotate_structural_line_mispricing(out)
    return annotate_safe_state(out)


def annotate_safe_state(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    edge_tier = _text(out, "edge_defendability_tier").str.upper().str.strip()
    price_status = _text(out, "price_validity_status").str.upper().str.strip()
    forecast_tier = _text(out, "forecastability_tier").str.upper().str.strip()
    similar_tier = _text(out, "similar_state_reliability_tier").str.upper().str.strip()
    structural_tier = _text(out, "structural_mispricing_tier").str.upper().str.strip()
    forecast_gap_severity = _text(out, "forecastability_gap_severity", "NONE").str.upper().str.strip()
    forecast_gap_fixability = _text(out, "forecastability_gap_fixability", "UNKNOWN").str.upper().str.strip()
    stress_edge = _numeric(out, "stress_edge", np.nan)
    lcb_edge = _numeric(out, "lcb_edge", np.nan)
    forecast_score = _clip01(_coalesce(out, ["overall_player_forecastability_score", "forecastability_score"], default=0.0))
    tightness = _clip01(_numeric(out, "similar_state_tightness_score", 0.0))
    structural_score = _clip01(_numeric(out, "overall_structural_mispricing_score", 0.0))
    chaos = _clip01(_numeric(out, "chaos_score", 0.50))
    scenario = _clip01(_numeric(out, "scenario_agreement", 0.50))
    board_risk = _clip01(_numeric(out, "board_readiness_risk_score", 0.0))
    failure_modes = (
        _text(out, "known_failure_modes")
        + ";"
        + _text(out, "forecastability_failure_modes")
        + ";"
        + _text(out, "failure_modes")
    ).str.upper()

    price_clear = edge_tier.eq("EDGE_DEFENDABLE") & price_status.eq("PRICE_VALID") & stress_edge.gt(0.0) & lcb_edge.gt(0.0)
    forecast_ok = forecast_tier.isin({"HIGH_FORECASTABILITY", "MEDIUM_FORECASTABILITY"})
    forecast_gap_ok = ~forecast_gap_severity.isin({"HIGH", "CRITICAL"})
    similar_ok = similar_tier.isin({"TIGHT", "ACCEPTABLE"})
    structural_ok = structural_tier.isin({"STRUCTURAL_MISPRICE_STRONG", "STRUCTURAL_MISPRICE_ACCEPTABLE"})
    severe_failure = failure_modes.str.contains("INJURY|NEWS_UNRESOLVED|PLAYER_STATE_UNFORECASTABLE|DATA_MISSING|MARKET_PRICE_FAILURE", regex=True)
    chaos_ok = chaos.le(0.35)
    scenario_ok = scenario.ge(0.65)

    safe_core = price_clear & forecast_ok & forecast_gap_ok & similar_ok & structural_ok & ~severe_failure & chaos_ok & scenario_ok
    safe_state_score = (
        0.30 * _clip01((lcb_edge.fillna(-0.10) + 0.08) / 0.16)
        + 0.22 * forecast_score
        + 0.18 * structural_score
        + 0.15 * tightness
        + 0.10 * scenario
        + 0.05 * (1.0 - board_risk)
        - 0.10 * chaos
    ).clip(lower=0.0, upper=1.0)

    blocker_frame = pd.DataFrame(
        {
            "forecastability": ~(forecast_ok & forecast_gap_ok),
            "similar_state": ~similar_ok,
            "structural": ~structural_ok,
            "scenario": ~(chaos_ok & scenario_ok),
            "failure_mode": severe_failure,
        },
        index=out.index,
    )
    major_blocker_count = blocker_frame.sum(axis=1)
    fixable_near_core_blocker = forecast_gap_fixability.isin(
        {"FIXABLE_WITH_EXISTING_LOGS", "FIXABLE_WITH_NEW_PIPELINE_DATA", "NEEDS_MORE_SAMPLE"}
    )
    near_core = price_clear & major_blocker_count.eq(1) & fixable_near_core_blocker & ~forecast_gap_fixability.eq("TRUE_UNSTABLE_STATE")

    gap_primary = _text(out, "forecastability_gap_primary", "").str.upper().str.strip()
    inferred_unstable_without_gap_detail = (
        forecast_tier.isin({"LOW_FORECASTABILITY", "UNFORECASTABLE"})
        & forecast_gap_fixability.eq("UNKNOWN")
        & gap_primary.eq("")
    )

    tier = pd.Series("SAFE_STATE_REJECT", index=out.index, dtype="object")
    tier = tier.mask(edge_tier.eq("EDGE_DEFENDABLE") & price_clear, "SAFE_STATE_PRICE_ONLY")
    tier = tier.mask(
        price_clear
        & (forecast_gap_fixability.eq("TRUE_UNSTABLE_STATE") | forecast_gap_severity.eq("CRITICAL") | inferred_unstable_without_gap_detail),
        "SAFE_STATE_UNSTABLE",
    )
    tier = tier.mask(price_clear & ~(forecast_ok & forecast_gap_ok), "SAFE_STATE_INSUFFICIENT_EVIDENCE")
    tier = tier.mask(
        price_clear
        & (forecast_gap_fixability.eq("TRUE_UNSTABLE_STATE") | forecast_gap_severity.eq("CRITICAL") | inferred_unstable_without_gap_detail),
        "SAFE_STATE_UNSTABLE",
    )
    tier = tier.mask(price_clear & forecast_ok & forecast_gap_ok & ~structural_ok, "SAFE_STATE_STRUCTURALLY_WEAK")
    insufficient = (
        price_clear
        & forecast_ok
        & forecast_gap_ok
        & structural_ok
        & ~similar_ok
    ) | similar_tier.isin({"INSUFFICIENT_SAMPLE", ""})
    tier = tier.mask(insufficient, "SAFE_STATE_INSUFFICIENT_EVIDENCE")
    tier = tier.mask(near_core, "SAFE_STATE_NEAR_CORE")
    tier = tier.mask(safe_core, "SAFE_STATE_CORE")
    tier = tier.mask(~price_clear, "SAFE_STATE_REJECT")

    expanded = price_clear & tier.eq("SAFE_STATE_PRICE_ONLY") & similar_tier.eq("TIGHT") & forecast_score.ge(0.70)
    tier = tier.mask(expanded, "SAFE_STATE_PRICE_ONLY")

    reasons: list[str] = []
    blockers: list[str] = []
    success_paths: list[str] = []
    failure_paths: list[str] = []
    why_price: list[str] = []
    why_forecast: list[str] = []
    why_structural: list[str] = []
    similar_summary: list[str] = []
    rejection: list[str] = []

    for idx in out.index:
        row_reasons: list[str] = []
        row_blockers: list[str] = []
        if price_clear.loc[idx]:
            row_reasons.append(f"price_defendable_lcb_edge={lcb_edge.loc[idx]:.3f}")
            why_price.append(f"timestamp-safe price clears stress and LCB edge; lcb_edge={lcb_edge.loc[idx]:.3f}")
        else:
            row_blockers.append(str(out.at[idx, "edge_defendability_reason"]) if "edge_defendability_reason" in out.columns else "price_or_lcb_failed")
            why_price.append("price defense failed or LCB edge did not clear")

        forecast_ready = bool(forecast_ok.loc[idx] and forecast_gap_ok.loc[idx])
        if forecast_ready:
            row_reasons.append(f"forecastability={forecast_tier.loc[idx]}")
            why_forecast.append(str(out.at[idx, "forecastability_reasons"]) if "forecastability_reasons" in out.columns else forecast_tier.loc[idx])
        else:
            gap_label = str(out.at[idx, "forecastability_gap_primary"]) if "forecastability_gap_primary" in out.columns else forecast_tier.loc[idx]
            row_blockers.append(f"forecastability={gap_label or forecast_tier.loc[idx] or 'missing'}")
            why_forecast.append("player state is unstable or missing role/minutes context")

        if structural_ok.loc[idx]:
            row_reasons.append(f"structural={structural_tier.loc[idx]}")
            why_structural.append(str(out.at[idx, "structural_mispricing_reasons"]) if "structural_mispricing_reasons" in out.columns else structural_tier.loc[idx])
        else:
            row_blockers.append(f"structural={structural_tier.loc[idx] or 'missing'}")
            why_structural.append("line lacks structural mispricing support beyond price edge")

        if similar_ok.loc[idx]:
            row_reasons.append(f"similar_states={similar_tier.loc[idx]}")
        else:
            row_blockers.append(f"similar_states={similar_tier.loc[idx] or 'missing'}")
        similar_summary.append(
            f"{int(_numeric(out.loc[[idx]], 'similar_state_count', 0).iloc[0])} comparable states; "
            f"tier={similar_tier.loc[idx] or 'missing'}; tightness={tightness.loc[idx]:.2f}"
        )
        if severe_failure.loc[idx]:
            row_blockers.append("major_active_failure_mode")

        success_paths.append(
            "price clears, player state holds, structural line gap materializes"
            if row_reasons
            else "no clear success path"
        )
        failure_paths.append(
            "price/LCB failure"
            if not price_clear.loc[idx]
            else "state volatility or structural thesis failure"
        )
        reasons.append(";".join(row_reasons) if row_reasons else "no_safe_state_confirmation")
        blockers.append(";".join(row_blockers))
        rejection.append(";".join(row_blockers) if tier.loc[idx] != "SAFE_STATE_CORE" else "")

    validation_status = pd.Series("SHADOW_ONLY", index=out.index, dtype="object")
    validation_status = validation_status.mask(tier.eq("SAFE_STATE_INSUFFICIENT_EVIDENCE"), "REJECT_INSUFFICIENT_EVIDENCE")
    validation_status = validation_status.mask(tier.eq("SAFE_STATE_REJECT"), "REJECT_INSUFFICIENT_EVIDENCE")

    out["safe_state_tier"] = tier
    out["safe_state_score"] = safe_state_score
    out["safe_state_reasons"] = reasons
    out["safe_state_blockers"] = blockers
    out["safe_state_success_path"] = success_paths
    out["safe_state_primary_failure_path"] = failure_paths
    out["safe_state_validation_status"] = validation_status
    out["why_price_defendable"] = why_price
    out["why_forecastable"] = why_forecast
    out["why_structurally_mispriced"] = why_structural
    out["similar_state_summary"] = similar_summary
    out["primary_success_path"] = success_paths
    out["primary_failure_path"] = failure_paths
    out["rejection_reason_if_not_safe"] = rejection
    out["safe_state_explanation"] = [
        (
            f"Accepted as {tier_value} because {reason_value}."
            if tier_value == "SAFE_STATE_CORE"
            else f"Classified as {tier_value}: {blocker_value or reason_value}."
        )
        for tier_value, reason_value, blocker_value in zip(tier.tolist(), reasons, blockers)
    ]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate candidate rows with shadow safe-state tiers.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--historical-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    history = pd.read_csv(args.historical_csv) if args.historical_csv and args.historical_csv.exists() else pd.DataFrame()
    annotated = annotate_safe_state_stack(candidates, history)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_csv, index=False)
    if args.summary_json:
        payload: dict[str, Any] = {
            "rows": int(len(annotated)),
            "safe_state_tier_counts": annotated["safe_state_tier"].value_counts(dropna=False).to_dict(),
            "shadow_only": True,
            "production_behavior_changed": False,
            "promotion_claim": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
