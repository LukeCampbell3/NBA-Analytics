from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _target(frame: pd.DataFrame) -> pd.Series:
    if "target" in frame.columns:
        out = frame["target"].fillna("").astype(str).str.upper().str.strip()
    else:
        out = pd.Series("", index=frame.index, dtype="object")
    market = _text(frame, "market_type").str.upper()
    for target in ["PTS", "TRB", "AST", "PRA", "PR", "PA", "RA", "3PM"]:
        out = out.mask(out.eq("") & market.str.contains(target), target)
    return out


def _side(frame: pd.DataFrame) -> pd.Series:
    side = _text(frame, "side").str.upper().str.strip()
    side = side.mask(side.eq(""), _text(frame, "direction").str.upper().str.strip())
    market = _text(frame, "market_type").str.upper()
    side = side.mask(side.eq("") & market.str.endswith("_OVER"), "OVER")
    side = side.mask(side.eq("") & market.str.endswith("_UNDER"), "UNDER")
    return side


def _quantile(frame: pd.DataFrame, q: str) -> pd.Series:
    candidates = [
        q,
        f"prediction_{q}",
        f"model_{q}",
        f"target_{q}",
        f"recent_{q}",
        f"trb_{q}_recent",
        f"pts_{q}_recent",
        f"ast_{q}_recent",
    ]
    return _coalesce(frame, candidates, default=np.nan)


def _line_zone(line: pd.Series, q25: pd.Series, q50: pd.Series, q75: pd.Series, q90: pd.Series) -> tuple[pd.Series, pd.Series]:
    zone = pd.Series("UNKNOWN", index=line.index, dtype="object")
    percentile = pd.Series(np.nan, index=line.index, dtype="float64")
    known = line.notna()
    zone = zone.mask(known & q25.notna() & line.lt(q25), "BELOW_Q25")
    percentile = percentile.mask(known & q25.notna() & line.lt(q25), 0.20)
    near_median = known & q50.notna() & (line - q50).abs().le(1.0)
    zone = zone.mask(near_median, "NEAR_MEDIAN")
    percentile = percentile.mask(near_median, 0.50)
    above_q75 = known & q75.notna() & line.gt(q75)
    zone = zone.mask(above_q75, "ABOVE_Q75")
    percentile = percentile.mask(above_q75, 0.80)
    extreme = known & q90.notna() & line.gt(q90)
    zone = zone.mask(extreme, "EXTREME_TAIL")
    percentile = percentile.mask(extreme, 0.95)
    mid_known = known & percentile.isna() & (q25.notna() | q50.notna() | q75.notna())
    zone = zone.mask(mid_known, "NEAR_MEDIAN")
    percentile = percentile.mask(mid_known, 0.50)
    return zone, percentile


def _structural_pathway_score(frame: pd.DataFrame, target: pd.Series) -> pd.Series:
    explicit = _numeric(frame, "structural_pathway_score", np.nan)
    forecastability = _clip01(_coalesce(frame, ["overall_player_forecastability_score", "forecastability_score"], default=0.55))
    scenario = _clip01(_numeric(frame, "scenario_agreement", 0.55).fillna(0.55))
    pts_path = _clip01(
        0.35 * _coalesce(frame, ["usage_forecastability_score", "usage_stability_score"], default=0.55)
        + 0.25 * (1.0 - _clip01(_coalesce(frame, ["fga_volatility", "usage_volatility"], default=0.35)))
        + 0.20 * _coalesce(frame, ["free_throw_path_score", "rim_pressure_score"], default=0.50)
        + 0.20 * scenario
    )
    ast_path = _clip01(
        0.35 * _coalesce(frame, ["team_assist_environment_score", "projected_assist_conversion_proxy"], default=0.50)
        + 0.25 * (1.0 - _clip01(_coalesce(frame, ["assist_opportunity_volatility", "teammate_shooting_volatility"], default=0.35)))
        + 0.20 * _coalesce(frame, ["passing_role_stability_score", "usage_forecastability_score"], default=0.55)
        + 0.20 * scenario
    )
    trb_path = _clip01(
        0.35 * _coalesce(frame, ["rebound_supply_score"], default=0.50)
        + 0.30 * _coalesce(frame, ["rebound_share_stability_score"], default=0.50)
        + 0.20 * (1.0 - _clip01(_coalesce(frame, ["teammate_rebound_competition_score", "wing_rebound_leakage_score"], default=0.35)))
        + 0.15 * scenario
    )
    combo_path = _clip01(0.50 * forecastability + 0.25 * pts_path + 0.25 * ast_path)
    path = pd.Series(0.50, index=frame.index, dtype="float64")
    path = path.mask(target.eq("PTS") | target.eq("3PM"), pts_path)
    path = path.mask(target.eq("AST"), ast_path)
    path = path.mask(target.eq("TRB"), trb_path)
    path = path.mask(target.isin({"PRA", "PR", "PA", "RA"}), combo_path)
    return explicit.where(explicit.notna(), path.clip(lower=0.0, upper=1.0))


def annotate_structural_line_mispricing(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    target = _target(out)
    side = _side(out)
    line = _coalesce(out, ["line", "market_line"], default=np.nan)
    model_mean = _coalesce(out, ["model_mean", "raw_prediction", "prediction", "projected_stat", "expected_stat"], default=np.nan)
    uncertainty = _coalesce(out, ["uncertainty_sigma", "sigma", "model_sigma"], default=3.0).fillna(3.0).clip(lower=0.25)
    conservative_mean = model_mean - (0.25 * uncertainty)

    q25 = _quantile(out, "q25")
    q50 = _coalesce(out, ["q50", "median", "recent_median", "trb_median_recent", "pts_median_recent", "ast_median_recent"], default=np.nan)
    q75 = _quantile(out, "q75")
    q90 = _quantile(out, "q90")
    zone, inferred_percentile = _line_zone(line, q25, q50, q75, q90)
    line_percentile = _numeric(out, "line_percentile", np.nan).where(_numeric(out, "line_percentile", np.nan).notna(), inferred_percentile)

    gap_over = conservative_mean - line
    gap_under = line - (model_mean + (0.25 * uncertainty))
    state_adjusted_gap = pd.Series(np.nan, index=out.index, dtype="float64")
    state_adjusted_gap = state_adjusted_gap.mask(side.eq("OVER"), gap_over)
    state_adjusted_gap = state_adjusted_gap.mask(side.eq("UNDER"), gap_under)
    gap_score = (state_adjusted_gap / (uncertainty + 1.0)).clip(lower=0.0, upper=1.0)

    zone_support = pd.Series(0.20, index=out.index, dtype="float64")
    zone_support = zone_support.mask(side.eq("OVER") & zone.eq("BELOW_Q25"), 0.85)
    zone_support = zone_support.mask(side.eq("OVER") & zone.eq("NEAR_MEDIAN") & gap_score.gt(0.35), 0.60)
    zone_support = zone_support.mask(side.eq("UNDER") & zone.isin({"ABOVE_Q75", "EXTREME_TAIL"}), 0.85)
    zone_support = zone_support.mask(side.eq("UNDER") & zone.eq("NEAR_MEDIAN") & gap_score.gt(0.35), 0.60)
    line_misplacement_score = (0.55 * zone_support + 0.45 * gap_score).clip(lower=0.0, upper=1.0)

    structural_side_agreement = (
        ((side.eq("OVER")) & state_adjusted_gap.gt(0.0))
        | ((side.eq("UNDER")) & state_adjusted_gap.gt(0.0))
    ).astype(int)

    pathway_score = _structural_pathway_score(out, target)
    role_mismatch = _clip01(
        _coalesce(
            out,
            [
                "market_role_mismatch_score",
                "stale_role_pricing_score",
                "teammate_out_usage_gain_score",
                "minutes_role_change_score",
            ],
            default=np.nan,
        )
    )
    role_mismatch = role_mismatch.where(role_mismatch.gt(0.0), 0.20 * _clip01(_coalesce(out, ["line_decision_instability_score"], default=0.0)))
    opposite_score = _clip01(_coalesce(out, ["opposite_side_discovery_score"], default=0.0))
    alt_score = _clip01(_coalesce(out, ["alt_line_better_framing_score"], default=0.0))
    opposite_or_alt = pd.concat([opposite_score, alt_score], axis=1).max(axis=1)
    similar_score = _clip01(
        _coalesce(out, ["similar_state_line_error_score"], default=np.nan).where(
            _coalesce(out, ["similar_state_line_error_score"], default=np.nan).notna(),
            _clip01(_numeric(out, "similar_state_win_rate", 0.50).fillna(0.50) - 0.50) * 2.0,
        )
    )
    forecast_support = _clip01(_coalesce(out, ["overall_player_forecastability_score", "forecastability_score"], default=0.50))
    state_adjusted_line_gap_score = gap_score

    overall = (
        0.20 * line_misplacement_score
        + 0.20 * state_adjusted_line_gap_score
        + 0.15 * pathway_score
        + 0.15 * role_mismatch
        + 0.15 * similar_score
        + 0.10 * opposite_or_alt
        + 0.05 * forecast_support
    ).clip(lower=0.0, upper=1.0)

    price_edge_positive = _numeric(out, "lcb_edge", np.nan).fillna(_numeric(out, "stress_edge", np.nan)).gt(0.0)
    structural_inputs_missing = (
        line.isna()
        | (
            model_mean.isna()
            & q25.isna()
            & q50.isna()
            & q75.isna()
            & _numeric(out, "similar_state_count", 0.0).fillna(0.0).lt(1)
            & pathway_score.le(0.50)
        )
    )
    tier = pd.Series("NO_STRUCTURAL_EDGE", index=out.index, dtype="object")
    tier = tier.mask(price_edge_positive, "PRICE_ONLY_EDGE")
    tier = tier.mask(overall.ge(0.58) & state_adjusted_gap.gt(0.0) & pathway_score.ge(0.50), "STRUCTURAL_MISPRICE_ACCEPTABLE")
    tier = tier.mask(overall.ge(0.72) & state_adjusted_gap.gt(0.0) & pathway_score.ge(0.55), "STRUCTURAL_MISPRICE_STRONG")
    tier = tier.mask(structural_inputs_missing, "UNKNOWN")

    reasons: list[str] = []
    for idx in out.index:
        row_reasons: list[str] = []
        if state_adjusted_gap.loc[idx] > 0:
            row_reasons.append(f"state_adjusted_line_gap={state_adjusted_gap.loc[idx]:.2f}")
        if line_misplacement_score.loc[idx] >= 0.65:
            row_reasons.append(f"line_zone={zone.loc[idx]}")
        if pathway_score.loc[idx] >= 0.60:
            row_reasons.append("basketball_pathway_supported")
        if role_mismatch.loc[idx] >= 0.55:
            row_reasons.append("market_role_mismatch_signal")
        if similar_score.loc[idx] >= 0.55:
            row_reasons.append("similar_states_favor_side")
        if opposite_or_alt.loc[idx] >= 0.55:
            row_reasons.append("opposite_or_alt_line_framing_signal")
        if not row_reasons:
            row_reasons.append("price_edge_without_structural_confirmation" if price_edge_positive.loc[idx] else "no_structural_support")
        reasons.append(";".join(row_reasons))

    out["line_zone"] = zone
    out["line_percentile"] = line_percentile
    out["line_misplacement_score"] = line_misplacement_score
    out["state_adjusted_line_gap"] = state_adjusted_gap
    out["state_adjusted_line_gap_score"] = state_adjusted_line_gap_score
    out["structural_side_agreement"] = structural_side_agreement
    out["structural_pathway_score"] = pathway_score
    out["market_role_mismatch_score"] = role_mismatch
    out["opposite_side_discovery_score"] = opposite_score
    out["alt_line_better_framing_score"] = alt_score
    out["similar_state_line_error_score"] = similar_score
    out["overall_structural_mispricing_score"] = overall
    out["structural_mispricing_tier"] = tier
    out["structural_mispricing_reasons"] = reasons
    out["structural_mispricing_blocks_safe_state_flag"] = ~tier.isin({"STRUCTURAL_MISPRICE_STRONG", "STRUCTURAL_MISPRICE_ACCEPTABLE"})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate structural line mispricing for candidate rows.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    annotated = annotate_structural_line_mispricing(frame)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_csv, index=False)
    if args.summary_json:
        payload: dict[str, Any] = {
            "rows": int(len(annotated)),
            "structural_mispricing_tier_counts": annotated["structural_mispricing_tier"].value_counts(dropna=False).to_dict(),
            "shadow_only": True,
            "production_behavior_changed": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
