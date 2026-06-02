from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import (
    as_string_list,
    coerce_market_family,
    safe_float,
    series_numeric,
    series_text,
    write_json,
)

EMPTY_CLUSTER_COLUMNS = [
    "candidate_failure_mode_id",
    "description",
    "affected_markets",
    "loss_count",
    "win_count_nearby",
    "loss_concentration",
    "defining_pre_event_features",
    "postgame_symptoms",
    "possible_causal_pathway",
    "suggested_diagnostic_features",
    "suggested_intervention_type",
    "overfit_risk",
    "sample_size_warning",
    "recommendation",
    "cluster_key",
    "player_count",
    "team_count",
    "date_span",
    "loss_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover recurring unexplained failure clusters from settled picks.")
    parser.add_argument("--failure-attribution-csv", type=Path, required=True)
    parser.add_argument("--clusters-csv-out", type=Path, required=True)
    parser.add_argument("--report-json-out", type=Path, required=True)
    parser.add_argument("--new-failure-mode-candidates-yaml-out", type=Path, required=True)
    parser.add_argument("--min-cluster-losses", type=int, default=3)
    parser.add_argument("--excluded-failure-mode", type=str, action="append", default=[])
    parser.add_argument("--excluded-market-family", type=str, action="append", default=[])
    parser.add_argument("--target-market-family", type=str, action="append", default=[])
    return parser.parse_args()


def _bucket_numeric(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(numeric, bins=edges, labels=labels, include_lowest=True).astype("object").fillna("unknown")


def _candidate_mode_id(market_family: str, cluster_key: str) -> str:
    digest = hashlib.sha1(cluster_key.encode("utf-8")).hexdigest()[:10]
    return f"UNKNOWN_{market_family}_{digest}".upper()


def _normalize_market_families(values: list[str] | set[str] | None) -> set[str]:
    return {str(item).strip().upper() for item in (values or []) if str(item).strip()}


def _postgame_symptoms(group: pd.DataFrame) -> str:
    symptoms: list[str] = []
    if safe_float(group.get("team_actual_ast_vs_trailing", pd.Series([np.nan])).mean(), default=np.nan) <= -0.15:
        symptoms.append("team_assist_collapse")
    if safe_float(group.get("team_actual_points_vs_trailing", pd.Series([np.nan])).mean(), default=np.nan) <= -0.12:
        symptoms.append("team_points_shortfall")
    if safe_float(group.get("team_actual_fg_pct_delta", pd.Series([np.nan])).mean(), default=np.nan) <= -0.05:
        symptoms.append("shooting_conversion_shortfall")
    if safe_float(group.get("player_actual_fga_vs_trailing", pd.Series([np.nan])).mean(), default=np.nan) <= -0.18:
        symptoms.append("usage_volume_shortfall")
    if safe_float(group.get("minutes_shortfall_ratio", pd.Series([np.nan])).mean(), default=np.nan) <= -0.12:
        symptoms.append("minutes_shortfall")
    if safe_float(group.get("price_gap", pd.Series([np.nan])).mean(), default=np.nan) >= 0.02:
        symptoms.append("expensive_price_region")
    return "|".join(symptoms) if symptoms else "unexplained_loss_cluster"


def _possible_causal_pathway(group: pd.DataFrame) -> str:
    market_family = str(group.get("market_family", pd.Series(["GENERIC"])).iloc[0]).upper()
    symptoms = _postgame_symptoms(group)
    if "team_assist_collapse" in symptoms:
        return "Assist-dependent overs entered a weak conversion environment that did not support the needed assist pool."
    if "team_points_shortfall" in symptoms or "shooting_conversion_shortfall" in symptoms:
        return "The broader team offense underperformed, shrinking the event pool required by the selected overs."
    if "usage_volume_shortfall" in symptoms:
        return "The player remained active but usage and shot volume landed below the required path."
    if "minutes_shortfall" in symptoms:
        return "The candidate relied on minutes that were less stable than the pre-event state suggested."
    if "expensive_price_region" in symptoms:
        return "The market appears directionally plausible but consistently too expensive at the accepted price band."
    return f"Recurring unexplained {market_family} losses are clustering in a repeatable pre-event state region."


def _suggested_intervention_type(group: pd.DataFrame) -> str:
    market_family = str(group.get("market_family", pd.Series(["GENERIC"])).iloc[0]).upper()
    symptoms = _postgame_symptoms(group)
    if "expensive_price_region" in symptoms:
        return "price_dependent_tier"
    if "minutes_shortfall" in symptoms:
        return "forecastability_penalty"
    if market_family == "AST" or "team_assist_collapse" in symptoms:
        return "scenario_stress_increase"
    if "usage_volume_shortfall" in symptoms:
        return "soft_downgrade"
    return "candidate_pool_reranking"


def _suggested_diagnostic_features(group: pd.DataFrame) -> str:
    features: list[str] = []
    for column in [
        "predicted_probability",
        "stress_probability",
        "market_side_break_even",
        "line_decision_fragility_score",
        "belief_uncertainty",
        "expected_minutes_band_width",
        "projected_team_fg_pct",
        "team_actual_points_vs_trailing",
        "team_actual_ast_vs_trailing",
        "player_actual_fga_vs_trailing",
    ]:
        if column in group.columns and group[column].notna().any():
            features.append(column)
    return "|".join(features)


def discover_unknown_failures(
    attributed_rows: pd.DataFrame,
    *,
    min_cluster_losses: int = 3,
    excluded_failure_modes: list[str] | set[str] | None = None,
    excluded_market_families: list[str] | set[str] | None = None,
    target_market_families: list[str] | set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if attributed_rows.empty:
        empty = pd.DataFrame(columns=EMPTY_CLUSTER_COLUMNS)
        return empty, {"cluster_count": 0, "register_count": 0}, {"failure_modes": []}

    work = attributed_rows.copy()
    work["failure_modes"] = work.get("failure_modes", pd.Series([[]] * len(work), index=work.index)).apply(as_string_list)
    work["result"] = series_text(work, "result").str.lower()
    work["market_family"] = coerce_market_family(work)
    excluded_modes = {str(item).strip() for item in (excluded_failure_modes or []) if str(item).strip()}
    excluded_markets = _normalize_market_families(excluded_market_families)
    target_markets = _normalize_market_families(target_market_families)
    if excluded_markets:
        work = work.loc[~work["market_family"].astype(str).str.upper().isin(excluded_markets)].copy()
    if target_markets:
        work = work.loc[work["market_family"].astype(str).str.upper().isin(target_markets)].copy()
    if work.empty:
        empty = pd.DataFrame(columns=EMPTY_CLUSTER_COLUMNS)
        return empty, {"cluster_count": 0, "register_count": 0}, {"failure_modes": []}

    work["known_failure_modes"] = work["failure_modes"].apply(
        lambda modes: [mode for mode in as_string_list(modes) if mode and mode not in excluded_modes]
    )
    work["recoverability_class"] = series_text(work, "recoverability_class")
    work["is_unexplained_loss"] = (
        work["result"].eq("loss")
        & (
            work["known_failure_modes"].map(len).eq(0)
            | work["recoverability_class"].isin(["ALEATORIC_OR_RANDOM", "DATA_MISSING"])
        )
    )
    if not bool(work["is_unexplained_loss"].any()):
        empty = pd.DataFrame(columns=EMPTY_CLUSTER_COLUMNS)
        return empty, {"cluster_count": 0, "register_count": 0}, {"failure_modes": []}

    work["direction_bucket"] = series_text(work, "direction").str.upper().replace("", "UNKNOWN")
    work["confidence_bucket"] = _bucket_numeric(
        series_numeric(
            work,
            "predicted_probability",
            default=np.nan,
        ).fillna(series_numeric(work, "stress_probability", default=np.nan)),
        [0.0, 0.54, 0.60, 0.68, 1.01],
        ["low", "medium", "high", "very_high"],
    )
    work["uncertainty_bucket"] = _bucket_numeric(
        series_numeric(work, "belief_uncertainty", default=np.nan),
        [0.0, 0.78, 0.92, 1.08, np.inf],
        ["steady", "normal", "fragile", "very_fragile"],
    )
    work["fragility_bucket"] = _bucket_numeric(
        series_numeric(work, "line_decision_fragility_score", default=np.nan),
        [0.0, 0.44, 0.54, 0.64, np.inf],
        ["steady", "soft", "fragile", "very_fragile"],
    )
    work["volatility_bucket"] = _bucket_numeric(
        series_numeric(work, "volatility_score", default=np.nan),
        [0.0, 0.44, 0.54, 0.64, np.inf],
        ["stable", "normal", "volatile", "very_volatile"],
    )
    work["team_environment_bucket"] = _bucket_numeric(
        series_numeric(work, "projected_team_fg_pct", default=np.nan),
        [0.0, 0.462, 0.472, 0.484, 1.01],
        ["low", "soft", "neutral", "strong"],
    )
    work["expected_minutes_band_bucket"] = _bucket_numeric(
        series_numeric(work, "expected_minutes_band_width", default=np.nan),
        [0.0, 4.0, 8.0, 12.0, np.inf],
        ["tight", "normal", "wide", "very_wide"],
    )
    work["price_gap"] = (
        series_numeric(work, "market_side_break_even", default=np.nan)
        - series_numeric(work, "stress_probability", default=np.nan).fillna(series_numeric(work, "expected_win_rate", default=np.nan))
    )
    work["price_bucket"] = _bucket_numeric(
        work["price_gap"],
        [-np.inf, -0.01, 0.015, 0.035, np.inf],
        ["cheap", "fair", "expensive", "very_expensive"],
    )
    work["minutes_shortfall_ratio"] = (
        series_numeric(work, "actual_minutes", default=np.nan)
        - series_numeric(work, "expected_minutes_band_low", default=np.nan)
    ) / series_numeric(work, "expected_minutes_band_low", default=np.nan).replace(0.0, np.nan)
    work["cluster_key"] = (
        work["market_family"].astype(str)
        + "|"
        + work["direction_bucket"].astype(str)
        + "|"
        + work["confidence_bucket"].astype(str)
        + "|"
        + work["uncertainty_bucket"].astype(str)
        + "|"
        + work["fragility_bucket"].astype(str)
        + "|"
        + work["price_bucket"].astype(str)
        + "|"
        + work["team_environment_bucket"].astype(str)
        + "|"
        + work["expected_minutes_band_bucket"].astype(str)
    )

    total_unexplained_losses = max(int(work["is_unexplained_loss"].sum()), 1)
    rows: list[dict[str, Any]] = []
    yaml_candidates: list[dict[str, Any]] = []
    for cluster_key, group in work.groupby("cluster_key", dropna=False):
        unexplained_losses = group.loc[group["is_unexplained_loss"]].copy()
        loss_count = int(len(unexplained_losses))
        if loss_count <= 0:
            continue
        win_count = int(group["result"].eq("win").sum())
        resolved = int(group["result"].isin(["win", "loss"]).sum())
        if resolved <= 0:
            continue
        player_count = int(series_text(unexplained_losses, "player").replace("", np.nan).nunique(dropna=True))
        team_columns = [column for column in ["team", "actual_team", "market_home_team", "market_away_team"] if column in unexplained_losses.columns]
        unique_teams: set[str] = set()
        for column in team_columns:
            unique_teams.update({value for value in series_text(unexplained_losses, column).tolist() if value})
        date_span = int(series_text(unexplained_losses, "game_date").replace("", np.nan).nunique(dropna=True))
        loss_rate = float(loss_count / max(1, loss_count + win_count))
        loss_concentration = float(loss_count / total_unexplained_losses)
        if loss_count < int(min_cluster_losses):
            recommendation = "NEEDS_MORE_SAMPLE"
        elif player_count <= 1 or len(unique_teams) <= 1 or date_span <= 1:
            recommendation = "REJECT_RANDOM"
        elif loss_rate >= 0.72 and win_count >= 1 and player_count >= 2 and len(unique_teams) >= 2 and date_span >= 2:
            recommendation = "REGISTER"
        elif loss_rate >= 0.60:
            recommendation = "WATCHLIST"
        else:
            recommendation = "REJECT_RANDOM"
        market_family = str(group["market_family"].iloc[0] or "GENERIC").upper()
        candidate_failure_mode_id = _candidate_mode_id(market_family, str(cluster_key))
        pre_event_features = [
            f"confidence_bucket={group['confidence_bucket'].iloc[0]}",
            f"uncertainty_bucket={group['uncertainty_bucket'].iloc[0]}",
            f"fragility_bucket={group['fragility_bucket'].iloc[0]}",
            f"price_bucket={group['price_bucket'].iloc[0]}",
            f"team_environment_bucket={group['team_environment_bucket'].iloc[0]}",
            f"expected_minutes_band_bucket={group['expected_minutes_band_bucket'].iloc[0]}",
        ]
        row = {
            "candidate_failure_mode_id": candidate_failure_mode_id,
            "description": f"Unexplained {market_family} losses are clustering in state region {cluster_key}.",
            "affected_markets": "|".join(sorted(series_text(group, "market_type").replace("", np.nan).dropna().unique().tolist())) or market_family,
            "loss_count": loss_count,
            "win_count_nearby": win_count,
            "loss_concentration": loss_concentration,
            "defining_pre_event_features": "|".join(pre_event_features),
            "postgame_symptoms": _postgame_symptoms(unexplained_losses),
            "possible_causal_pathway": _possible_causal_pathway(unexplained_losses),
            "suggested_diagnostic_features": _suggested_diagnostic_features(group),
            "suggested_intervention_type": _suggested_intervention_type(group),
            "overfit_risk": "high" if recommendation in {"REJECT_RANDOM", "NEEDS_MORE_SAMPLE"} else ("medium" if recommendation == "WATCHLIST" else "low"),
            "sample_size_warning": "" if loss_count >= int(min_cluster_losses) else "below_min_cluster_losses",
            "recommendation": recommendation,
            "cluster_key": cluster_key,
            "player_count": player_count,
            "team_count": len(unique_teams),
            "date_span": date_span,
            "loss_rate": loss_rate,
        }
        rows.append(row)
        if recommendation == "REGISTER":
            yaml_candidates.append(
                {
                    "failure_mode_id": candidate_failure_mode_id,
                    "market_families": [market_family],
                    "candidate_symptoms": pre_event_features,
                    "required_pre_event_features": as_string_list(row["suggested_diagnostic_features"]),
                    "postgame_attribution_signals": as_string_list(row["postgame_symptoms"]),
                    "likely_causal_pathway": row["possible_causal_pathway"],
                    "candidate_interventions": [
                        {
                            "intervention_type": row["suggested_intervention_type"],
                            "description": "Shadow-only candidate generated from unknown failure discovery.",
                        }
                    ],
                    "allowed_penalties_gates": ["soft_downgrade", "forecastability_penalty", "candidate_pool_reranking"],
                    "opposite_side_discovery_rules": ["require_valid_price"],
                    "validation_segments": [candidate_failure_mode_id],
                    "promotion_requirements": [
                        "trained_bundle_replay_required",
                        "broader_walk_forward_required",
                        "no_op_narrowness_required",
                        "active_window_improvement_required",
                    ],
                    "known_risks_overfit_traps": ["unknown_cluster_bootstrap_required", "one_off_loss_region"],
                }
            )
    clusters = (
        pd.DataFrame(rows).sort_values(
            ["recommendation", "loss_count", "loss_concentration"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=EMPTY_CLUSTER_COLUMNS)
    )
    report = {
        "cluster_count": int(len(clusters)),
        "register_count": int((clusters.get("recommendation", pd.Series(dtype="object")) == "REGISTER").sum()) if not clusters.empty else 0,
        "watchlist_count": int((clusters.get("recommendation", pd.Series(dtype="object")) == "WATCHLIST").sum()) if not clusters.empty else 0,
        "rejected_count": int((clusters.get("recommendation", pd.Series(dtype="object")) == "REJECT_RANDOM").sum()) if not clusters.empty else 0,
        "needs_more_sample_count": int((clusters.get("recommendation", pd.Series(dtype="object")) == "NEEDS_MORE_SAMPLE").sum()) if not clusters.empty else 0,
        "excluded_market_families": sorted(excluded_markets),
        "target_market_families": sorted(target_markets),
    }
    return clusters, report, {"failure_modes": yaml_candidates}


def main() -> None:
    args = parse_args()
    attributed = pd.read_csv(args.failure_attribution_csv)
    clusters, report, yaml_payload = discover_unknown_failures(
        attributed,
        min_cluster_losses=int(args.min_cluster_losses),
        excluded_failure_modes=args.excluded_failure_mode,
        excluded_market_families=args.excluded_market_family,
        target_market_families=args.target_market_family,
    )
    args.clusters_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(args.clusters_csv_out, index=False)
    write_json(args.report_json_out, report)
    args.new_failure_mode_candidates_yaml_out.resolve().write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
