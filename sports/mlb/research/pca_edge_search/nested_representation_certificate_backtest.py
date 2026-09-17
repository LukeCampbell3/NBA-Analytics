from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import sports.mlb.research.pca_edge_search.backtest_historical_universe_model_free as mf
import sports.mlb.research.pca_edge_search.nested_calibrated_edge_backtest as nested
from sports.mlb.research.pca_edge_search.backtest_historical_universe import SearchConfig


FEATURE_SETS = {
    "full": [
        "log_History_Rows",
        "days_since_history",
        "log_Market_Books",
        "Market_Line",
        "Market_Line_Std",
        "Is_Home",
        "is_pitcher",
        "pt_prior_mean",
        "pt_prior_std",
        "line_minus_prior_mean",
        "line_z",
        "no_vig_market_probability",
    ],
    "no_outcome_history": [
        "log_History_Rows",
        "days_since_history",
        "log_Market_Books",
        "Market_Line",
        "Market_Line_Std",
        "Is_Home",
        "is_pitcher",
        "no_vig_market_probability",
    ],
    "market_structure_only": [
        "log_Market_Books",
        "Market_Line",
        "Market_Line_Std",
        "Is_Home",
        "is_pitcher",
        "no_vig_market_probability",
    ],
    "market_core_only": [
        "log_Market_Books",
        "Market_Line",
        "Market_Line_Std",
        "no_vig_market_probability",
    ],
}

# Pre-specified before the outer holdout is scored. This is deliberately lower
# than the earlier rescue calibration's 10-play minimum because certificate-only
# filtering is narrower, but it still requires both inner validation dates.
MIN_INNER_PLAYS = 6


def dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    x = frame.copy()
    x = x.sort_values(
        ["contract_id", "conservative_residual", "base_edge_distance", "side"],
        ascending=[True, False, True, True],
    )
    return x.drop_duplicates("contract_id", keep="first")


def select_strict(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    return dedupe(scored[scored.strict_selected.fillna(False)].copy())


def select_certificate(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    mask = scored.strict_selected.fillna(False) & scored.base_edge_reachable.fillna(False)
    return dedupe(scored[mask].copy())


def select_rescue(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    mask = (
        scored.eligible.fillna(False)
        & scored.base_edge_reachable.fillna(False)
        & (pd.to_numeric(scored.conservative_residual, errors="coerce") > 0.0)
    )
    return dedupe(scored[mask].copy())


def by_date_roi(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    settled = frame[frame.settlement.isin(["win", "loss"])].copy()
    out: dict[str, float] = {}
    for day, group in settled.groupby(settled.Game_Date.dt.date):
        out[str(day)] = float(group.realized_units.mean())
    return out


def evaluate_selection(frame: pd.DataFrame, cfg: SearchConfig) -> dict:
    metrics = mf.metric_block(frame, cfg)
    date_roi = by_date_roi(frame)
    finite = [float(v) for v in date_roi.values() if np.isfinite(v)]
    return {
        "metrics": metrics,
        "by_date_roi": date_roi,
        "worst_date_roi": min(finite) if finite else None,
        "mean_date_roi": float(np.mean(finite)) if finite else None,
        "eligible_for_inner_choice": bool(
            int(metrics.get("plays", 0)) >= MIN_INNER_PLAYS and len(finite) >= 2
        ),
    }


def run(input_path: Path, output_json: Path, output_csv: Path) -> dict:
    cfg = SearchConfig(
        n_components=4,
        graph_k=8,
        local_k=24,
        min_support=20,
        edge_threshold=0.03,
        shrinkage=12.0,
        z_confidence=1.645,
        ood_quantile=0.95,
        path_radius_multiplier=3.0,
        initial_train_dates=4,
    )
    SearchConfig.seed = 20260917

    # prepare() creates the full leakage-safe state table once. Feature subsets
    # below only change the PCA/search geometry; the outcome model is never used.
    mf.FEATURES = list(FEATURE_SETS["full"])
    states, source_diag = mf.prepare(input_path, cfg)
    dates = sorted(states.Game_Date.dt.date.unique())
    outer_cut = max(1, min(len(dates) - 1, int(math.floor(0.75 * len(dates)))))
    development_dates = dates[:outer_cut]
    holdout_dates = dates[outer_cut:]
    inner_cut = max(1, len(development_dates) - 2)
    inner_train_dates = development_dates[:inner_cut]
    inner_validation_dates = development_dates[inner_cut:]

    inner_results: dict[str, dict] = {}
    for name, features in FEATURE_SETS.items():
        mf.FEATURES = list(features)
        space = nested.fit_space(states, set(inner_train_dates), cfg)
        scored = nested.score_dates(states, set(inner_validation_dates), space, cfg)
        strict = select_strict(scored)
        certificate = select_certificate(scored)
        rescue = select_rescue(scored)
        inner_results[name] = {
            "features": list(features),
            "pca_explained_variance_sum": float(space["pca"].explained_variance_ratio_.sum()),
            "edge_goal_nodes": int(space["base_goals"].sum()),
            "strict": evaluate_selection(strict, cfg),
            "certificate": evaluate_selection(certificate, cfg),
            "rescue": evaluate_selection(rescue, cfg),
        }

    candidates = [
        (name, block["certificate"])
        for name, block in inner_results.items()
        if block["certificate"]["eligible_for_inner_choice"]
    ]
    if candidates:
        # Primary objective is temporal robustness, not peak aggregate ROI.
        # Ties prefer higher mean date ROI, more plays, then fewer PCA inputs.
        selected_name, _ = max(
            candidates,
            key=lambda item: (
                item[1]["worst_date_roi"],
                item[1]["mean_date_roi"],
                item[1]["metrics"].get("plays", 0),
                -len(FEATURE_SETS[item[0]]),
            ),
        )
        fallback = False
    else:
        # Conservative deterministic fallback: use the simplest representation
        # that still contains player recency/sample support, never inspect holdout.
        selected_name = "no_outcome_history"
        fallback = True

    selected_features = list(FEATURE_SETS[selected_name])
    mf.FEATURES = selected_features
    final_space = nested.fit_space(states, set(development_dates), cfg)
    holdout_scored = nested.score_dates(states, set(holdout_dates), final_space, cfg)

    holdout_strict = select_strict(holdout_scored)
    holdout_certificate = select_certificate(holdout_scored)
    holdout_rescue = select_rescue(holdout_scored)

    cert_ids = set(holdout_certificate.contract_id) if not holdout_certificate.empty else set()
    strict_only = holdout_strict[~holdout_strict.contract_id.isin(cert_ids)].copy() if not holdout_strict.empty else holdout_strict.copy()
    rescue_ids = set(holdout_rescue.contract_id) if not holdout_rescue.empty else set()
    rescue_only = holdout_rescue[~holdout_rescue.contract_id.isin(set(holdout_strict.contract_id))].copy() if not holdout_rescue.empty else holdout_rescue.copy()

    # Evaluate all representations on outer holdout only for an audit table AFTER
    # the winner is frozen. These numbers are not used to alter selected_name.
    holdout_audit: dict[str, dict] = {}
    selected_scored_for_csv = holdout_scored.copy()
    for name, features in FEATURE_SETS.items():
        mf.FEATURES = list(features)
        if name == selected_name:
            space = final_space
            scored = holdout_scored
        else:
            space = nested.fit_space(states, set(development_dates), cfg)
            scored = nested.score_dates(states, set(holdout_dates), space, cfg)
        holdout_audit[name] = {
            "strict": evaluate_selection(select_strict(scored), cfg),
            "certificate": evaluate_selection(select_certificate(scored), cfg),
            "rescue": evaluate_selection(select_rescue(scored), cfg),
        }

    report = {
        "status": "NESTED_REPRESENTATION_SELECTION_OUTER_HOLDOUT",
        "strategy": "STRICT_LOCAL_EDGE_PLUS_SHORTEST_PATH_EDGE_CERTIFICATE",
        "source": source_diag,
        "feature_sets": FEATURE_SETS,
        "config": {
            "n_components": cfg.n_components,
            "graph_k": cfg.graph_k,
            "local_k": cfg.local_k,
            "min_support": cfg.min_support,
            "edge_threshold": cfg.edge_threshold,
            "shrinkage": cfg.shrinkage,
            "z_confidence": cfg.z_confidence,
            "ood_quantile": cfg.ood_quantile,
            "path_radius_multiplier": cfg.path_radius_multiplier,
            "min_inner_certificate_plays": MIN_INNER_PLAYS,
        },
        "split": {
            "inner_train_dates": [str(x) for x in inner_train_dates],
            "inner_validation_dates": [str(x) for x in inner_validation_dates],
            "outer_holdout_dates": [str(x) for x in holdout_dates],
        },
        "inner_development_selection": {
            "objective": "certificate only; maximize worst validation-date ROI, then mean date ROI, then plays, then fewer features; require >=6 plays across both validation dates",
            "results": inner_results,
            "selected_representation": selected_name,
            "selected_features": selected_features,
            "fallback_used": fallback,
        },
        "outer_holdout_selected_representation": {
            "strict_local": evaluate_selection(holdout_strict, cfg),
            "astar_certificate": evaluate_selection(holdout_certificate, cfg),
            "strict_not_certified": evaluate_selection(strict_only, cfg),
            "rescue_for_comparison_only": evaluate_selection(holdout_rescue, cfg),
            "rescue_incremental_additions": evaluate_selection(rescue_only, cfg),
            "certificate_by_target": mf.grouped(holdout_certificate, cfg, "Target"),
            "certificate_by_side": mf.grouped(holdout_certificate, cfg, "side"),
            "development_edge_goal_nodes": int(final_space["base_goals"].sum()),
            "pca_explained_variance_sum": float(final_space["pca"].explained_variance_ratio_.sum()),
        },
        "outer_holdout_audit_all_representations_not_used_for_selection": holdout_audit,
        "interpretation_guardrail": "Only outer_holdout_selected_representation is a nested representation-selection result. The audit table must not be used to change the selected representation after seeing holdout outcomes.",
        "computational_note": "The certificate uses exact precomputed directed shortest-path distance from every training graph node to a frozen edge goal. This preserves A* reachability semantics while avoiding repeated searches. If the graph changes incrementally, LPA*/D* Lite is the natural repair mechanism.",
        "limitations": [
            "Only 11 real-market dates are available: six inner-train, two inner-validation, and three outer-holdout dates.",
            "Representation selection is nested, but the candidate representation family itself was motivated by prior exploratory ablations on this corpus.",
            "The outer holdout is therefore cleaner than the earlier ablation result but not equivalent to a brand-new prospective season.",
            "Retrospective profitability does not establish future profitability.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    keep = [
        "Game_Date", "Game_ID", "Player", "Player_ID", "Target", "side", "Market_Line", "side_price",
        "no_vig_market_probability", "settlement", "realized_units", "empirical_probability", "raw_residual",
        "conservative_residual", "support", "nearest_distance", "eligible", "strict_selected",
        "base_edge_reachable", "base_edge_distance", "contract_id",
    ]
    selected_scored_for_csv[[c for c in keep if c in selected_scored_for_csv.columns]].to_csv(output_csv, index=False)
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()
    run(args.input, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
