from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sports.mlb.research.pca_edge_search.backtest_historical_universe_model_free as mf
from sports.mlb.research.pca_edge_search.backtest_historical_universe import (
    SearchConfig,
    build_graph,
    local_training_residuals,
)


FLOORS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03)
MIN_VALIDATION_PLAYS = 10


def reverse_multisource_distance(adj, goals: np.ndarray) -> np.ndarray:
    """Exact directed graph distance from every node to the nearest goal."""
    n = len(adj)
    rev: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, edges in enumerate(adj):
        for v, cost in edges:
            rev[v].append((u, float(cost)))
    dist = np.full(n, np.inf, dtype=float)
    q: list[tuple[float, int]] = []
    for g in np.flatnonzero(goals):
        dist[int(g)] = 0.0
        heapq.heappush(q, (0.0, int(g)))
    while q:
        d, u = heapq.heappop(q)
        if d != dist[u]:
            continue
        for v, cost in rev[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(q, (nd, v))
    return dist


def persistent_goal_mask(
    z: np.ndarray,
    train: pd.DataFrame,
    labels: list[tuple[str, str]],
    residual: np.ndarray,
    support: np.ndarray,
    cfg: SearchConfig,
) -> tuple[np.ndarray, dict]:
    """Require an edge node to survive a simple temporal concentration stress test.

    A node must already clear the frozen conservative 3% edge threshold. Its local
    neighborhood must cover at least three development dates, no single date may
    supply more than half the neighbors, and the raw residual must remain positive
    when any one represented date is removed (when at least 12 neighbors remain).
    This is intentionally stricter than the original edge-basin definition.
    """
    n = len(train)
    mask = np.zeros(n, dtype=bool)
    unique_dates = np.zeros(n, dtype=int)
    max_date_share = np.ones(n, dtype=float)
    min_loo_raw = np.full(n, -np.inf, dtype=float)
    market = train.no_vig_market_probability.to_numpy(float)
    outcomes = train.outcome.to_numpy(float)
    dates = train.Game_Date.dt.date.to_numpy()

    for label in sorted(set(labels)):
        ids = np.asarray([i for i, x in enumerate(labels) if x == label], dtype=int)
        if len(ids) < 2:
            continue
        zz = z[ids]
        q = np.sum(zz * zz, axis=1)
        dmat = np.sqrt(np.maximum(0.0, q[:, None] + q[None, :] - 2.0 * zz @ zz.T))
        np.fill_diagonal(dmat, np.inf)
        kk = min(cfg.local_k, len(ids) - 1)
        if kk <= 0:
            continue
        for a, i in enumerate(ids):
            if not (support[i] >= cfg.min_support and np.isfinite(residual[i]) and residual[i] >= cfg.edge_threshold):
                continue
            local = np.argpartition(dmat[a], kk - 1)[:kk]
            nbr = ids[local]
            good = np.isfinite(outcomes[nbr]) & np.isfinite(market[nbr])
            nbr = nbr[good]
            if len(nbr) < cfg.min_support:
                continue
            dvals, counts = np.unique(dates[nbr], return_counts=True)
            unique_dates[i] = len(dvals)
            max_date_share[i] = float(counts.max() / len(nbr)) if len(nbr) else 1.0
            loo_vals = []
            for day in dvals:
                keep = nbr[dates[nbr] != day]
                if len(keep) < 12:
                    continue
                p = (float(outcomes[keep].sum()) + cfg.shrinkage * float(market[keep].mean())) / (len(keep) + cfg.shrinkage)
                loo_vals.append(float(p - market[keep].mean()))
            min_loo_raw[i] = min(loo_vals) if loo_vals else -np.inf
            mask[i] = (
                unique_dates[i] >= 3
                and max_date_share[i] <= 0.50
                and min_loo_raw[i] > 0.0
            )

    base = (support >= cfg.min_support) & np.isfinite(residual) & (residual >= cfg.edge_threshold)
    return mask, {
        "base_edge_goal_nodes": int(base.sum()),
        "persistent_edge_goal_nodes": int(mask.sum()),
        "persistent_fraction": float(mask.sum() / base.sum()) if base.sum() else 0.0,
    }


def fit_space(states: pd.DataFrame, train_dates: set, cfg: SearchConfig) -> dict:
    train = states[states.Game_Date.dt.date.isin(train_dates) & states.outcome.notna()].copy().reset_index(drop=True)
    medians = train[mf.FEATURES].replace([np.inf, -np.inf], np.nan).median()
    xtrain = train[mf.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = StandardScaler().fit(xtrain)
    pca = PCA(n_components=min(cfg.n_components, len(mf.FEATURES)), svd_solver="full").fit(scaler.transform(xtrain))
    ztrain = pca.transform(scaler.transform(xtrain))
    labels = list(zip(train.Target.astype(str), train.side.astype(str)))
    adj, nearest, median_edge = build_graph(ztrain, labels, cfg.graph_k)
    residual, support = local_training_residuals(ztrain, train, labels, cfg)
    base_goals = (support >= cfg.min_support) & np.isfinite(residual) & (residual >= cfg.edge_threshold)
    persistent_goals, persistence_diag = persistent_goal_mask(ztrain, train, labels, residual, support, cfg)
    return {
        "train": train,
        "medians": medians,
        "scaler": scaler,
        "pca": pca,
        "ztrain": ztrain,
        "labels": labels,
        "adj": adj,
        "nearest": nearest,
        "median_edge": median_edge,
        "residual": residual,
        "support": support,
        "base_goals": base_goals,
        "persistent_goals": persistent_goals,
        "base_dist": reverse_multisource_distance(adj, base_goals),
        "persistent_dist": reverse_multisource_distance(adj, persistent_goals),
        "persistence_diag": persistence_diag,
    }


def score_dates(states: pd.DataFrame, candidate_dates: set, space: dict, cfg: SearchConfig) -> pd.DataFrame:
    frame = states[states.Game_Date.dt.date.isin(candidate_dates)].copy().reset_index(drop=True)
    x = frame[mf.FEATURES].replace([np.inf, -np.inf], np.nan).fillna(space["medians"]).fillna(0.0)
    z = space["pca"].transform(space["scaler"].transform(x))
    train = space["train"]
    labels = space["labels"]
    ztrain = space["ztrain"]
    nearest = space["nearest"]

    ood = {}
    for label in sorted(set(labels)):
        ids = np.asarray([i for i, value in enumerate(labels) if value == label], dtype=int)
        vals = nearest[ids]
        vals = vals[np.isfinite(vals)]
        ood[label] = float(np.quantile(vals, cfg.ood_quantile)) if len(vals) else math.inf

    rows = []
    for i, row in frame.iterrows():
        rec = row.to_dict()
        rec.update({"eligible": False})
        est = mf.candidate_local_estimate(row, z[i], train, ztrain, labels, cfg)
        if est is None:
            rec["rejection_reason"] = "LOCAL_SUPPORT"
            rows.append(rec)
            continue
        rec.update(est)
        label = est["label"]
        if est["nearest_distance"] > ood.get(label, math.inf):
            rec["rejection_reason"] = "OOD"
            rows.append(rec)
            continue
        anchor = int(est["start"])
        max_cost = cfg.path_radius_multiplier * float(space["median_edge"].get(label, 1.0))
        rec["eligible"] = True
        rec["strict_selected"] = bool(est["conservative_residual"] >= cfg.edge_threshold)
        rec["base_edge_distance"] = float(space["base_dist"][anchor]) if np.isfinite(space["base_dist"][anchor]) else math.inf
        rec["persistent_edge_distance"] = float(space["persistent_dist"][anchor]) if np.isfinite(space["persistent_dist"][anchor]) else math.inf
        rec["base_edge_reachable"] = bool(rec["base_edge_distance"] <= max_cost)
        rec["persistent_edge_reachable"] = bool(rec["persistent_edge_distance"] <= max_cost)
        rows.append(rec)
    return pd.DataFrame(rows)


def select_contracts(scored: pd.DataFrame, floor: float, distance_field: str) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    reachable = "base_edge_reachable" if distance_field == "base" else "persistent_edge_reachable"
    distance = "base_edge_distance" if distance_field == "base" else "persistent_edge_distance"
    mask = scored.eligible.fillna(False) & scored[reachable].fillna(False) & (pd.to_numeric(scored.conservative_residual, errors="coerce") >= floor)
    selected = scored[mask].copy()
    if selected.empty:
        return selected
    selected["distance_rank"] = pd.to_numeric(selected[distance], errors="coerce").fillna(1e12)
    selected = selected.sort_values(
        ["contract_id", "conservative_residual", "distance_rank", "side"],
        ascending=[True, False, True, True],
    )
    return selected.drop_duplicates("contract_id", keep="first").drop(columns=["distance_rank"])


def date_roi(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    out = {}
    for d, g in frame[frame.settlement.isin(["win", "loss"])].groupby(frame.Game_Date.dt.date):
        out[str(d)] = float(g.realized_units.mean()) if len(g) else math.nan
    return out


def calibrate_floor(scored: pd.DataFrame, field: str, cfg: SearchConfig) -> tuple[float, list[dict]]:
    table = []
    for floor in FLOORS:
        sel = select_contracts(scored, floor, field)
        metrics = mf.metric_block(sel, cfg)
        by_date = date_roi(sel)
        finite = [v for v in by_date.values() if np.isfinite(v)]
        worst = min(finite) if finite else -math.inf
        plays = int(metrics.get("plays", 0))
        eligible = plays >= MIN_VALIDATION_PLAYS and len(finite) >= 2
        table.append({
            "floor": floor,
            "plays": plays,
            "roi": metrics.get("roi_per_play"),
            "net_units": metrics.get("net_units"),
            "by_date_roi": by_date,
            "worst_date_roi": worst if np.isfinite(worst) else None,
            "eligible_for_selection": eligible,
        })
    eligible_rows = [r for r in table if r["eligible_for_selection"]]
    if not eligible_rows:
        return cfg.edge_threshold, table
    # Robust objective: maximize the worst validation-date ROI. Ties prefer the
    # higher floor, making the rescue rule more conservative rather than less.
    best = max(eligible_rows, key=lambda r: (r["worst_date_roi"], r["floor"]))
    return float(best["floor"]), table


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
    # metric_block's date bootstrap uses cfg.seed in the model-free helper.
    SearchConfig.seed = 20260917
    states, source_diag = mf.prepare(input_path, cfg)
    dates = sorted(states.Game_Date.dt.date.unique())
    outer_cut = max(1, min(len(dates) - 1, int(math.floor(0.75 * len(dates)))))
    dev = dates[:outer_cut]
    holdout = dates[outer_cut:]
    inner_cut = max(1, len(dev) - 2)
    inner_train = dev[:inner_cut]
    inner_val = dev[inner_cut:]

    inner_space = fit_space(states, set(inner_train), cfg)
    inner_scored = score_dates(states, set(inner_val), inner_space, cfg)
    base_floor, base_grid = calibrate_floor(inner_scored, "base", cfg)
    persistent_floor, persistent_grid = calibrate_floor(inner_scored, "persistent", cfg)

    final_space = fit_space(states, set(dev), cfg)
    final_scored = score_dates(states, set(holdout), final_space, cfg)

    strict = final_scored[final_scored.strict_selected.fillna(False)].copy()
    if not strict.empty:
        strict = strict.sort_values(["contract_id", "conservative_residual", "side"], ascending=[True, False, True]).drop_duplicates("contract_id")
    current = select_contracts(final_scored, 0.0, "base")
    calibrated = select_contracts(final_scored, base_floor, "base")
    persistent = select_contracts(final_scored, persistent_floor, "persistent")

    strict_ids = set(strict.contract_id) if not strict.empty else set()
    current_added = current[~current.contract_id.isin(strict_ids)].copy() if not current.empty else current.copy()
    calibrated_added = calibrated[~calibrated.contract_id.isin(strict_ids)].copy() if not calibrated.empty else calibrated.copy()
    persistent_added = persistent[~persistent.contract_id.isin(strict_ids)].copy() if not persistent.empty else persistent.copy()

    report = {
        "status": "NESTED_CHRONOLOGICAL_RESEARCH_BACKTEST",
        "strategy": "MODEL_FREE_PCA_EDGE_DISTANCE_WITH_CALIBRATED_RESCUE_AND_TEMPORAL_PERSISTENCE",
        "source": source_diag,
        "config": asdict(cfg),
        "features": list(mf.FEATURES),
        "split": {
            "inner_train_dates": [str(x) for x in inner_train],
            "inner_validation_dates": [str(x) for x in inner_val],
            "outer_holdout_dates": [str(x) for x in holdout],
        },
        "calibration": {
            "floor_grid": list(FLOORS),
            "selection_objective": "maximize worst validation-date ROI with >=10 plays and both inner validation dates represented; ties choose higher floor",
            "base_selected_floor": base_floor,
            "persistent_selected_floor": persistent_floor,
            "base_grid": base_grid,
            "persistent_grid": persistent_grid,
            "inner_persistence": inner_space["persistence_diag"],
        },
        "final_development_persistence": final_space["persistence_diag"],
        "holdout": {
            "strict_local_3pct": mf.metric_block(strict, cfg),
            "current_astar_equivalent_floor_0": mf.metric_block(current, cfg),
            "current_astar_incremental_additions": mf.metric_block(current_added, cfg),
            "calibrated_base_edge": mf.metric_block(calibrated, cfg),
            "calibrated_base_incremental_additions": mf.metric_block(calibrated_added, cfg),
            "persistent_calibrated_edge": mf.metric_block(persistent, cfg),
            "persistent_incremental_additions": mf.metric_block(persistent_added, cfg),
            "selection_counts": {
                "strict": len(strict),
                "current": len(current),
                "calibrated": len(calibrated),
                "persistent": len(persistent),
            },
            "persistent_by_target": mf.grouped(persistent, cfg, "Target"),
            "persistent_by_side": mf.grouped(persistent, cfg, "side"),
        },
        "computational_note": "Reverse multi-source Dijkstra precomputes exact graph distance to all frozen edge goals once. This is equivalent to repeated shortest-path reachability queries but avoids per-candidate A*/Dijkstra expansions; a future dynamic graph can replace this static field with LPA*/D* Lite repair.",
        "limitations": [
            "Only 11 real-market dates exist in this corpus; the inner validation uses two dates and the final holdout uses three dates.",
            "The rescue floor is selected only on inner development validation dates, never on the final holdout.",
            "Temporal persistence is a pre-specified concentration/leave-one-date-out stress rule, not a tuned final-holdout filter.",
            "This remains retrospective research and does not establish future profitability.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    keep = [
        "Game_Date", "Game_ID", "Player", "Player_ID", "Target", "side", "Market_Line", "side_price",
        "no_vig_market_probability", "settlement", "realized_units", "empirical_probability", "raw_residual",
        "conservative_residual", "support", "nearest_distance", "eligible", "strict_selected",
        "base_edge_reachable", "base_edge_distance", "persistent_edge_reachable", "persistent_edge_distance", "contract_id",
    ]
    final_scored[[c for c in keep if c in final_scored.columns]].to_csv(output_csv, index=False)
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
