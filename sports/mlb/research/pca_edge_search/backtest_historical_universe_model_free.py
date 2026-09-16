from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sports.mlb.research.pca_edge_search.backtest_historical_universe import (
    SearchConfig,
    american_implied,
    astar_edge_basin,
    build_graph,
    flat_units,
    local_training_residuals,
    residual_lipschitz,
    settle,
)


FEATURES = [
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
]

EXCLUDED_OUTCOME_MODEL_FIELDS = [
    "Prediction",
    "Edge",
    "Model_Selected",
    "Model_Members",
    "Model_Val_MAE",
    "Model_Val_RMSE",
]


def prepare(path: Path, cfg: SearchConfig) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False)
    source_rows = len(raw)
    raw["Game_Date"] = pd.to_datetime(raw["Game_Date"], errors="coerce", utc=True)
    raw["Actual"] = pd.to_numeric(raw["Actual"], errors="coerce")
    raw["Target"] = raw["Target"].astype(str).str.upper().str.strip()
    raw["Player_ID"] = raw["Player_ID"].astype(str).str.strip()

    # Build player/market representation features only from games strictly prior
    # to each candidate date. These are observed-history descriptors, not model
    # predictions of the candidate outcome.
    hist = (
        raw.loc[
            raw.Game_Date.notna() & raw.Actual.notna() & raw.Player_ID.ne(""),
            ["Game_Date", "Player_ID", "Target", "Actual"],
        ]
        .groupby(["Player_ID", "Target", "Game_Date"], as_index=False)["Actual"]
        .mean()
        .sort_values(["Player_ID", "Target", "Game_Date"])
    )
    group = hist.groupby(["Player_ID", "Target"], sort=False)["Actual"]
    hist["pt_prior_n"] = hist.groupby(["Player_ID", "Target"], sort=False).cumcount()
    hist["pt_prior_mean"] = group.transform(lambda s: s.shift(1).expanding().mean())
    hist["pt_prior_std"] = group.transform(lambda s: s.shift(1).expanding().std(ddof=1))
    raw = raw.merge(
        hist[["Player_ID", "Target", "Game_Date", "pt_prior_n", "pt_prior_mean", "pt_prior_std"]],
        on=["Player_ID", "Target", "Game_Date"],
        how="left",
        validate="many_to_one",
    )

    for col in ["Market_Line", "Market_Line_Std", "Market_Books", "History_Rows", "Is_Home"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["last_history_date"] = pd.to_datetime(raw["Last_History_Date"], errors="coerce", utc=True)
    raw["days_since_history"] = (raw.Game_Date - raw.last_history_date).dt.total_seconds() / 86400.0
    raw["is_pitcher"] = raw["Player_Type"].astype(str).str.lower().eq("pitcher").astype(float)

    real = raw[
        raw.Market_Source.astype(str).str.lower().eq("real")
        & raw.Game_Date.notna()
        & raw.Actual.notna()
        & raw.Market_Line.notna()
        & raw.Market_Over_Price.notna()
        & raw.Market_Under_Price.notna()
    ].copy()
    real["Market_Over_Price"] = pd.to_numeric(real.Market_Over_Price, errors="coerce")
    real["Market_Under_Price"] = pd.to_numeric(real.Market_Under_Price, errors="coerce")
    real["over_implied"] = real.Market_Over_Price.map(american_implied)
    real["under_implied"] = real.Market_Under_Price.map(american_implied)
    real = real[real.over_implied.notna() & real.under_implied.notna()].copy()
    real["vig_sum"] = real.over_implied + real.under_implied
    real = real[real.vig_sum.between(0.90, 1.35)].copy()
    real["p_over"] = real.over_implied / real.vig_sum
    real["p_under"] = real.under_implied / real.vig_sum

    identity = ["Game_Date", "Game_ID", "Player_ID", "Target", "Market_Line"]
    before_dedupe = len(real)
    real = real.sort_values(identity + ["vig_sum"]).drop_duplicates(identity, keep="first")

    states: list[dict] = []
    for _, row in real.iterrows():
        base = row.to_dict()
        for side in ("OVER", "UNDER"):
            rec = dict(base)
            rec["side"] = side
            if side == "OVER":
                rec["side_price"] = float(row.Market_Over_Price)
                rec["no_vig_market_probability"] = float(row.p_over)
            else:
                rec["side_price"] = float(row.Market_Under_Price)
                rec["no_vig_market_probability"] = float(row.p_under)
            rec["settlement"] = settle(float(row.Actual), float(row.Market_Line), side)
            rec["outcome"] = 1.0 if rec["settlement"] == "win" else 0.0 if rec["settlement"] == "loss" else math.nan
            rec["realized_units"] = flat_units(rec["settlement"], rec["side_price"])
            states.append(rec)

    df = pd.DataFrame(states)
    df["log_History_Rows"] = np.log1p(df.History_Rows.fillna(0).clip(lower=0))
    df["log_Market_Books"] = np.log1p(df.Market_Books.fillna(0).clip(lower=0))
    df["Market_Line_Std"] = df.Market_Line_Std.fillna(0.0)
    df["Is_Home"] = df.Is_Home.fillna(0.0)
    df["days_since_history"] = df.days_since_history.fillna(999.0)
    df["line_minus_prior_mean"] = df.Market_Line - df.pt_prior_mean
    denom = df.pt_prior_std.where(df.pt_prior_std.abs() > 1e-9)
    df["line_z"] = (df.Market_Line - df.pt_prior_mean) / denom
    df["partition"] = list(zip(df.Target.astype(str), df.side.astype(str)))
    df["contract_id"] = (
        df.Game_Date.dt.strftime("%Y-%m-%d") + "|" + df.Game_ID.astype(str) + "|"
        + df.Player_ID.astype(str) + "|" + df.Target.astype(str) + "|" + df.Market_Line.astype(str)
    )
    df = df.sort_values(["Game_Date", "Game_ID", "Player_ID", "Target", "side"]).reset_index(drop=True)

    same_book = (
        real.Market_Over_Book_Key.astype(str).str.lower().eq(real.Market_Under_Book_Key.astype(str).str.lower())
        & real.Market_Over_Book_Key.notna()
        & real.Market_Under_Book_Key.notna()
    )
    dates = sorted(df.Game_Date.dt.date.unique())
    return df, {
        "source_rows": source_rows,
        "real_two_sided_contracts_before_dedupe": before_dedupe,
        "real_two_sided_contracts": len(real),
        "expanded_side_states": len(df),
        "unique_real_market_dates": len(dates),
        "first_real_market_date": str(dates[0]) if dates else None,
        "last_real_market_date": str(dates[-1]) if dates else None,
        "same_book_two_sided_contracts": int(same_book.sum()),
    }


def candidate_local_estimate(row, zq, train, ztrain, labels, cfg: SearchConfig):
    label = (str(row.Target), str(row.side))
    ids = np.asarray([i for i, value in enumerate(labels) if value == label], dtype=int)
    if len(ids) < cfg.min_support:
        return None
    d = np.sqrt(np.sum((ztrain[ids] - zq) ** 2, axis=1))
    order = np.argsort(d)
    nbr = ids[order[: min(cfg.local_k, len(ids))]]
    y = train.iloc[nbr].outcome.to_numpy(float)
    good = np.isfinite(y)
    nbr = nbr[good]
    y = y[good]
    if len(nbr) < cfg.min_support:
        return None
    market_local = float(train.iloc[nbr].no_vig_market_probability.mean())
    p = (float(y.sum()) + cfg.shrinkage * market_local) / (len(y) + cfg.shrinkage)
    se = math.sqrt(max(1e-12, p * (1.0 - p) / (len(y) + cfg.shrinkage)))
    current_market = float(row.no_vig_market_probability)
    raw_residual = p - current_market
    conservative = raw_residual - cfg.z_confidence * se
    return {
        "label": label,
        "start": int(ids[order[0]]),
        "nearest_distance": float(d[order[0]]),
        "empirical_probability": p,
        "standard_error": se,
        "raw_residual": raw_residual,
        "conservative_residual": conservative,
        "support": int(len(y)),
    }


def dijkstra_edge_basin(start, adj, residual, support, cfg: SearchConfig, max_cost: float):
    def goal(i):
        return support[i] >= cfg.min_support and np.isfinite(residual[i]) and residual[i] >= cfg.edge_threshold
    q = [(0.0, start)]
    best = {start: 0.0}
    expanded = 0
    while q:
        g, u = __import__("heapq").heappop(q)
        if g != best.get(u) or g > max_cost:
            continue
        expanded += 1
        if goal(u):
            return True, g, expanded
        for v, step in adj[u]:
            ng = g + step
            if ng <= max_cost and ng < best.get(v, math.inf):
                best[v] = ng
                __import__("heapq").heappush(q, (ng, v))
    return False, math.inf, expanded


def resolve_contract_conflicts(scored: pd.DataFrame, flag: str) -> pd.DataFrame:
    selected = scored[scored[flag].fillna(False)].copy()
    if selected.empty:
        return selected
    selected["path_rank"] = pd.to_numeric(selected.astar_path_cost, errors="coerce").fillna(1e9)
    selected = selected.sort_values(
        ["contract_id", "conservative_residual", "path_rank", "side"],
        ascending=[True, False, True, True],
    )
    return selected.drop_duplicates("contract_id", keep="first").drop(columns=["path_rank"])


def metric_block(frame: pd.DataFrame, cfg: SearchConfig) -> dict:
    graded = frame[frame.settlement.isin(["win", "loss"])].copy()
    if graded.empty:
        return {"plays": 0, "dates": 0}
    n = len(graded)
    wins = int(graded.settlement.eq("win").sum())
    hit_rate = wins / n
    market_mean = float(graded.no_vig_market_probability.mean())
    market_brier = float(np.mean((graded.outcome - graded.no_vig_market_probability) ** 2))
    empirical = pd.to_numeric(graded.get("empirical_probability"), errors="coerce")
    empirical_brier = None
    if empirical.notna().any():
        mask = empirical.notna()
        empirical_brier = float(np.mean((graded.loc[mask, "outcome"] - empirical.loc[mask]) ** 2))

    probs = graded.no_vig_market_probability.to_numpy(float)
    pmf = np.zeros(n + 1, dtype=float)
    pmf[0] = 1.0
    for used, p in enumerate(probs, start=1):
        old = pmf.copy()
        pmf[: used + 1] = 0.0
        pmf[:used] += old[:used] * (1.0 - p)
        pmf[1 : used + 1] += old[:used] * p

    output = {
        "plays": n,
        "wins": wins,
        "losses": n - wins,
        "dates": int(graded.Game_Date.dt.date.nunique()),
        "hit_rate": hit_rate,
        "mean_no_vig_market_probability": market_mean,
        "realized_residual": hit_rate - market_mean,
        "net_units": float(graded.realized_units.sum()),
        "roi_per_play": float(graded.realized_units.mean()),
        "market_brier": market_brier,
        "empirical_brier": empirical_brier,
        "poisson_binomial_tail_p": float(pmf[wins:].sum()),
    }

    groups = {date: g for date, g in graded.groupby(graded.Game_Date.dt.date)}
    dates = list(groups)
    if len(dates) >= 2:
        rng = np.random.default_rng(cfg.seed)
        residuals, rois = [], []
        for _ in range(5000):
            sample = pd.concat([groups[d] for d in rng.choice(dates, len(dates), replace=True)], ignore_index=True)
            residuals.append(float(sample.outcome.mean() - sample.no_vig_market_probability.mean()))
            rois.append(float(sample.realized_units.mean()))
        output["residual_ci95"] = [float(v) for v in np.quantile(residuals, [0.025, 0.975])]
        output["roi_ci95"] = [float(v) for v in np.quantile(rois, [0.025, 0.975])]
    else:
        output["residual_ci95"] = [None, None]
        output["roi_ci95"] = [None, None]
    return output


def grouped(frame: pd.DataFrame, cfg: SearchConfig, field: str) -> dict:
    if frame.empty:
        return {}
    return {str(key): metric_block(group, cfg) for key, group in frame.groupby(field)}


def run(path: Path, output_json: Path, output_csv: Path) -> None:
    # Thresholds are fixed before the locked holdout; do not tune them on the
    # last 25% of dates.
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
    states, source_diag = prepare(path, cfg)
    dates = sorted(states.Game_Date.dt.date.unique())
    cut = max(1, min(len(dates) - 1, int(math.floor(0.75 * len(dates)))))
    development_dates = set(dates[:cut])
    holdout_dates = set(dates[cut:])
    train = states[states.Game_Date.dt.date.isin(development_dates) & states.outcome.notna()].copy().reset_index(drop=True)
    holdout = states[states.Game_Date.dt.date.isin(holdout_dates)].copy().reset_index(drop=True)

    medians = train[FEATURES].replace([np.inf, -np.inf], np.nan).median()
    xtrain = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    xhold = holdout[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = StandardScaler().fit(xtrain)
    pca = PCA(n_components=min(cfg.n_components, len(FEATURES)), svd_solver="full").fit(scaler.transform(xtrain))
    ztrain = pca.transform(scaler.transform(xtrain))
    zhold = pca.transform(scaler.transform(xhold))

    labels = list(zip(train.Target.astype(str), train.side.astype(str)))
    adj, nearest, median_edge = build_graph(ztrain, labels, cfg.graph_k)
    residual, support = local_training_residuals(ztrain, train, labels, cfg)
    L = residual_lipschitz(adj, residual)

    partition_ids: dict[tuple[str, str], np.ndarray] = {}
    partition_ood: dict[tuple[str, str], float] = {}
    for label in sorted(set(labels)):
        ids = np.asarray([i for i, x in enumerate(labels) if x == label], dtype=int)
        partition_ids[label] = ids
        vals = nearest[ids]
        vals = vals[np.isfinite(vals)]
        partition_ood[label] = float(np.quantile(vals, cfg.ood_quantile)) if len(vals) else math.inf

    # Confirm the A* heuristic is not changing the bounded shortest path result.
    rng = np.random.default_rng(20260916)
    starts = rng.choice(np.arange(len(train)), size=min(300, len(train)), replace=False)
    a_exp, d_exp, exact = [], [], 0
    for start in starts:
        label = labels[int(start)]
        max_cost = cfg.path_radius_multiplier * float(median_edge.get(label, 1.0))
        a = astar_edge_basin(int(start), label, adj, residual, support, cfg, L, max_cost)
        d = dijkstra_edge_basin(int(start), adj, residual, support, cfg, max_cost)
        same = a[0] == d[0] and ((not a[0]) or abs(float(a[1]) - float(d[1])) <= 1e-8)
        exact += int(same)
        a_exp.append(int(a[2])); d_exp.append(int(d[2]))

    cache: dict[int, tuple] = {}
    scored_rows = []
    for i, row in holdout.iterrows():
        est = candidate_local_estimate(row, zhold[i], train, ztrain, labels, cfg)
        rec = row.to_dict()
        rec.update({"eligible": False, "strict_selected": False, "astar_selected": False})
        if est is None:
            scored_rows.append(rec)
            continue
        rec.update(est)
        if est["nearest_distance"] > partition_ood.get(est["label"], math.inf):
            rec["rejection_reason"] = "OOD"
            scored_rows.append(rec)
            continue
        rec["eligible"] = True
        rec["strict_selected"] = est["conservative_residual"] >= cfg.edge_threshold
        anchor = est["start"]
        if anchor not in cache:
            max_cost = cfg.path_radius_multiplier * float(median_edge.get(est["label"], 1.0))
            cache[anchor] = astar_edge_basin(anchor, est["label"], adj, residual, support, cfg, L, max_cost)
        found, cost, expansions, path_nodes = cache[anchor]
        rec["astar_edge_found"] = bool(found)
        rec["astar_path_cost"] = None if not np.isfinite(cost) else float(cost)
        rec["astar_expansions"] = int(expansions)
        rec["astar_path_length"] = int(len(path_nodes))
        # A* is allowed to recover a locally positive candidate that has not
        # itself crossed the conservative threshold when it lies in a short,
        # frozen representation path to a development edge region.
        rec["astar_selected"] = bool(found and est["conservative_residual"] > 0.0)
        scored_rows.append(rec)

    scored = pd.DataFrame(scored_rows)
    strict = resolve_contract_conflicts(scored, "strict_selected")
    astar_selected = resolve_contract_conflicts(scored, "astar_selected")
    strict_ids = set(strict.contract_id) if not strict.empty else set()
    astar_added = astar_selected[~astar_selected.contract_id.isin(strict_ids)].copy() if not astar_selected.empty else astar_selected.copy()
    astar_overlap = astar_selected[astar_selected.contract_id.isin(strict_ids)].copy() if not astar_selected.empty else astar_selected.copy()

    report = {
        "status": "LOCKED_CHRONOLOGICAL_RESEARCH_BACKTEST",
        "strategy": "MODEL_FREE_PCA_ASTAR_MARKET_RESIDUAL_EDGE",
        "semantics": {
            "astar_success": "EDGE_FOUND means a short frozen PCA path reaches a supported development residual region; it does not mean the wager will win.",
            "settlement": "Holdout outcomes are used only after selection for evaluation.",
        },
        "source": source_diag,
        "split": {
            "development_dates": len(development_dates),
            "holdout_dates": len(holdout_dates),
            "development_start": str(dates[0]),
            "development_end": str(dates[cut - 1]),
            "holdout_start": str(dates[cut]),
            "holdout_end": str(dates[-1]),
            "development_side_states": len(train),
            "holdout_side_states": len(holdout),
        },
        "config": asdict(cfg),
        "features": FEATURES,
        "excluded_outcome_model_fields": EXCLUDED_OUTCOME_MODEL_FIELDS,
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "development_edge_goal_nodes": int(np.sum((support >= cfg.min_support) & np.isfinite(residual) & (residual >= cfg.edge_threshold))),
        "astar_vs_dijkstra": {
            "sampled_starts": len(starts),
            "exact_status_and_path_cost_matches": exact,
            "exact_match_rate": exact / len(starts) if len(starts) else None,
            "astar_mean_expansions": float(np.mean(a_exp)) if a_exp else None,
            "dijkstra_mean_expansions": float(np.mean(d_exp)) if d_exp else None,
            "expansion_reduction": float(1.0 - np.mean(a_exp) / np.mean(d_exp)) if d_exp and np.mean(d_exp) > 0 else None,
        },
        "holdout": {
            "eligible_side_states": int(scored.eligible.fillna(False).sum()),
            "strict_local_edge": metric_block(strict, cfg),
            "astar_path_edge": metric_block(astar_selected, cfg),
            "astar_incremental_additions": metric_block(astar_added, cfg),
            "astar_overlap_with_strict": metric_block(astar_overlap, cfg),
            "selection_overlap": {
                "strict_contracts": len(strict),
                "astar_contracts": len(astar_selected),
                "overlap_contracts": len(astar_overlap),
                "astar_added_contracts": len(astar_added),
            },
            "strict_by_target": grouped(strict, cfg, "Target"),
            "astar_by_target": grouped(astar_selected, cfg, "Target"),
            "astar_by_side": grouped(astar_selected, cfg, "side"),
        },
        "limitations": [
            "No Prediction, Edge, model-selection, model-error, or other outcome-model fields are used in the PCA representation.",
            "Two-sided probabilities normalize the stored over/under prices. When the best over and under prices come from different books, this is a composite market surface rather than a single-book vig removal.",
            "Player history features use only games strictly prior to each candidate date. The PCA basis, residual topology, thresholds, and path rule are frozen at the development boundary.",
            "This is retrospective research and does not establish future profitability.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    keep = [
        "Game_Date", "Game_ID", "Player", "Player_ID", "Target", "side", "Market_Line", "side_price",
        "no_vig_market_probability", "settlement", "realized_units", "empirical_probability",
        "raw_residual", "conservative_residual", "support", "nearest_distance", "eligible",
        "strict_selected", "astar_selected", "astar_edge_found", "astar_path_cost", "astar_expansions",
        "astar_path_length", "contract_id",
    ]
    scored[[c for c in keep if c in scored.columns]].to_csv(output_csv, index=False)
    print(json.dumps(report, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()
    run(args.input, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
