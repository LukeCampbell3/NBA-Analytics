from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Prediction",
    "Market_Line",
    "Edge",
    "log_History_Rows",
    "Model_Val_MAE",
    "Model_Val_RMSE",
    "Market_Books",
    "Market_Line_Std",
    "Is_Home",
    "no_vig_market_probability",
    "prediction_minus_line",
    "abs_prediction_minus_line",
]


@dataclass(frozen=True)
class SearchConfig:
    n_components: int = 6
    graph_k: int = 10
    local_k: int = 40
    min_support: int = 20
    edge_threshold: float = 0.03
    shrinkage: float = 20.0
    z_confidence: float = 1.0
    ood_quantile: float = 0.95
    path_radius_multiplier: float = 6.0
    initial_train_dates: int = 4
    same_book_max_quote_gap_minutes: float = 15.0


def american_implied(odds: Any) -> float:
    try:
        x = float(odds)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(x) or (-100.0 < x < 100.0):
        return math.nan
    return 100.0 / (x + 100.0) if x > 0 else (-x) / ((-x) + 100.0)


def flat_units(settlement: str, odds: float) -> float:
    if settlement == "push":
        return 0.0
    if settlement == "loss":
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def settle(actual: float, line: float, side: str) -> str:
    if actual == line:
        return "push"
    won = actual > line if side == "OVER" else actual < line
    return "win" if won else "loss"


def _clean_book(value: Any) -> str:
    return str(value or "").strip().lower()


def prepare_states(path: Path, *, strict_same_book: bool, cfg: SearchConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path, low_memory=False)
    required = {
        "Game_Date", "Game_ID", "Player_ID", "Target", "Prediction", "Market_Line",
        "Market_Source", "Market_Over_Price", "Market_Under_Price", "Actual",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"historical universe missing required fields: {missing}")

    for c in [
        "Prediction", "Market_Line", "Market_Over_Price", "Market_Under_Price", "Actual",
        "Edge", "History_Rows", "Model_Val_MAE", "Model_Val_RMSE", "Market_Books",
        "Market_Line_Std", "Is_Home",
    ]:
        if c in raw:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["Game_Date"] = pd.to_datetime(raw["Game_Date"], errors="coerce", utc=True)
    raw["over_time"] = pd.to_datetime(raw.get("Market_Over_Price_Time"), errors="coerce", utc=True)
    raw["under_time"] = pd.to_datetime(raw.get("Market_Under_Price_Time"), errors="coerce", utc=True)
    raw["over_book_key"] = raw.get("Market_Over_Book_Key", raw.get("Market_Over_Book", "")).map(_clean_book)
    raw["under_book_key"] = raw.get("Market_Under_Book_Key", raw.get("Market_Under_Book", "")).map(_clean_book)

    real = raw[raw["Market_Source"].astype(str).str.lower().eq("real")].copy()
    paired = real[
        real["Game_Date"].notna()
        & real["Prediction"].notna()
        & real["Market_Line"].notna()
        & real["Actual"].notna()
        & real["Market_Over_Price"].notna()
        & real["Market_Under_Price"].notna()
    ].copy()

    paired["over_implied"] = paired["Market_Over_Price"].map(american_implied)
    paired["under_implied"] = paired["Market_Under_Price"].map(american_implied)
    paired = paired[paired.over_implied.notna() & paired.under_implied.notna()].copy()
    paired["vig_sum"] = paired.over_implied + paired.under_implied
    paired = paired[paired.vig_sum.between(0.90, 1.35)].copy()

    if strict_same_book:
        same = paired.over_book_key.ne("") & paired.over_book_key.eq(paired.under_book_key)
        both_time = paired.over_time.notna() & paired.under_time.notna()
        gap_minutes = (paired.over_time - paired.under_time).abs().dt.total_seconds() / 60.0
        paired = paired[same & both_time & gap_minutes.le(cfg.same_book_max_quote_gap_minutes)].copy()

    paired["p_over_no_vig"] = paired.over_implied / paired.vig_sum
    paired["p_under_no_vig"] = paired.under_implied / paired.vig_sum
    paired["log_History_Rows"] = np.log1p(pd.to_numeric(paired.get("History_Rows"), errors="coerce").clip(lower=0))
    paired["prediction_minus_line"] = paired.Prediction - paired.Market_Line
    paired["abs_prediction_minus_line"] = paired.prediction_minus_line.abs()

    identity = ["Game_Date", "Game_ID", "Player_ID", "Target", "Market_Line"]
    before_dedupe = len(paired)
    paired = paired.sort_values(identity + ["vig_sum"]).drop_duplicates(identity, keep="first")

    states: list[dict[str, Any]] = []
    for _, r in paired.iterrows():
        base = r.to_dict()
        for side in ("OVER", "UNDER"):
            rec = dict(base)
            if side == "OVER":
                rec["side_price"] = float(r.Market_Over_Price)
                rec["no_vig_market_probability"] = float(r.p_over_no_vig)
                rec["book"] = str(r.get("Market_Over_Book") or r.get("Market_Over_Book_Key") or "")
                rec["quote_time"] = r.over_time
            else:
                rec["side_price"] = float(r.Market_Under_Price)
                rec["no_vig_market_probability"] = float(r.p_under_no_vig)
                rec["book"] = str(r.get("Market_Under_Book") or r.get("Market_Under_Book_Key") or "")
                rec["quote_time"] = r.under_time
            rec["side"] = side
            rec["settlement"] = settle(float(r.Actual), float(r.Market_Line), side)
            rec["outcome"] = 1.0 if rec["settlement"] == "win" else 0.0 if rec["settlement"] == "loss" else math.nan
            rec["realized_units"] = flat_units(rec["settlement"], rec["side_price"])
            states.append(rec)

    df = pd.DataFrame(states)
    if not df.empty:
        df = df.sort_values(["Game_Date", "Game_ID", "Player_ID", "Target", "Market_Line", "side"]).reset_index(drop=True)

    diag = {
        "raw_rows": int(len(raw)),
        "real_source_rows": int(len(real)),
        "paired_price_rows_before_strict_filter": int((real.Market_Over_Price.notna() & real.Market_Under_Price.notna()).sum()),
        "paired_contracts_before_dedupe": int(before_dedupe),
        "paired_contracts": int(len(paired)),
        "expanded_side_states": int(len(df)),
        "unique_game_dates": int(df.Game_Date.dt.date.nunique()) if not df.empty else 0,
        "strict_same_book": bool(strict_same_book),
        "same_book_max_quote_gap_minutes": cfg.same_book_max_quote_gap_minutes,
    }
    return df, diag


def _partition_labels(df: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(df.Target.astype(str), df.side.astype(str)))


def _pairwise(z: np.ndarray) -> np.ndarray:
    q = np.sum(z * z, axis=1)
    return np.sqrt(np.maximum(0.0, q[:, None] + q[None, :] - 2.0 * z @ z.T))


def build_graph(z: np.ndarray, labels: list[tuple[str, str]], k: int):
    n = len(z)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    nearest = np.full(n, np.nan)
    partition_median_edge: dict[tuple[str, str], float] = {}
    for label in sorted(set(labels)):
        ids = np.array([i for i, x in enumerate(labels) if x == label], dtype=int)
        if len(ids) < 2:
            continue
        d = _pairwise(z[ids])
        np.fill_diagonal(d, np.inf)
        kk = min(k, len(ids) - 1)
        edge_costs: list[float] = []
        for a, i in enumerate(ids):
            js = np.argpartition(d[a], kk - 1)[:kk]
            nearest[i] = float(d[a, js].min())
            for b in js:
                dist = float(d[a, b])
                if np.isfinite(dist) and dist > 0:
                    adj[i].append((int(ids[b]), dist))
                    edge_costs.append(dist)
        if edge_costs:
            partition_median_edge[label] = float(np.median(edge_costs))
    return adj, nearest, partition_median_edge


def local_training_residuals(z: np.ndarray, train: pd.DataFrame, labels: list[tuple[str, str]], cfg: SearchConfig):
    y = train.outcome.to_numpy(float)
    market = train.no_vig_market_probability.to_numpy(float)
    n = len(train)
    residual = np.full(n, np.nan)
    support = np.zeros(n, dtype=int)
    for label in sorted(set(labels)):
        ids = np.array([i for i, x in enumerate(labels) if x == label], dtype=int)
        if len(ids) < 2:
            continue
        d = _pairwise(z[ids])
        np.fill_diagonal(d, np.inf)
        for a, i in enumerate(ids):
            kk = min(cfg.local_k, len(ids) - 1)
            if kk <= 0:
                continue
            nbr_local = np.argpartition(d[a], kk - 1)[:kk]
            nbr = ids[nbr_local]
            good = np.isfinite(y[nbr]) & np.isfinite(market[nbr])
            nbr = nbr[good]
            if not len(nbr):
                continue
            yy = y[nbr]
            mm = market[nbr]
            market_mean = float(mm.mean())
            p = (float(yy.sum()) + cfg.shrinkage * market_mean) / (len(yy) + cfg.shrinkage)
            se = math.sqrt(max(1e-12, p * (1.0 - p) / (len(yy) + cfg.shrinkage)))
            residual[i] = p - market_mean - cfg.z_confidence * se
            support[i] = len(yy)
    return residual, support


def residual_lipschitz(adj, residual: np.ndarray) -> float:
    L = 0.0
    for i, edges in enumerate(adj):
        if not np.isfinite(residual[i]):
            continue
        for j, dist in edges:
            if dist > 0 and np.isfinite(residual[j]):
                L = max(L, abs(float(residual[i] - residual[j])) / dist)
    return max(L, 1e-9)


def astar_edge_basin(start: int, label: tuple[str, str], adj, residual, support, cfg: SearchConfig, L: float, max_cost: float):
    def is_goal(i: int) -> bool:
        return support[i] >= cfg.min_support and np.isfinite(residual[i]) and residual[i] >= cfg.edge_threshold

    def h(i: int) -> float:
        if not np.isfinite(residual[i]):
            return 0.0
        return max(0.0, cfg.edge_threshold - float(residual[i])) / L

    q = [(h(start), 0.0, start)]
    best = {start: 0.0}
    parent: dict[int, int] = {}
    expansions = 0
    while q:
        _, g, i = heapq.heappop(q)
        if g != best.get(i) or g > max_cost:
            continue
        expansions += 1
        if is_goal(i):
            path = [i]
            while i in parent:
                i = parent[i]
                path.append(i)
            path.reverse()
            return True, g, expansions, path
        for j, cost in adj[i]:
            ng = g + cost
            if ng <= max_cost and ng < best.get(j, math.inf):
                best[j] = ng
                parent[j] = i
                heapq.heappush(q, (ng + h(j), ng, j))
    return False, math.inf, expansions, []


def score_candidate(row: pd.Series, zq: np.ndarray, train: pd.DataFrame, ztrain: np.ndarray, labels, adj, nearest, med_edge, residual, support, L: float, cfg: SearchConfig):
    label = (str(row.Target), str(row.side))
    ids = np.array([i for i, x in enumerate(labels) if x == label], dtype=int)
    if len(ids) < cfg.min_support:
        return {"eligible": False, "reason": "PARTITION_SUPPORT"}

    distances = np.sqrt(np.sum((ztrain[ids] - zq) ** 2, axis=1))
    order = np.argsort(distances)
    start = int(ids[order[0]])

    partition_nearest = nearest[ids]
    partition_nearest = partition_nearest[np.isfinite(partition_nearest)]
    ood_limit = float(np.quantile(partition_nearest, cfg.ood_quantile)) if len(partition_nearest) else math.inf
    nearest_distance = float(distances[order[0]])
    if nearest_distance > ood_limit:
        return {"eligible": False, "reason": "OOD", "nearest_distance": nearest_distance, "ood_limit": ood_limit}

    nbr = ids[order[: min(cfg.local_k, len(ids))]]
    yy = train.iloc[nbr].outcome.to_numpy(float)
    good = np.isfinite(yy)
    nbr = nbr[good]
    yy = yy[good]
    if len(yy) < cfg.min_support:
        return {"eligible": False, "reason": "LOCAL_SUPPORT", "support": int(len(yy))}

    market_p = float(row.no_vig_market_probability)
    p = (float(yy.sum()) + cfg.shrinkage * market_p) / (len(yy) + cfg.shrinkage)
    se = math.sqrt(max(1e-12, p * (1.0 - p) / (len(yy) + cfg.shrinkage)))
    conservative = p - market_p - cfg.z_confidence * se

    max_cost = cfg.path_radius_multiplier * float(med_edge.get(label, 1.0))
    found, path_cost, expansions, path = astar_edge_basin(start, label, adj, residual, support, cfg, L, max_cost)
    selected = bool(found and conservative >= cfg.edge_threshold)
    return {
        "eligible": True,
        "selected": selected,
        "local_empirical_probability": p,
        "local_standard_error": se,
        "conservative_residual": conservative,
        "support": int(len(yy)),
        "nearest_distance": nearest_distance,
        "ood_limit": ood_limit,
        "astar_edge_found": bool(found),
        "astar_path_cost": None if not np.isfinite(path_cost) else float(path_cost),
        "astar_expansions": int(expansions),
        "astar_path_length": int(len(path)),
    }


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, cfg: SearchConfig):
    usable_features = [c for c in FEATURE_COLUMNS if c in train.columns and train[c].notna().sum() >= max(20, int(0.5 * len(train)))]
    if not usable_features:
        raise ValueError("no usable PCA features")
    medians = train[usable_features].median(numeric_only=True)
    xtr = train[usable_features].fillna(medians).fillna(0.0)
    xte = test[usable_features].fillna(medians).fillna(0.0)
    scaler = StandardScaler().fit(xtr)
    k = max(1, min(cfg.n_components, len(usable_features), len(train)))
    pca = PCA(n_components=k, svd_solver="full").fit(scaler.transform(xtr))
    ztr = pca.transform(scaler.transform(xtr))
    zte = pca.transform(scaler.transform(xte))
    labels = _partition_labels(train)
    adj, nearest, med_edge = build_graph(ztr, labels, cfg.graph_k)
    residual, support = local_training_residuals(ztr, train, labels, cfg)
    L = residual_lipschitz(adj, residual)
    goals = int(np.sum((support >= cfg.min_support) & np.isfinite(residual) & (residual >= cfg.edge_threshold)))

    rows = []
    for i, (_, row) in enumerate(test.iterrows()):
        rec = row.to_dict()
        rec.update(score_candidate(row, zte[i], train, ztr, labels, adj, nearest, med_edge, residual, support, L, cfg))
        rows.append(rec)
    scored = pd.DataFrame(rows)
    diagnostics = {
        "features": usable_features,
        "components": k,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "edge_goal_nodes": goals,
        "lipschitz_bound": float(L),
    }
    return scored, diagnostics


def expanding_walk_forward(df: pd.DataFrame, cfg: SearchConfig):
    dates = sorted(df.Game_Date.dt.date.unique())
    if len(dates) < 3:
        return pd.DataFrame(), [], dates
    initial = min(max(2, cfg.initial_train_dates), len(dates) - 1)
    outputs = []
    folds = []
    for test_index in range(initial, len(dates)):
        train_dates = set(dates[:test_index])
        test_date = dates[test_index]
        train = df[df.Game_Date.dt.date.isin(train_dates)].copy().reset_index(drop=True)
        test = df[df.Game_Date.dt.date.eq(test_date)].copy().reset_index(drop=True)
        if train.empty or test.empty:
            continue
        scored, diag = fit_and_score(train, test, cfg)
        scored["fold"] = test_index - initial + 1
        outputs.append(scored)
        folds.append({
            "fold": test_index - initial + 1,
            "test_date": str(test_date),
            "train_dates": int(len(train_dates)),
            "train_states": int(len(train)),
            "test_states": int(len(test)),
            "eligible": int(scored.get("eligible", False).fillna(False).sum()),
            "selected_states": int(scored.get("selected", False).fillna(False).sum()),
            **diag,
        })
    return (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()), folds, dates


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"plays": 0}
    graded = df[df.settlement.isin(["win", "loss"])].copy()
    if graded.empty:
        return {"plays": 0}
    n = len(graded)
    wins = int(graded.settlement.eq("win").sum())
    hit = wins / n
    market = float(graded.no_vig_market_probability.mean())
    units = float(graded.realized_units.sum())
    return {
        "plays": int(n),
        "wins": wins,
        "losses": int(n - wins),
        "hit_rate": hit,
        "mean_no_vig_market_probability": market,
        "realized_residual": hit - market,
        "units": units,
        "roi": units / n,
        "unique_dates": int(graded.Game_Date.dt.date.nunique()),
        "unique_games": int(graded.Game_ID.astype(str).nunique()),
        "unique_players": int(graded.Player_ID.astype(str).nunique()),
    }


def actionable_dedupe(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    selected = scored[scored.get("selected", False).fillna(False)].copy()
    if selected.empty:
        return selected
    selected["_resid"] = pd.to_numeric(selected.conservative_residual, errors="coerce").fillna(-math.inf)
    selected["_dist"] = pd.to_numeric(selected.nearest_distance, errors="coerce").fillna(math.inf)
    keys = ["Game_Date", "Game_ID", "Player_ID", "Target"]
    selected = selected.sort_values(keys + ["_resid", "_dist"], ascending=[True, True, True, True, False, True])
    return selected.drop_duplicates(keys, keep="first").drop(columns=["_resid", "_dist"])


def projection_baseline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    over = df[df.side.eq("OVER")].copy()
    under = df[df.side.eq("UNDER")].copy()
    keys = ["Game_Date", "Game_ID", "Player_ID", "Target", "Market_Line"]
    contracts = pd.concat([over, under], ignore_index=True)
    contracts["desired_side"] = np.where(contracts.prediction_minus_line >= 0, "OVER", "UNDER")
    contracts = contracts[contracts.side.eq(contracts.desired_side)].copy()
    contracts = contracts.sort_values(keys).drop_duplicates(keys, keep="first")
    return contracts


def date_cluster_bootstrap(df: pd.DataFrame, reps: int = 10000) -> dict[str, Any]:
    graded = df[df.settlement.isin(["win", "loss"])].copy()
    dates = list(graded.Game_Date.dt.date.unique())
    if len(dates) < 3:
        return {"roi_ci95": [None, None], "residual_ci95": [None, None], "bootstrap_dates": len(dates)}
    groups = {d: graded[graded.Game_Date.dt.date.eq(d)] for d in dates}
    rng = np.random.default_rng(20260916)
    roi = []
    residual = []
    for _ in range(reps):
        sample = pd.concat([groups[d] for d in rng.choice(dates, len(dates), replace=True)], ignore_index=True)
        roi.append(float(sample.realized_units.sum() / len(sample)))
        residual.append(float(sample.outcome.mean() - sample.no_vig_market_probability.mean()))
    return {
        "roi_ci95": [float(x) for x in np.quantile(roi, [0.025, 0.975])],
        "residual_ci95": [float(x) for x in np.quantile(residual, [0.025, 0.975])],
        "bootstrap_dates": len(dates),
    }


def per_date(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = []
    for date, g in df.groupby(df.Game_Date.dt.date):
        row = {"date": str(date)}
        row.update(summarize(g))
        out.append(row)
    return out


def per_market(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = []
    for (target, side), g in df.groupby(["Target", "side"]):
        row = {"target": str(target), "side": str(side)}
        row.update(summarize(g))
        out.append(row)
    return out


def run_analysis(path: Path, cfg: SearchConfig, strict_same_book: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    states, pipeline = prepare_states(path, strict_same_book=strict_same_book, cfg=cfg)
    scored, folds, all_dates = expanding_walk_forward(states, cfg)
    selected_all = scored[scored.get("selected", False).fillna(False)].copy() if not scored.empty else scored
    actionable = actionable_dedupe(scored)

    if not scored.empty:
        first_test_date = min(scored.Game_Date.dt.date)
        holdout_universe = states[states.Game_Date.dt.date >= first_test_date].copy()
    else:
        holdout_universe = states.iloc[0:0].copy()
    baseline = projection_baseline(holdout_universe)

    report = {
        "pipeline": pipeline,
        "chronology": {
            "all_real_quote_dates": [str(x) for x in all_dates],
            "folds": folds,
        },
        "holdout_states": summarize(scored),
        "astar_selected_states": summarize(selected_all),
        "actionable_one_per_player_target": summarize(actionable),
        "projection_direction_baseline": summarize(baseline),
        "actionable_bootstrap": date_cluster_bootstrap(actionable),
        "actionable_by_date": per_date(actionable),
        "actionable_by_market": per_market(actionable),
    }
    return report, scored


def run(input_path: Path, output_json: Path, output_csv: Path):
    cfg = SearchConfig()
    strict_report, strict_rows = run_analysis(input_path, cfg, strict_same_book=True)
    broad_report, broad_rows = run_analysis(input_path, cfg, strict_same_book=False)
    report = {
        "status": "DEVELOPMENT_BACKTEST",
        "semantics": {
            "astar_success": "EDGE_FOUND in a prior-data PCA residual basin; never a win label",
            "settlement": "derived after frozen selection from Actual versus Market_Line",
        },
        "config": asdict(cfg),
        "primary_same_book_two_sided": strict_report,
        "secondary_any_two_sided": broad_report,
        "limitations": [
            "Real sportsbook prices are concentrated on a small number of acquisition/slate dates, so date-level uncertainty dominates row count.",
            "The historical universe is retrospective infrastructure; only columns plausibly available by the frozen prediction/quote time are used as PCA features.",
            "Secondary any-two-sided analysis may de-vig quotes sourced from different books and is therefore diagnostic only.",
            "Research result only; no production or wagering authority.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    strict_out = strict_rows.copy()
    if not strict_out.empty:
        strict_out["analysis"] = "primary_same_book_two_sided"
    broad_out = broad_rows.copy()
    if not broad_out.empty:
        broad_out["analysis"] = "secondary_any_two_sided"
    pd.concat([strict_out, broad_out], ignore_index=True).to_csv(output_csv, index=False)
    print("HISTORICAL_PCA_ASTAR_REPORT=" + json.dumps(report, sort_keys=True, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()
    run(args.input, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
