from __future__ import annotations

"""Integrity wrapper for the historical-universe PCA+A* backtest.

This keeps the original backtest implementation auditable while replacing the
three pieces that could inflate evidence through alternate-line duplication:
book-key cleaning, graph neighbors, and local residual support.  Every search
neighbor/support observation must come from a distinct historical player-game.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "backtest_historical_universe.py"
spec = importlib.util.spec_from_file_location("pca_astar_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def clean_book(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return "" if text in {"", "nan", "none", "null"} else text


def identity_key(frame: pd.DataFrame, idx: int) -> tuple[str, str, str]:
    row = frame.iloc[int(idx)]
    return (
        str(pd.Timestamp(row["Game_Date"]).date()),
        str(row["Game_ID"]),
        str(row["Player_ID"]),
    )


def unique_neighbor_indices(
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    frame: pd.DataFrame,
    max_n: int,
    *,
    exclude_identity: tuple[str, str, str] | None = None,
) -> list[int]:
    order = np.argsort(candidate_distances)
    seen: set[tuple[str, str, str]] = set()
    out: list[int] = []
    for pos in order:
        idx = int(candidate_indices[int(pos)])
        key = identity_key(frame, idx)
        if exclude_identity is not None and key == exclude_identity:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(idx)
        if len(out) >= max_n:
            break
    return out


def build_graph(z: np.ndarray, labels: list[tuple[str, str]], k: int):
    """kNN graph with no edge to another alternate line from the same player-game."""
    n = len(z)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    nearest = np.full(n, np.nan)
    partition_median_edge: dict[tuple[str, str], float] = {}
    # The current caller passes the training frame through a temporary module global.
    train = _TRAIN_FRAME
    for label in sorted(set(labels)):
        ids = np.array([i for i, x in enumerate(labels) if x == label], dtype=int)
        if len(ids) < 2:
            continue
        d = base._pairwise(z[ids])
        np.fill_diagonal(d, np.inf)
        edge_costs: list[float] = []
        for a, i in enumerate(ids):
            own = identity_key(train, int(i))
            neighbors = unique_neighbor_indices(ids, d[a], train, k, exclude_identity=own)
            if not neighbors:
                continue
            dist_map = {int(ids[b]): float(d[a, b]) for b in range(len(ids))}
            valid = [(j, dist_map[j]) for j in neighbors if np.isfinite(dist_map.get(j, math.inf)) and dist_map[j] > 0]
            if not valid:
                continue
            nearest[int(i)] = min(c for _, c in valid)
            adj[int(i)].extend(valid)
            edge_costs.extend(c for _, c in valid)
        if edge_costs:
            partition_median_edge[label] = float(np.median(edge_costs))
    return adj, nearest, partition_median_edge


def local_training_residuals(z: np.ndarray, train: pd.DataFrame, labels: list[tuple[str, str]], cfg):
    y = train.outcome.to_numpy(float)
    market = train.no_vig_market_probability.to_numpy(float)
    residual = np.full(len(train), np.nan)
    support = np.zeros(len(train), dtype=int)
    for label in sorted(set(labels)):
        ids = np.array([i for i, x in enumerate(labels) if x == label], dtype=int)
        if len(ids) < 2:
            continue
        d = base._pairwise(z[ids])
        np.fill_diagonal(d, np.inf)
        for a, i in enumerate(ids):
            own = identity_key(train, int(i))
            nbr = unique_neighbor_indices(ids, d[a], train, cfg.local_k, exclude_identity=own)
            nbr = [j for j in nbr if np.isfinite(y[j]) and np.isfinite(market[j])]
            if not nbr:
                continue
            yy = y[nbr]
            mm = market[nbr]
            market_mean = float(mm.mean())
            p = (float(yy.sum()) + cfg.shrinkage * market_mean) / (len(yy) + cfg.shrinkage)
            se = math.sqrt(max(1e-12, p * (1.0 - p) / (len(yy) + cfg.shrinkage)))
            residual[int(i)] = p - market_mean - cfg.z_confidence * se
            support[int(i)] = len(yy)
    return residual, support


def score_candidate(row, zq, train, ztrain, labels, adj, nearest, med_edge, residual, support, L, cfg):
    label = (str(row.Target), str(row.side))
    ids = np.array([i for i, x in enumerate(labels) if x == label], dtype=int)
    if len(ids) < cfg.min_support:
        return {"eligible": False, "reason": "PARTITION_SUPPORT"}

    distances = np.sqrt(np.sum((ztrain[ids] - zq) ** 2, axis=1))
    independent = unique_neighbor_indices(ids, distances, train, max(cfg.local_k, 1))
    if len(independent) < cfg.min_support:
        return {"eligible": False, "reason": "INDEPENDENT_SUPPORT", "support": len(independent)}

    dist_by_idx = {int(idx): float(dist) for idx, dist in zip(ids, distances)}
    start = int(independent[0])
    nearest_distance = dist_by_idx[start]

    partition_nearest = nearest[ids]
    partition_nearest = partition_nearest[np.isfinite(partition_nearest)]
    ood_limit = float(np.quantile(partition_nearest, cfg.ood_quantile)) if len(partition_nearest) else math.inf
    if nearest_distance > ood_limit:
        return {"eligible": False, "reason": "OOD", "nearest_distance": nearest_distance, "ood_limit": ood_limit}

    nbr = independent[: cfg.local_k]
    yy = train.iloc[nbr].outcome.to_numpy(float)
    good = np.isfinite(yy)
    nbr = [j for j, ok in zip(nbr, good) if ok]
    yy = yy[good]
    if len(yy) < cfg.min_support:
        return {"eligible": False, "reason": "LOCAL_SUPPORT", "support": int(len(yy))}

    market_p = float(row.no_vig_market_probability)
    p = (float(yy.sum()) + cfg.shrinkage * market_p) / (len(yy) + cfg.shrinkage)
    se = math.sqrt(max(1e-12, p * (1.0 - p) / (len(yy) + cfg.shrinkage)))
    conservative = p - market_p - cfg.z_confidence * se

    max_cost = cfg.path_radius_multiplier * float(med_edge.get(label, 1.0))
    found, path_cost, expansions, path = base.astar_edge_basin(start, label, adj, residual, support, cfg, L, max_cost)
    selected = bool(found and conservative >= cfg.edge_threshold)
    return {
        "eligible": True,
        "selected": selected,
        "local_empirical_probability": p,
        "local_standard_error": se,
        "conservative_residual": conservative,
        "support": int(len(yy)),
        "support_unit": "unique_player_game",
        "nearest_distance": nearest_distance,
        "ood_limit": ood_limit,
        "astar_edge_found": bool(found),
        "astar_path_cost": None if not np.isfinite(path_cost) else float(path_cost),
        "astar_expansions": int(expansions),
        "astar_path_length": int(len(path)),
    }


_TRAIN_FRAME = pd.DataFrame()
_original_fit_and_score = base.fit_and_score


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, cfg):
    global _TRAIN_FRAME
    _TRAIN_FRAME = train.reset_index(drop=True)
    # Base implementation calls its module globals at runtime; all integrity-sensitive
    # functions are patched below before delegating.
    return _original_fit_and_score(_TRAIN_FRAME, test.reset_index(drop=True), cfg)


base._clean_book = clean_book
base.build_graph = build_graph
base.local_training_residuals = local_training_residuals
base.score_candidate = score_candidate
base.fit_and_score = fit_and_score


def run(input_path: Path, output_json: Path, output_csv: Path):
    cfg = base.SearchConfig()
    strict_report, strict_rows = base.run_analysis(input_path, cfg, strict_same_book=True)
    broad_report, broad_rows = base.run_analysis(input_path, cfg, strict_same_book=False)
    report = {
        "status": "DEVELOPMENT_BACKTEST_INDEPENDENT_SUPPORT",
        "semantics": {
            "astar_success": "EDGE_FOUND in a prior-data PCA residual basin; never a win label",
            "settlement": "derived after frozen selection from Actual versus Market_Line",
            "support_unit": "unique historical player-game; alternate lines from one event cannot multiply support",
        },
        "config": asdict(cfg),
        "primary_same_book_two_sided": strict_report,
        "secondary_any_two_sided": broad_report,
        "limitations": [
            "Real sportsbook prices are concentrated on a small number of acquisition/slate dates; date-level uncertainty dominates row count.",
            "Historical-universe feature provenance still requires leakage audit before certification.",
            "Secondary any-two-sided analysis may normalize quotes sourced from different books and is diagnostic only.",
            "Research result only; no production authority.",
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
    print("INDEPENDENT_SUPPORT_REPORT=" + json.dumps(report, sort_keys=True, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()
    run(args.input, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
