from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_impl(repo: Path):
    p = repo / "sports/mlb/research/pca_edge_search/pca_edge_search.py"
    spec = importlib.util.spec_from_file_location("pca_edge_search_impl", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def num(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def implied(american: float) -> float:
    if american < 0:
        return -american / (-american + 100.0)
    if american > 0:
        return 100.0 / (american + 100.0)
    raise ValueError("zero American odds")


def payout(result: str, american: float) -> float:
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    if result != "win":
        raise ValueError(result)
    return 100.0 / abs(american) if american < 0 else american / 100.0


def canonical(raw: dict[str, str]) -> dict[str, Any] | None:
    if str(raw.get("market_source", "")).lower() != "real":
        return None
    if str(raw.get("price_confirmed", "")).lower() not in {"true", "1", "yes"}:
        return None
    result = str(raw.get("result", "")).lower()
    if result not in {"win", "loss", "push"}:
        return None
    odds = num(raw, "side_price")
    target = str(raw.get("target", "")).upper()
    side = str(raw.get("direction", "")).upper()
    date = str(raw.get("date", ""))
    fields = [
        "line", "history_rows", "days_since_history",
        "historical_bet_profile_win_rate", "historical_bet_profile_support",
        "historical_market_availability_rate", "historical_market_availability_support", "books",
    ]
    values = {k: num(raw, k) for k in fields}
    if odds in (None, 0) or not target or side not in {"OVER", "UNDER"} or not date:
        return None
    if any(v is None for v in values.values()):
        return None
    p = implied(float(odds))
    return {
        "policy": str(raw.get("policy", "")), "date": date,
        "player": str(raw.get("player", "")), "player_id": str(raw.get("player_id", "")),
        "game_id": str(raw.get("game_id", "")), "market_type": target, "side": side,
        **{k: float(v) for k, v in values.items()},
        "american_odds": float(odds), "market_probability": p,
        "price_implied_probability": p, "result": result,
        "settlement": {"win": "won", "loss": "lost", "push": "push"}[result],
        "settled_hit": 1.0 if result == "win" else (0.0 if result == "loss" else None),
        "unit_return": payout(result, float(odds)),
    }


def key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r["date"], r["game_id"], r["player_id"], r["market_type"], r["side"], round(r["line"], 6), round(r["american_odds"], 6))


def load_rows(path: Path):
    raw_n = eligible_n = 0
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            raw_n += 1
            r = canonical(raw)
            if r is not None:
                eligible_n += 1
                groups[key(r)].append(r)
    rows = []
    disagreement = 0
    check = ["result", "history_rows", "days_since_history", "historical_bet_profile_win_rate", "historical_bet_profile_support", "historical_market_availability_rate", "historical_market_availability_support", "books"]
    for g in groups.values():
        g.sort(key=lambda x: x["policy"])
        base = g[0]
        if any(any(base[k] != x[k] for k in check) for x in g[1:]):
            disagreement += 1
        rows.append(base)
    rows.sort(key=lambda r: (r["date"], r["game_id"], r["player_id"], r["market_type"], r["side"], r["line"]))
    return rows, {"raw_rows": raw_n, "eligible_before_dedupe": eligible_n, "deduped_opportunities": len(rows), "duplicates_removed": eligible_n - len(rows), "duplicate_groups_with_feature_disagreement": disagreement}


def split_dates(rows: list[dict[str, Any]], frac: float):
    dates = sorted({r["date"] for r in rows})
    cut = max(1, min(len(dates) - 1, int(len(dates) * frac)))
    left_dates = set(dates[:cut])
    left = [r for r in rows if r["date"] in left_dates]
    right = [r for r in rows if r["date"] not in left_dates]
    return left, right, {"total_dates": len(dates), "left_dates": cut, "right_dates": len(dates) - cut, "left_start": dates[0], "left_end": dates[cut - 1], "right_start": dates[cut], "right_end": dates[-1]}


def wilson(k: int, n: int):
    if not n:
        return [None, None]
    z = 1.96; p = k / n; den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [center-half, center+half]


def metrics(rows: list[dict[str, Any]], idx: Iterable[int] | None = None):
    selected = rows if idx is None else [rows[i] for i in idx]
    wins = sum(r["result"] == "win" for r in selected); losses = sum(r["result"] == "loss" for r in selected); pushes = sum(r["result"] == "push" for r in selected)
    graded = [r for r in selected if r["result"] in {"win", "loss"}]
    hit = wins / len(graded) if graded else None
    market = float(np.mean([r["price_implied_probability"] for r in graded])) if graded else None
    residual = hit - market if hit is not None and market is not None else None
    roi = sum(r["unit_return"] for r in selected) / len(selected) if selected else None
    units = float(sum(r["unit_return"] for r in selected))
    brier = float(np.mean([(r["price_implied_probability"] - (1.0 if r["result"] == "win" else 0.0))**2 for r in graded])) if graded else None
    dates = {r["date"] for r in selected}; players = Counter(r["player_id"] for r in selected); ms = Counter((r["market_type"], r["side"]) for r in selected)
    return {"selected": len(selected), "graded": len(graded), "wins": wins, "losses": losses, "pushes": pushes, "hit_rate": hit, "hit_rate_wilson95": wilson(wins, len(graded)), "mean_price_implied_probability": market, "realized_residual": residual, "market_brier_score": brier, "net_units": units, "roi_per_play": roi, "unique_dates": len(dates), "unique_players": len(players), "max_player_share": max(players.values())/len(selected) if selected and players else None, "by_market_side": {f"{m}|{s}": n for (m,s),n in sorted(ms.items())}}


def bootstrap(rows: list[dict[str, Any]], idx: list[int], reps: int = 4000):
    if not idx:
        return {"clusters": 0, "residual_ci95": [None, None], "roi_ci95": [None, None]}
    by_date: dict[str, list[int]] = defaultdict(list)
    for i in idx: by_date[rows[i]["date"]].append(i)
    dates = sorted(by_date); rng = np.random.default_rng(20260916); residuals=[]; rois=[]
    for _ in range(reps):
        sample=[]
        for d in rng.choice(dates, size=len(dates), replace=True): sample.extend(rows[i] for i in by_date[str(d)])
        graded=[r for r in sample if r["result"] in {"win","loss"}]
        if graded:
            hit=sum(r["result"]=="win" for r in graded)/len(graded); mp=float(np.mean([r["price_implied_probability"] for r in graded])); residuals.append(hit-mp)
        if sample: rois.append(sum(r["unit_return"] for r in sample)/len(sample))
    ci=lambda x: [float(v) for v in np.quantile(x,[.025,.975])] if x else [None,None]
    return {"clusters": len(dates), "reps": reps, "residual_ci95": ci(residuals), "roi_ci95": ci(rois)}


def fit(mod, rows: list[dict[str, Any]], features: list[str]):
    train=[r for r in rows if r["result"] in {"win","loss"}]
    cfg=mod.PCAConfig(n_components=min(4,len(features)), neighbors=24, graph_neighbors=8, prior_strength=12.0, confidence_z=1.645, minimum_effective_support=20.0, edge_threshold=0.0, max_ood_distance=3.0)
    return mod.PCAStrategyEdgeSearch(cfg).fit(train, feature_names=features), len(train)


def nearest(model, row, z):
    ok=np.asarray([m==row["market_type"] and s==row["side"] for m,s in zip(model.training_market_type_,model.training_side_)])
    cand=np.flatnonzero(ok)
    if len(cand)==0: return None, math.inf
    d=np.linalg.norm(model.training_z_[cand]-z,axis=1); j=int(np.argmin(d)); return int(cand[j]), float(d[j])


def path_scores(model, rows: list[dict[str, Any]], threshold: float):
    model.config=replace(model.config, edge_threshold=float(threshold))
    est=model.score_current(rows); Z=model.transform(rows); cache={}; costs={}; direct=[]; out_support=0; exp=[]; plen=[]
    for i,(r,e,z) in enumerate(zip(rows,est,Z)):
        if not e.in_support:
            out_support += 1; continue
        if e.conservative_residual >= threshold: direct.append(i)
        anchor,entry=nearest(model,r,z)
        if anchor is None: continue
        if anchor not in cache: cache[anchor]=model.search_training_graph(anchor)
        result=cache[anchor]
        if result.status=="EDGE_FOUND" and result.path_cost is not None:
            costs[i]=entry+float(result.path_cost); exp.append(result.expansions); plen.append(len(result.path))
    return direct,costs,{"threshold":threshold,"out_of_support":out_support,"unique_anchor_searches":len(cache),"mean_expansions_for_found_paths":float(np.mean(exp)) if exp else None,"mean_path_length":float(np.mean(plen)) if plen else None}


def tune(mod, train, val, features):
    model,nfit=fit(mod,train,features); trials=[]; best=None
    for threshold in [0.0,0.01,0.02,0.03,0.04]:
        direct,costs,info=path_scores(model,val,threshold)
        for radius in [0.35,0.5,0.75,1.0,1.5,2.0,3.0]:
            astar=[i for i,c in costs.items() if c<=radius]; mm=metrics(val,astar); n=mm["graded"]
            if n:
                p=mm["hit_rate"]; lcb=mm["realized_residual"]-1.645*math.sqrt(max(p*(1-p),1e-9)/n)
            else: lcb=-math.inf
            feasible=mm["selected"]>=25 and mm["unique_dates"]>=8; score=lcb if feasible else -math.inf
            trial={"edge_threshold":threshold,"max_path_cost":radius,"feasible":feasible,"objective_residual_lcb":score if math.isfinite(score) else None,"astar":mm,"direct":metrics(val,direct)}; trials.append(trial)
            rank=(score,mm["roi_per_play"] if mm["roi_per_play"] is not None else -math.inf,mm["selected"])
            if best is None or rank>best[0]: best=(rank,trial)
    chosen={"edge_threshold":0.0,"max_path_cost":0.75,"fallback":True} if best is None or best[1]["objective_residual_lcb"] is None else {"edge_threshold":best[1]["edge_threshold"],"max_path_cost":best[1]["max_path_cost"],"validation_objective_residual_lcb":best[1]["objective_residual_lcb"],"fallback":False}
    return {"fit_rows":nfit,"chosen":chosen,"trials":trials}


def run_variant(mod,name,features,development,holdout):
    inner_train,inner_val,inner_split=split_dates(development,.75); tuning=tune(mod,inner_train,inner_val,features); chosen=tuning["chosen"]
    model,nfit=fit(mod,development,features); direct,costs,search=path_scores(model,holdout,float(chosen["edge_threshold"])); astar=[i for i,c in costs.items() if c<=float(chosen["max_path_cost"])]
    search.update({"max_path_cost":chosen["max_path_cost"],"direct_count":len(direct),"astar_count":len(astar),"astar_only_count":len(set(astar)-set(direct)),"overlap_count":len(set(astar)&set(direct)),"mean_total_path_cost":float(np.mean([costs[i] for i in astar])) if astar else None,"median_total_path_cost":float(np.median([costs[i] for i in astar])) if astar else None})
    examples=[{k:holdout[i][k] for k in ["date","player","market_type","side","line","american_odds","price_implied_probability","result"]}|{"astar_path_cost":costs[i]} for i in astar[:25]]
    return {"name":name,"features":features,"inner_split":inner_split,"tuning":tuning,"chosen":chosen,"final_fit_rows":nfit,"holdout":{"direct_pca_edge":metrics(holdout,direct),"direct_bootstrap":bootstrap(holdout,direct),"astar_edge":metrics(holdout,astar),"astar_bootstrap":bootstrap(holdout,astar),"search":search,"examples":examples}}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo-root",type=Path,default=Path.cwd()); ap.add_argument("--input",type=Path,default=Path("sports/mlb/data/predictions/backtests/mlb_walk_forward_backtest_rows.csv")); ap.add_argument("--output",type=Path,default=Path("sports/mlb/data/predictions/backtests/pca_astar_edge_backtest.json")); a=ap.parse_args()
    repo=a.repo_root.resolve(); inp=a.input if a.input.is_absolute() else repo/a.input; out=a.output if a.output.is_absolute() else repo/a.output; mod=load_impl(repo)
    rows,audit=load_rows(inp); development,holdout,outer=split_dates(rows,.75)
    variants=[("support_market_structure",["line","history_rows","days_since_history","historical_market_availability_rate","historical_market_availability_support","books"]),("support_plus_prior_strategy_history",["line","history_rows","days_since_history","historical_market_availability_rate","historical_market_availability_support","books","historical_bet_profile_win_rate","historical_bet_profile_support"])]
    result={"schema_version":1,"method":"locked_chronological_pca_astar_market_residual_backtest","source":str(a.input),"market_probability_definition":"one_sided_price_implied_break_even_probability_from_side_price_not_de_vig","audit":audit,"outer_split":outer,"development_rows":len(development),"holdout_rows":len(holdout),"holdout_baseline":metrics(holdout),"variants":[run_variant(mod,n,f,development,holdout) for n,f in variants],"limitations":["Recorded row corpus contains opportunities emitted by historical policies, not every quote in the sportsbook universe.","Only the selected-side price is preserved; market probability is therefore price-implied break-even probability including vig, not two-sided de-vig probability.","Historical profile/availability features inherit the repository walk-forward generator's prior-date construction and are not reconstructed here from the 53MB universe.","Hyperparameters are selected only on an inner chronological validation split; the final 25% of dates stay locked until final evaluation.","Historical backtest performance does not establish prospective profitability."]}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"audit":audit,"outer_split":outer,"holdout_baseline":result["holdout_baseline"],"variants":[{"name":v["name"],"chosen":v["chosen"],"direct":v["holdout"]["direct_pca_edge"],"astar":v["holdout"]["astar_edge"],"astar_bootstrap":v["holdout"]["astar_bootstrap"],"search":v["holdout"]["search"]} for v in result["variants"]]},indent=2))

if __name__=="__main__": main()
