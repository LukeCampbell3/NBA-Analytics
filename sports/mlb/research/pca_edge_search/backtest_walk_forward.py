from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "model_hit_probability", "hit_probability", "probability", "abs_edge",
    "log_history_rows", "days_since_history", "selection_score",
    "historical_bet_profile_win_rate", "log_historical_bet_profile_support",
    "historical_market_availability_rate", "log_historical_market_availability_support",
    "books", "line", "market_break_even_probability",
]


@dataclass(frozen=True)
class Config:
    n_components: int = 6
    graph_k: int = 8
    local_k: int = 30
    min_support: int = 18
    edge_threshold: float = 0.03
    z_confidence: float = 1.0
    shrinkage: float = 18.0
    ood_quantile: float = 0.95
    path_radius_multiplier: float = 6.0


def american_break_even(x):
    try:
        o = float(x)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(o) or -100 < o < 100:
        return np.nan
    return 100 / (o + 100) if o > 0 else (-o) / ((-o) + 100)


def normalize_result(x):
    x = str(x).strip().lower()
    if x in {"win", "won", "w"}: return "win"
    if x in {"loss", "lost", "l"}: return "loss"
    if x in {"push", "p"}: return "push"
    return ""


def prepare(path: Path):
    df = pd.read_csv(path, low_memory=False)
    required = {"date", "target", "direction", "line", "side_price", "result"}
    missing = sorted(required - set(df.columns))
    if missing: raise ValueError(f"missing required columns: {missing}")
    input_rows = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["market_type"] = df["target"].astype(str).str.upper().str.strip()
    df["side"] = df["direction"].astype(str).str.upper().str.strip()
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["side_price"] = pd.to_numeric(df["side_price"], errors="coerce")
    df["market_break_even_probability"] = df["side_price"].map(american_break_even)
    df["settlement"] = df["result"].map(normalize_result)
    if "market_source" in df:
        df = df[df["market_source"].astype(str).str.lower().eq("real")]
    if "price_confirmed" in df:
        ok = df["price_confirmed"].astype(str).str.lower().isin({"1","true","t","yes","y"})
        df = df[ok]
    df = df[df.date.notna() & df.line.notna() & df.market_break_even_probability.between(.01,.99) & df.settlement.isin(["win","loss","push"])].copy()
    df["outcome"] = np.where(df.settlement.eq("win"), 1.0, np.where(df.settlement.eq("loss"), 0.0, np.nan))
    numeric = ["model_hit_probability","hit_probability","probability","abs_edge","history_rows","days_since_history","selection_score","historical_bet_profile_win_rate","historical_bet_profile_support","historical_market_availability_rate","historical_market_availability_support","books"]
    for c in numeric:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    for src, dst in [("history_rows","log_history_rows"),("historical_bet_profile_support","log_historical_bet_profile_support"),("historical_market_availability_support","log_historical_market_availability_support")]:
        if src in df: df[dst] = np.log1p(df[src].clip(lower=0))
    identity = [c for c in ["date","game_id","player_id","player","market_type","side","line","side_price"] if c in df]
    before = len(df)
    if identity: df = df.sort_values(identity).drop_duplicates(identity, keep="first")
    def units(r):
        if r.settlement == "push": return 0.0
        if r.settlement == "loss": return -1.0
        o = float(r.side_price)
        return o / 100 if o > 0 else 100 / abs(o)
    df["realized_units"] = df.apply(units, axis=1)
    df = df.sort_values("date").reset_index(drop=True)
    return df, {"input_rows": input_rows, "eligible_before_dedupe": before, "deduplicated_rows": len(df), "duplicates_removed": before-len(df), "unique_dates": int(df.date.dt.date.nunique())}


def pairwise(z):
    q = np.sum(z*z, axis=1)
    return np.sqrt(np.maximum(0, q[:,None]+q[None,:]-2*z@z.T))


def graph(z, labels, k):
    n=len(z); adj=[[] for _ in range(n)]; nearest=np.full(n,np.nan)
    for lab in sorted(set(labels)):
        ids=np.array([i for i,x in enumerate(labels) if x==lab],int)
        if len(ids)<2: continue
        d=pairwise(z[ids]); np.fill_diagonal(d,np.inf); kk=min(k,len(ids)-1)
        for a,i in enumerate(ids):
            js=np.argpartition(d[a],kk-1)[:kk]; nearest[i]=float(d[a,js].min())
            for b in js:
                if np.isfinite(d[a,b]) and d[a,b]>0: adj[i].append((int(ids[b]),float(d[a,b])))
    return adj, nearest


def local_stats(z, train, labels, cfg):
    n=len(train); residual=np.full(n,np.nan); support=np.zeros(n,int)
    y=train.outcome.to_numpy(float); m=train.market_break_even_probability.to_numpy(float)
    for i in range(n):
        ids=np.array([j for j,x in enumerate(labels) if x==labels[i] and j!=i and np.isfinite(y[j])],int)
        if not len(ids): continue
        d=np.sqrt(np.sum((z[ids]-z[i])**2,axis=1)); kk=min(cfg.local_k,len(ids)); ids=ids[np.argpartition(d,kk-1)[:kk]]
        yy=y[ids]; mm=m[ids]; good=np.isfinite(yy)&np.isfinite(mm); yy=yy[good]; mm=mm[good]; nn=len(yy)
        if not nn: continue
        mb=float(mm.mean()); p=(float(yy.sum())+cfg.shrinkage*mb)/(nn+cfg.shrinkage)
        se=math.sqrt(max(1e-12,p*(1-p)/(nn+cfg.shrinkage)))
        residual[i]=p-mb-cfg.z_confidence*se; support[i]=nn
    return residual,support


def lipschitz(adj,residual):
    L=0.0
    for i,edges in enumerate(adj):
        if not np.isfinite(residual[i]): continue
        for j,d in edges:
            if d>0 and np.isfinite(residual[j]): L=max(L,abs(float(residual[i]-residual[j]))/d)
    return max(L,1e-9)


def astar(start,adj,residual,support,cfg,L,max_cost):
    def goal(i): return support[i]>=cfg.min_support and np.isfinite(residual[i]) and residual[i]>=cfg.edge_threshold
    def h(i): return 0.0 if not np.isfinite(residual[i]) else max(0.0,cfg.edge_threshold-float(residual[i]))/L
    q=[(h(start),0.0,start)]; best={start:0.0}; exp=0
    while q:
        _,g,i=heapq.heappop(q)
        if g!=best.get(i) or g>max_cost: continue
        exp+=1
        if goal(i): return True,g,exp
        for j,c in adj[i]:
            ng=g+c
            if ng<=max_cost and ng<best.get(j,math.inf):
                best[j]=ng; heapq.heappush(q,(ng+h(j),ng,j))
    return False,math.inf,exp


def score(row,zq,train,ztrain,labels,residual,support,adj,nearest,L,cfg):
    lab=(str(row.market_type),str(row.side)); ids=np.array([i for i,x in enumerate(labels) if x==lab],int)
    if len(ids)<cfg.min_support: return {"eligible":False,"reason":"PARTITION_SUPPORT"}
    d=np.sqrt(np.sum((ztrain[ids]-zq)**2,axis=1)); order=np.argsort(d); start=int(ids[order[0]])
    finite=nearest[np.isfinite(nearest)]; ood_limit=float(np.quantile(finite,cfg.ood_quantile)) if len(finite) else math.inf
    if float(d[order[0]])>ood_limit: return {"eligible":False,"reason":"OOD"}
    nn=ids[order[:min(cfg.local_k,len(ids))]]; yy=train.iloc[nn].outcome.to_numpy(float); yy=yy[np.isfinite(yy)]
    if len(yy)<cfg.min_support: return {"eligible":False,"reason":"LOCAL_SUPPORT"}
    mp=float(row.market_break_even_probability); p=(float(yy.sum())+cfg.shrinkage*mp)/(len(yy)+cfg.shrinkage)
    se=math.sqrt(max(1e-12,p*(1-p)/(len(yy)+cfg.shrinkage))); conservative=p-mp-cfg.z_confidence*se
    edge_costs=[c for edges in adj for _,c in edges]; max_cost=cfg.path_radius_multiplier*(float(np.median(edge_costs)) if edge_costs else 1.0)
    found,path_cost,exp=astar(start,adj,residual,support,cfg,L,max_cost)
    return {"eligible":True,"selected":bool(found and conservative>0),"local_empirical_probability":p,"conservative_residual":conservative,"support":len(yy),"astar_edge_found":found,"astar_path_cost":None if not np.isfinite(path_cost) else path_cost,"astar_expansions":exp}


def split_dates(df,folds=4,initial=.5):
    dates=sorted(df.date.dt.date.unique()); start=max(10,int(len(dates)*initial)); rem=len(dates)-start
    b=[start+round(i*rem/folds) for i in range(folds+1)]
    return [(set(dates[:b[i]]),set(dates[b[i]:b[i+1]])) for i in range(folds) if b[i+1]>b[i]]


def summarize(x):
    x=x[x.settlement.isin(["win","loss"])]
    if x.empty: return {"plays":0}
    wins=int(x.settlement.eq("win").sum()); n=len(x); hr=wins/n; mp=float(x.market_break_even_probability.mean()); units=float(x.realized_units.sum())
    return {"plays":n,"wins":wins,"losses":n-wins,"hit_rate":hr,"mean_market_break_even_probability":mp,"realized_residual":hr-mp,"units":units,"roi":units/n,"dates_with_plays":int(x.date.dt.date.nunique())}


def bootstrap(x,reps=5000):
    x=x[x.settlement.isin(["win","loss"])]
    days=list(x.date.dt.date.unique())
    if len(days)<2: return {"roi_ci95":[None,None],"residual_ci95":[None,None]}
    g={d:x[x.date.dt.date.eq(d)] for d in days}; rng=np.random.default_rng(20260916); roi=[]; res=[]
    for _ in range(reps):
        s=pd.concat([g[d] for d in rng.choice(days,len(days),replace=True)],ignore_index=True); n=len(s)
        roi.append(float(s.realized_units.sum()/n)); res.append(float(s.outcome.mean()-s.market_break_even_probability.mean()))
    return {"roi_ci95":[float(v) for v in np.quantile(roi,[.025,.975])],"residual_ci95":[float(v) for v in np.quantile(res,[.025,.975])]}


def run(path,out_json,out_csv):
    df,pdiag=prepare(path); cfg=Config(); scored=[]; fdiag=[]
    for fold,(trd,ted) in enumerate(split_dates(df),1):
        tr=df[df.date.dt.date.isin(trd)].copy(); te=df[df.date.dt.date.isin(ted)].copy()
        cols=[c for c in FEATURES if c in tr and tr[c].notna().sum()>=max(10,int(.5*len(tr)))]
        med=tr[cols].median(); xtr=tr[cols].fillna(med).fillna(0); xte=te[cols].fillna(med).fillna(0)
        scaler=StandardScaler().fit(xtr); k=min(cfg.n_components,len(cols),len(tr)); pca=PCA(n_components=k,svd_solver="full").fit(scaler.transform(xtr))
        ztr=pca.transform(scaler.transform(xtr)); zte=pca.transform(scaler.transform(xte)); labels=list(zip(tr.market_type.astype(str),tr.side.astype(str)))
        adj,nearest=graph(ztr,labels,cfg.graph_k); residual,support=local_stats(ztr,tr,labels,cfg); L=lipschitz(adj,residual)
        goals=int(np.sum((support>=cfg.min_support)&np.isfinite(residual)&(residual>=cfg.edge_threshold)))
        rows=[]
        for i,(_,r) in enumerate(te.iterrows()):
            rec=r.to_dict(); rec.update(score(r,zte[i],tr,ztr,labels,residual,support,adj,nearest,L,cfg)); rec["fold"]=fold; rows.append(rec)
        sf=pd.DataFrame(rows); scored.append(sf); fdiag.append({"fold":fold,"train_rows":len(tr),"test_rows":len(te),"train_dates":len(trd),"test_dates":len(ted),"features":cols,"components":k,"edge_goal_nodes":goals,"selected":int(sf.get("selected",False).fillna(False).sum())})
    s=pd.concat(scored,ignore_index=True); eligible=s[s.get("eligible",False).fillna(False)]; selected=s[s.get("selected",False).fillna(False)]
    report={"status":"DEVELOPMENT_BACKTEST","semantics":{"astar_success":"EDGE_FOUND, not a bet win","settlement":"evaluated after frozen selection"},"market_probability":"American side_price converted to break-even probability; includes vig","config":asdict(cfg),"pipeline":pdiag,"folds":fdiag,"all_holdout":summarize(s),"eligible_holdout":summarize(eligible),"astar_selected":summarize(selected),"limitations":["walk-forward artifact contains policy-selected rows rather than the complete candidate universe, so residual selection bias remains after deduplication","one-sided break-even probability includes vig","research result only; no production authority"]}
    report["astar_selected"].update(bootstrap(selected)); out_json.parent.mkdir(parents=True,exist_ok=True); out_csv.parent.mkdir(parents=True,exist_ok=True); out_json.write_text(json.dumps(report,indent=2,default=str)+"\n"); s.to_csv(out_csv,index=False); print("BACKTEST_REPORT="+json.dumps(report,sort_keys=True,default=str))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output-json",type=Path,required=True); ap.add_argument("--output-csv",type=Path,required=True); a=ap.parse_args(); run(a.input,a.output_json,a.output_csv)


if __name__=="__main__": main()
