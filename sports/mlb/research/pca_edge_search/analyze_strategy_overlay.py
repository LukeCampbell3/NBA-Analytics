from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"plays": 0}
    x = df[df["settlement"].isin(["win", "loss"])].copy()
    if x.empty:
        return {"plays": 0}
    n = len(x)
    wins = int(x["settlement"].eq("win").sum())
    hit = wins / n
    market = float(x["no_vig_market_probability"].mean())
    units = float(x["realized_units"].sum())
    return {
        "plays": int(n), "wins": wins, "losses": int(n-wins),
        "hit_rate": hit,
        "mean_no_vig_market_probability": market,
        "realized_residual": hit-market,
        "units": units, "roi": units/n,
        "unique_dates": int(pd.to_datetime(x["Game_Date"], utc=True).dt.date.nunique()),
        "unique_games": int(x["Game_ID"].astype(str).nunique()),
        "unique_players": int(x["Player_ID"].astype(str).nunique()),
    }


def bootstrap(df: pd.DataFrame, reps: int = 10000) -> dict:
    if df.empty:
        return {"roi_ci95": [None, None], "residual_ci95": [None, None], "dates": 0}
    x = df[df["settlement"].isin(["win", "loss"])].copy()
    x["_date"] = pd.to_datetime(x["Game_Date"], utc=True).dt.date
    dates = list(x["_date"].unique())
    if len(dates) < 3:
        return {"roi_ci95": [None, None], "residual_ci95": [None, None], "dates": len(dates)}
    groups = {d: x[x["_date"].eq(d)] for d in dates}
    rng = np.random.default_rng(20260916)
    roi, residual = [], []
    for _ in range(reps):
        s = pd.concat([groups[d] for d in rng.choice(dates, len(dates), replace=True)], ignore_index=True)
        roi.append(float(s["realized_units"].sum()/len(s)))
        residual.append(float(s["outcome"].mean()-s["no_vig_market_probability"].mean()))
    return {
        "roi_ci95": [float(v) for v in np.quantile(roi, [.025,.975])],
        "residual_ci95": [float(v) for v in np.quantile(residual, [.025,.975])],
        "dates": len(dates),
    }


def one_per_player_target(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["_resid"] = pd.to_numeric(x.get("conservative_residual"), errors="coerce").fillna(-np.inf)
    x["_dist"] = pd.to_numeric(x.get("nearest_distance"), errors="coerce").fillna(np.inf)
    keys = ["Game_Date", "Game_ID", "Player_ID", "Target"]
    return x.sort_values(keys+["_resid","_dist"], ascending=[True,True,True,True,False,True]).drop_duplicates(keys).drop(columns=["_resid","_dist"])


def per_date(df: pd.DataFrame) -> list[dict]:
    if df.empty: return []
    x=df.copy(); x["_date"]=pd.to_datetime(x["Game_Date"],utc=True).dt.date
    out=[]
    for d,g in x.groupby("_date"):
        r={"date":str(d)}; r.update(summarize(g)); out.append(r)
    return out


def analyze(input_csv: Path, output_json: Path):
    df = pd.read_csv(input_csv, low_memory=False)
    primary = df[df["analysis"].eq("primary_same_book_two_sided")].copy()
    primary["eligible"] = primary["eligible"].astype(str).str.lower().isin({"true","1"})
    primary["selected"] = primary["selected"].astype(str).str.lower().isin({"true","1"})
    primary["astar_edge_found"] = primary["astar_edge_found"].astype(str).str.lower().isin({"true","1"})
    primary["conservative_residual"] = pd.to_numeric(primary["conservative_residual"], errors="coerce")
    primary["Prediction"] = pd.to_numeric(primary["Prediction"], errors="coerce")
    primary["Market_Line"] = pd.to_numeric(primary["Market_Line"], errors="coerce")
    primary["strategy_side"] = np.where(primary["Prediction"] >= primary["Market_Line"], "OVER", "UNDER")
    primary["strategy_aligned"] = primary["side"].eq(primary["strategy_side"])

    strategy = primary[primary["strategy_aligned"]].copy()
    residual_only = primary[primary["eligible"] & primary["conservative_residual"].ge(.03)].copy()
    residual_only = one_per_player_target(residual_only)
    astar_any = one_per_player_target(primary[primary["selected"]])
    aligned_residual = one_per_player_target(strategy[strategy["eligible"] & strategy["conservative_residual"].ge(.03)])
    aligned_astar = one_per_player_target(strategy[strategy["selected"]])
    aligned_astar_loose = one_per_player_target(strategy[strategy["eligible"] & strategy["astar_edge_found"] & strategy["conservative_residual"].gt(0)])

    report = {
        "semantics": "A* is tested as a confirmation overlay on the existing projection direction, not as an outcome generator.",
        "primary_same_book_holdout": summarize(primary),
        "projection_direction_strategy": summarize(strategy),
        "direct_pca_residual_only": summarize(residual_only),
        "astar_any_side": summarize(astar_any),
        "strategy_aligned_direct_residual": summarize(aligned_residual),
        "strategy_aligned_astar_strict": summarize(aligned_astar),
        "strategy_aligned_astar_positive_residual": summarize(aligned_astar_loose),
        "bootstrap": {
            "projection_direction_strategy": bootstrap(strategy),
            "strategy_aligned_direct_residual": bootstrap(aligned_residual),
            "strategy_aligned_astar_strict": bootstrap(aligned_astar),
            "strategy_aligned_astar_positive_residual": bootstrap(aligned_astar_loose),
        },
        "strategy_aligned_astar_by_date": per_date(aligned_astar),
        "counts": {
            "astar_strict_subset_of_direct_residual": int(len(aligned_astar)),
            "direct_residual_strategy_aligned": int(len(aligned_residual)),
            "astar_confirmation_rate": None if len(aligned_residual)==0 else float(len(aligned_astar)/len(aligned_residual)),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, default=str)+"\n")
    print("PCA_ASTAR_OVERLAY_REPORT="+json.dumps(report, sort_keys=True, default=str))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-csv",type=Path,required=True); ap.add_argument("--output-json",type=Path,required=True); a=ap.parse_args(); analyze(a.input_csv,a.output_json)


if __name__=="__main__": main()
