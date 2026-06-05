#!/usr/bin/env python3
"""
v9.9 Market Weakness Residual

Separates two distinct signals:
  1. model_edge_score: "My distribution disagrees with the line"
  2. market_weakness_score: "This book's price is weak relative to consensus"

The key validation question:
  Within the same model edge bucket, does higher market_weakness_score
  produce better CLV and ROI?

If yes: the system identifies weak lines beyond raw model edge.
If no: it's just repackaging model confidence.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from market_odds_quality import add_american_odds_quality, is_valid_american_odds


def _no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    o = (-over_odds / (-over_odds + 100)) if over_odds < 0 else (100 / (over_odds + 100))
    u = (-under_odds / (-under_odds + 100)) if under_odds < 0 else (100 / (under_odds + 100))
    t = o + u
    if t <= 0:
        return 0.5, 0.5
    return o / t, u / t


def _unit_profit(odds: float, won: bool) -> float:
    if won:
        return (odds / 100.0) if odds > 0 else (100.0 / abs(odds))
    return -1.0


# ─── Execution Edge ───────────────────────────────────────────────

def compute_execution_edge(snapshot_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute how much better the best book's price is vs consensus.

    execution_edge = best_available_no_vig_for_side - consensus_no_vig_for_side

    For OVER: lower book_no_vig_over = cheaper over = better execution
    For UNDER: lower book_no_vig_under = cheaper under = better execution
    """
    valid = snapshot_df[snapshot_df["is_valid_american_odds"] == True].copy()
    if valid.empty:
        predictions["execution_edge"] = 0.0
        predictions["best_book_no_vig"] = 0.5
        predictions["consensus_no_vig"] = 0.5
        return predictions

    # Compute no-vig per book row
    nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    valid["book_nv_over"], valid["book_nv_under"] = zip(*nv)

    # Consensus per prop (line-specific)
    group_keys = ["player", "market", "line", "date"]
    available_keys = [k for k in group_keys if k in valid.columns]
    if len(available_keys) < 3:
        predictions["execution_edge"] = 0.0
        return predictions

    consensus = valid.groupby(available_keys).agg(
        consensus_nv_over=("book_nv_over", "mean"),
        consensus_nv_under=("book_nv_under", "mean"),
        n_books=("book", "nunique"),
    ).reset_index()

    # For each prediction, find the best book and compute execution edge
    preds = predictions.copy()
    preds["execution_edge"] = 0.0
    preds["best_book_no_vig"] = 0.5
    preds["consensus_no_vig"] = 0.5
    preds["n_books_available"] = 0

    for idx in preds.index:
        player = preds.loc[idx, "player"]
        market = preds.loc[idx, "market"]
        date = preds.loc[idx, "date"]
        side = preds.loc[idx, "selected_side"]

        # Find matching books
        mask = (valid["player"] == player) & (valid["market"] == market) & (valid["date"] == date)
        books = valid.loc[mask]
        if len(books) < 2:
            continue

        cons_over = books["book_nv_over"].mean()
        cons_under = books["book_nv_under"].mean()
        preds.loc[idx, "consensus_no_vig"] = cons_over if side == "OVER" else cons_under
        preds.loc[idx, "n_books_available"] = int(books["book"].nunique())

        if side == "OVER":
            # Best book for OVER = lowest no-vig over (cheapest price)
            best_nv = books["book_nv_over"].min()
            preds.loc[idx, "best_book_no_vig"] = best_nv
            preds.loc[idx, "execution_edge"] = cons_over - best_nv  # positive = book is cheaper
        else:
            best_nv = books["book_nv_under"].min()
            preds.loc[idx, "best_book_no_vig"] = best_nv
            preds.loc[idx, "execution_edge"] = cons_under - best_nv

    return preds


# ─── Market Weakness Score ────────────────────────────────────────

def compute_market_weakness_score(preds: pd.DataFrame) -> pd.DataFrame:
    """Compute market weakness score INDEPENDENT of model edge.

    This answers: "Given that the model likes this side, is this
    specific book/price unusually exploitable?"
    """
    out = preds.copy()

    # Execution edge normalized (0 to 1)
    out["execution_edge_normalized"] = np.clip(out["execution_edge"] / 0.03, 0, 1)

    # Stale line score (from weak_line_detector if available)
    if "stale_line_score" not in out.columns:
        out["stale_line_score"] = 0.0

    # Velocity alignment (from weak_line_detector if available)
    if "velocity_alignment_normalized" not in out.columns:
        out["velocity_alignment_normalized"] = 0.5

    # Market weakness score — deliberately excludes model_edge
    out["market_weakness_score"] = (
        0.45 * out["execution_edge_normalized"]
        + 0.25 * out["velocity_alignment_normalized"]
        + 0.15 * out["stale_line_score"]
        + 0.15 * np.clip(out["n_books_available"] / 6.0, 0, 1)  # more books = more reliable consensus
    ).clip(0, 1)

    return out


# ─── Combined Score ───────────────────────────────────────────────

# ─── Within-Edge-Bucket Validation ────────────────────────────────

def validate_within_edge_buckets(df: pd.DataFrame) -> dict:
    """The key test: does market_weakness_score add value WITHIN the same edge bucket?"""
    edge_bins = [0.0, 0.05, 0.10, 0.15, 0.20, 1.0]
    edge_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    df["edge_bucket"] = pd.cut(df["model_edge"], bins=edge_bins, labels=edge_labels).astype(str)

    results = []
    for edge_bucket in edge_labels:
        bucket = df[df["edge_bucket"] == edge_bucket]
        if len(bucket) < 50:
            continue

        # Split by market weakness: above/below median within this edge bucket
        mw_median = bucket["market_weakness_score"].median()
        high_mw = bucket[bucket["market_weakness_score"] > mw_median]
        low_mw = bucket[bucket["market_weakness_score"] <= mw_median]

        if len(high_mw) < 20 or len(low_mw) < 20:
            continue

        results.append({
            "edge_bucket": edge_bucket,
            "total_rows": int(len(bucket)),
            "high_market_weakness": {
                "rows": int(len(high_mw)),
                "hit_rate": float(high_mw["hit"].mean()),
                "roi": float(high_mw["unit_profit"].mean()),
                "brier": float(high_mw["brier"].mean()),
                "mean_mw_score": float(high_mw["market_weakness_score"].mean()),
            },
            "low_market_weakness": {
                "rows": int(len(low_mw)),
                "hit_rate": float(low_mw["hit"].mean()),
                "roi": float(low_mw["unit_profit"].mean()),
                "brier": float(low_mw["brier"].mean()),
                "mean_mw_score": float(low_mw["market_weakness_score"].mean()),
            },
            "high_beats_low_roi": float(high_mw["unit_profit"].mean()) > float(low_mw["unit_profit"].mean()),
            "high_beats_low_hit": float(high_mw["hit"].mean()) > float(low_mw["hit"].mean()),
            "roi_improvement": float(high_mw["unit_profit"].mean()) - float(low_mw["unit_profit"].mean()),
        })

    beats_count = sum(1 for r in results if r["high_beats_low_roi"])
    total = len(results)

    return {
        "edge_buckets_tested": total,
        "high_mw_beats_low_mw_rate": beats_count / total if total > 0 else 0.0,
        "results": results,
        "verdict": "market_weakness_adds_value" if beats_count / max(total, 1) > 0.5 else "market_weakness_not_proven",
    }


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("V9.9 MARKET WEAKNESS RESIDUAL ANALYSIS")
    print("=" * 70)

    # Load live snapshot data (multi-book)
    snapshot_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "collected_book_snapshots.csv"
    predictions_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "model_slate_for_clv.csv"
    attachable_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "market_snapshot_attachable.csv"

    if not predictions_path.exists():
        print("ERROR: Run build_clv_slate_from_live_odds.py first")
        return

    predictions = pd.read_csv(predictions_path)
    snapshot_df = pd.read_csv(snapshot_path)
    snapshot_df = add_american_odds_quality(snapshot_df)

    print(f"\nInput: {len(predictions)} predictions, {len(snapshot_df)} book snapshots")
    print(f"  Books in snapshots: {snapshot_df['book'].nunique()}")

    # Compute model edge and side selection
    predictions["p_model_over"] = predictions["p_model_over"].clip(0.01, 0.99)
    predictions["p_model_under"] = predictions["p_model_under"].clip(0.01, 0.99)

    # Get consensus from snapshots for edge computation
    valid_snaps = snapshot_df[snapshot_df["is_valid_american_odds"] == True]
    snap_consensus = valid_snaps.groupby(["player", "market", "date"]).apply(
        lambda g: pd.Series({
            "snap_consensus_over": g.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"])[0], axis=1).mean(),
        })
    ).reset_index()

    predictions = predictions.merge(snap_consensus, on=["player", "market", "date"], how="left")
    predictions["snap_consensus_over"] = predictions["snap_consensus_over"].fillna(0.5)
    predictions["model_edge_over"] = predictions["p_model_over"] - predictions["snap_consensus_over"]
    predictions["model_edge_under"] = predictions["p_model_under"] - (1 - predictions["snap_consensus_over"])
    predictions["selected_side"] = np.where(predictions["model_edge_over"] >= predictions["model_edge_under"], "OVER", "UNDER")
    predictions["model_edge"] = np.maximum(predictions["model_edge_over"], predictions["model_edge_under"])

    # Compute execution edge (book-vs-consensus)
    print("\nComputing execution edge...")
    predictions = compute_execution_edge(snapshot_df, predictions)
    print(f"  Mean execution edge: {predictions['execution_edge'].mean():.4f}")
    print(f"  Rows with execution edge > 0: {(predictions['execution_edge'] > 0).sum()}")
    print(f"  Mean books available: {predictions['n_books_available'].mean():.1f}")

    # Compute market weakness score and two-stage gate
    predictions = compute_market_weakness_score(predictions)
    # Two-stage gate is now inside compute_market_weakness_score
    # Compute model_edge_score and final_play_score inline
    predictions["model_edge_score"] = np.clip(predictions["model_edge"] / 0.15, 0, 1)
    predictions["final_play_score"] = (
        0.50 * predictions["model_edge_score"]
        + 0.35 * predictions["market_weakness_score"]
        + 0.15 * 0.5
    ).clip(0, 1)
    uncertainty = predictions.get("belief_uncertainty", pd.Series(0.5, index=predictions.index)).fillna(0.5)
    predictions["risk_penalty"] = np.clip(uncertainty * 0.25, 0, 0.25)
    predictions["final_play_score_adjusted"] = (predictions["final_play_score"] - predictions["risk_penalty"]).clip(0, 1)

    # Two-stage tier assignment
    min_model_edge = 0.06
    model_passes = predictions["model_edge"] >= min_model_edge
    strong_mw = predictions["market_weakness_score"] >= 0.50
    very_strong_edge = predictions["model_edge"] >= 0.18
    predictions["reliability_tier"] = "no_action"
    predictions.loc[model_passes & strong_mw, "reliability_tier"] = "shadow"
    predictions.loc[model_passes & ~strong_mw & very_strong_edge, "reliability_tier"] = "model_edge_monitor"
    predictions.loc[model_passes & ~strong_mw & ~very_strong_edge, "reliability_tier"] = "monitor"

    print(f"\n  Model edge score (mean):       {predictions['model_edge_score'].mean():.3f}")
    print(f"  Market weakness score (mean):  {predictions['market_weakness_score'].mean():.3f}")
    print(f"  Final play score (mean):       {predictions['final_play_score_adjusted'].mean():.3f}")
    print(f"  Correlation(edge, mw):         {predictions['model_edge_score'].corr(predictions['market_weakness_score']):.3f}")

    # ── Attach CLV for validation ──
    print(f"\n{'=' * 70}")
    print("CLV VALIDATION (within-edge-bucket test)")
    print(f"{'=' * 70}")

    att = pd.read_csv(attachable_path)
    att = add_american_odds_quality(att)
    true_clv = att[att.get("close_status", pd.Series()) == "true_sequence_close"].copy()

    # Join predictions to CLV
    true_clv["player_norm"] = true_clv["player"].str.replace("_", " ").str.lower().str.strip()
    predictions["player_norm"] = predictions["player"].str.replace("_", " ").str.lower().str.strip()
    true_clv["date"] = pd.to_datetime(true_clv["date"], errors="coerce").dt.date.astype(str)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.date.astype(str)

    merged = true_clv.merge(
        predictions[["player_norm", "market", "date", "model_edge", "market_weakness_score",
                     "execution_edge", "selected_side", "final_play_score_adjusted",
                     "model_edge_score", "p_model_over", "p_model_under"]],
        on=["player_norm", "market", "date"],
        how="inner",
    )

    if len(merged) == 0:
        print("  No CLV matches (expected if dates don't overlap)")
        print("  Using historical backtest for within-edge validation instead...")

        # Fall back to historical data for the within-edge test
        hist_path = ROOT / "model" / "props" / "v9_5_prelock_availability_w050" / "data" / "prop_training_rows.csv"
        hist = pd.read_csv(hist_path, low_memory=False)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.dropna(subset=["date", "result_over", "model_mean", "sigma", "line"]).copy()
        valid_odds = hist["over_odds"].apply(is_valid_american_odds) & hist["under_odds"].apply(is_valid_american_odds)
        hist = hist[valid_odds].copy()
        hist["game_key"] = hist["date"].dt.date.astype(str) + "|" + hist["player"] + "|" + hist["market"]
        hist = hist.drop_duplicates(subset="game_key", keep="first").copy()

        # Compute features on historical
        hist["p_model_over"] = hist["p_over_raw"].clip(0.01, 0.99)
        hist["p_model_under"] = 1.0 - hist["p_model_over"]
        nv = hist.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
        hist["mkt_nv_over"], hist["mkt_nv_under"] = zip(*nv)
        hist["model_edge_over"] = hist["p_model_over"] - hist["mkt_nv_over"]
        hist["model_edge_under"] = hist["p_model_under"] - hist["mkt_nv_under"]
        hist["selected_side"] = np.where(hist["model_edge_over"] >= hist["model_edge_under"], "OVER", "UNDER")
        hist["model_edge"] = np.maximum(hist["model_edge_over"], hist["model_edge_under"])
        hist["selected_outcome"] = np.where(hist["selected_side"] == "OVER", hist["result_over"], 1 - hist["result_over"])
        hist["hit"] = (hist["selected_outcome"] > 0.5).astype(int)
        hist["selected_odds"] = np.where(hist["selected_side"] == "OVER", hist["over_odds"], hist["under_odds"])
        hist["unit_profit"] = hist.apply(lambda r: _unit_profit(r["selected_odds"], r["hit"] == 1), axis=1)
        hist["p_selected"] = np.where(hist["selected_side"] == "OVER", hist["p_model_over"], hist["p_model_under"])
        hist["brier"] = (hist["p_selected"] - hist["selected_outcome"]) ** 2

        # Market weakness proxy for historical: use odds deviation from -110/-110 as proxy
        # (since we only have one book, use the odds asymmetry as a weakness signal)
        hist["odds_asymmetry"] = (hist["over_odds"] - hist["under_odds"]).abs()
        hist["market_weakness_score"] = np.clip(hist["odds_asymmetry"] / 30.0, 0, 1)

        print(f"\n  Historical rows for within-edge test: {len(hist)}")
        result = validate_within_edge_buckets(hist)
    else:
        # Compute CLV on merged data
        entry_valid = merged["over_odds"].apply(is_valid_american_odds) & merged["under_odds"].apply(is_valid_american_odds)
        close_valid = merged["close_over_odds"].apply(is_valid_american_odds) & merged["close_under_odds"].apply(is_valid_american_odds)
        both = entry_valid & close_valid
        valid = merged.loc[both].copy()

        if len(valid) > 0:
            entry_nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
            close_nv = valid.apply(lambda r: _no_vig(r["close_over_odds"], r["close_under_odds"]), axis=1)
            valid["entry_nv_over"], valid["entry_nv_under"] = zip(*entry_nv)
            valid["close_nv_over"], valid["close_nv_under"] = zip(*close_nv)
            valid["clv_over"] = valid["close_nv_over"] - valid["entry_nv_over"]
            valid["clv_under"] = valid["close_nv_under"] - valid["entry_nv_under"]
            valid["model_side_clv"] = np.where(valid["selected_side"] == "OVER", valid["clv_over"], valid["clv_under"])
            valid["hit"] = (valid["model_side_clv"] > 0).astype(int)
            valid["unit_profit"] = valid["model_side_clv"]  # Use CLV as profit proxy
            valid["brier"] = 0.25  # placeholder
            result = validate_within_edge_buckets(valid)
        else:
            result = {"edge_buckets_tested": 0, "verdict": "no_data"}

    # Print results
    print(f"\n  Edge buckets tested: {result['edge_buckets_tested']}")
    print(f"  High MW beats low MW rate: {result.get('high_mw_beats_low_mw_rate', 0):.0%}")
    print(f"  Verdict: {result['verdict']}")

    if result.get("results"):
        print(f"\n  {'Edge Bucket':<10s} {'High MW ROI':>11s} {'Low MW ROI':>11s} {'Improvement':>12s} {'Beats'}")
        print(f"  {'-'*55}")
        for r in result["results"]:
            h = r["high_market_weakness"]
            l = r["low_market_weakness"]
            beats = "Y" if r["high_beats_low_roi"] else "N"
            print(f"  {r['edge_bucket']:<10s} {h['roi']:>+10.3f} {l['roi']:>+10.3f} {r['roi_improvement']:>+11.3f} {beats:>5s}")

    # Save report
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": int(len(predictions)),
        "mean_execution_edge": float(predictions["execution_edge"].mean()),
        "mean_market_weakness_score": float(predictions["market_weakness_score"].mean()),
        "edge_mw_correlation": float(predictions["model_edge_score"].corr(predictions["market_weakness_score"])),
        "within_edge_validation": result,
    }
    out_path = ROOT / "model" / "props" / "v9_6" / "validation" / "market_weakness_residual_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report: {out_path}")


if __name__ == "__main__":
    main()
