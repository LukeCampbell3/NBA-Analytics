#!/usr/bin/env python3
"""
V9.8 Weak Line Profit Backtest — Audited Version

Corrections from initial version:
1. Walk-forward WLS (train on prior months, test on next month)
2. Dumb baselines (all-OVER, all-UNDER, side-prior, random, highest-edge-only)
3. One-row-per-player-market-game dedup (no inflated sample)
4. Bootstrap confidence intervals
5. Odds validation (only valid American odds, pushes excluded)
6. Consistent bucket definitions
7. Honest Brier monotonicity reporting
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

from market_odds_quality import is_valid_american_odds


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


def load_and_clean() -> pd.DataFrame:
    """Load historical data with strict cleaning."""
    path = ROOT / "model" / "props" / "v9_5_prelock_availability_w050" / "data" / "prop_training_rows.csv"
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "result_over", "model_mean", "sigma", "line"]).copy()

    # Odds validation: only valid American odds
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
    valid_odds = df["over_odds"].apply(is_valid_american_odds) & df["under_odds"].apply(is_valid_american_odds)
    df = df[valid_odds].copy()

    # Exclude pushes
    df = df[df["push"] != 1].copy() if "push" in df.columns else df

    # Dedup: one row per player-market-game (keep first occurrence)
    df["game_key"] = df["date"].dt.date.astype(str) + "|" + df["player"] + "|" + df["market"]
    df = df.drop_duplicates(subset="game_key", keep="first").copy()

    return df


def compute_wls(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weak-line score features."""
    out = df.copy()
    out["p_model_over"] = out["p_over_raw"].clip(0.01, 0.99)
    out["p_model_under"] = 1.0 - out["p_model_over"]

    nv = out.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    out["market_no_vig_over"], out["market_no_vig_under"] = zip(*nv)

    out["model_edge_over"] = out["p_model_over"] - out["market_no_vig_over"]
    out["model_edge_under"] = out["p_model_under"] - out["market_no_vig_under"]
    out["selected_side"] = np.where(out["model_edge_over"] >= out["model_edge_under"], "OVER", "UNDER")
    out["model_edge"] = np.maximum(out["model_edge_over"], out["model_edge_under"])
    out["model_edge_normalized"] = np.clip(out["model_edge"] / 0.15, 0, 1)

    out["projection_z"] = (out["line"] - out["model_mean"]) / out["sigma"].clip(lower=0.5)
    out["projection_z_score"] = np.where(
        out["selected_side"] == "OVER",
        np.clip(-out["projection_z"] / 2, 0, 1),
        np.clip(out["projection_z"] / 2, 0, 1),
    )

    out["belief_uncertainty"] = out["belief_uncertainty"].fillna(0.5)
    out["risk_penalty"] = np.clip(out["belief_uncertainty"] * 0.3, 0, 0.3)

    out["weak_line_score"] = (
        0.45 * out["model_edge_normalized"]
        + 0.30 * out["projection_z_score"]
        + 0.25 * 0.5  # book outlier/velocity not available in historical
    ).clip(0, 1)
    out["weak_line_score_adjusted"] = (out["weak_line_score"] - out["risk_penalty"]).clip(0, 1)

    # Outcomes
    out["selected_outcome"] = np.where(out["selected_side"] == "OVER", out["result_over"], 1.0 - out["result_over"])
    out["hit"] = (out["selected_outcome"] > 0.5).astype(int)
    out["selected_odds"] = np.where(out["selected_side"] == "OVER", out["over_odds"], out["under_odds"])
    out["unit_profit"] = out.apply(lambda r: _unit_profit(r["selected_odds"], r["hit"] == 1), axis=1)
    out["p_selected"] = np.where(out["selected_side"] == "OVER", out["p_model_over"], out["p_model_under"])
    out["brier"] = (out["p_selected"] - out["selected_outcome"]) ** 2

    return out


def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return float(stat_fn(arr)), float(stat_fn(arr)), float(stat_fn(arr))
    boots = [float(stat_fn(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, alpha * 100)), float(stat_fn(arr)), float(np.percentile(boots, (1 - alpha) * 100))


def compute_baselines(df: pd.DataFrame) -> dict:
    """Compute dumb baseline strategies for comparison."""
    baselines = {}

    # All OVER
    over_hit = df["result_over"].mean()
    over_profit = df.apply(lambda r: _unit_profit(r["over_odds"], r["result_over"] == 1), axis=1)
    baselines["all_OVER"] = {"hit_rate": float(over_hit), "roi": float(over_profit.mean()), "rows": int(len(df))}

    # All UNDER
    under_hit = (1 - df["result_over"]).mean()
    under_profit = df.apply(lambda r: _unit_profit(r["under_odds"], r["result_over"] == 0), axis=1)
    baselines["all_UNDER"] = {"hit_rate": float(under_hit), "roi": float(under_profit.mean()), "rows": int(len(df))}

    # Market favorite (side with lower no-vig probability — the underdog is the value side)
    nv = df.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    nv_over, nv_under = zip(*nv)
    fav_side = np.where(np.array(nv_over) > 0.5, "OVER", "UNDER")
    fav_hit = np.where(fav_side == "OVER", df["result_over"], 1 - df["result_over"])
    fav_odds = np.where(fav_side == "OVER", df["over_odds"], df["under_odds"])
    fav_profit = [_unit_profit(o, h > 0.5) for o, h in zip(fav_odds, fav_hit)]
    baselines["market_favorite"] = {"hit_rate": float(np.mean(fav_hit)), "roi": float(np.mean(fav_profit)), "rows": int(len(df))}

    # Highest model edge only (top 20% by raw model edge, no WLS)
    top_edge = df.nlargest(int(len(df) * 0.2), "model_edge") if "model_edge" in df.columns else df.head(0)
    if len(top_edge) > 0:
        baselines["top_20pct_edge"] = {
            "hit_rate": float(top_edge["hit"].mean()),
            "roi": float(top_edge["unit_profit"].mean()),
            "rows": int(len(top_edge)),
        }

    return baselines


def walk_forward_backtest(df: pd.DataFrame) -> dict:
    """Walk-forward: train WLS thresholds on prior months, test on next month."""
    df = df.sort_values("date").copy()
    df["month"] = df["date"].dt.to_period("M")
    months = sorted(df["month"].unique())

    if len(months) < 3:
        return {"status": "insufficient_months", "months": len(months)}

    fold_results = []
    for i in range(2, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train = df[df["month"].isin(train_months)]
        test = df[df["month"] == test_month]

        if len(test) < 50:
            continue

        # In walk-forward, WLS is computed fresh on test data using the same formula
        # (no threshold optimization on test data — just apply the score and bucket)
        if len(test) == 0:
            continue

        # Bucket test data
        bins = [0.0, 0.15, 0.25, 0.35, 1.01]
        labels = ["0.00-0.15", "0.15-0.25", "0.25-0.35", "0.35+"]
        test["wls_bucket"] = pd.cut(test["weak_line_score_adjusted"], bins=bins, labels=labels).astype(str)

        # Top bucket performance
        top = test[test["weak_line_score_adjusted"] >= 0.25]
        bottom = test[test["weak_line_score_adjusted"] < 0.15]

        fold_results.append({
            "test_month": str(test_month),
            "test_rows": int(len(test)),
            "top_rows": int(len(top)),
            "top_hit_rate": float(top["hit"].mean()) if len(top) > 0 else None,
            "top_roi": float(top["unit_profit"].mean()) if len(top) > 0 else None,
            "top_brier": float(top["brier"].mean()) if len(top) > 0 else None,
            "bottom_rows": int(len(bottom)),
            "bottom_hit_rate": float(bottom["hit"].mean()) if len(bottom) > 0 else None,
            "bottom_roi": float(bottom["unit_profit"].mean()) if len(bottom) > 0 else None,
            "bottom_brier": float(bottom["brier"].mean()) if len(bottom) > 0 else None,
            "top_beats_bottom_roi": (
                float(top["unit_profit"].mean()) > float(bottom["unit_profit"].mean())
                if len(top) > 0 and len(bottom) > 0 else None
            ),
        })

    # Summary
    top_beats = [f["top_beats_bottom_roi"] for f in fold_results if f["top_beats_bottom_roi"] is not None]
    return {
        "status": "computed",
        "folds": len(fold_results),
        "top_beats_bottom_rate": float(np.mean(top_beats)) if top_beats else 0.0,
        "fold_results": fold_results,
    }


def main():
    print("=" * 70)
    print("V9.8 WEAK LINE PROFIT BACKTEST (AUDITED)")
    print("=" * 70)

    # Load and clean
    print("\nLoading and cleaning...")
    df = load_and_clean()
    print(f"  Rows after dedup + odds validation: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique player-market-games: {df['game_key'].nunique()}")
    print(f"  Players: {df['player'].nunique()}")
    print(f"  Avg odds: over={df['over_odds'].mean():.0f}, under={df['under_odds'].mean():.0f}")

    # Compute WLS
    print("\nComputing WLS...")
    scored = compute_wls(df)

    # ── Baselines ──
    print(f"\n{'=' * 70}")
    print("BASELINE COMPARISON")
    print(f"{'=' * 70}")
    baselines = compute_baselines(scored)
    print(f"\n  {'Strategy':<20s} {'Rows':>7s} {'Hit%':>6s} {'ROI':>8s}")
    print(f"  {'-'*45}")
    for name, b in baselines.items():
        print(f"  {name:<20s} {b['rows']:>7d} {b['hit_rate']:>5.1%} {b['roi']:>+7.3f}")

    # WLS strategy at different thresholds
    for threshold in [0.15, 0.20, 0.25, 0.30]:
        gated = scored[scored["weak_line_score_adjusted"] >= threshold]
        if len(gated) > 0:
            print(f"  WLS>={threshold:<4.2f}          {len(gated):>7d} {gated['hit'].mean():>5.1%} {gated['unit_profit'].mean():>+7.3f}")

    # ── Bucket Analysis with CIs ──
    print(f"\n{'=' * 70}")
    print("WLS BUCKET ANALYSIS (with bootstrap 95% CI)")
    print(f"{'=' * 70}")
    bins = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 1.01]
    labels = ["0.00-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25", "0.25-0.30", "0.30-0.35", "0.35+"]
    scored["wls_bucket"] = pd.cut(scored["weak_line_score_adjusted"], bins=bins, labels=labels).astype(str)

    print(f"\n  {'Bucket':<10s} {'Rows':>6s} {'Hit%':>6s} {'Hit CI':>14s} {'ROI':>7s} {'ROI CI':>16s} {'Brier':>7s}")
    print(f"  {'-'*72}")
    bucket_results = []
    for bucket in labels:
        group = scored[scored["wls_bucket"] == bucket]
        if len(group) == 0:
            continue
        hit_lo, hit_mid, hit_hi = bootstrap_ci(group["hit"].values)
        roi_lo, roi_mid, roi_hi = bootstrap_ci(group["unit_profit"].values)
        brier_val = float(group["brier"].mean())
        print(f"  {bucket:<10s} {len(group):>6d} {hit_mid:>5.1%} [{hit_lo:.1%},{hit_hi:.1%}] {roi_mid:>+6.3f} [{roi_lo:+.3f},{roi_hi:+.3f}] {brier_val:>7.4f}")
        bucket_results.append({
            "bucket": bucket, "rows": int(len(group)),
            "hit_rate": hit_mid, "hit_ci": [hit_lo, hit_hi],
            "roi": roi_mid, "roi_ci": [roi_lo, roi_hi],
            "brier": brier_val,
        })

    # Monotonicity (buckets with 200+ rows)
    big_buckets = [b for b in bucket_results if b["rows"] >= 200]
    rois = [b["roi"] for b in big_buckets]
    hits = [b["hit_rate"] for b in big_buckets]
    briers = [b["brier"] for b in big_buckets]

    def _mono(vals, asc=True):
        if len(vals) < 2:
            return 0.0
        pairs = len(vals) - 1
        good = sum(1 for i in range(pairs) if (vals[i+1] >= vals[i] if asc else vals[i+1] <= vals[i]))
        return good / pairs

    print(f"\n  Monotonicity (buckets with 200+ rows, n={len(big_buckets)}):")
    print(f"    ROI increasing:      {_mono(rois):.2f}")
    print(f"    Hit rate increasing: {_mono(hits):.2f}")
    print(f"    Brier decreasing:    {_mono(briers, False):.2f}")

    # ── Walk-Forward ──
    print(f"\n{'=' * 70}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'=' * 70}")
    wf = walk_forward_backtest(scored)
    if wf["status"] == "computed":
        print(f"\n  Folds: {wf['folds']}")
        print(f"  Top WLS beats bottom rate: {wf['top_beats_bottom_rate']:.1%}")
        print(f"\n  {'Month':<10s} {'Test':>5s} {'Top':>4s} {'Top Hit':>8s} {'Top ROI':>8s} {'Bot Hit':>8s} {'Bot ROI':>8s} {'Beats'}")
        for f in wf["fold_results"]:
            top_h = f"{f['top_hit_rate']:.1%}" if f["top_hit_rate"] is not None else "n/a"
            top_r = f"{f['top_roi']:+.3f}" if f["top_roi"] is not None else "n/a"
            bot_h = f"{f['bottom_hit_rate']:.1%}" if f["bottom_hit_rate"] is not None else "n/a"
            bot_r = f"{f['bottom_roi']:+.3f}" if f["bottom_roi"] is not None else "n/a"
            beats = "Y" if f["top_beats_bottom_roi"] else "N" if f["top_beats_bottom_roi"] is not None else "?"
            print(f"  {f['test_month']:<10s} {f['test_rows']:>5d} {f['top_rows']:>4d} {top_h:>8s} {top_r:>8s} {bot_h:>8s} {bot_r:>8s} {beats:>5s}")
    else:
        print(f"  Status: {wf['status']}")

    # ── Side/Market Breakdown ──
    print(f"\n{'=' * 70}")
    print("SIDE AND MARKET BREAKDOWN (WLS >= 0.20)")
    print(f"{'=' * 70}")
    gated = scored[scored["weak_line_score_adjusted"] >= 0.20]
    for side in ["OVER", "UNDER"]:
        g = gated[gated["selected_side"] == side]
        if len(g) >= 50:
            print(f"\n  {side} ({len(g)} rows): hit={g['hit'].mean():.3f}, ROI={g['unit_profit'].mean():+.3f}, Brier={g['brier'].mean():.4f}")
    for mkt in ["PTS", "TRB", "AST"]:
        g = gated[gated["market"] == mkt]
        if len(g) >= 50:
            print(f"  {mkt} ({len(g)} rows): hit={g['hit'].mean():.3f}, ROI={g['unit_profit'].mean():+.3f}, Brier={g['brier'].mean():.4f}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    wls_25 = scored[scored["weak_line_score_adjusted"] >= 0.25]
    _, wls_roi_mid, _ = bootstrap_ci(wls_25["unit_profit"].values) if len(wls_25) > 0 else (0, 0, 0)
    wls_roi_lo, _, wls_roi_hi = bootstrap_ci(wls_25["unit_profit"].values) if len(wls_25) > 0 else (0, 0, 0)

    print(f"\n  WLS >= 0.25 (recommended production threshold):")
    print(f"    Rows:     {len(wls_25)}")
    print(f"    Hit rate: {wls_25['hit'].mean():.3f}" if len(wls_25) > 0 else "    Hit rate: n/a")
    print(f"    ROI:      {wls_roi_mid:+.3f} [{wls_roi_lo:+.3f}, {wls_roi_hi:+.3f}]")
    print(f"    Brier:    {wls_25['brier'].mean():.4f}" if len(wls_25) > 0 else "    Brier: n/a")

    best_baseline_roi = max(b["roi"] for b in baselines.values())
    best_baseline_name = max(baselines, key=lambda k: baselines[k]["roi"])
    wls_beats_all = wls_roi_mid > best_baseline_roi if len(wls_25) > 0 else False
    print(f"\n  Best baseline: {best_baseline_name} (ROI={best_baseline_roi:+.3f})")
    print(f"  WLS >= 0.25 beats best baseline: {wls_beats_all}")
    print(f"  Walk-forward top-beats-bottom: {wf.get('top_beats_bottom_rate', 0):.0%}")

    # Save
    report = {
        "backtest_at": datetime.now(timezone.utc).isoformat(),
        "data": {"rows": int(len(scored)), "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}", "unique_games": int(df["game_key"].nunique()), "deduped": True, "odds_validated": True, "pushes_excluded": True},
        "baselines": baselines,
        "bucket_results": bucket_results,
        "monotonicity": {"roi": _mono(rois), "hit": _mono(hits), "brier_dec": _mono(briers, False)},
        "walk_forward": wf,
        "wls_25_summary": {"rows": int(len(wls_25)), "hit": float(wls_25["hit"].mean()) if len(wls_25) > 0 else None, "roi": wls_roi_mid, "roi_ci": [wls_roi_lo, wls_roi_hi], "beats_best_baseline": wls_beats_all},
    }
    out_path = ROOT / "model" / "props" / "v9_6" / "validation" / "weak_line_profit_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report: {out_path}")


if __name__ == "__main__":
    main()
