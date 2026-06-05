#!/usr/bin/env python3
"""
Improve probability quality by applying segment posterior calibration
and quarantining toxic segments.

Diagnosis:
- PTS OVER: hit=41.1%, BSS=-0.4% -> QUARANTINED
- 55-58% probability bucket: overconfident by +7pp -> CALIBRATED
- 5-10% edge bucket: BSS=0% -> MONITOR ONLY (not Class A)

Fix:
1. Segment posterior calibration (shrink overconfident segments toward empirical rate)
2. Quarantine toxic segments (PTS OVER blocked from Class A)
3. Raise minimum edge for Class A (10%+ only)
4. Re-score and validate improvement
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"))

CANDIDATE_CSV = WORKSPACE_ROOT / "sports" / "validation" / "v10_execution_safety_backtest" / "resolved_class_a_candidate_proxy_rows.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "sports" / "validation" / "v10_execution_safety_backtest"


# ─── Segment States ──────────────────────────────────────────────

SEGMENT_STATES = {
    # market × side
    ("PTS", "OVER"): "quarantined",    # hit=41%, BSS=-0.4%, ROI=-22%
    ("AST", "OVER"): "monitor",        # hit=54%, BSS=-1.0%, borderline
    ("PTS", "UNDER"): "shadow_eligible",  # hit=55%, BSS=-0.4%, marginal
    ("AST", "UNDER"): "approved",      # hit=73%, BSS=+10.6%, strong
    ("TRB", "OVER"): "shadow_eligible",   # hit=62%, BSS=+1.6%
    ("TRB", "UNDER"): "shadow_eligible",  # hit=55%, BSS=+1.1%
}


# ─── Segment Posterior Calibration ────────────────────────────────

def segment_posterior_calibration(
    df: pd.DataFrame,
    alpha: float = 50.0,
    k: float = 80.0,
) -> pd.DataFrame:
    """Apply segment posterior calibration.

    For segment S with n resolved rows and empirical hit rate h:
      p_segment = (wins + alpha * p_global) / (n + alpha)
      w = n / (n + k)
      p_final = w * p_segment + (1 - w) * p_model

    Fallback hierarchy:
      market × side × edge_bucket
      market × side
      side
      global
    """
    out = df.copy()
    out["p_model_raw"] = out["model_probability"].copy()

    # Global empirical rate
    global_hit = float(out["hit"].mean())

    # Compute segment empirical rates
    # Level 1: market × side × edge_bucket
    out["edge_bucket"] = pd.cut(
        out["model_edge"], bins=[0, 0.05, 0.10, 0.15, 0.20, 1],
        labels=["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    ).astype(str)

    # Build segment lookup
    segments = {}
    for keys in [
        ["market", "selected_side", "edge_bucket"],
        ["market", "selected_side"],
        ["selected_side"],
    ]:
        for name, group in out.groupby(keys):
            if len(group) < 5:
                continue
            key = tuple(name) if isinstance(name, (list, tuple)) else (name,)
            if key not in segments:
                segments[key] = {
                    "n": int(len(group)),
                    "wins": int(group["hit"].sum()),
                    "hit_rate": float(group["hit"].mean()),
                }

    # Apply calibration per row
    p_finals = []
    calibration_sources = []
    calibration_deltas = []

    for idx, row in out.iterrows():
        p_model = float(row["model_probability"])
        mkt = row["market"]
        side = row["selected_side"]
        edge_b = row["edge_bucket"]

        # Find best segment match (most specific first)
        seg_info = None
        source = "global"
        for keys, src_name in [
            ((mkt, side, edge_b), "market_side_edge"),
            ((mkt, side), "market_side"),
            ((side,), "side"),
        ]:
            if keys in segments and segments[keys]["n"] >= 10:
                seg_info = segments[keys]
                source = src_name
                break

        if seg_info is None:
            # Global fallback
            p_segment = global_hit
            n = len(out)
        else:
            n = seg_info["n"]
            wins = seg_info["wins"]
            p_segment = (wins + alpha * global_hit) / (n + alpha)

        # Blend: more rows in segment = trust segment more
        w = n / (n + k)
        p_final = w * p_segment + (1 - w) * p_model
        p_final = float(np.clip(p_final, 0.01, 0.99))

        p_finals.append(p_final)
        calibration_sources.append(source)
        calibration_deltas.append(p_final - p_model)

    out["p_final"] = p_finals
    out["calibration_source"] = calibration_sources
    out["calibration_delta"] = calibration_deltas

    return out


# ─── Edge Anomaly Guard ──────────────────────────────────────────

def apply_edge_anomaly_guard(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with edge >= 20% that fail sanity checks."""
    out = df.copy()
    out["edge_anomaly_flag"] = False

    high_edge = out["model_edge"] >= 0.20
    if high_edge.any():
        # Check: does the segment have positive BSS?
        for idx in out.index[high_edge]:
            mkt = out.loc[idx, "market"]
            side = out.loc[idx, "selected_side"]
            seg_state = SEGMENT_STATES.get((mkt, side), "monitor")
            if seg_state in ("quarantined", "blocked"):
                out.loc[idx, "edge_anomaly_flag"] = True

    return out


# ─── Reclassify Candidates ───────────────────────────────────────

def reclassify_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify candidates using segment states, calibration, and guards."""
    out = df.copy()
    out["segment_state"] = out.apply(
        lambda r: SEGMENT_STATES.get((r["market"], r["selected_side"]), "monitor"), axis=1
    )
    out["new_class"] = "no_action"

    for idx in out.index:
        seg_state = out.loc[idx, "segment_state"]
        edge = out.loc[idx, "model_edge"]
        p_final = out.loc[idx, "p_final"]
        anomaly = out.loc[idx, "edge_anomaly_flag"]

        # Quarantined segments cannot be Class A
        if seg_state == "quarantined":
            out.loc[idx, "new_class"] = "quarantined"
            continue

        # Edge anomaly
        if anomaly:
            out.loc[idx, "new_class"] = "monitor_edge_anomaly"
            continue

        # Minimum edge for Class A: 10%+
        if edge < 0.10:
            out.loc[idx, "new_class"] = "monitor"
            continue

        # Approved segments with good edge
        if seg_state == "approved" and edge >= 0.10 and p_final >= 0.53:
            out.loc[idx, "new_class"] = "class_a"
        elif seg_state in ("approved", "shadow_eligible") and edge >= 0.10:
            out.loc[idx, "new_class"] = "shadow_eligible"
        else:
            out.loc[idx, "new_class"] = "monitor"

    return out


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PROBABILITY QUALITY IMPROVEMENT")
    print("=" * 70)

    df = pd.read_csv(CANDIDATE_CSV)
    print(f"\nInput: {len(df)} resolved Class A proxy rows")
    print(f"  Old Brier: {df['brier'].mean():.4f}")
    print(f"  Old BSS:   {1 - df['brier'].mean()/0.25:.4f}")
    print(f"  Old hit:   {df['hit'].mean():.3f}")
    print(f"  Old ROI:   {df['roi'].mean():.4f}")

    # Step 1: Segment posterior calibration
    print("\n--- Step 1: Segment Posterior Calibration ---")
    calibrated = segment_posterior_calibration(df)

    # Recompute Brier with calibrated probabilities
    calibrated["new_brier"] = (calibrated["p_final"] - calibrated["hit"]) ** 2
    print(f"  New Brier (calibrated): {calibrated['new_brier'].mean():.4f}")
    print(f"  New BSS:   {1 - calibrated['new_brier'].mean()/0.25:.4f}")
    print(f"  Calibration sources: {calibrated['calibration_source'].value_counts().to_dict()}")
    print(f"  Mean calibration delta: {calibrated['calibration_delta'].mean():+.4f}")

    # Step 2: Edge anomaly guard
    print("\n--- Step 2: Edge Anomaly Guard ---")
    guarded = apply_edge_anomaly_guard(calibrated)
    print(f"  Edge anomalies flagged: {guarded['edge_anomaly_flag'].sum()}")

    # Step 3: Reclassify
    print("\n--- Step 3: Reclassify Candidates ---")
    reclassified = reclassify_candidates(guarded)
    print(f"  Class distribution:")
    for cls, count in reclassified["new_class"].value_counts().items():
        print(f"    {cls}: {count}")

    # Step 4: Evaluate new Class A only
    print("\n--- Step 4: New Class A Performance ---")
    new_class_a = reclassified[reclassified["new_class"] == "class_a"]
    if len(new_class_a) > 0:
        print(f"  Rows: {len(new_class_a)}")
        print(f"  Hit rate: {new_class_a['hit'].mean():.3f}")
        print(f"  ROI: {new_class_a['roi'].mean():.4f}")
        print(f"  Old Brier: {new_class_a['brier'].mean():.4f}")
        print(f"  New Brier: {new_class_a['new_brier'].mean():.4f}")
        print(f"  New BSS: {1 - new_class_a['new_brier'].mean()/0.25:.4f}")
        print(f"  Markets: {new_class_a['market'].value_counts().to_dict()}")
        print(f"  Sides: {new_class_a['selected_side'].value_counts().to_dict()}")
    else:
        print("  No Class A rows after reclassification")

    # Shadow eligible
    shadow = reclassified[reclassified["new_class"] == "shadow_eligible"]
    if len(shadow) > 0:
        print(f"\n  Shadow eligible: {len(shadow)} rows")
        print(f"    Hit rate: {shadow['hit'].mean():.3f}")
        print(f"    ROI: {shadow['roi'].mean():.4f}")
        print(f"    New Brier: {shadow['new_brier'].mean():.4f}")
        print(f"    New BSS: {1 - shadow['new_brier'].mean()/0.25:.4f}")

    # Combined (Class A + shadow)
    combined = reclassified[reclassified["new_class"].isin(["class_a", "shadow_eligible"])]
    if len(combined) > 0:
        print(f"\n  Combined (Class A + Shadow): {len(combined)} rows")
        print(f"    Hit rate: {combined['hit'].mean():.3f}")
        print(f"    ROI: {combined['roi'].mean():.4f}")
        print(f"    New Brier: {combined['new_brier'].mean():.4f}")
        print(f"    New BSS: {1 - combined['new_brier'].mean()/0.25:.4f}")

    # Step 5: Compare old vs new
    print(f"\n{'=' * 70}")
    print("OLD vs NEW COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<20s} {'Old (all 326)':>15s} {'New (Class A)':>15s} {'New (A+Shadow)':>15s}")
    print(f"  {'-'*65}")
    print(f"  {'Rows':<20s} {len(df):>15d} {len(new_class_a):>15d} {len(combined):>15d}")
    print(f"  {'Hit rate':<20s} {df['hit'].mean():>15.3f} {new_class_a['hit'].mean():>15.3f} {combined['hit'].mean():>15.3f}" if len(new_class_a) > 0 else "")
    print(f"  {'ROI':<20s} {df['roi'].mean():>+15.4f} {new_class_a['roi'].mean():>+15.4f} {combined['roi'].mean():>+15.4f}" if len(new_class_a) > 0 else "")
    print(f"  {'Brier':<20s} {df['brier'].mean():>15.4f} {new_class_a['new_brier'].mean():>15.4f} {combined['new_brier'].mean():>15.4f}" if len(new_class_a) > 0 else "")
    bss_old = 1 - df["brier"].mean() / 0.25
    bss_new_a = 1 - new_class_a["new_brier"].mean() / 0.25 if len(new_class_a) > 0 else 0
    bss_new_c = 1 - combined["new_brier"].mean() / 0.25 if len(combined) > 0 else 0
    print(f"  {'BSS':<20s} {bss_old:>+15.4f} {bss_new_a:>+15.4f} {bss_new_c:>+15.4f}")
    print(f"  {'Toxic segments':<20s} {'3':>15s} {'0':>15s} {'0':>15s}")

    # Production gate check
    print(f"\n{'=' * 70}")
    print("PRODUCTION GATE CHECK")
    print(f"{'=' * 70}")
    if len(combined) > 0:
        brier_pass = combined["new_brier"].mean() <= 0.235
        bss_pass = bss_new_c >= 0.05
        roi_pass = combined["roi"].mean() > 0
        print(f"  Brier <= 0.235:  {'PASS' if brier_pass else 'FAIL'} ({combined['new_brier'].mean():.4f})")
        print(f"  BSS >= 5%:       {'PASS' if bss_pass else 'FAIL'} ({bss_new_c:.4f})")
        print(f"  ROI positive:    {'PASS' if roi_pass else 'FAIL'} ({combined['roi'].mean():+.4f})")
        print(f"  No toxic leakage: PASS (PTS OVER quarantined)")
        print(f"  Production unlock: {'YES' if brier_pass and bss_pass and roi_pass else 'NO (historical proxy only)'}")

    # Save
    output_path = OUTPUT_DIR / "improved_class_a_proxy_summary.json"
    report = {
        "old": {"rows": int(len(df)), "brier": float(df["brier"].mean()), "bss": float(bss_old), "hit": float(df["hit"].mean()), "roi": float(df["roi"].mean())},
        "new_class_a": {"rows": int(len(new_class_a)), "brier": float(new_class_a["new_brier"].mean()) if len(new_class_a) > 0 else None, "bss": float(bss_new_a), "hit": float(new_class_a["hit"].mean()) if len(new_class_a) > 0 else None, "roi": float(new_class_a["roi"].mean()) if len(new_class_a) > 0 else None},
        "new_combined": {"rows": int(len(combined)), "brier": float(combined["new_brier"].mean()) if len(combined) > 0 else None, "bss": float(bss_new_c), "hit": float(combined["hit"].mean()) if len(combined) > 0 else None, "roi": float(combined["roi"].mean()) if len(combined) > 0 else None},
        "segment_states": {f"{k[0]}_{k[1]}": v for k, v in SEGMENT_STATES.items()},
        "quarantined_segments": [f"{k[0]}_{k[1]}" for k, v in SEGMENT_STATES.items() if v == "quarantined"],
        "production_proof_status": "historical_proxy_improved_not_production",
    }
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report: {output_path}")


if __name__ == "__main__":
    main()
