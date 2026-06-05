#!/usr/bin/env python3
"""
Evaluate CLV promotion gates with proper separation:

  market_sequence_gates: Does the CLV pipeline produce valid, time-separated
    snapshots with real price movement? (Infrastructure proof)

  model_promotion_gates: Does the v9.6 model select sides that beat the
    closing market? (Strategy proof — requires model predictions attached)

These are different questions. The first proves plumbing. The second proves
the model is worth deploying.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from market_odds_quality import add_american_odds_quality, is_valid_american_odds


def _american_to_implied(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    over = _american_to_implied(over_odds)
    under = _american_to_implied(under_odds)
    total = over + under
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    return over / total, under / total


# ─── Market Sequence Gates ─────────────────────────────────────────

def evaluate_market_sequence_gates(attachable_path: Path) -> dict:
    """Evaluate whether the CLV pipeline produces valid market data.

    This answers: "Is the infrastructure working?"
    NOT: "Is the model good?"
    """
    df = pd.read_csv(attachable_path)
    df = add_american_odds_quality(df)

    true_clv = df[df.get("close_status", pd.Series()) == "true_sequence_close"].copy() if "close_status" in df.columns else df.copy()

    # Compute entry/close no-vig
    entry_valid = (
        true_clv["over_odds"].apply(is_valid_american_odds)
        & true_clv["under_odds"].apply(is_valid_american_odds)
    )
    close_valid = pd.Series(False, index=true_clv.index)
    if "close_over_odds" in true_clv.columns:
        close_valid = (
            true_clv["close_over_odds"].apply(is_valid_american_odds)
            & true_clv["close_under_odds"].apply(is_valid_american_odds)
        )
    both_valid = entry_valid & close_valid

    # Check entry != close (not same row)
    if both_valid.any():
        subset = true_clv.loc[both_valid]
        entry_close_differ = (subset["over_odds"] != subset["close_over_odds"]).sum()
    else:
        entry_close_differ = 0

    # Distinct snapshot times
    if "snapshot_time" in df.columns:
        distinct_times = df["snapshot_time"].nunique()
    else:
        distinct_times = 0

    gates = {
        "valid_american_odds_rate": float(df["is_valid_american_odds"].mean()) >= 0.98 if len(df) > 0 else False,
        "sufficient_clv_rows": int(both_valid.sum()) >= 500,
        "sufficient_price_movement": int(entry_close_differ) >= 50,
        "distinct_snapshot_times": distinct_times >= 3,
        "entry_close_not_same_row": int(entry_close_differ) > 0,
    }

    return {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "details": {
            "total_rows": int(len(df)),
            "true_clv_rows": int(len(true_clv)),
            "rows_with_both_valid": int(both_valid.sum()),
            "rows_with_price_movement": int(entry_close_differ),
            "distinct_snapshot_times": distinct_times,
            "valid_odds_rate": float(df["is_valid_american_odds"].mean()) if len(df) > 0 else 0.0,
        },
    }


# ─── Market Movement Proxy ─────────────────────────────────────────

def evaluate_market_movement_proxy(attachable_path: Path) -> dict:
    """Compute market-only open-to-close directional movement metrics.

    This is NOT model CLV. It measures whether early prices are less
    stable than later prices and whether there is enough movement to
    make CLV validation meaningful.
    """
    df = pd.read_csv(attachable_path)
    df = add_american_odds_quality(df)
    true_clv = df[df.get("close_status", pd.Series()) == "true_sequence_close"].copy() if "close_status" in df.columns else df.copy()

    entry_valid = true_clv["over_odds"].apply(is_valid_american_odds) & true_clv["under_odds"].apply(is_valid_american_odds)
    close_valid = pd.Series(False, index=true_clv.index)
    if "close_over_odds" in true_clv.columns:
        close_valid = true_clv["close_over_odds"].apply(is_valid_american_odds) & true_clv["close_under_odds"].apply(is_valid_american_odds)
    both_valid = entry_valid & close_valid

    if both_valid.sum() == 0:
        return {"status": "no_data", "rows": 0}

    subset = true_clv.loc[both_valid].copy()

    # Compute no-vig
    entry_nv = subset.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    close_nv = subset.apply(lambda r: _no_vig(r["close_over_odds"], r["close_under_odds"]), axis=1)
    subset["entry_no_vig_over"], subset["entry_no_vig_under"] = zip(*entry_nv)
    subset["close_no_vig_over"], subset["close_no_vig_under"] = zip(*close_nv)

    subset["clv_over"] = subset["close_no_vig_over"] - subset["entry_no_vig_over"]
    subset["clv_under"] = subset["close_no_vig_under"] - subset["entry_no_vig_under"]

    # Movement metrics (market-only, no model)
    price_movement = (subset["close_over_odds"] - subset["over_odds"]).abs()
    moved = subset[price_movement > 0]
    no_vig_movement = subset["clv_over"].abs()
    moved_nv = subset[no_vig_movement > 1e-6]

    return {
        "status": "computed",
        "total_rows": int(len(subset)),
        "rows_with_odds_movement": int(len(moved)),
        "pct_with_odds_movement": float(len(moved) / len(subset)) if len(subset) > 0 else 0.0,
        "mean_abs_odds_movement": float(price_movement.mean()),
        "mean_abs_no_vig_movement": float(no_vig_movement.mean()),
        "rows_with_no_vig_movement": int(len(moved_nv)),
        "mean_clv_over": float(subset["clv_over"].mean()),
        "mean_clv_under": float(subset["clv_under"].mean()),
        "clv_over_std": float(subset["clv_over"].std()),
        "note": "This is market-only open-to-close movement, NOT model CLV.",
    }


# ─── Model Promotion Gates ────────────────────────────────────────

def evaluate_model_promotion_gates(attachable_path: Path, model_manifest_path: Path) -> dict:
    """Evaluate whether the model's selected sides beat the closing market.

    This requires actual model predictions joined to the CLV rows.
    If predictions are not available, gates are reported as 'unavailable'.
    """
    df = pd.read_csv(attachable_path)
    df = add_american_odds_quality(df)
    true_clv = df[df.get("close_status", pd.Series()) == "true_sequence_close"].copy() if "close_status" in df.columns else df.copy()

    # Check if model manifest exists and has scored data
    if not model_manifest_path.exists():
        return _unavailable_result("model manifest not found")

    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_output = Path(manifest.get("output", ""))
    if not model_output.is_absolute():
        model_output = REPO_ROOT / model_output
    scored_path = model_output / "data" / "prop_training_rows.csv"

    if not scored_path.exists():
        return _unavailable_result("model scored data not found")

    # Load model predictions
    model_rows = pd.read_csv(scored_path, low_memory=False)
    model_rows["date"] = pd.to_datetime(model_rows["date"], errors="coerce").dt.date.astype(str)

    # The model data has: player, market, date, p_over_raw, p_v96_calibrated, selected_side, edge_over, edge_under
    # The CLV data has: player, market, date, over_odds, under_odds, close_over_odds, close_under_odds
    # Join on [player, market, date]

    # Prepare CLV data
    true_clv["date"] = pd.to_datetime(true_clv["date"], errors="coerce").dt.date.astype(str)

    # Check for join keys
    join_keys = ["player", "market", "date"]
    if not all(k in true_clv.columns for k in join_keys) or not all(k in model_rows.columns for k in join_keys):
        return _unavailable_result("join keys missing from data")

    # Determine model's selected side from scored data
    if "selected_side" in model_rows.columns:
        side_col = "selected_side"
    elif "edge_over" in model_rows.columns and "edge_under" in model_rows.columns:
        model_rows["selected_side"] = np.where(
            model_rows["edge_over"] >= model_rows["edge_under"], "OVER", "UNDER"
        )
        side_col = "selected_side"
    else:
        return _unavailable_result("no side selection columns in model data")

    # Get model's selected side and edge per player/market/date
    model_selections = model_rows.groupby(join_keys).agg(
        model_selected_side=(side_col, "last"),
        model_edge_over=("edge_over", "last") if "edge_over" in model_rows.columns else ("p_over_raw", "last"),
        model_edge_under=("edge_under", "last") if "edge_under" in model_rows.columns else ("p_over_raw", "last"),
        model_p_calibrated=("p_v96_calibrated", "last") if "p_v96_calibrated" in model_rows.columns else ("p_over_raw", "last"),
    ).reset_index()

    # Join model selections to CLV data
    merged = true_clv.merge(model_selections, on=join_keys, how="inner")

    if len(merged) == 0:
        return _unavailable_result(
            f"no date overlap between model data ({model_rows['date'].min()} to {model_rows['date'].max()}) "
            f"and CLV data ({true_clv['date'].min()} to {true_clv['date'].max()})"
        )

    # Compute no-vig for entry and close
    entry_valid = merged["over_odds"].apply(is_valid_american_odds) & merged["under_odds"].apply(is_valid_american_odds)
    close_valid = merged["close_over_odds"].apply(is_valid_american_odds) & merged["close_under_odds"].apply(is_valid_american_odds)
    both_valid = entry_valid & close_valid

    if both_valid.sum() == 0:
        return _unavailable_result("no rows with valid entry and close odds after model join")

    valid = merged.loc[both_valid].copy()
    entry_nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    close_nv = valid.apply(lambda r: _no_vig(r["close_over_odds"], r["close_under_odds"]), axis=1)
    valid["entry_no_vig_over"], valid["entry_no_vig_under"] = zip(*entry_nv)
    valid["close_no_vig_over"], valid["close_no_vig_under"] = zip(*close_nv)

    # Model-selected-side CLV
    valid["clv_over"] = valid["close_no_vig_over"] - valid["entry_no_vig_over"]
    valid["clv_under"] = valid["close_no_vig_under"] - valid["entry_no_vig_under"]
    valid["model_side_clv"] = np.where(
        valid["model_selected_side"] == "OVER",
        valid["clv_over"],
        valid["clv_under"],
    )

    # Only evaluate on rows with actual movement
    moved = valid[valid["clv_over"].abs() > 1e-6]

    if len(moved) < 30:
        return {
            "status": "insufficient_moved_rows",
            "model_predictions_attached": True,
            "matched_rows": int(len(valid)),
            "moved_rows": int(len(moved)),
            "note": "Need at least 30 rows with price movement for reliable CLV evaluation",
            "gates": {
                "model_predictions_attached": True,
                "model_selected_side_clv_positive": "unavailable",
                "model_positive_clv_rate_above_50": "unavailable",
                "model_edge_clv_correlation_positive": "unavailable",
                "model_bss_vs_true_market_positive": "unavailable",
            },
        }

    # Compute model CLV metrics
    mean_model_clv = float(moved["model_side_clv"].mean())
    positive_clv_rate = float((moved["model_side_clv"] > 0).mean())

    # Model edge vs CLV correlation
    moved["model_edge_abs"] = np.where(
        moved["model_selected_side"] == "OVER",
        moved["model_edge_over"].abs(),
        moved["model_edge_under"].abs(),
    )
    edge_clv_corr = float(moved["model_edge_abs"].corr(moved["model_side_clv"]))
    if not np.isfinite(edge_clv_corr):
        edge_clv_corr = 0.0

    # Model BSS vs market: compare model probability to market no-vig
    # If model is better than market, its Brier vs outcomes should be lower
    # Since we don't have outcomes yet, use close as proxy for truth
    model_brier_vs_close = float(((valid["model_p_calibrated"] - valid["close_no_vig_over"]) ** 2).mean())
    entry_brier_vs_close = float(((valid["entry_no_vig_over"] - valid["close_no_vig_over"]) ** 2).mean())
    model_beats_entry = model_brier_vs_close < entry_brier_vs_close

    gates = {
        "model_predictions_attached": True,
        "model_selected_side_clv_positive": mean_model_clv > 0,
        "model_positive_clv_rate_above_50": positive_clv_rate > 0.50,
        "model_edge_clv_correlation_positive": edge_clv_corr > 0,
        "model_bss_vs_true_market_positive": model_beats_entry,
    }

    return {
        "status": "pass" if all(v is True for v in gates.values()) else "partial",
        "model_version": manifest.get("model_version", "unknown"),
        "matched_rows": int(len(valid)),
        "moved_rows": int(len(moved)),
        "metrics": {
            "mean_model_side_clv": mean_model_clv,
            "positive_clv_rate": positive_clv_rate,
            "edge_clv_correlation": edge_clv_corr,
            "model_brier_vs_close": model_brier_vs_close,
            "entry_brier_vs_close": entry_brier_vs_close,
            "model_beats_entry_market": model_beats_entry,
        },
        "gates": gates,
    }


def _unavailable_result(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "gates": {
            "model_predictions_attached": False,
            "model_selected_side_clv_positive": "unavailable",
            "model_positive_clv_rate_above_50": "unavailable",
            "model_edge_clv_correlation_positive": "unavailable",
            "model_bss_vs_true_market_positive": "unavailable",
        },
    }


# ─── Main ─────────────────────────────────────────────────────────

def main():
    attachable_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "market_snapshot_attachable.csv"
    model_manifest_path = ROOT / "model" / "props" / "v9_6" / "manifest.json"

    if not attachable_path.exists():
        print("ERROR: market_snapshot_attachable.csv not found")
        return

    print("=" * 70)
    print("CLV PROMOTION GATE EVALUATION")
    print("=" * 70)

    # ── 1. Market Sequence Gates ──
    print("\n[1/3] Market Sequence Gates (infrastructure proof)")
    seq_result = evaluate_market_sequence_gates(attachable_path)
    print(f"  Status: {seq_result['status'].upper()}")
    for gate, passed in seq_result["gates"].items():
        marker = "✓" if passed else "✗"
        print(f"    {marker} {gate}")
    d = seq_result["details"]
    print(f"  Details: {d['true_clv_rows']} CLV rows, {d['rows_with_price_movement']} moved, {d['distinct_snapshot_times']} timestamps")

    # ── 2. Market Movement Proxy ──
    print("\n[2/3] Market Movement Proxy (open-to-close directional movement)")
    proxy_result = evaluate_market_movement_proxy(attachable_path)
    if proxy_result["status"] == "computed":
        print(f"  Rows with odds movement: {proxy_result['rows_with_odds_movement']} / {proxy_result['total_rows']} ({proxy_result['pct_with_odds_movement']:.1%})")
        print(f"  Mean abs odds movement:  {proxy_result['mean_abs_odds_movement']:.1f} cents")
        print(f"  Mean abs no-vig movement:{proxy_result['mean_abs_no_vig_movement']:.6f}")
        print(f"  Mean CLV over:           {proxy_result['mean_clv_over']:+.6f}")
        print(f"  Mean CLV under:          {proxy_result['mean_clv_under']:+.6f}")
        print(f"  NOTE: {proxy_result['note']}")
    else:
        print(f"  Status: {proxy_result['status']}")

    # ── 3. Model Promotion Gates ──
    print("\n[3/3] Model Promotion Gates (strategy proof)")
    model_result = evaluate_model_promotion_gates(attachable_path, model_manifest_path)
    print(f"  Status: {model_result['status'].upper()}")
    if model_result["status"] == "unavailable":
        print(f"  Reason: {model_result['reason']}")
    elif model_result["status"] == "insufficient_moved_rows":
        print(f"  Matched rows: {model_result['matched_rows']}, Moved: {model_result['moved_rows']}")
        print(f"  Note: {model_result['note']}")
    elif "metrics" in model_result:
        m = model_result["metrics"]
        print(f"  Model version:           {model_result.get('model_version', '?')}")
        print(f"  Matched rows:            {model_result['matched_rows']}")
        print(f"  Moved rows:              {model_result['moved_rows']}")
        print(f"  Mean model-side CLV:     {m['mean_model_side_clv']:+.6f}")
        print(f"  Positive CLV rate:       {m['positive_clv_rate']:.3f}")
        print(f"  Edge-CLV correlation:    {m['edge_clv_correlation']:+.4f}")
        print(f"  Model beats entry mkt:   {m['model_beats_entry_market']}")
    for gate, passed in model_result["gates"].items():
        if passed == "unavailable":
            marker = "?"
        elif passed:
            marker = "✓"
        else:
            marker = "✗"
        print(f"    {marker} {gate}: {passed}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Market sequence gates:    {seq_result['status'].upper()}")
    print(f"  Market movement proxy:    {'COMPUTED' if proxy_result['status'] == 'computed' else proxy_result['status'].upper()}")
    print(f"  Model promotion gates:    {model_result['status'].upper()}")

    all_infra_pass = seq_result["status"] == "pass"
    model_pass = model_result["status"] == "pass"
    if all_infra_pass and model_pass:
        overall = "PROMOTED"
    elif all_infra_pass:
        overall = "INFRASTRUCTURE PASS — MODEL PROMOTION BLOCKED"
    else:
        overall = "BLOCKED"
    print(f"  Overall:                  {overall}")
    print("=" * 70)

    # Write report
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "market_sequence_gates": seq_result,
        "market_movement_proxy": proxy_result,
        "model_promotion_gates": model_result,
        "overall_status": overall,
    }
    output_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "clv_promotion_gate_report.json"
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {output_path}")


if __name__ == "__main__":
    main()
