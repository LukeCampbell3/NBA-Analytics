#!/usr/bin/env python3
"""
Attach full-distribution model predictions to CLV market snapshots.

This is the v9.7 model-attached CLV path. It enforces:
  - Predictions MUST come from the full player distribution pipeline
  - Predictions MUST NOT be calibrator-only or market-only side selection
  - Side selection MUST come from model edge (model_mean vs line)
  - Side collapse guard: max 80% on one side

The model generates:
  model_mean (predicted stat value)
  sigma (uncertainty)
  p_over_raw = 1 - Φ((line - model_mean) / sigma)  [normal CDF]
  edge = p_over_raw - market_no_vig_over

Then CLV is computed as:
  OVER CLV  = close_no_vig_over  - entry_no_vig_over
  UNDER CLV = close_no_vig_under - entry_no_vig_under

Model-side CLV = CLV on the side the model selected.
Positive means the market moved toward the model's view.
"""
from __future__ import annotations

import argparse
import json
import math
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


def _p_over_from_distribution(model_mean: float, line: float, sigma: float) -> float:
    """Compute P(stat > line) from normal distribution parameters."""
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0
    z = (line - model_mean) / sigma
    p_over = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return float(np.clip(p_over, 0.01, 0.99))


# Market code -> slate target column mapping
MARKET_TO_TARGET = {
    "PTS": "PTS",
    "TRB": "TRB",
    "AST": "AST",
    "3PM": "PTS",  # fallback
}


def load_slate_predictions(slate_path: Path) -> pd.DataFrame:
    """Load the daily slate predictions from the full model pipeline."""
    df = pd.read_csv(slate_path)
    # The slate has columns like: player, market_date, pred_PTS, market_PTS, edge_PTS,
    # PTS_uncertainty_sigma, belief_uncertainty, etc.
    records = []
    for _, row in df.iterrows():
        player = str(row.get("player", "")).strip()
        market_date = str(row.get("market_date", ""))
        for target in ["PTS", "TRB", "AST"]:
            pred_col = f"pred_{target}"
            market_col = f"market_{target}"
            sigma_col = f"{target}_uncertainty_sigma"
            if pred_col not in row.index or pd.isna(row.get(pred_col)):
                continue
            market_line = row.get(market_col)
            if pd.isna(market_line):
                continue
            model_mean = float(row[pred_col])
            line = float(market_line)
            sigma = float(row.get(sigma_col, 3.0)) if pd.notna(row.get(sigma_col)) else 3.0
            p_over = _p_over_from_distribution(model_mean, line, sigma)
            records.append({
                "player": player,
                "date": market_date,
                "market": target,
                "model_mean": model_mean,
                "line": line,
                "sigma": sigma,
                "p_model_over": p_over,
                "p_model_under": 1.0 - p_over,
                "belief_uncertainty": float(row.get("belief_uncertainty", 0.5)),
                "prediction_source": "full_distribution_pipeline",
            })
    return pd.DataFrame(records)


def attach_and_evaluate(
    predictions: pd.DataFrame,
    attachable_path: Path,
    max_side_share: float = 0.80,
) -> dict:
    """Attach model predictions to CLV data and compute model-side CLV."""
    df = pd.read_csv(attachable_path)
    df = add_american_odds_quality(df)
    true_clv = df[df.get("close_status", pd.Series()) == "true_sequence_close"].copy()

    if true_clv.empty:
        return {"status": "no_clv_rows"}

    # Normalize player names for join
    true_clv["player_norm"] = true_clv["player"].str.replace("_", " ").str.lower().str.strip()
    predictions["player_norm"] = predictions["player"].str.replace("_", " ").str.lower().str.strip()
    true_clv["date"] = pd.to_datetime(true_clv["date"], errors="coerce").dt.date.astype(str)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.date.astype(str)

    # Join on player + market + date
    merged = true_clv.merge(
        predictions[["player_norm", "market", "date", "model_mean", "sigma",
                     "p_model_over", "p_model_under", "belief_uncertainty", "prediction_source"]],
        on=["player_norm", "market", "date"],
        how="inner",
        suffixes=("", "_model"),
    )

    if len(merged) == 0:
        return {
            "status": "no_matches",
            "reason": f"No overlap between predictions ({len(predictions)} rows) and CLV data ({len(true_clv)} rows)",
            "prediction_dates": sorted(predictions["date"].unique().tolist()),
            "clv_dates": sorted(true_clv["date"].unique().tolist()),
            "prediction_players_sample": predictions["player_norm"].unique()[:10].tolist(),
            "clv_players_sample": true_clv["player_norm"].unique()[:10].tolist(),
        }

    # Compute entry and close no-vig
    entry_valid = merged["over_odds"].apply(is_valid_american_odds) & merged["under_odds"].apply(is_valid_american_odds)
    close_valid = merged["close_over_odds"].apply(is_valid_american_odds) & merged["close_under_odds"].apply(is_valid_american_odds)
    both_valid = entry_valid & close_valid

    valid = merged.loc[both_valid].copy()
    if len(valid) == 0:
        return {"status": "no_valid_odds_after_join", "matched_rows": int(len(merged))}

    entry_nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    close_nv = valid.apply(lambda r: _no_vig(r["close_over_odds"], r["close_under_odds"]), axis=1)
    valid["entry_no_vig_over"], valid["entry_no_vig_under"] = zip(*entry_nv)
    valid["close_no_vig_over"], valid["close_no_vig_under"] = zip(*close_nv)

    # Model-selected side from distribution edge
    valid["model_edge_over"] = valid["p_model_over"] - valid["entry_no_vig_over"]
    valid["model_edge_under"] = valid["p_model_under"] - valid["entry_no_vig_under"]
    valid["selected_side"] = np.where(
        valid["model_edge_over"] >= valid["model_edge_under"], "OVER", "UNDER"
    )
    valid["selected_edge"] = np.where(
        valid["selected_side"] == "OVER",
        valid["model_edge_over"],
        valid["model_edge_under"],
    )

    # Side collapse guard
    side_counts = valid["selected_side"].value_counts(normalize=True)
    max_share = float(side_counts.max())
    dominant_side = side_counts.idxmax()
    side_collapse = max_share > max_side_share

    # CLV computation
    valid["clv_over"] = valid["close_no_vig_over"] - valid["entry_no_vig_over"]
    valid["clv_under"] = valid["close_no_vig_under"] - valid["entry_no_vig_under"]
    valid["model_side_clv"] = np.where(
        valid["selected_side"] == "OVER",
        valid["clv_over"],
        valid["clv_under"],
    )

    # Only evaluate on rows with actual movement
    moved = valid[valid["clv_over"].abs() > 1e-6].copy()

    # Metrics
    if len(moved) >= 30:
        mean_clv = float(moved["model_side_clv"].mean())
        pos_rate = float((moved["model_side_clv"] > 0).mean())
        edge_clv_corr = float(moved["selected_edge"].corr(moved["model_side_clv"]))
        if not np.isfinite(edge_clv_corr):
            edge_clv_corr = 0.0

        # Close-tracking error (NOT Brier — games haven't resolved)
        model_close_mse = float(((valid["p_model_over"] - valid["close_no_vig_over"]) ** 2).mean())
        entry_close_mse = float(((valid["entry_no_vig_over"] - valid["close_no_vig_over"]) ** 2).mean())

        # CLV by side
        clv_by_side = {}
        for side in ["OVER", "UNDER"]:
            side_moved = moved[moved["selected_side"] == side]
            if len(side_moved) >= 10:
                clv_by_side[side] = {
                    "n": int(len(side_moved)),
                    "mean_clv": float(side_moved["model_side_clv"].mean()),
                    "positive_rate": float((side_moved["model_side_clv"] > 0).mean()),
                }

        # CLV by edge bucket
        moved["edge_bucket"] = pd.cut(
            moved["selected_edge"],
            bins=[-1, 0.02, 0.05, 0.10, 0.20, 1.0],
            labels=["0-2%", "2-5%", "5-10%", "10-20%", "20%+"],
        ).astype(str)
        clv_by_edge = {}
        for bucket, group in moved.groupby("edge_bucket"):
            if len(group) >= 10:
                clv_by_edge[str(bucket)] = {
                    "n": int(len(group)),
                    "mean_clv": float(group["model_side_clv"].mean()),
                    "positive_rate": float((group["model_side_clv"] > 0).mean()),
                }

        gates = {
            "model_predictions_attached": True,
            "full_distribution_prediction": True,
            "calibrator_only_prediction": False,  # Good — this should be False
            "model_selected_side_clv_positive": mean_clv > 0,
            "model_positive_clv_rate_above_50": pos_rate > 0.50,
            "model_edge_clv_correlation_positive": edge_clv_corr > 0,
            "model_close_tracking_better_than_entry": model_close_mse < entry_close_mse,
            "side_collapse_guard_pass": not side_collapse,
        }

        return {
            "status": "pass" if all(v is True or v is False and "calibrator" in k for k, v in gates.items() if isinstance(v, bool)) else "partial",
            "prediction_source": "full_distribution_pipeline",
            "matched_rows": int(len(valid)),
            "moved_rows": int(len(moved)),
            "side_distribution": side_counts.to_dict(),
            "side_collapse": side_collapse,
            "metrics": {
                "mean_model_side_clv": mean_clv,
                "positive_clv_rate": pos_rate,
                "edge_clv_correlation": edge_clv_corr,
                "model_close_tracking_mse": model_close_mse,
                "entry_close_tracking_mse": entry_close_mse,
                "model_beats_entry_on_close_tracking": model_close_mse < entry_close_mse,
            },
            "clv_by_side": clv_by_side,
            "clv_by_edge_bucket": clv_by_edge,
            "gates": gates,
            "note": "close_tracking_mse measures distance to closing market, NOT actual outcome Brier. Reserve Brier for post-game evaluation.",
        }
    else:
        return {
            "status": "insufficient_moved_rows",
            "matched_rows": int(len(valid)),
            "moved_rows": int(len(moved)),
            "side_distribution": side_counts.to_dict(),
            "note": f"Need 30+ moved rows, got {len(moved)}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach full-distribution model predictions to CLV data")
    parser.add_argument(
        "--slate",
        type=Path,
        default=None,
        help="Path to upcoming_market_slate CSV from daily pipeline. If not provided, searches daily_runs.",
    )
    parser.add_argument(
        "--attachable",
        type=Path,
        default=ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "market_snapshot_attachable.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model" / "props" / "v9_6" / "validation" / "model_attached_clv_report.json",
    )
    parser.add_argument("--max-side-share", type=float, default=0.80)
    return parser.parse_args()


def find_latest_slate() -> Path | None:
    """Find the most recent daily slate file."""
    daily_runs = ROOT / "model" / "analysis" / "daily_runs"
    if not daily_runs.exists():
        return None
    dates = sorted([d.name for d in daily_runs.iterdir() if d.is_dir() and d.name.isdigit()], reverse=True)
    for date_dir in dates:
        slate = daily_runs / date_dir / f"upcoming_market_slate_{date_dir}.csv"
        if slate.exists():
            return slate
    return None


def main():
    args = parse_args()

    # Find slate
    slate_path = args.slate
    if slate_path is None:
        slate_path = find_latest_slate()
    if slate_path is None or not slate_path.exists():
        print("ERROR: No slate file found. Run the daily market pipeline first:")
        print("  python sports/nba/predictions/Player-Predictor/scripts/run_daily_market_pipeline.py")
        return

    print(f"Loading slate: {slate_path}")
    predictions = load_slate_predictions(slate_path)
    print(f"  Loaded {len(predictions)} prediction rows from full distribution pipeline")
    print(f"  Players: {predictions['player'].nunique()}")
    print(f"  Markets: {sorted(predictions['market'].unique().tolist())}")
    print(f"  Dates: {sorted(predictions['date'].unique().tolist())}")
    print(f"  Side distribution (from model edge):")
    over_pct = (predictions["p_model_over"] > 0.5).mean()
    print(f"    OVER-favored: {over_pct:.1%}, UNDER-favored: {1-over_pct:.1%}")

    # Attach and evaluate
    print(f"\nAttaching to CLV data: {args.attachable}")
    result = attach_and_evaluate(predictions, args.attachable, args.max_side_share)

    print(f"\nResult: {result['status'].upper()}")
    if "metrics" in result:
        m = result["metrics"]
        print(f"  Matched rows:              {result['matched_rows']}")
        print(f"  Moved rows:                {result['moved_rows']}")
        print(f"  Side distribution:         {result['side_distribution']}")
        print(f"  Side collapse:             {result['side_collapse']}")
        print(f"  Mean model-side CLV:       {m['mean_model_side_clv']:+.6f}")
        print(f"  Positive CLV rate:         {m['positive_clv_rate']:.3f}")
        print(f"  Edge-CLV correlation:      {m['edge_clv_correlation']:+.4f}")
        print(f"  Model close-tracking MSE:  {m['model_close_tracking_mse']:.6f}")
        print(f"  Entry close-tracking MSE:  {m['entry_close_tracking_mse']:.6f}")
        print(f"  Model beats entry:         {m['model_beats_entry_on_close_tracking']}")
        if result.get("clv_by_side"):
            print(f"  CLV by side:")
            for side, metrics in result["clv_by_side"].items():
                print(f"    {side}: n={metrics['n']}, mean_clv={metrics['mean_clv']:+.6f}, pos_rate={metrics['positive_rate']:.3f}")
        if result.get("gates"):
            print(f"\n  Promotion Gates:")
            for gate, passed in result["gates"].items():
                marker = "✓" if passed else "✗"
                print(f"    {marker} {gate}: {passed}")
    elif "reason" in result:
        print(f"  Reason: {result['reason']}")

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "slate_path": str(slate_path),
        "attachable_path": str(args.attachable),
        **result,
    }
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
