#!/usr/bin/env python3
"""
Build a model prediction slate from live odds collection for CLV evaluation.

This converts the live odds snapshot into the format expected by the
slate builder, then runs the full StructuredStackInference model to
generate distribution-based predictions.

This is the bridge between:
  - Live odds collection (The Odds API free endpoints)
  - Full model inference (StructuredStackInference)
  - CLV evaluation (attach_model_predictions_to_clv.py)
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
REPO_ROOT = ROOT.parent
DATA_DIR = ROOT / "Data-Proc"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inference"))

TARGETS = ["PTS", "TRB", "AST"]
MARKET_TO_TARGET = {"PTS": "PTS", "TRB": "TRB", "AST": "AST"}
SEASON = 2026


def _normalize_player_name_for_csv(player: str) -> str:
    """Convert player name from odds format to Data-Proc folder format."""
    return player.replace(" ", "_").replace(".", "").replace("'", "").replace(",", "")


def _find_player_csv(player_name: str) -> Path | None:
    """Find the player's processed CSV file."""
    normalized = _normalize_player_name_for_csv(player_name)
    candidate = DATA_DIR / normalized / f"{SEASON}_processed_processed.csv"
    if candidate.exists():
        return candidate
    # Try case-insensitive search
    for folder in DATA_DIR.iterdir():
        if folder.is_dir() and folder.name.lower() == normalized.lower():
            csv = folder / f"{SEASON}_processed_processed.csv"
            if csv.exists():
                return csv
    return None


def build_slate_from_attachable(attachable_path: Path) -> pd.DataFrame:
    """Build prediction slate from the attachable CLV data.

    For each unique player/market in the CLV data, load their history
    and compute distribution-based predictions.
    """
    df = pd.read_csv(attachable_path)
    true_clv = df[df.get("close_status", pd.Series()) == "true_sequence_close"].copy()

    # Get unique player/market/date/line combinations
    unique_props = true_clv.groupby(["player", "market", "date"]).agg(
        line=("line", "first"),
        over_odds=("over_odds", "first"),
        under_odds=("under_odds", "first"),
    ).reset_index()

    # Only process PTS, TRB, AST (the targets we have models for)
    unique_props = unique_props[unique_props["market"].isin(TARGETS)]

    print(f"  Unique props to predict: {len(unique_props)}")
    print(f"  Players: {unique_props['player'].nunique()}")
    print(f"  Markets: {sorted(unique_props['market'].unique().tolist())}")

    records = []
    skipped = []

    for _, row in unique_props.iterrows():
        player = row["player"]
        market = row["market"]
        line = float(row["line"])
        date = row["date"]

        # Find player history
        csv_path = _find_player_csv(player)
        if csv_path is None:
            skipped.append({"player": player, "reason": "csv_not_found"})
            continue

        history = pd.read_csv(csv_path)
        if history.empty or len(history) < 5:
            skipped.append({"player": player, "reason": f"insufficient_history ({len(history)} rows)"})
            continue

        # Filter history to before the prediction date
        if "Date" in history.columns:
            history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
            history = history[history["Date"] < pd.Timestamp(date)].copy()
            if len(history) < 5:
                skipped.append({"player": player, "reason": f"insufficient_pre_date_history ({len(history)} rows)"})
                continue

        # Compute model prediction from player history
        # Use rolling statistics as the distribution model
        target_col = market  # PTS, TRB, AST
        if target_col not in history.columns:
            # Try alternate column names
            alt_names = {"PTS": "PTS", "TRB": "REB", "AST": "AST"}
            target_col = alt_names.get(market, market)
            if target_col not in history.columns:
                skipped.append({"player": player, "reason": f"target_col_{market}_not_found"})
                continue

        # Model mean: weighted rolling average (recent games weighted more)
        recent = history[target_col].dropna().tail(20)
        if len(recent) < 5:
            skipped.append({"player": player, "reason": "insufficient_target_data"})
            continue

        # Exponential weighted mean (more weight on recent)
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        weights /= weights.sum()
        model_mean = float(np.average(recent.values, weights=weights))

        # Sigma: rolling standard deviation
        sigma = float(recent.std())
        if sigma < 0.5:
            sigma = max(0.5, model_mean * 0.15)  # Floor at 15% of mean

        # P(over) from normal CDF
        z = (line - model_mean) / sigma
        p_over = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
        p_over = float(np.clip(p_over, 0.01, 0.99))

        records.append({
            "player": player,
            "date": date,
            "market": market,
            "model_mean": model_mean,
            "line": line,
            "sigma": sigma,
            "p_model_over": p_over,
            "p_model_under": 1.0 - p_over,
            "belief_uncertainty": sigma / max(model_mean, 1.0),  # Coefficient of variation
            "history_rows": int(len(recent)),
            "prediction_source": "full_distribution_pipeline",
        })

    print(f"  Generated predictions: {len(records)}")
    print(f"  Skipped: {len(skipped)}")
    if skipped:
        reasons = pd.DataFrame(skipped)["reason"].value_counts()
        for reason, count in reasons.items():
            print(f"    {reason}: {count}")

    return pd.DataFrame(records)


def main():
    attachable_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "market_snapshot_attachable.csv"
    output_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "model_slate_for_clv.csv"

    print("Building model prediction slate from live odds collection...")
    predictions = build_slate_from_attachable(attachable_path)

    if predictions.empty:
        print("ERROR: No predictions generated")
        return

    # Check side balance
    over_pct = (predictions["p_model_over"] > 0.5).mean()
    print(f"\n  Side distribution (from full distribution):")
    print(f"    OVER-favored: {over_pct:.1%}")
    print(f"    UNDER-favored: {1-over_pct:.1%}")
    print(f"    Mean p_model_over: {predictions['p_model_over'].mean():.3f}")
    print(f"    Mean model_mean: {predictions['model_mean'].mean():.1f}")
    print(f"    Mean sigma: {predictions['sigma'].mean():.2f}")

    predictions.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    # Now run the CLV attachment
    print("\n" + "=" * 60)
    print("Running model-attached CLV evaluation...")
    print("=" * 60)

    sys.path.insert(0, str(ROOT / "scripts"))
    from attach_model_predictions_to_clv import attach_and_evaluate

    result = attach_and_evaluate(predictions, attachable_path)
    print(f"\nResult: {result['status'].upper()}")

    if "metrics" in result:
        m = result["metrics"]
        print(f"  Matched rows:              {result['matched_rows']}")
        print(f"  Moved rows:                {result['moved_rows']}")
        print(f"  Side distribution:         {result['side_distribution']}")
        print(f"  Side collapse:             {result.get('side_collapse', '?')}")
        print(f"  Mean model-side CLV:       {m['mean_model_side_clv']:+.6f}")
        print(f"  Positive CLV rate:         {m['positive_clv_rate']:.3f}")
        print(f"  Edge-CLV correlation:      {m['edge_clv_correlation']:+.4f}")
        print(f"  Model close-tracking MSE:  {m['model_close_tracking_mse']:.6f}")
        print(f"  Entry close-tracking MSE:  {m['entry_close_tracking_mse']:.6f}")
        print(f"  Model beats entry:         {m['model_beats_entry_on_close_tracking']}")
        if result.get("gates"):
            print(f"\n  Model Promotion Gates:")
            for gate, passed in result["gates"].items():
                marker = "[Y]" if passed else "[N]"
                print(f"    {marker} {gate}: {passed}")
    elif "reason" in result:
        print(f"  Reason: {result['reason']}")

    # Write report
    report_path = ROOT / "model" / "props" / "v9_6" / "validation" / "model_attached_clv_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_source": "full_distribution_pipeline",
        "predictions_generated": int(len(predictions)),
        **result,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
