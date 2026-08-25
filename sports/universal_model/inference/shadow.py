"""SHADOW UNIVERSAL PREDICTOR integration (spec sections 31/52/54).

Produces a daily comparison artifact between the existing, validated
per-sport MLB predictor's real output and the universal model's real
output for the same real players/targets/lines -- never replacing the
existing predictor, never feeding the universal model's confidence into
PolicyStatus/G_C/G_L/G_V (this module imports nothing from
sports.mlb.research.parlay_certification_v2 and writes only a comparison
report, never a certification decision).

Run: python -m sports.universal_model.inference.shadow --date 20260811
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sports.universal_model.inference.predict import predict_for_date

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def build_shadow_comparison(date_compact: str, checkpoint_name: str = "drm_final") -> list[dict]:
    """date_compact: 'YYYYMMDD', matching the existing daily_runs directory
    naming convention."""
    pool_path = REPO_ROOT / f"sports/mlb/data/predictions/daily_runs/{date_compact}/daily_prediction_pool_{date_compact}.csv"
    if not pool_path.exists():
        return []
    existing = pd.read_csv(pool_path, low_memory=False)
    date_iso = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:]}"

    checkpoint_path = Path(__file__).resolve().parents[1] / "manifests" / "checkpoints" / f"{checkpoint_name}.pt"
    universal_rows = predict_for_date("mlb", date_iso, checkpoint_path)
    universal_by_key = {(r["entity_id"], r["target"], r["line"]): r for r in universal_rows}

    comparisons = []
    for row in existing.itertuples(index=False):
        r = row._asdict()
        key = (str(r["Player_ID"]), str(r["Target"]), float(r["Market_Line"]) if pd.notna(r["Market_Line"]) else None)
        universal = universal_by_key.get(key)
        if universal is None:
            continue
        existing_pred = r.get("Prediction")
        market_line = r.get("Market_Line")
        existing_direction = None
        if pd.notna(existing_pred) and pd.notna(market_line):
            existing_direction = "over" if existing_pred > market_line else "under"
        universal_direction = "over" if universal["prob_over"] > 0.5 else "under"
        comparisons.append(
            {
                "sport": "mlb",
                "date": date_iso,
                "player": r.get("Player"),
                "player_id": r.get("Player_ID"),
                "target": r.get("Target"),
                "line": market_line,
                "existing_model_point_prediction": existing_pred,
                "existing_model_direction": existing_direction,
                "universal_model_prob_over": universal["prob_over"],
                "universal_model_direction": universal_direction,
                "agree": existing_direction == universal_direction if existing_direction else None,
                "market_probability": universal.get("market_probability"),
                "eventual_outcome": None,  # not yet settled at generation time; joined later from settled history if needed
                "existing_model_version": "mlb_daily_pool_pipeline",
                "universal_model_checkpoint": checkpoint_name,
            }
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    args = parser.parse_args()
    comparisons = build_shadow_comparison(args.date)
    out_path = REPORTS_DIR / f"shadow_comparison_{args.date}.json"
    out_path.write_text(json.dumps(comparisons, indent=2))
    agree_rate = (
        sum(1 for c in comparisons if c["agree"]) / sum(1 for c in comparisons if c["agree"] is not None)
        if any(c["agree"] is not None for c in comparisons)
        else None
    )
    print(f"wrote {len(comparisons)} comparisons to {out_path}; direction agreement rate: {agree_rate}")


if __name__ == "__main__":
    main()
