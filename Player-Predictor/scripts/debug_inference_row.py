#!/usr/bin/env python3
"""
Debug one walk-forward inference row end-to-end.

This script helps isolate:
- baseline vs delta reconstruction issues
- inference feature parity issues
- fallback / guardrail behavior
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from structured_stack_inference import StructuredStackInference  # noqa: E402


MODEL_DIR = REPO_ROOT / "model"


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path | None:
    if run_id:
        return MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return MODEL_DIR / "latest_structured_lstm_stack.json"
    return MODEL_DIR / "production_structured_lstm_stack.json"


def normalize_input_frame(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    out = df.copy()
    if "Player" not in out.columns:
        out["Player"] = player_name
    if "Game_Index" not in out.columns:
        out["Game_Index"] = np.arange(len(out), dtype=np.int32)
    return out


def is_prepared_frame(df: pd.DataFrame, required_columns: list[str]) -> bool:
    return all(column in df.columns for column in required_columns)


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_dict(title: str, payload: dict):
    print(f"\n{title}:")
    for key, value in payload.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Debug one walk-forward inference row.")
    parser.add_argument("--csv", required=True, help="Processed player CSV")
    parser.add_argument("--target-index", type=int, required=True, help="Row index to predict using all prior rows as history")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save debug payload")
    args = parser.parse_args()
    np.random.seed(42)

    csv_path = Path(args.csv)
    player_name = csv_path.parent.name
    raw_df = normalize_input_frame(pd.read_csv(csv_path), player_name)
    if args.target_index <= 0 or args.target_index >= len(raw_df):
        raise ValueError(f"--target-index must be between 1 and {len(raw_df) - 1}")

    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    assume_prepared = is_prepared_frame(raw_df, predictor.feature_columns)

    history_df = raw_df.iloc[: args.target_index].copy()
    actual_row = raw_df.iloc[args.target_index]
    explanation = predictor.predict(history_df, assume_prepared=assume_prepared, return_debug=True)

    print_section("INFERENCE ROW DEBUG")
    print(f"Manifest target: {manifest_path}")
    print(f"Run id: {predictor.metadata.get('run_id')}")
    print(f"CSV: {csv_path}")
    print(f"Player: {player_name}")
    print(f"Target index: {args.target_index}")
    print(f"History rows used: {len(history_df)}")
    print(f"Assume prepared: {assume_prepared}")

    recent_cols = [c for c in ["Date", "PTS", "TRB", "AST", "MP", "USG%", "Rest_Days", "Did_Not_Play", "PTS_rolling_avg", "TRB_rolling_avg", "AST_rolling_avg"] if c in raw_df.columns]
    print("\nRecent history:")
    print(history_df.tail(5)[recent_cols].to_string(index=True))

    actual_payload = {k: actual_row[k] for k in recent_cols if k in actual_row.index}
    print_dict("Actual target row", actual_payload)
    print_dict("Baseline raw", explanation["debug"]["baseline_raw"])
    print_dict("Baseline scaled", explanation["debug"]["baseline_scaled"])
    print_dict("CatBoost delta scaled", explanation["debug"]["catboost_delta_scaled"])
    print_dict("Raw model prediction", explanation["debug"]["predicted_raw_model"])
    print_dict("Prediction post-split pre-guard", explanation["debug"]["predicted_post_split_pre_guard"])
    print_dict("Prediction post-fallback", explanation["debug"]["predicted_post_fallback"])
    print_dict("Data quality", explanation["data_quality"])
    print_dict("Latent environment", explanation["latent_environment"])
    if explanation["data_quality"].get("nan_feature_columns"):
        print("\nNaN feature columns:")
        for column in explanation["data_quality"]["nan_feature_columns"]:
            print(f"  {column}")
    print_dict("PTS residual split", explanation["debug"]["pts_residual_split"])

    print("\nTarget comparison:")
    for target in predictor.target_columns:
        pred = explanation["predicted"][target]
        actual = float(actual_row[target])
        base = explanation["baseline"][target]
        print(
            f"  {target}: baseline={base:.2f} pred={pred:.2f} actual={actual:.2f} "
            f"err={pred - actual:+.2f} pred_minus_base={pred - base:+.2f}"
        )

    print("\nFeature versions:")
    print(json.dumps(explanation["catboost_feature_versions"], indent=2))

    print("\nFeature set stats:")
    for name, stats in explanation["debug"]["feature_set_stats"].items():
        print(
            f"  {name}: shape={stats['shape']} nan_count={stats['nan_count']} "
            f"zero_fraction={stats['zero_fraction']:.3f} min={stats['min']:.3f} "
            f"max={stats['max']:.3f} mean={stats['mean']:.3f} std={stats['std']:.3f}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_path": str(manifest_path),
            "run_id": predictor.metadata.get("run_id"),
            "csv": str(csv_path),
            "player": player_name,
            "target_index": int(args.target_index),
            "actual_row": {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.generic)) else v)) for k, v in actual_payload.items()},
            "explanation": explanation,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved JSON debug payload: {out_path}")


if __name__ == "__main__":
    main()
