#!/usr/bin/env python3
"""
Reality-check inference validator for the structured stack.

This script is meant for practical failure analysis:
- run one or many CSVs through the chosen manifest
- show the recent context the model actually sees
- surface derived pressure / elasticity inputs
- flag suspicious input conditions
- print the production prediction breakdown
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
DATA_DIR = REPO_ROOT / "Data-Proc"


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path | None:
    if run_id:
        return MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return MODEL_DIR / "latest_structured_lstm_stack.json"
    return MODEL_DIR / "production_structured_lstm_stack.json"


def gather_csvs(csvs: list[str] | None, player_dir: str | None, limit: int) -> list[Path]:
    if csvs:
        return [Path(item).resolve() for item in csvs]
    if player_dir:
        root = Path(player_dir)
        return sorted(root.glob("*.csv"))[:limit]
    paths = sorted(DATA_DIR.glob("*/*.csv"))
    return paths[:limit]


def infer_player_name(path: Path) -> str:
    return path.parent.name


def normalize_input_frame(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    out = df.copy()
    if "Player" not in out.columns:
        out["Player"] = player_name
    if "Game_Index" not in out.columns:
        out["Game_Index"] = np.arange(len(out), dtype=np.int32)
    return out


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def summarize_recent_games(df: pd.DataFrame, n_rows: int = 3) -> list[dict]:
    cols = [col for col in ["Date", "PTS", "TRB", "AST", "MP", "FGA", "USG%", "Rest_Days", "Did_Not_Play"] if col in df.columns]
    rows = []
    for _, row in df.tail(n_rows).iterrows():
        payload = {}
        for col in cols:
            value = row[col]
            if isinstance(value, (np.generic,)):
                value = value.item()
            payload[col] = value
        rows.append(payload)
    return rows


def collect_context_snapshot(df_feat: pd.DataFrame) -> dict:
    row = df_feat.iloc[-1]
    keys = [
        "PTS", "TRB", "AST", "MP", "FGA", "USG%", "Rest_Days", "Did_Not_Play",
        "MP_pressure", "FGA_pressure", "USG_pressure",
        "Opportunity_Trend_Score", "Role_Instability_Score",
        "PTS_Trend_Trust", "Context_Pressure_Score", "Opportunity_Score",
        "PTS_Elasticity_Score", "PTS_Downside_Pressure",
        "Low_Rest_Flag", "Strong_Defense_Flag", "oppDfRtg_3",
    ]
    return {key: safe_float(row[key]) for key in keys if key in df_feat.columns}


def find_extreme_scaled_features(predictor: StructuredStackInference, df_feat: pd.DataFrame) -> list[tuple[str, float]]:
    seq_df = df_feat.tail(predictor.seq_len).copy()
    numeric_features = predictor.feature_columns[3:]
    numeric_values = seq_df[numeric_features].values.astype(np.float32)
    scaled = predictor.scaler_x.transform(numeric_values).astype(np.float32)
    abs_max = np.max(np.abs(scaled), axis=0)
    ranked = sorted(zip(numeric_features, abs_max.tolist()), key=lambda item: item[1], reverse=True)
    return [(name, value) for name, value in ranked[:8] if value >= 3.0]


def build_flags(context: dict, extreme_scaled: list[tuple[str, float]]) -> list[str]:
    flags = []
    if context.get("Low_Rest_Flag", 0.0) >= 0.5:
        flags.append("low_rest")
    if context.get("Strong_Defense_Flag", 0.0) >= 0.5:
        flags.append("strong_defense")
    if (context.get("Role_Instability_Score") or 0.0) >= 1.15:
        flags.append("role_instability")
    if (context.get("PTS_Elasticity_Score") or 0.0) >= 6.0:
        flags.append("high_pts_elasticity")
    if (context.get("Context_Pressure_Score") or 0.0) >= 1.0:
        flags.append("context_pressure")
    if (context.get("Did_Not_Play") or 0.0) >= 0.5:
        flags.append("recent_dnp")
    if extreme_scaled:
        flags.append("extreme_scaled_inputs")
    return flags


def run_single_check(predictor: StructuredStackInference, csv_path: Path) -> dict:
    player_name = infer_player_name(csv_path)
    raw_df = normalize_input_frame(pd.read_csv(csv_path), player_name)
    feat_df = predictor.feature_trainer.create_hybrid_features(raw_df.copy())
    explanation = predictor.predict(raw_df)
    context = collect_context_snapshot(feat_df)
    extreme_scaled = find_extreme_scaled_features(predictor, feat_df)
    report = {
        "csv": str(csv_path),
        "player": player_name,
        "n_rows": int(len(raw_df)),
        "recent_games": summarize_recent_games(raw_df),
        "context_snapshot": context,
        "flags": build_flags(context, extreme_scaled),
        "extreme_scaled_features": [{"feature": name, "max_abs_z": float(value)} for name, value in extreme_scaled],
        "prediction": explanation,
    }
    return report


def print_report(report: dict):
    print("\n" + "=" * 100)
    print(f"REALITY CHECK: {report['player']}")
    print("=" * 100)
    print(f"CSV: {report['csv']}")
    print(f"Rows available: {report['n_rows']}")
    print("\nRecent games:")
    for row in report["recent_games"]:
        print(f"  {row}")
    print("\nContext snapshot:")
    for key, value in report["context_snapshot"].items():
        if value is None:
            continue
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("\nFlags:")
    if report["flags"]:
        print(f"  {', '.join(report['flags'])}")
    else:
        print("  none")
    if report["extreme_scaled_features"]:
        print("\nExtreme scaled features:")
        for item in report["extreme_scaled_features"]:
            print(f"  {item['feature']}: max_abs_z={item['max_abs_z']:.2f}")
    pred = report["prediction"]
    print("\nBaseline vs predicted:")
    for target in pred["baseline"]:
        base = pred["baseline"][target]
        out = pred["predicted"][target]
        print(f"  {target}: baseline={base:.2f} predicted={out:.2f} delta={out - base:+.2f}")
    env = pred["latent_environment"]
    print("\nLatent environment:")
    for key in [
        "slow_state_strength", "environment_strength", "belief_uncertainty", "feasibility",
        "role_shift_risk", "volatility_regime_risk", "context_pressure_risk",
        "pts_trend_trust", "pts_baseline_trust", "pts_elasticity", "pts_opportunity_jump",
    ]:
        if key in env:
            print(f"  {key}: {env[key]:.3f}")
    print("\nTarget factors:")
    for target, factors in pred["target_factors"].items():
        print(
            f"  {target}: baseline={factors['baseline_anchor']:.2f} "
            f"normal={factors['normal_adjustment']:+.2f} tail={factors['tail_adjustment']:+.2f} "
            f"spike_prob={factors['spike_probability']:.3f} pred={factors['production_prediction']:.2f}"
        )
        extra_keys = [
            "continuation_probability", "reversion_probability", "opportunity_jump_probability",
            "trend_trust", "baseline_trust", "elasticity", "downside_risk",
        ]
        extra = {k: factors[k] for k in extra_keys if k in factors}
        if extra:
            print(f"    pts_state={json.dumps(extra)}")
    print("\nFeature versions:")
    print(f"  {pred['catboost_feature_versions']}")


def main():
    parser = argparse.ArgumentParser(description="Reality-check structured stack inference on one or more CSVs.")
    parser.add_argument("--csv", nargs="+", help="One or more CSVs to inspect")
    parser.add_argument("--player-dir", help="Directory of player CSVs to sample from")
    parser.add_argument("--limit", type=int, default=3, help="Max CSVs when using --player-dir or default sampling")
    parser.add_argument("--run-id", type=str, default=None, help="Validate against a specific immutable run")
    parser.add_argument("--latest", action="store_true", help="Validate against latest manifest instead of production")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    csv_paths = gather_csvs(args.csv, args.player_dir, args.limit)
    if not csv_paths:
        raise FileNotFoundError("No CSVs found for reality check")

    reports = [run_single_check(predictor, path) for path in csv_paths]
    print("\n" + "=" * 100)
    print("STRUCTURED STACK REALITY CHECK")
    print("=" * 100)
    print(f"Manifest target: {manifest_path}")
    print(f"Run id: {predictor.metadata.get('run_id')}")
    print(f"Average MAE: {predictor.metadata.get('avg_mae')}")
    for report in reports:
        print_report(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved JSON report: {out_path}")


if __name__ == "__main__":
    main()
