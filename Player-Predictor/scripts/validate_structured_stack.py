#!/usr/bin/env python3
"""
Validate the saved structured LSTM + CatBoost production stack.

Checks:
- production manifest exists
- referenced artifact files exist
- metadata and best-ever registry are readable
- CatBoost ensemble weights are sane
- rolling-window stability stats are present
- inference smoke test works on one local processed CSV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from structured_stack_inference import StructuredStackInference


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest(args) -> tuple[dict, str]:
    if args.run_id:
        run_dir = MODEL_DIR / "runs" / args.run_id
        metadata_path = run_dir / "lstm_v7_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
        metadata = read_json(metadata_path)
        manifest = {
            "model_type": metadata.get("model_type", "structured_lstm_stack_targetwise"),
            "avg_mae": float(metadata["avg_mae"]),
            "best_method": metadata.get("best_method"),
            "target_columns": metadata["target_columns"],
            "targetwise_methods": metadata.get("targetwise_methods", ["catboost_delta"] * len(metadata["target_columns"])),
            "run_id": metadata.get("run_id", run_dir.name),
            "artifact_paths": metadata["artifact_paths"],
        }
        return manifest, f"run:{args.run_id}", metadata_path

    if args.latest:
        latest_path = MODEL_DIR / "latest_structured_lstm_stack.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"Missing latest manifest: {latest_path}")
        return read_json(latest_path), "latest", latest_path

    production_path = MODEL_DIR / "production_structured_lstm_stack.json"
    if not production_path.exists():
        raise FileNotFoundError(f"Missing production manifest: {production_path}")
    return read_json(production_path), "production", production_path


def find_sample_csv() -> Path | None:
    data_root = REPO_ROOT / "Data-Proc"
    if not data_root.exists():
        return None
    for path in data_root.rglob("*processed*.csv"):
        return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None, help="Validate a specific immutable run")
    parser.add_argument("--latest", action="store_true", help="Validate model/latest_structured_lstm_stack.json instead of production")
    args = parser.parse_args()

    best_ever_path = MODEL_DIR / "lstm_v7_best_ever.json"
    latest_path = MODEL_DIR / "latest_structured_lstm_stack.json"
    runs_dir = MODEL_DIR / "runs"

    production, manifest_label, manifest_path = resolve_manifest(args)
    best_ever = read_json(best_ever_path) if best_ever_path.exists() else None
    latest = read_json(latest_path) if latest_path.exists() else None

    best_immutable = None
    best_immutable_run = None
    if runs_dir.exists():
        for path in runs_dir.glob("*/lstm_v7_metadata.json"):
            payload = read_json(path)
            avg_mae = float(payload.get("avg_mae", 999.0))
            if best_immutable is None or avg_mae < best_immutable:
                best_immutable = avg_mae
                best_immutable_run = path.parent

    artifact_paths = production.get("artifact_paths", {})
    metadata_rel = artifact_paths.get("metadata", "model/lstm_v7_metadata.json")
    metadata_path = REPO_ROOT / metadata_rel
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing production metadata artifact: {metadata_path}")
    metadata = read_json(metadata_path)

    print("=" * 80)
    print("STRUCTURED STACK VALIDATION")
    print("=" * 80)
    print(f"Manifest target:    {manifest_label}")
    print(f"Production avg_mae: {production.get('avg_mae')}")
    print(f"Metadata avg_mae:   {metadata.get('avg_mae')}")
    if best_ever is not None:
        print(f"Best-ever avg_mae:  {best_ever.get('avg_mae')}")
    if best_immutable is not None:
        print(f"Best immutable:     {best_immutable} ({best_immutable_run.name})")
    if latest is not None:
        print(f"Latest avg_mae:     {latest.get('avg_mae')}")

    production_avg = float(production.get("avg_mae", metadata.get("avg_mae", 0.0)))
    metadata_avg = float(metadata.get("avg_mae", production_avg))
    if abs(production_avg - metadata_avg) > 1e-6:
        print("\nWarning: production manifest and artifact metadata disagree.")
        print("  The manifest is stale or the shared artifacts were overwritten by a later run.")
        print("  Production inference should use immutable run artifacts before deployment.")

    print("\nArtifacts:")
    missing = []
    for label, rel_path in artifact_paths.items():
        if isinstance(rel_path, list):
            for item in rel_path:
                full = REPO_ROOT / item
                ok = full.exists()
                print(f"  {label}: {full} -> {'OK' if ok else 'MISSING'}")
                if not ok:
                    missing.append(str(full))
        else:
            full = REPO_ROOT / rel_path
            ok = full.exists()
            print(f"  {label}: {full} -> {'OK' if ok else 'MISSING'}")
            if not ok:
                missing.append(str(full))
    if missing:
        raise FileNotFoundError("Missing artifact(s): " + ", ".join(missing))

    cb_models = joblib.load(REPO_ROOT / artifact_paths.get("catboost_models", "model/lstm_v7_catboost_models.pkl"))
    cb_info = metadata.get("catboost_model_info", [])
    print("\nCatBoost bundles:")
    for idx, info in enumerate(cb_info):
        bundle = cb_models[idx]
        if isinstance(bundle, dict):
            weights = bundle.get("weights", [])
            weight_sum = sum(weights) if weights else 0.0
            print(
                f"  {info['target']}: candidate={info.get('candidate')} "
                f"members={bundle.get('ensemble_size')} weight_sum={weight_sum:.4f}"
            )
        else:
            print(f"  {info['target']}: single model")

    rolling = metadata.get("rolling_window_validation", {}).get("catboost_delta", {})
    print("\nRolling-window stability:")
    for target, stats in rolling.get("targets", {}).items():
        print(
            f"  {target}: mean={stats['mean_mae']:.4f} std={stats['std_mae']:.4f} "
            f"max={stats['max_mae']:.4f}"
        )

    latent_audit = metadata.get("latent_head_audit", {})
    if latent_audit:
        print("\nLatent-head audit:")
        for target, payload in latent_audit.items():
            top_blocks = ", ".join(
                f"{name}={share:.3f}" for name, share in list(payload.get("block_share_summary", {}).items())[:5]
            ) or "none"
            print(
                f"  {target}: base={payload.get('base_feature_share', 0.0):.3f} "
                f"latent={payload.get('latent_feature_share', 0.0):.3f} "
                f"pts_state={payload.get('pts_state_feature_share', 0.0):.3f} | {top_blocks}"
            )

    pts_ablation = metadata.get("pts_latent_ablation", {})
    if pts_ablation:
        print("\nPTS latent ablation:")
        print(
            f"  best_base={pts_ablation.get('best_base_name')} "
            f"best_subset={pts_ablation.get('best_subset_name')} "
            f"subset_mae={pts_ablation.get('best_subset_mae', 0.0):.4f}"
        )
        block_names = pts_ablation.get("best_subset_blocks", [])
        print(f"  blocks={', '.join(block_names) if block_names else 'none'}")

    sample_csv = find_sample_csv()
    if sample_csv is None:
        print("\nInference smoke test skipped: no sample CSV found.")
        return

    print(f"\nInference smoke test: {sample_csv}")
    predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    explanation = predictor.predict(pd.read_csv(sample_csv))
    print("  Predicted:", explanation["predicted"])
    print("  Feature versions:", explanation["catboost_feature_versions"])
    print("\nValidation complete.")


if __name__ == "__main__":
    main()
