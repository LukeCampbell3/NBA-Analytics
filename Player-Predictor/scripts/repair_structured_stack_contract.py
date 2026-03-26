#!/usr/bin/env python3
"""Repair/normalize structured stack metadata contract against current artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "training")))

from improved_lstm_v7 import build_feature_spec
from structured_stack_contract import (
    ARTIFACT_CONTRACT_VERSION,
    build_schema_payload,
    build_schema_signature,
    normalize_catboost_model_info,
    validate_metadata_contract,
)
from unified_moe_trainer import UnifiedMoETrainer


def _resolve_path(value, model_dir: Path, fallback_name: str) -> Path:
    repo_root = model_dir.parent
    if isinstance(value, str) and value.strip():
        candidate = Path(value)
        return candidate if candidate.is_absolute() else repo_root / candidate
    return model_dir / fallback_name


def _load_manifest(model_dir: Path, explicit: str | None):
    if explicit:
        path = Path(explicit)
        return path, json.loads(path.read_text(encoding="utf-8"))
    for name in ["production_structured_lstm_stack.json", "latest_structured_lstm_stack.json"]:
        path = model_dir / name
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None, {}


def main():
    parser = argparse.ArgumentParser(description="Repair structured stack metadata contract")
    parser.add_argument("--model-dir", default="model", help="Model directory")
    parser.add_argument("--manifest", default=None, help="Optional explicit manifest path")
    parser.add_argument(
        "--force-schema-refresh",
        action="store_true",
        help="Rebuild schema from training data even if metadata already validates",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    manifest_path, manifest_payload = _load_manifest(model_dir, args.manifest)
    artifact_paths = manifest_payload.get("artifact_paths", {}) if isinstance(manifest_payload, dict) else {}

    metadata_path = _resolve_path(artifact_paths.get("metadata"), model_dir, "lstm_v7_metadata.json")
    scaler_x_path = _resolve_path(artifact_paths.get("scaler_x"), model_dir, "lstm_v7_scaler_x.pkl")
    scaler_y_path = _resolve_path(artifact_paths.get("scaler_y"), model_dir, "lstm_v7_scaler_y.pkl")
    catboost_path = _resolve_path(artifact_paths.get("catboost_models"), model_dir, "lstm_v7_catboost_models.pkl")
    schema_path = _resolve_path(artifact_paths.get("schema"), model_dir, "lstm_v7_feature_schema.json")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "No structured-stack artifacts are present yet. "
            "Run: python train.py --mode improved_lstm"
        )
    if not scaler_x_path.exists():
        raise FileNotFoundError(f"Scaler-X file not found: {scaler_x_path}")
    if not scaler_y_path.exists():
        raise FileNotFoundError(f"Scaler-Y file not found: {scaler_y_path}")
    if not catboost_path.exists():
        raise FileNotFoundError(f"CatBoost file not found: {catboost_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    cb_models = joblib.load(catboost_path)

    metadata["catboost_model_info"] = normalize_catboost_model_info(
        metadata.get("catboost_model_info"),
        metadata.get("target_columns", []),
        cb_models,
    )

    errors = validate_metadata_contract(metadata, scaler_x=scaler_x, scaler_y=scaler_y, cb_models=cb_models)
    if errors and not args.force_schema_refresh:
        print("Current metadata contract is invalid; rebuilding schema from training data.")
    elif not errors and not args.force_schema_refresh:
        print("Metadata contract is already valid.")
        return
    else:
        print("Force-refresh requested; rebuilding schema from training data.")

    trainer = UnifiedMoETrainer()
    trainer.prepare_data()

    feature_columns = list(trainer.feature_columns)
    target_columns = list(trainer.target_columns)
    baseline_features = list(trainer.baseline_features)
    seq_len = int(getattr(trainer, "config", {}).get("seq_len", metadata.get("seq_len", 10)))
    n_features = int(len(feature_columns))
    n_targets = int(len(target_columns))
    counts = {
        "players": int(len(getattr(trainer, "player_mapping", {}))),
        "teams": int(len(getattr(trainer, "team_mapping", {}))),
        "opponents": int(len(getattr(trainer, "opponent_mapping", {}))),
    }

    metadata["model_type"] = metadata.get("model_type", "structured_lstm_stack")
    metadata["artifact_contract_version"] = ARTIFACT_CONTRACT_VERSION
    metadata["feature_columns"] = feature_columns
    metadata["feature_spec"] = build_feature_spec(feature_columns)
    metadata["baseline_features"] = baseline_features
    metadata["target_columns"] = target_columns
    metadata["seq_len"] = seq_len
    metadata["n_features"] = n_features
    metadata["n_targets"] = n_targets
    metadata["counts"] = counts
    metadata["player_mapping"] = getattr(trainer, "player_mapping", {})
    metadata["team_mapping"] = getattr(trainer, "team_mapping", {})
    metadata["opponent_mapping"] = getattr(trainer, "opponent_mapping", {})
    metadata["scaler_x_n_features"] = int(getattr(scaler_x, "n_features_in_", max(0, n_features - 3)))
    metadata["scaler_y_n_features"] = int(getattr(scaler_y, "n_features_in_", n_targets))
    metadata["catboost_model_info"] = normalize_catboost_model_info(
        metadata.get("catboost_model_info"),
        target_columns,
        cb_models,
    )
    if "n_models" not in metadata:
        if isinstance(metadata.get("artifact_paths", {}).get("lstm_weights"), list):
            metadata["n_models"] = len(metadata["artifact_paths"]["lstm_weights"])
        else:
            metadata["n_models"] = 3

    metadata["schema_signature"] = build_schema_signature(
        feature_columns,
        target_columns,
        baseline_features,
        seq_len,
        n_features,
        n_targets,
        counts,
    )

    repaired_errors = validate_metadata_contract(
        metadata,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        cb_models=cb_models,
    )
    if repaired_errors:
        if any("scaler_x expects" in line or "n_features mismatch with scaler_x" in line for line in repaired_errors):
            lines = "\n".join(f"  - {line}" for line in repaired_errors)
            raise RuntimeError(
                "Current model artifacts are not compatible with the current trainer schema.\n"
                "A full structured-stack retrain is required before metadata can be repaired safely.\n"
                "Run: python train.py --mode improved_lstm\n"
                f"{lines}"
            )
        lines = "\n".join(f"  - {line}" for line in repaired_errors)
        raise ValueError(f"Repaired metadata is still invalid:\n{lines}")

    schema_payload = build_schema_payload(metadata)

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    if isinstance(manifest_payload, dict):
        manifest_payload.setdefault("artifact_paths", {})
        manifest_payload["artifact_paths"]["metadata"] = str(metadata_path.as_posix())
        manifest_payload["artifact_paths"]["schema"] = str(schema_path.as_posix())
        if manifest_path is not None:
            manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"Metadata repaired: {metadata_path}")
    print(f"Schema sidecar:   {schema_path}")
    if manifest_path is not None:
        print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
