#!/usr/bin/env python3
"""
Promote an immutable structured-stack run directory to production.

Usage:
  python scripts/promote_structured_stack_run.py --run-id lstm_v7_20260317_031337
  python scripts/promote_structured_stack_run.py --run-dir model/runs/lstm_v7_20260317_031337

This writes model/production_structured_lstm_stack.json to point at the chosen
run's artifacts. It does not copy files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"
RUNS_DIR = MODEL_DIR / "runs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(args) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        return run_dir
    if args.run_id:
        return RUNS_DIR / args.run_id
    latest_path = MODEL_DIR / "latest_structured_lstm_stack.json"
    if latest_path.exists():
        latest = read_json(latest_path)
        run_id = latest.get("run_id")
        if run_id:
            return RUNS_DIR / run_id
    raise FileNotFoundError("No run specified and no latest run manifest found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args)
    metadata_path = run_dir / "lstm_v7_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")

    metadata = read_json(metadata_path)
    artifact_paths = metadata.get("artifact_paths")
    if not artifact_paths:
        artifact_paths = {
            "lstm_weights": [str((run_dir / f"lstm_v7_member_{idx}.weights.h5").as_posix()) for idx in range(3)],
            "pts_branch_weights": str((run_dir / "lstm_v7_pts_branch.weights.h5").as_posix()),
            "catboost_models": str((run_dir / "lstm_v7_catboost_models.pkl").as_posix()),
            "meta_models": str((run_dir / "lstm_v7_meta_models.pkl").as_posix()),
            "scaler_x": str((run_dir / "lstm_v7_scaler_x.pkl").as_posix()),
            "scaler_y": str((run_dir / "lstm_v7_scaler_y.pkl").as_posix()),
            "metadata": str(metadata_path.as_posix()),
        }

    production_manifest = {
        "model_type": "structured_lstm_stack_targetwise",
        "avg_mae": float(metadata["avg_mae"]),
        "best_method": metadata["best_method"],
        "target_columns": metadata["target_columns"],
        "targetwise_methods": metadata.get("targetwise_methods", ["catboost_delta"] * len(metadata["target_columns"])),
        "run_id": metadata.get("run_id", run_dir.name),
        "artifact_paths": artifact_paths,
    }
    output_path = MODEL_DIR / "production_structured_lstm_stack.json"
    output_path.write_text(json.dumps(production_manifest, indent=2), encoding="utf-8")

    best_ever_path = MODEL_DIR / "lstm_v7_best_ever.json"
    best_ever = read_json(best_ever_path) if best_ever_path.exists() else None
    if best_ever is None or float(best_ever.get("avg_mae", 999.0)) > float(metadata["avg_mae"]):
        best_ever_payload = {
            "avg_mae": float(metadata["avg_mae"]),
            "best_method": metadata["best_method"],
            "run_id": metadata.get("run_id", run_dir.name),
            "target_columns": metadata["target_columns"],
            "catboost_model_info": metadata.get("catboost_model_info", []),
            "artifact_paths": artifact_paths,
            "rolling_window_validation": metadata.get("rolling_window_validation", {}),
        }
        best_ever_path.write_text(json.dumps(best_ever_payload, indent=2), encoding="utf-8")
        print(f"Best-ever registry updated: {best_ever_path}")

    print(f"Promoted run: {run_dir}")
    print(f"Production manifest written to: {output_path}")
    print(f"avg_mae: {production_manifest['avg_mae']:.4f}")


if __name__ == "__main__":
    main()
