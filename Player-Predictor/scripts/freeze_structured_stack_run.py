#!/usr/bin/env python3
"""
Freeze the current generic structured-stack artifacts into an immutable run directory.

This is a one-way safety helper for older runs that saved only to model/lstm_v7_*.
It snapshots the current artifact set under model/runs/<run_id>/ and writes a
latest manifest that points at the snapshot. If the production manifest already
uses immutable run artifacts, this script leaves it alone.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def uses_immutable_run_artifacts(manifest: dict) -> bool:
    metadata_path = manifest.get("artifact_paths", {}).get("metadata", "")
    return isinstance(metadata_path, str) and "model/runs/" in metadata_path.replace("\\", "/")


def main():
    production_path = MODEL_DIR / "production_structured_lstm_stack.json"
    metadata_path = MODEL_DIR / "lstm_v7_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    production = read_json(production_path) if production_path.exists() else {}
    metadata = read_json(metadata_path)

    run_id = metadata.get("run_id") or f"lstm_v7_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_names = [
        "lstm_v7_member_0.weights.h5",
        "lstm_v7_member_1.weights.h5",
        "lstm_v7_member_2.weights.h5",
        "lstm_v7_pts_branch.weights.h5",
        "lstm_v7_catboost_models.pkl",
        "lstm_v7_meta_models.pkl",
        "lstm_v7_scaler_x.pkl",
        "lstm_v7_scaler_y.pkl",
        "lstm_v7_metadata.json",
    ]
    copied = []
    for name in artifact_names:
        src = MODEL_DIR / name
        if not src.exists():
            continue
        dst = run_dir / name
        shutil.copy2(src, dst)
        copied.append(dst)

    run_artifact_paths = {
        "lstm_weights": [str((run_dir / f"lstm_v7_member_{idx}.weights.h5").as_posix()) for idx in range(3) if (run_dir / f"lstm_v7_member_{idx}.weights.h5").exists()],
        "pts_branch_weights": str((run_dir / "lstm_v7_pts_branch.weights.h5").as_posix()),
        "catboost_models": str((run_dir / "lstm_v7_catboost_models.pkl").as_posix()),
        "meta_models": str((run_dir / "lstm_v7_meta_models.pkl").as_posix()),
        "scaler_x": str((run_dir / "lstm_v7_scaler_x.pkl").as_posix()),
        "scaler_y": str((run_dir / "lstm_v7_scaler_y.pkl").as_posix()),
        "metadata": str((run_dir / "lstm_v7_metadata.json").as_posix()),
    }

    latest_manifest = {
        "run_id": run_id,
        "avg_mae": float(metadata.get("avg_mae", 0.0)),
        "best_method": metadata.get("best_method"),
        "artifact_paths": run_artifact_paths,
    }
    write_json(MODEL_DIR / "latest_structured_lstm_stack.json", latest_manifest)

    if production and not uses_immutable_run_artifacts(production):
        print("Production manifest still points at shared generic artifacts.")
        print("Latest manifest was updated to an immutable snapshot instead.")
        print("Promote a future run, or manually review before repointing production.")
    else:
        print("Production manifest already uses immutable run artifacts or does not exist.")

    print(f"Snapshot written to: {run_dir}")
    print(f"Copied {len(copied)} artifact(s).")


if __name__ == "__main__":
    main()
