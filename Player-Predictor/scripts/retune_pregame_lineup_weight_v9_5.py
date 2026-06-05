#!/usr/bin/env python3
"""Retune v9.5 pregame lineup weight without rebuilding availability joins."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _resolve(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("/workspace/"):
        return REPO_ROOT / text.replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / text).resolve()


def _normal_sf(x: np.ndarray) -> np.ndarray:
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return 1.0 - cdf


def _copy_tree_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retune v9.5 pregame lineup weight")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "model" / "props" / "v9_5_prelock_availability" / "manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineup-weight", type=float, required=True)
    parser.add_argument("--sigma-inflation-per-expected-out", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = _resolve(args.source_manifest)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_output = _resolve(Path(manifest["output"]))
    rows = pd.read_csv(source_output / "data" / "prop_training_rows.csv")
    base_mean_col = "v92_model_mean" if "v92_model_mean" in rows.columns else "model_mean"
    base_sigma_col = "v92_sigma" if "v92_sigma" in rows.columns else "sigma"
    rows["v95_pregame_lineup_model_mean"] = (
        pd.to_numeric(rows[base_mean_col], errors="coerce").fillna(rows["model_mean"])
        + args.lineup_weight * pd.to_numeric(rows["pregame_lineup_adjustment"], errors="coerce").fillna(0.0)
    )
    base_sigma = pd.to_numeric(rows[base_sigma_col], errors="coerce").fillna(rows.get("sigma", 3.0)).clip(lower=0.25)
    rows["v95_pregame_lineup_sigma"] = (
        base_sigma
        * (1.0 + args.sigma_inflation_per_expected_out * pd.to_numeric(rows["pregame_teammate_out_expected_count"], errors="coerce").fillna(0.0).clip(upper=5))
    ).clip(lower=0.25)
    z = (pd.to_numeric(rows["line"], errors="coerce") - rows["v95_pregame_lineup_model_mean"]) / rows["v95_pregame_lineup_sigma"]
    if "p_over_raw_v94_safe" not in rows.columns:
        rows["p_over_raw_v94_safe"] = rows["p_over_raw"]
    rows["p_over_raw"] = np.clip(_normal_sf(z.to_numpy(dtype=float)), 0.001, 0.999)
    rows["pregame_lineup_probability_delta"] = rows["p_over_raw"] - rows["p_over_raw_v94_safe"]

    output = _resolve(args.output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "data" / "prop_training_rows.csv", index=False)
    _copy_tree_if_exists(source_output / "calibration", output / "calibration")
    tuned_manifest = dict(manifest)
    tuned_manifest["output"] = str(output.relative_to(REPO_ROOT)) if output.is_relative_to(REPO_ROOT) else str(output)
    tuned_manifest["trained_at"] = datetime.now(timezone.utc).isoformat()
    tuned_manifest["status"] = "pregame_lineup_shadow_candidate_weight_tuned"
    tuned_manifest["pregame_lineup_application"] = dict(manifest.get("pregame_lineup_application", {}))
    tuned_manifest["pregame_lineup_application"]["lineup_weight"] = args.lineup_weight
    tuned_manifest["pregame_lineup_application"]["avg_abs_probability_delta"] = float(rows["pregame_lineup_probability_delta"].abs().mean())
    tuned_manifest["pregame_lineup_application"]["retuned_from"] = str(source_manifest_path)
    (output / "manifest.json").write_text(json.dumps(tuned_manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": tuned_manifest["status"], "lineup_weight": args.lineup_weight, "manifest": str(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
