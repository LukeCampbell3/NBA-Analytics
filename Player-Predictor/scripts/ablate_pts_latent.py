#!/usr/bin/env python3
"""
Run a PTS-only latent ablation table for a saved structured-stack run.

This script:
- rebuilds the train/validation split used by the trainer
- loads the saved structured LSTM ensemble for a chosen run
- exports latent blocks
- fits a fixed CatBoost PTS regressor on:
  - base_v2
  - base_v3
  - best base + selected latent block groups

The goal is to measure which latent blocks are actually moving PTS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT / "inference"))

from improved_lstm_v7 import (  # noqa: E402
    CatBoostRegressor,
    build_structured_latent_feature_matrix,
    create_shared_trainer,
    prepare_gbm_features_v3,
    weighted_ensemble_latent_export,
)
from improved_stacking_trainer import prepare_gbm_features_v2  # noqa: E402
from structured_stack_inference import StructuredStackInference  # noqa: E402


MODEL_DIR = REPO_ROOT / "model"


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path:
    if run_id:
        path = MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing run metadata: {path}")
        return path
    if latest:
        path = MODEL_DIR / "latest_structured_lstm_stack.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing latest manifest: {path}")
        return path
    path = MODEL_DIR / "production_structured_lstm_stack.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing production manifest: {path}")
    return path


def get_block_matrix(latents: dict[str, np.ndarray], block_names: list[str]) -> np.ndarray:
    return np.hstack([latents[name] for name in block_names])


def fixed_pts_catboost_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        iterations=1100,
        learning_rate=0.023,
        depth=8,
        l2_leaf_reg=3.0,
        min_data_in_leaf=24,
        subsample=0.75,
        colsample_bylevel=0.72,
        early_stopping_rounds=65,
        verbose=0,
        random_seed=int(seed),
    )


def evaluate_pts_model(
    X_train,
    y_train_delta,
    X_val,
    y_val_scaled,
    b_val_scaled,
    scaler_y,
    seeds: list[int],
):
    maes: list[float] = []
    r2s: list[float] = []
    true_orig = scaler_y.inverse_transform(y_val_scaled)[:, 0]

    for seed in seeds:
        model = CatBoostRegressor(**fixed_pts_catboost_params(seed))
        model.fit(X_train, y_train_delta, eval_set=(X_val, y_val_scaled[:, 0] - b_val_scaled[:, 0]), verbose=0)
        pred_delta_scaled = np.asarray(model.predict(X_val), dtype=np.float32)
        pred_y_scaled = pred_delta_scaled + b_val_scaled[:, 0]
        pred_scaled_full = np.zeros_like(y_val_scaled, dtype=np.float32)
        pred_scaled_full[:, 0] = pred_y_scaled
        pred_orig = scaler_y.inverse_transform(pred_scaled_full)[:, 0]
        maes.append(float(mean_absolute_error(true_orig, pred_orig)))
        r2s.append(float(r2_score(true_orig, pred_orig)))

    return {
        "mae": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "seeds": [int(seed) for seed in seeds],
    }


def make_row(name: str, block_names: list[str], n_features: int, metrics: dict, best_base_mae: float) -> dict:
    return {
        "name": name,
        "block_names": list(block_names),
        "n_features": int(n_features),
        "mae": float(metrics["mae"]),
        "mae_std": float(metrics["mae_std"]),
        "r2": float(metrics["r2"]),
        "r2_std": float(metrics["r2_std"]),
        "delta_vs_best_base": float(metrics["mae"] - best_base_mae),
        "seeds": list(metrics["seeds"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[842, 943, 1044])
    args = parser.parse_args()

    if CatBoostRegressor is None:
        raise ImportError("catboost is required for PTS latent ablation")

    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    metadata = predictor.metadata

    trainer = create_shared_trainer()
    X, baselines, y, _df = trainer.prepare_data()
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    b_train, b_val = baselines[:split_idx], baselines[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    delta_train = y_train - b_train
    pts_train = delta_train[:, 0]

    train_latents = weighted_ensemble_latent_export(predictor.models, predictor.val_losses, X_train, b_train)
    val_latents = weighted_ensemble_latent_export(predictor.models, predictor.val_losses, X_val, b_val)

    base_sets = {
        "v2": (prepare_gbm_features_v2(X_train, b_train), prepare_gbm_features_v2(X_val, b_val)),
        "v3": (prepare_gbm_features_v3(X_train, b_train), prepare_gbm_features_v3(X_val, b_val)),
    }

    base_results = []
    for base_name, (Xtr, Xva) in base_sets.items():
        metrics = evaluate_pts_model(Xtr, pts_train, Xva, y_val, b_val, trainer.scaler_y, args.seeds)
        base_results.append((base_name, metrics))
    base_results.sort(key=lambda item: item[1]["mae"])
    best_base_name, best_base_metrics = base_results[0]
    best_base_mae = best_base_metrics["mae"]
    best_base_r2 = best_base_metrics["r2"]
    X_base_train, X_base_val = base_sets[best_base_name]

    state_blocks = {
        "belief_mu": ["belief_mu"],
        "belief_std": ["belief_std"],
        "belief_pair": ["belief_mu", "belief_std"],
        "bottleneck": ["bottleneck"],
        "stable_env": ["stable_env"],
        "slow_mode": ["slow_mode"],
        "feasibility": ["feasibility"],
    }
    risk_blocks = {
        "sigma": ["sigma"],
        "spike_prob": ["spike_prob"],
    }
    regime_blocks = {
        "role_shift_prob": ["role_shift_prob"],
        "volatility_regime_prob": ["volatility_regime_prob"],
        "context_pressure_prob": ["context_pressure_prob"],
        "regime_heads": ["role_shift_prob", "volatility_regime_prob", "context_pressure_prob"],
    }
    block_groups = {}
    block_groups.update(state_blocks)
    block_groups.update(risk_blocks)
    block_groups.update(regime_blocks)
    block_groups.update(
        {
            "state_only": ["bottleneck", "belief_mu", "belief_std", "slow_mode", "stable_env", "feasibility"],
            "risk_only": ["sigma", "spike_prob"],
            "regime_only": ["role_shift_prob", "volatility_regime_prob", "context_pressure_prob"],
            "state_plus_risk": [
                "bottleneck",
                "belief_mu",
                "belief_std",
                "slow_mode",
                "stable_env",
                "feasibility",
                "sigma",
                "spike_prob",
            ],
            "state_plus_regime": [
                "bottleneck",
                "belief_mu",
                "belief_std",
                "slow_mode",
                "stable_env",
                "feasibility",
                "role_shift_prob",
                "volatility_regime_prob",
                "context_pressure_prob",
            ],
            "belief_pair_plus_bottleneck": ["belief_mu", "belief_std", "bottleneck"],
            "belief_pair_plus_bottleneck_plus_feasibility": ["belief_mu", "belief_std", "bottleneck", "feasibility"],
            "belief_pair_plus_bottleneck_plus_feasibility_plus_stable_env": [
                "belief_mu",
                "belief_std",
                "bottleneck",
                "feasibility",
                "stable_env",
            ],
            "all_latent": [
                "bottleneck",
                "belief_mu",
                "belief_std",
                "slow_mode",
                "stable_env",
                "sigma",
                "spike_prob",
                "feasibility",
                "role_shift_prob",
                "volatility_regime_prob",
                "context_pressure_prob",
            ],
        }
    )

    baseline_orig = trainer.scaler_y.inverse_transform(b_val)[:, 0]
    true_orig = trainer.scaler_y.inverse_transform(y_val)[:, 0]
    baseline_mae = float(mean_absolute_error(true_orig, baseline_orig))
    baseline_r2 = float(r2_score(true_orig, baseline_orig))

    rows = [
        {
            "name": "baseline_anchor",
            "block_names": [],
            "n_features": 0,
            "mae": baseline_mae,
            "mae_std": 0.0,
            "r2": baseline_r2,
            "r2_std": 0.0,
            "delta_vs_best_base": float(baseline_mae - best_base_mae),
            "seeds": [],
        },
    ]
    for base_name, metrics in base_results:
        Xtr, _Xva = base_sets[base_name]
        rows.append(make_row(f"base_{base_name}", [], Xtr.shape[1], metrics, best_base_mae))

    for name, block_names in block_groups.items():
        Xtr = np.hstack([X_base_train, get_block_matrix(train_latents, block_names)])
        Xva = np.hstack([X_base_val, get_block_matrix(val_latents, block_names)])
        metrics = evaluate_pts_model(Xtr, pts_train, Xva, y_val, b_val, trainer.scaler_y, args.seeds)
        rows.append(make_row(f"{best_base_name}+{name}", block_names, Xtr.shape[1], metrics, best_base_mae))

    rows = sorted(rows, key=lambda item: item["mae"])

    print("=" * 80)
    print("PTS LATENT ABLATION")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Best base family: {best_base_name} | MAE={best_base_mae:.4f} | R2={best_base_r2:.4f}")
    for row in rows:
        print(
            f"  {row['name']:44s} "
            f"MAE={row['mae']:.4f}±{row['mae_std']:.4f} "
            f"R2={row['r2']:.4f}±{row['r2_std']:.4f} "
            f"delta={row['delta_vs_best_base']:+.4f} "
            f"n={row['n_features']}"
        )

    output = {
        "manifest_path": str(manifest_path),
        "run_id": metadata.get("run_id"),
        "seeds": [int(seed) for seed in args.seeds],
        "best_base_name": best_base_name,
        "best_base_mae": float(best_base_mae),
        "best_base_r2": float(best_base_r2),
        "baseline_anchor_mae": baseline_mae,
        "baseline_anchor_r2": baseline_r2,
        "rows": rows,
    }
    out_dir = MODEL_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pts_latent_ablation_{metadata.get('run_id', 'unknown')}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
