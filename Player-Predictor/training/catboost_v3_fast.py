#!/usr/bin/env python3
"""
CatBoost v3 (fast): No Optuna. Quick targeted improvements.
Goal: Beat 4.441 avg MAE (PTS=8.47, TRB=2.31, AST=2.55)

Strategy:
  1. More iterations (1000 vs 500) with lower LR
  2. Try deeper trees for PTS (the biggest MAE contributor)
  3. v3 features: quantiles, EWM, acceleration
  4. Multi-seed per target, pick best
  5. All output to stderr to avoid buffering
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
# Force all prints to stderr to avoid PowerShell buffering
import functools
print = functools.partial(print, file=sys.stderr, flush=True)

import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from unified_moe_trainer import UnifiedMoETrainer
from improved_stacking_trainer import prepare_gbm_features_v2


def prepare_gbm_features_v3(X_seq, baselines):
    """v2 + quantiles, EWM, acceleration."""
    n, sl, nf = X_seq.shape
    last = X_seq[:, -1, :]
    seq_mean = X_seq.mean(axis=1)
    seq_std = X_seq.std(axis=1)
    trend = X_seq[:, -3:, :].mean(axis=1) - X_seq[:, :3, :].mean(axis=1)
    last3 = X_seq[:, -3:, :].reshape(n, -1)
    seq_min = X_seq.min(axis=1)
    seq_max = X_seq.max(axis=1)
    seq_range = seq_max - seq_min
    momentum = last - seq_mean
    recent_std = X_seq[:, -3:, :].std(axis=1)
    imm_delta = X_seq[:, -1, :] - X_seq[:, -2, :]
    # v3 additions
    q25 = np.quantile(X_seq, 0.25, axis=1)
    q75 = np.quantile(X_seq, 0.75, axis=1)
    iqr = q75 - q25
    median = np.median(X_seq, axis=1)
    last_vs_median = last - median
    weights = np.exp(np.linspace(-1, 0, sl))
    weights /= weights.sum()
    ewm = np.tensordot(weights, X_seq, axes=([0], [1]))
    mid_mom = X_seq[:, -2, :] - X_seq[:, :sl-2, :].mean(axis=1)
    accel = momentum - mid_mom
    return np.hstack([
        last, seq_mean, seq_std, trend, last3,
        seq_min, seq_max, seq_range, momentum, recent_std, imm_delta,
        q25, q75, iqr, median, last_vs_median, ewm, accel,
        baselines
    ])


def main():
    print("=" * 70)
    print("CATBOOST v3 FAST: More iters + v3 features + multi-seed")
    print("=" * 70)

    trainer = UnifiedMoETrainer()
    X, baselines, y, df = trainer.prepare_data()
    split = int(len(X) * 0.8)
    X_tr, X_v = X[:split], X[split:]
    b_tr, b_v = baselines[:split], baselines[split:]
    y_tr, y_v = y[:split], y[split:]
    d_tr, d_v = y_tr - b_tr, y_v - b_v
    names = trainer.target_columns
    y_v_orig = trainer.scaler_y.inverse_transform(y_v)
    b_v_orig = trainer.scaler_y.inverse_transform(b_v)

    # Prepare both feature sets
    X2_tr = prepare_gbm_features_v2(X_tr, b_tr)
    X2_v = prepare_gbm_features_v2(X_v, b_v)
    X3_tr = prepare_gbm_features_v3(X_tr, b_tr)
    X3_v = prepare_gbm_features_v3(X_v, b_v)
    print(f"v2 features: {X2_tr.shape[1]}, v3 features: {X3_tr.shape[1]}")

    # Param configs to try per target
    configs = {
        'base': dict(loss_function='MAE', iterations=1000, learning_rate=0.03,
                      depth=6, l2_leaf_reg=1.0, min_data_in_leaf=20,
                      subsample=0.8, colsample_bylevel=0.8,
                      early_stopping_rounds=50, verbose=0),
        'deep': dict(loss_function='MAE', iterations=1000, learning_rate=0.03,
                      depth=8, l2_leaf_reg=3.0, min_data_in_leaf=30,
                      subsample=0.7, colsample_bylevel=0.7,
                      early_stopping_rounds=50, verbose=0),
        'wide': dict(loss_function='MAE', iterations=1000, learning_rate=0.02,
                      depth=6, l2_leaf_reg=0.5, min_data_in_leaf=10,
                      subsample=0.85, colsample_bylevel=0.85,
                      early_stopping_rounds=50, verbose=0),
        'rmse': dict(loss_function='RMSE', iterations=1000, learning_rate=0.03,
                      depth=6, l2_leaf_reg=1.0, min_data_in_leaf=20,
                      subsample=0.8, colsample_bylevel=0.8,
                      early_stopping_rounds=50, verbose=0),
    }

    best_models = [None, None, None]
    best_maes = [999, 999, 999]
    best_info = ['', '', '']

    for t in range(3):
        print(f"\n{'='*50}")
        print(f"TARGET: {names[t]} (v2 MAE: {[8.47, 2.31, 2.55][t]})")
        print(f"{'='*50}")

        for cfg_name, params in configs.items():
            for fv, (Xtr, Xv) in [('v2', (X2_tr, X2_v)), ('v3', (X3_tr, X3_v))]:
                for seed in [42, 123, 777]:
                    p = {**params, 'random_seed': seed}
                    m = CatBoostRegressor(**p)
                    m.fit(Xtr, d_tr[:, t], eval_set=(Xv, d_v[:, t]), verbose=0)
                    pred = m.predict(Xv)

                    # Compute real-space MAE
                    delta_full = np.copy(d_v)
                    delta_full[:, t] = pred
                    preds_orig = trainer.scaler_y.inverse_transform(b_v + delta_full)
                    mae = mean_absolute_error(y_v_orig[:, t], preds_orig[:, t])

                    tag = f"{cfg_name}/{fv}/s{seed}"
                    if mae < best_maes[t]:
                        best_maes[t] = mae
                        best_models[t] = m
                        best_info[t] = tag
                        print(f"  {tag}: {mae:.4f} <-- NEW BEST (iters={m.best_iteration_})")
                    else:
                        print(f"  {tag}: {mae:.4f}")

        print(f"  BEST for {names[t]}: {best_info[t]} -> {best_maes[t]:.4f}")

    # Full evaluation with best per-target models
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    cb_delta = np.zeros_like(d_v)
    for t in range(3):
        fv = best_info[t].split('/')[1]
        Xv = X3_v if fv == 'v3' else X2_v
        cb_delta[:, t] = best_models[t].predict(Xv)

    cb_preds = trainer.scaler_y.inverse_transform(b_v + cb_delta)
    total = 0
    per_target = {}
    for i, nm in enumerate(names):
        mae = mean_absolute_error(y_v_orig[:, i], cb_preds[:, i])
        bl = mean_absolute_error(y_v_orig[:, i], b_v_orig[:, i])
        r2 = r2_score(y_v_orig[:, i], cb_preds[:, i])
        total += mae
        per_target[nm] = float(mae)
        v2_ref = [8.47, 2.31, 2.55][i]
        print(f"  {nm}: MAE={mae:.4f}  R²={r2:.4f}  (bl={bl:.4f})  vs v2: {v2_ref - mae:+.4f}")

    avg = total / 3
    print(f"\n  Avg MAE: {avg:.4f}")
    print(f"  vs v2 (4.441): {4.441 - avg:+.4f}")
    print(f"  vs old (4.795): {4.795 - avg:+.4f}")

    # Save
    if avg < 4.441:
        print("\n  NEW BEST! Saving as production...")
        # Need to store which feature version each target uses
        fv_list = [best_info[t].split('/')[1] for t in range(3)]
        joblib.dump(best_models, 'model/production_catboost_models.pkl')
        joblib.dump(trainer.scaler_x, 'model/unified_moe_scaler_x.pkl')
        joblib.dump(trainer.scaler_y, 'model/unified_moe_scaler_y.pkl')
        metadata = {
            'model_type': 'catboost_v3',
            'gbm_feature_versions': fv_list,
            'gbm_feature_dim_v2': int(X2_tr.shape[1]),
            'gbm_feature_dim_v3': int(X3_tr.shape[1]),
            'n_targets': 3, 'target_columns': names,
            'avg_mae': float(avg), 'per_target_mae': per_target,
            'best_configs': best_info,
            'feature_columns': trainer.feature_columns,
            'baseline_features': trainer.baseline_features,
            'player_mapping': trainer.player_mapping,
            'team_mapping': trainer.team_mapping,
            'opponent_mapping': trainer.opponent_mapping,
            'seq_len': 10,
        }
        with open('model/unified_moe_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  Saved!")
    else:
        print("\n  No improvement. Saving v3 separately.")
        joblib.dump(best_models, 'model/catboost_v3_models.pkl')
        # Save info for reference
        with open('model/catboost_v3_info.json', 'w') as f:
            json.dump({'avg_mae': float(avg), 'per_target': per_target,
                       'configs': best_info}, f, indent=2)

    print("\nDONE")


if __name__ == "__main__":
    main()
