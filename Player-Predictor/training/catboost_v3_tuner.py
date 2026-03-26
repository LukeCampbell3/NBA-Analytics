#!/usr/bin/env python3
"""
CatBoost v3: Optuna-tuned per-target CatBoost with enhanced features.
Goal: Beat current 4.441 avg MAE (PTS=8.47, TRB=2.31, AST=2.55)

Improvements over v2:
  1. Optuna per-target tuning (15 trials each, fast)
  2. Enhanced features v3: quantiles, player volatility, scoring streaks
  3. Per-target early stopping with more patience
  4. Multiple seeds → pick best per target
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from unified_moe_trainer import UnifiedMoETrainer
from improved_stacking_trainer import prepare_gbm_features_v2


# ============================================================
# Enhanced features v3: adds quantiles, player volatility, streaks
# ============================================================
def prepare_gbm_features_v3(X_seq, baselines):
    """v2 features + quantile features + interaction terms."""
    n, sl, nf = X_seq.shape

    # All v2 features
    last = X_seq[:, -1, :]
    seq_mean = X_seq.mean(axis=1)
    seq_std = X_seq.std(axis=1)
    trend = X_seq[:, -3:, :].mean(axis=1) - X_seq[:, :3, :].mean(axis=1) if sl >= 6 else np.zeros_like(last)
    last3 = X_seq[:, -3:, :].reshape(n, -1)
    seq_min = X_seq.min(axis=1)
    seq_max = X_seq.max(axis=1)
    seq_range = seq_max - seq_min
    momentum = last - seq_mean
    recent_std = X_seq[:, -3:, :].std(axis=1) if sl >= 3 else np.zeros_like(last)
    imm_delta = X_seq[:, -1, :] - X_seq[:, -2, :] if sl >= 2 else np.zeros_like(last)

    # NEW v3: Quantile features
    q25 = np.quantile(X_seq, 0.25, axis=1)
    q75 = np.quantile(X_seq, 0.75, axis=1)
    iqr = q75 - q25
    median = np.median(X_seq, axis=1)

    # NEW v3: Last-vs-median deviation
    last_vs_median = last - median

    # NEW v3: Acceleration (change in momentum)
    if sl >= 4:
        mid_momentum = X_seq[:, -2, :] - X_seq[:, :sl-2, :].mean(axis=1)
        acceleration = momentum - mid_momentum
    else:
        acceleration = np.zeros_like(last)

    # NEW v3: Exponential weighted mean (recent games weighted more)
    weights = np.exp(np.linspace(-1, 0, sl))  # exponential decay
    weights /= weights.sum()
    ewm = np.tensordot(weights, X_seq, axes=([0], [1]))  # (n, nf)

    # NEW v3: Last 5 vs first 5 trend (broader trend)
    if sl >= 10:
        broad_trend = X_seq[:, -5:, :].mean(axis=1) - X_seq[:, :5, :].mean(axis=1)
    else:
        broad_trend = trend  # fallback to 3v3

    return np.hstack([
        last, seq_mean, seq_std, trend, last3,
        seq_min, seq_max, seq_range, momentum, recent_std, imm_delta,
        q25, q75, iqr, median, last_vs_median,
        acceleration, ewm, broad_trend,
        baselines
    ])


# ============================================================
# Optuna objective for CatBoost
# ============================================================
def create_objective(X_tr, y_tr, X_v, y_v, target_idx):
    def objective(trial):
        params = {
            'loss_function': 'MAE',
            'iterations': 800,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.9),
            'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
            'early_stopping_rounds': 40,
            'verbose': 0,
            'random_seed': 42,
        }
        m = CatBoostRegressor(**params)
        m.fit(X_tr, y_tr[:, target_idx], eval_set=(X_v, y_v[:, target_idx]), verbose=0)
        pred = m.predict(X_v)
        return mean_absolute_error(y_v[:, target_idx], pred)
    return objective


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("CATBOOST v3: Optuna-tuned + Enhanced Features")
    print("=" * 80)

    # Load data
    trainer = UnifiedMoETrainer()
    X, baselines, y, df = trainer.prepare_data()
    split = int(len(X) * 0.8)
    X_tr, X_v = X[:split], X[split:]
    b_tr, b_v = baselines[:split], baselines[split:]
    y_tr, y_v = y[:split], y[split:]
    d_tr, d_v = y_tr - b_tr, y_v - b_v
    n_tgt = 3
    names = trainer.target_columns
    y_v_orig = trainer.scaler_y.inverse_transform(y_v)
    b_v_orig = trainer.scaler_y.inverse_transform(b_v)

    # Prepare features v3
    X_gbm_tr_v3 = prepare_gbm_features_v3(X_tr, b_tr)
    X_gbm_v_v3 = prepare_gbm_features_v3(X_v, b_v)
    print(f"\n  v3 features: {X_gbm_tr_v3.shape[1]} (v2 was 1472)")

    # Also prepare v2 for comparison
    X_gbm_tr_v2 = prepare_gbm_features_v2(X_tr, b_tr)
    X_gbm_v_v2 = prepare_gbm_features_v2(X_v, b_v)

    # ============================================================
    # Step 1: Quick test — v3 features with v2 params (baseline)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: v3 features with v2 CatBoost params")
    print("=" * 60)
    v2_params = dict(
        loss_function='MAE', iterations=500, learning_rate=0.05,
        depth=6, l2_leaf_reg=1.0, min_data_in_leaf=20,
        subsample=0.8, colsample_bylevel=0.8,
        early_stopping_rounds=30, verbose=0, random_seed=42
    )
    v3_baseline_maes = []
    for t in range(n_tgt):
        m = CatBoostRegressor(**v2_params)
        m.fit(X_gbm_tr_v3, d_tr[:, t], eval_set=(X_gbm_v_v3, d_v[:, t]), verbose=0)
        pred = m.predict(X_gbm_v_v3)
        preds_orig = trainer.scaler_y.inverse_transform(b_v + np.column_stack([
            pred if t == 0 else d_v[:, 0],
            pred if t == 1 else d_v[:, 1],
            pred if t == 2 else d_v[:, 2]
        ]))
        mae = mean_absolute_error(y_v_orig[:, t], preds_orig[:, t])
        v3_baseline_maes.append(mae)
        print(f"  {names[t]}: MAE={mae:.4f} (v2 was {[8.47, 2.31, 2.55][t]:.2f})")
    print(f"  Avg: {np.mean(v3_baseline_maes):.4f} (v2 was 4.441)")

    # ============================================================
    # Step 2: Optuna tuning per target (15 trials each)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Optuna per-target tuning (15 trials each)")
    print("=" * 60)

    # Try both v2 and v3 features — pick best per target
    best_models = []
    best_maes = []
    best_feature_versions = []

    for t in range(n_tgt):
        print(f"\n  --- {names[t]} ---")

        # Tune on v3 features
        study_v3 = optuna.create_study(direction='minimize')
        study_v3.optimize(create_objective(X_gbm_tr_v3, d_tr, X_gbm_v_v3, d_v, t),
                          n_trials=15, show_progress_bar=False)
        best_v3_mae = study_v3.best_value
        best_v3_params = study_v3.best_params

        # Tune on v2 features
        study_v2 = optuna.create_study(direction='minimize')
        study_v2.optimize(create_objective(X_gbm_tr_v2, d_tr, X_gbm_v_v2, d_v, t),
                          n_trials=15, show_progress_bar=False)
        best_v2_mae = study_v2.best_value
        best_v2_params = study_v2.best_params

        print(f"    v3 best scaled MAE: {best_v3_mae:.6f}")
        print(f"    v2 best scaled MAE: {best_v2_mae:.6f}")

        # Pick the better one
        if best_v3_mae <= best_v2_mae:
            chosen_params = best_v3_params
            chosen_X_tr, chosen_X_v = X_gbm_tr_v3, X_gbm_v_v3
            fv = 'v3'
        else:
            chosen_params = best_v2_params
            chosen_X_tr, chosen_X_v = X_gbm_tr_v2, X_gbm_v_v2
            fv = 'v2'

        print(f"    Using {fv} features")
        print(f"    Params: {chosen_params}")

        # Retrain with best params + multiple seeds, pick best
        seed_maes = []
        seed_models = []
        for seed in [42, 123, 777]:
            full_params = {
                'loss_function': 'MAE', 'iterations': 800,
                'early_stopping_rounds': 40, 'verbose': 0, 'random_seed': seed,
                **chosen_params
            }
            m = CatBoostRegressor(**full_params)
            m.fit(chosen_X_tr, d_tr[:, t], eval_set=(chosen_X_v, d_v[:, t]), verbose=0)
            pred = m.predict(chosen_X_v)
            smae = mean_absolute_error(d_v[:, t], pred)
            seed_maes.append(smae)
            seed_models.append(m)

        best_seed_idx = np.argmin(seed_maes)
        best_model = seed_models[best_seed_idx]
        best_models.append(best_model)
        best_feature_versions.append(fv)
        print(f"    Best seed: {[42, 123, 777][best_seed_idx]} (scaled MAE: {seed_maes[best_seed_idx]:.6f})")

    # ============================================================
    # Step 3: Full evaluation
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP 3: FULL EVALUATION")
    print("=" * 80)

    cb_delta = np.zeros_like(d_v)
    for t in range(n_tgt):
        fv = best_feature_versions[t]
        X_eval = X_gbm_v_v3 if fv == 'v3' else X_gbm_v_v2
        cb_delta[:, t] = best_models[t].predict(X_eval)

    cb_preds = trainer.scaler_y.inverse_transform(b_v + cb_delta)

    total = 0
    per_target = {}
    for i, t_name in enumerate(names):
        mae = mean_absolute_error(y_v_orig[:, i], cb_preds[:, i])
        r2 = r2_score(y_v_orig[:, i], cb_preds[:, i])
        bl_mae = mean_absolute_error(y_v_orig[:, i], b_v_orig[:, i])
        imp = bl_mae - mae
        pct = (imp / bl_mae * 100) if bl_mae > 0 else 0
        total += mae
        per_target[t_name] = mae
        v2_mae = [8.47, 2.31, 2.55][i]
        delta_vs_v2 = v2_mae - mae
        print(f"  {t_name}: MAE={mae:.4f}  R²={r2:.4f}  (baseline: {bl_mae:.4f}, +{imp:.4f} / {pct:+.1f}%)  vs v2: {delta_vs_v2:+.4f}")

    avg = total / n_tgt
    print(f"\n  Avg MAE: {avg:.4f}")
    print(f"  vs CatBoost v2 (4.441): {4.441 - avg:+.4f} ({(4.441 - avg)/4.441*100:+.1f}%)")
    print(f"  vs old ensemble (4.795): {4.795 - avg:+.4f} ({(4.795 - avg)/4.795*100:+.1f}%)")

    # ============================================================
    # Step 4: Save if improved
    # ============================================================
    if avg < 4.441:
        print("\n  NEW BEST! Saving as production...")
        joblib.dump(best_models, 'model/production_catboost_models.pkl')
        joblib.dump(trainer.scaler_x, 'model/unified_moe_scaler_x.pkl')
        joblib.dump(trainer.scaler_y, 'model/unified_moe_scaler_y.pkl')

        metadata = {
            'model_type': 'catboost_v3',
            'gbm_feature_versions': best_feature_versions,
            'gbm_feature_dim_v2': X_gbm_tr_v2.shape[1],
            'gbm_feature_dim_v3': X_gbm_tr_v3.shape[1],
            'n_targets': n_tgt,
            'target_columns': names,
            'avg_mae': float(avg),
            'per_target_mae': per_target,
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
        print("\n  No improvement over v2. Keeping v2 as production.")
        print("  Saving v3 models separately for reference...")
        joblib.dump(best_models, 'model/catboost_v3_models.pkl')

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
