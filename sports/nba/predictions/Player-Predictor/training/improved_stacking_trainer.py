#!/usr/bin/env python3
"""
Improved Stacking Trainer v2 (Fast)
====================================
Improvements over v1 (single LSTM + single LightGBM + Ridge):
  1. Multi-seed LSTM ensemble (3 seeds)
  2. Triple GBM: LightGBM + XGBoost + CatBoost (hand-tuned, no Optuna)
  3. Richer GBM features (quantiles, momentum, min/max, volatility)
  4. Enhanced meta-learner with all 6 base model predictions

Run: python training/improved_stacking_trainer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# ============================================================
def build_simple_lstm(seq_len, n_features, n_targets, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, Concatenate, LayerNormalization
    )
    from tensorflow.keras.models import Model

    seq_in = Input(shape=(seq_len, n_features), name="sequence_input")
    base_in = Input(shape=(n_targets,), name="baseline_pred_input")
    x = LSTM(128, return_sequences=True)(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    c = Concatenate()([x, base_in])
    x = Dense(128, activation="relu")(c)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    out = Dense(n_targets)(x)
    return Model(inputs=[seq_in, base_in], outputs=out)


def train_lstm(X_tr, b_tr, d_tr, X_v, b_v, d_v, seq_len, n_feat, n_tgt, seed=42):
    model = build_simple_lstm(seq_len, n_feat, n_tgt, seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
                  loss='mse', metrics=['mae'])
    model.fit(
        [X_tr, b_tr], d_tr,
        validation_data=([X_v, b_v], d_v),
        epochs=120, batch_size=128, verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping('val_loss', patience=25,
                                             restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5,
                                                  patience=10, min_lr=1e-6, verbose=0),
        ]
    )
    return model, model.predict([X_v, b_v], verbose=0, batch_size=256)


# ============================================================
# Enhanced GBM Features
# ============================================================
def prepare_gbm_features_v2(X_seq, baselines):
    n, sl, nf = X_seq.shape
    last = X_seq[:, -1, :]
    seq_mean = X_seq.mean(axis=1)
    seq_std  = X_seq.std(axis=1)
    trend = X_seq[:, -3:, :].mean(axis=1) - X_seq[:, :3, :].mean(axis=1) if sl >= 6 else np.zeros_like(last)
    last3 = X_seq[:, -3:, :].reshape(n, -1)
    seq_min = X_seq.min(axis=1)
    seq_max = X_seq.max(axis=1)
    seq_range = seq_max - seq_min
    momentum = last - seq_mean
    recent_std = X_seq[:, -3:, :].std(axis=1) if sl >= 3 else np.zeros_like(last)
    imm_delta = X_seq[:, -1, :] - X_seq[:, -2, :] if sl >= 2 else np.zeros_like(last)
    return np.hstack([last, seq_mean, seq_std, trend, last3,
                      seq_min, seq_max, seq_range, momentum, recent_std, imm_delta,
                      baselines])


# ============================================================
# Train all 3 GBM types (no Optuna — fast hand-tuned params)
# ============================================================
def train_all_gbms(X_tr, d_tr, X_v, d_v, n_tgt, names):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    results = {}

    # --- LightGBM ---
    print("\n  === LightGBM ===")
    lgb_models, lgb_preds = [], np.zeros_like(d_v)
    for t in range(n_tgt):
        params = {'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
                  'n_jobs': -1, 'learning_rate': 0.05, 'num_leaves': 31,
                  'max_depth': 6, 'min_child_samples': 20, 'subsample': 0.8,
                  'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
        td = lgb.Dataset(X_tr, label=d_tr[:, t])
        vd = lgb.Dataset(X_v, label=d_v[:, t], reference=td)
        m = lgb.train(params, td, num_boost_round=500, valid_sets=[vd],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        lgb_preds[:, t] = m.predict(X_v)
        mae = mean_absolute_error(d_v[:, t], lgb_preds[:, t])
        print(f"    {names[t]}: MAE={mae:.4f}, iters={m.best_iteration}")
        lgb_models.append(m)
    results['lgb'] = (lgb_models, lgb_preds)

    # --- XGBoost ---
    print("\n  === XGBoost ===")
    xgb_models, xgb_preds = [], np.zeros_like(d_v)
    for t in range(n_tgt):
        params = {'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                  'verbosity': 0, 'nthread': -1, 'learning_rate': 0.05,
                  'max_depth': 6, 'min_child_weight': 20, 'subsample': 0.8,
                  'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
                  'gamma': 0.1}
        dtrain = xgb.DMatrix(X_tr, label=d_tr[:, t])
        dval = xgb.DMatrix(X_v, label=d_v[:, t])
        m = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)
        xgb_preds[:, t] = m.predict(dval)
        mae = mean_absolute_error(d_v[:, t], xgb_preds[:, t])
        print(f"    {names[t]}: MAE={mae:.4f}, iters={m.best_iteration}")
        xgb_models.append(m)
    results['xgb'] = (xgb_models, xgb_preds)

    # --- CatBoost ---
    print("\n  === CatBoost ===")
    cb_models, cb_preds = [], np.zeros_like(d_v)
    for t in range(n_tgt):
        m = CatBoostRegressor(
            loss_function='MAE', iterations=500, learning_rate=0.05,
            depth=6, l2_leaf_reg=1.0, min_data_in_leaf=20,
            subsample=0.8, colsample_bylevel=0.8,
            early_stopping_rounds=30, verbose=0)
        m.fit(X_tr, d_tr[:, t], eval_set=(X_v, d_v[:, t]), verbose=0)
        cb_preds[:, t] = m.predict(X_v)
        mae = mean_absolute_error(d_v[:, t], cb_preds[:, t])
        print(f"    {names[t]}: MAE={mae:.4f}, iters={m.best_iteration_}")
        cb_models.append(m)
    results['cb'] = (cb_models, cb_preds)

    return results


# ============================================================
# Enhanced Meta-Learner
# ============================================================
def train_enhanced_meta(lstm_deltas, gbm_results, b_val, true_delta, n_tgt, names):
    meta_models = []
    meta_delta = np.zeros_like(true_delta)
    for t in range(n_tgt):
        feats, fnames = [], []
        for i, ld in enumerate(lstm_deltas):
            feats.append(ld[:, t:t+1]); fnames.append(f"lstm{i}")
        for gn in ['lgb', 'xgb', 'cb']:
            if gn in gbm_results:
                feats.append(gbm_results[gn][1][:, t:t+1]); fnames.append(gn)
        feats.append(b_val); fnames.extend([f"b{i}" for i in range(b_val.shape[1])])
        X_m = np.hstack(feats)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_m, true_delta[:, t])
        meta_models.append(ridge)
        pred = ridge.predict(X_m)
        mae = np.mean(np.abs(pred - true_delta[:, t]))
        meta_delta[:, t] = pred
        n_model = len(fnames) - b_val.shape[1]
        cs = ", ".join(f"{fnames[i]}={ridge.coef_[i]:.3f}" for i in range(n_model))
        print(f"  {names[t]}: MAE={mae:.4f} | {cs}")
    return meta_models, meta_delta


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "="*80)
    print("IMPROVED STACKING v2: 3-seed LSTM + LGB/XGB/CatBoost + Ridge Meta")
    print("="*80)

    from unified_moe_trainer import UnifiedMoETrainer
    trainer = UnifiedMoETrainer()
    X, baselines, y, df = trainer.prepare_data()

    split = int(len(X) * 0.8)
    X_tr, X_v = X[:split], X[split:]
    b_tr, b_v = baselines[:split], baselines[split:]
    y_tr, y_v = y[:split], y[split:]
    d_tr, d_v = y_tr - b_tr, y_v - b_v
    n_tgt = y_tr.shape[1]
    sl, nf = X_tr.shape[1], X_tr.shape[2]
    names = trainer.target_columns

    print(f"\n  Train: {len(X_tr)}, Val: {len(X_v)}, Features: {nf}, Targets: {n_tgt}")

    # --- Phase 1: Multi-seed LSTM ---
    print("\n" + "="*60)
    print("PHASE 1: Multi-seed LSTM (3 seeds)")
    print("="*60)
    seeds = [42, 123, 777]
    lstm_models, lstm_deltas = [], []
    y_v_orig = trainer.scaler_y.inverse_transform(y_v)
    b_v_orig = trainer.scaler_y.inverse_transform(b_v)

    for i, s in enumerate(seeds):
        print(f"  Training seed={s} ({i+1}/3)...", end=" ", flush=True)
        m, vp = train_lstm(X_tr, b_tr, d_tr, X_v, b_v, d_v, sl, nf, n_tgt, s)
        lstm_models.append(m)
        lstm_deltas.append(vp)
        po = trainer.scaler_y.inverse_transform(b_v + vp)
        maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
        print(f"PTS={maes[0]:.2f} TRB={maes[1]:.2f} AST={maes[2]:.2f} Avg={np.mean(maes):.4f}")

    ens_delta = np.mean(lstm_deltas, axis=0)
    po = trainer.scaler_y.inverse_transform(b_v + ens_delta)
    maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
    print(f"  Ensemble avg: PTS={maes[0]:.2f} TRB={maes[1]:.2f} AST={maes[2]:.2f} Avg={np.mean(maes):.4f}")

    # Save best LSTM
    best_i = min(range(len(seeds)), key=lambda i: np.mean([
        mean_absolute_error(y_v_orig[:, t],
                            trainer.scaler_y.inverse_transform(b_v + lstm_deltas[i])[:, t])
        for t in range(n_tgt)]))
    lstm_models[best_i].save_weights('model/unified_moe_best.weights.h5')
    lstm_models[best_i].save_weights('model/unified_moe_final.weights.h5')

    # --- Phase 2: Triple GBM ---
    print("\n" + "="*60)
    print("PHASE 2: Triple GBM (LGB + XGB + CatBoost)")
    print("="*60)
    X_gbm_tr = prepare_gbm_features_v2(X_tr, b_tr)
    X_gbm_v  = prepare_gbm_features_v2(X_v, b_v)
    print(f"  GBM features: {X_gbm_tr.shape[1]} (v1 was ~794)")
    gbm_results = train_all_gbms(X_gbm_tr, d_tr, X_gbm_v, d_v, n_tgt, names)

    # --- Phase 3: Meta-learner ---
    print("\n" + "="*60)
    print("PHASE 3: Enhanced Ridge Meta-Learner (6 base models)")
    print("="*60)
    meta_models, meta_delta = train_enhanced_meta(
        lstm_deltas, gbm_results, b_v, d_v, n_tgt, names)

    # --- Phase 4: Evaluation ---
    print("\n" + "="*80)
    print("FULL EVALUATION")
    print("="*80)

    approaches = {}
    # Individual GBMs
    for gn, (_, gp) in gbm_results.items():
        approaches[gn] = trainer.scaler_y.inverse_transform(b_v + gp)
    # GBM average
    gbm_avg_d = np.mean([gp for _, gp in gbm_results.values()], axis=0)
    approaches['gbm_avg'] = trainer.scaler_y.inverse_transform(b_v + gbm_avg_d)
    # LSTM ensemble
    approaches['lstm_ens'] = trainer.scaler_y.inverse_transform(b_v + ens_delta)
    # Stacked v2
    approaches['stacked_v2'] = trainer.scaler_y.inverse_transform(b_v + meta_delta)
    # v1-style (best single LSTM + LGB only)
    v1d = np.zeros_like(d_v)
    for t in range(n_tgt):
        f = np.hstack([lstm_deltas[best_i][:, t:t+1], gbm_results['lgb'][1][:, t:t+1], b_v])
        r = Ridge(alpha=1.0); r.fit(f, d_v[:, t]); v1d[:, t] = r.predict(f)
    approaches['stacked_v1'] = trainer.scaler_y.inverse_transform(b_v + v1d)

    summary = {}
    for name, preds in sorted(approaches.items()):
        maes = [mean_absolute_error(y_v_orig[:, i], preds[:, i]) for i in range(n_tgt)]
        avg = np.mean(maes)
        summary[name] = avg
        print(f"  {name:20s}: PTS={maes[0]:.2f}  TRB={maes[1]:.2f}  AST={maes[2]:.2f}  Avg={avg:.4f}")

    print(f"\n{'='*60}")
    print("RANKED")
    print(f"{'='*60}")
    for name, avg in sorted(summary.items(), key=lambda x: x[1]):
        m = " <-- BEST" if avg == min(summary.values()) else ""
        print(f"  {name:20s}: {avg:.4f}{m}")
    print(f"  {'old_ensemble':20s}: 4.7950")
    print(f"  {'prev_stacked_v1':20s}: 4.5984")
    print(f"{'='*60}")

    best_name = min(summary, key=summary.get)
    best_avg = summary[best_name]
    if best_avg < 4.5984:
        print(f"\n  NEW BEST! {best_name}: {best_avg:.4f} (improvement: {4.5984 - best_avg:.4f})")
    else:
        print(f"\n  Best: {best_name}: {best_avg:.4f} (prev best: 4.5984)")

    # --- Save ---
    print("\n Saving...")
    joblib.dump({n: ms for n, (ms, _) in gbm_results.items()}, 'model/stacked_gbm_models_v2.pkl')
    joblib.dump(meta_models, 'model/stacked_meta_models_v2.pkl')
    for i, m in enumerate(lstm_models):
        m.save_weights(f'model/lstm_seed_{seeds[i]}.weights.h5')
    if best_name == 'stacked_v2':
        joblib.dump({n: ms for n, (ms, _) in gbm_results.items()}, 'model/stacked_gbm_models.pkl')
        joblib.dump(meta_models, 'model/stacked_meta_models.pkl')
        print("  v2 saved as production models")
    joblib.dump(trainer.scaler_x, 'model/unified_moe_scaler_x.pkl')
    joblib.dump(trainer.scaler_y, 'model/unified_moe_scaler_y.pkl')
    metadata = {
        'model_type': 'improved_stacking_v2', 'lstm_seeds': seeds,
        'best_lstm_seed': seeds[best_i], 'gbm_types': list(gbm_results.keys()),
        'gbm_feature_dim': X_gbm_tr.shape[1], 'n_targets': n_tgt,
        'target_columns': names, 'best_method': best_name,
        'best_avg_mae': float(best_avg),
        'feature_columns': trainer.feature_columns,
        'baseline_features': trainer.baseline_features,
        'player_mapping': trainer.player_mapping,
        'team_mapping': trainer.team_mapping,
        'opponent_mapping': trainer.opponent_mapping,
    }
    with open('model/unified_moe_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  Done!")
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
