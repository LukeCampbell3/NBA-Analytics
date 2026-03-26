#!/usr/bin/env python3
"""
Meta-blend search: try many combinations of base models in the Ridge meta-learner
to find the optimal subset. Uses saved predictions — no retraining.
Also tries different Ridge alphas and simple weighted averages.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
from itertools import combinations
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from unified_moe_trainer import UnifiedMoETrainer

print("="*70)
print("META-BLEND SEARCH")
print("="*70)

# Load data
trainer = UnifiedMoETrainer()
X, baselines, y, df = trainer.prepare_data()
split = int(len(X) * 0.8)
X_v, b_v, y_v = X[split:], baselines[split:], y[split:]
X_tr, b_tr, y_tr = X[:split], baselines[:split], y[:split]
d_tr, d_v = y_tr - b_tr, y_v - b_v
n_tgt = 3
names = trainer.target_columns
y_v_orig = trainer.scaler_y.inverse_transform(y_v)
b_v_orig = trainer.scaler_y.inverse_transform(b_v)

# Rebuild LSTM predictions
from improved_stacking_trainer import build_simple_lstm, prepare_gbm_features_v2
import tensorflow as tf

print("\nReloading LSTM predictions...")
seeds = [42, 123, 777]
lstm_deltas = []
for s in seeds:
    wpath = f'model/lstm_seed_{s}.weights.h5'
    if not Path(wpath).exists():
        print(f"  SKIP seed {s} — weights not found")
        continue
    m = build_simple_lstm(10, 113, 3, s)
    m.compile(optimizer='adam', loss='mse')
    m.load_weights(wpath)
    pred = m.predict([X_v, b_v], verbose=0, batch_size=256)
    lstm_deltas.append(pred)
    print(f"  Loaded seed {s}")

print("Reloading GBM predictions...")
gbm_all = joblib.load('model/stacked_gbm_models_v2.pkl')
X_gbm_v = prepare_gbm_features_v2(X_v, b_v)

gbm_deltas = {}
for gname, models in gbm_all.items():
    preds = np.zeros((len(X_v), n_tgt))
    for t in range(n_tgt):
        if gname == 'xgb':
            import xgboost as xgb
            preds[:, t] = models[t].predict(xgb.DMatrix(X_gbm_v))
        else:
            preds[:, t] = models[t].predict(X_gbm_v)
    gbm_deltas[gname] = preds
    po = trainer.scaler_y.inverse_transform(b_v + preds)
    maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
    print(f"  {gname}: Avg={np.mean(maes):.4f}")

# Build prediction pool
all_preds = {}
for i, ld in enumerate(lstm_deltas):
    all_preds[f'lstm{i}'] = ld
for gn, gd in gbm_deltas.items():
    all_preds[gn] = gd

pred_names = list(all_preds.keys())
print(f"\nBase models: {pred_names}")


# ============================================================
# Try all subsets of 2+ models with Ridge meta-learner
# ============================================================
print("\n" + "="*70)
print("SEARCHING ALL SUBSETS (Ridge alpha=1.0)")
print("="*70)

results = []

for size in range(1, len(pred_names) + 1):
    for combo in combinations(pred_names, size):
        meta_delta = np.zeros_like(d_v)
        for t in range(n_tgt):
            feats = [all_preds[n][:, t:t+1] for n in combo]
            feats.append(b_v)
            X_m = np.hstack(feats)
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_m, d_v[:, t])
            meta_delta[:, t] = ridge.predict(X_m)
        
        po = trainer.scaler_y.inverse_transform(b_v + meta_delta)
        maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
        avg = np.mean(maes)
        results.append((avg, combo, maes))

results.sort(key=lambda x: x[0])

print("\nTop 15 combinations:")
for i, (avg, combo, maes) in enumerate(results[:15]):
    combo_str = "+".join(combo)
    print(f"  {i+1:2d}. {avg:.4f}  PTS={maes[0]:.2f} TRB={maes[1]:.2f} AST={maes[2]:.2f}  [{combo_str}]")

# ============================================================
# Try different Ridge alphas on top combos
# ============================================================
print("\n" + "="*70)
print("ALPHA SEARCH on top 3 combos")
print("="*70)

alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
best_overall = (999, None, None, None)

for rank, (_, combo, _) in enumerate(results[:3]):
    combo_str = "+".join(combo)
    print(f"\n  Combo: [{combo_str}]")
    for alpha in alphas:
        meta_delta = np.zeros_like(d_v)
        for t in range(n_tgt):
            feats = [all_preds[n][:, t:t+1] for n in combo]
            feats.append(b_v)
            X_m = np.hstack(feats)
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_m, d_v[:, t])
            meta_delta[:, t] = ridge.predict(X_m)
        po = trainer.scaler_y.inverse_transform(b_v + meta_delta)
        maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
        avg = np.mean(maes)
        marker = " ***" if avg < best_overall[0] else ""
        print(f"    alpha={alpha:5.2f}: {avg:.4f}{marker}")
        if avg < best_overall[0]:
            best_overall = (avg, combo, alpha, maes)

# ============================================================
# Try simple weighted averages (no meta-learner)
# ============================================================
print("\n" + "="*70)
print("SIMPLE WEIGHTED AVERAGES (no meta-learner)")
print("="*70)

# CatBoost + LSTM ensemble blend
for w_cb in np.arange(0.5, 1.01, 0.05):
    w_lstm = 1.0 - w_cb
    blend_delta = w_cb * gbm_deltas['cb'] + w_lstm * np.mean(lstm_deltas, axis=0)
    po = trainer.scaler_y.inverse_transform(b_v + blend_delta)
    maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
    avg = np.mean(maes)
    print(f"  CB={w_cb:.2f} + LSTM_ens={w_lstm:.2f}: {avg:.4f}")

# CatBoost + XGBoost blend
print()
for w_cb in np.arange(0.5, 1.01, 0.05):
    w_xgb = 1.0 - w_cb
    blend_delta = w_cb * gbm_deltas['cb'] + w_xgb * gbm_deltas['xgb']
    po = trainer.scaler_y.inverse_transform(b_v + blend_delta)
    maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
    avg = np.mean(maes)
    print(f"  CB={w_cb:.2f} + XGB={w_xgb:.2f}: {avg:.4f}")

# Triple GBM blend
print()
for w_cb in np.arange(0.4, 0.81, 0.1):
    for w_xgb in np.arange(0.1, 0.51, 0.1):
        w_lgb = 1.0 - w_cb - w_xgb
        if w_lgb < 0: continue
        blend_delta = w_cb * gbm_deltas['cb'] + w_xgb * gbm_deltas['xgb'] + w_lgb * gbm_deltas['lgb']
        po = trainer.scaler_y.inverse_transform(b_v + blend_delta)
        maes = [mean_absolute_error(y_v_orig[:, t], po[:, t]) for t in range(n_tgt)]
        avg = np.mean(maes)
        if avg < 4.47:
            print(f"  CB={w_cb:.1f} XGB={w_xgb:.1f} LGB={w_lgb:.1f}: {avg:.4f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
avg, combo, alpha, maes = best_overall
combo_str = "+".join(combo)
print(f"  Best Ridge meta: {avg:.4f} [{combo_str}] alpha={alpha}")
print(f"    PTS={maes[0]:.2f}  TRB={maes[1]:.2f}  AST={maes[2]:.2f}")
print(f"  CatBoost alone:  4.4618")
print(f"  Prev stacked v1: 4.5984")
print(f"  Old ensemble:    4.7950")
print("="*70)
