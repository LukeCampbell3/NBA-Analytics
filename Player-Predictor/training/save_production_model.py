#!/usr/bin/env python3
"""Save CatBoost as the production model and update inference artifacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, r2_score
from unified_moe_trainer import UnifiedMoETrainer
from improved_stacking_trainer import prepare_gbm_features_v2

print("="*60)
print("SAVING CATBOOST AS PRODUCTION MODEL")
print("="*60)

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

# Prepare features
X_gbm_tr = prepare_gbm_features_v2(X_tr, b_tr)
X_gbm_v = prepare_gbm_features_v2(X_v, b_v)
print(f"GBM features: {X_gbm_tr.shape[1]}")

# Train final CatBoost models
from catboost import CatBoostRegressor

cb_models = []
for t in range(n_tgt):
    print(f"\nTraining CatBoost for {names[t]}...")
    m = CatBoostRegressor(
        loss_function='MAE', iterations=500, learning_rate=0.05,
        depth=6, l2_leaf_reg=1.0, min_data_in_leaf=20,
        subsample=0.8, colsample_bylevel=0.8,
        early_stopping_rounds=30, verbose=0, random_seed=42)
    m.fit(X_gbm_tr, d_tr[:, t], eval_set=(X_gbm_v, d_v[:, t]), verbose=0)
    pred = m.predict(X_gbm_v)
    mae_scaled = mean_absolute_error(d_v[:, t], pred)
    print(f"  Scaled MAE: {mae_scaled:.4f}, iters: {m.best_iteration_}")
    cb_models.append(m)

# Evaluate
cb_delta = np.zeros_like(d_v)
for t in range(n_tgt):
    cb_delta[:, t] = cb_models[t].predict(X_gbm_v)

cb_preds = trainer.scaler_y.inverse_transform(b_v + cb_delta)
print("\n" + "="*60)
print("FINAL PRODUCTION MODEL RESULTS")
print("="*60)
total = 0
for i, t in enumerate(names):
    mae = mean_absolute_error(y_v_orig[:, i], cb_preds[:, i])
    r2 = r2_score(y_v_orig[:, i], cb_preds[:, i])
    bl_mae = mean_absolute_error(y_v_orig[:, i], b_v_orig[:, i])
    imp = bl_mae - mae
    pct = (imp / bl_mae * 100) if bl_mae > 0 else 0
    total += mae
    print(f"  {t}: MAE={mae:.4f}  R²={r2:.4f}  (baseline: {bl_mae:.4f}, +{imp:.4f} / {pct:+.1f}%)")
avg = total / n_tgt
print(f"\n  Avg MAE: {avg:.4f}")
print(f"  vs old ensemble (4.795): {4.795 - avg:+.4f} ({(4.795 - avg)/4.795*100:.1f}% better)")
print(f"  vs prev stacked (4.598): {4.598 - avg:+.4f} ({(4.598 - avg)/4.598*100:.1f}% better)")

# Save
print("\nSaving production artifacts...")
joblib.dump(cb_models, 'model/production_catboost_models.pkl')
joblib.dump(trainer.scaler_x, 'model/unified_moe_scaler_x.pkl')
joblib.dump(trainer.scaler_y, 'model/unified_moe_scaler_y.pkl')

metadata = {
    'model_type': 'catboost_v2',
    'gbm_feature_dim': X_gbm_tr.shape[1],
    'gbm_feature_version': 'v2',
    'n_targets': n_tgt,
    'target_columns': names,
    'avg_mae': float(avg),
    'per_target_mae': {names[i]: float(mean_absolute_error(y_v_orig[:, i], cb_preds[:, i])) for i in range(n_tgt)},
    'feature_columns': trainer.feature_columns,
    'baseline_features': trainer.baseline_features,
    'player_mapping': trainer.player_mapping,
    'team_mapping': trainer.team_mapping,
    'opponent_mapping': trainer.opponent_mapping,
    'seq_len': 10,
}
with open('model/unified_moe_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Done! Production model saved.")
print("="*60)
