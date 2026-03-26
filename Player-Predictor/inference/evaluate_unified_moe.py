#!/usr/bin/env python3
"""
Evaluate the production model.
Supports: CatBoost v2 (current best), stacked LSTM+GBM, simple LSTM, or MoE.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

import numpy as np
import json
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

print("="*80)
print("EVALUATING PRODUCTION MODEL")
print("="*80)

# Load metadata to determine model type
meta_path = Path('model/unified_moe_metadata.json')
if meta_path.exists():
    with open(meta_path) as f:
        metadata = json.load(f)
    model_type = metadata.get('model_type', 'unknown')
    print(f"  Model type: {model_type}")
else:
    model_type = 'unknown'

# Load data
print("\n Loading data...")
from training.unified_moe_trainer import UnifiedMoETrainer
trainer = UnifiedMoETrainer()
X, baselines, y, df = trainer.prepare_data()

split_idx = int(len(X) * 0.8)
X_test, b_test, y_test = X[split_idx:], baselines[split_idx:], y[split_idx:]
n_targets = len(trainer.target_columns)
names = trainer.target_columns

print(f" Test set: {len(X_test)} samples")

y_test_orig = trainer.scaler_y.inverse_transform(y_test)
b_test_orig = trainer.scaler_y.inverse_transform(b_test)

# Baseline
print("\n--- BASELINE (rolling avg) ---")
for i, t in enumerate(names):
    mae = mean_absolute_error(y_test_orig[:, i], b_test_orig[:, i])
    print(f"  {t}: {mae:.4f}")

# ============================================================
# CatBoost v2 path (current production)
# ============================================================
cb_path = Path('model/production_catboost_models.pkl')
if model_type == 'catboost_v2' and cb_path.exists():
    print("\n--- CATBOOST v2 (production) ---")
    from training.improved_stacking_trainer import prepare_gbm_features_v2

    cb_models = joblib.load(str(cb_path))
    X_gbm_test = prepare_gbm_features_v2(X_test, b_test)
    delta_test = y_test - b_test

    cb_delta = np.zeros((len(X_test), n_targets))
    for t in range(n_targets):
        cb_delta[:, t] = cb_models[t].predict(X_gbm_test)

    cb_preds = trainer.scaler_y.inverse_transform(b_test + cb_delta)
    maes = []
    for i, t in enumerate(names):
        mae = mean_absolute_error(y_test_orig[:, i], cb_preds[:, i])
        r2 = r2_score(y_test_orig[:, i], cb_preds[:, i])
        bl_mae = mean_absolute_error(y_test_orig[:, i], b_test_orig[:, i])
        imp = bl_mae - mae
        pct = (imp / bl_mae * 100) if bl_mae > 0 else 0
        maes.append(mae)
        print(f"  {t}: MAE={mae:.4f}  R²={r2:.4f}  (baseline: {bl_mae:.4f}, +{imp:.4f} / {pct:+.1f}%)")
    avg = np.mean(maes)
    print(f"  Avg MAE: {avg:.4f}")
    best_name, best_avg, best_maes = 'catboost_v2', avg, maes

# ============================================================
# Stacked LSTM+GBM path (fallback)
# ============================================================
elif Path('model/stacked_gbm_models.pkl').exists() and Path('model/stacked_meta_models.pkl').exists():
    import tensorflow as tf
    print("\n--- STACKED LSTM+GBM+META ---")
    use_simple = trainer.config.get("use_simple_model", False)

    if use_simple:
        net = trainer.build_simple_lstm_model()
        from tensorflow.keras.models import Model as KerasModel
        simple_model = KerasModel(inputs=net.inputs, outputs=net.outputs[0])
        simple_model.compile(optimizer='adam', loss='mse')
        simple_model.load_weights(str(Path('model/unified_moe_best.weights.h5')))
        lstm_delta = simple_model.predict([X_test, b_test], verbose=0, batch_size=256)
    else:
        from training.unified_moe_trainer import DeltaTrainedKerasModel
        net = trainer.build_model_with_anti_collapse()
        model = DeltaTrainedKerasModel(net=net, target_columns=names,
            use_probabilistic=False, huber_delta=1.0, band=None, band_weight=0.0,
            outlier_gamma=2.0, outlier_alpha=0.25, outlier_weight=0.5,
            spike_z_thr=2.75, spike_min_pos_frac=0.03)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0))
        model.load_weights(str(Path('model/unified_moe_best.weights.h5')))
        preds = model.predict([X_test, b_test], verbose=0, batch_size=64)
        if isinstance(preds, list): preds = preds[0]
        lstm_delta = preds - b_test

    gbm_models = joblib.load('model/stacked_gbm_models.pkl')
    meta_models = joblib.load('model/stacked_meta_models.pkl')
    from training.unified_moe_trainer import UnifiedMoETrainer as _T
    X_gbm_test = _T().prepare_gbm_features(X_test, b_test) if not hasattr(trainer, 'prepare_gbm_features_v2') else trainer.prepare_gbm_features(X_test, b_test)

    gbm_delta = np.zeros((len(X_test), n_targets))
    for t in range(n_targets):
        gbm_delta[:, t] = gbm_models[t].predict(X_gbm_test)

    meta_delta = np.zeros_like(lstm_delta)
    for t in range(n_targets):
        feats = np.hstack([lstm_delta[:, t:t+1], gbm_delta[:, t:t+1], b_test])
        meta_delta[:, t] = meta_models[t].predict(feats)

    meta_preds = trainer.scaler_y.inverse_transform(b_test + meta_delta)
    maes = []
    for i, t in enumerate(names):
        mae = mean_absolute_error(y_test_orig[:, i], meta_preds[:, i])
        r2 = r2_score(y_test_orig[:, i], meta_preds[:, i])
        maes.append(mae)
        print(f"  {t}: MAE={mae:.4f}  R²={r2:.4f}")
    avg = np.mean(maes)
    print(f"  Avg MAE: {avg:.4f}")
    best_name, best_avg, best_maes = 'stacked', avg, maes

else:
    print("\n  No production model found. Run training first.")
    sys.exit(1)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"  Current model ({best_name}): {best_avg:.4f}")
print(f"  Old ensemble:                4.7950")
if best_avg < 4.795:
    diff = 4.795 - best_avg
    print(f"  Improvement: {diff:.4f} ({diff/4.795*100:.1f}% better)")
print(f"{'='*60}")

# Save results
results = {
    'model_type': best_name,
    'n_samples': len(y_test),
    'targets': names,
    'avg_mae': float(best_avg),
    'per_target': {names[i]: float(best_maes[i]) for i in range(n_targets)},
    'comparison': {'old_ensemble': 4.795, 'current': float(best_avg),
                   'improvement': float(4.795 - best_avg)}
}
with open('inference/unified_moe_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n Results saved to inference/unified_moe_results.json")
print(" Evaluation complete!")
