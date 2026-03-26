#!/usr/bin/env python3
"""
Evaluate the saved .h5 model directly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

print("Loading cached sequences...")
X_seq = np.load('cache/X_sequences.npy')
y_seq = np.load('cache/y_sequences.npy')

# Load metadata
with open('cache/metadata.json', 'r') as f:
    metadata = json.load(f)

feature_columns = metadata['feature_columns']
all_targets = metadata['target_columns']
target_columns = all_targets[:3]  # PTS, TRB, AST
y_seq = y_seq[:, :3]

print(f"Data shape: X={X_seq.shape}, y={y_seq.shape}")
print(f"Targets: {target_columns}")

# Split data (80/20)
split_idx = int(len(X_seq) * 0.8)
X_test = X_seq[split_idx:]
y_test = y_seq[split_idx:]

# Extract baselines
baseline_feature_names = ['PTS_rolling_avg', 'TRB_rolling_avg', 'AST_rolling_avg']
baseline_indices = [feature_columns.index(f) for f in baseline_feature_names]
b_test = X_test[:, -1, baseline_indices]

print(f"Test set: {len(X_test)} samples")
print(f"Baselines shape: {b_test.shape}")

# Load model
print("\nLoading model from improved_baseline_final.h5...")
try:
    # Need to import custom objects
    from training.integrate_moe_improvements import DeltaTrainedKerasModel
    
    model = tf.keras.models.load_model(
        'model/improved_baseline_final.h5',
        custom_objects={'DeltaTrainedKerasModel': DeltaTrainedKerasModel},
        compile=False
    )
    print("✓ Model loaded successfully")
    print(f"Model type: {type(model)}")
    print(f"Model inputs: {[inp.name for inp in model.inputs]}")
    print(f"Model outputs: {[out.name for out in model.outputs]}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Make predictions
print("\nMaking predictions...")
try:
    preds = model.predict([X_test, b_test], verbose=1, batch_size=64)
    print(f"Predictions shape: {preds.shape if not isinstance(preds, list) else [p.shape for p in preds]}")
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Handle output format
if isinstance(preds, list):
    print(f"Model returned list with {len(preds)} elements")
    preds = preds[0]  # Take first output

if preds.shape[1] > len(target_columns):
    pred_means = preds[:, :len(target_columns)]
    print(f"Extracted means from probabilistic output")
else:
    pred_means = preds

# Compute metrics
print("\n" + "="*70)
print("MODEL EVALUATION RESULTS (scaled space)")
print("="*70)

maes = []
r2s = []
baseline_maes = []
baseline_r2s = []

for i, target in enumerate(target_columns):
    mae = mean_absolute_error(y_test[:, i], pred_means[:, i])
    r2 = r2_score(y_test[:, i], pred_means[:, i])
    maes.append(mae)
    r2s.append(r2)
    
    baseline_mae = mean_absolute_error(y_test[:, i], b_test[:, i])
    baseline_r2 = r2_score(y_test[:, i], b_test[:, i])
    baseline_maes.append(baseline_mae)
    baseline_r2s.append(baseline_r2)
    
    print(f"\n{target}:")
    print(f"  Model MAE:    {mae:.4f}")
    print(f"  Model R²:     {r2:.4f}")
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    print(f"  Baseline R²:  {baseline_r2:.4f}")
    improvement = baseline_mae - mae
    pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
    print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%) {'✓' if improvement > 0 else '✗'}")

print(f"\n{'='*70}")
print(f"MACRO AVERAGES:")
print(f"  Model MAE:    {np.mean(maes):.4f}")
print(f"  Model R²:     {np.mean(r2s):.4f}")
print(f"  Baseline MAE: {np.mean(baseline_maes):.4f}")
print(f"  Baseline R²:  {np.mean(baseline_r2s):.4f}")
macro_imp = np.mean(baseline_maes) - np.mean(maes)
macro_pct = (macro_imp / np.mean(baseline_maes) * 100) if np.mean(baseline_maes) > 0 else 0
print(f"  Improvement:  {macro_imp:+.4f} ({macro_pct:+.1f}%)")
print(f"{'='*70}\n")

# Check for issues
negative_r2 = [target for i, target in enumerate(target_columns) if r2s[i] < 0]
if negative_r2:
    print(f"⚠️  WARNING: Negative R² for {negative_r2} - worse than mean baseline!")

worse_than_baseline = [target for i, target in enumerate(target_columns) if maes[i] > baseline_maes[i]]
if worse_than_baseline:
    print(f"⚠️  WARNING: Worse MAE than baseline for {worse_than_baseline}")

# Save results
results = {
    'n_samples': len(y_test),
    'targets': target_columns,
    'note': 'Metrics in scaled space',
    'individual_metrics': {
        target: {
            'model_mae': float(maes[i]),
            'model_r2': float(r2s[i]),
            'baseline_mae': float(baseline_maes[i]),
            'baseline_r2': float(baseline_r2s[i]),
            'improvement': float(baseline_maes[i] - maes[i]),
            'improvement_pct': float((baseline_maes[i] - maes[i]) / baseline_maes[i] * 100) if baseline_maes[i] > 0 else 0
        }
        for i, target in enumerate(target_columns)
    },
    'macro': {
        'model_mae': float(np.mean(maes)),
        'model_r2': float(np.mean(r2s)),
        'baseline_mae': float(np.mean(baseline_maes)),
        'baseline_r2': float(np.mean(baseline_r2s)),
        'improvement': float(macro_imp),
        'improvement_pct': float(macro_pct)
    }
}

with open('inference/new_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to inference/new_model_results.json")
