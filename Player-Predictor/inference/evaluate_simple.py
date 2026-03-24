#!/usr/bin/env python3
"""
Simple evaluation of the newly trained MoE model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

print("Loading cached data...")
# Load preprocessed sequences
X_seq = np.load('cache/X_sequences.npy')
y_seq = np.load('cache/y_sequences.npy')

# Load scaler (for features - X is already scaled in cache)
with open('model/improved_baseline_scaler.pkl', 'rb') as f:
    scaler_x = pickle.load(f)

# For targets, we'll use the same scaler (it scales both X and y in the trainer)
scaler_y = scaler_x

# Load feature and target names
with open('cache/features.txt', 'r') as f:
    feature_columns = [line.strip() for line in f]
with open('cache/targets.txt', 'r') as f:
    target_columns = [line.strip() for line in f]

print(f"Data shape: X={X_seq.shape}, y={y_seq.shape}")
print(f"Features: {len(feature_columns)}, Targets: {target_columns}")

# Use last 20% as test set
test_size = int(len(X_seq) * 0.2)
X_test = X_seq[-test_size:]
y_test = y_seq[-test_size:]

print(f"Test set size: {len(X_test)}")

# Extract baselines (last 3 features are PTS, TRB, AST from previous game)
# Assuming baseline features are at the end of feature list
baseline_indices = [i for i, f in enumerate(feature_columns) if f in ['PTS', 'TRB', 'AST']]
print(f"Baseline feature indices: {baseline_indices}")

# Get baselines from last timestep
b_test = X_test[:, -1, baseline_indices]
print(f"Baselines shape: {b_test.shape}")

# Rebuild model
print("\nRebuilding model architecture...")
from training.integrate_moe_improvements import MoEImprovedTrainer, build_moe_model

trainer = MoEImprovedTrainer()
model = build_moe_model(
    seq_len=X_test.shape[1],
    n_features=X_test.shape[2],
    n_targets=len(target_columns),
    config=trainer.config
)

# Load weights
print("Loading trained weights...")
try:
    model.load_weights('model/improved_baseline_final.weights.h5')
    print("✓ Weights loaded successfully")
except Exception as e:
    print(f"✗ Error loading weights: {e}")
    sys.exit(1)

# Make predictions
print("\nMaking predictions...")
preds = model.predict([X_test, b_test], verbose=1, batch_size=64)

# Handle probabilistic outputs
if isinstance(preds, list):
    preds = preds[0]

if preds.shape[1] > len(target_columns):
    # Extract means from probabilistic output
    pred_means = preds[:, :len(target_columns)]
    print(f"Extracted means from probabilistic output: {pred_means.shape}")
else:
    pred_means = preds

# Inverse transform predictions and targets
pred_means_orig = scaler_y.inverse_transform(pred_means)
y_test_orig = scaler_y.inverse_transform(y_test)
b_test_orig = scaler_y.inverse_transform(b_test)

# Compute metrics
print("\n" + "="*70)
print("MODEL EVALUATION RESULTS")
print("="*70)

maes = []
r2s = []
baseline_maes = []
baseline_r2s = []

for i, target in enumerate(target_columns):
    mae = mean_absolute_error(y_test_orig[:, i], pred_means_orig[:, i])
    r2 = r2_score(y_test_orig[:, i], pred_means_orig[:, i])
    maes.append(mae)
    r2s.append(r2)
    
    baseline_mae = mean_absolute_error(y_test_orig[:, i], b_test_orig[:, i])
    baseline_r2 = r2_score(y_test_orig[:, i], b_test_orig[:, i])
    baseline_maes.append(baseline_mae)
    baseline_r2s.append(baseline_r2)
    
    print(f"\n{target}:")
    print(f"  Model MAE:    {mae:.3f}")
    print(f"  Model R²:     {r2:.3f}")
    print(f"  Baseline MAE: {baseline_mae:.3f}")
    print(f"  Baseline R²:  {baseline_r2:.3f}")
    improvement = baseline_mae - mae
    print(f"  Improvement:  {improvement:+.3f} MAE {'✓' if improvement > 0 else '✗'}")

print(f"\n{'='*70}")
print(f"MACRO AVERAGES:")
print(f"  Model MAE:    {np.mean(maes):.3f}")
print(f"  Model R²:     {np.mean(r2s):.3f}")
print(f"  Baseline MAE: {np.mean(baseline_maes):.3f}")
print(f"  Baseline R²:  {np.mean(baseline_r2s):.3f}")
print(f"  Improvement:  {np.mean(baseline_maes) - np.mean(maes):+.3f} MAE")
print(f"{'='*70}\n")

# Save results
results = {
    'n_samples': len(y_test),
    'targets': target_columns,
    'individual_metrics': {
        target: {
            'model_mae': float(maes[i]),
            'model_r2': float(r2s[i]),
            'baseline_mae': float(baseline_maes[i]),
            'baseline_r2': float(baseline_r2s[i]),
            'improvement': float(baseline_maes[i] - maes[i])
        }
        for i, target in enumerate(target_columns)
    },
    'macro': {
        'model_mae': float(np.mean(maes)),
        'model_r2': float(np.mean(r2s)),
        'baseline_mae': float(np.mean(baseline_maes)),
        'baseline_r2': float(np.mean(baseline_r2s)),
        'improvement': float(np.mean(baseline_maes) - np.mean(maes))
    }
}

with open('inference/new_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to inference/new_model_results.json")
