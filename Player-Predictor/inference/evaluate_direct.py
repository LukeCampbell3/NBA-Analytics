#!/usr/bin/env python3
"""
Direct evaluation using the same data loading as training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

print("Loading cached sequences...")
# Load preprocessed sequences (already scaled)
X_seq = np.load('cache/X_sequences.npy')
y_seq = np.load('cache/y_sequences.npy')

# Load metadata
with open('cache/metadata.json', 'r') as f:
    metadata = json.load(f)

feature_columns = metadata['feature_columns']
all_targets = metadata['target_columns']

# We only train on first 3 targets: PTS, TRB, AST
target_columns = all_targets[:3]
y_seq = y_seq[:, :3]

print(f"Data shape: X={X_seq.shape}, y={y_seq.shape}")
print(f"Features: {len(feature_columns)}")
print(f"Targets: {target_columns}")

# Split data (80/20 train/test)
split_idx = int(len(X_seq) * 0.8)
X_test = X_seq[split_idx:]
y_test = y_seq[split_idx:]

print(f"Test set size: {len(X_test)}")

# Extract baselines from last timestep
# Baseline features are PTS, TRB, AST rolling averages
baseline_feature_names = ['PTS_rolling_avg', 'TRB_rolling_avg', 'AST_rolling_avg']
baseline_indices = [feature_columns.index(f) for f in baseline_feature_names]
print(f"Baseline feature indices: {baseline_indices}")

b_test = X_test[:, -1, baseline_indices]
print(f"Baselines shape: {b_test.shape}")

# Rebuild model
print("\nRebuilding model architecture...")
from training.integrate_moe_improvements import MoEImprovedTrainer

trainer = MoEImprovedTrainer()

# Load data to initialize mappings
print("Initializing trainer (loading data for mappings)...")
trainer.load_data()

# Build the inner network (same as training)
net = trainer.build_model_with_phase2()

# Wrap it the same way as training
from training.improved_baseline_trainer import DeltaTrainedKerasModel

model = DeltaTrainedKerasModel(
    net=net,
    target_columns=target_columns,
    use_probabilistic=trainer.config.get("use_probabilistic", False),
    huber_delta=float(trainer.config.get("huber_delta", 1.0)),
    band=trainer.config.get("band_per_target", None),
    band_weight=float(trainer.config.get("band_weight", 0.0)),
    outlier_gamma=float(trainer.config.get("outlier_gamma", 2.0)),
    outlier_alpha=float(trainer.config.get("outlier_alpha", 0.25)),
    outlier_weight=float(trainer.config.get("outlier_weight", 0.5)),
    spike_z_thr=float(trainer.config.get("spike_z_thr", 2.75)),
    spike_min_pos_frac=float(trainer.config.get("spike_min_pos_frac", 0.03)),
)

# Compile (needed before loading weights)
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer)

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

# Note: Data is already scaled, so we work in scaled space
# Compute metrics
print("\n" + "="*70)
print("MODEL EVALUATION RESULTS (in scaled space)")
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
    pct_improvement = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
    print(f"  Improvement:  {improvement:+.4f} MAE ({pct_improvement:+.1f}%) {'✓' if improvement > 0 else '✗'}")

print(f"\n{'='*70}")
print(f"MACRO AVERAGES:")
print(f"  Model MAE:    {np.mean(maes):.4f}")
print(f"  Model R²:     {np.mean(r2s):.4f}")
print(f"  Baseline MAE: {np.mean(baseline_maes):.4f}")
print(f"  Baseline R²:  {np.mean(baseline_r2s):.4f}")
macro_improvement = np.mean(baseline_maes) - np.mean(maes)
macro_pct = (macro_improvement / np.mean(baseline_maes) * 100) if np.mean(baseline_maes) > 0 else 0
print(f"  Improvement:  {macro_improvement:+.4f} MAE ({macro_pct:+.1f}%)")
print(f"{'='*70}\n")

# Check for negative R² (worse than baseline)
negative_r2 = [target for i, target in enumerate(target_columns) if r2s[i] < 0]
if negative_r2:
    print(f"⚠️  WARNING: Negative R² for {negative_r2} - model worse than mean baseline!")

# Save results
results = {
    'n_samples': len(y_test),
    'targets': target_columns,
    'note': 'Metrics computed in scaled space',
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
        'improvement': float(np.mean(baseline_maes) - np.mean(maes)),
        'improvement_pct': float((np.mean(baseline_maes) - np.mean(maes)) / np.mean(baseline_maes) * 100) if np.mean(baseline_maes) > 0 else 0
    }
}

with open('inference/new_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to inference/new_model_results.json")
