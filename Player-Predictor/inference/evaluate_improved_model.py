#!/usr/bin/env python3
"""
Evaluate the NEW MoE Improved Model (with expert collapse fixes)

This script evaluates the model trained with integrate_moe_improvements.py
which has all the expert collapse fixes applied.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

print("="*80)
print("EVALUATING NEW MOE IMPROVED MODEL (with expert collapse fixes)")
print("="*80)

# Load cached sequences
print("\n📂 Loading cached data...")
X_seq = np.load('cache/X_sequences.npy')
y_seq = np.load('cache/y_sequences.npy')

with open('cache/metadata.json', 'r') as f:
    metadata = json.load(f)

feature_columns = metadata['feature_columns']
all_targets = metadata['target_columns']
target_columns = all_targets[:3]  # PTS, TRB, AST
y_seq = y_seq[:, :3]

print(f"✓ Data shape: X={X_seq.shape}, y={y_seq.shape}")
print(f"✓ Targets: {target_columns}")

# Split data (80/20)
split_idx = int(len(X_seq) * 0.8)
X_test = X_seq[split_idx:]
y_test = y_seq[split_idx:]

# Extract baselines
baseline_feature_names = ['PTS_rolling_avg', 'TRB_rolling_avg', 'AST_rolling_avg']
baseline_indices = [feature_columns.index(f) for f in baseline_feature_names]
b_test = X_test[:, -1, baseline_indices]

print(f"✓ Test set: {len(X_test)} samples")

# Rebuild model architecture
print("\n🏗️  Rebuilding model architecture...")
from training.integrate_moe_improvements import MoEImprovedTrainer

trainer = MoEImprovedTrainer()

# Check if model file exists
model_path = Path('model/improved_baseline_final.weights.h5')
if not model_path.exists():
    print(f"\n❌ ERROR: Model file not found: {model_path}")
    print("\n⚠️  You need to train the model first!")
    print("Run: python training/integrate_moe_improvements.py")
    sys.exit(1)

# Build model
print("✓ Building model with MoE improvements...")
net = trainer.build_model_with_phase2()

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

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer)

# Load weights
print(f"📥 Loading weights from {model_path}...")
try:
    model.load_weights(str(model_path))
    print("✓ Weights loaded successfully")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    sys.exit(1)

# Make predictions
print("\n🔮 Making predictions...")
preds = model.predict([X_test, b_test], verbose=0, batch_size=64)

# Handle output format
if isinstance(preds, list):
    preds = preds[0]

if preds.shape[1] > len(target_columns):
    pred_means = preds[:, :len(target_columns)]
else:
    pred_means = preds

# Compute metrics
print("\n" + "="*80)
print("EVALUATION RESULTS (scaled space)")
print("="*80)

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
    
    improvement = baseline_mae - mae
    pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
    
    print(f"\n{target}:")
    print(f"  Model MAE:    {mae:.4f}")
    print(f"  Model R²:     {r2:.4f}")
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    print(f"  Baseline R²:  {baseline_r2:.4f}")
    print(f"  Improvement:  {improvement:+.4f} ({pct:+.1f}%) {'✓' if improvement > 0 else '✗'}")

print(f"\n{'='*80}")
print(f"MACRO AVERAGES:")
macro_mae = np.mean(maes)
macro_r2 = np.mean(r2s)
macro_baseline_mae = np.mean(baseline_maes)
macro_baseline_r2 = np.mean(baseline_r2s)
macro_imp = macro_baseline_mae - macro_mae
macro_pct = (macro_imp / macro_baseline_mae * 100) if macro_baseline_mae > 0 else 0

print(f"  Model MAE:    {macro_mae:.4f}")
print(f"  Model R²:     {macro_r2:.4f}")
print(f"  Baseline MAE: {macro_baseline_mae:.4f}")
print(f"  Baseline R²:  {macro_baseline_r2:.4f}")
print(f"  Improvement:  {macro_imp:+.4f} ({macro_pct:+.1f}%)")
print(f"{'='*80}\n")

# Compare to old ensemble
print("📊 COMPARISON TO OLD ENSEMBLE:")
print(f"  Old Ensemble MAE: 4.78")
print(f"  New Model MAE:    {macro_mae:.4f}")
if macro_mae < 4.78:
    print(f"  ✓ NEW MODEL IS BETTER by {4.78 - macro_mae:.4f} MAE")
else:
    print(f"  ✗ Old ensemble is still better by {macro_mae - 4.78:.4f} MAE")

# Check for issues
negative_r2 = [target for i, target in enumerate(target_columns) if r2s[i] < 0]
if negative_r2:
    print(f"\n⚠️  WARNING: Negative R² for {negative_r2}")

# Save results
results = {
    'model_type': 'MoE_Improved_with_collapse_fixes',
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
        'model_mae': float(macro_mae),
        'model_r2': float(macro_r2),
        'baseline_mae': float(macro_baseline_mae),
        'baseline_r2': float(macro_baseline_r2),
        'improvement': float(macro_imp),
        'improvement_pct': float(macro_pct)
    },
    'comparison_to_old_ensemble': {
        'old_ensemble_mae': 4.78,
        'new_model_mae': float(macro_mae),
        'improvement': float(4.78 - macro_mae),
        'is_better': bool(macro_mae < 4.78)
    }
}

output_path = 'inference/improved_model_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to {output_path}")
print("\n✅ Evaluation complete!")
