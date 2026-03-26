#!/usr/bin/env python3
"""
Quick evaluation of the newly trained MoE model with all 4 phases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

# Load the model
print("Loading model...")
try:
    from training.integrate_moe_improvements import MoEImprovedTrainer
    
    # Initialize trainer to get config
    trainer = MoEImprovedTrainer()
    
    # Load data
    print("Loading data...")
    from training.improved_baseline_trainer import ImprovedBaselineTrainer
    base_trainer = ImprovedBaselineTrainer()
    base_trainer.load_data()
    
    # Create sequences
    X_seq, baselines, y, metadata = base_trainer.create_sequences(
        base_trainer.train_data,
        base_trainer.feature_columns,
        base_trainer.baseline_features,
        base_trainer.target_columns
    )
    
    print(f"Data shape: X={X_seq.shape}, baselines={baselines.shape}, y={y.shape}")
    
    # Use last 20% as test set
    test_size = int(len(X_seq) * 0.2)
    X_test = X_seq[-test_size:]
    b_test = baselines[-test_size:]
    y_test = y[-test_size:]
    
    # Load scaler
    with open('model/improved_baseline_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale features
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Rebuild and load model
    print("Rebuilding model architecture...")
    from training.integrate_moe_improvements import build_moe_model
    
    # Get model config from trainer
    model = build_moe_model(
        seq_len=trainer.seq_len,
        n_features=len(base_trainer.feature_columns),
        n_targets=len(base_trainer.target_columns),
        config=trainer.config
    )
    
    # Load weights
    print("Loading trained weights...")
    model.load_weights('model/improved_baseline_final.weights.h5')
    
    # Make predictions
    print("Making predictions...")
    preds = model.predict([X_test_scaled, b_test], verbose=0, batch_size=64)
    
    # Handle probabilistic outputs
    if preds.shape[1] > 3:
        # Extract means from probabilistic output
        pred_means = preds[:, :3]
    else:
        pred_means = preds
    
    # Compute metrics
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    targets = ['PTS', 'TRB', 'AST']
    maes = []
    r2s = []
    
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_test[:, i], pred_means[:, i])
        r2 = r2_score(y_test[:, i], pred_means[:, i])
        maes.append(mae)
        r2s.append(r2)
        
        print(f"\n{target}:")
        print(f"  MAE: {mae:.3f}")
        print(f"  R²:  {r2:.3f}")
        
        # Baseline comparison
        baseline_mae = mean_absolute_error(y_test[:, i], b_test[:, i])
        baseline_r2 = r2_score(y_test[:, i], b_test[:, i])
        print(f"  Baseline MAE: {baseline_mae:.3f}")
        print(f"  Baseline R²:  {baseline_r2:.3f}")
        print(f"  Improvement: {baseline_mae - mae:.3f} MAE reduction")
    
    print(f"\n{'='*70}")
    print(f"MACRO AVERAGES:")
    print(f"  MAE: {np.mean(maes):.3f}")
    print(f"  R²:  {np.mean(r2s):.3f}")
    print(f"{'='*70}\n")
    
    # Save results
    results = {
        'n_samples': len(y_test),
        'targets': targets,
        'individual_metrics': {
            target: {
                'mae': float(maes[i]),
                'r2': float(r2s[i])
            }
            for i, target in enumerate(targets)
        },
        'macro': {
            'mae': float(np.mean(maes)),
            'r2': float(np.mean(r2s))
        }
    }
    
    with open('inference/new_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to inference/new_model_results.json")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
