#!/usr/bin/env python3
"""
Debug R² Calculation Discrepancy
Compare Keras metric vs final evaluation
"""

import numpy as np
import tensorflow as tf
from fixed_epochs_trainer import create_fixed_epochs_trainer
import joblib

def test_r2_calculations():
    """Test different R² calculation methods"""
    
    print("🔍 Debugging R² Calculation Discrepancy")
    print("=" * 70)
    
    # Load the trained model
    print("\n📊 Loading model and data...")
    trainer = create_fixed_epochs_trainer()
    
    # Load data
    X, baselines, y, df = trainer.prepare_data()
    
    # Train/validation split
    split_idx = int(0.8 * len(X))
    X_val = X[split_idx:]
    baselines_val = baselines[split_idx:]
    y_val = y[split_idx:]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    
    X_train = X[:split_idx]
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_numeric = X_train_flat[:, 3:]
    
    scaler = StandardScaler()
    scaler.fit(X_train_numeric)
    
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_numeric = X_val_flat[:, 3:]
    X_val_numeric_scaled = scaler.transform(X_val_numeric)
    X_val_scaled = np.concatenate([X_val_flat[:, :3], X_val_numeric_scaled], axis=1)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    # Load model
    print("🏗️ Building model...")
    model = trainer.build_model()
    
    try:
        model.load_weights("model/fixed_epochs_final.h5")
        print("✅ Loaded trained weights")
    except:
        print("❌ Could not load weights - using random initialization")
        print("   This is just for testing the calculation methods")
    
    # Get predictions
    print("\n🔮 Getting predictions...")
    predictions = model.predict([X_val_scaled, baselines_val], verbose=0)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"Target columns: {trainer.target_columns}")
    
    # Extract means
    if trainer.config["use_probabilistic"]:
        pred_means = predictions[:, :len(trainer.target_columns)]
        print(f"Extracted means shape: {pred_means.shape}")
        print(f"Expected: ({len(y_val)}, {len(trainer.target_columns)})")
    else:
        pred_means = predictions
    
    print(f"\nPrediction stats:")
    print(f"  Mean: {np.mean(pred_means, axis=0)}")
    print(f"  Std: {np.std(pred_means, axis=0)}")
    print(f"\nActual stats:")
    print(f"  Mean: {np.mean(y_val, axis=0)}")
    print(f"  Std: {np.std(y_val, axis=0)}")
    
    # Method 1: Keras metric (WRONG - global mean)
    print("\n" + "=" * 70)
    print("METHOD 1: Keras Metric (Global Mean)")
    print("=" * 70)
    
    ss_res_keras = np.sum((y_val - pred_means) ** 2)
    ss_tot_keras = np.sum((y_val - np.mean(y_val)) ** 2)  # Global mean
    r2_keras = 1 - ss_res_keras / ss_tot_keras
    
    print(f"SS_res: {ss_res_keras:.2f}")
    print(f"SS_tot (global mean): {ss_tot_keras:.2f}")
    print(f"R² (Keras method): {r2_keras:.4f}")
    
    # Method 2: Correct R² (per-stat mean)
    print("\n" + "=" * 70)
    print("METHOD 2: Correct R² (Per-Stat Mean)")
    print("=" * 70)
    
    ss_res_correct = np.sum((y_val - pred_means) ** 2)
    ss_tot_correct = np.sum((y_val - np.mean(y_val, axis=0)) ** 2)  # Per-stat mean
    r2_correct = 1 - ss_res_correct / ss_tot_correct
    
    print(f"SS_res: {ss_res_correct:.2f}")
    print(f"SS_tot (per-stat mean): {ss_tot_correct:.2f}")
    print(f"R² (Correct method): {r2_correct:.4f}")
    
    # Method 3: Per-stat R² then average
    print("\n" + "=" * 70)
    print("METHOD 3: Per-Stat R² Then Average")
    print("=" * 70)
    
    r2_per_stat = []
    for i, stat in enumerate(trainer.target_columns):
        ss_res_i = np.sum((y_val[:, i] - pred_means[:, i]) ** 2)
        ss_tot_i = np.sum((y_val[:, i] - np.mean(y_val[:, i])) ** 2)
        r2_i = 1 - ss_res_i / ss_tot_i
        r2_per_stat.append(r2_i)
        print(f"  {stat}: R²={r2_i:.4f}, SS_res={ss_res_i:.2f}, SS_tot={ss_tot_i:.2f}")
    
    r2_avg = np.mean(r2_per_stat)
    print(f"\nAverage R²: {r2_avg:.4f}")
    
    # Method 4: Weighted by variance
    print("\n" + "=" * 70)
    print("METHOD 4: Variance-Weighted R²")
    print("=" * 70)
    
    variances = [np.var(y_val[:, i]) for i in range(len(trainer.target_columns))]
    total_var = sum(variances)
    weights = [v / total_var for v in variances]
    
    r2_weighted = sum(r2 * w for r2, w in zip(r2_per_stat, weights))
    
    for i, stat in enumerate(trainer.target_columns):
        print(f"  {stat}: variance={variances[i]:.2f}, weight={weights[i]:.3f}, R²={r2_per_stat[i]:.4f}")
    
    print(f"\nWeighted R²: {r2_weighted:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Method 1 (Keras - WRONG):     R² = {r2_keras:.4f}")
    print(f"Method 2 (Correct):           R² = {r2_correct:.4f}")
    print(f"Method 3 (Per-stat average):  R² = {r2_avg:.4f}")
    print(f"Method 4 (Variance-weighted): R² = {r2_weighted:.4f}")
    
    print(f"\n🔍 DIAGNOSIS:")
    if abs(r2_keras - 0.666) < 0.05:
        print("✅ Method 1 matches Keras training metric (~0.666)")
        print("   This confirms Keras is using GLOBAL mean (incorrect)")
    
    if abs(r2_correct - 0.041) < 0.05:
        print("✅ Method 2 matches final evaluation (~0.041)")
        print("   This confirms final evaluation uses PER-STAT mean (correct)")
    
    print(f"\n💡 CONCLUSION:")
    print("The discrepancy is caused by different R² calculation methods:")
    print("  - Keras metric uses GLOBAL mean across all stats")
    print("  - Final evaluation uses PER-STAT mean (correct)")
    print("\nThe correct R² is the one from final evaluation.")
    
    # Check if predictions are reasonable
    print("\n" + "=" * 70)
    print("PREDICTION QUALITY CHECK")
    print("=" * 70)
    
    for i, stat in enumerate(trainer.target_columns):
        mae = np.mean(np.abs(y_val[:, i] - pred_means[:, i]))
        rmse = np.sqrt(np.mean((y_val[:, i] - pred_means[:, i]) ** 2))
        
        pred_var = np.var(pred_means[:, i])
        true_var = np.var(y_val[:, i])
        var_ratio = pred_var / true_var if true_var > 0 else 0
        
        print(f"\n{stat}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2_per_stat[i]:.4f}")
        print(f"  Pred variance: {pred_var:.2f}")
        print(f"  True variance: {true_var:.2f}")
        print(f"  Variance ratio: {var_ratio:.3f}")
        
        if var_ratio < 0.2:
            print(f"  ⚠️ WARNING: Predictions have very low variance!")
            print(f"     Model is being too conservative")

if __name__ == "__main__":
    test_r2_calculations()
