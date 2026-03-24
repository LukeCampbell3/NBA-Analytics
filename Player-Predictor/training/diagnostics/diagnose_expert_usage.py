#!/usr/bin/env python3
"""
Quick diagnostic to check current expert usage distribution
"""

import numpy as np
import tensorflow as tf
from hybrid_spike_moe_trainer import HybridSpikeMoETrainer
import warnings
warnings.filterwarnings("ignore")

def diagnose_current_expert_usage():
    """Diagnose current expert usage patterns"""
    
    print("🔍 Diagnosing Current Expert Usage Distribution")
    print("=" * 60)
    
    # Create trainer
    trainer = HybridSpikeMoETrainer(ensemble_size=1)
    
    # Load and prepare small sample of data
    print("📊 Loading sample data...")
    X, baselines, y, df = trainer.prepare_data()
    
    # Use small sample for quick diagnosis
    sample_size = min(1000, len(X))
    X_sample = X[:sample_size]
    baselines_sample = baselines[:sample_size]
    y_sample = y[:sample_size]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    
    X_flat = X_sample.reshape(-1, X_sample.shape[-1])
    X_numeric = X_flat[:, 3:]
    
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_scaled = np.concatenate([X_flat[:, :3], X_numeric_scaled], axis=1)
    X_scaled = X_scaled.reshape(X_sample.shape)
    
    # Build model
    print("🏗️ Building model for diagnosis...")
    model = trainer.build_model()
    
    # Compile with dummy loss for prediction
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    
    print(f"📈 Model Architecture:")
    print(f"  Total Experts: {trainer.config['num_experts'] + trainer.config['num_spike_experts']}")
    print(f"  Regular Experts: {trainer.config['num_experts']}")
    print(f"  Spike Experts: {trainer.config['num_spike_experts']}")
    print(f"  Router Temperature: {trainer.config['router_temperature']}")
    print(f"  Load Balance Weight: {trainer.config['load_balance_weight']}")
    
    # Get predictions to see routing behavior
    print(f"\n🎯 Analyzing routing behavior on {sample_size} samples...")
    
    try:
        # Make predictions to trigger routing
        predictions = model.predict([X_scaled, baselines_sample], verbose=0)
        print("✅ Model prediction successful")
        
        # Try to extract routing information if available
        # Note: This is a simplified analysis since we can't directly access router probs
        print(f"\n📊 Prediction Analysis:")
        print(f"  Prediction shape: {predictions.shape}")
        print(f"  Sample predictions (first 5):")
        for i in range(min(5, len(predictions))):
            if trainer.config["use_probabilistic"]:
                means = predictions[i, :len(trainer.target_columns)]
                print(f"    Sample {i+1}: PTS={means[0]:.1f}, TRB={means[1]:.1f}, AST={means[2]:.1f}")
            else:
                print(f"    Sample {i+1}: PTS={predictions[i,0]:.1f}, TRB={predictions[i,1]:.1f}, AST={predictions[i,2]:.1f}")
        
        # Check prediction variance as indicator of expert diversity
        if trainer.config["use_probabilistic"]:
            pred_means = predictions[:, :len(trainer.target_columns)]
        else:
            pred_means = predictions
        
        pred_vars = np.var(pred_means, axis=0)
        true_vars = np.var(y_sample, axis=0)
        
        print(f"\n📈 Variance Analysis (indicator of expert diversity):")
        stats = ["PTS", "TRB", "AST"]
        for i, stat in enumerate(stats):
            var_ratio = pred_vars[i] / true_vars[i] if true_vars[i] > 0 else 0
            print(f"  {stat}: Pred Var={pred_vars[i]:.2f}, True Var={true_vars[i]:.2f}, Ratio={var_ratio:.3f}")
            
            if var_ratio < 0.2:
                print(f"    ❌ Very low variance - likely expert concentration")
            elif var_ratio < 0.4:
                print(f"    ⚠️ Low variance - possible expert concentration")
            elif var_ratio > 0.6:
                print(f"    ✅ Good variance - likely diverse expert usage")
            else:
                print(f"    ✅ Moderate variance - reasonable expert usage")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("This might indicate model architecture issues")
    
    # Analyze current configuration for expert diversity
    print(f"\n🔧 Current Configuration Analysis:")
    
    config_issues = []
    config_good = []
    
    # Check router temperature
    temp = trainer.config["router_temperature"]
    if temp < 1.5:
        config_issues.append(f"Router temperature too low ({temp}) - increases expert concentration")
    else:
        config_good.append(f"Router temperature good ({temp}) - promotes diversity")
    
    # Check load balance weight
    lb_weight = trainer.config["load_balance_weight"]
    if lb_weight < 0.05:
        config_issues.append(f"Load balance weight too low ({lb_weight}) - allows expert concentration")
    else:
        config_good.append(f"Load balance weight good ({lb_weight}) - promotes balance")
    
    # Check expert utilization balancing
    if not trainer.config.get("expert_utilization_balancing", False):
        config_issues.append("Expert utilization balancing disabled - allows concentration")
    else:
        config_good.append("Expert utilization balancing enabled - promotes diversity")
    
    # Check entropy target
    entropy_target = trainer.config.get("entropy_target", 1.0)
    max_entropy = np.log(trainer.config['num_experts'] + trainer.config['num_spike_experts'])
    if entropy_target < max_entropy * 0.7:
        config_issues.append(f"Entropy target too low ({entropy_target:.2f} vs max {max_entropy:.2f}) - allows concentration")
    else:
        config_good.append(f"Entropy target good ({entropy_target:.2f}) - promotes diversity")
    
    print(f"\n✅ Good Configuration Settings:")
    for item in config_good:
        print(f"  {item}")
    
    print(f"\n⚠️ Configuration Issues for Expert Diversity:")
    for item in config_issues:
        print(f"  {item}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR EXPERT DIVERSITY:")
    print("=" * 50)
    
    if config_issues:
        print("🔧 Configuration Fixes Needed:")
        if temp < 1.5:
            print(f"  1. Increase router_temperature: {temp} → 2.0+ (higher = more diverse)")
        if lb_weight < 0.05:
            print(f"  2. Increase load_balance_weight: {lb_weight} → 0.08+ (stronger balancing)")
        if not trainer.config.get("expert_utilization_balancing", False):
            print(f"  3. Enable expert_utilization_balancing: False → True")
        if entropy_target < max_entropy * 0.7:
            print(f"  4. Increase entropy_target: {entropy_target:.2f} → {max_entropy*0.8:.2f}+")
        
        print(f"\n🚀 Quick Fix Command:")
        print(f"Run: python expert_diversity_fixes.py")
        print(f"This will train with optimized settings for expert diversity")
        
    else:
        print("✅ Configuration looks good for expert diversity!")
        print("If still seeing concentration, consider:")
        print("  1. Further increasing router temperature")
        print("  2. Adding explicit expert diversity loss")
        print("  3. Using expert rotation during training")
    
    print(f"\n📋 Summary:")
    print(f"  Current router temperature: {temp}")
    print(f"  Current load balance weight: {lb_weight}")
    print(f"  Expert utilization balancing: {trainer.config.get('expert_utilization_balancing', False)}")
    print(f"  Total experts available: {trainer.config['num_experts'] + trainer.config['num_spike_experts']}")
    
    if len(config_issues) == 0:
        print(f"  Status: ✅ Configuration optimized for diversity")
    elif len(config_issues) <= 2:
        print(f"  Status: ⚠️ Minor fixes needed for better diversity")
    else:
        print(f"  Status: ❌ Major fixes needed for expert diversity")

if __name__ == "__main__":
    diagnose_current_expert_usage()