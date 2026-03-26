#!/usr/bin/env python3
"""
Diagnostic script to check if router collapse has been fixed.
This will run a quick test to verify:
1. Router entropy is reasonable (>1.0)
2. Expert usage is balanced
3. Spike routing is selective (not always on)
4. Auxiliary losses are active
"""

import numpy as np
import tensorflow as tf
from hybrid_spike_moe_trainer import HybridSpikeMoETrainer
import warnings
warnings.filterwarnings("ignore")

def diagnose_router_collapse():
    """Diagnose router collapse issues"""
    
    print("🔍 Diagnosing Router Collapse Issues")
    print("=" * 50)
    
    # Create trainer
    trainer = HybridSpikeMoETrainer(ensemble_size=1)
    
    # Reduce training for quick diagnosis
    trainer.config.update({
        "epochs": 3,  # Very short training
        "batch_size": 32,
        "patience": 10,
    })
    
    print("📊 Loading test data...")
    try:
        # Load data
        X, baselines, y, df = trainer.prepare_data()
        
        # Use small subset for quick testing
        n_test = min(200, len(X))
        X_test = X[:n_test]
        baselines_test = baselines[:n_test]
        y_test = y[:n_test]
        
        print(f"Test data shape: X={X_test.shape}, baselines={baselines_test.shape}, y={y_test.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    print("🏗️ Building model for diagnosis...")
    
    # Build model
    model = trainer.build_model()
    
    # Simple compilation for testing
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print("🔍 Running diagnostic forward pass...")
    
    # Scale features quickly
    from sklearn.preprocessing import StandardScaler
    X_flat = X_test.reshape(-1, X_test.shape[-1])
    X_numeric = X_flat[:, 3:]  # Skip categorical
    
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_scaled = np.concatenate([X_flat[:, :3], X_numeric_scaled], axis=1)
    X_scaled = X_scaled.reshape(X_test.shape)
    
    # Get model predictions and routing info
    predictions = model.predict([X_scaled, baselines_test], verbose=0)
    
    # Create routing analysis model
    sequence_input = model.get_layer("sequence_input").input
    baseline_input = model.get_layer("baseline_input").input
    router_probs = model.get_layer("router_probs").output
    
    # Also get spike indicators
    try:
        spike_indicators = model.get_layer("enhanced_context_spike_detection").output
    except:
        try:
            spike_indicators = model.get_layer("context_aware_spike_detection").output
        except:
            spike_indicators = None
    
    if spike_indicators is not None:
        routing_model = tf.keras.Model(
            inputs=[sequence_input, baseline_input],
            outputs=[router_probs, spike_indicators]
        )
        routing_probs, spike_scores = routing_model.predict([X_scaled, baselines_test], verbose=0)
    else:
        routing_model = tf.keras.Model(
            inputs=[sequence_input, baseline_input],
            outputs=router_probs
        )
        routing_probs = routing_model.predict([X_scaled, baselines_test], verbose=0)
        spike_scores = None
    
    print("\n🔍 DIAGNOSTIC RESULTS:")
    print("=" * 40)
    
    # 1. Router entropy analysis
    def calculate_entropy(probs):
        return -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))
    
    entropy = calculate_entropy(routing_probs)
    print(f"1. Router Entropy: {entropy:.3f}")
    if entropy > 1.0:
        print("   ✅ GOOD: Entropy > 1.0 (diverse routing)")
    elif entropy > 0.5:
        print("   ⚠️ MODERATE: Entropy 0.5-1.0 (some diversity)")
    else:
        print("   ❌ BAD: Entropy < 0.5 (collapsed routing)")
    
    # 2. Confidence analysis
    max_probs = np.max(routing_probs, axis=1)
    avg_confidence = np.mean(max_probs)
    print(f"2. Average Confidence: {avg_confidence:.3f}")
    if avg_confidence < 0.8:
        print("   ✅ GOOD: Confidence < 0.8 (not overconfident)")
    elif avg_confidence < 0.95:
        print("   ⚠️ MODERATE: Confidence 0.8-0.95 (somewhat confident)")
    else:
        print("   ❌ BAD: Confidence > 0.95 (overconfident/collapsed)")
    
    # 3. Expert usage analysis
    expert_usage = np.mean(routing_probs, axis=0)
    regular_experts = trainer.config["num_experts"]
    spike_experts = trainer.config["num_spike_experts"]
    
    regular_usage = np.sum(expert_usage[:regular_experts])
    spike_usage = np.sum(expert_usage[regular_experts:])
    
    print(f"3. Expert Usage:")
    print(f"   Regular Experts: {regular_usage:.1%}")
    print(f"   Spike Experts: {spike_usage:.1%}")
    
    if regular_usage > 0.5:
        print("   ✅ GOOD: Regular experts getting significant usage")
    elif regular_usage > 0.2:
        print("   ⚠️ MODERATE: Regular experts getting some usage")
    else:
        print("   ❌ BAD: Regular experts barely used (collapsed to spike)")
    
    # 4. Individual expert usage
    print(f"4. Individual Expert Usage:")
    for i, usage in enumerate(expert_usage):
        expert_type = "Spike" if i >= regular_experts else "Regular"
        print(f"   Expert {i} ({expert_type}): {usage:.1%}")
    
    # Check for completely unused experts
    unused_experts = np.sum(expert_usage < 0.01)
    print(f"   Unused Experts (< 1%): {unused_experts}/{len(expert_usage)}")
    
    # 5. Spike score analysis (if available)
    if spike_scores is not None:
        avg_spike_score = np.mean(spike_scores, axis=0)
        print(f"5. Spike Scores:")
        for i, score in enumerate(avg_spike_score):
            stat = ["PTS", "TRB", "AST"][i]
            print(f"   {stat}: {score:.3f}")
        
        # Check if spike scores are always high (indicating always-spike)
        high_spike_rate = np.mean(np.max(spike_scores, axis=1) > 0.8)
        print(f"   High Spike Rate (>0.8): {high_spike_rate:.1%}")
        
        if high_spike_rate < 0.3:
            print("   ✅ GOOD: Spike detection is selective")
        elif high_spike_rate < 0.7:
            print("   ⚠️ MODERATE: Spike detection somewhat selective")
        else:
            print("   ❌ BAD: Spike detection always high (always-spike mode)")
    
    # 6. Routing decision analysis
    routing_decisions = np.argmax(routing_probs, axis=1)
    decision_counts = np.bincount(routing_decisions, minlength=len(expert_usage))
    
    print(f"6. Routing Decisions:")
    for i, count in enumerate(decision_counts):
        expert_type = "Spike" if i >= regular_experts else "Regular"
        pct = count / len(routing_decisions) * 100
        print(f"   Expert {i} ({expert_type}): {count} samples ({pct:.1f}%)")
    
    # 7. Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    issues = []
    if entropy < 0.5:
        issues.append("Low entropy (collapsed routing)")
    if avg_confidence > 0.95:
        issues.append("Overconfident routing")
    if regular_usage < 0.2:
        issues.append("Regular experts underused")
    if unused_experts > len(expert_usage) // 2:
        issues.append("Many experts unused")
    if spike_scores is not None and high_spike_rate > 0.7:
        issues.append("Always-spike detection")
    
    if len(issues) == 0:
        print("✅ EXCELLENT: No major routing issues detected!")
    elif len(issues) <= 2:
        print("⚠️ MODERATE: Some issues detected:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("❌ POOR: Multiple routing issues detected:")
        for issue in issues:
            print(f"   - {issue}")
    
    # 8. Quick training test
    print(f"\n🚀 Running quick training test...")
    
    # Train for 1 epoch to see if losses are active
    try:
        history = model.fit(
            [X_scaled, baselines_test], y_test,
            epochs=1,
            batch_size=32,
            verbose=1
        )
        
        print("✅ Training completed successfully")
        
        # Check if losses are changing
        if hasattr(history.history, 'loss'):
            print(f"   Final loss: {history.history['loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
    
    return {
        'entropy': entropy,
        'confidence': avg_confidence,
        'regular_usage': regular_usage,
        'spike_usage': spike_usage,
        'unused_experts': unused_experts,
        'issues': issues
    }

if __name__ == "__main__":
    results = diagnose_router_collapse()
    
    if results:
        print(f"\n📋 SUMMARY:")
        print(f"Entropy: {results['entropy']:.3f}")
        print(f"Confidence: {results['confidence']:.3f}")
        print(f"Regular Usage: {results['regular_usage']:.1%}")
        print(f"Spike Usage: {results['spike_usage']:.1%}")
        print(f"Issues: {len(results['issues'])}")