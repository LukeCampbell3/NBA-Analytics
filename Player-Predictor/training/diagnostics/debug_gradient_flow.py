#!/usr/bin/env python3
"""
Debug gradient flow in Stage B to understand why some layers aren't getting gradients
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from properly_fixed_trainer import ProperlyFixedTrainer
import warnings
warnings.filterwarnings("ignore")

def debug_gradient_flow():
    """Debug why some layers aren't getting gradients in Stage B"""
    
    print("🔍 DEBUGGING GRADIENT FLOW IN STAGE B")
    print("=" * 60)
    
    # Create trainer and load minimal data
    trainer = ProperlyFixedTrainer()
    
    # Set up minimal mappings for model building
    trainer.player_mapping = {"test": 0}
    trainer.team_mapping = {"test": 0}
    trainer.opponent_mapping = {"test": 0}
    trainer.feature_columns = ["Player_ID_mapped", "Team_ID_mapped", "Opponent_ID_mapped"] + ["feat_" + str(i) for i in range(21)]
    trainer.baseline_features = ["PTS_rolling_avg", "TRB_rolling_avg", "AST_rolling_avg"]
    trainer.regime_stats = {
        "PTS": {"boom_threshold": 8.0, "slump_threshold": -8.0},
        "TRB": {"boom_threshold": 3.0, "slump_threshold": -3.0},
        "AST": {"boom_threshold": 3.0, "slump_threshold": -3.0}
    }
    
    # Build model
    model = trainer._build_full_model()
    
    print(f"Model built with {len(model.layers)} layers")
    
    # Apply Stage B freezing
    print("\n🔧 Applying Stage B freezing logic...")
    frozen_count = 0
    unfrozen_count = 0
    
    for i, layer in enumerate(model.layers):
        if i >= 40:  # After backbone
            if not any(skip in layer.name for skip in ['tf.', 'lambda', 'concatenate', 'add', 'softmax']):
                layer.trainable = True
                unfrozen_count += 1
            else:
                layer.trainable = False
                frozen_count += 1
        else:
            layer.trainable = False
            frozen_count += 1
    
    print(f"Frozen: {frozen_count}, Unfrozen: {unfrozen_count}")
    
    # Create dummy data
    batch_size = 4
    seq_len = 10
    n_features = len(trainer.feature_columns)
    n_baselines = len(trainer.baseline_features)
    
    X_dummy = np.random.randn(batch_size, seq_len, n_features).astype(np.float32)
    baselines_dummy = np.random.randn(batch_size, n_baselines).astype(np.float32)
    y_dummy = np.random.randn(batch_size, 3).astype(np.float32)
    
    print(f"\nDummy data shapes: X={X_dummy.shape}, baselines={baselines_dummy.shape}, y={y_dummy.shape}")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    try:
        outputs = model([X_dummy, baselines_dummy], training=True)
        print(f"✅ Forward pass successful")
        
        if isinstance(outputs, dict):
            predictions = outputs["predictions"]
            gate_probs = outputs.get("gate_probs")
            print(f"Predictions shape: {predictions.shape}")
            if gate_probs is not None:
                print(f"Gate probs shape: {gate_probs.shape}")
        else:
            predictions = outputs
            gate_probs = None
            print(f"Predictions shape: {predictions.shape}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Test gradient computation
    print("\n🎯 Testing gradient computation...")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    with tf.GradientTape() as tape:
        outputs = model([X_dummy, baselines_dummy], training=True)
        
        if isinstance(outputs, dict):
            predictions = outputs["predictions"]
            gate_probs = outputs.get("gate_probs")
        else:
            predictions = outputs
            gate_probs = None
        
        # Compute Stage B loss (gate supervision only)
        if gate_probs is not None:
            loss = trainer._compute_fixed_stable_gate_loss(y_dummy, predictions, baselines_dummy, gate_probs)
        else:
            # Fallback to simple MSE
            loss = tf.reduce_mean(tf.square(y_dummy - predictions[:, :3]))
    
    print(f"Loss computed: {loss.numpy():.4f}")
    
    # Get gradients
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    
    print(f"Trainable variables: {len(trainable_vars)}")
    print(f"Gradients computed: {len([g for g in gradients if g is not None])}")
    print(f"None gradients: {len([g for g in gradients if g is None])}")
    
    # Analyze which layers have gradients
    print("\n📊 Gradient Analysis:")
    print("-" * 40)
    
    layers_with_gradients = []
    layers_without_gradients = []
    
    for var, grad in zip(trainable_vars, gradients):
        layer_name = var.name.split('/')[0]  # Extract layer name
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            layers_with_gradients.append((layer_name, grad_norm))
        else:
            layers_without_gradients.append(layer_name)
    
    print("Layers WITH gradients:")
    for layer_name, grad_norm in layers_with_gradients:
        print(f"  ✅ {layer_name}: grad_norm={grad_norm:.6f}")
    
    print(f"\nLayers WITHOUT gradients ({len(layers_without_gradients)}):")
    for layer_name in set(layers_without_gradients):  # Remove duplicates
        print(f"  ❌ {layer_name}")
    
    # Check if the problematic layers are in the computation graph
    print("\n🔍 Analyzing computation graph connectivity...")
    
    # Get the layers that should be trainable in Stage B
    stage_b_layers = []
    for i, layer in enumerate(model.layers):
        if i >= 40 and layer.trainable:
            stage_b_layers.append(layer.name)
    
    print(f"Stage B trainable layers: {len(stage_b_layers)}")
    
    # Check which Stage B layers are getting gradients
    stage_b_with_grads = set([name for name, _ in layers_with_gradients])
    stage_b_without_grads = set(layers_without_gradients)
    
    connected_stage_b = [name for name in stage_b_layers if any(name in grad_layer for grad_layer in stage_b_with_grads)]
    disconnected_stage_b = [name for name in stage_b_layers if any(name in no_grad_layer for no_grad_layer in stage_b_without_grads)]
    
    print(f"Stage B layers WITH gradients: {len(connected_stage_b)}")
    for name in connected_stage_b[:5]:  # Show first 5
        print(f"  ✅ {name}")
    
    print(f"Stage B layers WITHOUT gradients: {len(disconnected_stage_b)}")
    for name in disconnected_stage_b[:5]:  # Show first 5
        print(f"  ❌ {name}")
    
    # DIAGNOSIS
    print("\n" + "="*60)
    print("🔬 DIAGNOSIS")
    print("="*60)
    
    if len(layers_without_gradients) > len(layers_with_gradients):
        print("❌ MAJOR ISSUE: More layers without gradients than with gradients")
        print("   This suggests the loss function isn't connected to most trainable layers")
        print("   Possible causes:")
        print("   1. Gate probabilities not being used in loss computation")
        print("   2. Expert outputs not contributing to final predictions")
        print("   3. Computational graph disconnected")
    elif len(disconnected_stage_b) > 0:
        print("⚠️  PARTIAL ISSUE: Some Stage B layers not getting gradients")
        print(f"   {len(disconnected_stage_b)} Stage B layers disconnected from loss")
        print("   This explains the gradient warnings during training")
    else:
        print("✅ GRADIENT FLOW LOOKS GOOD")
        print("   All trainable layers are getting gradients")
    
    # Check if gate probabilities are actually being used
    if gate_probs is not None:
        print(f"\n🎯 Gate probabilities found in model output")
        print(f"   Shape: {gate_probs.shape}")
        print(f"   This suggests gates should be trainable")
    else:
        print(f"\n❌ No gate probabilities in model output")
        print(f"   This suggests the gate computation isn't working")
    
    return {
        'total_layers': len(model.layers),
        'trainable_vars': len(trainable_vars),
        'layers_with_gradients': len(layers_with_gradients),
        'layers_without_gradients': len(layers_without_gradients),
        'stage_b_connected': len(connected_stage_b),
        'stage_b_disconnected': len(disconnected_stage_b)
    }

if __name__ == "__main__":
    results = debug_gradient_flow()
    
    print(f"\n📋 SUMMARY:")
    print(f"  Total layers: {results['total_layers']}")
    print(f"  Trainable variables: {results['trainable_vars']}")
    print(f"  Variables with gradients: {results['layers_with_gradients']}")
    print(f"  Variables without gradients: {results['layers_without_gradients']}")
    print(f"  Stage B layers connected: {results['stage_b_connected']}")
    print(f"  Stage B layers disconnected: {results['stage_b_disconnected']}")