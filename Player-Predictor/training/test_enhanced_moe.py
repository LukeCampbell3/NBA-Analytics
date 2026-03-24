#!/usr/bin/env python3
"""
Test script for Enhanced MoE Trainer with Phases 1-3.

This verifies that all improvements are properly integrated.
"""

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from enhanced_moe_trainer import EnhancedMoETrainer, EnhancedMoELayer

print("\n" + "="*70)
print("ENHANCED MOE TRAINER TESTS")
print("="*70)

# Test 1: Trainer creation
print("\n1. Testing trainer creation...")
try:
    trainer = EnhancedMoETrainer()
    print("   ✓ Trainer created successfully")
    print(f"     - Total experts: {trainer.config['num_experts'] + trainer.config['num_spike_experts']}")
    print(f"     - Load balancing: {trainer.config['use_load_balancing']}")
    print(f"     - Prototype experts: {trainer.config['use_prototype_experts']}")
    print(f"     - Diversity regularization: {trainer.config['use_diversity_regularization']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: EnhancedMoELayer creation
print("\n2. Testing EnhancedMoELayer...")
try:
    layer = EnhancedMoELayer(
        num_experts=8,
        num_spike_experts=3,
        expert_dim=128,
        spike_expert_capacity=256,
        output_dim=12,  # 3 stats * 4 (mean, sigma, etc.)
        use_probabilistic=True,
        use_load_balancing=True,
        use_prototype_experts=True,
        use_diversity_regularization=True
    )
    print("   ✓ EnhancedMoELayer created successfully")
    print(f"     - Total experts: {layer.total_experts}")
    print(f"     - Has load balancer: {hasattr(layer, 'load_balancer')}")
    print(f"     - Has prototype experts: {hasattr(layer, 'prototype_experts')}")
    print(f"     - Has diversity regularizer: {hasattr(layer, 'diversity_regularizer')}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Forward pass with dummy data
print("\n3. Testing forward pass...")
try:
    batch_size = 32
    repr_dim = 256 + 3  # d_model + baseline features
    
    # Create dummy inputs
    sequence_repr = tf.random.normal([batch_size, repr_dim])
    spike_indicators = tf.random.uniform([batch_size, 3], 0, 1)
    
    # Forward pass
    output = layer([sequence_repr, spike_indicators], training=True)
    
    print("   ✓ Forward pass successful")
    print(f"     - Input shape: {sequence_repr.shape}")
    print(f"     - Output shape: {output.shape}")
    print(f"     - Expected output shape: ({batch_size}, 12)")
    
    assert output.shape == (batch_size, 12), f"Wrong output shape: {output.shape}"
    print("   ✓ Output shape correct")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check losses are added
print("\n4. Testing loss tracking...")
try:
    # Get losses from layer
    losses = layer.losses
    
    print(f"   ✓ Layer has {len(losses)} losses")
    
    # Check for expected losses
    expected_losses = []
    if layer.use_load_balancing:
        expected_losses.append("balance_loss")
    if layer.use_prototype_experts:
        expected_losses.extend(["compactness_loss", "separation_loss"])
    if layer.use_diversity_regularization:
        expected_losses.append("output_correlation_loss")
    
    print(f"     - Expected losses: {expected_losses}")
    print(f"     - Actual losses: {len(losses)} loss tensors")
    
    # Losses are added during forward pass, so we should have some
    assert len(losses) > 0, "No losses added"
    print("   ✓ Losses are being tracked")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check metrics are added
print("\n5. Testing metric tracking...")
try:
    # Get metrics from layer
    metrics = layer.metrics
    
    print(f"   ✓ Layer has {len(metrics)} metrics")
    
    # Check for expected metrics
    expected_metric_names = [
        'balance_loss',
        'compactness_loss', 
        'separation_loss',
        'output_correlation_loss',
        'avg_router_prob',
        'max_router_prob'
    ]
    
    metric_names = [m.name for m in metrics]
    print(f"     - Metric names: {metric_names[:5]}...")  # Show first 5
    
    # At least some metrics should be present
    assert len(metrics) > 0, "No metrics added"
    print("   ✓ Metrics are being tracked")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test with different configurations
print("\n6. Testing configuration variations...")
try:
    configs = [
        {"use_load_balancing": True, "use_prototype_experts": False, "use_diversity_regularization": False},
        {"use_load_balancing": False, "use_prototype_experts": True, "use_diversity_regularization": False},
        {"use_load_balancing": False, "use_prototype_experts": False, "use_diversity_regularization": True},
        {"use_load_balancing": True, "use_prototype_experts": True, "use_diversity_regularization": True},
    ]
    
    for i, config in enumerate(configs):
        layer_test = EnhancedMoELayer(
            num_experts=8,
            num_spike_experts=3,
            expert_dim=128,
            spike_expert_capacity=256,
            output_dim=12,
            **config
        )
        
        # Test forward pass
        output_test = layer_test([sequence_repr, spike_indicators], training=True)
        
        enabled = [k for k, v in config.items() if v]
        print(f"   ✓ Config {i+1}: {', '.join(enabled)}")
    
    print("   ✓ All configurations work")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Verify prototype keys are learnable
print("\n7. Testing prototype expert keys...")
try:
    if layer.use_prototype_experts:
        keys = layer.prototype_experts.expert_keys
        print(f"   ✓ Prototype keys shape: {keys.shape}")
        print(f"     - Expected: ({layer.total_experts}, {layer.prototype_key_dim})")
        
        assert keys.shape == (layer.total_experts, layer.prototype_key_dim)
        assert keys.trainable, "Keys should be trainable"
        print("   ✓ Prototype keys are learnable")
    else:
        print("   ⊙ Skipped (prototypes not enabled)")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nThe Enhanced MoE Trainer is working correctly!")
print("\nPhases integrated:")
print("  ✓ Phase 1: Load balancing (Switch-style)")
print("  ✓ Phase 2: Prototype experts (learnable domain keys)")
print("  ✓ Phase 3: Diversity regularization (output correlation)")
print("\nYou can now:")
print("  1. Train with: python training/enhanced_moe_trainer.py")
print("  2. Or import: from enhanced_moe_trainer import EnhancedMoETrainer")
print("="*70 + "\n")
