#!/usr/bin/env python3
"""Simple test for enhanced MoE - no unicode issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
import numpy as np
from enhanced_moe_trainer import EnhancedMoELayer

print("Testing EnhancedMoELayer...")

# Create layer
layer = EnhancedMoELayer(
    num_experts=8,
    num_spike_experts=3,
    expert_dim=128,
    spike_expert_capacity=256,
    output_dim=6,  # 3 stats * 2 (mean + scale)
    use_probabilistic=True,
    use_load_balancing=True,
    use_prototype_experts=True,
    use_diversity_regularization=True
)

print(f"Layer created: {layer.total_experts} experts")

# Test forward pass
batch_size = 32
repr_dim = 259  # 256 + 3 baseline features
sequence_repr = tf.random.normal([batch_size, repr_dim])
spike_indicators = tf.random.uniform([batch_size, 3], 0, 1)

output = layer([sequence_repr, spike_indicators], training=True)

print(f"Input shape: {sequence_repr.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: ({batch_size}, 6)")

assert output.shape == (batch_size, 6), f"Wrong shape: {output.shape}"
print("SUCCESS: Output shape correct!")

# Check losses
print(f"Losses added: {len(layer.losses)}")
print(f"Metrics added: {len(layer.metrics)}")

print("\nAll tests passed!")
