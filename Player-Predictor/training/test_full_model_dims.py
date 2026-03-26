#!/usr/bin/env python3
"""Test full model dimensions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_moe_trainer import EnhancedMoETrainer

print("Creating trainer...")
trainer = EnhancedMoETrainer()

print("Preparing data...")
X, baselines, y, df = trainer.prepare_data()

print(f"Data shapes:")
print(f"  X: {X.shape}")
print(f"  baselines: {baselines.shape}")
print(f"  y: {y.shape}")

print("\nBuilding model...")
model = trainer.build_model()

print(f"\nModel built!")
print(f"Model name: {model.name}")
print(f"Input shapes: {[inp.shape for inp in model.inputs]}")
print(f"Output shape: {model.output.shape}")

# Test forward pass
import numpy as np
batch_size = 32
X_test = X[:batch_size]
baselines_test = baselines[:batch_size]

print(f"\nTesting forward pass...")
print(f"  X_test shape: {X_test.shape}")
print(f"  baselines_test shape: {baselines_test.shape}")

output = model.predict([X_test, baselines_test], verbose=0)
print(f"  Output shape: {output.shape}")
print(f"  Expected: ({batch_size}, 6) for probabilistic")

if output.shape[1] == 6:
    print("\nSUCCESS: Model outputs correct dimensions!")
else:
    print(f"\nERROR: Model outputs {output.shape[1]} dimensions, expected 6")
    print("This will cause the loss function to fail.")
