#!/usr/bin/env python3
"""
Quick diagnostic to check if the fixes are working
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

def analyze_training_results():
    """Analyze the training results to see if fixes are working"""
    
    print("🔍 Quick Diagnostic of Training Results")
    print("=" * 50)
    
    # Check if Stage B actually learned anything
    print("\n📊 Stage Analysis:")
    print("Stage A (Normal head): Final val_mae = 4.1020")
    print("Stage B (Gate training): Final val_loss = 2.8452")
    print("  - Started at 2.8774, ended at 2.8452")
    print("  - Loss decreased by 0.0322 (1.1%)")
    print("  - Early stopping after 7 epochs")
    print("  - ⚠️ Only 1 layer unfrozen - this is still wrong!")
    
    print("\nStage C (Joint training): Final val_loss = 24.1658")
    print("  - Started at 25.4345, ended at 24.1658")
    print("  - Loss decreased by 1.2687 (5.0%)")
    print("  - Completed 30 epochs")
    
    print("\n🎯 Key Observations:")
    print("✅ Stage B loss is now changing (was flatlined before)")
    print("❌ Stage B still only unfroze 1 layer (conditional_spike_output)")
    print("❌ Gate computation layers (gate_logits_dense, etc.) still frozen")
    print("❌ This means gates still can't learn properly")
    
    print("\n🚨 Critical Issue Identified:")
    print("The layer naming in the model doesn't match the unfreezing keywords!")
    print("Need to check actual layer names and fix the unfreezing logic.")
    
    # Check what the actual layer names are
    print("\n📋 Next Steps:")
    print("1. Fix the unfreezing logic to target correct layer names")
    print("2. Ensure gate computation layers are actually trainable in Stage B")
    print("3. Use stable regime thresholds instead of batch-dependent z-scores")
    print("4. Remove component ordering penalty (output layer handles it)")
    print("5. Use gate probabilities directly as mixture weights")

def check_model_architecture():
    """Check the actual layer names in the model"""
    
    print("\n🔍 Checking Model Architecture...")
    
    try:
        # Load metadata to see what was saved
        import json
        with open("model/conditional_spike_metadata.json", "r") as f:
            metadata = json.load(f)
        
        print(f"Model type: {metadata.get('model_type', 'unknown')}")
        print(f"Architecture version: {metadata.get('architecture_version', 'unknown')}")
        print(f"Use probabilistic: {metadata.get('use_probabilistic', 'unknown')}")
        print(f"Mixture components: {metadata.get('mixture_components', 'unknown')}")
        
        # The issue is that we need to actually look at the layer names
        # in the built model to fix the unfreezing logic
        
    except Exception as e:
        print(f"Could not load metadata: {e}")

def main():
    """Run quick diagnostic"""
    
    analyze_training_results()
    check_model_architecture()
    
    print("\n🎯 Summary:")
    print("The fixes are partially working, but Stage B unfreezing is still broken.")
    print("Need to fix the layer name matching to actually unfreeze gate layers.")
    print("Once that's fixed, we should see:")
    print("  ✓ Slump predictions (not all zeros)")
    print("  ✓ Better boom discrimination")
    print("  ✓ Higher variance ratios (not ~0.05)")
    print("  ✓ Positive σ-error correlation")

if __name__ == "__main__":
    main()