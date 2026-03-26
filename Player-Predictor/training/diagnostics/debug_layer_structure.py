#!/usr/bin/env python3
"""
Debug script to understand the model structure and identify gate/expert layers
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from enhanced_two_stage_trainer import EnhancedTwoStageTrainer

def analyze_model_structure():
    """Analyze the model structure to identify gate and expert layers"""
    
    print("🔍 Analyzing Model Structure")
    print("=" * 50)
    
    # Create a minimal trainer to build the model
    trainer = EnhancedTwoStageTrainer()
    
    # Set up minimal mappings for model building
    trainer.player_mapping = {"test": 0}
    trainer.team_mapping = {"test": 0}
    trainer.opponent_mapping = {"test": 0}
    trainer.feature_columns = ["Player_ID_mapped", "Team_ID_mapped", "Opponent_ID_mapped"] + ["feat_" + str(i) for i in range(21)]
    trainer.baseline_features = ["PTS_rolling_avg", "TRB_rolling_avg", "AST_rolling_avg"]
    
    # Build the full model
    print("Building model to analyze structure...")
    model = trainer._build_full_model()
    
    print(f"\nTotal layers: {len(model.layers)}")
    
    # Analyze the structure by looking at layer connections and shapes
    print("\n📋 Key Layers Analysis:")
    print("-" * 30)
    
    # Look for patterns in the layer sequence
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        
        # Check if this is likely a gate or expert layer based on position and connections
        if "dense" in layer_name:
            try:
                output_shape = layer.output_shape if hasattr(layer, 'output_shape') else "unknown"
                print(f"{i:2d}: {layer_name} - shape: {output_shape}")
                
                # Analyze based on output shape
                if hasattr(layer, 'output_shape') and layer.output_shape:
                    if isinstance(layer.output_shape, tuple) and len(layer.output_shape) > 1:
                        last_dim = layer.output_shape[-1]
                        
                        # Gate layer should output 9 (3 stats * 3 regimes)
                        if last_dim == 9:
                            print(f"    🎯 LIKELY GATE LAYER: {layer_name} (outputs 9 = 3 stats * 3 regimes)")
                        
                        # Expert layers should output 18 (3 stats * 6 components)
                        elif last_dim == 18:
                            print(f"    🤖 LIKELY EXPERT LAYER: {layer_name} (outputs 18 = 3 stats * 6 components)")
                        
                        # Router layer should output 10 (8 regular + 2 spike experts)
                        elif last_dim == 10:
                            print(f"    🔀 LIKELY ROUTER LAYER: {layer_name} (outputs 10 = 8 + 2 experts)")
                        
                        # Spike detection layers
                        elif last_dim == 32:
                            print(f"    🔍 LIKELY SPIKE DETECTION: {layer_name} (outputs 32)")
                        
                        # Other sizes
                        else:
                            print(f"    📊 Other: {layer_name} (outputs {last_dim})")
                            
            except Exception as e:
                print(f"{i:2d}: {layer_name} - error: {e}")
        
        elif layer_name in ["gru", "conditional_spike_output"]:
            print(f"{i:2d}: {layer_name} - IMPORTANT LAYER")
    
    print("\n🔧 Based on Analysis, FIXED Unfreezing Should Target:")
    print("-" * 55)
    
    # Based on the structure, identify the layers that should be unfrozen
    # We need to unfreeze layers after the backbone (around layer 40+)
    
    trainable_layer_indices = []
    trainable_layer_names = []
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        
        # Unfreeze layers after the backbone (global_average_pooling1d is at index 40)
        if i >= 40:  # After backbone
            # Skip certain TensorFlow ops that aren't trainable
            if not any(skip in layer_name for skip in ["tf.", "lambda", "concatenate", "add", "softmax"]):
                trainable_layer_indices.append(i)
                trainable_layer_names.append(layer_name)
    
    print("Layers to unfreeze (by index):")
    for idx in trainable_layer_indices:
        layer = model.layers[idx]
        print(f"  {idx:2d}: {layer.name}")
    
    print(f"\nTotal layers to unfreeze: {len(trainable_layer_indices)}")
    
    # Generate the correct unfreezing logic
    print("\n🎯 CORRECT Unfreezing Logic:")
    print("-" * 30)
    print("# Unfreeze all layers after the backbone (index >= 40)")
    print("for i, layer in enumerate(model.layers):")
    print("    if i >= 40:")  # After global_average_pooling1d
    print("        if not any(skip in layer.name for skip in ['tf.', 'lambda', 'concatenate', 'add', 'softmax']):")
    print("            layer.trainable = True")
    print("        else:")
    print("            layer.trainable = False")
    print("    else:")
    print("        layer.trainable = False")

if __name__ == "__main__":
    analyze_model_structure()