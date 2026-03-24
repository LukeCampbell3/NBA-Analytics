#!/usr/bin/env python3
"""
Debug script to check actual layer names in the model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from enhanced_two_stage_trainer import EnhancedTwoStageTrainer

def check_actual_layer_names():
    """Build the model and check actual layer names"""
    
    print("🔍 Checking Actual Layer Names in Model")
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
    print("Building model to inspect layer names...")
    model = trainer._build_full_model()
    
    print(f"\nTotal layers: {len(model.layers)}")
    print("\n📋 All Layer Names:")
    print("-" * 30)
    
    gate_related = []
    spike_related = []
    router_related = []
    expert_related = []
    backbone_related = []
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.name.lower()
        print(f"{i:2d}: {layer.name}")
        
        # Categorize layers
        if "gate" in layer_name:
            gate_related.append(layer.name)
        elif "spike" in layer_name:
            spike_related.append(layer.name)
        elif "router" in layer_name:
            router_related.append(layer.name)
        elif "expert" in layer_name:
            expert_related.append(layer.name)
        elif any(keyword in layer_name for keyword in ["embed", "attention", "norm", "add", "ff", "projection", "pooling"]):
            backbone_related.append(layer.name)
    
    print("\n🎯 Layer Categories:")
    print("-" * 20)
    print(f"Gate-related layers: {gate_related}")
    print(f"Spike-related layers: {spike_related}")
    print(f"Router-related layers: {router_related}")
    print(f"Expert-related layers: {expert_related}")
    print(f"Backbone layers: {len(backbone_related)} layers")
    
    print("\n🔧 FIXED Unfreezing Keywords Should Be:")
    print("-" * 40)
    
    # Generate the correct keywords based on actual layer names
    trainable_keywords = []
    
    if gate_related:
        for name in gate_related:
            if "dense" in name.lower():
                trainable_keywords.append(name.split("_")[0] + "_")
    
    if spike_related:
        for name in spike_related:
            if "dense" in name.lower() or "gru" in name.lower():
                trainable_keywords.append(name.split("_")[0] + "_")
    
    if router_related:
        for name in router_related:
            trainable_keywords.append(name.split("_")[0] + "_")
    
    if expert_related:
        trainable_keywords.append("expert_")
    
    trainable_keywords.append("conditional_spike_output")
    
    # Remove duplicates
    trainable_keywords = list(set(trainable_keywords))
    
    print("trainable_keywords = [")
    for keyword in trainable_keywords:
        print(f'    "{keyword}",')
    print("]")
    
    print("\n🧪 Test Unfreezing Logic:")
    print("-" * 25)
    
    # Test the unfreezing logic
    frozen_count = 0
    unfrozen_count = 0
    
    print("Would be UNFROZEN:")
    for layer in model.layers:
        layer_name = layer.name.lower()
        should_be_trainable = any(keyword.lower() in layer_name for keyword in trainable_keywords)
        
        if should_be_trainable:
            print(f"  ✓ {layer.name}")
            unfrozen_count += 1
        else:
            frozen_count += 1
    
    print(f"\nSummary: {unfrozen_count} unfrozen, {frozen_count} frozen")
    
    if unfrozen_count < 5:
        print("⚠️ Still too few layers unfrozen! Need to fix keywords.")
    else:
        print("✅ Looks good! Gate layers should be trainable now.")

if __name__ == "__main__":
    check_actual_layer_names()