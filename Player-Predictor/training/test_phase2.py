#!/usr/bin/env python3
"""
Test Phase 2 Implementation

Quick test to verify Phase 2 specialization mechanisms are working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.integrate_moe_improvements import MoEImprovedTrainer


def test_phase2_config():
    """Test that Phase 2 configuration is properly set."""
    print("\n" + "="*70)
    print("TESTING PHASE 2 CONFIGURATION")
    print("="*70)
    
    trainer = MoEImprovedTrainer()
    
    # Check Phase 2 config
    assert trainer.config["use_prototype_experts"] == True, "Prototype experts should be enabled"
    assert trainer.config["prototype_key_dim"] == 256, "Key dimension should be 256"
    assert trainer.config["compactness_coef"] > 0, "Compactness coefficient should be positive"
    assert trainer.config["separation_coef"] > 0, "Separation coefficient should be positive"
    
    # Check Phase 1 still enabled
    assert trainer.config["use_load_balancing"] == True, "Phase 1 should still be enabled"
    
    # Check Phase 3-4 still disabled
    assert trainer.config["use_diversity_regularization"] == False, "Phase 3 should be disabled"
    assert trainer.config["use_expert_replay"] == False, "Phase 4 should be disabled"
    
    print("✅ Phase 2 configuration correct")
    print(f"   - Prototype keys: {trainer.config['prototype_key_dim']} dimensions")
    print(f"   - Compactness: {trainer.config['compactness_start']} → {trainer.config['compactness_final']}")
    print(f"   - Separation: {trainer.config['separation_coef']} (margin={trainer.config['separation_margin']})")
    
    return trainer


def test_compactness_schedule(trainer):
    """Test compactness weight scheduling."""
    print("\n" + "="*70)
    print("TESTING COMPACTNESS SCHEDULE")
    print("="*70)
    
    # Test ramp schedule
    weights = []
    for epoch in range(20):
        weight = trainer.get_compactness_weight(epoch)
        weights.append(weight)
        if epoch % 5 == 0 or epoch in [4, 5, 14, 15]:
            print(f"Epoch {epoch:2d}: weight = {weight:.4f}")
    
    # Verify ramping behavior
    assert weights[0] == trainer.config["compactness_start"], "Should start at start weight"
    assert weights[4] == trainer.config["compactness_start"], "Should stay at start until epoch 5"
    assert weights[10] > weights[5], "Should ramp up between epochs 5-15"
    assert weights[15] == trainer.config["compactness_final"], "Should reach final weight after epoch 15"
    
    print("\n✅ Compactness schedule working correctly")
    print(f"   Start: {weights[0]:.4f}")
    print(f"   Mid:   {weights[10]:.4f}")
    print(f"   Final: {weights[15]:.4f}")


def test_phase2_losses():
    """Test Phase 2 loss creation."""
    print("\n" + "="*70)
    print("TESTING PHASE 2 LOSSES")
    print("="*70)
    
    import tensorflow as tf
    import numpy as np
    
    trainer = MoEImprovedTrainer()
    
    # Create dummy data
    batch_size = 32
    key_dim = 256
    num_experts = 11
    
    # Router query
    router_query = tf.random.normal([batch_size, key_dim])
    
    # Expert keys
    expert_keys = [tf.random.normal([key_dim]) for _ in range(num_experts)]
    
    # Router probs
    router_probs = tf.nn.softmax(tf.random.normal([batch_size, num_experts]))
    
    # Create losses
    losses = trainer.create_phase2_losses(router_query, expert_keys, router_probs, num_experts)
    
    # Check losses exist
    assert 'compactness_loss' in losses, "Compactness loss should exist"
    assert 'separation_loss' in losses, "Separation loss should exist"
    assert 'key_similarity_mean' in losses, "Key similarity metric should exist"
    
    # Check loss values are reasonable
    compactness = losses['compactness_loss'].numpy()
    separation = losses['separation_loss'].numpy()
    similarity = losses['key_similarity_mean'].numpy()
    
    assert compactness > 0, "Compactness loss should be positive"
    assert separation >= 0, "Separation loss should be non-negative"
    assert -1 <= similarity <= 1, "Cosine similarity should be in [-1, 1]"
    
    print("✅ Phase 2 losses computed correctly")
    print(f"   Compactness loss: {compactness:.4f}")
    print(f"   Separation loss: {separation:.4f}")
    print(f"   Key similarity: {similarity:.4f}")


def test_model_building():
    """Test that Phase 2 model can be built."""
    print("\n" + "="*70)
    print("TESTING PHASE 2 MODEL BUILDING")
    print("="*70)
    
    trainer = MoEImprovedTrainer()
    
    # Initialize required attributes
    trainer.player_mapping = {"Player1": 0, "Player2": 1}
    trainer.team_mapping = {"Team1": 0, "Team2": 1}
    trainer.opponent_mapping = {"Opp1": 0, "Opp2": 1}
    trainer.feature_columns = ["feat" + str(i) for i in range(33)]
    trainer.baseline_features = ["PTS_rolling_avg", "TRB_rolling_avg", "AST_rolling_avg"]
    trainer.target_columns = ["PTS", "TRB", "AST"]
    trainer.spike_features = ["MP_trend", "High_MP_Flag", "FGA_trend", "AST_trend", 
                              "AST_variance", "USG_AST_ratio_trend", "High_Playmaker_Flag"]
    
    try:
        model = trainer.build_model_with_phase2()
        
        # Check model structure
        assert model is not None, "Model should be created"
        assert len(model.inputs) == 2, "Model should have 2 inputs (sequence, baseline)"
        assert len(model.outputs) == 1, "Model should have 1 output"
        
        # Check for Phase 2 metrics
        metric_names = [m.name for m in model.metrics]
        assert any('compactness' in name for name in metric_names), "Should have compactness metric"
        assert any('separation' in name for name in metric_names), "Should have separation metric"
        assert any('key_similarity' in name for name in metric_names), "Should have key similarity metric"
        
        print("✅ Phase 2 model built successfully")
        print(f"   Model name: {model.name}")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Phase 2 metrics: compactness, separation, key_similarity")
        
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 2 IMPLEMENTATION TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Configuration
        trainer = test_phase2_config()
        
        # Test 2: Compactness schedule
        test_compactness_schedule(trainer)
        
        # Test 3: Phase 2 losses
        test_phase2_losses()
        
        # Test 4: Model building
        test_model_building()
        
        print("\n" + "="*80)
        print("✅ ALL PHASE 2 TESTS PASSED")
        print("="*80)
        print("\nPhase 2 implementation is ready!")
        print("\nNext steps:")
        print("  1. Run full training: python training/integrate_moe_improvements.py")
        print("  2. Monitor Phase 2 metrics during training:")
        print("     - compactness_loss (should decrease)")
        print("     - separation_loss (should decrease)")
        print("     - key_similarity_mean (should decrease)")
        print("     - expert_overlap (should be < 0.40)")
        print("  3. Check Phase 2 success criteria at end")
        print("  4. If successful, enable Phase 3 (Diversity)")
        print("="*80 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
