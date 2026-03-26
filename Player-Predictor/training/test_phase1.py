#!/usr/bin/env python3
"""
Test Phase 1 Implementation

Quick test to verify Phase 1 anti-collapse mechanisms are working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.integrate_moe_improvements import MoEImprovedTrainer


def test_phase1_config():
    """Test that Phase 1 configuration is properly set."""
    print("\n" + "="*70)
    print("TESTING PHASE 1 CONFIGURATION")
    print("="*70)
    
    trainer = MoEImprovedTrainer()
    
    # Check Phase 1 config
    assert trainer.config["use_load_balancing"] == True, "Load balancing should be enabled"
    assert trainer.config["use_capacity_enforcement"] == True, "Capacity enforcement should be enabled"
    assert trainer.config["use_importance_loss"] == True, "Importance loss should be enabled"
    assert trainer.config["use_load_loss"] == True, "Load loss should be enabled"
    
    # Check Phase 2-4 are disabled
    assert trainer.config["use_prototype_experts"] == False, "Phase 2 should be disabled"
    assert trainer.config["use_diversity_regularization"] == False, "Phase 3 should be disabled"
    assert trainer.config["use_expert_replay"] == False, "Phase 4 should be disabled"
    
    print("✅ Phase 1 configuration correct")
    print(f"   - Load balancing: {trainer.config['load_balance_schedule']}")
    print(f"   - Capacity factor: {trainer.config['capacity_factor']}")
    print(f"   - Importance weight: {trainer.config['importance_weight']}")
    print(f"   - Load weight: {trainer.config['load_weight']}")
    
    return trainer


def test_load_balance_schedule(trainer):
    """Test load balance weight scheduling."""
    print("\n" + "="*70)
    print("TESTING LOAD BALANCE SCHEDULE")
    print("="*70)
    
    # Test ramp schedule
    weights = []
    for epoch in range(15):
        weight = trainer.get_load_balance_weight(epoch)
        weights.append(weight)
        print(f"Epoch {epoch:2d}: weight = {weight:.4f}")
    
    # Verify ramping behavior
    assert weights[0] == trainer.config["load_balance_weight_start"], "Should start at start weight"
    assert weights[2] == trainer.config["load_balance_weight_start"], "Should stay at start until epoch 3"
    assert weights[7] > weights[3], "Should ramp up between epochs 3-8"
    assert weights[10] == trainer.config["load_balance_weight_final"], "Should reach final weight after epoch 8"
    
    print("\n✅ Load balance schedule working correctly")
    print(f"   Start: {weights[0]:.4f}")
    print(f"   Mid:   {weights[5]:.4f}")
    print(f"   Final: {weights[10]:.4f}")


def test_moe_metrics():
    """Test MoE metrics tracker initialization."""
    print("\n" + "="*70)
    print("TESTING MOE METRICS TRACKER")
    print("="*70)
    
    trainer = MoEImprovedTrainer()
    
    assert trainer.moe_metrics is not None, "MoE metrics should be initialized"
    assert trainer.moe_metrics.num_experts == 11, "Should have 11 total experts (8 regular + 3 spike)"
    assert trainer.moe_metrics.num_spike_experts == 3, "Should have 3 spike experts"
    assert trainer.moe_metrics.dead_threshold == 0.02, "Dead threshold should be 0.02"
    
    print("✅ MoE metrics tracker initialized correctly")
    print(f"   Total experts: {trainer.moe_metrics.num_experts}")
    print(f"   Spike experts: {trainer.moe_metrics.num_spike_experts}")
    print(f"   Dead threshold: {trainer.moe_metrics.dead_threshold}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 1 IMPLEMENTATION TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Configuration
        trainer = test_phase1_config()
        
        # Test 2: Load balance schedule
        test_load_balance_schedule(trainer)
        
        # Test 3: MoE metrics
        test_moe_metrics()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nPhase 1 implementation is ready!")
        print("\nNext steps:")
        print("  1. Run full training: python training/integrate_moe_improvements.py")
        print("  2. Monitor MoE metrics during training")
        print("  3. Check Phase 1 success criteria at end")
        print("  4. If successful, enable Phase 2 (Specialization)")
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
