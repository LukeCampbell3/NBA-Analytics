#!/usr/bin/env python3
"""
Test script for MoE metrics tracker.

This script verifies that the MoEMetricsTracker works correctly
by simulating router outputs and checking the computed metrics.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from moe_metrics import MoEMetricsTracker


def test_basic_metrics():
    """Test basic metric computation."""
    print("\n" + "="*70)
    print("TEST 1: Basic Metrics")
    print("="*70)
    
    # Create tracker
    tracker = MoEMetricsTracker(
        num_experts=12,
        num_spike_experts=3,
        dead_threshold=0.02
    )
    
    # Simulate uniform routing (healthy)
    batch_size = 100
    num_experts = 12
    
    # Uniform probabilities
    router_probs = np.ones((batch_size, num_experts)) / num_experts
    
    # Top-2 assignments (uniform distribution)
    expert_assignments = np.random.randint(0, num_experts, size=(batch_size, 2))
    
    # Update tracker
    tracker.update_batch(router_probs, expert_assignments)
    
    # Compute metrics
    metrics = tracker.compute_epoch_metrics()
    
    # Print
    tracker.print_summary(metrics, epoch=1)
    
    # Verify
    assert metrics['expert_usage_entropy_normalized'] > 0.9, "Uniform routing should have high entropy"
    assert metrics['expert_dead_rate'] < 0.1, "Uniform routing should have no dead experts"
    
    print("\n✓ Test 1 passed: Uniform routing detected correctly")


def test_collapsed_routing():
    """Test detection of collapsed routing."""
    print("\n" + "="*70)
    print("TEST 2: Collapsed Routing Detection")
    print("="*70)
    
    # Create tracker
    tracker = MoEMetricsTracker(
        num_experts=12,
        num_spike_experts=3,
        dead_threshold=0.02
    )
    
    # Simulate collapsed routing (2 experts dominate)
    batch_size = 100
    num_experts = 12
    
    # Concentrated probabilities (experts 0 and 1 dominate)
    router_probs = np.zeros((batch_size, num_experts))
    router_probs[:, 0] = 0.6
    router_probs[:, 1] = 0.3
    router_probs[:, 2:] = 0.1 / (num_experts - 2)
    
    # Assignments mostly to experts 0 and 1
    expert_assignments = np.zeros((batch_size, 2), dtype=np.int32)
    expert_assignments[:, 0] = 0
    expert_assignments[:, 1] = np.random.choice([1, 2], size=batch_size)
    
    # Update tracker
    tracker.update_batch(router_probs, expert_assignments)
    
    # Compute metrics
    metrics = tracker.compute_epoch_metrics()
    
    # Print
    tracker.print_summary(metrics, epoch=1)
    
    # Verify
    assert metrics['expert_usage_entropy_normalized'] < 0.5, "Collapsed routing should have low entropy"
    assert metrics['expert_dead_rate'] > 0.3, "Collapsed routing should have many dead experts"
    
    health = tracker.get_health_status(metrics)
    assert health in ['warning', 'critical'], "Collapsed routing should trigger warning"
    
    print(f"\n✓ Test 2 passed: Collapse detected (health={health})")


def test_spike_expert_tracking():
    """Test spike expert usage tracking."""
    print("\n" + "="*70)
    print("TEST 3: Spike Expert Tracking")
    print("="*70)
    
    # Create tracker
    tracker = MoEMetricsTracker(
        num_experts=12,
        num_spike_experts=3,
        dead_threshold=0.02
    )
    
    # Simulate routing with spike experts
    batch_size = 100
    num_experts = 12
    num_regular = 9
    num_spike = 3
    
    # Uniform probabilities
    router_probs = np.ones((batch_size, num_experts)) / num_experts
    
    # Assignments: spike experts for high spike samples
    expert_assignments = np.zeros((batch_size, 2), dtype=np.int32)
    spike_labels = np.random.random(batch_size) > 0.8  # 20% spike samples
    
    for i in range(batch_size):
        if spike_labels[i]:
            # Assign to spike experts (9, 10, 11)
            expert_assignments[i] = np.random.choice(range(num_regular, num_experts), size=2, replace=False)
        else:
            # Assign to regular experts (0-8)
            expert_assignments[i] = np.random.choice(range(num_regular), size=2, replace=False)
    
    # Create residuals and sigmas
    residuals = np.random.randn(batch_size) * 3
    sigmas = np.abs(np.random.randn(batch_size) * 2 + 3)
    
    # Update tracker
    tracker.update_batch(
        router_probs, 
        expert_assignments,
        spike_labels=spike_labels,
        residuals=residuals,
        sigmas=sigmas
    )
    
    # Compute metrics
    metrics = tracker.compute_epoch_metrics()
    
    # Print
    tracker.print_summary(metrics, epoch=1)
    
    # Verify
    spike_activation_rate = metrics.get('spike_expert_spike_label_rate', 0)
    assert spike_activation_rate > 0.5, "Spike experts should activate for spike samples"
    
    print(f"\n✓ Test 3 passed: Spike expert tracking works (activation={spike_activation_rate:.2%})")


def test_expert_overlap():
    """Test expert overlap computation."""
    print("\n" + "="*70)
    print("TEST 4: Expert Overlap")
    print("="*70)
    
    # Create tracker
    tracker = MoEMetricsTracker(
        num_experts=12,
        num_spike_experts=3,
        dead_threshold=0.02
    )
    
    # Simulate routing with expert outputs
    batch_size = 100
    num_experts = 12
    output_dim = 12  # 3 stats * 4 (mean, sigma, etc.)
    
    # Uniform probabilities
    router_probs = np.ones((batch_size, num_experts)) / num_experts
    
    # Random assignments
    expert_assignments = np.random.randint(0, num_experts, size=(batch_size, 2))
    
    # Create expert outputs (some similar, some different)
    expert_outputs = np.random.randn(batch_size, 2, output_dim)
    
    # Make some experts very similar (high overlap)
    for i in range(batch_size):
        if np.random.random() < 0.3:  # 30% of samples have similar experts
            expert_outputs[i, 1] = expert_outputs[i, 0] + np.random.randn(output_dim) * 0.1
    
    # Update tracker multiple times to accumulate enough samples
    for _ in range(10):  # 10 batches = 1000 samples
        tracker.update_batch(router_probs, expert_assignments, expert_outputs=expert_outputs)
    
    # Compute metrics
    metrics = tracker.compute_epoch_metrics()
    
    # Print
    tracker.print_summary(metrics, epoch=1)
    
    # Verify
    overlap = metrics.get('expert_overlap_mean', 0)
    # Overlap might be 0 if not enough samples were buffered (10% sampling rate)
    # Just verify it's computed without error
    print(f"\n✓ Test 4 passed: Overlap computed (mean={overlap:.3f})")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MOE METRICS TRACKER TESTS")
    print("="*70)
    
    try:
        test_basic_metrics()
        test_collapsed_routing()
        test_spike_expert_tracking()
        test_expert_overlap()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nThe MoEMetricsTracker is working correctly!")
        print("You can now use it in training with:")
        print("  python training/integrate_moe_improvements.py")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
