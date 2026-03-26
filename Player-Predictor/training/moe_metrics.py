#!/usr/bin/env python3
"""
MoE Metrics Tracker - Detect Expert Collapse and Monitor Specialization

Implements comprehensive metrics from update.txt Section A to detect:
- Expert collapse (dead experts, low entropy)
- Redundancy (high overlap between experts)
- Spike expert behavior
"""

import numpy as np
import tensorflow as tf
from collections import defaultdict


class MoEMetricsTracker:
    """
    Tracks MoE health metrics to detect collapse and monitor specialization.
    
    Key metrics:
    - expert_usage_mean: Average routing fraction per expert
    - expert_usage_entropy: Entropy of expert usage distribution
    - expert_dead_rate: Fraction of experts with usage < threshold
    - expert_overlap: Similarity between expert outputs on shared inputs
    - spike_expert_usage: Spike expert activation by regime
    """
    
    def __init__(self, num_experts, num_spike_experts=3, dead_threshold=0.02):
        """
        Args:
            num_experts: Total number of experts (regular + spike)
            num_spike_experts: Number of spike experts
            dead_threshold: Usage threshold below which expert is considered dead
        """
        self.num_experts = num_experts
        self.num_spike_experts = num_spike_experts
        self.num_regular_experts = num_experts - num_spike_experts
        self.dead_threshold = dead_threshold
        
        # Accumulators for batch-level metrics
        self.reset_epoch()
    
    def reset_epoch(self):
        """Reset accumulators at start of epoch."""
        self.expert_counts = np.zeros(self.num_experts)
        self.total_samples = 0
        self.expert_outputs_buffer = defaultdict(list)  # For overlap computation
        self.spike_regime_counts = defaultdict(int)  # Track spike expert usage by regime
    
    def update_batch(self, router_probs, expert_assignments, expert_outputs=None, 
                     spike_labels=None, residuals=None, sigmas=None):
        """
        Update metrics with batch data.
        
        Args:
            router_probs: [B, E] routing probabilities after softmax
            expert_assignments: [B, k] top-k expert indices assigned
            expert_outputs: [B, k, d] expert outputs (optional, for overlap)
            spike_labels: [B] binary spike indicators (optional)
            residuals: [B] prediction residuals (optional)
            sigmas: [B] predicted uncertainties (optional)
        """
        batch_size = router_probs.shape[0]
        self.total_samples += batch_size
        
        # Update expert usage counts
        for expert_idx in expert_assignments.flatten():
            if 0 <= expert_idx < self.num_experts:
                self.expert_counts[expert_idx] += 1
        
        # Store expert outputs for overlap computation (sample to avoid memory issues)
        if expert_outputs is not None and np.random.random() < 0.1:  # Sample 10% of batches
            for i in range(batch_size):
                for k_idx in range(expert_assignments.shape[1]):
                    expert_idx = expert_assignments[i, k_idx]
                    if 0 <= expert_idx < self.num_experts:
                        output = expert_outputs[i, k_idx]
                        self.expert_outputs_buffer[expert_idx].append(output)
        
        # Track spike expert usage by regime
        if spike_labels is not None and residuals is not None:
            spike_expert_start = self.num_regular_experts
            for i in range(batch_size):
                # Determine regime
                is_spike = spike_labels[i] > 0.5
                high_residual = abs(residuals[i]) > 2.0 if residuals is not None else False
                high_sigma = sigmas[i] > 3.0 if sigmas is not None else False
                
                # Check if spike expert was assigned
                assigned_experts = expert_assignments[i]
                spike_expert_assigned = any(e >= spike_expert_start for e in assigned_experts)
                
                # Track by regime
                if is_spike:
                    self.spike_regime_counts['spike_label'] += 1
                    if spike_expert_assigned:
                        self.spike_regime_counts['spike_label_activated'] += 1
                
                if high_residual:
                    self.spike_regime_counts['high_residual'] += 1
                    if spike_expert_assigned:
                        self.spike_regime_counts['high_residual_activated'] += 1
                
                if high_sigma:
                    self.spike_regime_counts['high_sigma'] += 1
                    if spike_expert_assigned:
                        self.spike_regime_counts['high_sigma_activated'] += 1
    
    def compute_epoch_metrics(self):
        """
        Compute epoch-level metrics from accumulated data.
        
        Returns:
            dict: Metrics including usage, entropy, dead rate, overlap, spike usage
        """
        metrics = {}
        
        # 1. Expert usage mean (fraction per expert)
        expert_usage = self.expert_counts / (self.total_samples + 1e-9)
        metrics['expert_usage_mean'] = float(np.mean(expert_usage))
        metrics['expert_usage_std'] = float(np.std(expert_usage))
        metrics['expert_usage_min'] = float(np.min(expert_usage))
        metrics['expert_usage_max'] = float(np.max(expert_usage))
        
        # Per-expert usage for detailed analysis
        for i in range(self.num_experts):
            expert_type = 'spike' if i >= self.num_regular_experts else 'regular'
            metrics[f'expert_{i}_{expert_type}_usage'] = float(expert_usage[i])
        
        # 2. Expert usage entropy
        # Normalize to get probability distribution
        expert_probs = expert_usage / (np.sum(expert_usage) + 1e-9)
        # Compute entropy: -sum(p * log(p))
        entropy = -np.sum(expert_probs * np.log(expert_probs + 1e-9))
        max_entropy = np.log(self.num_experts)  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        metrics['expert_usage_entropy'] = float(entropy)
        metrics['expert_usage_entropy_normalized'] = float(normalized_entropy)
        
        # 3. Expert dead rate
        dead_experts = np.sum(expert_usage < self.dead_threshold)
        dead_rate = dead_experts / self.num_experts
        metrics['expert_dead_count'] = int(dead_experts)
        metrics['expert_dead_rate'] = float(dead_rate)
        
        # Separate dead rates for regular vs spike experts
        regular_dead = np.sum(expert_usage[:self.num_regular_experts] < self.dead_threshold)
        spike_dead = np.sum(expert_usage[self.num_regular_experts:] < self.dead_threshold)
        metrics['regular_expert_dead_rate'] = float(regular_dead / self.num_regular_experts)
        metrics['spike_expert_dead_rate'] = float(spike_dead / self.num_spike_experts) if self.num_spike_experts > 0 else 0.0
        
        # 4. Expert overlap (similarity between expert outputs)
        if len(self.expert_outputs_buffer) >= 2:
            overlaps = []
            expert_ids = list(self.expert_outputs_buffer.keys())
            
            # Sample pairs to avoid O(E^2) computation
            num_pairs = min(50, len(expert_ids) * (len(expert_ids) - 1) // 2)
            for _ in range(num_pairs):
                i, j = np.random.choice(expert_ids, size=2, replace=False)
                
                outputs_i = np.array(self.expert_outputs_buffer[i])
                outputs_j = np.array(self.expert_outputs_buffer[j])
                
                # Sample common size
                min_size = min(len(outputs_i), len(outputs_j), 50)
                if min_size > 0:
                    sample_i = outputs_i[np.random.choice(len(outputs_i), min_size, replace=False)]
                    sample_j = outputs_j[np.random.choice(len(outputs_j), min_size, replace=False)]
                    
                    # Compute mean cosine similarity
                    for out_i, out_j in zip(sample_i, sample_j):
                        cos_sim = np.dot(out_i, out_j) / (np.linalg.norm(out_i) * np.linalg.norm(out_j) + 1e-9)
                        overlaps.append(cos_sim)
            
            if overlaps:
                metrics['expert_overlap_mean'] = float(np.mean(overlaps))
                metrics['expert_overlap_std'] = float(np.std(overlaps))
                metrics['expert_overlap_max'] = float(np.max(overlaps))
            else:
                metrics['expert_overlap_mean'] = 0.0
                metrics['expert_overlap_std'] = 0.0
                metrics['expert_overlap_max'] = 0.0
        else:
            metrics['expert_overlap_mean'] = 0.0
            metrics['expert_overlap_std'] = 0.0
            metrics['expert_overlap_max'] = 0.0
        
        # 5. Spike expert usage by regime
        for regime in ['spike_label', 'high_residual', 'high_sigma']:
            total = self.spike_regime_counts.get(regime, 0)
            activated = self.spike_regime_counts.get(f'{regime}_activated', 0)
            activation_rate = activated / total if total > 0 else 0.0
            metrics[f'spike_expert_{regime}_rate'] = float(activation_rate)
            metrics[f'spike_expert_{regime}_count'] = int(total)
        
        return metrics
    
    def print_summary(self, metrics, epoch=None):
        """Print human-readable summary of metrics."""
        prefix = f"[Epoch {epoch}] " if epoch is not None else ""
        
        print(f"\n{prefix}MoE Health Metrics:")
        print("=" * 70)
        
        # Usage statistics
        print(f"Expert Usage:")
        print(f"  Mean: {metrics['expert_usage_mean']:.4f}")
        print(f"  Std:  {metrics['expert_usage_std']:.4f}")
        print(f"  Range: [{metrics['expert_usage_min']:.4f}, {metrics['expert_usage_max']:.4f}]")
        
        # Entropy
        print(f"\nExpert Entropy:")
        print(f"  Absolute: {metrics['expert_usage_entropy']:.4f}")
        print(f"  Normalized: {metrics['expert_usage_entropy_normalized']:.4f} (1.0 = uniform)")
        
        # Dead experts
        print(f"\nDead Experts (usage < {self.dead_threshold}):")
        print(f"  Total: {metrics['expert_dead_count']}/{self.num_experts} ({metrics['expert_dead_rate']*100:.1f}%)")
        print(f"  Regular: {metrics['regular_expert_dead_rate']*100:.1f}%")
        print(f"  Spike: {metrics['spike_expert_dead_rate']*100:.1f}%")
        
        # Overlap
        if metrics['expert_overlap_mean'] > 0:
            print(f"\nExpert Overlap (cosine similarity):")
            print(f"  Mean: {metrics['expert_overlap_mean']:.4f}")
            print(f"  Max:  {metrics['expert_overlap_max']:.4f}")
        
        # Spike expert behavior
        if any(k.startswith('spike_expert_') for k in metrics):
            print(f"\nSpike Expert Activation:")
            for regime in ['spike_label', 'high_residual', 'high_sigma']:
                rate = metrics.get(f'spike_expert_{regime}_rate', 0)
                count = metrics.get(f'spike_expert_{regime}_count', 0)
                print(f"  {regime}: {rate*100:.1f}% ({count} samples)")
        
        # Warnings
        print(f"\nHealth Warnings:")
        warnings = []
        if metrics['expert_dead_rate'] > 0.3:
            warnings.append(f"⚠️  HIGH DEAD RATE: {metrics['expert_dead_rate']*100:.1f}% experts dead")
        if metrics['expert_usage_entropy_normalized'] < 0.5:
            warnings.append(f"⚠️  LOW ENTROPY: {metrics['expert_usage_entropy_normalized']:.2f} (collapse risk)")
        if metrics['expert_overlap_mean'] > 0.7:
            warnings.append(f"⚠️  HIGH OVERLAP: {metrics['expert_overlap_mean']:.2f} (redundancy)")
        if metrics.get('spike_expert_spike_label_rate', 0) < 0.02 and metrics.get('spike_expert_spike_label_count', 0) > 0:
            warnings.append(f"⚠️  SPIKE EXPERTS INACTIVE: {metrics['spike_expert_spike_label_rate']*100:.1f}% activation")
        
        if warnings:
            for warning in warnings:
                print(f"  {warning}")
        else:
            print(f"  ✓ No critical warnings")
        
        print("=" * 70)
    
    def get_health_status(self, metrics):
        """
        Determine overall MoE health status.
        
        Returns:
            str: 'healthy', 'warning', or 'critical'
        """
        # Critical conditions
        if metrics['expert_dead_rate'] > 0.5:
            return 'critical'
        if metrics['expert_usage_entropy_normalized'] < 0.3:
            return 'critical'
        
        # Warning conditions
        if metrics['expert_dead_rate'] > 0.3:
            return 'warning'
        if metrics['expert_usage_entropy_normalized'] < 0.5:
            return 'warning'
        if metrics['expert_overlap_mean'] > 0.7:
            return 'warning'
        
        return 'healthy'
