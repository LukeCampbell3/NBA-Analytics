#!/usr/bin/env python3
"""
MoE Routing Mechanisms - Anti-Collapse and Specialization

Implements advanced routing from update.txt:
- Load-balancing loss (Switch-style, soft)
- Capacity enforcement with overflow rerouting
- Expert choice routing
- Differentiable sparse gates
- Prototype/centroid experts
"""

import numpy as np
import tensorflow as tf


class LoadBalancingLoss:
    """
    Switch-style load-balancing loss to prevent expert collapse.
    
    Implements two components:
    1. Importance loss: Penalizes concentrating probability mass
    2. Load loss: Penalizes concentrating actual assignments
    
    From update.txt Section B.1
    """
    
    def __init__(self, num_experts, schedule='ramp'):
        """
        Args:
            num_experts: Number of experts
            schedule: 'ramp' for gradual increase, 'constant' for fixed weight
        """
        self.num_experts = num_experts
        self.schedule = schedule
        self.current_coef = 0.001  # Start small
    
    def get_coefficient(self, epoch):
        """Get balance coefficient for current epoch."""
        if self.schedule == 'constant':
            return self.current_coef
        elif self.schedule == 'ramp':
            # Ramp schedule from update.txt
            if epoch < 3:
                return 0.001
            elif epoch < 8:
                return 0.003
            else:
                return 0.005
        return self.current_coef
    
    def compute_loss(self, router_probs, expert_assignments, epoch=0):
        """
        Compute load-balancing loss.
        
        Args:
            router_probs: [B, E] routing probabilities after softmax
            expert_assignments: [B, k] top-k expert indices
            epoch: Current epoch for scheduling
            
        Returns:
            balance_loss: Scalar loss
            metrics: Dict with importance_loss, load_loss, balance_coef
        """
        batch_size = tf.cast(tf.shape(router_probs)[0], tf.float32)
        num_experts = tf.cast(self.num_experts, tf.float32)
        
        # 1. Importance loss: E * sum(p_e^2)
        # p_e = mean probability mass per expert
        importance = tf.reduce_mean(router_probs, axis=0)  # [E]
        importance_loss = num_experts * tf.reduce_sum(tf.square(importance))
        
        # 2. Load loss: E * sum(c_e^2)
        # c_e = realized assignment fraction per expert
        # Count how many times each expert was actually assigned
        expert_counts = tf.zeros([self.num_experts], dtype=tf.float32)
        for e in range(self.num_experts):
            # Count assignments to expert e
            mask = tf.cast(tf.equal(expert_assignments, e), tf.float32)
            count = tf.reduce_sum(mask)
            expert_counts = tf.tensor_scatter_nd_update(
                expert_counts,
                [[e]],
                [count]
            )
        
        load = expert_counts / batch_size  # [E]
        load_loss = num_experts * tf.reduce_sum(tf.square(load))
        
        # 3. Combined loss with scheduling
        balance_coef = self.get_coefficient(epoch)
        balance_loss = balance_coef * (importance_loss + load_loss)
        
        metrics = {
            'importance_loss': importance_loss,
            'load_loss': load_loss,
            'balance_coef': balance_coef,
            'balance_loss': balance_loss
        }
        
        return balance_loss, metrics


class CapacityEnforcement:
    """
    Capacity enforcement with overflow rerouting.
    
    Prevents one expert from being overloaded by:
    1. Setting capacity per expert
    2. Rerouting overflow to next-best expert
    3. Penalizing excessive rerouting
    
    From update.txt Section B.2
    """
    
    def __init__(self, num_experts, capacity_factor=1.25, overflow_penalty=0.01):
        """
        Args:
            num_experts: Number of experts
            capacity_factor: Multiplier for capacity (C = factor * B / E)
            overflow_penalty: Penalty coefficient for rerouting
        """
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.overflow_penalty = overflow_penalty
    
    def enforce_capacity(self, router_logits, top_k=2):
        """
        Enforce capacity constraints with overflow rerouting.
        
        Args:
            router_logits: [B, E] routing logits before softmax
            top_k: Number of experts to select per sample
            
        Returns:
            assignments: [B, k] expert indices after capacity enforcement
            overflow_rate: Fraction of samples that were rerouted
            overflow_loss: Penalty for excessive rerouting
        """
        batch_size = tf.shape(router_logits)[0]
        
        # Compute capacity per expert
        capacity = tf.cast(
            tf.math.ceil(self.capacity_factor * tf.cast(batch_size, tf.float32) / self.num_experts),
            tf.int32
        )
        
        # Get top-k experts by score for each sample
        top_k_scores, top_k_indices = tf.nn.top_k(router_logits, k=top_k)
        
        # Track expert loads
        expert_loads = tf.zeros([self.num_experts], dtype=tf.int32)
        assignments = []
        overflow_count = 0
        
        # Process each sample
        for i in range(batch_size):
            sample_assignments = []
            
            for k_idx in range(top_k):
                best_expert = top_k_indices[i, k_idx]
                
                # Check if expert has capacity
                if expert_loads[best_expert] < capacity:
                    # Assign to best expert
                    sample_assignments.append(best_expert)
                    expert_loads = tf.tensor_scatter_nd_add(
                        expert_loads,
                        [[best_expert]],
                        [1]
                    )
                else:
                    # Overflow: find next-best expert with capacity
                    overflow_count += 1
                    assigned = False
                    
                    # Try remaining experts in order of score
                    for fallback_idx in range(k_idx + 1, self.num_experts):
                        if fallback_idx < top_k:
                            fallback_expert = top_k_indices[i, fallback_idx]
                        else:
                            # Need to look beyond top-k
                            # Find best available expert
                            available_mask = tf.cast(expert_loads < capacity, tf.float32)
                            masked_logits = router_logits[i] * available_mask - 1e9 * (1 - available_mask)
                            fallback_expert = tf.argmax(masked_logits)
                        
                        if expert_loads[fallback_expert] < capacity:
                            sample_assignments.append(fallback_expert)
                            expert_loads = tf.tensor_scatter_nd_add(
                                expert_loads,
                                [[fallback_expert]],
                                [1]
                            )
                            assigned = True
                            break
                    
                    if not assigned:
                        # Last resort: assign to least loaded expert
                        fallback_expert = tf.argmin(expert_loads)
                        sample_assignments.append(fallback_expert)
                        expert_loads = tf.tensor_scatter_nd_add(
                            expert_loads,
                            [[fallback_expert]],
                            [1]
                        )
            
            assignments.append(sample_assignments)
        
        assignments = tf.constant(assignments, dtype=tf.int32)
        
        # Compute overflow metrics
        overflow_rate = tf.cast(overflow_count, tf.float32) / tf.cast(batch_size * top_k, tf.float32)
        overflow_loss = self.overflow_penalty * overflow_rate
        
        return assignments, overflow_rate, overflow_loss


class PrototypeExperts:
    """
    Prototype/centroid experts with learnable domain keys.
    
    Each expert has a learnable key vector that represents its domain.
    Routing becomes "which domain does this sample belong to?"
    
    From update.txt Section D.5
    """
    
    def __init__(self, num_experts, key_dim, compactness_coef=0.01, separation_coef=0.001, margin=0.5):
        """
        Args:
            num_experts: Number of experts
            key_dim: Dimension of key vectors
            compactness_coef: Weight for compactness loss
            separation_coef: Weight for separation loss
            margin: Margin for separation loss
        """
        self.num_experts = num_experts
        self.key_dim = key_dim
        self.compactness_coef = compactness_coef
        self.separation_coef = separation_coef
        self.margin = margin
        
        # Learnable expert keys (initialized randomly)
        self.expert_keys = tf.Variable(
            tf.random.normal([num_experts, key_dim], stddev=0.1),
            trainable=True,
            name='expert_keys'
        )
    
    def compute_affinity(self, query):
        """
        Compute affinity scores between query and expert keys.
        
        Args:
            query: [B, key_dim] router query vectors
            
        Returns:
            affinity: [B, E] cosine similarity scores
        """
        # Normalize query and keys
        query_norm = tf.nn.l2_normalize(query, axis=-1)  # [B, key_dim]
        keys_norm = tf.nn.l2_normalize(self.expert_keys, axis=-1)  # [E, key_dim]
        
        # Compute cosine similarity
        affinity = tf.matmul(query_norm, keys_norm, transpose_b=True)  # [B, E]
        
        return affinity
    
    def compute_domain_losses(self, query, assigned_experts):
        """
        Compute compactness and separation losses.
        
        Args:
            query: [B, key_dim] router query vectors
            assigned_experts: [B] or [B, k] assigned expert indices
            
        Returns:
            domain_loss: Combined compactness + separation loss
            metrics: Dict with individual losses
        """
        # Handle both [B] and [B, k] assignments
        if len(assigned_experts.shape) == 2:
            # Take first assignment for simplicity
            assigned_experts = assigned_experts[:, 0]
        
        # 1. Compactness loss: pull queries toward assigned keys
        assigned_keys = tf.gather(self.expert_keys, assigned_experts)  # [B, key_dim]
        compactness_loss = tf.reduce_mean(tf.square(query - assigned_keys))
        
        # 2. Separation loss: push expert keys away from each other
        # Sample pairs to avoid O(E^2)
        num_pairs = min(32, self.num_experts * (self.num_experts - 1) // 2)
        separation_losses = []
        
        for _ in range(num_pairs):
            i, j = tf.random.uniform([2], 0, self.num_experts, dtype=tf.int32)
            if i != j:
                key_i = self.expert_keys[i]
                key_j = self.expert_keys[j]
                
                # Cosine similarity
                cos_sim = tf.reduce_sum(
                    tf.nn.l2_normalize(key_i, axis=-1) * tf.nn.l2_normalize(key_j, axis=-1)
                )
                
                # Hinge loss: max(0, margin - cos_sim)
                # We want cos_sim < margin (keys should be separated)
                separation_loss = tf.maximum(0.0, self.margin - cos_sim)
                separation_losses.append(separation_loss)
        
        separation_loss = tf.reduce_mean(separation_losses) if separation_losses else 0.0
        
        # Combined loss
        domain_loss = (
            self.compactness_coef * compactness_loss +
            self.separation_coef * separation_loss
        )
        
        metrics = {
            'compactness_loss': compactness_loss,
            'separation_loss': separation_loss,
            'domain_loss': domain_loss
        }
        
        return domain_loss, metrics


class DiversityRegularizer:
    """
    Diversity regularizers to reduce expert redundancy.
    
    Implements:
    - Option A: Output correlation penalty (most robust)
    - Option B: Orthogonal last-layer weights (cheap guardrail)
    
    From update.txt Section E.6
    """
    
    def __init__(self, diversity_coef=0.001, use_output_correlation=True, use_weight_orthogonal=True):
        """
        Args:
            diversity_coef: Weight for diversity losses
            use_output_correlation: Enable Option A
            use_weight_orthogonal: Enable Option B
        """
        self.diversity_coef = diversity_coef
        self.use_output_correlation = use_output_correlation
        self.use_weight_orthogonal = use_weight_orthogonal
    
    def output_correlation_penalty(self, expert_outputs, expert_assignments):
        """
        Option A: Penalize correlation between co-activated expert outputs.
        
        Args:
            expert_outputs: [B, k, d] outputs from top-k experts
            expert_assignments: [B, k] expert indices
            
        Returns:
            correlation_penalty: Scalar loss
        """
        if not self.use_output_correlation:
            return 0.0
        
        batch_size = tf.shape(expert_outputs)[0]
        k = tf.shape(expert_outputs)[1]
        
        if k < 2:
            return 0.0  # Need at least 2 experts for correlation
        
        # Compute pairwise cosine similarities within each sample
        correlations = []
        
        for i in range(batch_size):
            for j in range(k):
                for l in range(j + 1, k):
                    out_j = expert_outputs[i, j]
                    out_l = expert_outputs[i, l]
                    
                    # Cosine similarity
                    cos_sim = tf.reduce_sum(
                        tf.nn.l2_normalize(out_j, axis=-1) * tf.nn.l2_normalize(out_l, axis=-1)
                    )
                    correlations.append(cos_sim)
        
        if not correlations:
            return 0.0
        
        # Mean correlation (we want to minimize this)
        correlation_penalty = self.diversity_coef * tf.reduce_mean(correlations)
        
        return correlation_penalty
    
    def weight_orthogonal_penalty(self, expert_weights):
        """
        Option B: Penalize non-orthogonal expert last-layer weights.
        
        Args:
            expert_weights: List of [d_in, d_out] weight matrices for each expert
            
        Returns:
            orthogonal_penalty: Scalar loss
        """
        if not self.use_weight_orthogonal or len(expert_weights) < 2:
            return 0.0
        
        num_experts = len(expert_weights)
        
        # Sample pairs to avoid O(E^2)
        num_pairs = min(32, num_experts * (num_experts - 1) // 2)
        penalties = []
        
        for _ in range(num_pairs):
            i, j = np.random.choice(num_experts, size=2, replace=False)
            
            W_i = expert_weights[i]
            W_j = expert_weights[j]
            
            # Normalize weights
            W_i_norm = tf.nn.l2_normalize(W_i, axis=None)
            W_j_norm = tf.nn.l2_normalize(W_j, axis=None)
            
            # Frobenius norm of W_i^T W_j
            product = tf.matmul(W_i_norm, W_j_norm, transpose_a=True)
            penalty = tf.reduce_sum(tf.square(product))
            penalties.append(penalty)
        
        if not penalties:
            return 0.0
        
        orthogonal_penalty = self.diversity_coef * tf.reduce_mean(penalties)
        
        return orthogonal_penalty
