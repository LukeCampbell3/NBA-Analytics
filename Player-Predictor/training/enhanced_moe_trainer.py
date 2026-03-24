#!/usr/bin/env python3
"""
Enhanced MoE Trainer with Full Phase 1-3 Integration

This extends the hybrid_spike_moe_trainer.py to add:
- Phase 1: Load balancing + capacity enforcement
- Phase 2: Prototype experts with domain keys  
- Phase 3: Diversity regularization

Based on update.txt comprehensive MoE improvements.
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
import numpy as np
import io
import warnings

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_spike_moe_trainer import HybridSpikeMoETrainer
from moe_routing import LoadBalancingLoss, PrototypeExperts, DiversityRegularizer


class EnhancedMoELayer(Layer):
    """
    Custom Keras layer that implements enhanced MoE routing with:
    - Load balancing
    - Prototype experts
    - Diversity regularization
    """
    
    def __init__(self, 
                 num_experts,
                 num_spike_experts,
                 expert_dim,
                 spike_expert_capacity,
                 output_dim,
                 use_probabilistic=True,
                 # Phase 1: Load balancing
                 use_load_balancing=True,
                 load_balance_schedule='ramp',
                 # Phase 2: Prototypes
                 use_prototype_experts=True,
                 prototype_key_dim=256,
                 compactness_coef=0.01,
                 separation_coef=0.001,
                 separation_margin=0.5,
                 # Phase 3: Diversity
                 use_diversity_regularization=True,
                 diversity_coef=0.001,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_experts = num_experts
        self.num_spike_experts = num_spike_experts
        self.total_experts = num_experts + num_spike_experts
        self.expert_dim = expert_dim
        self.spike_expert_capacity = spike_expert_capacity
        self.output_dim = output_dim
        self.use_probabilistic = use_probabilistic
        
        # Phase 1
        self.use_load_balancing = use_load_balancing
        self.load_balance_schedule = load_balance_schedule
        
        # Phase 2
        self.use_prototype_experts = use_prototype_experts
        self.prototype_key_dim = prototype_key_dim
        self.compactness_coef = compactness_coef
        self.separation_coef = separation_coef
        self.separation_margin = separation_margin
        
        # Phase 3
        self.use_diversity_regularization = use_diversity_regularization
        self.diversity_coef = diversity_coef
        
        # Initialize components
        self._build_components()
    
    def _build_components(self):
        """Build all MoE components."""
        
        # Phase 2: Prototype experts (if enabled)
        if self.use_prototype_experts:
            self.router_query_layer = Dense(
                self.prototype_key_dim, 
                activation='relu', 
                name='router_query'
            )
            self.prototype_experts = PrototypeExperts(
                num_experts=self.total_experts,
                key_dim=self.prototype_key_dim,
                compactness_coef=self.compactness_coef,
                separation_coef=self.separation_coef,
                margin=self.separation_margin
            )
        else:
            # Standard router
            self.router_layer = Dense(self.total_experts, name='router')
        
        # Phase 1: Load balancing (if enabled)
        if self.use_load_balancing:
            self.load_balancer = LoadBalancingLoss(
                num_experts=self.total_experts,
                schedule=self.load_balance_schedule
            )
        
        # Phase 3: Diversity regularizer (if enabled)
        if self.use_diversity_regularization:
            self.diversity_regularizer = DiversityRegularizer(
                diversity_coef=self.diversity_coef,
                use_output_correlation=True,
                use_weight_orthogonal=False  # Will implement separately
            )
        
        # Expert networks
        self.expert_layers = []
        for i in range(self.total_experts):
            expert_type = "spike" if i >= self.num_experts else "regular"
            expert_dim = self.spike_expert_capacity if expert_type == "spike" else self.expert_dim
            
            # Each expert is a small MLP
            expert = tf.keras.Sequential([
                Dense(expert_dim, activation="relu", name=f"expert_{expert_type}_{i}_1"),
                Dense(self.output_dim, name=f"expert_{expert_type}_{i}_out")
            ], name=f"expert_{expert_type}_{i}")
            
            self.expert_layers.append(expert)
    
    def call(self, inputs, training=False):
        """
        Forward pass with enhanced MoE routing.
        
        Args:
            inputs: [sequence_repr_with_baseline, spike_indicators]
            training: Whether in training mode
            
        Returns:
            Mixed expert outputs
        """
        sequence_repr_with_baseline, spike_indicators = inputs
        batch_size = tf.shape(sequence_repr_with_baseline)[0]
        
        # Phase 2: Compute router logits (with or without prototypes)
        if self.use_prototype_experts:
            # Use prototype-based routing
            router_query = self.router_query_layer(sequence_repr_with_baseline)
            router_logits = self.prototype_experts.compute_affinity(router_query)
        else:
            # Standard routing
            router_logits = self.router_layer(sequence_repr_with_baseline)
        
        # Apply spike routing bias
        spike_expert_mask = tf.concat([
            tf.zeros([batch_size, self.num_experts]),
            tf.ones([batch_size, self.num_spike_experts])
        ], axis=1)
        avg_spike_score = tf.reduce_mean(spike_indicators, axis=1, keepdims=True)
        spike_routing_bias = avg_spike_score * spike_expert_mask * 1.0
        
        adjusted_router_logits = router_logits + spike_routing_bias
        router_probs = tf.nn.softmax(adjusted_router_logits)
        
        # Get top-k expert assignments (k=2)
        top_k = 2
        top_k_probs, top_k_indices = tf.nn.top_k(router_probs, k=top_k)
        
        # Normalize top-k probabilities
        top_k_probs_normalized = top_k_probs / (tf.reduce_sum(top_k_probs, axis=1, keepdims=True) + 1e-9)
        
        # Phase 1: Add load balancing loss (if enabled and training)
        if self.use_load_balancing and training:
            # Use epoch=0 for now (would need to pass from training loop)
            balance_loss, balance_metrics = self.load_balancer.compute_loss(
                router_probs=router_probs,
                expert_assignments=top_k_indices,
                epoch=0
            )
            self.add_loss(balance_loss)
            self.add_metric(balance_metrics['balance_loss'], name='balance_loss')
        
        # Phase 2: Add domain losses (if enabled and training)
        if self.use_prototype_experts and training:
            domain_loss, domain_metrics = self.prototype_experts.compute_domain_losses(
                query=router_query,
                assigned_experts=top_k_indices[:, 0]  # Use top-1 for domain loss
            )
            self.add_loss(domain_loss)
            self.add_metric(domain_metrics['compactness_loss'], name='compactness_loss')
            self.add_metric(domain_metrics['separation_loss'], name='separation_loss')
        
        # Compute expert outputs
        expert_outputs_list = []
        for i in range(self.total_experts):
            expert_out = self.expert_layers[i](sequence_repr_with_baseline, training=training)
            expert_outputs_list.append(expert_out)
        
        expert_stack = tf.stack(expert_outputs_list, axis=1)  # [B, E, D]
        
        # Gather top-k expert outputs
        batch_indices = tf.range(batch_size)[:, None]  # [B, 1]
        batch_indices = tf.tile(batch_indices, [1, top_k])  # [B, k]
        
        gather_indices = tf.stack([batch_indices, top_k_indices], axis=-1)  # [B, k, 2]
        top_k_expert_outputs = tf.gather_nd(expert_stack, gather_indices)  # [B, k, D]
        
        # Phase 3: Add diversity loss (if enabled and training)
        if self.use_diversity_regularization and training:
            output_corr_loss = self.diversity_regularizer.output_correlation_penalty(
                expert_outputs=top_k_expert_outputs,
                expert_assignments=top_k_indices
            )
            self.add_loss(output_corr_loss)
            self.add_metric(output_corr_loss, name='output_correlation_loss')
        
        # Mix expert outputs with top-k probabilities
        top_k_probs_expanded = top_k_probs_normalized[:, :, None]  # [B, k, 1]
        mixed_output = tf.reduce_sum(top_k_expert_outputs * top_k_probs_expanded, axis=1)  # [B, D]
        
        # Add routing metrics
        if training:
            self.add_metric(tf.reduce_mean(router_probs), name='avg_router_prob')
            self.add_metric(tf.reduce_max(router_probs, axis=1), name='max_router_prob')
            
            # Expert usage
            for i in range(self.total_experts):
                self.add_metric(tf.reduce_mean(router_probs[:, i]), name=f'expert_{i}_usage')
        
        return mixed_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
            'num_spike_experts': self.num_spike_experts,
            'expert_dim': self.expert_dim,
            'spike_expert_capacity': self.spike_expert_capacity,
            'output_dim': self.output_dim,
            'use_probabilistic': self.use_probabilistic,
            'use_load_balancing': self.use_load_balancing,
            'load_balance_schedule': self.load_balance_schedule,
            'use_prototype_experts': self.use_prototype_experts,
            'prototype_key_dim': self.prototype_key_dim,
            'compactness_coef': self.compactness_coef,
            'separation_coef': self.separation_coef,
            'separation_margin': self.separation_margin,
            'use_diversity_regularization': self.use_diversity_regularization,
            'diversity_coef': self.diversity_coef,
        })
        return config


class EnhancedMoETrainer(HybridSpikeMoETrainer):
    """
    Enhanced trainer with full Phase 1-3 MoE improvements integrated.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add Phase 1-3 configuration
        self.config.update({
            # Phase 1: Load balancing
            "use_load_balancing": True,
            "load_balance_schedule": "ramp",
            
            # Phase 2: Prototype experts
            "use_prototype_experts": True,
            "prototype_key_dim": 256,
            "compactness_coef": 0.01,
            "separation_coef": 0.001,
            "separation_margin": 0.5,
            
            # Phase 3: Diversity regularization
            "use_diversity_regularization": True,
            "diversity_coef": 0.001,
        })
        
        print("\n" + "="*70)
        print("ENHANCED MOE TRAINER - PHASES 1-3 INTEGRATED")
        print("="*70)
        print("Phase 1: Load balancing (Switch-style)")
        print("Phase 2: Prototype experts (learnable domain keys)")
        print("Phase 3: Diversity regularization (output correlation)")
        print("="*70 + "\n")
    
    def build_model(self):
        """
        Build model with enhanced MoE layer.
        
        This replaces the standard MoE routing with the enhanced version
        that includes Phases 1-3 improvements.
        """
        from tensorflow.keras.layers import (
            Input, Embedding, Lambda, Concatenate, Dense, 
            MultiHeadAttention, Add, LayerNormalization, Dropout,
            GlobalAveragePooling1D, Softmax
        )
        from tensorflow.keras.models import Model
        from hybrid_spike_moe_trainer import ConditionalSpikeOutput
        
        print("🏗️ Building Enhanced MoE with Phases 1-3...")
        
        # Input layers
        sequence_input = Input(shape=(self.config["seq_len"], len(self.feature_columns)), name="sequence_input")
        baseline_input = Input(shape=(len(self.baseline_features),), name="baseline_input")
        
        # Extract categorical features for embedding
        player_ids = Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32), name="player_ids")(sequence_input)
        team_ids = Lambda(lambda x: tf.cast(x[:, :, 1], tf.int32), name="team_ids")(sequence_input)
        opponent_ids = Lambda(lambda x: tf.cast(x[:, :, 2], tf.int32), name="opponent_ids")(sequence_input)
        
        # Embeddings
        player_embed = Embedding(len(self.player_mapping), 16, name="player_embed")(player_ids)
        team_embed = Embedding(len(self.team_mapping), 8, name="team_embed")(team_ids)
        opponent_embed = Embedding(len(self.opponent_mapping), 8, name="opponent_embed")(opponent_ids)
        
        # Numeric features (skip first 3 categorical)
        numeric_features = Lambda(lambda x: x[:, :, 3:], name="numeric_features")(sequence_input)
        
        # Combine features
        combined = Concatenate(axis=-1, name="combined_features")([
            player_embed, team_embed, opponent_embed, numeric_features
        ])
        
        # Project to model dimension
        x = Dense(self.config["d_model"], name="input_projection")(combined)
        
        # Transformer layers
        for i in range(self.config["n_layers"]):
            # Multi-head attention
            attn = MultiHeadAttention(
                num_heads=self.config["n_heads"],
                key_dim=self.config["d_model"] // self.config["n_heads"],
                name=f"attention_{i}"
            )(x, x)
            
            x = Add(name=f"add_attn_{i}")([x, attn])
            x = LayerNormalization(name=f"norm_attn_{i}")(x)
            
            # Feed forward
            ff = Dense(self.config["d_model"] * 2, activation="relu", name=f"ff1_{i}")(x)
            ff = Dropout(self.config["dropout"], name=f"dropout_ff_{i}")(ff)
            ff = Dense(self.config["d_model"], name=f"ff2_{i}")(ff)
            
            x = Add(name=f"add_ff_{i}")([x, ff])
            x = LayerNormalization(name=f"norm_ff_{i}")(x)
        
        # Use full sequence for routing
        sequence_repr = GlobalAveragePooling1D(name="sequence_pooling")(x)
        
        # Condition router/experts on baseline
        sequence_repr_with_baseline = Concatenate(name="sequence_baseline_concat")([sequence_repr, baseline_input])
        
        # Per-target spike indicators
        spike_features_last = Lambda(lambda x: x[:, -1, -len(self.spike_features):], name="spike_features")(sequence_input)
        spike_indicators = Dense(len(self.target_columns), activation="sigmoid", name="per_target_spike_indicators")(spike_features_last)
        
        # ENHANCED MOE LAYER (Phases 1-3)
        # Output delta predictions in format expected by ConditionalSpikeOutput
        # For probabilistic: [means, scales, spike_deltas, spike_scales] (12 dims)
        # For non-probabilistic: [means] (3 dims)
        if self.config["use_probabilistic"]:
            output_dim = len(self.target_columns) * 4  # means + scales + spike_deltas + spike_scales
        else:
            output_dim = len(self.target_columns)
        
        enhanced_moe = EnhancedMoELayer(
            num_experts=self.config["num_experts"],
            num_spike_experts=self.config["num_spike_experts"],
            expert_dim=self.config["expert_dim"],
            spike_expert_capacity=self.config["spike_expert_capacity"],
            output_dim=output_dim,
            use_probabilistic=self.config["use_probabilistic"],
            # Phase 1
            use_load_balancing=self.config["use_load_balancing"],
            load_balance_schedule=self.config["load_balance_schedule"],
            # Phase 2
            use_prototype_experts=self.config["use_prototype_experts"],
            prototype_key_dim=self.config["prototype_key_dim"],
            compactness_coef=self.config["compactness_coef"],
            separation_coef=self.config["separation_coef"],
            separation_margin=self.config["separation_margin"],
            # Phase 3
            use_diversity_regularization=self.config["use_diversity_regularization"],
            diversity_coef=self.config["diversity_coef"],
            name="enhanced_moe"
        )
        
        # Get MoE output (delta predictions)
        delta_output = enhanced_moe([sequence_repr_with_baseline, spike_indicators])
        
        # Apply ConditionalSpikeOutput layer to get final predictions
        # This layer adds baseline, applies bounds, and formats output correctly
        final_output = ConditionalSpikeOutput(
            use_probabilistic=self.config["use_probabilistic"],
            min_scale=self.config["min_scale"],
            max_scale_pts=self.config["max_scale_pts"],
            max_scale_trb=self.config["max_scale_trb"],
            max_scale_ast=self.config["max_scale_ast"],
            name="conditional_spike_output"
        )([delta_output, baseline_input, spike_indicators])
        
        # Create model
        model = Model(inputs=[sequence_input, baseline_input], outputs=final_output, name="EnhancedMoE")
        
        print(f"✓ Enhanced MoE built with {self.config['num_experts'] + self.config['num_spike_experts']} experts")
        print(f"  - Phase 1: Load balancing ({self.config['load_balance_schedule']})")
        print(f"  - Phase 2: Prototype experts (key_dim={self.config['prototype_key_dim']})")
        print(f"  - Phase 3: Diversity regularization (coef={self.config['diversity_coef']})")
        
        return model


def main():
    """Test the enhanced trainer."""
    print("\n" + "="*70)
    print("ENHANCED MOE TRAINER - FULL INTEGRATION TEST")
    print("="*70)
    
    trainer = EnhancedMoETrainer()
    trainer.train()


if __name__ == "__main__":
    main()
