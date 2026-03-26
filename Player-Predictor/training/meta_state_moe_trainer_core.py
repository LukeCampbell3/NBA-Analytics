#!/usr/bin/env python3
"""
Meta-State MoE Trainer - Core Architecture
Implements "meta-state → routing → delta prediction (+ uncertainty)"

This file contains ONLY the model architecture changes.
Use with improved_baseline_trainer.py for training loop.
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np


class MetaStateHead(Layer):
    """
    Meta-state head: explicit "reason encoder"
    Transforms sequence representation into interpretable meta-state z
    """
    def __init__(self, z_dim=48, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.dropout = dropout
        
    def build(self, input_shape):
        self.projection = Dense(self.z_dim, name="meta_projection")
        self.norm = LayerNormalization(name="meta_norm")
        self.activation = Activation("gelu", name="meta_activation")
        self.dropout_layer = Dropout(self.dropout, name="meta_dropout")
        
    def call(self, inputs, training=False):
        z = self.projection(inputs)
        z = self.norm(z)
        z = self.activation(z)
        z = self.dropout_layer(z, training=training)
        return z


class EventPredictionHead(Layer):
    """
    Event prediction head: predicts discrete events/regimes
    Events: minutes_spike, usage_spike, pace_tier, blowout_risk
    """
    def __init__(self, num_events=4, **kwargs):
        super().__init__(**kwargs)
        self.num_events = num_events
        
    def build(self, input_shape):
        self.event_dense = Dense(self.num_events, activation="sigmoid", name="event_logits")
        
    def call(self, inputs):
        return self.event_dense(inputs)


class EpistemicUncertaintyHead(Layer):
    """
    Epistemic uncertainty head: model uncertainty (not data noise)
    Separate from aleatoric uncertainty (sigma_ale)
    """
    def __init__(self, n_targets=3, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets
        
    def build(self, input_shape):
        self.epi_dense = Dense(self.n_targets, activation="softplus", name="epistemic")
        
    def call(self, inputs):
        return self.epi_dense(inputs)


class EventGatedOutput(Layer):
    """
    Event-gated spike expert output
    Replaces threshold-based spike activation with learned event gate
    """
    def __init__(self, n_targets=3, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets
        
    def build(self, input_shape):
        # Gate network: z + u_epi → gate probability
        self.gate_dense1 = Dense(16, activation="relu", name="gate_hidden")
        self.gate_dense2 = Dense(1, activation="sigmoid", name="gate_output")
        
    def call(self, inputs):
        """
        inputs: [delta_combined, z, u_epi]
        delta_combined: [batch, n_targets*2] - base + spike deltas
        z: [batch, z_dim]
        u_epi: [batch, n_targets]
        """
        delta_combined, z, u_epi = inputs
        
        # Compute gate
        gate_input = Concatenate()([z, u_epi])
        gate_hidden = self.gate_dense1(gate_input)
        gate = self.gate_dense2(gate_hidden)  # [batch, 1]
        
        # Split deltas
        delta_base = delta_combined[:, :self.n_targets]
        delta_spike = delta_combined[:, self.n_targets:2*self.n_targets]
        
        # Gated combination
        delta_final = delta_base + gate * delta_spike
        
        return delta_final, gate


def build_meta_state_moe(config, feature_columns, target_columns, 
                         player_mapping, team_mapping, opponent_mapping):
    """
    Build meta-state MoE architecture
    
    Key changes from original:
    1. Meta-state head z after sequence encoding
    2. Router uses z (not raw sequence_repr)
    3. Event prediction head
    4. Event-gated spike experts
    5. Separated epistemic uncertainty
    """
    
    # Input layers
    sequence_input = Input(shape=(config["seq_len"], len(feature_columns)), name="sequence_input")
    baseline_input = Input(shape=(len(target_columns),), name="baseline_input")
    
    # Extract and embed categorical features (same as original)
    player_ids = Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32))(sequence_input)
    team_ids = Lambda(lambda x: tf.cast(x[:, :, 1], tf.int32))(sequence_input)
    opponent_ids = Lambda(lambda x: tf.cast(x[:, :, 2], tf.int32))(sequence_input)
    
    player_embed = Embedding(len(player_mapping), 16)(player_ids)
    team_embed = Embedding(len(team_mapping), 8)(team_ids)
    opponent_embed = Embedding(len(opponent_mapping), 8)(opponent_ids)
    
    numeric_features = Lambda(lambda x: x[:, :, 3:])(sequence_input)
    combined = Concatenate(axis=-1)([player_embed, team_embed, opponent_embed, numeric_features])
    
    # Transformer encoder (same as original)
    x = Dense(config["d_model"])(combined)
    
    for i in range(config["n_layers"]):
        attn = MultiHeadAttention(
            num_heads=config["n_heads"],
            key_dim=config["d_model"] // config["n_heads"]
        )(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        
        ff = Dense(config["d_model"] * 2, activation="relu")(x)
        ff = Dropout(config["dropout"])(ff)
        ff = Dense(config["d_model"])(ff)
        x = Add()([x, ff])
        x = LayerNormalization()(x)
    
    sequence_repr = GlobalAveragePooling1D()(x)
    
    # NEW: Meta-state head
    z = MetaStateHead(z_dim=config.get("z_dim", 48), dropout=config["dropout"])(sequence_repr)
    
    # NEW: Event prediction
    event_logits = EventPredictionHead(num_events=config.get("num_events", 4))(z)
    
    # NEW: Epistemic uncertainty
    u_epi = EpistemicUncertaintyHead(n_targets=len(target_columns))(z)
    
    # Router uses z + baseline (not raw sequence_repr)
    router_input = Concatenate()([z, baseline_input])
    
    total_experts = config["num_experts"] + config["num_spike_experts"]
    router_logits = Dense(total_experts)(router_input)
    router_probs = Softmax()(router_logits)
    
    # Expert networks (output delta_mean + sigma_ale)
    expert_outputs = []
    for i in range(total_experts):
        expert_dim = config.get("expert_dim", 128)
        expert = Dense(expert_dim, activation="relu")(router_input)
        expert = Dropout(config["dropout"])(expert)
        # Output: delta_mean (3) + sigma_ale (3) = 6 total
        expert_out = Dense(len(target_columns) * 2)(expert)
        expert_outputs.append(expert_out)
    
    # Combine expert outputs
    expert_stack = tf.stack(expert_outputs, axis=1)
    router_probs_expanded = tf.expand_dims(router_probs, axis=-1)
    combined_output = tf.reduce_sum(expert_stack * router_probs_expanded, axis=1)
    
    # Split into base and spike components for gating
    # Base experts: first num_experts
    # Spike experts: last num_spike_experts
    base_mask = tf.concat([
        tf.ones([tf.shape(router_probs)[0], config["num_experts"]]),
        tf.zeros([tf.shape(router_probs)[0], config["num_spike_experts"]])
    ], axis=1)
    spike_mask = 1.0 - base_mask
    
    base_output = tf.reduce_sum(expert_stack * tf.expand_dims(router_probs * base_mask, -1), axis=1)
    spike_output = tf.reduce_sum(expert_stack * tf.expand_dims(router_probs * spike_mask, -1), axis=1)
    
    # Combine base and spike with event gate
    delta_combined = Concatenate()([base_output[:, :len(target_columns)], 
                                    spike_output[:, :len(target_columns)]])
    delta_final, gate = EventGatedOutput(n_targets=len(target_columns))([delta_combined, z, u_epi])
    
    # CRITICAL FIX: Output delta, not absolute prediction
    # The trainer will reconstruct mu = baseline + delta
    # This matches the original trainer's predict_delta=True convention
    
    # Aleatoric uncertainty from combined expert output
    sigma_ale = combined_output[:, len(target_columns):2*len(target_columns)]
    sigma_ale = tf.clip_by_value(sigma_ale, 0.5, 15.0)
    
    # Final output: [delta_mean, sigma_ale, u_epi, event_logits, gate]
    # Output delta (3), not mu, so trainer can reconstruct properly
    final_output = Concatenate()([delta_final, sigma_ale, u_epi, event_logits, gate])
    
    model = Model(inputs=[sequence_input, baseline_input], outputs=final_output)
    
    # Add regularization losses
    # 1. Expert usage (6-10 active)
    active_experts = tf.reduce_sum(tf.cast(router_probs > 0.05, tf.float32), axis=1)
    expert_usage_loss = tf.reduce_mean(tf.square(active_experts - 8.0))
    model.add_loss(config.get("expert_usage_weight", 0.1) * expert_usage_loss)
    
    # 2. Gate penalty (prevent always-on)
    gate_mean = tf.reduce_mean(gate)
    gate_penalty = tf.square(gate_mean - config.get("gate_mean_target", 0.2))
    model.add_loss(config.get("gate_penalty_weight", 0.05) * gate_penalty)
    
    # 3. Router entropy (prevent collapse)
    router_entropy = -tf.reduce_mean(tf.reduce_sum(router_probs * tf.math.log(router_probs + 1e-8), axis=1))
    model.add_loss(0.01 * tf.square(router_entropy - 1.5))
    
    # Add metrics
    model.add_metric(gate_mean, name="gate_activation")
    model.add_metric(tf.reduce_mean(active_experts), name="active_experts")
    model.add_metric(router_entropy, name="router_entropy")
    model.add_metric(tf.reduce_mean(u_epi), name="epistemic_mean")
    
    return model
