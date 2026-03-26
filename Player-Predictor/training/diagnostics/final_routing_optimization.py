#!/usr/bin/env python3
"""
Final routing optimization for the hybrid spike MoE trainer.
Goal: Achieve the perfect balance of:
- Good confidence (>0.50) and low entropy (<0.90)
- Realistic spike frequency (12-15%)
- Good regime separation
- All experts utilized
- Accurate predictions
"""

import numpy as np
import tensorflow as tf

def apply_final_routing_optimization():
    """Apply final routing optimization for perfect balance"""
    
    print("🎯 Applying Final Routing Optimization")
    print("=" * 50)
    
    # Read the current trainer file
    with open('hybrid_spike_moe_trainer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Optimize temperature for better confidence without losing balance
    print("1. Optimizing temperature for confidence...")
    content = content.replace(
        '"router_temperature": 0.8,  # BALANCED: Moderate sharpness for intelligent decisions',
        '"router_temperature": 0.6,  # OPTIMIZED: Better confidence while maintaining flexibility'
    )
    
    # Fix 2: Adjust confidence and entropy targets
    print("2. Adjusting confidence and entropy targets...")
    content = content.replace(
        '"confidence_threshold": 0.55,  # BALANCED: Moderate confidence for flexibility',
        '"confidence_threshold": 0.52,  # OPTIMIZED: Achievable confidence target'
    )
    content = content.replace(
        '"entropy_target": 1.2,  # BALANCED: Allow some uncertainty for intelligent routing',
        '"entropy_target": 0.85,  # OPTIMIZED: Lower entropy for better decisions'
    )
    
    # Fix 3: Enhance the context-aware spike detection for better regime separation
    print("3. Enhancing context-aware spike detection...")
    
    context_aware_old = '''        # Context-aware spike detection: consider recent performance and game situation
        # Use baseline values to inform spike likelihood
        baseline_pts = baseline_input[:, 0:1]  # PTS baseline
        baseline_trb = baseline_input[:, 1:2]  # TRB baseline  
        baseline_ast = baseline_input[:, 2:3]  # AST baseline
        
        # Higher baselines = higher spike potential
        baseline_spike_factor = tf.concat([
            tf.sigmoid((baseline_pts - 20.0) / 10.0),  # PTS spike factor
            tf.sigmoid((baseline_trb - 8.0) / 5.0),    # TRB spike factor
            tf.sigmoid((baseline_ast - 6.0) / 4.0)     # AST spike factor
        ], axis=1)
        
        # Combine raw indicators with baseline context
        context_aware_indicators = spike_indicators * (0.7 + 0.3 * baseline_spike_factor)
        spike_indicators = Lambda(lambda x: tf.minimum(x * sensitivity, 1.0), name="context_aware_spike_detection")(context_aware_indicators)'''
    
    context_aware_new = '''        # Enhanced context-aware spike detection for better regime separation
        # Use baseline values and recent trends to inform spike likelihood
        baseline_pts = baseline_input[:, 0:1]  # PTS baseline
        baseline_trb = baseline_input[:, 1:2]  # TRB baseline  
        baseline_ast = baseline_input[:, 2:3]  # AST baseline
        
        # More nuanced baseline spike factors
        baseline_spike_factor = tf.concat([
            tf.sigmoid((baseline_pts - 18.0) / 8.0),   # PTS spike factor (adjusted thresholds)
            tf.sigmoid((baseline_trb - 7.0) / 4.0),    # TRB spike factor
            tf.sigmoid((baseline_ast - 5.0) / 3.0)     # AST spike factor
        ], axis=1)
        
        # Get recent performance trends from spike features
        recent_trends = spike_features_last[:, -3:]  # Last 3 trend features
        trend_factor = tf.reduce_mean(tf.abs(recent_trends), axis=1, keepdims=True)
        trend_boost = tf.sigmoid(trend_factor - 0.5) * 0.2  # 0-20% boost based on trends
        
        # Combine raw indicators with baseline context and trends
        context_multiplier = 0.6 + 0.3 * baseline_spike_factor + tf.tile(trend_boost, [1, 3])
        context_aware_indicators = spike_indicators * context_multiplier
        spike_indicators = Lambda(lambda x: tf.minimum(x * sensitivity, 1.0), name="enhanced_context_spike_detection")(context_aware_indicators)'''
    
    content = content.replace(context_aware_old, context_aware_new)
    
    # Fix 4: Improve the intelligent routing bias for better separation
    print("4. Improving intelligent routing bias...")
    
    routing_bias_old = '''        # Apply intelligent, conditional routing strength for spike experts
        base_routing_strength = strategy.get("spike_routing_strength", self.config.get("spike_routing_strength", 1.0))
        
        # Conditional routing: stronger bias only when spike indicators are actually high
        spike_confidence_threshold = 0.4  # Only apply strong bias when spike confidence > 40%
        confident_spike_mask = tf.cast(avg_spike_score > spike_confidence_threshold, tf.float32)
        
        # Graduated routing strength based on spike confidence
        routing_strength = base_routing_strength * (0.2 + 0.8 * avg_spike_score)  # Scale from 20% to 100% of base strength
        spike_routing_bias = confident_spike_mask * avg_spike_score * spike_expert_mask * routing_strength'''
    
    routing_bias_new = '''        # Apply intelligent, graduated routing strength for spike experts
        base_routing_strength = strategy.get("spike_routing_strength", self.config.get("spike_routing_strength", 1.0))
        
        # Multi-threshold routing for better regime separation
        low_spike_threshold = 0.2   # Weak spike indicators
        med_spike_threshold = 0.5   # Medium spike indicators  
        high_spike_threshold = 0.7  # Strong spike indicators
        
        # Graduated routing strength based on spike confidence levels
        low_spike_mask = tf.cast((avg_spike_score > low_spike_threshold) & (avg_spike_score <= med_spike_threshold), tf.float32)
        med_spike_mask = tf.cast((avg_spike_score > med_spike_threshold) & (avg_spike_score <= high_spike_threshold), tf.float32)
        high_spike_mask = tf.cast(avg_spike_score > high_spike_threshold, tf.float32)
        
        # Different routing strengths for different confidence levels
        routing_strength = (
            low_spike_mask * base_routing_strength * 0.3 +   # 30% strength for low confidence
            med_spike_mask * base_routing_strength * 0.7 +   # 70% strength for medium confidence
            high_spike_mask * base_routing_strength * 1.2    # 120% strength for high confidence
        )
        
        spike_routing_bias = avg_spike_score * spike_expert_mask * routing_strength'''
    
    content = content.replace(routing_bias_old, routing_bias_new)
    
    # Fix 5: Add expert utilization balancing
    print("5. Adding expert utilization balancing...")
    
    # Find the expert outputs section and add utilization tracking
    expert_outputs_section = '''        # Combine expert outputs
        expert_stack = tf.stack(expert_outputs, axis=1)
        router_probs_expanded = tf.expand_dims(router_probs, axis=-1)
        delta_output = tf.reduce_sum(expert_stack * router_probs_expanded, axis=1)'''
    
    enhanced_expert_outputs = '''        # Combine expert outputs with utilization balancing
        expert_stack = tf.stack(expert_outputs, axis=1)
        router_probs_expanded = tf.expand_dims(router_probs, axis=-1)
        
        # Add expert utilization balancing for better gradient flow
        if self.config.get("expert_utilization_balancing", False):
            # Compute expert usage statistics
            expert_usage = tf.reduce_mean(router_probs, axis=0)
            target_usage = 1.0 / total_experts
            
            # Apply gentle balancing to underused experts
            usage_ratio = expert_usage / (target_usage + 1e-8)
            underused_boost = tf.maximum(0.0, 1.0 - usage_ratio) * 0.1  # 10% boost for underused
            
            # Apply boost to router probabilities
            router_probs_balanced = router_probs + tf.expand_dims(underused_boost, 0)
            router_probs_balanced = router_probs_balanced / tf.reduce_sum(router_probs_balanced, axis=1, keepdims=True)
            
            router_probs_expanded = tf.expand_dims(router_probs_balanced, axis=-1)
        
        delta_output = tf.reduce_sum(expert_stack * router_probs_expanded, axis=1)'''
    
    content = content.replace(expert_outputs_section, enhanced_expert_outputs)
    
    # Fix 6: Update ensemble strategies for optimal performance
    print("6. Updating ensemble strategies for optimal performance...")
    
    # Update all three strategies for better balance
    old_strategies = '''        # Define different training strategies with EXTREME diversity focus and routing fixes
        ensemble_strategies = [
            {
                "name": "balanced_spike_aware",
                "spike_loss_weights": {"PTS": 1.3, "TRB": 1.2, "AST": 1.15},  # Moderate tail emphasis
                "delta_l2_weight": 0.0001,  # Light L2 for stability
                "student_t_df": 2.0,  # Heavy tails for spikes
                "dropout": 0.08,  # Moderate dropout
                "router_temperature": 0.7,  # BALANCED routing decisions
                "confidence_threshold": 0.6,  # MODERATE confidence target
                "entropy_target": 1.0,  # BALANCED entropy target
                "spike_routing_strength": 2.0,  # MODERATE spike routing
                # Balanced variance and spike focus
                "variance_encouragement_weight": 0.25,  # Moderate variance focus
                "uncertainty_calibration_weight": 0.01,  # Balanced calibration
                "diversity_encouragement": 0.05,  # Moderate diversity
            },
            {
                "name": "balanced_confidence_variance", 
                "spike_loss_weights": {"PTS": 1.4, "TRB": 1.3, "AST": 1.2},  # Moderate tail emphasis
                "delta_l2_weight": 0.0001,  # Very light L2
                "student_t_df": 2.5,  # Heavy tails
                "dropout": 0.08,
                "router_temperature": 0.5,  # Moderate sharpness
                "confidence_threshold": 0.6,  # Moderate confidence target
                "entropy_target": 0.8,  # Moderate entropy target
                "spike_routing_strength": 2.0,  # Strong spike routing
                # Strong variance focus with moderate spikes
                "variance_encouragement_weight": 0.35,  # Strong variance focus
                "uncertainty_calibration_weight": 0.005,  # Light calibration
                "diversity_encouragement": 0.05,  # Moderate diversity
            },
            {
                "name": "stable_confident_routing",
                "spike_loss_weights": {"PTS": 1.6, "TRB": 1.5, "AST": 1.4},  # Moderate tail emphasis
                "delta_l2_weight": 0.0003,  # Light L2 for some stability
                "student_t_df": 3.0,  # Moderate tails
                "dropout": 0.12,
                "router_temperature": 0.7,  # Less sharp but still confident
                "confidence_threshold": 0.55,  # Moderate confidence target
                "entropy_target": 0.9,  # Higher entropy allowed
                "spike_routing_strength": 1.5,  # Moderate spike routing
                # Strong variance focus with stability
                "variance_encouragement_weight": 0.25,  # Strong but controlled variance focus
                "uncertainty_calibration_weight": 0.01,  # Light calibration focus
                "diversity_encouragement": 0.02,  # Light diversity
            }
        ]'''
    
    new_strategies = '''        # Define optimized training strategies for balanced performance
        ensemble_strategies = [
            {
                "name": "confident_spike_aware",
                "spike_loss_weights": {"PTS": 1.4, "TRB": 1.3, "AST": 1.2},  # Moderate tail emphasis
                "delta_l2_weight": 0.0001,  # Light L2 for stability
                "student_t_df": 2.2,  # Heavy tails for spikes
                "dropout": 0.08,  # Moderate dropout
                "router_temperature": 0.5,  # CONFIDENT routing decisions
                "confidence_threshold": 0.55,  # ACHIEVABLE confidence target
                "entropy_target": 0.8,  # GOOD entropy target
                "spike_routing_strength": 1.8,  # STRONG but balanced spike routing
                # Optimized variance and spike focus
                "variance_encouragement_weight": 0.30,  # Strong variance focus
                "uncertainty_calibration_weight": 0.01,  # Balanced calibration
                "diversity_encouragement": 0.05,  # Moderate diversity
            },
            {
                "name": "balanced_performance", 
                "spike_loss_weights": {"PTS": 1.3, "TRB": 1.25, "AST": 1.15},  # Balanced tail emphasis
                "delta_l2_weight": 0.0002,  # Light L2
                "student_t_df": 2.5,  # Heavy tails
                "dropout": 0.10,
                "router_temperature": 0.6,  # Balanced sharpness
                "confidence_threshold": 0.52,  # Moderate confidence target
                "entropy_target": 0.85,  # Balanced entropy target
                "spike_routing_strength": 1.5,  # Balanced spike routing
                # Balanced focus
                "variance_encouragement_weight": 0.25,  # Balanced variance focus
                "uncertainty_calibration_weight": 0.008,  # Light calibration
                "diversity_encouragement": 0.04,  # Light diversity
            },
            {
                "name": "stable_intelligent_routing",
                "spike_loss_weights": {"PTS": 1.5, "TRB": 1.4, "AST": 1.3},  # Moderate tail emphasis
                "delta_l2_weight": 0.0003,  # Light L2 for stability
                "student_t_df": 2.8,  # Moderate tails
                "dropout": 0.12,
                "router_temperature": 0.65,  # Stable but confident
                "confidence_threshold": 0.50,  # Achievable confidence target
                "entropy_target": 0.90,  # Stable entropy
                "spike_routing_strength": 1.3,  # Moderate spike routing
                # Stable focus
                "variance_encouragement_weight": 0.20,  # Controlled variance focus
                "uncertainty_calibration_weight": 0.012,  # Balanced calibration
                "diversity_encouragement": 0.03,  # Light diversity
            }
        ]'''
    
    content = content.replace(old_strategies, new_strategies)
    
    # Fix 7: Add new configuration parameters for optimization
    print("7. Adding optimization configuration parameters...")
    
    # Add expert utilization balancing config
    config_addition = '''            "expert_utilization_balancing": True,  # NEW: Balance expert utilization
            "multi_threshold_routing": True,  # NEW: Multi-level spike routing
            "enhanced_context_detection": True,  # NEW: Enhanced context-aware detection
            '''
    
    # Find a good place to insert the config
    config_insert_point = content.find('"regime_separation_weight": 0.008')
    if config_insert_point != -1:
        # Insert before this line
        content = content[:config_insert_point] + config_addition + content[config_insert_point:]
    
    # Write the optimized file
    with open('hybrid_spike_moe_trainer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Final routing optimization applied!")
    print("\nKey optimizations for perfect balance:")
    print("- Temperature: 0.8 → 0.6 (better confidence)")
    print("- Entropy target: 1.2 → 0.85 (sharper decisions)")
    print("- Multi-threshold routing (20%, 50%, 70% spike confidence levels)")
    print("- Enhanced context-aware spike detection with trends")
    print("- Expert utilization balancing for better gradient flow")
    print("- Optimized ensemble strategies for balanced performance")
    print("- Graduated routing strength based on spike confidence")
    
    return True

if __name__ == "__main__":
    apply_final_routing_optimization()
    print("\n🧪 Run 'python test_routing_fixes.py' to validate the final optimization!")