#!/usr/bin/env python3
"""
Simple Meta-State MoE Trainer
Complete standalone implementation - just run this file!

Usage:
    python training/meta_state_trainer_simple.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from improved_baseline_trainer import ImprovedBaselineTrainer
from meta_state_moe_trainer_core import build_meta_state_moe
from meta_state_losses import (
    create_meta_state_loss,
    create_delta_variance_loss,
    create_baseline_correlation_penalty
)
import tensorflow as tf


class MetaStateTrainer(ImprovedBaselineTrainer):
    """
    Meta-State MoE Trainer
    
    Inherits from ImprovedBaselineTrainer but uses:
    1. Meta-state head z (reason encoder)
    2. Event prediction
    3. Event-gated spike experts
    4. Separated epistemic/aleatoric uncertainty
    5. Delta-focused loss functions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add meta-state specific config
        self.config.update({
            # Meta-state architecture
            "z_dim": 48,
            "num_events": 4,
            "use_meta_state": True,
            
            # Loss weights (delta-focused)
            "delta_huber_weight": 0.5,
            "nll_weight": 1.0,
            "event_loss_weight": 0.1,
            "epistemic_weight": 0.01,
            "calibration_weight": 0.05,
            "mean_loss_weight": 0.05,
            
            # Expert regularization
            "expert_usage_weight": 0.1,
            "gate_penalty_weight": 0.05,
            "gate_mean_target": 0.2,
            
            # Reduce sigma reg (avoid scale cheating)
            "sigma_regularization_weight": 0.0001,
            
            # Keep variance encouragement
            "variance_encouragement_weight": 3.5,
            "cov_penalty_weight": 0.25,
            "neg_corr_penalty_weight": 1.0,
        })
        
        print("\n" + "="*70)
        print("META-STATE MOE TRAINER")
        print("="*70)
        print(f"z_dim: {self.config['z_dim']}")
        print(f"num_events: {self.config['num_events']}")
        print(f"delta_huber_weight: {self.config['delta_huber_weight']}")
        print(f"gate_penalty_weight: {self.config['gate_penalty_weight']}")
        print("="*70 + "\n")
    
    def build_model(self):
        """Build meta-state MoE architecture"""
        print("🏗️ Building Meta-State MoE...")
        
        model = build_meta_state_moe(
            config=self.config,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            player_mapping=self.player_mapping,
            team_mapping=self.team_mapping,
            opponent_mapping=self.opponent_mapping
        )
        
        print(f"✅ Model built: {model.count_params():,} parameters")
        return model
    
    def create_enhanced_loss(self):
        """Create meta-state loss function with compatibility wrapper"""
        print("📊 Creating meta-state loss functions...")
        
        # Import meta-state losses
        from meta_state_losses import (
            create_meta_state_loss,
            create_delta_variance_loss,
            create_baseline_correlation_penalty
        )
        
        # Main loss
        main_loss = create_meta_state_loss(self.config, self.target_columns)
        
        # Variance loss
        var_loss = create_delta_variance_loss(self.config, self.target_columns)
        
        # Baseline correlation penalty
        cov_loss = create_baseline_correlation_penalty(self.config, self.target_columns)
        
        def combined_loss(y_true, y_pred, baselines=None, bucket_ids=None, event_true=None):
            """
            Combined loss function with compatibility wrapper
            
            y_pred structure from meta-state model: [delta (3), sigma_ale (3), u_epi (3), event_logits (4), gate (1)]
            Total: 14 outputs
            
            Original trainer expects: [delta (3), sigma (3)] = 6 outputs (when predict_delta=True)
            
            Both output deltas, so they're compatible!
            """
            
            # Check if this is meta-state output (14-dim) or original output (6-dim)
            n = len(self.target_columns)
            
            if y_pred.shape[-1] == 14:
                # Meta-state model output
                # Extract delta and sigma for compatibility
                delta_pred = y_pred[:, :n]
                sigma_ale = y_pred[:, n:2*n]
                
                # Create compatible y_pred for variance/cov losses (they expect delta + sigma)
                y_pred_compat = tf.concat([delta_pred, sigma_ale], axis=1)
                
                # Main loss (uses full meta-state output)
                loss = main_loss(y_true, y_pred, baselines, event_true)
                
            else:
                # Original model output (6-dim) - fallback
                y_pred_compat = y_pred
                
                # Use simplified loss for original model
                delta_pred = y_pred[:, :n]
                sigma = y_pred[:, n:2*n]
                
                # Reconstruct mu for NLL
                mu = baselines + delta_pred if baselines is not None else delta_pred
                
                # Delta Huber
                if baselines is not None:
                    delta_true = y_true - baselines
                    delta_loss = tf.keras.losses.huber(delta_true, delta_pred, delta=2.0)
                else:
                    delta_loss = 0.0
                
                # Student-t NLL
                df = 4.0
                resid = (y_true - mu) / (sigma + 1e-6)
                nll = -tf.reduce_mean(
                    tf.math.lgamma((df + 1) / 2) - tf.math.lgamma(df / 2) -
                    0.5 * tf.math.log(df * 3.14159) - tf.math.log(sigma + 1e-6) -
                    ((df + 1) / 2) * tf.math.log(1 + tf.square(resid) / df)
                )
                
                loss = 0.5 * delta_loss + 1.0 * nll
            
            # Variance encouragement (if bucket_ids provided)
            if baselines is not None and bucket_ids is not None:
                var_term = var_loss(y_true, y_pred_compat, baselines, bucket_ids)
                loss += self.config["variance_encouragement_weight"] * var_term
            
            # Baseline correlation penalty
            if baselines is not None:
                cov_term = cov_loss(y_true, y_pred_compat, baselines)
                loss += self.config["cov_penalty_weight"] * cov_term
            
            return loss
        
        return combined_loss
    
    def _make_mae_metric(self):
        """MAE metric for meta-state output"""
        def mae_metric(y_true, y_pred, baselines=None):
            # Extract delta from y_pred
            # Meta-state: 14-dim [delta, sigma_ale, u_epi, events, gate]
            # Original: 6-dim [delta, sigma] (when predict_delta=True)
            n = len(self.target_columns)
            
            # Both models output delta (first n values)
            delta_pred = y_pred[:, :n]
            
            # Reconstruct absolute prediction
            if baselines is not None:
                mu = baselines + delta_pred
            else:
                mu = delta_pred
            
            return tf.reduce_mean(tf.abs(y_true - mu))
        return mae_metric


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("STARTING META-STATE MOE TRAINING")
    print("="*70)
    
    # Create trainer
    trainer = MetaStateTrainer(ensemble_size=1)
    
    # Train model
    model, meta = trainer.train()
    
    # Print results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"R²_macro: {meta['final_performance']['r2_macro']:.3f}")
    print(f"MAE: {meta['final_performance']['validation_mae']:.3f}")
    print("\nModel saved to:")
    print("  model/improved_baseline_final.weights.h5")
    print("  model/improved_baseline_scaler.pkl")
    print("  model/improved_baseline_metadata.json")
    print("="*70)
    
    return model, meta


if __name__ == "__main__":
    main()
