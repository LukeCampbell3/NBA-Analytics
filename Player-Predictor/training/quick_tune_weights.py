#!/usr/bin/env python3
"""
Quick Loss Weight Tuning - Research-Based Configurations

Tests a curated set of configurations based on research and domain knowledge.
Much faster than full optimization, focuses on known-good combinations.
"""

import numpy as np
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from improved_baseline_trainer import ImprovedBaselineTrainer


class QuickTuner:
    """
    Tests research-based configurations for quick tuning.
    """
    
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_r2 = -np.inf
    
    def get_configurations(self):
        """
        Return curated configurations based on research.
        
        Each config targets a specific failure mode:
        1. Baseline: Current settings
        2. High Variance: Aggressive variance encouragement
        3. Anti-Collapse: Strong delta learning
        4. Balanced: Middle ground
        5. Conservative: Light regularization
        """
        
        configs = {
            'baseline': {
                'name': 'Baseline (Current)',
                'description': 'Current configuration from user request',
                'weights': {
                    'variance_encouragement_weight': 1.0,
                    'delta_huber_weight': 0.1,
                    'cov_penalty_weight': 0.15,
                    'neg_corr_penalty_weight': 0.15,
                    'sigma_calibration_weight': 0.05,
                    'final_mean_variance_weight': 0.5,
                    'mean_loss_weight': 0.08,
                    'sigma_regularization_weight': 0.0001,
                }
            },
            
            'high_variance': {
                'name': 'High Variance Focus',
                'description': 'Aggressive variance encouragement for flat predictions',
                'weights': {
                    'variance_encouragement_weight': 2.5,  # Much higher
                    'delta_huber_weight': 0.2,  # Stronger delta signal
                    'cov_penalty_weight': 0.2,
                    'neg_corr_penalty_weight': 0.2,
                    'sigma_calibration_weight': 0.05,
                    'final_mean_variance_weight': 1.0,  # Stronger floor
                    'mean_loss_weight': 0.05,  # Reduced to not fight variance
                    'sigma_regularization_weight': 0.0001,
                }
            },
            
            'anti_collapse': {
                'name': 'Anti-Collapse',
                'description': 'Prevents delta collapse and baseline cancellation',
                'weights': {
                    'variance_encouragement_weight': 1.5,
                    'delta_huber_weight': 0.3,  # Very strong delta signal
                    'cov_penalty_weight': 0.4,  # Strong cancellation prevention
                    'neg_corr_penalty_weight': 0.4,  # Strong negative corr penalty
                    'sigma_calibration_weight': 0.05,
                    'final_mean_variance_weight': 0.7,
                    'mean_loss_weight': 0.05,  # Light MSE
                    'sigma_regularization_weight': 0.0001,
                }
            },
            
            'balanced': {
                'name': 'Balanced',
                'description': 'Balanced approach between all objectives',
                'weights': {
                    'variance_encouragement_weight': 1.5,
                    'delta_huber_weight': 0.15,
                    'cov_penalty_weight': 0.2,
                    'neg_corr_penalty_weight': 0.2,
                    'sigma_calibration_weight': 0.08,  # Slightly higher
                    'final_mean_variance_weight': 0.6,
                    'mean_loss_weight': 0.06,
                    'sigma_regularization_weight': 0.0005,
                }
            },
            
            'conservative': {
                'name': 'Conservative',
                'description': 'Light regularization, let NLL dominate',
                'weights': {
                    'variance_encouragement_weight': 0.8,  # Lighter
                    'delta_huber_weight': 0.08,
                    'cov_penalty_weight': 0.1,
                    'neg_corr_penalty_weight': 0.1,
                    'sigma_calibration_weight': 0.03,
                    'final_mean_variance_weight': 0.3,
                    'mean_loss_weight': 0.05,
                    'sigma_regularization_weight': 0.0001,
                }
            },
            
            'ultra_aggressive': {
                'name': 'Ultra Aggressive',
                'description': 'Maximum variance encouragement for stubborn collapse',
                'weights': {
                    'variance_encouragement_weight': 3.0,  # Maximum
                    'delta_huber_weight': 0.4,  # Very strong
                    'cov_penalty_weight': 0.5,  # Maximum
                    'neg_corr_penalty_weight': 0.5,  # Maximum
                    'sigma_calibration_weight': 0.1,
                    'final_mean_variance_weight': 1.5,  # Very strong floor
                    'mean_loss_weight': 0.03,  # Minimal MSE
                    'sigma_regularization_weight': 0.0001,
                }
            },
            
            'nll_focused': {
                'name': 'NLL Focused',
                'description': 'Minimal regularization, trust the likelihood',
                'weights': {
                    'variance_encouragement_weight': 0.5,  # Light
                    'delta_huber_weight': 0.05,  # Light
                    'cov_penalty_weight': 0.05,  # Light
                    'neg_corr_penalty_weight': 0.05,  # Light
                    'sigma_calibration_weight': 0.02,  # Light
                    'final_mean_variance_weight': 0.2,  # Light
                    'mean_loss_weight': 0.03,  # Light
                    'sigma_regularization_weight': 0.0001,
                }
            },
            
            'calibration_focused': {
                'name': 'Calibration Focused',
                'description': 'Emphasize uncertainty calibration',
                'weights': {
                    'variance_encouragement_weight': 1.0,
                    'delta_huber_weight': 0.1,
                    'cov_penalty_weight': 0.15,
                    'neg_corr_penalty_weight': 0.15,
                    'sigma_calibration_weight': 0.15,  # Much higher
                    'final_mean_variance_weight': 0.5,
                    'mean_loss_weight': 0.08,
                    'sigma_regularization_weight': 0.001,  # Slightly higher
                }
            },
        }
        
        return configs
    
    def test_configuration(self, config_name, config_data):
        """Test a single configuration."""
        print("\n" + "="*70)
        print(f"TESTING: {config_data['name']}")
        print("="*70)
        print(f"Description: {config_data['description']}")
        print("\nWeights:")
        for k, v in config_data['weights'].items():
            print(f"  {k}: {v}")
        
        try:
            # Create trainer
            trainer = ImprovedBaselineTrainer(ensemble_size=1)
            
            # Update weights
            trainer.config.update(config_data['weights'])
            
            # Quick training for comparison
            trainer.config['phase1_epochs'] = 35  # Reasonable for comparison
            trainer.config['phase2_epochs'] = 20
            trainer.config['patience'] = 12
            
            # Train
            model, metadata = trainer.train()
            
            # Extract metrics
            r2_macro = metadata['final_performance']['r2_macro']
            mae = metadata['final_performance']['validation_mae']
            
            result = {
                'config_name': config_name,
                'name': config_data['name'],
                'description': config_data['description'],
                'weights': config_data['weights'].copy(),
                'r2_macro': r2_macro,
                'mae': mae,
            }
            
            self.results.append(result)
            
            # Update best
            if r2_macro > self.best_r2:
                self.best_r2 = r2_macro
                self.best_config = config_name
                print(f"\n🎯 NEW BEST: {config_data['name']}")
                print(f"   R² = {r2_macro:.4f}, MAE = {mae:.3f}")
            
            print(f"\nResult: R² = {r2_macro:.4f}, MAE = {mae:.3f}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Configuration failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self, configs_to_test=None):
        """Run quick tuning."""
        print("="*70)
        print("QUICK LOSS WEIGHT TUNING")
        print("="*70)
        print("Testing research-based configurations for optimal accuracy")
        print("="*70)
        
        configs = self.get_configurations()
        
        # Test specific configs or all
        if configs_to_test:
            configs = {k: v for k, v in configs.items() if k in configs_to_test}
        
        print(f"\nTesting {len(configs)} configurations...")
        
        for config_name, config_data in configs.items():
            result = self.test_configuration(config_name, config_data)
            
            # Save intermediate results
            self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.best_config
    
    def _save_results(self):
        """Save results to file."""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save all results
        with open(output_dir / "quick_tune_results.json", "w") as f:
            json.dump({
                'results': self.results,
                'best_config': self.best_config,
                'best_r2': self.best_r2
            }, f, indent=2)
    
    def _print_summary(self):
        """Print summary of results."""
        print("\n" + "="*70)
        print("QUICK TUNING SUMMARY")
        print("="*70)
        
        if not self.results:
            print("No successful results")
            return
        
        # Sort by R²
        sorted_results = sorted(self.results, key=lambda x: x['r2_macro'], reverse=True)
        
        print("\nRanking (by R² macro):")
        print("-" * 70)
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result['name']}")
            print(f"   R² = {result['r2_macro']:.4f}, MAE = {result['mae']:.3f}")
            print(f"   {result['description']}")
            print()
        
        # Best configuration details
        best = sorted_results[0]
        print("="*70)
        print("BEST CONFIGURATION")
        print("="*70)
        print(f"Name: {best['name']}")
        print(f"R² macro: {best['r2_macro']:.4f}")
        print(f"MAE: {best['mae']:.3f}")
        print("\nWeights to use:")
        for k, v in best['weights'].items():
            print(f"  {k}: {v}")
        
        print("\n" + "="*70)
        print("To apply this configuration, update improved_baseline_trainer.py")
        print("with the weights shown above.")
        print("="*70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick loss weight tuning")
    parser.add_argument('--configs', nargs='+', 
                       choices=['baseline', 'high_variance', 'anti_collapse', 
                               'balanced', 'conservative', 'ultra_aggressive',
                               'nll_focused', 'calibration_focused'],
                       help='Specific configurations to test (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available configurations')
    
    args = parser.parse_args()
    
    tuner = QuickTuner()
    
    if args.list:
        print("Available configurations:")
        configs = tuner.get_configurations()
        for name, data in configs.items():
            print(f"\n{name}:")
            print(f"  Name: {data['name']}")
            print(f"  Description: {data['description']}")
        return
    
    # Run tuning
    best = tuner.run(configs_to_test=args.configs)
    
    print(f"\n✅ Best configuration: {best}")
    print(f"   Results saved to optimization_results/quick_tune_results.json")


if __name__ == "__main__":
    main()
