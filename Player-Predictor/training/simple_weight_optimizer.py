#!/usr/bin/env python3
"""
Simple Weight Optimizer - Directly test weight configurations

This script directly runs the improved_baseline_trainer with different
weight configurations to find optimal settings for peak accuracy.
"""

import subprocess
import json
from pathlib import Path
import sys

def test_configuration(config_name, weights):
    """Test a single configuration by modifying the trainer and running it."""
    print("\n" + "="*70)
    print(f"TESTING: {config_name}")
    print("="*70)
    print("Weights:")
    for k, v in weights.items():
        print(f"  {k}: {v}")
    
    # Create a temporary config file
    config_path = Path("training/temp_config.json")
    with open(config_path, "w") as f:
        json.dump(weights, f, indent=2)
    
    # Run the trainer with this config
    try:
        result = subprocess.run(
            [sys.executable, "training/improved_baseline_trainer.py", "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Parse output for R² score
        output = result.stdout
        r2_macro = None
        for line in output.split('\n'):
            if 'R²_macro:' in line or 'r2_macro=' in line:
                try:
                    r2_macro = float(line.split(':')[-1].strip())
                except:
                    try:
                        r2_macro = float(line.split('=')[-1].strip())
                    except:
                        pass
        
        if r2_macro is None:
            print(f"❌ Could not extract R² from output")
            return None
        
        print(f"\n✅ R² macro: {r2_macro:.4f}")
        return {
            'config_name': config_name,
            'weights': weights,
            'r2_macro': r2_macro
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ Configuration timed out")
        return None
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return None
    finally:
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()


def get_configurations():
    """Return curated configurations to test."""
    return {
        'baseline': {
            'variance_encouragement_weight': 1.0,
            'delta_huber_weight': 0.1,
            'cov_penalty_weight': 0.15,
            'neg_corr_penalty_weight': 0.15,
            'sigma_calibration_weight': 0.05,
            'final_mean_variance_weight': 0.5,
        },
        'high_variance': {
            'variance_encouragement_weight': 2.5,
            'delta_huber_weight': 0.2,
            'cov_penalty_weight': 0.2,
            'neg_corr_penalty_weight': 0.2,
            'sigma_calibration_weight': 0.05,
            'final_mean_variance_weight': 1.0,
        },
        'anti_collapse': {
            'variance_encouragement_weight': 1.5,
            'delta_huber_weight': 0.3,
            'cov_penalty_weight': 0.4,
            'neg_corr_penalty_weight': 0.4,
            'sigma_calibration_weight': 0.05,
            'final_mean_variance_weight': 0.7,
        },
        'ultra_aggressive': {
            'variance_encouragement_weight': 3.0,
            'delta_huber_weight': 0.4,
            'cov_penalty_weight': 0.5,
            'neg_corr_penalty_weight': 0.5,
            'sigma_calibration_weight': 0.1,
            'final_mean_variance_weight': 1.5,
        },
    }


def main():
    """Main entry point."""
    print("="*70)
    print("SIMPLE WEIGHT OPTIMIZER")
    print("="*70)
    print("Testing configurations to find optimal weights for peak accuracy")
    print("="*70)
    
    configs = get_configurations()
    results = []
    
    for config_name, weights in configs.items():
        result = test_configuration(config_name, weights)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    if not results:
        print("No successful results")
        return
    
    # Sort by R²
    results.sort(key=lambda x: x['r2_macro'], reverse=True)
    
    print("\nRanking:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['config_name']}: R² = {result['r2_macro']:.4f}")
    
    # Best configuration
    best = results[0]
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    print(f"Name: {best['config_name']}")
    print(f"R² macro: {best['r2_macro']:.4f}")
    print("\nWeights:")
    for k, v in best['weights'].items():
        print(f"  {k}: {v}")
    
    # Save results
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "simple_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}/simple_optimization_results.json")


if __name__ == "__main__":
    main()
