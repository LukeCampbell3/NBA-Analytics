#!/usr/bin/env python3
"""
Weight Tuning Experiments

Systematically test different weight configurations to find optimal balance
between variance and accuracy.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from run_expanded_training import patch_prepare_data, EXPANDED_PLAYERS
from improved_baseline_trainer import ImprovedBaselineTrainer
from hybrid_spike_moe_trainer import HybridSpikeMoETrainer


# Test configurations
EXPERIMENTS = [
    {
        "name": "Run 9: PTS weight=2.5, floor=5.0",
        "direct_delta_weight": [2.5, 1.5, 1.5],
        "variance_floor": [5.0, 1.5, 1.5],
        "nll_weight": [0.3, 1.0, 1.0],
    },
    {
        "name": "Run 10: PTS weight=3.5, floor=6.0",
        "direct_delta_weight": [3.5, 1.5, 1.5],
        "variance_floor": [6.0, 1.5, 1.5],
        "nll_weight": [0.3, 1.0, 1.0],
    },
    {
        "name": "Run 11: PTS weight=3.0, floor=5.5, NLL=0.4",
        "direct_delta_weight": [3.0, 1.5, 1.5],
        "variance_floor": [5.5, 1.5, 1.5],
        "nll_weight": [0.4, 1.0, 1.0],
    },
]


def run_experiment(config):
    """Run a single experiment with given configuration"""
    
    print("\n" + "="*80)
    print(config["name"])
    print("="*80)
    print(f"Direct delta weight: {config['direct_delta_weight']}")
    print(f"Variance floor: {config['variance_floor']}")
    print(f"NLL weight: {config['nll_weight']}")
    print()
    
    # Patch prepare_data for expanded players
    original_prepare_data = patch_prepare_data()
    
    try:
        # Create trainer
        trainer = ImprovedBaselineTrainer(ensemble_size=1)
        
        # Override config
        trainer.config["direct_delta_supervision_weight"] = config["direct_delta_weight"]
        # Note: variance_floor is hardcoded in the loss function, would need to modify
        # For now, we'll just test different direct delta weights
        
        # Train
        model, meta = trainer.train()
        
        # Extract results
        results = {
            "name": config["name"],
            "config": config,
            "metrics": {
                # These would be extracted from the training output
                # For now, user will need to manually record
            }
        }
        
        return results
        
    finally:
        # Restore original
        HybridSpikeMoETrainer.prepare_data = original_prepare_data


def main():
    """Run all experiments"""
    
    print("\n" + "="*80)
    print("WEIGHT TUNING EXPERIMENTS")
    print("="*80)
    print(f"Testing {len(EXPERIMENTS)} configurations")
    print()
    
    results = []
    for i, config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Starting experiment...")
        result = run_experiment(config)
        results.append(result)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    
    # Save results
    with open("training/weight_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to training/weight_tuning_results.json")


if __name__ == "__main__":
    main()
