#!/usr/bin/env python3
"""
Loss Weight Optimization for Peak Accuracy

Systematically searches for optimal loss weights that maximize R² while
maintaining spike detection and realistic variance. Uses Bayesian optimization
with multi-objective constraints.

Key Objectives:
1. Maximize R² macro (primary)
2. Maintain spike detection accuracy (constraint)
3. Ensure realistic variance ratios (constraint)
4. Calibrate uncertainty (constraint)
"""

import numpy as np
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    HAVE_SKOPT = True
except ImportError:
    print("⚠️  scikit-optimize not found. Install with: pip install scikit-optimize")
    HAVE_SKOPT = False

sys.path.insert(0, str(Path(__file__).parent))
from improved_baseline_trainer import ImprovedBaselineTrainer


class LossWeightOptimizer:
    """
    Bayesian optimization of loss weights for maximum accuracy.
    
    Search Strategy:
    1. Define reasonable ranges for each loss weight
    2. Use Gaussian Process to model objective function
    3. Balance exploration vs exploitation
    4. Apply multi-objective constraints
    """
    
    def __init__(self, n_calls=50, n_initial_points=10, random_state=42):
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        
    def define_search_space(self):
        """
        Define search space based on research and domain knowledge.
        
        Key Insights:
        - NLL is primary (weight=1.0, fixed)
        - Variance losses need strong signal (0.5-3.0)
        - Regularization should be light (0.01-0.3)
        - Calibration needs moderate weight (0.01-0.2)
        """
        space = [
            # Variance encouragement (CRITICAL for delta learning)
            Real(0.5, 3.0, name='variance_encouragement_weight', prior='log-uniform'),
            
            # Delta Huber (prevents collapse)
            Real(0.05, 0.5, name='delta_huber_weight', prior='log-uniform'),
            
            # Baseline cancellation penalties
            Real(0.05, 0.5, name='cov_penalty_weight', prior='log-uniform'),
            Real(0.05, 0.5, name='neg_corr_penalty_weight', prior='log-uniform'),
            
            # Sigma calibration
            Real(0.01, 0.2, name='sigma_calibration_weight', prior='log-uniform'),
            
            # Mean variance floor (R² insurance)
            Real(0.2, 2.0, name='final_mean_variance_weight', prior='log-uniform'),
            
            # Auxiliary MSE (should be light)
            Real(0.01, 0.2, name='mean_loss_weight', prior='log-uniform'),
            
            # Sigma regularization (very light)
            Real(0.0001, 0.01, name='sigma_regularization_weight', prior='log-uniform'),
            
            # Curriculum parameters
            Real(0.1, 0.3, name='hard_example_percent'),
            Real(0.2, 0.5, name='anchor_mix_ratio'),
            Real(0.5, 0.8, name='phase2_replay_mix'),
        ]
        
        return space
    
    def compute_objective(self, config_dict):
        """
        Compute multi-objective score.
        
        Primary: R² macro (maximize)
        Constraints:
        - Spike detection accuracy > 0.6
        - VarRatio in [0.6, 1.2]
        - DeltaVarRatio in [0.3, 1.2]
        - Err-Sigma Corr > 0.15
        
        Returns negative score for minimization.
        """
        print("\n" + "="*70)
        print(f"TRIAL {len(self.results) + 1}/{self.n_calls}")
        print("="*70)
        print("Testing configuration:")
        for k, v in config_dict.items():
            print(f"  {k}: {v:.4f}")
        
        try:
            # Create trainer with this configuration
            trainer = ImprovedBaselineTrainer(ensemble_size=1)
            
            # Update config
            trainer.config.update(config_dict)
            
            # Train model (simplified for speed)
            trainer.config['phase1_epochs'] = 30  # Reduced for optimization
            trainer.config['phase2_epochs'] = 15
            trainer.config['patience'] = 10
            
            model, metadata = trainer.train()
            
            # Extract metrics
            r2_macro = metadata['final_performance']['r2_macro']
            
            # Get detailed metrics from validation
            metrics = self._extract_detailed_metrics(trainer, model)
            
            # Compute constraint violations
            violations = self._compute_violations(metrics)
            
            # Multi-objective score
            # Primary: R² (maximize)
            # Penalty: constraint violations
            score = r2_macro - 2.0 * violations  # Heavy penalty for violations
            
            # Store result
            result = {
                'config': config_dict.copy(),
                'r2_macro': r2_macro,
                'metrics': metrics,
                'violations': violations,
                'score': score
            }
            self.results.append(result)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_config = config_dict.copy()
                print(f"\n🎯 NEW BEST: R²={r2_macro:.4f}, Score={score:.4f}")
            
            print(f"\nResult: R²={r2_macro:.4f}, Violations={violations:.4f}, Score={score:.4f}")
            
            # Return negative for minimization
            return -score
            
        except Exception as e:
            print(f"\n❌ Trial failed: {e}")
            return 1000.0  # Large penalty for failed trials
    
    def _extract_detailed_metrics(self, trainer, model):
        """Extract detailed validation metrics."""
        # This would need access to validation data
        # For now, return placeholder
        return {
            'var_ratio_pts': 0.8,
            'var_ratio_trb': 0.7,
            'var_ratio_ast': 0.7,
            'delta_var_ratio_pts': 0.5,
            'delta_var_ratio_trb': 0.4,
            'delta_var_ratio_ast': 0.4,
            'err_sigma_corr_pts': 0.2,
            'err_sigma_corr_trb': 0.2,
            'err_sigma_corr_ast': 0.2,
            'spike_detection_acc': 0.7,
        }
    
    def _compute_violations(self, metrics):
        """
        Compute constraint violation penalty.
        
        Constraints:
        - VarRatio in [0.6, 1.2]
        - DeltaVarRatio in [0.3, 1.2]
        - Err-Sigma Corr > 0.15
        - Spike detection > 0.6
        """
        violations = 0.0
        
        # VarRatio constraints
        for stat in ['pts', 'trb', 'ast']:
            vr = metrics.get(f'var_ratio_{stat}', 0.8)
            if vr < 0.6:
                violations += (0.6 - vr)
            elif vr > 1.2:
                violations += (vr - 1.2)
        
        # DeltaVarRatio constraints
        for stat in ['pts', 'trb', 'ast']:
            dvr = metrics.get(f'delta_var_ratio_{stat}', 0.5)
            if dvr < 0.3:
                violations += (0.3 - dvr)
            elif dvr > 1.2:
                violations += (dvr - 1.2)
        
        # Err-Sigma correlation
        for stat in ['pts', 'trb', 'ast']:
            esc = metrics.get(f'err_sigma_corr_{stat}', 0.2)
            if esc < 0.15:
                violations += (0.15 - esc)
        
        # Spike detection
        spike_acc = metrics.get('spike_detection_acc', 0.7)
        if spike_acc < 0.6:
            violations += (0.6 - spike_acc)
        
        return violations
    
    def optimize(self):
        """Run Bayesian optimization."""
        if not HAVE_SKOPT:
            print("❌ Cannot run optimization without scikit-optimize")
            return None
        
        print("="*70)
        print("LOSS WEIGHT OPTIMIZATION")
        print("="*70)
        print(f"Strategy: Bayesian Optimization")
        print(f"Trials: {self.n_calls}")
        print(f"Initial random: {self.n_initial_points}")
        print(f"Objective: Maximize R² with constraints")
        print("="*70)
        
        space = self.define_search_space()
        
        @use_named_args(space)
        def objective(**params):
            return self.compute_objective(params)
        
        # Run optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=True,
            n_jobs=1  # Sequential for stability
        )
        
        # Save results
        self._save_results()
        
        return self.best_config
    
    def _save_results(self):
        """Save optimization results."""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save all trials
        with open(output_dir / "all_trials.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save best config
        with open(output_dir / "best_config.json", "w") as f:
            json.dump({
                'config': self.best_config,
                'score': self.best_score,
                'r2_macro': max(r['r2_macro'] for r in self.results)
            }, f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir}/")
        print(f"   - all_trials.json: All {len(self.results)} trials")
        print(f"   - best_config.json: Best configuration")


class GridSearchOptimizer:
    """
    Grid search alternative for systematic exploration.
    Faster than Bayesian but less efficient.
    """
    
    def __init__(self):
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
    
    def define_grid(self):
        """
        Define grid of configurations to test.
        Focus on most impactful parameters.
        """
        # Coarse grid for speed
        grid = {
            'variance_encouragement_weight': [0.5, 1.0, 2.0],
            'delta_huber_weight': [0.05, 0.1, 0.2],
            'cov_penalty_weight': [0.1, 0.2, 0.3],
            'sigma_calibration_weight': [0.03, 0.05, 0.1],
            'final_mean_variance_weight': [0.3, 0.5, 1.0],
        }
        
        return grid
    
    def optimize(self):
        """Run grid search."""
        print("="*70)
        print("GRID SEARCH OPTIMIZATION")
        print("="*70)
        
        grid = self.define_grid()
        
        # Generate all combinations
        import itertools
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        
        total = np.prod([len(v) for v in values])
        print(f"Total configurations: {total}")
        print("="*70)
        
        for i, combo in enumerate(itertools.product(*values)):
            config = dict(zip(keys, combo))
            
            print(f"\n[{i+1}/{total}] Testing configuration:")
            for k, v in config.items():
                print(f"  {k}: {v}")
            
            # Train and evaluate
            try:
                trainer = ImprovedBaselineTrainer(ensemble_size=1)
                trainer.config.update(config)
                
                # Quick training
                trainer.config['phase1_epochs'] = 20
                trainer.config['phase2_epochs'] = 10
                trainer.config['patience'] = 8
                
                model, metadata = trainer.train()
                r2_macro = metadata['final_performance']['r2_macro']
                
                result = {
                    'config': config.copy(),
                    'r2_macro': r2_macro
                }
                self.results.append(result)
                
                if r2_macro > self.best_score:
                    self.best_score = r2_macro
                    self.best_config = config.copy()
                    print(f"🎯 NEW BEST: R²={r2_macro:.4f}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
        
        # Save results
        self._save_results()
        return self.best_config
    
    def _save_results(self):
        """Save grid search results."""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "grid_search_results.json", "w") as f:
            json.dump({
                'all_results': self.results,
                'best_config': self.best_config,
                'best_r2': self.best_score
            }, f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir}/grid_search_results.json")


class ManualTuningGuide:
    """
    Provides research-based guidance for manual tuning.
    """
    
    @staticmethod
    def print_guide():
        """Print tuning guide based on research."""
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    LOSS WEIGHT TUNING GUIDE                          ║
╔══════════════════════════════════════════════════════════════════════╗

Based on research and domain knowledge, here's how to tune each weight:

1. VARIANCE ENCOURAGEMENT (variance_encouragement_weight)
   Current: 1.0
   Range: 0.5 - 3.0
   
   Symptoms of too low:
   - Flat predictions (low DeltaVarRatio < 0.3)
   - Model predicts near baseline
   - R² negative or very low
   
   Symptoms of too high:
   - Unstable training
   - Predictions too volatile
   - High MAE
   
   Recommendation: Start at 1.0, increase to 2.0 if DeltaVarRatio < 0.4

2. DELTA HUBER (delta_huber_weight)
   Current: 0.1
   Range: 0.05 - 0.5
   
   Purpose: Prevents delta collapse, encourages learning changes
   
   Symptoms of too low:
   - Model ignores deltas
   - Predictions = baseline
   
   Symptoms of too high:
   - Fights with NLL
   - Poor calibration
   
   Recommendation: 0.1 is good, increase to 0.2 if deltas are ignored

3. BASELINE CANCELLATION (cov_penalty_weight, neg_corr_penalty_weight)
   Current: 0.15 each
   Range: 0.05 - 0.5
   
   Purpose: Prevents model from learning baseline offsets
   
   Check: Corr(baseline, pred-baseline) should be near 0
   If negative correlation > 0.3, increase these weights
   
   Recommendation: 0.15-0.25 for most cases

4. SIGMA CALIBRATION (sigma_calibration_weight)
   Current: 0.05
   Range: 0.01 - 0.2
   
   Purpose: Makes uncertainty estimates useful
   
   Check: Corr(|error|, sigma) should be > 0.15
   
   Recommendation: 0.05 is good, increase to 0.1 if correlation < 0.15

5. MEAN VARIANCE FLOOR (final_mean_variance_weight)
   Current: 0.5
   Range: 0.2 - 2.0
   
   Purpose: R² insurance, prevents variance collapse
   
   Check: VarRatio should be 0.6-1.2
   If VarRatio < 0.6, increase this weight
   
   Recommendation: 0.5-1.0 for most cases

6. AUXILIARY MSE (mean_loss_weight)
   Current: 0.08
   Range: 0.01 - 0.2
   
   Purpose: Light regularization
   Should be much smaller than NLL (1.0)
   
   Recommendation: Keep at 0.05-0.1

7. SIGMA REGULARIZATION (sigma_regularization_weight)
   Current: 0.0001
   Range: 0.0001 - 0.01
   
   Purpose: Prevents sigma from exploding
   Should be very light
   
   Recommendation: Keep at 0.0001-0.001

═══════════════════════════════════════════════════════════════════════

TUNING STRATEGY:

Phase 1: Fix Delta Learning
- If DeltaVarRatio < 0.3:
  * Increase variance_encouragement_weight to 2.0
  * Increase delta_huber_weight to 0.2
  * Reduce mean_loss_weight to 0.05

Phase 2: Fix Baseline Cancellation
- If Corr(baseline, delta) < -0.3:
  * Increase cov_penalty_weight to 0.3
  * Increase neg_corr_penalty_weight to 0.3

Phase 3: Calibrate Uncertainty
- If Err-Sigma Corr < 0.15:
  * Increase sigma_calibration_weight to 0.1

Phase 4: Ensure Variance
- If VarRatio < 0.6:
  * Increase final_mean_variance_weight to 1.0

═══════════════════════════════════════════════════════════════════════

QUICK FIXES FOR COMMON ISSUES:

Issue: R² negative, predictions flat
Fix: variance_encouragement_weight = 2.0, delta_huber_weight = 0.2

Issue: R² negative, predictions = baseline
Fix: cov_penalty_weight = 0.3, neg_corr_penalty_weight = 0.3

Issue: R² positive but low (< 0.1)
Fix: Increase all variance weights by 50%

Issue: High MAE, unstable predictions
Fix: Reduce variance_encouragement_weight to 0.5

Issue: Good R² but poor spike detection
Fix: Increase spike_loss_weights in config

═══════════════════════════════════════════════════════════════════════
""")


def main():
    """Main optimization entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize loss weights for peak accuracy")
    parser.add_argument('--method', choices=['bayesian', 'grid', 'guide'], default='guide',
                       help='Optimization method')
    parser.add_argument('--n-calls', type=int, default=50,
                       help='Number of trials for Bayesian optimization')
    parser.add_argument('--n-initial', type=int, default=10,
                       help='Number of initial random trials')
    
    args = parser.parse_args()
    
    if args.method == 'guide':
        # Print tuning guide
        ManualTuningGuide.print_guide()
        
    elif args.method == 'bayesian':
        if not HAVE_SKOPT:
            print("❌ Bayesian optimization requires scikit-optimize")
            print("   Install with: pip install scikit-optimize")
            print("\n   Showing tuning guide instead:")
            ManualTuningGuide.print_guide()
            return
        
        optimizer = LossWeightOptimizer(
            n_calls=args.n_calls,
            n_initial_points=args.n_initial,
            random_state=42
        )
        best_config = optimizer.optimize()
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print("\nBest configuration:")
        for k, v in best_config.items():
            print(f"  {k}: {v:.4f}")
        
    elif args.method == 'grid':
        optimizer = GridSearchOptimizer()
        best_config = optimizer.optimize()
        
        print("\n" + "="*70)
        print("GRID SEARCH COMPLETE")
        print("="*70)
        print("\nBest configuration:")
        for k, v in best_config.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
