#!/usr/bin/env python3
"""
Diagnostic-Driven Tuning

Analyzes current model performance and suggests specific weight adjustments
based on observed failure modes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from improved_baseline_trainer import ImprovedBaselineTrainer


class ModelDiagnostics:
    """
    Comprehensive model diagnostics to identify failure modes.
    """
    
    def __init__(self, trainer, model, X_val, baselines_val, y_val):
        self.trainer = trainer
        self.model = model
        self.X_val = X_val
        self.baselines_val = baselines_val
        self.y_val = y_val
        self.diagnostics = {}
    
    def run_full_diagnostics(self):
        """Run all diagnostic checks."""
        print("="*70)
        print("RUNNING COMPREHENSIVE DIAGNOSTICS")
        print("="*70)
        
        # Get predictions
        preds = self.model.net.predict([self.X_val, self.baselines_val], verbose=0)
        
        if self.trainer.config.get("predict_delta", False):
            delta_means = preds[:, :3]
            pred_means = self.baselines_val + delta_means
        else:
            pred_means = preds[:, :3]
        
        pred_sigma = preds[:, 3:6]
        
        # Run diagnostics
        self._check_r2(pred_means)
        self._check_variance_ratios(pred_means)
        self._check_delta_variance(pred_means)
        self._check_baseline_correlation(pred_means)
        self._check_sigma_calibration(pred_means, pred_sigma)
        self._check_spike_detection(pred_means)
        self._check_trajectory_realism(pred_means)
        
        # Analyze and suggest
        self._analyze_failure_modes()
        self._suggest_weight_adjustments()
        
        return self.diagnostics
    
    def _check_r2(self, pred_means):
        """Check R² scores."""
        print("\n1. R² ANALYSIS")
        print("-" * 70)
        
        stats = ["PTS", "TRB", "AST"]
        r2_scores = []
        
        for i, stat in enumerate(stats):
            yt = self.y_val[:, i]
            yp = pred_means[:, i]
            
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            r2_scores.append(r2)
            
            print(f"{stat}: R² = {r2:.4f}")
        
        r2_macro = np.mean(r2_scores)
        print(f"\nMacro R²: {r2_macro:.4f}")
        
        self.diagnostics['r2_scores'] = r2_scores
        self.diagnostics['r2_macro'] = r2_macro
        
        # Diagnosis
        if r2_macro < 0:
            print("❌ CRITICAL: Negative R² - model worse than baseline")
        elif r2_macro < 0.1:
            print("⚠️  WARNING: Very low R² - significant issues")
        elif r2_macro < 0.3:
            print("⚠️  Low R² - needs improvement")
        else:
            print("✓ Acceptable R²")
    
    def _check_variance_ratios(self, pred_means):
        """Check variance ratios."""
        print("\n2. VARIANCE RATIO ANALYSIS")
        print("-" * 70)
        
        stats = ["PTS", "TRB", "AST"]
        var_ratios = []
        
        for i, stat in enumerate(stats):
            yt = self.y_val[:, i]
            yp = pred_means[:, i]
            
            var_pred = np.var(yp)
            var_true = np.var(yt)
            ratio = var_pred / var_true
            var_ratios.append(ratio)
            
            status = "✓" if 0.6 <= ratio <= 1.2 else "❌"
            print(f"{stat}: {ratio:.3f} {status} (target: 0.6-1.2)")
        
        self.diagnostics['var_ratios'] = var_ratios
        
        # Diagnosis
        if all(r < 0.6 for r in var_ratios):
            print("\n❌ CRITICAL: All stats under-predicting variance")
            print("   → Increase variance_encouragement_weight")
        elif any(r < 0.6 for r in var_ratios):
            print("\n⚠️  WARNING: Some stats under-predicting variance")
    
    def _check_delta_variance(self, pred_means):
        """Check delta variance ratios."""
        print("\n3. DELTA VARIANCE ANALYSIS")
        print("-" * 70)
        
        stats = ["PTS", "TRB", "AST"]
        delta_var_ratios = []
        
        for i, stat in enumerate(stats):
            yt = self.y_val[:, i]
            yp = pred_means[:, i]
            bs = self.baselines_val[:, i]
            
            delta_pred = yp - bs
            delta_true = yt - bs
            
            var_delta_pred = np.var(delta_pred)
            var_delta_true = np.var(delta_true)
            ratio = var_delta_pred / var_delta_true
            delta_var_ratios.append(ratio)
            
            status = "✓" if 0.3 <= ratio <= 1.2 else "❌"
            print(f"{stat}: {ratio:.3f} {status} (target: 0.3-1.2)")
        
        self.diagnostics['delta_var_ratios'] = delta_var_ratios
        
        # Diagnosis
        if all(r < 0.3 for r in delta_var_ratios):
            print("\n❌ CRITICAL: Delta collapse - model not learning changes")
            print("   → Increase variance_encouragement_weight to 2.0+")
            print("   → Increase delta_huber_weight to 0.2+")
        elif any(r < 0.3 for r in delta_var_ratios):
            print("\n⚠️  WARNING: Some stats showing delta collapse")
    
    def _check_baseline_correlation(self, pred_means):
        """Check for baseline cancellation."""
        print("\n4. BASELINE CORRELATION ANALYSIS")
        print("-" * 70)
        
        stats = ["PTS", "TRB", "AST"]
        correlations = []
        
        for i, stat in enumerate(stats):
            yp = pred_means[:, i]
            bs = self.baselines_val[:, i]
            delta_pred = yp - bs
            
            corr = np.corrcoef(bs, delta_pred)[0, 1]
            correlations.append(corr)
            
            status = "✓" if abs(corr) < 0.3 else "❌"
            print(f"{stat}: Corr(baseline, delta) = {corr:.3f} {status}")
        
        self.diagnostics['baseline_correlations'] = correlations
        
        # Diagnosis
        if any(c < -0.3 for c in correlations):
            print("\n❌ CRITICAL: Baseline cancellation detected")
            print("   → Increase cov_penalty_weight to 0.3+")
            print("   → Increase neg_corr_penalty_weight to 0.3+")
    
    def _check_sigma_calibration(self, pred_means, pred_sigma):
        """Check uncertainty calibration."""
        print("\n5. UNCERTAINTY CALIBRATION ANALYSIS")
        print("-" * 70)
        
        stats = ["PTS", "TRB", "AST"]
        calibrations = []
        
        for i, stat in enumerate(stats):
            yt = self.y_val[:, i]
            yp = pred_means[:, i]
            sigma = pred_sigma[:, i]
            
            errors = np.abs(yt - yp)
            corr = np.corrcoef(errors, sigma)[0, 1]
            calibrations.append(corr)
            
            status = "✓" if corr > 0.15 else "❌"
            print(f"{stat}: Corr(|error|, sigma) = {corr:.3f} {status} (target: >0.15)")
        
        self.diagnostics['sigma_calibrations'] = calibrations
        
        # Diagnosis
        if all(c < 0.15 for c in calibrations):
            print("\n⚠️  WARNING: Poor uncertainty calibration")
            print("   → Increase sigma_calibration_weight to 0.1")
    
    def _check_spike_detection(self, pred_means):
        """Check spike detection accuracy."""
        print("\n6. SPIKE DETECTION ANALYSIS")
        print("-" * 70)
        
        thresholds = {"PTS": 35.0, "TRB": 11.0, "AST": 9.0}
        stats = ["PTS", "TRB", "AST"]
        
        for i, stat in enumerate(stats):
            yt = self.y_val[:, i]
            yp = pred_means[:, i]
            threshold = thresholds[stat]
            
            # True spikes
            true_spikes = yt >= threshold
            n_true_spikes = np.sum(true_spikes)
            
            if n_true_spikes == 0:
                print(f"{stat}: No spike games in validation")
                continue
            
            # Predicted spikes
            pred_spikes = yp >= threshold
            
            # Accuracy
            correct = np.sum(true_spikes & pred_spikes)
            recall = correct / n_true_spikes if n_true_spikes > 0 else 0
            
            # Average prediction for spike games
            avg_pred_on_spikes = np.mean(yp[true_spikes])
            avg_true_on_spikes = np.mean(yt[true_spikes])
            
            print(f"{stat}:")
            print(f"  True spikes: {n_true_spikes}")
            print(f"  Recall: {recall:.2%}")
            print(f"  Avg pred on spikes: {avg_pred_on_spikes:.1f} (true: {avg_true_on_spikes:.1f})")
        
        print("\nNote: Spike detection is secondary to overall R²")
    
    def _check_trajectory_realism(self, pred_means):
        """Check if predictions look realistic."""
        print("\n7. TRAJECTORY REALISM CHECK")
        print("-" * 70)
        
        # Sample a few sequences
        print("Sample predictions (first 5 games):")
        print("PTS: pred vs true")
        for i in range(min(5, len(pred_means))):
            print(f"  Game {i+1}: {pred_means[i, 0]:.1f} vs {self.y_val[i, 0]:.1f}")
        
        # Check for constant predictions
        std_preds = np.std(pred_means, axis=0)
        print(f"\nPrediction std devs: PTS={std_preds[0]:.2f}, TRB={std_preds[1]:.2f}, AST={std_preds[2]:.2f}")
        
        if np.any(std_preds < 1.0):
            print("⚠️  WARNING: Very low prediction variance - model may be stuck")
    
    def _analyze_failure_modes(self):
        """Analyze diagnostics to identify failure modes."""
        print("\n" + "="*70)
        print("FAILURE MODE ANALYSIS")
        print("="*70)
        
        failure_modes = []
        
        # Check R²
        if self.diagnostics['r2_macro'] < 0:
            failure_modes.append("negative_r2")
        
        # Check variance collapse
        if all(r < 0.6 for r in self.diagnostics['var_ratios']):
            failure_modes.append("variance_collapse")
        
        # Check delta collapse
        if all(r < 0.3 for r in self.diagnostics['delta_var_ratios']):
            failure_modes.append("delta_collapse")
        
        # Check baseline cancellation
        if any(c < -0.3 for c in self.diagnostics['baseline_correlations']):
            failure_modes.append("baseline_cancellation")
        
        # Check calibration
        if all(c < 0.15 for c in self.diagnostics['sigma_calibrations']):
            failure_modes.append("poor_calibration")
        
        self.diagnostics['failure_modes'] = failure_modes
        
        if not failure_modes:
            print("✓ No critical failure modes detected")
        else:
            print("Detected failure modes:")
            for mode in failure_modes:
                print(f"  - {mode}")
    
    def _suggest_weight_adjustments(self):
        """Suggest specific weight adjustments."""
        print("\n" + "="*70)
        print("RECOMMENDED WEIGHT ADJUSTMENTS")
        print("="*70)
        
        suggestions = []
        
        failure_modes = self.diagnostics.get('failure_modes', [])
        
        if 'delta_collapse' in failure_modes:
            suggestions.append({
                'issue': 'Delta Collapse',
                'adjustments': {
                    'variance_encouragement_weight': 2.5,
                    'delta_huber_weight': 0.3,
                    'mean_loss_weight': 0.03,
                },
                'reason': 'Model not learning changes from baseline'
            })
        
        if 'baseline_cancellation' in failure_modes:
            suggestions.append({
                'issue': 'Baseline Cancellation',
                'adjustments': {
                    'cov_penalty_weight': 0.4,
                    'neg_corr_penalty_weight': 0.4,
                },
                'reason': 'Model learning baseline offsets instead of true dynamics'
            })
        
        if 'variance_collapse' in failure_modes:
            suggestions.append({
                'issue': 'Variance Collapse',
                'adjustments': {
                    'variance_encouragement_weight': 2.0,
                    'final_mean_variance_weight': 1.0,
                },
                'reason': 'Predictions too flat, not matching true variance'
            })
        
        if 'poor_calibration' in failure_modes:
            suggestions.append({
                'issue': 'Poor Calibration',
                'adjustments': {
                    'sigma_calibration_weight': 0.1,
                },
                'reason': 'Uncertainty estimates not useful'
            })
        
        if not suggestions:
            print("✓ No critical adjustments needed")
            print("\nFor incremental improvement, try:")
            print("  - Increase variance_encouragement_weight by 50%")
            print("  - Increase delta_huber_weight by 50%")
        else:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"\n{i}. {suggestion['issue']}")
                print(f"   Reason: {suggestion['reason']}")
                print("   Adjustments:")
                for param, value in suggestion['adjustments'].items():
                    print(f"     {param}: {value}")
        
        self.diagnostics['suggestions'] = suggestions


def run_diagnostic_tuning():
    """Run diagnostic-driven tuning."""
    print("="*70)
    print("DIAGNOSTIC-DRIVEN TUNING")
    print("="*70)
    print("Training model with current weights, then analyzing performance...")
    print("="*70)
    
    # Train model
    print("\n1. Training model with current configuration...")
    trainer = ImprovedBaselineTrainer(ensemble_size=1)
    
    # Quick training for diagnosis
    trainer.config['phase1_epochs'] = 30
    trainer.config['phase2_epochs'] = 15
    trainer.config['patience'] = 10
    
    model, metadata = trainer.train()
    
    # Get validation data (need to re-prepare)
    print("\n2. Preparing validation data...")
    X, baselines, y, df = trainer.prepare_data()
    
    # Split
    if trainer.config["use_walk_forward_split"]:
        season_ids = df["Year"].values[:len(X)] if "Year" in df.columns else np.full(len(X), 2025)
        game_idx = df["Game_Index"].values[:len(X)] if "Game_Index" in df.columns else np.arange(len(X))
        metadata_df = pd.DataFrame({
            "player_id": df["Player_ID_mapped"].values[:len(X)],
            "season_id": season_ids,
            "game_index": game_idx,
        })
        train_idx, val_idx = trainer.walk_forward_split(X, baselines, y, metadata_df)
    else:
        split = int(0.8 * len(X))
        train_idx = np.arange(split)
        val_idx = np.arange(split, len(X))
    
    X_val = X[val_idx]
    baselines_val = baselines[val_idx]
    y_val = y[val_idx]
    
    # Scale validation data
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_num = X_val_flat[:, 3:]
    X_val_num_s = trainer.scaler_x.transform(X_val_num)
    X_val_s = np.concatenate([X_val_flat[:, :3], X_val_num_s], axis=1).reshape(X_val.shape)
    
    # Run diagnostics
    print("\n3. Running diagnostics...")
    diagnostics = ModelDiagnostics(trainer, model, X_val_s, baselines_val, y_val)
    results = diagnostics.run_full_diagnostics()
    
    # Save results
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "diagnostic_results.json", "w") as f:
        # Convert numpy types to Python types for JSON
        results_json = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                results_json[k] = float(v)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.float32, np.float64)):
                results_json[k] = [float(x) for x in v]
            else:
                results_json[k] = v
        
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Diagnostic results saved to {output_dir}/diagnostic_results.json")
    
    return results


if __name__ == "__main__":
    run_diagnostic_tuning()
