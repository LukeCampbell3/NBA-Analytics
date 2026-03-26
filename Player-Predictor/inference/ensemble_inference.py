#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Inference Script for Hybrid Spike-Driver MoE Models

Loads the trained ensemble and provides comprehensive evaluation:
- Individual model predictions
- Ensemble predictions (mean, median, weighted)
- Uncertainty quantification
- Spike detection analysis
- Expert routing analysis
- Performance metrics (MAE, R², calibration)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import warnings
import io
warnings.filterwarnings("ignore")

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.hybrid_spike_moe_trainer import HybridSpikeMoETrainer, ConditionalSpikeOutput


class EnsembleInference:
    """
    Comprehensive inference for Hybrid Spike-Driver MoE ensemble.
    """
    
    def __init__(self, model_dir="model"):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing saved models and metadata
        """
        self.model_dir = Path(model_dir)
        self.models = []
        self.metadata = None
        self.scaler_x = None
        self.trainer = None
        
        print("🔧 Initializing Ensemble Inference Engine...")
        self._load_models()
        print("✅ Inference engine ready!")
    
    def _load_models(self):
        """Load ensemble models and metadata."""
        
        # Load metadata
        metadata_path = self.model_dir / "hybrid_spike_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"📋 Loaded metadata: {self.metadata['model_type']}")
        print(f"   Ensemble size: {self.metadata['ensemble_size']}")
        print(f"   Strategies: {', '.join(self.metadata['ensemble_strategies'])}")
        
        # Load scaler
        scaler_path = self.model_dir / "hybrid_spike_scaler_x.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler_x = joblib.load(scaler_path)
        print(f"✅ Loaded feature scaler")
        
        # Initialize trainer to get model architecture
        self.trainer = HybridSpikeMoETrainer(ensemble_size=self.metadata['ensemble_size'])
        self.trainer.feature_columns = self.metadata['feature_columns']
        self.trainer.baseline_features = self.metadata['baseline_features']
        self.trainer.target_columns = self.metadata['target_columns']
        self.trainer.player_mapping = self.metadata['player_mapping']
        self.trainer.team_mapping = self.metadata['team_mapping']
        self.trainer.opponent_mapping = self.metadata['opponent_mapping']
        self.trainer.scaler_x = self.scaler_x
        
        # Initialize spike_features (required for model building)
        self.trainer.spike_features = [
            "MP_trend", "High_MP_Flag", "FGA_trend", "AST_trend", 
            "AST_variance", "USG_AST_ratio_trend", "High_Playmaker_Flag"
        ]
        
        # Load each model in ensemble
        for i in range(self.metadata['ensemble_size']):
            weights_path = self.model_dir / f"hybrid_spike_ensemble_{i}_weights.h5"
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")
            
            # Build model architecture
            model = self.trainer.build_model()
            
            # Try to load weights, skip layer count check if needed
            try:
                model.load_weights(str(weights_path))
            except ValueError as e:
                if "Layer count mismatch" in str(e):
                    print(f"⚠️  Warning: Layer count mismatch for model {i+1}, attempting to load by name...")
                    model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
                else:
                    raise
            
            self.models.append(model)
            strategy = self.metadata['ensemble_strategies'][i]
            print(f"✅ Loaded model {i+1}/{self.metadata['ensemble_size']}: {strategy}")
    
    def prepare_input(self, player_name, recent_games_df):
        """
        Prepare input features for inference.
        
        Args:
            player_name: Name of player
            recent_games_df: DataFrame with last seq_len games (must have all required features)
        
        Returns:
            X_scaled: Scaled sequence features [1, seq_len, n_features]
            baseline: Baseline from last game [1, 3]
        """
        seq_len = self.metadata['config']['seq_len']
        
        if len(recent_games_df) < seq_len:
            raise ValueError(f"Need at least {seq_len} recent games, got {len(recent_games_df)}")
        
        # Take last seq_len games
        recent_games_df = recent_games_df.iloc[-seq_len:].copy()
        
        # Extract features
        X = recent_games_df[self.metadata['feature_columns']].values
        baseline = recent_games_df.iloc[-1][self.metadata['baseline_features']].values
        
        # Scale features (skip first 3 categorical)
        X_numeric = X[:, 3:]
        X_numeric_scaled = self.scaler_x.transform(X_numeric)
        X_scaled = np.concatenate([X[:, :3], X_numeric_scaled], axis=1)
        
        # Add batch dimension
        X_scaled = X_scaled.reshape(1, seq_len, -1)
        baseline = baseline.reshape(1, -1)
        
        return X_scaled.astype(np.float32), baseline.astype(np.float32)
    
    def predict_single_model(self, X, baseline, model_idx=0):
        """
        Get prediction from a single model in ensemble.
        
        Args:
            X: Scaled sequence features [1, seq_len, n_features]
            baseline: Baseline [1, 3]
            model_idx: Index of model in ensemble
        
        Returns:
            dict with predictions and uncertainty
        """
        model = self.models[model_idx]
        
        # Get raw output [1, 15]: [normal_means(3), normal_scales(3), spike_means(3), spike_scales(3), spike_indicators(3)]
        output = model.predict([X, baseline], verbose=0)
        
        # Parse output
        normal_means = output[0, :3]
        normal_scales = output[0, 3:6]
        spike_means = output[0, 6:9]
        spike_scales = output[0, 9:12]
        spike_indicators = output[0, 12:15]
        
        return {
            'normal_means': normal_means,
            'normal_scales': normal_scales,
            'spike_means': spike_means,
            'spike_scales': spike_scales,
            'spike_indicators': spike_indicators,
            'strategy': self.metadata['ensemble_strategies'][model_idx]
        }
    
    def predict_ensemble(self, X, baseline, method='mean'):
        """
        Get ensemble prediction.
        
        Args:
            X: Scaled sequence features [1, seq_len, n_features]
            baseline: Baseline [1, 3]
            method: 'mean', 'median', or 'weighted'
        
        Returns:
            dict with ensemble predictions and individual model predictions
        """
        # Get predictions from all models
        individual_preds = []
        for i in range(len(self.models)):
            pred = self.predict_single_model(X, baseline, model_idx=i)
            individual_preds.append(pred)
        
        # Aggregate predictions
        if method == 'mean':
            ensemble_means = np.mean([p['normal_means'] for p in individual_preds], axis=0)
            ensemble_scales = np.mean([p['normal_scales'] for p in individual_preds], axis=0)
            ensemble_spike_indicators = np.mean([p['spike_indicators'] for p in individual_preds], axis=0)
        elif method == 'median':
            ensemble_means = np.median([p['normal_means'] for p in individual_preds], axis=0)
            ensemble_scales = np.median([p['normal_scales'] for p in individual_preds], axis=0)
            ensemble_spike_indicators = np.median([p['spike_indicators'] for p in individual_preds], axis=0)
        elif method == 'weighted':
            # Weight by inverse uncertainty (more confident models get more weight)
            weights = 1.0 / (np.array([p['normal_scales'] for p in individual_preds]) + 1e-6)
            weights = weights / weights.sum(axis=0, keepdims=True)
            
            ensemble_means = np.sum([w * p['normal_means'] for w, p in zip(weights, individual_preds)], axis=0)
            ensemble_scales = np.mean([p['normal_scales'] for p in individual_preds], axis=0)
            ensemble_spike_indicators = np.mean([p['spike_indicators'] for p in individual_preds], axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute ensemble uncertainty (includes model disagreement)
        model_means = np.array([p['normal_means'] for p in individual_preds])
        epistemic_uncertainty = np.std(model_means, axis=0)  # Model disagreement
        aleatoric_uncertainty = ensemble_scales  # Average predicted uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return {
            'ensemble_means': ensemble_means,
            'ensemble_scales': ensemble_scales,
            'ensemble_spike_indicators': ensemble_spike_indicators,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'individual_predictions': individual_preds,
            'baseline': baseline[0]
        }
    
    def evaluate_on_dataset(self, X, baselines, y_true, metadata_df=None):
        """
        Evaluate ensemble on a dataset.
        
        Args:
            X: Scaled sequence features [N, seq_len, n_features]
            baselines: Baselines [N, 3]
            y_true: True targets [N, 3]
            metadata_df: Optional metadata for analysis
        
        Returns:
            dict with comprehensive evaluation metrics
        """
        print(f"\n📊 Evaluating ensemble on {len(X)} samples...")
        
        # Get predictions from all models
        all_preds = []
        for i, model in enumerate(self.models):
            print(f"   Model {i+1}/{len(self.models)}: {self.metadata['ensemble_strategies'][i]}")
            preds = model.predict([X, baselines], verbose=0)
            all_preds.append(preds)
        
        # Parse predictions
        individual_means = [p[:, :3] for p in all_preds]
        individual_scales = [p[:, 3:6] for p in all_preds]
        individual_spike_indicators = [p[:, 12:15] for p in all_preds]
        
        # Ensemble predictions (mean)
        ensemble_means = np.mean(individual_means, axis=0)
        ensemble_scales = np.mean(individual_scales, axis=0)
        ensemble_spike_indicators = np.mean(individual_spike_indicators, axis=0)
        
        # Compute uncertainties
        epistemic_uncertainty = np.std(individual_means, axis=0)
        aleatoric_uncertainty = ensemble_scales
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        # Compute metrics
        results = {
            'n_samples': len(X),
            'targets': self.metadata['target_columns'],
            'individual_models': {},
            'ensemble': {}
        }
        
        # Individual model metrics
        for i, (means, scales) in enumerate(zip(individual_means, individual_scales)):
            strategy = self.metadata['ensemble_strategies'][i]
            model_metrics = self._compute_metrics(y_true, means, scales)
            results['individual_models'][strategy] = model_metrics
        
        # Ensemble metrics
        ensemble_metrics = self._compute_metrics(y_true, ensemble_means, ensemble_scales)
        ensemble_metrics['epistemic_uncertainty_mean'] = epistemic_uncertainty.mean(axis=0).tolist()
        ensemble_metrics['aleatoric_uncertainty_mean'] = aleatoric_uncertainty.mean(axis=0).tolist()
        ensemble_metrics['total_uncertainty_mean'] = total_uncertainty.mean(axis=0).tolist()
        results['ensemble'] = ensemble_metrics
        
        # Spike detection analysis
        spike_thresholds = self.metadata['config']['spike_thresholds']
        spike_analysis = self._analyze_spike_detection(
            y_true, ensemble_means, ensemble_spike_indicators, spike_thresholds
        )
        results['spike_analysis'] = spike_analysis
        
        # Calibration analysis
        calibration = self._analyze_calibration(y_true, ensemble_means, total_uncertainty)
        results['calibration'] = calibration
        
        return results
    
    def _compute_metrics(self, y_true, y_pred, scales):
        """Compute prediction metrics."""
        metrics = {}
        
        # Per-target metrics
        for i, target in enumerate(self.metadata['target_columns']):
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            # Calibration: correlation between |error| and predicted uncertainty
            errors = np.abs(y_true[:, i] - y_pred[:, i])
            corr = np.corrcoef(errors, scales[:, i])[0, 1]
            
            metrics[f'{target}_mae'] = float(mae)
            metrics[f'{target}_r2'] = float(r2)
            metrics[f'{target}_calibration_corr'] = float(corr)
            metrics[f'{target}_mean_uncertainty'] = float(scales[:, i].mean())
        
        # Overall metrics
        metrics['mae_macro'] = float(np.mean([metrics[f'{t}_mae'] for t in self.metadata['target_columns']]))
        metrics['r2_macro'] = float(np.mean([metrics[f'{t}_r2'] for t in self.metadata['target_columns']]))
        metrics['calibration_corr_macro'] = float(np.mean([metrics[f'{t}_calibration_corr'] for t in self.metadata['target_columns']]))
        
        return metrics
    
    def _analyze_spike_detection(self, y_true, y_pred, spike_indicators, spike_thresholds):
        """Analyze spike detection performance."""
        analysis = {}
        
        for i, (target, threshold) in enumerate(spike_thresholds.items()):
            # True spikes
            true_spikes = y_true[:, i] >= threshold
            
            # Predicted spike probability
            spike_prob = spike_indicators[:, i]
            
            # Metrics at different thresholds
            for prob_threshold in [0.3, 0.5, 0.7]:
                pred_spikes = spike_prob >= prob_threshold
                
                # Confusion matrix
                tp = np.sum(true_spikes & pred_spikes)
                fp = np.sum(~true_spikes & pred_spikes)
                tn = np.sum(~true_spikes & ~pred_spikes)
                fn = np.sum(true_spikes & ~pred_spikes)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                analysis[f'{target}_threshold_{prob_threshold}'] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'true_spike_rate': float(true_spikes.mean()),
                    'pred_spike_rate': float(pred_spikes.mean())
                }
        
        return analysis
    
    def _analyze_calibration(self, y_true, y_pred, uncertainty):
        """Analyze uncertainty calibration."""
        calibration = {}
        
        # Compute errors
        errors = np.abs(y_true - y_pred)
        
        # For each target
        for i, target in enumerate(self.metadata['target_columns']):
            # Bin by predicted uncertainty
            n_bins = 10
            bins = np.percentile(uncertainty[:, i], np.linspace(0, 100, n_bins + 1))
            
            bin_errors = []
            bin_uncertainties = []
            
            for j in range(n_bins):
                mask = (uncertainty[:, i] >= bins[j]) & (uncertainty[:, i] < bins[j+1])
                if mask.sum() > 0:
                    bin_errors.append(errors[mask, i].mean())
                    bin_uncertainties.append(uncertainty[mask, i].mean())
            
            calibration[target] = {
                'bin_errors': [float(x) for x in bin_errors],
                'bin_uncertainties': [float(x) for x in bin_uncertainties],
                'correlation': float(np.corrcoef(errors[:, i], uncertainty[:, i])[0, 1])
            }
        
        return calibration
    
    def print_evaluation_summary(self, results):
        """Print human-readable evaluation summary."""
        print("\n" + "="*80)
        print("ENSEMBLE EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Dataset: {results['n_samples']} samples")
        print(f"🎯 Targets: {', '.join(results['targets'])}")
        
        # Individual models
        print("\n" + "-"*80)
        print("INDIVIDUAL MODELS")
        print("-"*80)
        for strategy, metrics in results['individual_models'].items():
            print(f"\n{strategy}:")
            print(f"  MAE: {metrics['mae_macro']:.3f}")
            print(f"  R²:  {metrics['r2_macro']:.3f}")
            for target in results['targets']:
                print(f"    {target}: MAE={metrics[f'{target}_mae']:.3f}, R²={metrics[f'{target}_r2']:.3f}")
        
        # Ensemble
        print("\n" + "-"*80)
        print("ENSEMBLE (Mean Aggregation)")
        print("-"*80)
        metrics = results['ensemble']
        print(f"\nOverall Performance:")
        print(f"  MAE: {metrics['mae_macro']:.3f}")
        print(f"  R²:  {metrics['r2_macro']:.3f}")
        print(f"  Calibration: {metrics['calibration_corr_macro']:.3f}")
        
        print(f"\nPer-Target Performance:")
        for target in results['targets']:
            print(f"  {target}:")
            print(f"    MAE: {metrics[f'{target}_mae']:.3f}")
            print(f"    R²:  {metrics[f'{target}_r2']:.3f}")
            print(f"    Calibration: {metrics[f'{target}_calibration_corr']:.3f}")
            print(f"    Mean Uncertainty: {metrics[f'{target}_mean_uncertainty']:.3f}")
        
        print(f"\nUncertainty Decomposition:")
        for i, target in enumerate(results['targets']):
            print(f"  {target}:")
            print(f"    Epistemic (model disagreement): {metrics['epistemic_uncertainty_mean'][i]:.3f}")
            print(f"    Aleatoric (data noise):         {metrics['aleatoric_uncertainty_mean'][i]:.3f}")
            print(f"    Total:                          {metrics['total_uncertainty_mean'][i]:.3f}")
        
        # Spike detection
        print("\n" + "-"*80)
        print("SPIKE DETECTION ANALYSIS")
        print("-"*80)
        spike_analysis = results['spike_analysis']
        for key, metrics in spike_analysis.items():
            if 'threshold_0.5' in key:  # Show results at 0.5 threshold
                target = key.split('_threshold')[0]
                print(f"\n{target} (threshold=0.5):")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall:    {metrics['recall']:.3f}")
                print(f"  F1:        {metrics['f1']:.3f}")
                print(f"  True spike rate: {metrics['true_spike_rate']*100:.1f}%")
                print(f"  Pred spike rate: {metrics['pred_spike_rate']*100:.1f}%")
        
        print("\n" + "="*80)


def main():
    """Example usage of ensemble inference."""
    
    print("\n" + "="*80)
    print("HYBRID SPIKE-DRIVER MOE ENSEMBLE INFERENCE")
    print("="*80)
    
    # Initialize inference engine
    inference = EnsembleInference(model_dir="model")
    
    # Load test data (using trainer's data loading)
    print("\n📂 Loading test data...")
    trainer = HybridSpikeMoETrainer()
    X, baselines, y, df = trainer.prepare_data()
    
    # Use last 20% as test set
    test_idx = int(0.8 * len(X))
    X_test = X[test_idx:]
    baselines_test = baselines[test_idx:]
    y_test = y[test_idx:]
    
    # Scale test data
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_numeric = X_test_flat[:, 3:]
    X_test_numeric_scaled = inference.scaler_x.transform(X_test_numeric)
    X_test_scaled = np.concatenate([X_test_flat[:, :3], X_test_numeric_scaled], axis=1)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    print(f"✅ Loaded {len(X_test)} test samples")
    
    # Evaluate ensemble
    results = inference.evaluate_on_dataset(X_test_scaled, baselines_test, y_test)
    
    # Print summary
    inference.print_evaluation_summary(results)
    
    # Save results
    output_path = Path("inference/evaluation_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
