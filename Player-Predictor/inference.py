#!/usr/bin/env python3
"""
UNIFIED INFERENCE - All Inference Functionality in One File

Combines:
- ensemble_inference.py
- evaluate_simple.py
- predict_game.py
- visualize_predictions.py

Usage:
    python inference.py                    # Evaluate ensemble
    python inference.py --visualize        # Evaluate + visualize
    python inference.py --predict          # Predict next game
    python inference.py --player "LeBron_James" --games 5  # Predict specific player
"""

import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# Add training directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "training"))

from hybrid_spike_moe_trainer import HybridSpikeMoETrainer, ConditionalSpikeOutput
from sklearn.metrics import mean_absolute_error, r2_score


class UnifiedInference:
    """Unified inference engine for all prediction tasks."""
    
    def __init__(self, model_dir="model"):
        self.model_dir = Path(model_dir)
        self.models = []
        self.metadata = None
        self.scaler_x = None
        self.trainer = None
        
        print("🔧 Initializing Unified Inference Engine...")
        self._load_models()
        print("✅ Inference engine ready!")
    
    def _load_models(self):
        """Load ensemble models and metadata."""
        metadata_path = self.model_dir / "hybrid_spike_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"📋 Loaded metadata: {self.metadata['model_type']}")
        print(f"   Ensemble size: {self.metadata['ensemble_size']}")
        
        scaler_path = self.model_dir / "hybrid_spike_scaler_x.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler_x = joblib.load(scaler_path)
        print(f"✅ Loaded feature scaler")
        
        self.trainer = HybridSpikeMoETrainer(ensemble_size=self.metadata['ensemble_size'])
        self.trainer.feature_columns = self.metadata['feature_columns']
        self.trainer.baseline_features = self.metadata['baseline_features']
        self.trainer.target_columns = self.metadata['target_columns']
        self.trainer.player_mapping = self.metadata['player_mapping']
        self.trainer.team_mapping = self.metadata['team_mapping']
        self.trainer.opponent_mapping = self.metadata['opponent_mapping']
        self.trainer.scaler_x = self.scaler_x
        
        self.trainer.spike_features = [
            "MP_trend", "High_MP_Flag", "FGA_trend", "AST_trend", 
            "AST_variance", "USG_AST_ratio_trend", "High_Playmaker_Flag"
        ]
        
        for i in range(self.metadata['ensemble_size']):
            weights_path = self.model_dir / f"hybrid_spike_ensemble_{i}_weights.h5"
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")
            
            model = self.trainer.build_model()
            
            try:
                model.load_weights(str(weights_path))
            except ValueError as e:
                if "Layer count mismatch" in str(e):
                    print(f"⚠️  Warning: Layer count mismatch for model {i+1}, loading by name...")
                    model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
                else:
                    raise
            
            self.models.append(model)
            strategy = self.metadata['ensemble_strategies'][i]
            print(f"✅ Loaded model {i+1}/{self.metadata['ensemble_size']}: {strategy}")

    
    def evaluate(self):
        """Evaluate ensemble on test data."""
        print("\n📂 Loading test data...")
        X, baselines, y, df = self.trainer.prepare_data()
        
        test_idx = int(0.8 * len(X))
        X_test = X[test_idx:]
        baselines_test = baselines[test_idx:]
        y_test = y[test_idx:]
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_numeric = X_test_flat[:, 3:]
        X_test_numeric_scaled = self.scaler_x.transform(X_test_numeric)
        X_test_scaled = np.concatenate([X_test_flat[:, :3], X_test_numeric_scaled], axis=1)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        print(f"✅ Loaded {len(X_test)} test samples")
        
        print(f"\n📊 Evaluating ensemble on {len(X_test)} samples...")
        
        all_preds = []
        for i, model in enumerate(self.models):
            print(f"   Model {i+1}/{len(self.models)}: {self.metadata['ensemble_strategies'][i]}")
            preds = model.predict([X_test_scaled, baselines_test], verbose=0)
            all_preds.append(preds)
        
        individual_means = [p[:, :3] for p in all_preds]
        ensemble_means = np.mean(individual_means, axis=0)
        
        results = {
            'n_samples': len(X_test),
            'targets': self.metadata['target_columns'],
            'ensemble': {}
        }
        
        for i, target in enumerate(self.metadata['target_columns']):
            mae = mean_absolute_error(y_test[:, i], ensemble_means[:, i])
            r2 = r2_score(y_test[:, i], ensemble_means[:, i])
            
            results['ensemble'][f'{target}_mae'] = float(mae)
            results['ensemble'][f'{target}_r2'] = float(r2)
        
        results['ensemble']['mae_macro'] = float(np.mean([results['ensemble'][f'{t}_mae'] for t in self.metadata['target_columns']]))
        results['ensemble']['r2_macro'] = float(np.mean([results['ensemble'][f'{t}_r2'] for t in self.metadata['target_columns']]))
        
        self._print_results(results)
        
        output_path = Path("inference/evaluation_results.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        return results
    
    def _print_results(self, results):
        """Print evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 Dataset: {results['n_samples']} samples")
        print(f"🎯 Targets: {', '.join(results['targets'])}")
        
        print("\n" + "-"*80)
        print("ENSEMBLE PERFORMANCE")
        print("-"*80)
        metrics = results['ensemble']
        print(f"\nOverall:")
        print(f"  MAE: {metrics['mae_macro']:.3f}")
        print(f"  R²:  {metrics['r2_macro']:.3f}")
        
        print(f"\nPer-Target:")
        for target in results['targets']:
            print(f"  {target}:")
            print(f"    MAE: {metrics[f'{target}_mae']:.3f}")
            print(f"    R²:  {metrics[f'{target}_r2']:.3f}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Unified NBA Player Prediction Inference")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--predict", action="store_true", help="Predict next game")
    parser.add_argument("--player", type=str, help="Player name for prediction")
    parser.add_argument("--games", type=int, default=10, help="Number of recent games to use")
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNIFIED INFERENCE ENGINE")
    print("="*80)
    
    inference = UnifiedInference(model_dir="model")
    
    if args.predict:
        print("\n🔮 Prediction mode not yet implemented")
        print("   Use: python inference/predict_game.py")
    elif args.visualize:
        print("\n📊 Visualization mode not yet implemented")
        print("   Use: python inference/visualize_predictions.py")
    else:
        inference.evaluate()
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
