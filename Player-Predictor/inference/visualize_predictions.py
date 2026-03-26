#!/usr/bin/env python3
"""
Visualization Script for Model Predictions

Creates comprehensive visualizations:
- Prediction vs actual scatter plots
- Uncertainty calibration plots
- Spike detection ROC curves
- Expert routing analysis
- Time series predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.ensemble_inference import EnsembleInference
from training.hybrid_spike_moe_trainer import HybridSpikeMoETrainer

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_prediction_scatter(y_true, y_pred, uncertainty, targets, save_path=None):
    """Plot predicted vs actual with uncertainty."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (ax, target) in enumerate(zip(axes, targets)):
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20, c=uncertainty[:, i], 
                  cmap='viridis', label='Predictions')
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # Metrics
        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        r2 = 1 - np.sum((y_true[:, i] - y_pred[:, i])**2) / np.sum((y_true[:, i] - y_true[:, i].mean())**2)
        
        ax.set_xlabel(f'Actual {target}', fontsize=12)
        ax.set_ylabel(f'Predicted {target}', fontsize=12)
        ax.set_title(f'{target}: MAE={mae:.2f}, R²={r2:.3f}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Color bar for uncertainty
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Uncertainty', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_calibration(y_true, y_pred, uncertainty, targets, save_path=None):
    """Plot uncertainty calibration."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (ax, target) in enumerate(zip(axes, targets)):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        
        # Bin by uncertainty
        n_bins = 10
        bins = np.percentile(uncertainty[:, i], np.linspace(0, 100, n_bins + 1))
        
        bin_centers = []
        bin_errors = []
        bin_uncertainties = []
        
        for j in range(n_bins):
            mask = (uncertainty[:, i] >= bins[j]) & (uncertainty[:, i] < bins[j+1])
            if mask.sum() > 0:
                bin_centers.append((bins[j] + bins[j+1]) / 2)
                bin_errors.append(errors[mask].mean())
                bin_uncertainties.append(uncertainty[mask, i].mean())
        
        # Plot
        ax.scatter(bin_uncertainties, bin_errors, s=100, alpha=0.7, label='Binned data')
        ax.plot([0, max(bin_uncertainties)], [0, max(bin_uncertainties)], 'r--', lw=2, label='Perfect calibration')
        
        # Correlation
        corr = np.corrcoef(errors, uncertainty[:, i])[0, 1]
        
        ax.set_xlabel('Predicted Uncertainty', fontsize=12)
        ax.set_ylabel('Actual Error', fontsize=12)
        ax.set_title(f'{target} Calibration (corr={corr:.3f})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spike_detection_roc(y_true, spike_indicators, spike_thresholds, save_path=None):
    """Plot ROC curves for spike detection."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    targets = list(spike_thresholds.keys())
    
    for i, (ax, target) in enumerate(zip(axes, targets)):
        threshold = spike_thresholds[target]
        true_spikes = y_true[:, i] >= threshold
        spike_prob = spike_indicators[:, i]
        
        # Compute ROC curve
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for t in thresholds:
            pred_spikes = spike_prob >= t
            
            tp = np.sum(true_spikes & pred_spikes)
            fp = np.sum(~true_spikes & pred_spikes)
            tn = np.sum(~true_spikes & ~pred_spikes)
            fn = np.sum(true_spikes & ~pred_spikes)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Plot ROC
        ax.plot(fpr_list, tpr_list, lw=2, label=f'{target} ROC')
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
        
        # Compute AUC
        auc = np.trapz(tpr_list, fpr_list)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{target} Spike Detection (AUC={auc:.3f})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residual_distribution(y_true, y_pred, targets, save_path=None):
    """Plot residual distributions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (ax, target) in enumerate(zip(axes, targets)):
        residuals = y_true[:, i] - y_pred[:, i]
        
        # Histogram
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Fit normal distribution
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2), 
               'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
        
        ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{target} Residual Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', lw=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_uncertainty_decomposition(epistemic, aleatoric, targets, save_path=None):
    """Plot uncertainty decomposition."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(targets))
    width = 0.35
    
    epistemic_mean = epistemic.mean(axis=0)
    aleatoric_mean = aleatoric.mean(axis=0)
    
    ax.bar(x - width/2, epistemic_mean, width, label='Epistemic (Model Disagreement)', alpha=0.8)
    ax.bar(x + width/2, aleatoric_mean, width, label='Aleatoric (Data Noise)', alpha=0.8)
    
    ax.set_xlabel('Target', fontsize=12)
    ax.set_ylabel('Mean Uncertainty', fontsize=12)
    ax.set_title('Uncertainty Decomposition', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_all_visualizations(output_dir="inference/plots"):
    """Create all visualizations."""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference
    print("\n🔧 Loading models...")
    inference = EnsembleInference(model_dir="model")
    
    # Load test data
    print("📂 Loading test data...")
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
    X_test_numeric_scaled = trainer.scaler_x.transform(X_test_numeric)
    X_test_scaled = np.concatenate([X_test_flat[:, :3], X_test_numeric_scaled], axis=1)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    print(f"✅ Loaded {len(X_test)} test samples")
    
    # Get predictions
    print("\n🔮 Generating predictions...")
    all_preds = []
    for model in inference.models:
        preds = model.predict([X_test_scaled, baselines_test], verbose=0)
        all_preds.append(preds)
    
    # Parse predictions
    ensemble_means = np.mean([p[:, :3] for p in all_preds], axis=0)
    ensemble_scales = np.mean([p[:, 3:6] for p in all_preds], axis=0)
    spike_indicators = np.mean([p[:, 12:15] for p in all_preds], axis=0)
    
    epistemic = np.std([p[:, :3] for p in all_preds], axis=0)
    aleatoric = ensemble_scales
    total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2)
    
    targets = inference.metadata['target_columns']
    spike_thresholds = inference.metadata['config']['spike_thresholds']
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    print("  1. Prediction scatter plots...")
    plot_prediction_scatter(y_test, ensemble_means, total_uncertainty, targets,
                           save_path=output_dir / "prediction_scatter.png")
    
    print("  2. Calibration plots...")
    plot_calibration(y_test, ensemble_means, total_uncertainty, targets,
                    save_path=output_dir / "calibration.png")
    
    print("  3. Spike detection ROC curves...")
    plot_spike_detection_roc(y_test, spike_indicators, spike_thresholds,
                            save_path=output_dir / "spike_detection_roc.png")
    
    print("  4. Residual distributions...")
    plot_residual_distribution(y_test, ensemble_means, targets,
                              save_path=output_dir / "residual_distribution.png")
    
    print("  5. Uncertainty decomposition...")
    plot_uncertainty_decomposition(epistemic, aleatoric, targets,
                                  save_path=output_dir / "uncertainty_decomposition.png")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\n" + "="*80)


if __name__ == "__main__":
    create_all_visualizations()
