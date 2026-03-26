#!/usr/bin/env python3
"""
Scale Cheating Diagnostic Tool
Analyzes whether the model is gaming sigma to avoid learning
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import pandas as pd

def analyze_scale_cheating(y_true, y_pred, sigma, stat_names=["PTS", "TRB", "AST"]):
    """
    Comprehensive analysis of scale cheating
    
    Args:
        y_true: True values (N, 3)
        y_pred: Predicted means (N, 3)
        sigma: Predicted uncertainties (N, 3)
        stat_names: Names of statistics
    
    Returns:
        Dictionary with diagnostic results
    """
    results = {}
    
    print("="*70)
    print("🔍 SCALE CHEATING DIAGNOSTIC")
    print("="*70)
    
    for i, stat in enumerate(stat_names):
        print(f"\n📊 {stat} Analysis:")
        print("-"*70)
        
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        sig = sigma[:, i]
        
        # 1. Correlation between error and sigma
        errors = np.abs(y_t - y_p)
        corr_pearson, p_pearson = pearsonr(errors, sig)
        corr_spearman, p_spearman = spearmanr(errors, sig)
        
        print(f"1. Error-Sigma Correlation:")
        print(f"   Pearson:  {corr_pearson:.3f} (p={p_pearson:.4f})")
        print(f"   Spearman: {corr_spearman:.3f} (p={p_spearman:.4f})")
        
        if corr_pearson > 0.7:
            print(f"   ⚠️  HIGH CORRELATION - Possible scale cheating!")
        elif corr_pearson > 0.4:
            print(f"   ⚠️  MODERATE CORRELATION - Some calibration")
        else:
            print(f"   ✅ LOW CORRELATION - Good calibration")
        
        # 2. Calibration by sigma bins
        print(f"\n2. Calibration by Sigma Bins:")
        n_bins = 10
        sigma_bins = np.percentile(sig, np.linspace(0, 100, n_bins + 1))
        
        for j in range(n_bins):
            mask = (sig >= sigma_bins[j]) & (sig < sigma_bins[j + 1])
            if np.sum(mask) > 0:
                avg_sigma = np.mean(sig[mask])
                rmse = np.sqrt(np.mean(errors[mask] ** 2))
                ratio = rmse / avg_sigma if avg_sigma > 0 else 0
                
                if j % 3 == 0:  # Print every 3rd bin
                    print(f"   Bin {j+1}: σ={avg_sigma:.2f}, RMSE={rmse:.2f}, "
                          f"Ratio={ratio:.2f}")
        
        # 3. Sigma distribution analysis
        print(f"\n3. Sigma Distribution:")
        print(f"   Mean: {np.mean(sig):.2f}")
        print(f"   Std:  {np.std(sig):.2f}")
        print(f"   Min:  {np.min(sig):.2f}")
        print(f"   Max:  {np.max(sig):.2f}")
        print(f"   Q25:  {np.percentile(sig, 25):.2f}")
        print(f"   Q50:  {np.percentile(sig, 50):.2f}")
        print(f"   Q75:  {np.percentile(sig, 75):.2f}")
        
        # Check for collapse
        if np.std(sig) < 0.5:
            print(f"   ⚠️  SIGMA COLLAPSED - Very low variance!")
        elif np.mean(sig) > 10:
            print(f"   ⚠️  SIGMA INFLATED - Very high average!")
        else:
            print(f"   ✅ SIGMA DISTRIBUTION - Reasonable")
        
        # 4. Overconfident vs underconfident
        print(f"\n4. Confidence Analysis:")
        overconfident = np.sum((errors > 2 * sig)) / len(errors)
        underconfident = np.sum((errors < 0.5 * sig)) / len(errors)
        well_calibrated = 1 - overconfident - underconfident
        
        print(f"   Overconfident:  {overconfident*100:.1f}% (error > 2σ)")
        print(f"   Well-calibrated: {well_calibrated*100:.1f}%")
        print(f"   Underconfident: {underconfident*100:.1f}% (error < 0.5σ)")
        
        if overconfident > 0.3:
            print(f"   ⚠️  TOO OVERCONFIDENT - Sigma too small!")
        elif underconfident > 0.5:
            print(f"   ⚠️  TOO UNDERCONFIDENT - Sigma too large!")
        else:
            print(f"   ✅ REASONABLE CALIBRATION")
        
        # 5. Predictive quality
        print(f"\n5. Predictive Quality:")
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        avg_sigma = np.mean(sig)
        
        print(f"   MAE:  {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   Avg σ: {avg_sigma:.2f}")
        print(f"   RMSE/σ: {rmse/avg_sigma:.2f}")
        
        if rmse / avg_sigma > 1.5:
            print(f"   ⚠️  POOR CALIBRATION - Errors exceed uncertainty!")
        elif rmse / avg_sigma < 0.5:
            print(f"   ⚠️  OVER-CAUTIOUS - Uncertainty too high!")
        else:
            print(f"   ✅ GOOD CALIBRATION")
        
        # Store results
        results[stat] = {
            'error_sigma_corr': corr_pearson,
            'error_sigma_corr_p': p_pearson,
            'sigma_mean': np.mean(sig),
            'sigma_std': np.std(sig),
            'overconfident_pct': overconfident,
            'underconfident_pct': underconfident,
            'mae': mae,
            'rmse': rmse,
            'rmse_sigma_ratio': rmse / avg_sigma
        }
    
    return results


def plot_scale_cheating_diagnostics(y_true, y_pred, sigma, stat_names=["PTS", "TRB", "AST"]):
    """Create diagnostic plots for scale cheating"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Scale Cheating Diagnostics', fontsize=16, fontweight='bold')
    
    for i, stat in enumerate(stat_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        sig = sigma[:, i]
        errors = np.abs(y_t - y_p)
        
        # Plot 1: Error vs Sigma scatter
        ax = axes[i, 0]
        ax.scatter(sig, errors, alpha=0.3, s=10)
        ax.plot([sig.min(), sig.max()], [sig.min(), sig.max()], 
                'r--', label='Perfect calibration')
        ax.set_xlabel('Predicted σ')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'{stat}: Error vs Sigma')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Sigma distribution
        ax = axes[i, 1]
        ax.hist(sig, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(sig), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(sig):.2f}')
        ax.set_xlabel('Predicted σ')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{stat}: Sigma Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Calibration curve
        ax = axes[i, 2]
        n_bins = 10
        sigma_bins = np.percentile(sig, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        bin_rmse = []
        
        for j in range(n_bins):
            mask = (sig >= sigma_bins[j]) & (sig < sigma_bins[j + 1])
            if np.sum(mask) > 0:
                bin_centers.append(np.mean(sig[mask]))
                bin_rmse.append(np.sqrt(np.mean(errors[mask] ** 2)))
        
        ax.plot(bin_centers, bin_rmse, 'o-', label='Observed')
        ax.plot([min(bin_centers), max(bin_centers)], 
                [min(bin_centers), max(bin_centers)], 
                'r--', label='Perfect')
        ax.set_xlabel('Predicted σ (binned)')
        ax.set_ylabel('Observed RMSE')
        ax.set_title(f'{stat}: Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Example usage"""
    # Load model predictions
    # This is a placeholder - replace with actual model loading
    print("Load your model and generate predictions on validation set")
    print("Then call:")
    print("  results = analyze_scale_cheating(y_true, y_pred, sigma)")
    print("  fig = plot_scale_cheating_diagnostics(y_true, y_pred, sigma)")
    print("  plt.savefig('scale_cheating_diagnostics.png')")


if __name__ == "__main__":
    main()
