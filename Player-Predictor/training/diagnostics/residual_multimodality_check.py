#!/usr/bin/env python3
"""
Residual Multimodality Analysis
Check if our residuals show genuine multimodality that would justify MDN complexity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from fixed_ensemble_inference import FixedEnsembleInference

def analyze_residual_multimodality():
    """Check for multimodality in prediction residuals"""
    
    print("🔍 Analyzing Residual Multimodality")
    print("=" * 50)
    
    # Load ensemble and make predictions
    inference = FixedEnsembleInference()
    if not inference.load_ensemble_components():
        return
    
    players = ["Stephen_Curry", "LeBron_James"]
    all_residuals = {"PTS": [], "TRB": [], "AST": []}
    all_contexts = []
    
    for player in players:
        print(f"\n📊 Analyzing {player}...")
        
        X, baselines, y_true, game_indices, df = inference.prepare_test_data(player)
        if X is None:
            continue
            
        y_pred_means, y_pred_stds = inference.predict_ensemble(X, baselines)
        
        # Calculate residuals
        residuals = y_true - y_pred_means
        
        # Add to overall collection
        for i, stat in enumerate(["PTS", "TRB", "AST"]):
            all_residuals[stat].extend(residuals[:, i])
        
        # Collect context information
        for i in range(len(residuals)):
            game_idx = game_indices[i]
            game_row = df[df['Game_Index'] == game_idx].iloc[0]
            
            context = {
                'player': player,
                'home': game_row.get('Home', 0),
                'b2b': game_row.get('BackToBack', 0),
                'mp': game_row.get('MP', 30),
                'usage': game_row.get('USG%', 20),
                'pts_actual': y_true[i, 0],
                'pts_pred': y_pred_means[i, 0],
                'pts_residual': residuals[i, 0],
                'trb_residual': residuals[i, 1],
                'ast_residual': residuals[i, 2],
            }
            all_contexts.append(context)
    
    # Convert to arrays
    for stat in all_residuals:
        all_residuals[stat] = np.array(all_residuals[stat])
    
    context_df = pd.DataFrame(all_contexts)
    
    print(f"\n📈 Overall Residual Statistics:")
    for stat in ["PTS", "TRB", "AST"]:
        residuals = all_residuals[stat]
        print(f"  {stat}: mean={np.mean(residuals):.2f}, std={np.std(residuals):.2f}, skew={stats.skew(residuals):.2f}")
    
    # Test for multimodality using various methods
    print(f"\n🔍 Multimodality Tests:")
    
    for stat in ["PTS", "TRB", "AST"]:
        residuals = all_residuals[stat]
        
        print(f"\n{stat} Residuals:")
        
        # 1. Hartigan's Dip Test (if available)
        try:
            from diptest import diptest
            dip_stat, p_value = diptest(residuals)
            print(f"  Hartigan's Dip Test: stat={dip_stat:.4f}, p={p_value:.4f}")
            if p_value < 0.05:
                print(f"    → Significant multimodality detected!")
            else:
                print(f"    → No significant multimodality")
        except ImportError:
            print(f"  Hartigan's Dip Test: diptest package not available")
        
        # 2. Bimodality Coefficient
        n = len(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        bimodality_coeff = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3)))
        print(f"  Bimodality Coefficient: {bimodality_coeff:.4f}")
        if bimodality_coeff > 0.555:
            print(f"    → Suggests bimodality (>0.555)")
        else:
            print(f"    → Suggests unimodality (≤0.555)")
        
        # 3. Visual inspection of histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
        plt.title(f'{stat} Residuals Distribution')
        plt.xlabel('Residual')
        plt.ylabel('Density')
        
        # Fit normal and check for deviations
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal fit')
        plt.legend()
        plt.savefig(f'{stat.lower()}_residuals_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    → Histogram saved as {stat.lower()}_residuals_distribution.png")
    
    # 4. Conditional multimodality analysis
    print(f"\n🎯 Conditional Multimodality Analysis:")
    
    # Check residuals by context
    contexts_to_check = [
        ('High Usage', context_df['usage'] > context_df['usage'].quantile(0.75)),
        ('Low Usage', context_df['usage'] < context_df['usage'].quantile(0.25)),
        ('High Minutes', context_df['mp'] > 35),
        ('Low Minutes', context_df['mp'] < 25),
        ('Home Games', context_df['home'] == 1),
        ('Away Games', context_df['home'] == 0),
        ('Back-to-Back', context_df['b2b'] == 1),
        ('Rested', context_df['b2b'] == 0),
    ]
    
    for context_name, mask in contexts_to_check:
        if np.sum(mask) < 10:  # Need enough samples
            continue
            
        print(f"\n  {context_name} ({np.sum(mask)} games):")
        
        for stat in ["PTS"]:  # Focus on PTS for brevity
            residuals = context_df[mask][f'{stat.lower()}_residual'].values
            
            if len(residuals) < 5:
                continue
                
            # Bimodality coefficient for this context
            if len(residuals) > 3:
                skewness = stats.skew(residuals)
                kurtosis = stats.kurtosis(residuals)
                n = len(residuals)
                bimodality_coeff = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3)))
                print(f"    {stat} Bimodality Coeff: {bimodality_coeff:.4f}")
                
                if bimodality_coeff > 0.555:
                    print(f"      → MULTIMODAL in {context_name}!")
    
    # 5. Spike vs Non-spike residual analysis
    print(f"\n🎯 Spike vs Non-Spike Residual Analysis:")
    
    spike_thresholds = {"PTS": 35.0, "TRB": 11.0, "AST": 9.0}
    
    for stat in ["PTS", "TRB", "AST"]:
        threshold = spike_thresholds[stat]
        actual_values = context_df[f'{stat.lower()}_actual'] if f'{stat.lower()}_actual' in context_df.columns else context_df['pts_actual']
        
        if stat != "PTS":  # We only have PTS actual in context
            continue
            
        spike_mask = actual_values >= threshold
        non_spike_mask = ~spike_mask
        
        spike_residuals = context_df[spike_mask]['pts_residual'].values
        non_spike_residuals = context_df[non_spike_mask]['pts_residual'].values
        
        print(f"\n  {stat}:")
        print(f"    Spike games (≥{threshold}): {len(spike_residuals)} games")
        print(f"      Mean residual: {np.mean(spike_residuals):.2f}")
        print(f"      Std residual: {np.std(spike_residuals):.2f}")
        
        print(f"    Non-spike games (<{threshold}): {len(non_spike_residuals)} games")
        print(f"      Mean residual: {np.mean(non_spike_residuals):.2f}")
        print(f"      Std residual: {np.std(non_spike_residuals):.2f}")
        
        # Two-sample KS test
        if len(spike_residuals) > 5 and len(non_spike_residuals) > 5:
            ks_stat, ks_p = stats.ks_2samp(spike_residuals, non_spike_residuals)
            print(f"    KS Test (different distributions): stat={ks_stat:.4f}, p={ks_p:.4f}")
            if ks_p < 0.05:
                print(f"      → Spike and non-spike residuals are significantly different!")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"Based on the multimodality analysis:")
    print(f"1. If bimodality coefficients > 0.555 in multiple contexts → Consider conditional 2-component mixture")
    print(f"2. If spike vs non-spike residuals are significantly different → Spike-gated mixture is justified")
    print(f"3. If residuals are mostly unimodal → Stick with Student-t likelihood")

if __name__ == "__main__":
    analyze_residual_multimodality()