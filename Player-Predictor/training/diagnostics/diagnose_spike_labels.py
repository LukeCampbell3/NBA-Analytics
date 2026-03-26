#!/usr/bin/env python3
"""
Diagnose Spike Label Alignment and AST Issues
Step 1 of the high-leverage fix plan
"""

import numpy as np
import pandas as pd
from pathlib import Path

def diagnose_spike_labels():
    """Diagnose spike label alignment and AST threshold issues"""
    
    print("🔍 Diagnosing Spike Label Alignment")
    print("=" * 60)
    
    # Load Stephen Curry data for diagnosis
    data_dir = Path("Data-Proc-OG") / "Stephen_Curry"
    csv_files = list(data_dir.glob("*_Optimized.csv"))
    latest_file = sorted(csv_files)[-1]
    df = pd.read_csv(latest_file)
    
    print(f"✓ Loaded {len(df)} games from {latest_file.name}")
    
    # Sort by game index to ensure proper sequence
    df = df.sort_values('Game_Index').reset_index(drop=True)
    
    # Current fixed thresholds
    fixed_thresholds = {"PTS": 35.0, "TRB": 11.0, "AST": 9.0}
    
    print(f"\n📊 Current Fixed Thresholds Analysis:")
    for stat, threshold in fixed_thresholds.items():
        if stat in df.columns:
            spike_games = df[stat] >= threshold
            spike_count = np.sum(spike_games)
            spike_rate = spike_count / len(df)
            
            print(f"  {stat} ≥ {threshold}: {spike_count}/{len(df)} games ({spike_rate:.1%})")
            print(f"    Mean: {df[stat].mean():.1f}, Std: {df[stat].std():.1f}")
            print(f"    Min: {df[stat].min():.1f}, Max: {df[stat].max():.1f}")
    
    # Player-relative quantile analysis
    print(f"\n📈 Player-Relative Quantile Analysis:")
    quantiles = [0.75, 0.80, 0.85, 0.90, 0.95]
    
    for stat in ["PTS", "TRB", "AST"]:
        if stat in df.columns:
            print(f"\n  {stat} Quantiles:")
            for q in quantiles:
                threshold = df[stat].quantile(q)
                spike_count = np.sum(df[stat] >= threshold)
                print(f"    Q{q:.0%}: {threshold:.1f} → {spike_count} games ({spike_count/len(df):.1%})")
    
    # Check sequence alignment (critical for spike detection)
    print(f"\n🔍 Sequence Alignment Check (last 20 games):")
    seq_len = 10
    
    if len(df) >= seq_len + 5:
        print(f"Game_Index | PTS | AST | PTS_spike | AST_spike | Notes")
        print("-" * 70)
        
        for i in range(len(df) - 20, len(df)):
            if i >= seq_len:  # Can create sequence
                game_idx = df.iloc[i]['Game_Index']
                pts = df.iloc[i]['PTS']
                ast = df.iloc[i]['AST']
                
                # Fixed threshold labels
                pts_spike = 1 if pts >= fixed_thresholds['PTS'] else 0
                ast_spike = 1 if ast >= fixed_thresholds['AST'] else 0
                
                # Player-relative labels (85th percentile)
                pts_q85 = df['PTS'].quantile(0.85)
                ast_q85 = df['AST'].quantile(0.85)
                pts_spike_rel = 1 if pts >= pts_q85 else 0
                ast_spike_rel = 1 if ast >= ast_q85 else 0
                
                notes = ""
                if pts_spike != pts_spike_rel:
                    notes += f"PTS: fixed≠rel "
                if ast_spike != ast_spike_rel:
                    notes += f"AST: fixed≠rel "
                
                print(f"{game_idx:10d} | {pts:3.0f} | {ast:3.0f} | {pts_spike:9d} | {ast_spike:9d} | {notes}")
    
    # AST component analysis - check if there's a structural issue
    print(f"\n🎯 AST Spike Context Analysis:")
    
    # Look for patterns in AST spikes
    ast_spikes = df[df['AST'] >= fixed_thresholds['AST']]
    ast_normal = df[df['AST'] < fixed_thresholds['AST']]
    
    if len(ast_spikes) > 0 and len(ast_normal) > 0:
        print(f"  AST Spike Games ({len(ast_spikes)}):")
        print(f"    Mean AST: {ast_spikes['AST'].mean():.1f}")
        if 'MP' in ast_spikes.columns:
            print(f"    Mean MP: {ast_spikes['MP'].mean():.1f}")
        if 'FGA' in ast_spikes.columns:
            print(f"    Mean FGA: {ast_spikes['FGA'].mean():.1f}")
        
        print(f"  AST Normal Games ({len(ast_normal)}):")
        print(f"    Mean AST: {ast_normal['AST'].mean():.1f}")
        if 'MP' in ast_normal.columns:
            print(f"    Mean MP: {ast_normal['MP'].mean():.1f}")
        if 'FGA' in ast_normal.columns:
            print(f"    Mean FGA: {ast_normal['FGA'].mean():.1f}")
    
    # Recommend player-relative thresholds
    print(f"\n💡 Recommended Player-Relative Thresholds (85th percentile):")
    for stat in ["PTS", "TRB", "AST"]:
        if stat in df.columns:
            q85 = df[stat].quantile(0.85)
            q95 = df[stat].quantile(0.95)
            current_fixed = fixed_thresholds[stat]
            
            print(f"  {stat}:")
            print(f"    Current fixed: {current_fixed:.1f}")
            print(f"    Recommended spike (Q85): {q85:.1f}")
            print(f"    Recommended extreme (Q95): {q95:.1f}")
            
            # Show impact
            fixed_spikes = np.sum(df[stat] >= current_fixed)
            rel_spikes = np.sum(df[stat] >= q85)
            print(f"    Impact: {fixed_spikes} → {rel_spikes} spike games")

if __name__ == "__main__":
    diagnose_spike_labels()