#!/usr/bin/env python3
"""
Run Training with Expanded Player Set

This script runs the improved baseline trainer with an expanded set of 20 players
to test if more training data improves delta variance learning.

Expanded from 10 to 20 players:
- Original 10: Curry, LeBron, Giannis, Luka, Tatum, KD, Jokic, Embiid, Dame, Butler
- Added 10: Kawhi, Ant Edwards, Booker, Mitchell, Morant, Trae, SGA, Haliburton, Fox, Banchero
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the trainer
from improved_baseline_trainer import ImprovedBaselineTrainer
from hybrid_spike_moe_trainer import HybridSpikeMoETrainer

# Expanded player list
EXPANDED_PLAYERS = [
    # Original 10
    "Stephen_Curry", "LeBron_James", "Giannis_Antetokounmpo", 
    "Luka_Doncic", "Jayson_Tatum", "Kevin_Durant", "Nikola_Jokic",
    "Joel_Embiid", "Damian_Lillard", "Jimmy_Butler",
    
    # Additional 10
    "Kawhi_Leonard", "Anthony_Edwards", "Devin_Booker",
    "Donovan_Mitchell", "Ja_Morant", "Trae_Young",
    "Shai_Gilgeous-Alexander", "Tyrese_Haliburton", "De'Aaron_Fox",
    "Paolo_Banchero"
]


def patch_prepare_data():
    """Monkey patch the prepare_data method to use expanded player list"""
    
    original_prepare_data = HybridSpikeMoETrainer.prepare_data
    
    def expanded_prepare_data(self):
        """Load and prepare training data with EXPANDED player list"""
        
        print("📊 Loading training data (EXPANDED)...")
        print(f"[EXPANDED] Using {len(EXPANDED_PLAYERS)} players (was 10)")
        data_dir = Path("Data-Proc-OG")
        
        training_players = EXPANDED_PLAYERS
        
        all_data = []
        for player_dir in data_dir.iterdir():
            if not player_dir.is_dir():
                continue
            
            if player_dir.name in training_players:
                for csv_file in player_dir.glob("*_Optimized.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        df["Player_Name"] = player_dir.name
                        all_data.append(df)
                        print(f"✓ Loaded {len(df)} games for {player_dir.name}")
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
        
        if not all_data:
            raise ValueError("No training data loaded!")
        
        df = pd.concat(all_data, ignore_index=True)
        print(f"[EXPANDED] Total training games: {len(df):,}")
        
        # Create ID mappings
        self.player_mapping = {name: idx for idx, name in enumerate(df['Player_Name'].unique())}
        self.team_mapping = {team_id: idx for idx, team_id in enumerate(sorted(df['Team_ID'].unique()))}
        self.opponent_mapping = {opp_id: idx for idx, opp_id in enumerate(sorted(df['Opponent_ID'].unique()))}
        
        df['Player_ID_mapped'] = df['Player_Name'].map(self.player_mapping)
        df['Team_ID_mapped'] = df['Team_ID'].map(self.team_mapping)
        df['Opponent_ID_mapped'] = df['Opponent_ID'].map(self.opponent_mapping)
        
        # Create hybrid features
        df = self.create_hybrid_features(df)
        
        # Clean data
        import numpy as np
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values for model features
        for col in self.feature_columns:
            if col in df.columns:
                if col in self.categorical_features:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fill baseline features
        for col in self.baseline_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create sequences
        print("Creating training sequences...")
        X, baselines, y = self.create_sequences(df)
        
        print(f"[EXPANDED] Training data shape: X={X.shape}, baselines={baselines.shape}, y={y.shape}")
        
        return X, baselines, y, df
    
    # Apply the patch
    HybridSpikeMoETrainer.prepare_data = expanded_prepare_data
    
    return original_prepare_data


def main():
    """Main training function"""
    
    print("\n" + "="*80)
    print("EXPANDED DATA TRAINING: 20 Players (2x more data)")
    print("="*80)
    print("\nExpanded Player List:")
    for i, player in enumerate(EXPANDED_PLAYERS, 1):
        print(f"  {i:2d}. {player}")
    print()
    
    # Patch the prepare_data method
    original_prepare_data = patch_prepare_data()
    
    try:
        # Create and run trainer
        trainer = ImprovedBaselineTrainer(ensemble_size=1)
        model, meta = trainer.train()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE - EXPANDED DATA")
        print("="*80)
        
    finally:
        # Restore original method
        HybridSpikeMoETrainer.prepare_data = original_prepare_data


if __name__ == "__main__":
    main()
