#!/usr/bin/env python3
"""
Single Game Prediction Script

Predict a player's performance for an upcoming game using the ensemble.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.ensemble_inference import EnsembleInference


def predict_next_game(player_name, recent_games_csv=None, recent_games_df=None):
    """
    Predict player's next game performance.
    
    Args:
        player_name: Name of player (e.g., "Stephen_Curry")
        recent_games_csv: Path to CSV with recent games
        recent_games_df: Or provide DataFrame directly
    
    Returns:
        dict with predictions
    """
    # Initialize inference
    inference = EnsembleInference(model_dir="model")
    
    # Load recent games
    if recent_games_df is None:
        if recent_games_csv is None:
            raise ValueError("Must provide either recent_games_csv or recent_games_df")
        recent_games_df = pd.read_csv(recent_games_csv)
    
    # Prepare input
    X, baseline = inference.prepare_input(player_name, recent_games_df)
    
    # Get ensemble prediction
    prediction = inference.predict_ensemble(X, baseline, method='mean')
    
    return prediction


def print_prediction(player_name, prediction):
    """Print prediction in human-readable format."""
    
    print("\n" + "="*80)
    print(f"PREDICTION FOR {player_name.replace('_', ' ').upper()}")
    print("="*80)
    
    targets = ['PTS', 'TRB', 'AST']
    baseline = prediction['baseline']
    means = prediction['ensemble_means']
    scales = prediction['ensemble_scales']
    spike_indicators = prediction['ensemble_spike_indicators']
    epistemic = prediction['epistemic_uncertainty']
    aleatoric = prediction['aleatoric_uncertainty']
    total = prediction['total_uncertainty']
    
    print("\n📊 BASELINE (Recent Average):")
    for i, target in enumerate(targets):
        print(f"  {target}: {baseline[i]:.1f}")
    
    print("\n🎯 PREDICTED PERFORMANCE:")
    for i, target in enumerate(targets):
        print(f"  {target}: {means[i]:.1f} ± {total[i]:.1f}")
        print(f"       Range: [{means[i] - 2*total[i]:.1f}, {means[i] + 2*total[i]:.1f}] (95% confidence)")
        print(f"       Spike probability: {spike_indicators[i]*100:.1f}%")
    
    print("\n📈 UNCERTAINTY BREAKDOWN:")
    for i, target in enumerate(targets):
        print(f"  {target}:")
        print(f"    Model disagreement (epistemic): {epistemic[i]:.2f}")
        print(f"    Data noise (aleatoric):         {aleatoric[i]:.2f}")
        print(f"    Total uncertainty:              {total[i]:.2f}")
    
    print("\n🤖 INDIVIDUAL MODEL PREDICTIONS:")
    for pred in prediction['individual_predictions']:
        strategy = pred['strategy']
        means_i = pred['normal_means']
        print(f"  {strategy}:")
        print(f"    PTS: {means_i[0]:.1f}, TRB: {means_i[1]:.1f}, AST: {means_i[2]:.1f}")
    
    print("\n" + "="*80)


def main():
    """Example: Predict Stephen Curry's next game."""
    
    # Example: Load Stephen Curry's recent games
    player_name = "Stephen_Curry"
    data_path = Path("Data") / player_name / "2024_processed.csv"
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        print("\nUsage:")
        print("  python predict_game.py")
        print("\nOr in Python:")
        print("  from inference.predict_game import predict_next_game")
        print("  prediction = predict_next_game('Stephen_Curry', recent_games_csv='path/to/games.csv')")
        return
    
    # Load recent games
    df = pd.read_csv(data_path)
    print(f"📂 Loaded {len(df)} games for {player_name}")
    
    # Predict next game
    print(f"\n🔮 Predicting next game for {player_name}...")
    prediction = predict_next_game(player_name, recent_games_df=df)
    
    # Print results
    print_prediction(player_name, prediction)


if __name__ == "__main__":
    main()
