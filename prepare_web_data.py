#!/usr/bin/env python3
"""
prepare_web_data.py - Prepare data for web frontend

This script:
1. Copies player card JSON files from data/processed/player_cards to web/data
2. Generates valuation JSON files for each player
3. Creates combined cards.json and valuations.json for the web app
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from value_players import PlayerValuator


def prepare_web_data():
    """Prepare all data for web frontend"""
    
    # Paths - use full data directory
    source_dir = Path('data/processed/player_cards')
    web_data_dir = Path('web/data')
    
    # Create web data directory
    web_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all player card files
    card_files = list(source_dir.glob('*_final.json'))
    
    if not card_files:
        print(f"Error: No player card files found in {source_dir}")
        return 1
    
    print(f"Found {len(card_files)} player cards")
    
    # Initialize valuator
    valuator = PlayerValuator()
    
    # Process each player
    all_cards = []
    all_valuations = []
    
    for i, card_file in enumerate(card_files, 1):
        try:
            # Load card
            with open(card_file, 'r') as f:
                card = json.load(f)
            
            all_cards.append(card)
            
            # Generate valuation
            result = valuator.valuate_player(card)
            valuation = valuator.generate_report(result)
            all_valuations.append(valuation)
            
            if i % 50 == 0:
                print(f"  Processed {i}/{len(card_files)} players...")
            
        except Exception as e:
            print(f"  Error processing {card_file.name}: {e}")
            continue
    
    # Save combined files
    cards_path = web_data_dir / 'cards.json'
    with open(cards_path, 'w') as f:
        json.dump(all_cards, f, indent=2)
    
    valuations_path = web_data_dir / 'valuations.json'
    with open(valuations_path, 'w') as f:
        json.dump(all_valuations, f, indent=2)
    
    print(f"\n[SUCCESS] Prepared web data:")
    print(f"  - {len(all_cards)} player cards -> {cards_path}")
    print(f"  - {len(all_valuations)} valuations -> {valuations_path}")
    print(f"\nTo view the web app:")
    print(f"  python serve_web.py")
    
    return 0


if __name__ == "__main__":
    exit(prepare_web_data())
