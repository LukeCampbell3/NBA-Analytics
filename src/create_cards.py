"""
create_cards.py - Generate player cards from raw data

Consolidates all card generation logic into a single file.
Combines functionality from:
- data/generate_player_cards.py
- data/generate_enhanced_player_cards.py
- nba_var/src/cards/build_card.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class PlayerCard:
    """Simplified player card structure"""
    player: Dict[str, Any]
    identity: Dict[str, Any]
    offense: Dict[str, Any]
    defense: Dict[str, Any]
    impact: Dict[str, Any]
    metadata: Dict[str, Any]
    trust: Optional[Dict[str, Any]] = None
    uncertainty: Optional[Dict[str, Any]] = None


def load_player_data(input_path: Path) -> pd.DataFrame:
    """Load player data from CSV or Parquet"""
    if input_path.suffix == '.csv':
        return pd.read_csv(input_path)
    elif input_path.suffix == '.parquet':
        return pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def calculate_identity(row: pd.Series) -> Dict[str, Any]:
    """Calculate player identity (usage, archetype)"""
    # Usage band
    usage = row.get('usage_rate', 0.20)
    if usage >= 0.28:
        usage_band = "high"
    elif usage >= 0.22:
        usage_band = "med"
    else:
        usage_band = "low"
    
    # Simple archetype classification
    position = str(row.get('position', '')).upper()
    height = row.get('height_in', 75)
    
    if position in ['PG', 'SG']:
        if usage >= 0.25:
            archetype = "initiator_creator"
        else:
            archetype = "shooting_specialist"
    elif position in ['SF']:
        archetype = "versatile_wing"
    elif position in ['PF']:
        if height >= 80:
            archetype = "rim_protector"
        else:
            archetype = "versatile_wing"
    elif position in ['C']:
        archetype = "rim_protector"
    else:
        archetype = "connector"
    
    return {
        "usage_band": usage_band,
        "primary_archetype": archetype,
        "position": position
    }


def calculate_offense(row: pd.Series) -> Dict[str, Any]:
    """Calculate offensive metrics"""
    pts = row.get('points_per_game', 0.0)
    ast = row.get('assists_per_game', 0.0)
    tov = row.get('turnovers_per_game', 0.0)
    fg3a = row.get('three_point_attempts_per_game', 0.0)
    fga = row.get('field_goal_attempts_per_game', 1.0)
    
    # Shot profile
    three_rate = fg3a / fga if fga > 0 else 0.0
    
    return {
        "shot_profile": {
            "three_rate": round(three_rate, 3),
            "volume": round(fga, 1)
        },
        "creation": {
            "scoring": round(pts, 1),
            "playmaking": round(ast, 1)
        },
        "efficiency": {
            "ast_tov_ratio": round(ast / tov, 2) if tov > 0 else 99.0
        }
    }


def calculate_defense(row: pd.Series) -> Dict[str, Any]:
    """Calculate defensive metrics"""
    stl = row.get('steals_per_game', 0.0)
    blk = row.get('blocks_per_game', 0.0)
    dreb = row.get('defensive_rebounds_per_game', 0.0)
    
    stocks = stl + blk
    
    # Defensive burden estimate
    if stocks >= 2.0:
        burden_level = "high"
    elif stocks >= 1.0:
        burden_level = "med"
    else:
        burden_level = "low"
    
    return {
        "burden": {
            "level": burden_level,
            "score": round(min(stocks / 3.0, 1.0), 3)
        },
        "performance": {
            "stocks_per_game": round(stocks, 2),
            "dreb_per_game": round(dreb, 1)
        }
    }


def calculate_impact(row: pd.Series) -> Dict[str, Any]:
    """Calculate impact metrics"""
    plus_minus = row.get('plus_minus', 0.0)
    
    # Simple impact estimate
    offensive_impact = row.get('points_per_game', 0.0) / 10.0
    defensive_impact = (row.get('steals_per_game', 0.0) + row.get('blocks_per_game', 0.0)) * 0.5
    
    return {
        "net": round(plus_minus, 2),
        "offensive": round(offensive_impact, 2),
        "defensive": round(defensive_impact, 2),
        "source": "estimated"
    }


def calculate_trust(row: pd.Series) -> Dict[str, Any]:
    """Calculate trust/uncertainty scores"""
    games = row.get('games_played', 0)
    minutes = row.get('minutes_per_game', 0.0)
    
    # Trust based on sample size
    if games >= 60 and minutes >= 25:
        trust_level = "high"
        trust_score = 0.85
    elif games >= 40 and minutes >= 20:
        trust_level = "medium"
        trust_score = 0.70
    else:
        trust_level = "low"
        trust_score = 0.50
    
    return {
        "score": trust_score,
        "level": trust_level,
        "data_quality": "actual" if games >= 50 else "estimated"
    }


def create_player_card(row: pd.Series) -> PlayerCard:
    """Create a complete player card from a data row"""
    player_info = {
        "id": str(row.get('player_id', '')),
        "name": str(row.get('player_name', row.get('name', 'Unknown'))),
        "team": str(row.get('team', 'UNK')).upper(),
        "season": int(row.get('season', 2025)),
        "position": str(row.get('position', '')),
        "age": float(row.get('age', 25.0))
    }
    
    identity = calculate_identity(row)
    offense = calculate_offense(row)
    defense = calculate_defense(row)
    impact = calculate_impact(row)
    trust = calculate_trust(row)
    
    metadata = {
        "games_played": str(int(row.get('games_played', 0))),
        "minutes": float(row.get('minutes_per_game', 0.0)),
        "data_quality": trust["data_quality"]
    }
    
    uncertainty = {
        "overall": round(1.0 - trust["score"], 3),
        "sample_size": "adequate" if int(row.get('games_played', 0)) >= 50 else "limited"
    }
    
    return PlayerCard(
        player=player_info,
        identity=identity,
        offense=offense,
        defense=defense,
        impact=impact,
        metadata=metadata,
        trust=trust,
        uncertainty=uncertainty
    )


def generate_cards(input_path: Path, output_dir: Path, limit: Optional[int] = None):
    """Generate player cards from input data"""
    print(f"Loading data from {input_path}...")
    df = load_player_data(input_path)
    
    if limit:
        df = df.head(limit)
    
    print(f"Generating {len(df)} player cards...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_created = 0
    
    for idx, row in df.iterrows():
        try:
            card = create_player_card(row)
            
            # Save individual card
            player_name = card.player['name'].replace(' ', '_')
            team = card.player['team']
            season = card.player['season']
            
            filename = f"{player_name}_{team}_{season}.json"
            output_path = output_dir / filename
            
            with open(output_path, 'w') as f:
                json.dump(asdict(card), f, indent=2)
            
            cards_created += 1
            
            if cards_created % 100 == 0:
                print(f"  Created {cards_created} cards...")
                
        except Exception as e:
            print(f"  Error creating card for row {idx}: {e}")
            continue
    
    print(f"\n[SUCCESS] Created {cards_created} player cards in {output_dir}")
    
    # Create summary file
    summary = {
        "total_cards": cards_created,
        "output_directory": str(output_dir),
        "source_file": str(input_path)
    }
    
    with open(output_dir / "cards_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return cards_created


def main():
    parser = argparse.ArgumentParser(description="Generate player cards from raw data")
    parser.add_argument('--input', type=Path, required=True, help="Input CSV or Parquet file")
    parser.add_argument('--output', type=Path, default=Path('data/player_cards'), help="Output directory")
    parser.add_argument('--limit', type=int, help="Limit number of cards to generate")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        generate_cards(args.input, args.output, args.limit)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
