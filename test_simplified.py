"""
test_simplified.py - Test the simplified NBA-VAR system

Quick test to verify all simplified modules work correctly.
"""

import json
import tempfile
from pathlib import Path
import pandas as pd


def test_create_cards():
    """Test card creation"""
    print("Testing create_cards.py...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'player_name': ['Test Player A', 'Test Player B'],
        'team': ['LAL', 'BOS'],
        'season': [2025, 2025],
        'position': ['SG', 'PF'],
        'age': [24.5, 27.0],
        'points_per_game': [18.5, 22.3],
        'assists_per_game': [3.2, 2.1],
        'rebounds_per_game': [4.5, 8.2],
        'steals_per_game': [1.1, 0.8],
        'blocks_per_game': [0.3, 1.5],
        'turnovers_per_game': [2.1, 2.5],
        'field_goal_attempts_per_game': [14.2, 17.8],
        'three_point_attempts_per_game': [6.5, 2.3],
        'minutes_per_game': [32.5, 35.2],
        'games_played': [72, 68],
        'usage_rate': [0.24, 0.28],
        'plus_minus': [2.5, 4.1]
    })
    
    # Use a persistent temp directory
    tmpdir = Path(tempfile.mkdtemp())
    input_file = tmpdir / 'test_data.csv'
    output_dir = tmpdir / 'cards'
    
    sample_data.to_csv(input_file, index=False)
    
    # Import and run
    from create_cards import generate_cards
    
    count = generate_cards(input_file, output_dir)
    
    # Verify
    assert count == 2, f"Expected 2 cards, got {count}"
    
    cards = list(output_dir.glob('*.json'))
    cards = [c for c in cards if not c.name.startswith('cards_summary')]
    assert len(cards) == 2, f"Expected 2 card files, found {len(cards)}"
    
    # Check card structure
    with open(cards[0], 'r') as f:
        card = json.load(f)
    
    required_fields = ['player', 'identity', 'offense', 'defense', 'impact', 'metadata']
    for field in required_fields:
        assert field in card, f"Missing field: {field}"
    
    print("  [PASS] Card creation works")
    return output_dir, tmpdir


def test_value_players(cards_dir):
    """Test player valuation"""
    print("\nTesting value_players.py...")
    
    # Import and run
    from value_players import PlayerValuator
    
    valuator = PlayerValuator()
    
    card_paths = list(Path(cards_dir).glob('*.json'))
    card_paths = [p for p in card_paths if not p.name.startswith('cards_summary')]
    
    print(f"  Found {len(card_paths)} card files")
    
    results = []
    for card_path in card_paths:
        try:
            print(f"  Processing {card_path.name}...")
            card = valuator.load_player_card(card_path)
            result = valuator.valuate_player(card)
            report = valuator.generate_report(result)
            results.append(report)
        except Exception as e:
            print(f"  Error processing {card_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    assert len(results) >= 1, f"Expected at least 1 valuation, got {len(results)}"
    
    # Check report structure
    if results:
        report = results[0]
        required_sections = ['player', 'impact', 'market_value', 'contract', 'trade_value', 'aging']
        for section in required_sections:
            assert section in report, f"Missing section: {section}"
    
    print(f"  [PASS] Player valuation works ({len(results)} players)")


def test_analyze_players(cards_dir):
    """Test player analysis"""
    print("\nTesting analyze_players.py...")
    
    # Import and run
    from analyze_players import analyze_player, load_player_card
    
    card_paths = list(Path(cards_dir).glob('*.json'))
    card_paths = [p for p in card_paths if not p.name.startswith('cards_summary')]
    
    results = []
    for card_path in card_paths:
        try:
            card = load_player_card(card_path)
            analysis = analyze_player(card)
            results.append(analysis)
        except Exception as e:
            print(f"  Error processing {card_path.name}: {e}")
            continue
    
    assert len(results) >= 1, f"Expected at least 1 analysis, got {len(results)}"
    
    # Check analysis structure
    if results:
        analysis = results[0]
        required_sections = ['scouting_report', 'breakout_potential', 'defense_portability', 'impact_sanity']
        for section in required_sections:
            assert section in analysis, f"Missing section: {section}"
    
    print(f"  [PASS] Player analysis works ({len(results)} players)")


def test_utils():
    """Test utility functions"""
    print("\nTesting utils.py...")
    
    from utils import safe_float, safe_int, clamp, normalize, sanitize_filename
    
    # Test safe conversions
    assert safe_float("3.14") == 3.14
    assert safe_float(None, 0.0) == 0.0
    assert safe_int("42") == 42
    assert safe_int("1,234") == 1234
    
    # Test clamp
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    
    # Test normalize
    assert normalize(5, 0, 10) == 0.5
    assert normalize(0, 0, 10) == 0.0
    assert normalize(10, 0, 10) == 1.0
    
    # Test sanitize
    assert sanitize_filename("Test Player") == "Test_Player"
    assert sanitize_filename("Test/Player") == "Test_Player"
    
    print("  [PASS] Utilities work")


def test_models():
    """Test data models"""
    print("\nTesting models.py...")
    
    from models import PlayerCard, PlayerInfo, validate_player_card
    
    # Test PlayerInfo
    player_info = PlayerInfo(
        id="test123",
        name="Test Player",
        team="LAL",
        season=2025,
        position="SG",
        age=25.0
    )
    
    assert player_info.name == "Test Player"
    assert player_info.to_dict()['team'] == "LAL"
    
    # Test validation
    valid_card = {
        'player': {},
        'identity': {},
        'offense': {},
        'defense': {},
        'impact': {},
        'metadata': {}
    }
    
    assert validate_player_card(valid_card) == True
    
    invalid_card = {
        'player': {},
        'identity': {}
    }
    
    assert validate_player_card(invalid_card) == False
    
    print("  [PASS] Models work")


def main():
    """Run all tests"""
    print("="*60)
    print("Testing Simplified NBA-VAR System")
    print("="*60)
    
    tmpdir = None
    
    try:
        # Test utilities and models first
        test_utils()
        test_models()
        
        # Test card creation (returns temp dir with cards)
        cards_dir, tmpdir = test_create_cards()
        
        # Test valuation and analysis with created cards
        test_value_players(cards_dir)
        test_analyze_players(cards_dir)
        
        print("\n" + "="*60)
        print("SUCCESS: All tests PASSED!")
        print("="*60)
        print("\nThe simplified system is working correctly.")
        print("\nNext steps:")
        print("  1. Prepare your player data (CSV or Parquet)")
        print("  2. Run: python create_cards.py --input your_data.csv --output data/cards")
        print("  3. Run: python value_players.py --cards data/cards --output valuations")
        print("  4. Run: python analyze_players.py --cards data/cards --output analysis")
        
        return 0
        
    except AssertionError as e:
        print(f"\nFAILED: Test assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup temp directory
        if tmpdir and Path(tmpdir).exists():
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    exit(main())
