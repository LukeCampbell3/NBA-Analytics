# NBA-VAR Simplified - Quick Start Guide

## Installation

```bash
# Install dependencies
pip install pandas numpy pyyaml

# Optional: For Parquet support
pip install pyarrow
```

## Verify Installation

```bash
python test_simplified.py
```

Expected output:
```
✅ All tests PASSED!
```

## Basic Workflow

### Step 1: Prepare Your Data

Your CSV or Parquet file should have these columns:

**Required:**
- `player_name` or `name`
- `team`
- `season`
- `position`
- `age`
- `points_per_game`
- `assists_per_game`
- `rebounds_per_game`
- `minutes_per_game`
- `games_played`

**Recommended:**
- `steals_per_game`, `blocks_per_game`, `turnovers_per_game`
- `field_goal_attempts_per_game`, `three_point_attempts_per_game`
- `usage_rate`, `plus_minus`

### Step 2: Generate Player Cards

```bash
python create_cards.py --input data/your_stats.csv --output data/cards
```

This creates:
- Individual JSON files for each player
- `cards_summary.json` with metadata

### Step 3: Run Valuation

```bash
python value_players.py --cards data/cards --output valuations
```

This creates:
- Individual valuation reports for each player
- `valuation_summary.json` with top surplus players

### Step 4: Run Analysis

```bash
python analyze_players.py --cards data/cards --output analysis
```

This creates:
- Individual analysis reports for each player
- `analysis_summary.json` with interesting cases

## Example with Sample Data

```bash
# Create sample data
cat > sample_data.csv << EOF
player_name,team,season,position,age,points_per_game,assists_per_game,rebounds_per_game,steals_per_game,blocks_per_game,turnovers_per_game,field_goal_attempts_per_game,three_point_attempts_per_game,minutes_per_game,games_played,usage_rate,plus_minus
LeBron James,LAL,2025,SF,40.0,25.7,8.3,7.3,1.3,0.5,3.5,18.5,5.2,35.5,71,0.31,5.2
Stephen Curry,GSW,2025,PG,36.5,26.4,5.1,4.5,0.9,0.4,3.0,18.7,11.7,32.7,74,0.32,7.8
EOF

# Run the workflow
python create_cards.py --input sample_data.csv --output data/cards
python value_players.py --cards data/cards --output valuations
python analyze_players.py --cards data/cards --output analysis

# Check results
cat valuations/valuation_summary.json
cat analysis/analysis_summary.json
```

## Understanding the Outputs

### Player Card Structure

```json
{
  "player": {
    "name": "LeBron James",
    "team": "LAL",
    "age": 40.0,
    "position": "SF"
  },
  "identity": {
    "usage_band": "high",
    "primary_archetype": "initiator_creator"
  },
  "offense": {
    "shot_profile": { "three_rate": 0.28, "volume": 18.5 },
    "creation": { "scoring": 25.7, "playmaking": 8.3 }
  },
  "defense": {
    "burden": { "level": "med", "score": 0.6 }
  },
  "impact": {
    "net": 5.2,
    "offensive": 2.57,
    "defensive": 0.9
  }
}
```

### Valuation Report Structure

```json
{
  "player": { "name": "LeBron James" },
  "impact": { "wins_added": 4.2 },
  "market_value": {
    "by_year": {
      "2025": 14.7,
      "2026": 13.2,
      "2027": 11.5
    }
  },
  "contract": {
    "surplus_by_year": { "2025": 3.2, "2026": 1.8 },
    "npv_surplus": 4.5
  },
  "trade_value": {
    "low": 3.2,
    "base": 4.5,
    "high": 5.8
  },
  "aging": {
    "current_phase": "decline",
    "peak_age": 28.5
  }
}
```

### Analysis Report Structure

```json
{
  "scouting_report": {
    "role_summary": "High-usage initiator_creator",
    "strengths": ["High-volume scorer", "Strong positive impact"],
    "weaknesses": []
  },
  "breakout_potential": {
    "can_breakout": false,
    "opportunity_score": 28.0,
    "signal_strength": 85.5
  },
  "defense_portability": {
    "defensive_role": "wing_defender",
    "portability": { "score": 0.72, "level": "high" }
  },
  "impact_sanity": {
    "sanity_level": "pass",
    "flags": []
  }
}
```

## Common Use Cases

### 1. Find Breakout Candidates

```bash
python analyze_players.py --cards data/cards --output analysis
cat analysis/analysis_summary.json | grep -A 10 "breakout_candidates"
```

### 2. Identify Contract Value

```bash
python value_players.py --cards data/cards --output valuations
cat valuations/valuation_summary.json | grep -A 20 "top_surplus_players"
```

### 3. Assess Defensive Versatility

```bash
python analyze_players.py --cards data/cards --output analysis
cat analysis/analysis_summary.json | grep -A 10 "high_portability_defenders"
```

### 4. Generate Single Player Report

```bash
# Create card for one player
python create_cards.py --input data/stats.csv --output data/cards --limit 1

# Get their valuation
python value_players.py --cards data/cards/Player_Name_TEAM_2025.json --output valuations

# Get their analysis
python analyze_players.py --cards data/cards/Player_Name_TEAM_2025.json --output analysis
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Market parameters
market:
  dollars_per_win: 3.5  # Adjust based on current market
  salary_cap: 141.0     # Update for current season
  discount_rate: 0.08   # Change discount rate

# Analysis thresholds
thresholds:
  breakout:
    min_opportunity_score: 40  # Lower to find more candidates
    min_signal_strength: 35
    min_confidence: 0.5
  
  defense:
    high_portability: 0.7  # Adjust portability thresholds
    medium_portability: 0.5
```

## Troubleshooting

### Issue: "No cards found"

**Solution:** Check that your input file has the required columns and data.

```bash
# Verify your data
head -5 data/your_stats.csv
```

### Issue: "Missing field errors"

**Solution:** Ensure your CSV has all required columns. Use default values if needed:

```python
# In your CSV, add missing columns with defaults
df['usage_rate'] = df.get('usage_rate', 0.20)
df['plus_minus'] = df.get('plus_minus', 0.0)
```

### Issue: "Import errors"

**Solution:** Install missing dependencies:

```bash
pip install pandas numpy pyyaml
```

## Advanced Usage

### Batch Processing

```bash
# Process multiple seasons
for year in 2023 2024 2025; do
  python create_cards.py --input data/stats_${year}.csv --output data/cards_${year}
  python value_players.py --cards data/cards_${year} --output valuations_${year}
  python analyze_players.py --cards data/cards_${year} --output analysis_${year}
done
```

### Custom Archetypes

Edit `models.py` to add new archetypes:

```python
class Archetype(str, Enum):
    # ... existing archetypes ...
    MY_NEW_ARCHETYPE = "my_new_archetype"
```

Then update `value_players.py` to add aging curve:

```python
self.aging_curves = {
    # ... existing curves ...
    "my_new_archetype": {
        "peak": 29.0,
        "growth": (21, 27),
        "plateau": (27, 32),
        "decline": -0.04
    }
}
```

## Getting Help

1. Read `README_SIMPLIFIED.md` for complete documentation
2. Check `SIMPLIFICATION_SUMMARY.md` for technical details
3. Run `python test_simplified.py` to verify your setup
4. Review example outputs in test results

## Next Steps

1. ✅ Install dependencies
2. ✅ Run tests
3. ✅ Prepare your data
4. ✅ Generate cards
5. ✅ Run valuation
6. ✅ Run analysis
7. ✅ Review outputs
8. ✅ Customize config
9. ✅ Iterate and refine

Happy analyzing! 🏀
