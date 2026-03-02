# NBA-VAR Simplified

A minimal, concise NBA player valuation and analysis system.

## Overview

This repository provides a streamlined approach to NBA player evaluation with three core components:

1. **Player Card Generation** - Create comprehensive player profiles from raw data
2. **Player Valuation** - Calculate contract value and aging projections
3. **Player Analysis** - Breakout detection, defense portability, and impact sanity checks

## Architecture

The simplified system uses just 5 core files:

```
├── create_cards.py          # Generate player cards from data
├── value_players.py         # Contract valuation & aging curves
├── analyze_players.py       # Comprehensive player analysis
├── utils.py                 # Shared utilities
├── models.py                # Data models
└── config.yaml              # Configuration
```

## Features

### 1. Player Cards (`create_cards.py`)
Consolidates card generation logic from multiple modules:
- Generate player profiles from raw statistics (CSV/Parquet)
- Calculate offensive/defensive metrics
- Classify player archetypes
- Compute trust/uncertainty scores
- Output: Individual JSON cards + summary

### 2. Player Valuation (`value_players.py`)
Combines contract valuation and aging curve analysis:
- Convert impact metrics to wins added
- Calculate market value using $/win conversion
- Estimate contract surplus value (market value - salary)
- Apply archetype-based aging curves
- Generate trade value bands (low/base/high)
- Output: Valuation reports + top surplus players

### 3. Player Analysis (`analyze_players.py`)
Comprehensive analysis combining multiple tools:
- **Breakout Detection**: Identify players with growth potential
- **Defense Portability**: Assess defensive versatility and matchup flexibility
- **Impact Sanity Check**: Validate impact metrics for consistency
- **Scouting Report**: Generate strengths/weaknesses summary
- Output: Individual analysis + summary of interesting cases

## Quick Start

### 1. Generate Player Cards

```bash
python create_cards.py --input data/raw/stats.csv --output data/player_cards
```

Options:
- `--input`: Input CSV or Parquet file with player stats
- `--output`: Output directory for player cards (default: `data/player_cards`)
- `--limit`: Limit number of cards to generate (optional)

### 2. Run Valuation Analysis

```bash
python value_players.py --cards data/player_cards --output valuations/
```

Options:
- `--cards`: Player cards directory or single card file
- `--output`: Output directory for valuation reports (default: `valuations`)

### 3. Run Comprehensive Analysis

```bash
python analyze_players.py --cards data/player_cards --output analysis/
```

Options:
- `--cards`: Player cards directory or single card file
- `--output`: Output directory for analysis reports (default: `analysis`)

## Data Requirements

### Input Data Format

Player statistics file (CSV or Parquet) should include:

**Required fields:**
- `player_name` or `name`: Player name
- `team`: Team abbreviation
- `season`: Season year
- `position`: Position (PG, SG, SF, PF, C)
- `age`: Player age

**Performance metrics:**
- `points_per_game`, `assists_per_game`, `rebounds_per_game`
- `steals_per_game`, `blocks_per_game`, `turnovers_per_game`
- `field_goal_attempts_per_game`, `three_point_attempts_per_game`
- `minutes_per_game`, `games_played`
- `usage_rate`, `plus_minus`

**Optional fields:**
- `player_id`: Unique player identifier
- `height_in`: Height in inches
- `weight_lb`: Weight in pounds
- `defensive_rebounds_per_game`

## Output Structure

### Player Cards
```json
{
  "player": {
    "id": "...",
    "name": "...",
    "team": "...",
    "season": 2025,
    "position": "SG",
    "age": 25.5
  },
  "identity": {
    "usage_band": "med",
    "primary_archetype": "shooting_specialist",
    "position": "SG"
  },
  "offense": { ... },
  "defense": { ... },
  "impact": { ... },
  "metadata": { ... },
  "trust": { ... },
  "uncertainty": { ... }
}
```

### Valuation Reports
```json
{
  "player": { ... },
  "impact": {
    "wins_added": 3.2
  },
  "market_value": {
    "by_year": { "2025": 11.2, "2026": 11.8, ... }
  },
  "contract": {
    "surplus_by_year": { ... },
    "npv_surplus": 8.5
  },
  "trade_value": {
    "low": 6.8,
    "base": 8.5,
    "high": 10.2
  },
  "aging": {
    "current_phase": "growth",
    "peak_age": 30.0,
    "multipliers": { ... }
  }
}
```

### Analysis Reports
```json
{
  "scouting_report": {
    "role_summary": "Medium-usage shooting_specialist",
    "strengths": [ ... ],
    "weaknesses": [ ... ]
  },
  "breakout_potential": {
    "can_breakout": true,
    "opportunity_score": 72.5,
    "signal_strength": 68.3
  },
  "defense_portability": {
    "defensive_role": "wing_defender",
    "portability": {
      "score": 0.75,
      "level": "high"
    }
  },
  "impact_sanity": {
    "sanity_level": "pass",
    "flags": []
  }
}
```

## Configuration

Edit `config.yaml` to customize:
- Market parameters ($/win, salary cap, discount rate)
- Aging curve defaults
- Analysis thresholds (breakout, defense, trust)
- Data quality requirements
- Output settings

## Archetypes

Supported player archetypes:
- `initiator_creator`: Primary ball handlers, playmakers
- `shooting_specialist`: 3-point specialists, catch-and-shoot
- `rim_protector`: Centers, shot blockers
- `versatile_wing`: 3&D, multi-position defenders
- `connector`: Role players, high IQ glue guys
- `athletic_finisher`: Slashers, rim runners
- `combo_guard`: Secondary creators

Each archetype has unique aging curves and valuation parameters.

## Aging Phases

Players progress through aging phases:
- `pre_growth`: Very young, developing (< 20)
- `growth`: Rapid improvement phase (20-26)
- `plateau`: Peak performance window (26-32)
- `decline`: Gradual decline phase (32+)

Timing varies by archetype (e.g., shooters peak later, athletic players earlier).

## Dependencies

```bash
pip install pandas numpy pyyaml
```

Optional:
```bash
pip install pyarrow  # For Parquet support
```

## Example Workflow

```bash
# 1. Generate cards from your data
python create_cards.py --input data/nba_stats_2025.csv --output data/cards

# 2. Run valuation
python value_players.py --cards data/cards --output reports/valuations

# 3. Run analysis
python analyze_players.py --cards data/cards --output reports/analysis

# 4. Check outputs
ls reports/valuations/valuation_summary.json
ls reports/analysis/analysis_summary.json
```

## Legacy System

The original complex system is preserved in:
- `core/`: Original modular implementation
- `nba_var/`: NBA-VAR 7-engine architecture
- `data/`: Data pipeline and processing

This simplified system consolidates that functionality into minimal files for easier use and maintenance.

## License

MIT License
