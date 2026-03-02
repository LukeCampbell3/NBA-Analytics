# NBA-VAR Project Structure

## Root Directory (Simplified System)

```
.
├── README.md                    # Main documentation
├── QUICK_START.md              # Quick start guide
├── SIMPLIFICATION_SUMMARY.md   # Technical summary
├── PROJECT_STRUCTURE.md        # This file
│
├── create_cards.py             # Generate player cards from data
├── value_players.py            # Contract valuation & aging curves
├── analyze_players.py          # Comprehensive player analysis
├── utils.py                    # Shared utilities
├── models.py                   # Data models & schemas
├── config.yaml                 # Configuration
│
├── test_simplified.py          # Test suite
├── .gitignore                  # Git ignore rules
│
└── legacy/                     # Original complex system (archived)
    ├── README.md               # Legacy system documentation
    ├── core/                   # Original modular implementation
    ├── nba_var/                # NBA-VAR 7-engine architecture
    ├── data/                   # Data pipeline scripts
    └── ...                     # Other legacy files
```

## Core Files (5 files, ~1,300 lines)

### 1. create_cards.py (~180 lines)
**Purpose:** Generate player cards from raw statistics

**Input:** CSV or Parquet file with player stats
**Output:** Individual JSON player cards

**Key Functions:**
- `load_player_data()` - Load data from file
- `calculate_identity()` - Determine usage band and archetype
- `calculate_offense()` - Compute offensive metrics
- `calculate_defense()` - Compute defensive metrics
- `calculate_impact()` - Estimate impact metrics
- `create_player_card()` - Generate complete card
- `generate_cards()` - Main entry point

**Usage:**
```bash
python create_cards.py --input data.csv --output cards/
```

### 2. value_players.py (~280 lines)
**Purpose:** Player valuation with contract analysis and aging curves

**Input:** Player cards (JSON)
**Output:** Valuation reports with trade value bands

**Key Classes:**
- `PlayerValuator` - Main valuation engine

**Key Methods:**
- `convert_impact_to_wins()` - Convert metrics to wins added
- `calculate_aging_multiplier()` - Apply aging curves
- `calculate_market_value()` - Compute market value
- `calculate_surplus()` - Calculate contract surplus
- `valuate_player()` - Main valuation function

**Usage:**
```bash
python value_players.py --cards cards/ --output valuations/
```

### 3. analyze_players.py (~350 lines)
**Purpose:** Comprehensive player analysis

**Input:** Player cards (JSON)
**Output:** Analysis reports with breakout/defense/impact assessments

**Key Functions:**
- `detect_breakout_potential()` - Identify growth candidates
- `analyze_defense_portability()` - Assess defensive versatility
- `check_impact_sanity()` - Validate impact metrics
- `generate_scouting_report()` - Create scouting summary
- `analyze_player()` - Run all analyses

**Usage:**
```bash
python analyze_players.py --cards cards/ --output analysis/
```

### 4. utils.py (~200 lines)
**Purpose:** Shared utility functions

**Key Functions:**
- Type conversions: `safe_float()`, `safe_int()`, `safe_str()`
- Math utilities: `clamp()`, `normalize()`, `weighted_average()`
- File I/O: `load_json()`, `save_json()`, `find_files()`
- String utilities: `sanitize_filename()`, `format_currency()`
- Statistics: `summarize_stats()`, `calculate_percentile()`

### 5. models.py (~250 lines)
**Purpose:** Data models and schemas

**Key Classes:**
- `PlayerCard` - Complete player card structure
- `PlayerInfo` - Basic player information
- `ValuationResult` - Valuation output
- `BreakoutAnalysis` - Breakout detection result
- `DefensePortability` - Defense analysis result

**Key Enums:**
- `Archetype` - Player archetypes (7 types)
- `UsageBand` - Usage levels (high/med/low)
- `AgingPhase` - Aging phases (4 phases)

## Configuration (config.yaml)

```yaml
market:
  dollars_per_win: 3.5
  salary_cap: 141.0
  discount_rate: 0.08

aging:
  default_peak_age: 28.0
  growth_rate: 0.07
  decline_rate: -0.05

thresholds:
  breakout:
    min_opportunity_score: 40
    min_signal_strength: 35
  defense:
    high_portability: 0.7
```

## Workflow

```
Raw Data (CSV/Parquet)
    ↓
[create_cards.py]
    ↓
Player Cards (JSON)
    ↓
    ├─→ [value_players.py] → Valuation Reports
    └─→ [analyze_players.py] → Analysis Reports
```

## Data Flow

1. **Input:** Player statistics (CSV/Parquet)
   - Required: name, team, season, position, age, basic stats
   - Optional: advanced metrics, physical attributes

2. **Player Cards:** Structured JSON profiles
   - Player info, identity, offense, defense, impact
   - Trust/uncertainty scores, metadata

3. **Valuation:** Contract and aging analysis
   - Wins added, market value, surplus value
   - Trade value bands, aging projections

4. **Analysis:** Comprehensive assessments
   - Breakout potential, defense portability
   - Impact sanity checks, scouting reports

## Testing

```bash
python test_simplified.py
```

Tests cover:
- ✓ Utility functions
- ✓ Data models
- ✓ Card creation
- ✓ Player valuation
- ✓ Player analysis

## Legacy System

The `legacy/` directory contains the original implementation:
- 85+ files, 15,000+ lines
- Complex multi-engine architecture
- Preserved for reference and advanced features

## Quick Reference

| Task | Command |
|------|---------|
| Generate cards | `python create_cards.py --input data.csv --output cards/` |
| Run valuation | `python value_players.py --cards cards/ --output valuations/` |
| Run analysis | `python analyze_players.py --cards cards/ --output analysis/` |
| Run tests | `python test_simplified.py` |
| View config | `cat config.yaml` |

## File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| create_cards.py | 180 | Card generation |
| value_players.py | 280 | Valuation & aging |
| analyze_players.py | 350 | Analysis tools |
| utils.py | 200 | Utilities |
| models.py | 250 | Data models |
| **Total** | **~1,260** | **Complete system** |

## Dependencies

```bash
pip install pandas numpy pyyaml
pip install pyarrow  # Optional: Parquet support
```

## Documentation

- **README.md** - Complete system documentation
- **QUICK_START.md** - Quick start guide with examples
- **SIMPLIFICATION_SUMMARY.md** - Technical details of simplification
- **PROJECT_STRUCTURE.md** - This file (project structure)
- **legacy/README.md** - Legacy system documentation

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review QUICK_START.md for examples
3. Run test_simplified.py to verify setup
4. Check legacy/ for reference implementation
