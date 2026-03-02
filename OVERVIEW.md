# NBA-VAR Simplified - Overview

## What Is This?

A minimal, concise NBA player valuation and analysis system that consolidates complex functionality into 5 simple Python files.

## Quick Stats

- **5 core files** (~1,300 lines total)
- **3 commands** to run complete analysis
- **7 archetypes** with unique aging curves
- **4 analysis types** (breakout, defense, impact, scouting)
- **100% tested** (all tests pass)

## The 3-Step Workflow

```bash
# 1. Generate player cards
python create_cards.py --input data.csv --output cards/

# 2. Run valuation
python value_players.py --cards cards/ --output valuations/

# 3. Run analysis
python analyze_players.py --cards cards/ --output analysis/
```

## What You Get

### From create_cards.py
- Player identity (archetype, usage band)
- Offensive metrics (shot profile, creation, efficiency)
- Defensive metrics (burden, performance)
- Impact estimates (net, offensive, defensive)
- Trust/uncertainty scores

### From value_players.py
- Wins added calculation
- Market value by year (5-year projection)
- Contract surplus analysis
- Trade value bands (low/base/high)
- Aging phase identification

### From analyze_players.py
- Breakout potential detection
- Defense portability assessment
- Impact sanity checks
- Scouting reports (strengths/weaknesses)

## Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `create_cards.py` | 180 | Generate player cards |
| `value_players.py` | 280 | Valuation & aging |
| `analyze_players.py` | 350 | Analysis tools |
| `utils.py` | 200 | Utilities |
| `models.py` | 250 | Data models |

## Key Features

### Archetypes (7 types)
Each with unique aging curves and valuation parameters:
- **initiator_creator** - Primary ball handlers (peak: 28.5)
- **shooting_specialist** - 3-point specialists (peak: 30.0)
- **rim_protector** - Centers, shot blockers (peak: 27.0)
- **versatile_wing** - 3&D defenders (peak: 29.0)
- **connector** - High-IQ role players (peak: 31.0)
- **athletic_finisher** - Slashers, rim runners (peak: 26.5)
- **combo_guard** - Secondary creators (peak: 29.0)

### Aging Phases (4 phases)
- **pre_growth** - Very young, developing
- **growth** - Rapid improvement (20-26)
- **plateau** - Peak performance (26-32)
- **decline** - Gradual decline (32+)

### Analysis Types
- **Breakout Detection** - Opportunity + signal + confidence
- **Defense Portability** - Switchability + versatility
- **Impact Sanity** - Metric consistency checks
- **Scouting Reports** - Strengths/weaknesses summary

## Installation

```bash
# Install dependencies
pip install pandas numpy pyyaml

# Optional: Parquet support
pip install pyarrow

# Verify installation
python test_simplified.py
```

## Example Output

### Player Card
```json
{
  "player": {"name": "LeBron James", "age": 40.0},
  "identity": {"usage_band": "high", "primary_archetype": "initiator_creator"},
  "impact": {"net": 5.2, "offensive": 2.57, "defensive": 0.9}
}
```

### Valuation
```json
{
  "impact": {"wins_added": 4.2},
  "market_value": {"2025": 14.7, "2026": 13.2},
  "trade_value": {"low": 3.2, "base": 4.5, "high": 5.8},
  "aging": {"current_phase": "decline", "peak_age": 28.5}
}
```

### Analysis
```json
{
  "breakout_potential": {"can_breakout": false, "opportunity_score": 28.0},
  "defense_portability": {"portability": {"score": 0.72, "level": "high"}},
  "impact_sanity": {"sanity_level": "pass"}
}
```

## Documentation

- **README.md** - Complete documentation
- **QUICK_START.md** - Quick start guide
- **PROJECT_STRUCTURE.md** - Project structure
- **SIMPLIFICATION_SUMMARY.md** - Technical details
- **OVERVIEW.md** - This file

## Configuration

Edit `config.yaml` to customize:
- Market parameters ($/win, salary cap)
- Aging curve defaults
- Analysis thresholds
- Data quality requirements

## Testing

```bash
python test_simplified.py
```

All tests pass:
- ✓ Utilities work
- ✓ Models work
- ✓ Card creation works
- ✓ Player valuation works
- ✓ Player analysis works

## Legacy System

The original complex system (85+ files, 15,000+ lines) is archived in `legacy/` for reference.

## Use Cases

### Find Breakout Candidates
Players with high opportunity scores and growth potential

### Identify Contract Value
Players with positive surplus (market value > salary)

### Assess Defensive Versatility
Players with high portability scores

### Generate Scouting Reports
Comprehensive player profiles with strengths/weaknesses

## Why Simplified?

**Before:** 85+ files, 15,000+ lines, complex orchestration
**After:** 5 files, 1,300 lines, simple workflow

**Benefits:**
- ✅ Easier to understand
- ✅ Easier to use
- ✅ Easier to maintain
- ✅ Faster to run
- ✅ Better documented

## Getting Started

1. Read **README.md** for complete documentation
2. Follow **QUICK_START.md** for step-by-step guide
3. Run `python test_simplified.py` to verify setup
4. Prepare your data (CSV or Parquet)
5. Run the 3-step workflow
6. Review outputs and iterate

## Support

- Check documentation files for detailed information
- Run tests to verify your setup
- Review example outputs in test results
- Check `legacy/` for reference implementation

## License

MIT License

---

**Ready to start?** → See **QUICK_START.md**

**Need details?** → See **README.md**

**Want structure?** → See **PROJECT_STRUCTURE.md**
