# NBA-VAR Simplified - Documentation Index

## Start Here

👉 **New users:** Start with [OVERVIEW.md](OVERVIEW.md) for a quick introduction

👉 **Ready to use:** Follow [QUICK_START.md](QUICK_START.md) for step-by-step instructions

👉 **Need details:** Read [README.md](README.md) for complete documentation

## Documentation Files

### Getting Started
- **[OVERVIEW.md](OVERVIEW.md)** - Quick introduction and key features
- **[QUICK_START.md](QUICK_START.md)** - Step-by-step guide with examples
- **[README.md](README.md)** - Complete system documentation

### Technical Details
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project structure and file descriptions
- **[SIMPLIFICATION_SUMMARY.md](SIMPLIFICATION_SUMMARY.md)** - Technical summary of simplification

### Configuration
- **[config.yaml](config.yaml)** - System configuration parameters

## Core Files

### Main Scripts
- **[create_cards.py](create_cards.py)** - Generate player cards from data
- **[value_players.py](value_players.py)** - Contract valuation & aging curves
- **[analyze_players.py](analyze_players.py)** - Comprehensive player analysis

### Supporting Code
- **[utils.py](utils.py)** - Shared utility functions
- **[models.py](models.py)** - Data models and schemas

### Testing
- **[test_simplified.py](test_simplified.py)** - Test suite

## Quick Reference

### Installation
```bash
pip install pandas numpy pyyaml
python test_simplified.py  # Verify installation
```

### Basic Usage
```bash
# 1. Generate cards
python create_cards.py --input data.csv --output cards/

# 2. Run valuation
python value_players.py --cards cards/ --output valuations/

# 3. Run analysis
python analyze_players.py --cards cards/ --output analysis/
```

### Get Help
```bash
python create_cards.py --help
python value_players.py --help
python analyze_players.py --help
```

## Navigation Guide

### I want to...

**...understand what this system does**
→ Read [OVERVIEW.md](OVERVIEW.md)

**...get started quickly**
→ Follow [QUICK_START.md](QUICK_START.md)

**...understand the complete system**
→ Read [README.md](README.md)

**...understand the code structure**
→ Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**...understand the simplification**
→ Read [SIMPLIFICATION_SUMMARY.md](SIMPLIFICATION_SUMMARY.md)

**...customize the system**
→ Edit [config.yaml](config.yaml)

**...see the original system**
→ Check [legacy/](legacy/)

**...run tests**
→ Run `python test_simplified.py`

**...generate player cards**
→ Run `python create_cards.py --input data.csv --output cards/`

**...value players**
→ Run `python value_players.py --cards cards/ --output valuations/`

**...analyze players**
→ Run `python analyze_players.py --cards cards/ --output analysis/`

## File Organization

```
Root Directory
├── Documentation (5 files)
│   ├── OVERVIEW.md              ← Start here
│   ├── QUICK_START.md           ← Step-by-step guide
│   ├── README.md                ← Complete docs
│   ├── PROJECT_STRUCTURE.md     ← Code structure
│   └── SIMPLIFICATION_SUMMARY.md ← Technical details
│
├── Core System (5 files)
│   ├── create_cards.py          ← Generate cards
│   ├── value_players.py         ← Valuation
│   ├── analyze_players.py       ← Analysis
│   ├── utils.py                 ← Utilities
│   └── models.py                ← Data models
│
├── Configuration & Testing
│   ├── config.yaml              ← Configuration
│   └── test_simplified.py       ← Tests
│
├── Other
│   ├── INDEX.md                 ← This file
│   └── .gitignore               ← Git ignore
│
└── legacy/                      ← Original system (archived)
```

## Common Tasks

### First Time Setup
1. Read [OVERVIEW.md](OVERVIEW.md)
2. Install dependencies: `pip install pandas numpy pyyaml`
3. Run tests: `python test_simplified.py`
4. Follow [QUICK_START.md](QUICK_START.md)

### Daily Usage
1. Prepare your data (CSV or Parquet)
2. Run `create_cards.py` to generate cards
3. Run `value_players.py` for valuation
4. Run `analyze_players.py` for analysis
5. Review outputs

### Customization
1. Edit [config.yaml](config.yaml) for parameters
2. Modify core files for new features
3. Update [models.py](models.py) for new data structures
4. Run tests to verify changes

### Troubleshooting
1. Check [QUICK_START.md](QUICK_START.md) troubleshooting section
2. Run `python test_simplified.py` to verify setup
3. Review [README.md](README.md) for detailed information
4. Check [legacy/](legacy/) for reference implementation

## Key Concepts

### Archetypes (7 types)
Player types with unique aging curves:
- initiator_creator, shooting_specialist, rim_protector
- versatile_wing, connector, athletic_finisher, combo_guard

### Aging Phases (4 phases)
- pre_growth → growth → plateau → decline

### Analysis Types (4 types)
- Breakout detection
- Defense portability
- Impact sanity checks
- Scouting reports

### Valuation Metrics
- Wins added
- Market value
- Contract surplus
- Trade value bands

## Support

- **Documentation:** Read the 5 documentation files
- **Examples:** Check [QUICK_START.md](QUICK_START.md)
- **Testing:** Run `python test_simplified.py`
- **Reference:** Check [legacy/](legacy/) for original implementation

## Version

- **System:** NBA-VAR Simplified v1.0
- **Files:** 13 files + 1 directory
- **Lines:** ~1,300 lines (core system)
- **Reduction:** 90% smaller than original

---

**Ready to start?** → [QUICK_START.md](QUICK_START.md)

**Need overview?** → [OVERVIEW.md](OVERVIEW.md)

**Want details?** → [README.md](README.md)
