# NBA Parlay Subsystem

Robust parlay creation subsystem for NBA-Analytics / Player-Predictor system.

**Status: SHADOW MODE ONLY** - All operations are non-production. The system does not alter production selections, staking, live betting, or promotion behavior.

## Core Principle

> A parlay is not a group of picks.
>
> A parlay is a joint probability position over shared and independent game states.

The system does not ask: "Which predictions look good?"

It asks: "Which priced pre-event binary events belong to reliable/mispriced state regions, and which combinations remain positive-EV after joint-state stress?"

## Architecture Overview

### Phases

1. **Phase 1: Build Priced Binary-Event Universe**
   - Input: scenario probabilities, forecastability board, player state, odds snapshots
   - Output: `priced_event_universe_latest.csv`
   - Each row represents one priced binary event with:
     - Identity (player, game, market, side, line, book, time)
     - Price (American, decimal, implied, break-even)
     - Distribution (quantiles, percentile)
     - Edge metrics (raw, robust, LCB)
     - Reliability metrics (forecastability, plan reliability, scenario agreement)
     - Scenario breakdown (positive/negative masses, failure modes)
     - News status

2. **Phase 2: Line-Zone and Alternate-Line Scanner**
   - Scans all available lines for each player/market
   - Finds best binary framing by robust EV or LCB edge
   - Classifies lines into zones (NEAR_MEDIAN, TAIL, etc.)
   - Supports main lines, alternate lines, both sides

3. **Phase 3: Single-Leg Set Membership with Tiers**
   - Evaluates each leg for acceptance into bettable set B_t
   - Tiers:
     - **SEED_PLAYABLE**: High conviction (high LCB edge, high forecastability, low chaos)
     - **BALANCED_PLAYABLE**: Acceptable edge (positive LCB, positive robust EV)
     - **PRICE_DEPENDENT**: Model valid, but price not current sufficient
     - **NEWS_DEPENDENT**: Model valid, but injury/lineup not resolved
     - **BOUNDARY_SHADOW**: Close to acceptance (for validation)
     - **PASS**: Fails checks
   - Computes min_acceptable_odds for PRICE_DEPENDENT legs

4. **Phase 4: Anchor + Companion Parlay Construction**
   - Selects anchor legs from SEED and strong BALANCED tiers
   - Finds compatible companion legs (not Cartesian product)
   - Excludes PRICE_DEPENDENT companions if current price doesn't meet min odds
   - Excludes NEWS_DEPENDENT legs until news clarity
   - Validates 2-leg subsets before allowing 3-leg

5. **Phase 5: Shared Event-Supply Engine**
   - Detects when legs draw from same limited event pool
   - Event pools: rebounds, assists, points, three-pointers, steals, blocks, FT, turnovers
   - Rejects parlays with excessive shared pool conflict
   - Applies shared_event_supply_penalty

6. **Phase 6: Team Environment Failure Modes**
   - (Implemented in Phase 10 stress engine)
   - Named failure modes: TEAM_OFFENSE_COLLAPSE, LOW_ASSIST_ENVIRONMENT, PACE_COLLAPSE, etc.
   - Exposure analysis per leg
   - Shared exposure penalty

7. **Phase 7: Joint State and Correlation Engine**
   - (Implemented in Phase 9)
   - Builds joint state object from leg state vectors
   - Identifies shared game state, shared event supply, failure overlap
   - Correlation classes: CROSS_GAME_WEAK, SAME_GAME_PACE_POSITIVE, etc.

8. **Phase 8: Parlay Price Engine**
   - Computes synthetic parlay price (product of decimal odds)
   - Supports book-quoted SGP prices (overrides synthetic if available)
   - Detects SGP payout reduction vs synthetic
   - Validates price validity

9. **Phase 9: Parlay Probability Engine**
   - Naive product of individual probabilities
   - Applies correlation adjustment (cross-game vs same-game)
   - Applies shared event-supply penalty
   - Applies calibration penalty
   - Computes LCB probability

10. **Phase 10: Parlay Stress Engine**
    - Applies stress downshifts to probability
    - Stress factors: plan holds, blowout, foul trouble, minutes loss, role shift, etc.
    - Validates stressed probability > break-even + margin
    - Validates LCB edge remains positive

11. **Phase 11: Parlay Selector**
    - Final scoring (weighted product of quality factors)
    - Tiers: SEED_SHADOW, BALANCED_SHADOW, BOUNDARY_SHADOW
    - Final decision label (accepted or specific rejection reason)

12. **Phase 12: Timing Checkpoints**
    - CHECKPOINT_OPEN, CHECKPOINT_MIDDAY, CHECKPOINT_INJURY_REPORT, CHECKPOINT_LINEUP_CONFIRMED, CHECKPOINT_FINAL_ODDS
    - Supports rerunning at specific times
    - Tracks tier changes and price movements

13. **Phase 13: Validation**
    - Backtests against historical outcomes
    - Compares vs naive stacking approaches
    - Measures hit rate, ROI, calibration, CLV
    - Validates that joint-state filtering improves ROI

14. **Phase 14: Reporting**
    - Generates detailed parlay reports
    - Includes legs, prices, break-even, probabilities, EV
    - Includes joint state analysis, failure modes, event supply risk
    - Explains every acceptance/rejection

15. **Phase 15: Config**
    - YAML policy file: `config/parlay_policy.yaml`
    - Configurable thresholds for all tiers, price, stress, generation

16. **Phase 16: Tests**
    - American odds conversions
    - Break-even calculations
    - Min acceptable odds
    - Line-zone classification
    - Tier membership
    - Anchor + companion generation (no Cartesian product)
    - SGP price handling
    - Shared event-supply rejection
    - Parlay stress EV acceptance
    - 3-leg subset validation

17. **Phase 17: Minimal Vertical Slice (Current Implementation)**
    - Phases 1-4 + 5, 8-11 implemented
    - Cross-game parlays only (2-leg primary, 3-leg future)
    - Synthetic price fallback (book-quoted SGP future)

## Data Contracts

### Input Files

- `scenario_probability_matrix_latest.csv` - Probability distributions by player/game/market
- `forecastability_board_latest.csv` - Forecastability metrics by player/market
- `trusted_player_state_registry_latest.csv` - Current player status, injury, expected minutes
- `current_odds_snapshot.csv` - Current market prices (game_id, player_id, market, side, line, odds, book, snapshot_time)

### Output Files

- `outputs/priced_event_universe_latest.csv` - Complete priced binary event universe
- `outputs/parlays_latest.json` - Final accepted parlays with full details

## Key Concepts

### Break-Even Probability

The probability at which a bet is neutral (no edge). For American odds:
```
break_even_prob = 1 / (1 + abs(odds) / 100) for positive odds
break_even_prob = abs(odds) / (abs(odds) + 100) for negative odds
```

A 50% edge is **playable** at +130 (break-even 43.5%).
A 56% edge is **unplayable** at -150 (break-even 60%).

**The system judges probability relative to price.**

### Single-Leg Bettable Set

$$B_t = P_t \cap R_t \cap C_t \cap M_t \cap Z_t$$

Where:
- $P_t$ = probability/edge set
- $R_t$ = reliability/forecastability set
- $C_t$ = chaos-stable set
- $M_t$ = market-mispricing/positive robust-EV set
- $Z_t$ = price-valid binary framing set

A leg enters $B_t$ only if ALL conditions pass.

### Parlay Accepted Set

$$J_t = \{(i, j, k, ...): \text{conditions}\}$$

- Each leg $\in B_t$
- Every 2-leg subset is rational
- Joint probability beats actual parlay break-even
- Shared failure risk is low
- Stress-tested joint EV remains positive

**Important:** $J_t \ne B_t \times B_t \times B_t$

Do not create every combination of good legs.

## Configuration

See `config/parlay_policy.yaml` for all tunable parameters:

- Single leg thresholds (min_lcb_edge_seed, min_forecastability, etc.)
- Price parameters (max_stale_minutes, price_dependent_watchlist)
- Parlay thresholds (joint LCB edge, shared failure risk, event-supply penalty)
- Generation parameters (top N anchors, companions, max legs)
- Stress parameters (plan holds downshift, calibration penalty)
- Output and validation parameters

## Usage

### Running the Full Pipeline

```python
from orchestrator import ParlaySubsystemOrchestrator

orchestrator = ParlaySubsystemOrchestrator("config/parlay_policy.yaml")

summary = orchestrator.run_pipeline(
    scenario_prob_path="outputs/scenario_probability_matrix_latest.csv",
    forecastability_path="outputs/forecastability_board_latest.csv",
    player_state_path="outputs/trusted_player_state_registry_latest.csv",
    odds_snapshot_path="outputs/current_odds_snapshot.csv",
    output_dir="outputs"
)
```

### Running Individual Phases

```python
# Phase 1: Build priced event universe
from build_priced_event_universe import PricedEventUniverseBuilder

builder = PricedEventUniverseBuilder(...)
events = builder.build_universe()
builder.export_to_csv("outputs/priced_event_universe_latest.csv")

# Phase 3: Evaluate single legs
from single_leg_set_membership import SingleLegSetMembership

evaluator = SingleLegSetMembership(config)
evaluations = evaluator.evaluate(events)

# Phase 4: Generate parlays
from anchor_companion_generator import AnchorCompanionGenerator

generator = AnchorCompanionGenerator(config)
parlay_specs = generator.generate_parlay_candidates(evaluations)
```

## Testing

```bash
python -m pytest tests/test_parlay_subsystem.py -v
```

Key tests:
- Odds conversions (American ↔ Decimal, implied probability)
- Break-even calculations
- Min acceptable odds
- Single-leg tier membership
- Parlay price engine (synthetic and book-quoted)
- Shared event supply detection
- All edges and EV calculations

## Edge Cases and Validations

1. **50/50 is playable** - A 50% probability leg at +110 (break-even 47.6%) has positive edge
2. **50/50 is not playable** - A 50% probability leg at -120 (break-even 54.5%) has negative edge
3. **Price-dependent watchlist** - Legs tracked until price improves to min acceptable odds
4. **News-dependent watchlist** - Legs excluded until after injury report/lineup confirmation
5. **No Cartesian product** - Anchor + companion prevents explosion of low-quality combinations
6. **2-leg subset gate** - For 3-leg parlays, all 2-leg subsets must pass filters
7. **SGP payout reduction** - Rejects same-game parlays if book reduces payout below 75% of synthetic
8. **Shared event supply** - Rejects if too many legs draw from same limited pool
9. **Shared failure modes** - Applies penalty if all legs fail under common scenario

## Future Enhancements

1. **Phase 6 expansion**: Team environment failure mode analysis with per-leg exposure
2. **Phase 7 expansion**: Empirical covariance matrix from historical outcomes
3. **Phase 12 expansion**: Full checkpoint system with automatic reruns
4. **Phase 13 expansion**: Comprehensive backtesting vs naive stacking
5. **3-leg parlays**: Once 2-leg validation is strong
6. **Same-game scenario simulation**: Complex scenario trees for SGP correlations
7. **Trusted joint-state registry**: Pre-computed joint states from historical validation
8. **Alt-line ladder optimization**: Find optimal alt line for each leg
9. **Multi-book comparison**: Track best prices across books
10. **Live parlay adjustment**: Track in-game developments vs pre-event state

## Files

- `core_utils.py` - Odds conversions, probability calculations, enums
- `data_types.py` - All data models and types
- `build_priced_event_universe.py` - Phase 1
- `line_zone_scanner.py` - Phase 2
- `single_leg_set_membership.py` - Phase 3
- `anchor_companion_generator.py` - Phase 4
- `shared_event_supply_engine.py` - Phase 5
- `parlay_price_engine.py` - Phase 8
- `parlay_probability_engine.py` - Phase 9
- `parlay_stress_engine.py` - Phase 10
- `parlay_selector.py` - Phase 11
- `orchestrator.py` - Main orchestration
- `config/parlay_policy.yaml` - Configuration
- `tests/test_parlay_subsystem.py` - Tests

## Logging

Enable debug logging to see detailed decision reasoning:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Shadow Mode

All output is tagged with `SHADOW_MODE`. The system:
- Does not execute bets
- Does not track real money
- Does not integrate with production selection/staking
- Does not affect user-facing promotion behavior
- Validates approach through backtesting before integration

## References

### Probability and Odds

- American odds: -110 (favored), +110 (underdog)
- Implied probability: probability built into odds (with vigorish)
- Break-even: probability at which bet is neutral
- Edge: probability advantage over break-even

### Edge Cases

- A leg can be playable at unfavorable price if LCB edge is high enough
- A leg can be unplayable at favorable price if LCB edge is low or uncertain
- Parlay EV depends on joint probability AND price, not just individual probabilities
- Shared failure modes can kill multiple legs despite individual high probabilities

## Contact

Built as part of NBA-Analytics modernization effort.
Questions or issues: see project documentation.
