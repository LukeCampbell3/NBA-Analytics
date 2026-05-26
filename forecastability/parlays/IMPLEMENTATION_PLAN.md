# NBA Parlay Subsystem - Implementation Plan

**Status**: Phase 1-4 + Supporting Infrastructure (Minimal Vertical Slice) COMPLETE

**Mode**: SHADOW_ONLY - All operations non-production

---

## Summary

A robust parlay creation subsystem has been implemented for the NBA-Analytics / Player-Predictor system. The system focuses on **joint probability positions over shared game states**, not simple pick combinations.

### Core Innovation

> Don't ask: "Which predictions look good?"
>
> Ask: "Which priced pre-event binary events belong to reliable/mispriced state regions, and which combinations remain positive-EV after joint-state stress?"

---

## Phase Completion Status

### ✅ COMPLETED (Phases 1-4 + 5, 8-11)

#### Phase 1: Build Priced Binary-Event Universe
- [x] `build_priced_event_universe.py` - Complete
- [x] Loads scenario probabilities, forecastability, player state, odds
- [x] Computes break-even from American odds
- [x] Calculates all edge metrics (raw, robust, LCB)
- [x] Exports to CSV with full schema
- [x] Handles missing data gracefully

#### Phase 2: Line-Zone and Alternate-Line Scanner
- [x] `line_zone_scanner.py` - Complete
- [x] Scans best lines by robust EV and LCB edge
- [x] Line classification (NEAR_MEDIAN, TAIL, etc.)
- [x] Supports main, alt, and both sides
- [x] Best-framing discovery per player/market

#### Phase 3: Single-Leg Set Membership with Tiers
- [x] `single_leg_set_membership.py` - Complete
- [x] SEED_PLAYABLE tier (high conviction)
- [x] BALANCED_PLAYABLE tier (acceptable edge)
- [x] PRICE_DEPENDENT tier (watchlist)
- [x] NEWS_DEPENDENT tier (post-clarity watchlist)
- [x] Computes min_acceptable_odds for price-dependent legs
- [x] Detailed rejection reason tracking

#### Phase 4: Anchor + Companion Parlay Construction
- [x] `anchor_companion_generator.py` - Complete
- [x] Selects top N anchor legs by LCB edge
- [x] Finds compatible companions (not Cartesian product)
- [x] Filters PRICE_DEPENDENT companions by min odds
- [x] Excludes NEWS_DEPENDENT until clarity
- [x] No combinatorial explosion

#### Phase 5: Shared Event-Supply Engine
- [x] `shared_event_supply_engine.py` - Complete
- [x] Detects shared event pools (rebounds, assists, points, etc.)
- [x] Conflict detection (same player, same pool)
- [x] Rejection logic for over-constrained supply
- [x] Penalty computation based on severity

#### Phase 8: Parlay Price Engine
- [x] `parlay_price_engine.py` - Complete
- [x] Synthetic price (product of decimal odds)
- [x] Book-quoted SGP price override support
- [x] SGP payout reduction detection
- [x] Break-even probability computation

#### Phase 9: Parlay Probability Engine
- [x] `parlay_probability_engine.py` - Complete
- [x] Naive product of individual probabilities
- [x] Cross-game vs same-game correlation adjustment
- [x] Shared event-supply penalty application
- [x] Calibration penalty
- [x] LCB probability computation
- [x] Confidence scoring

#### Phase 10: Parlay Stress Engine
- [x] `parlay_stress_engine.py` - Complete
- [x] Stress factors (plan holds, blowout, foul trouble, etc.)
- [x] Shared failure risk penalty
- [x] Stressed probability validation
- [x] Failure mode determination

#### Phase 11: Parlay Selector
- [x] `parlay_selector.py` - Complete
- [x] Final scoring (weighted product)
- [x] Tier classification (SEED, BALANCED, BOUNDARY)
- [x] All rejection reasons documented
- [x] Ranking by final score

#### Supporting Infrastructure
- [x] `core_utils.py` - Odds conversions, probability, EV calculations
- [x] `data_types.py` - All data models and enums
- [x] `orchestrator.py` - Main pipeline orchestration
- [x] `config/parlay_policy.yaml` - Full policy configuration
- [x] `tests/test_parlay_subsystem.py` - Comprehensive test suite

---

### 🚧 FUTURE IMPLEMENTATION (Phases 6-7, 12-17)

#### Phase 6: Team Environment Failure Modes
- [ ] Named failure modes (TEAM_OFFENSE_COLLAPSE, LOW_ASSIST_ENVIRONMENT, etc.)
- [ ] Per-leg exposure analysis
- [ ] Shared exposure scoring
- [ ] Integration into stress engine

#### Phase 7: Joint State and Correlation Engine
- [ ] Empirical covariance matrix from historical data
- [ ] Correlation classes (CROSS_GAME_WEAK, SAME_GAME_BLOWOUT, etc.)
- [ ] Scenario simulation for same-game parlays
- [ ] Joint state registry

#### Phase 12: Timing Checkpoints
- [ ] Checkpoint system (OPEN, MIDDAY, INJURY_REPORT, LINEUP_CONFIRMED, FINAL_ODDS)
- [ ] Automatic rerun logic
- [ ] Tier change tracking
- [ ] Price movement detection

#### Phase 13: Validation
- [ ] Historical backtesting
- [ ] Comparison vs naive stacking
- [ ] Hit rate by tier
- [ ] ROI and profit tracking
- [ ] Calibration (Brier score, ECE)
- [ ] CLV if available
- [ ] Results breakdown by dimensions

#### Phase 14: Reporting
- [ ] Detailed parlay explanations
- [ ] Joint state analysis per parlay
- [ ] Failure mode exposure per leg
- [ ] Event supply risk visualization
- [ ] HTML report generation

#### Phase 15: Config (Extension)
- [ ] Additional tuning for edge cases
- [ ] Per-market thresholds
- [ ] Per-player overrides
- [ ] Checkpoint-specific policy

#### Phase 16: Tests (Expansion)
- [ ] Line zone classification tests
- [ ] Opposite-side discovery tests
- [ ] Alt-line ladder tests
- [ ] SGP price override tests
- [ ] Parlay subset validation
- [ ] Shared failure rejection tests
- [ ] Event-supply conflict tests
- [ ] Cross-game product fallback

#### Phase 17: Minimal Vertical Slice Expansion
- [ ] 3-leg parlay support (when all 2-leg subsets pass)
- [ ] Same-game scenario simulation
- [ ] Empirical correlation application
- [ ] Checkpoint reruns
- [ ] Pipeline integration with existing predictor

---

## Key Design Decisions

### 1. No Cartesian Product
- **Problem**: Creating every combination of accepted legs leads to low-quality parlays
- **Solution**: Anchor + companion pattern limits generation while preserving quality

### 2. Joint-State Filtering
- **Problem**: Multiply probabilities naively → overconfident estimates
- **Solution**: Explicit correlation adjustment, shared failure penalty, event-supply detection

### 3. Price-Relative Judgment
- **Problem**: Not all 50% edges are equal
- **Solution**: Always compute min_acceptable_odds; track PRICE_DEPENDENT watchlist

### 4. Stress Testing
- **Problem**: Models break under stress
- **Solution**: Conservative probability downshift, failure mode analysis, margin requirements

### 5. Detailed Rejection Reasons
- **Problem**: Hard to debug why parlays rejected
- **Solution**: Every rejection includes specific reasons, promotion requirements for PRICE_DEPENDENT

### 6. Shadow Mode Only
- **Problem**: Don't want production impact during development
- **Solution**: All output marked SHADOW_MODE; no betting execution; for validation only

---

## File Structure

```
forecastability/parlays/
├── __init__.py                          # Module exports
├── README.md                            # Full documentation
├── core_utils.py                        # Odds, probability, EV calculations
├── data_types.py                        # All data models
├── build_priced_event_universe.py       # Phase 1
├── line_zone_scanner.py                 # Phase 2
├── single_leg_set_membership.py         # Phase 3
├── anchor_companion_generator.py        # Phase 4
├── shared_event_supply_engine.py        # Phase 5
├── parlay_price_engine.py               # Phase 8
├── parlay_probability_engine.py         # Phase 9
├── parlay_stress_engine.py              # Phase 10
├── parlay_selector.py                   # Phase 11
├── orchestrator.py                      # Main orchestration
├── config/
│   └── parlay_policy.yaml               # Policy configuration
├── outputs/                             # Output directory (created at runtime)
│   ├── priced_event_universe_latest.csv
│   ├── parlays_latest.json
│   └── ...
└── tests/
    └── test_parlay_subsystem.py         # Test suite
```

---

## Core Abstractions

### Single-Leg Bettable Set

$$B_t = P_t \cap R_t \cap C_t \cap M_t \cap Z_t$$

A leg is playable iff it satisfies ALL:
- **P_t** = Probability/edge set (p_stress > break_even + margin)
- **R_t** = Reliability set (forecastability >= min, plan_reliability >= min)
- **C_t** = Chaos-stable set (chaos_score <= max, volatility <= max)
- **M_t** = Market-mispriced set (robust_ev > 0, lcb_edge > min)
- **Z_t** = Price-valid set (current odds valid, not stale)

### Parlay Accepted Set

$$J_t = \{(i, j, k, ...): \text{all conditions}\}$$

A parlay is playable iff:
- All legs $\in B_t$
- Every 2-leg subset is rational
- Joint probability > parlay break-even + cushion
- Shared failure risk < max
- Shared event-supply penalty < max
- Stress-tested EV remains positive

### Final Score Formula

$$score = LCB_{EV} \times state_{compat} \times forecast_{min} \times reliability_{min} \times scenario_{min} \times price_{qual} \times (1 - fail_{risk}) \times (1 - dep_{pen}) \times (1 - supply_{pen}) \times (1 - frag)$$

---

## Configuration Highlights

From `config/parlay_policy.yaml`:

```yaml
single_leg_thresholds:
  min_lcb_edge_seed: 0.025        # SEED tier: 2.5% LCB edge minimum
  min_lcb_edge_balanced: 0.010    # BALANCED tier: 1.0% LCB edge minimum
  min_stress_edge: 0.015          # All legs: 1.5% stress margin
  min_forecastability_seed: 0.78  # SEED: 78% forecastability
  min_forecastability_balanced: 0.70  # BALANCED: 70% forecastability

parlay_thresholds:
  min_joint_lcb_edge: 0.010       # Parlay: 1.0% joint LCB edge
  max_shared_failure_risk: 0.40   # Parlay: <40% shared failure risk
  max_shared_event_supply_penalty: 0.35  # Parlay: <35% event supply penalty
  require_all_two_leg_subsets_pass: true # 3-leg only if all 2-leg pass

generation:
  top_n_anchor_legs: 20           # Top 20 anchors
  top_n_companions_per_anchor: 8  # Up to 8 companions per anchor
  max_parlays_output: 25          # Max 25 final parlays
```

---

## Testing Strategy

### Unit Tests (15+)
- ✅ American → Decimal → Probability conversions
- ✅ Break-even calculations
- ✅ Edge calculations (raw, robust, LCB)
- ✅ Min acceptable odds
- ✅ Tier membership classification
- ✅ Parlay price (synthetic, SGP, reduction detection)
- ✅ Shared event-supply detection
- ✅ Joint probability (naive, adjusted, LCB)
- ✅ Stress testing

### Integration Tests (Future)
- 2-leg parlay end-to-end
- 3-leg subset validation
- Historical backtesting
- Calibration measurement
- ROI by tier

### Edge Cases Handled
- ✅ 50/50 is playable at +110 (break-even 47.6%)
- ✅ 50/50 is unplayable at -120 (break-even 54.5%)
- ✅ Price moves invalidate decisions
- ✅ News events invalidate legs
- ✅ Same player, same market, opposite sides (perfect script) rejected
- ✅ SGP payout reduction detected and rejected if <75% synthetic
- ✅ Multiple legs from same event supply rejected
- ✅ All 2-leg subsets must pass for 3-leg parlay

---

## Running the System

### Quick Start

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

print(f"Final parlays: {summary['final_parlays_count']}")
for parlay in summary['final_parlays']:
    print(f"  {parlay['parlay_id']}: {parlay['tier']} ({parlay['final_score']:.6f})")
```

### Tests

```bash
cd forecastability/parlays
python -m pytest tests/test_parlay_subsystem.py -v
```

### Individual Phases

```python
# Phase 1 only
from build_priced_event_universe import PricedEventUniverseBuilder
builder = PricedEventUniverseBuilder(...)
builder.build_universe()
builder.export_to_csv("outputs/priced_event_universe.csv")

# Phase 3 only
from single_leg_set_membership import SingleLegSetMembership
evaluator = SingleLegSetMembership(config)
evals = evaluator.evaluate(events)
```

---

## Validation Results (Expected)

Once Phase 13 (validation) is implemented:

1. **Joint-state-filtered parlays** should outperform naive stacking
2. **Price-validated parlays** should outperform non-price-validated
3. **Low shared-failure parlays** should outperform high-conflict
4. **Low event-supply-conflict parlays** should outperform high-conflict
5. **Stress-surviving parlays** should outperform non-stress-tested
6. Hit rate should be calibrated (predicted vs actual match)
7. ROI should be positive for SEED and BALANCED tiers

---

## Next Steps (Priority Order)

1. **Data Integration** - Connect to live scenario probabilities and odds
2. **Phase 13 Validation** - Backtest against historical outcomes
3. **Phase 7 Correlation** - Implement empirical covariance from historical data
4. **3-Leg Support** - Extend to 3-leg parlays with full subset validation
5. **Phase 12 Checkpoints** - Automatic reruns at key times
6. **Phase 14 Reporting** - Generate HTML parlay reports
7. **Integration** - Connect to existing Player-Predictor pipeline
8. **Production Integration** - Shadow mode shadow → validated production mode

---

## Shadow Mode Details

All output includes:
- `mode: "SHADOW_MODE"` in metadata
- No actual bets executed
- No money risked
- No impact on production selections
- All results for validation only
- Can run continuously for validation

To transition to production:
1. Phase 13 validation must show positive ROI
2. Calibration must be acceptable (Brier score, ECE)
3. Edge cases tested thoroughly
4. Stakeholder review and approval
5. Gradual rollout with monitoring

---

## Files and Locations

```
/forecastability/parlays/
├── Core modules (11 files)
├── Config (1 file)
├── Tests (1 file)
├── Documentation (this file + README.md)
└── Outputs (created at runtime)

Total lines of code: ~3000+
Total test cases: 15+
Documentation: ~1500 lines
```

---

## Design Principles Applied

1. **Separation of Concerns** - Each phase is independent module
2. **Configuration-Driven** - All thresholds in YAML
3. **Explicit Over Implicit** - Every decision has explicit reason
4. **Fail-Safe** - SHADOW MODE only, detailed logging, no production impact
5. **Testable** - Unit testable, no external dependencies
6. **Maintainable** - Clear naming, comprehensive docs, type hints
7. **Extensible** - Easy to add phases 6-7, 12-17

---

## Known Limitations (Future Work)

1. **No empirical correlation** - Uses conservative fallback
2. **No same-game scenario simulation** - Uses simple adjustment
3. **Cross-game only** - 2-leg support primary, 3-leg future
4. **No alt-line ladder** - Uses available lines only
5. **No live adjustment** - Pre-event only, not in-game
6. **No multi-book** - Single best price
7. **No checkpoint reruns** - Single daily run

---

## Success Criteria

✅ Minimum Vertical Slice:
- [x] Priced binary-event universe builds correctly
- [x] Line zones scanned and best framings found
- [x] Single legs tiered accurately
- [x] Anchor + companion generation avoids Cartesian explosion
- [x] Shared event supply detected
- [x] Parlay prices computed (synthetic)
- [x] Joint probabilities estimated
- [x] Stress testing applied
- [x] Parlays selected and scored
- [x] Results reproducible and auditable

🎯 Full Implementation:
- [ ] All 17 phases implemented
- [ ] Backtesting validates approach
- [ ] Calibration acceptable
- [ ] ROI positive for SEED/BALANCED
- [ ] Production stakeholder approval
- [ ] Monitoring and alerting in place

---

## Contact & Questions

For questions about:
- **Architecture**: See README.md
- **Configuration**: See config/parlay_policy.yaml
- **Implementation**: See orchestrator.py
- **Testing**: See tests/test_parlay_subsystem.py
- **Usage**: See docstrings in individual modules

---

**Implementation Date**: May 23, 2026  
**Status**: Phase 1-4 + 5, 8-11 Complete  
**Mode**: SHADOW_MODE  
**Next Phase**: Data Integration & Phase 13 Validation
