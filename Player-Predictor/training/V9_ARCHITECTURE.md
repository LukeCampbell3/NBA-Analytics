# NBA v9 Architecture: Hierarchical Stochastic Player-Prop Engine

## Overview

v9 implements the research-backed concepts from `nba-upgrade.txt` as a layered probabilistic system. The core shift: **the model is not a classifier ("will he go over?") but a distribution + market inefficiency detector ("is the sportsbook line wrong relative to the player's full conditional distribution tonight?").**

## New Modules

### 1. `nba_v9_regime_hmm.py` — Hidden Markov Model Regime Detection

**What it replaces:** Heuristic regime flags (USG_spike_flag, Blowout_Signal, etc.)

**What it adds:**
- Gaussian HMM fitted per player archetype (high_usage_guard, secondary_scorer, low_usage_big, playmaker, role_player)
- Probabilistic regime inference: instead of binary flags, outputs P(regime=k) for each state
- Regime-conditional stat distributions: different mean/std per regime
- **Mixture P(over):** `P(over) = Σ P(regime=k) × P(over | regime=k)`
- Regime uncertainty (entropy) as a signal to abstain

**Key insight:** A player tonight is not one projection. It's a probability mixture over possible game states.

---

### 2. `nba_v9_copula_dependency.py` — Joint Distribution Modeling

**What it replaces:** Independent stat predictions

**What it adds:**
- Gaussian copula capturing rank correlations between PTS/TRB/AST
- Joint P(all over) for same-game parlays (accounts for correlation)
- Conditional P(over | other stat hit): `P(AST > 8.5 | PTS hit over)`
- Teammate correlation model: how one player's stats affect another's
- Game script correlation detection (blowout vs competitive)
- Diversification scoring for portfolio construction

**Key insight:** A prop model that predicts PTS, REB, AST separately underrepresents reality. Stats are correlated through game context.

---

### 3. `nba_v9_adaptive_calibration.py` — Online Calibration with Drift Detection

**What it replaces:** Static isotonic regression calibration

**What it adds:**
- Sliding window calibration that updates with each resolved game
- CUSUM and Page-Hinkley drift detection (detects when calibration shifts)
- Automatic recalibration when drift exceeds threshold
- Regime-aware calibration (separate curves per regime)
- Calibration confidence intervals
- Calibration health monitoring (Brier, ECE, drift status)

**Key insight:** A calibrator fit on October data may be wrong by February. The NBA evolves within a season.

---

### 4. `nba_v9_lineup_impact.py` — Teammate Dependency & Lineup Effects

**What it replaces:** No explicit lineup modeling

**What it adds:**
- Quantified teammate-out impacts (stat delta, usage delta, minutes delta)
- Statistical significance testing for each impact
- Lineup context computation (usage boost, minutes boost, opportunity score)
- Opportunity environment classification (elevated/normal/suppressed)
- Role change detection from lineup changes
- Lineup-adjusted stat projections

**Key insight:** One player's usage rises when another player sits. This shifts the entire distribution, not just the mean.

---

### 5. `nba_v9_residual_chaos.py` — Chaos Measurement & Diagnostics

**What it replaces:** No systematic chaos quantification

**What it adds:**
- Brier score decomposition (reliability, resolution, uncertainty)
- Permutation entropy of residual signs (measures randomness)
- Residual autocorrelation (detects missing temporal structure)
- Mutual information vs market (does model add info beyond market?)
- Rolling degradation detection
- Chaos level classification (low/moderate/high/extreme)
- Exploitable signal estimation

**Key insight:** "Is the NBA chaotic, or is my model missing state?" This module answers that question quantitatively.

---

### 6. `nba_v9_feature_engineering.py` — Extended Feature Set

**What it adds (56 new features):**
- Tail pressure features: CV, upper/lower tail rates, IQR ratios
- Regime transition features: usage/minutes/efficiency trends and acceleration
- Opportunity environment features: opportunity index, per-minute rates, game competitiveness
- Entropy/uncertainty features: rolling stat entropy, line distance in std units

---

### 7. `nba_v9_prop_engine.py` — Master Integration

Orchestrates all components into a single assessment pipeline:

```
Input: player, stat, line, market_odds, recent_games, lineup_context
  │
  ├─ Step 1: Distribution (quantiles from v8 model)
  ├─ Step 2: Raw P(over) from distribution
  ├─ Step 3: Regime detection (HMM)
  ├─ Step 4: Lineup adjustment
  ├─ Step 5: Probability blending (raw + regime + lineup)
  ├─ Step 6: Adaptive calibration
  ├─ Step 7: Market comparison (edge, EV)
  ├─ Step 8: Uncertainty assessment
  └─ Step 9: Decision (bet_over / bet_under / no_bet / monitor)
  │
Output: PropAssessment with full probabilistic analysis
```

## Integration with Existing Pipeline

The v9 modules are **additive** — they don't replace v8, they extend it:

1. **Feature engineering:** `add_v9_enhanced_features()` calls `add_v8_enhanced_features()` first, then adds v9 features
2. **Distributional head:** v8's quantile regression still produces the base distribution; v9 adds regime mixture and lineup adjustment on top
3. **Calibration:** v9's adaptive calibrator can replace or wrap v8's static isotonic regression
4. **Decision engine:** v9's PropEngine can be called from the existing `score_candidates()` pipeline to enrich each candidate with regime/lineup/chaos context

## Usage

```python
from nba_v9_prop_engine import PropEngine

engine = PropEngine(edge_threshold=0.04)
engine.load_models("model/v9/")

assessment = engine.assess_prop(
    player="Jalen Brunson",
    stat="PTS",
    line=27.5,
    market_over_odds=-110,
    recent_games=recent_games_df,
    teammates_out=["Julius Randle"],
    pace_factor=1.05,
    quantiles=model_quantiles,  # from v8 distributional head
    sigma=model_sigma,          # from v8 sigma head
)

print(engine.format_assessment(assessment))
```

## Research Alignment

| Research Concept | v9 Module |
|---|---|
| EPV / stochastic process models | `nba_v9_regime_hmm.py` |
| HMM / network models for hidden states | `nba_v9_regime_hmm.py` |
| Bayesian hierarchical archetypes | `ArchetypeRegimeLibrary` |
| Copula / Bayesian network dependencies | `nba_v9_copula_dependency.py` |
| Calibration > accuracy for betting | `nba_v9_adaptive_calibration.py` |
| Teammate/lineup causal effects | `nba_v9_lineup_impact.py` |
| Residual chaos quantification | `nba_v9_residual_chaos.py` |
| Distribution + market inefficiency detection | `nba_v9_prop_engine.py` |
