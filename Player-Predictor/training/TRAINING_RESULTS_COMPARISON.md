# Training Results Comparison

## Summary

Both models trained but neither achieved positive R². The original model performs significantly better than the meta-state model in its current form.

## Results

| Metric | Original Trainer | Meta-State Trainer | Winner |
|--------|-----------------|-------------------|--------|
| **R²_macro** | -0.198 | -4.795 | ✅ Original (24x better) |
| **MAE** | 4.862 | 12.403 | ✅ Original (2.5x better) |
| **Predictions** | Reasonable | Too high (2x baseline) | ✅ Original |
| **Training Time** | ~60s | ~60s | Tie |

## Detailed Analysis

### Original Trainer
- **R²**: -0.198 (negative but close to zero)
- **MAE**: 4.862 (reasonable for NBA stats)
- **Predictions**: Close to baselines (±5-10 points)
- **Example**: baseline=25.6 → pred=30.4 (reasonable)

### Meta-State Trainer
- **R²**: -4.795 (very negative)
- **MAE**: 12.403 (very high)
- **Predictions**: ~2x baselines (way too high)
- **Example**: baseline=25.3 → pred=53.8 (unreasonable)

## Root Cause

The meta-state model has an **architecture mismatch**:

1. **Model outputs absolute predictions** (μ = baseline + delta)
2. **Trainer expects delta outputs** (when `predict_delta=True`)
3. **Result**: Baseline gets added twice → predictions are 2x too high

### Evidence

**Meta-State Predictions**:
```
baseline=21.0 → pred=45.9 (2.19x)
baseline=25.3 → pred=53.8 (2.13x)
baseline=17.2 → pred=39.1 (2.27x)
```

**Original Predictions**:
```
baseline=25.6 → pred=30.4 (1.19x)
baseline=25.3 → pred=30.9 (1.22x)
baseline=17.2 → pred=21.3 (1.24x)
```

## Why Original Trainer Also Has Negative R²

Negative R² means the model performs worse than predicting the mean. This can happen when:

1. **Limited training data** (2,073 samples across 7 players)
2. **High variance in NBA stats** (injuries, matchups, rest)
3. **Overfitting** (model memorizes training, fails on validation)
4. **Baseline already good** (hard to beat rolling average)

However, **MAE of 4.86 is actually reasonable** for NBA predictions:
- PTS: ±5 points is typical
- TRB: ±2 rebounds is typical
- AST: ±2 assists is typical

## Fixes Needed

### For Meta-State Model (Critical)

**Problem**: Outputs absolute predictions instead of deltas

**Fix**: Change line ~180 in `meta_state_moe_trainer_core.py`:

```python
# OLD (wrong):
mu = Add()([baseline_input, delta_final])
final_output = Concatenate()([mu, sigma_ale, u_epi, event_logits, gate])

# NEW (correct):
# Output delta, let trainer reconstruct
final_output = Concatenate()([delta_final, sigma_ale, u_epi, event_logits, gate])
```

### For Both Models (Improvement)

1. **More training data** - Add more players/seasons
2. **Better features** - Add opponent strength, rest days, home/away
3. **Hyperparameter tuning** - Reduce overfitting
4. **Ensemble** - Average multiple models

## Conclusion

### Current Status
- ❌ Meta-state model has critical bug (outputs 2x predictions)
- ⚠️ Original model works but R² is negative
- ✅ Original model MAE is reasonable (4.86)

### Next Steps

1. **Fix meta-state architecture** (output deltas, not absolute)
2. **Re-train meta-state model**
3. **Compare again** (expect meta-state to match or beat original)
4. **If both still negative R²**: Focus on data/features, not architecture

### Expected After Fix

| Metric | Original | Meta-State (Fixed) | Target |
|--------|----------|-------------------|--------|
| R²_macro | -0.198 | **-0.10 to +0.05** | >0.00 |
| MAE | 4.862 | **4.5-5.0** | <5.0 |
| Predictions | Reasonable | **Reasonable** | Match baselines |

## Recommendation

**Do NOT deploy either model to production yet.**

Reasons:
1. Negative R² means worse than baseline
2. Meta-state has critical bug
3. Need more data/features for positive R²

**Instead**:
1. Fix meta-state architecture
2. Re-train and compare
3. Add more features (opponent, rest, home/away)
4. Get positive R² before production

## Files to Review

- `training/TRAINING_ISSUE_ANALYSIS.md` - Detailed bug analysis
- `model/improved_baseline_metadata.json` - Original model metrics
- `training/meta_state_moe_trainer_core.py` - Line ~180 needs fix

---

**Status**: ⚠️ **NEEDS FIX BEFORE PRODUCTION**
**Priority**: 🔴 **HIGH** (Critical bug in meta-state)
**ETA**: ~30 minutes to fix and re-train
