# Training Issue Analysis

## Problem Identified

The meta-state MoE training completed but produced very poor results:
- **R²_macro**: -4.795 (should be positive, ideally >0.10)
- **MAE**: 12.403 (very high)
- **Predictions**: ~2x baseline values (e.g., baseline=25, pred=53)

## Root Cause

The meta-state model outputs **absolute predictions** (μ = baseline + delta), but the original trainer's phased training logic expects **delta outputs** when `predict_delta=True`.

### Architecture Mismatch

**Meta-State Model Output**:
```python
# In meta_state_moe_trainer_core.py
mu = Add()([baseline_input, delta_final])  # Absolute prediction
output = [mu, sigma_ale, u_epi, events, gate]  # 14-dim
```

**Original Trainer Expectation**:
```python
# In improved_baseline_trainer.py
if self.config.get("predict_delta", False):
    delta_means = preds[:, :3]  # Expects delta, not absolute
    pred_means = baselines + delta_means  # Reconstructs absolute
```

### Why This Causes Issues

1. **Loss Function**: Expects delta but receives absolute → wrong gradients
2. **Evaluation**: Adds baseline twice → predictions are 2x too high
3. **Phased Training**: All variance/calibration logic breaks

## Solutions

### Option 1: Fix Meta-State Model (Recommended)

Change meta-state model to output deltas:

```python
# In meta_state_moe_trainer_core.py, line ~180
# OLD:
mu = Add()([baseline_input, delta_final])

# NEW:
# Just output delta, let trainer reconstruct
delta_output = delta_final  # Don't add baseline here
```

### Option 2: Fix Trainer Compatibility

Update trainer to handle both delta and absolute outputs:

```python
# In meta_state_trainer_simple.py
def _make_mae_metric(self):
    def mae_metric(y_true, y_pred, baselines=None):
        n = len(self.target_columns)
        
        if y_pred.shape[-1] >= 14:
            # Meta-state outputs absolute mu (already reconstructed)
            mu = y_pred[:, :n]
        else:
            # Original outputs delta
            delta = y_pred[:, :n]
            mu = baselines + delta if baselines is not None else delta
        
        return tf.reduce_mean(tf.abs(y_true - mu))
    return mae_metric
```

### Option 3: Run Original Trainer First (Immediate)

Establish baseline performance with original architecture:

```bash
python training/improved_baseline_trainer.py
```

Then compare with meta-state after fixing the architecture.

## Recommendation

**Immediate**: Run original trainer to get baseline metrics
**Next**: Fix meta-state model to output deltas (Option 1)
**Then**: Re-train and compare

## Expected Results After Fix

| Metric | Original | Meta-State (Fixed) | Target |
|--------|----------|-------------------|--------|
| R²_macro | 0.10-0.15 | 0.15-0.25 | >0.15 |
| MAE | 4.5-5.5 | 4.0-5.0 | <5.0 |
| Predictions | Reasonable | Reasonable | Match baselines ±10 |

## Files to Modify

1. `training/meta_state_moe_trainer_core.py` - Line ~180 (remove Add baseline)
2. `training/meta_state_losses.py` - Update to expect delta outputs
3. `training/meta_state_trainer_simple.py` - Update MAE metric

## Next Steps

1. ✅ Run original trainer: `python training/improved_baseline_trainer.py`
2. ⏳ Fix meta-state architecture (output deltas)
3. ⏳ Re-train meta-state model
4. ⏳ Compare results
