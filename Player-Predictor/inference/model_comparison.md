# Model Comparison: Old vs New Trainer

## Current Evaluation Results (OLD Trainer)

**Models Evaluated**: `hybrid_spike_ensemble_*_weights.h5` (saved 8:12 PM, Feb 13, 2026)
**Trainer Used**: `training/hybrid_spike_moe_trainer.py` (OLD)

### Performance
- **MAE**: 4.675
- **R²**: 0.036
- **Per-Target**:
  - PTS: MAE=8.59, R²=-0.40
  - TRB: MAE=2.86, R²=0.45
  - AST: MAE=2.57, R²=0.06

### Expert Usage (SEVERE COLLAPSE)
- **Expert 0**: 99.44% ❌
- **Experts 1-10**: <0.002% each ❌
- **Router Entropy**: 0.032 (should be ~2.4) ❌

### Configuration Issues
```json
{
  "router_temperature": 5.0,  // ✅ Good
  "load_balance_weight": 0.08,  // ❌ Single value, not ramping
  "router_z_loss_weight": 0.001,  // ✅ Present
  "capacity_factor": 2.0,  // ✅ Good
  // ❌ MISSING: load_balance_weight_start, _mid, _final
  // ❌ MISSING: Orthogonal initialization
  // ❌ MISSING: Auxiliary variance loss
  // ❌ MISSING: Compactness delay schedule
}
```

## Expected Results (NEW Trainer)

**Trainer**: `training/integrate_moe_improvements.py` (NEW)

### Key Improvements

1. **Load Balance Ramping Schedule**:
   ```python
   Epoch 0-3:   0.01 (start)
   Epoch 3-15:  0.01 → 0.05 (ramp up)
   Epoch 15+:   0.1 (final, 10x stronger than old 0.01)
   ```

2. **Expert Key Initialization**:
   ```python
   OLD: RandomNormal(stddev=0.02)  # Similar keys → collapse
   NEW: Orthogonal(gain=0.5)       # Maximally different keys
   ```

3. **Router Z-Loss**:
   ```python
   L_z = mean(square(log_sum_exp(router_logits)))
   Prevents logit explosion that causes collapse
   ```

4. **Auxiliary Variance Loss**:
   ```python
   Directly penalizes low std deviation in expert usage
   Forces distribution to spread out
   ```

5. **Compactness Delay**:
   ```python
   OLD: Starts at epoch 0 (locks in collapse early)
   NEW: Starts at epoch 10 (allows exploration first)
   ```

### Expected Metrics

**Expert Usage** (should be balanced):
```
expert_0_usage:  8-12%
expert_1_usage:  8-12%
expert_2_usage:  8-12%
expert_3_usage:  8-12%
expert_4_usage:  8-12%
expert_5_usage:  8-12%
expert_6_usage:  8-12%
expert_7_usage:  8-12%
expert_8_usage:  5-10% (spike expert)
expert_9_usage:  5-10% (spike expert)
expert_10_usage: 5-10% (spike expert)
```

**Router Entropy**: >1.5 (ideally 1.8-2.2)
- 11 experts → max entropy = log(11) = 2.4
- Target: 85% of max = 2.0

**Performance** (expected improvement):
- **MAE**: <4.5 (vs 4.675 old)
- **R²**: >0.1 (vs 0.036 old)
- Better calibration (uncertainty matches errors)

## How to Run New Training

### Step 1: Run Training
```cmd
python training/integrate_moe_improvements.py
```

### Step 2: Monitor First 10 Epochs
Watch for:
- Expert usage distribution (should be 5-15% each)
- Router entropy (should be >1.5)
- No single expert >30%

### Step 3: Evaluate
```cmd
python inference/ensemble_inference.py
```

### Step 4: Compare
Check `inference/evaluation_results.json`:
- MAE should decrease
- R² should increase
- Expert usage should be balanced

## Troubleshooting

### If Collapse Still Happens

Edit `training/integrate_moe_improvements.py` and increase:

```python
"router_temperature": 7.0,  # Currently 5.0
"load_balance_weight_final": 0.2,  # Currently 0.1
"load_balance_weight_mid": 0.1,  # Currently 0.05
```

### If Training is Too Slow

Reduce epochs for testing:
```python
"epochs": 20,  # Instead of 50
```

### If Memory Issues

Reduce batch size:
```python
"batch_size": 16,  # Instead of 32
```

## References

- **Research**: `inference/expert_collapse_fixes.md`
- **Analysis**: `inference/training_analysis.md`
- **Instructions**: `TRAINING_INSTRUCTIONS.md`
- **Old Trainer**: `training/hybrid_spike_moe_trainer.py`
- **New Trainer**: `training/integrate_moe_improvements.py`
