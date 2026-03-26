# Unified MoE Trainer - Expert Collapse Prevention

## What Was Done

I've created a **single unified training file** that combines:
1. `hybrid_spike_moe_trainer.py` (spike features, data preparation)
2. `integrate_moe_improvements.py` (expert collapse fixes)

All in one place: `training/unified_moe_trainer.py`

## Key Anti-Collapse Features

### 1. Router Temperature: 5.0 (CRITICAL)
- **Before**: 2.0 (too low, winner-takes-all)
- **Now**: 5.0 (high exploration, prevents single expert dominance)
- **Effect**: Softer probability distributions, all experts get gradient signal

### 2. Strong Load Balancing (10x Stronger)
- **Before**: 0.001 → 0.008
- **Now**: 0.01 → 0.1 (ramped over 15 epochs)
- **Effect**: Strong pressure toward balanced expert usage

### 3. Router Z-Loss: 0.001 (NEW)
- **What**: Penalizes large router logits
- **Why**: Prevents numerical instability and overconfident routing
- **Effect**: Maintains softer routing, better gradient flow

### 4. Orthogonal Expert Key Initialization (NEW)
- **Before**: RandomNormal (experts start similar)
- **Now**: Orthogonal(gain=0.5) (experts start maximally different)
- **Effect**: Easier for router to learn meaningful distinctions

### 5. Delayed Compactness Schedule
- **Before**: Starts epoch 5
- **Now**: Starts epoch 10, reduced strength (0.005 final)
- **Effect**: Allows exploration before forcing specialization

### 6. Auxiliary Variance Penalty (NEW)
- **What**: Direct penalty on std of router probabilities
- **Effect**: Additional pressure toward uniform expert utilization

### 7. Diversity Regularization (Phase 3)
- **What**: Penalizes similar expert outputs
- **Effect**: Encourages experts to specialize differently

### 8. Expert Replay Buffer (Phase 4)
- **What**: Stores samples per expert for stability
- **Effect**: Prevents catastrophic forgetting

## Expected Results

### Healthy Expert Usage
- Each expert: **5-15% usage**
- Router entropy: **> 1.5** (currently 0.032 in collapsed model)
- No single expert > 30%

### Monitoring During Training
Watch these metrics:
```
expert_0_usage: should be ~9-15% (not 99%)
expert_1_usage: should be ~9-15% (not 0%)
...
expert_10_usage: should be ~9-15% (not 0%)

router_entropy: should increase from 0.032 to > 1.5
router_z_loss: should stay < 100
```

## How to Use

### 1. Train the Model
```bash
python training/unified_moe_trainer.py
```

This will:
- Load data from all players
- Create spike features
- Build MoE model with anti-collapse mechanisms
- Train for 50 epochs
- Save to `model/unified_moe_best.weights.h5`

### 2. Monitor Training
During training, you'll see:
```
📊 Expert Usage (Epoch 10):
   Expert 0 (regular): 12.34%
   Expert 1 (regular): 10.56%
   Expert 2 (regular): 9.87%
   ...
   Expert 8 (spike): 11.23%
   Expert 9 (spike): 8.91%
   Expert 10 (spike): 10.45%
   Router entropy: 2.145
   Router Z-loss: 45.32
```

### 3. Evaluate the Model
```bash
python inference/evaluate_unified_moe.py
```

This will:
- Load the trained model
- Evaluate on test set
- Compare to old ensemble (MAE=4.795)
- Save results to `inference/unified_moe_results.json`

## What to Expect

### If Training Succeeds
- All 11 experts at 5-15% usage
- Router entropy > 1.5
- No collapse warnings
- Improved accuracy over baseline

### If Experts Still Collapse
If after 10 epochs you still see collapse, increase:
1. `router_temperature` to 7.0 or 10.0
2. `load_balance_weight_final` to 0.2
3. `router_z_loss_weight` to 0.01

Edit these in `unified_moe_trainer.py` around line 250.

## Files Created

1. **training/unified_moe_trainer.py** - Main training file (all-in-one)
2. **inference/evaluate_unified_moe.py** - Evaluation script
3. **UNIFIED_MOE_GUIDE.md** - This guide

## Old Files (Can Be Archived)

These are now replaced by the unified trainer:
- `training/hybrid_spike_moe_trainer.py` (data prep now in unified)
- `training/integrate_moe_improvements.py` (fixes now in unified)

## Configuration

All anti-collapse settings are in `UnifiedMoETrainer.__init__()`:

```python
self.config.update({
    "router_temperature": 5.0,           # High exploration
    "load_balance_weight_final": 0.1,    # Strong balancing
    "router_z_loss_weight": 0.001,       # Logit stability
    "compactness_ramp_epochs": [10, 25], # Delayed specialization
    "capacity_factor": 2.0,              # More tokens per expert
    # ... and more
})
```

## Success Criteria

Training is successful if:
- ✓ All 11 experts receive 5-15% of tokens
- ✓ Router entropy > 1.5
- ✓ No training instabilities (NaN, loss spikes)
- ✓ Model accuracy beats old ensemble (MAE < 4.795)

## Troubleshooting

### Problem: Experts still collapsing
**Solution**: Increase router_temperature to 7.0-10.0

### Problem: Training unstable (NaN loss)
**Solution**: Decrease router_z_loss_weight to 0.0001

### Problem: Low accuracy despite balanced experts
**Solution**: Increase compactness_final to 0.01 after epoch 15

## Next Steps

1. Run training: `python training/unified_moe_trainer.py`
2. Monitor expert usage during training
3. Evaluate: `python inference/evaluate_unified_moe.py`
4. Compare results to old ensemble (MAE=4.795)

If experts are balanced and accuracy improves, the anti-collapse mechanisms worked!
