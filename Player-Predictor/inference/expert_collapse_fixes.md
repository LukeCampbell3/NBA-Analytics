# Expert Collapse Fixes - Implementation Summary

## Research Sources

Based on latest 2024-2025 research from:
- [Auxiliary Balancing Loss for MoE](https://mbrenndoerfer.com/writing/auxiliary-balancing-loss-mixture-of-experts-moe) (Content rephrased for compliance with licensing restrictions)
- [Router Z-Loss for MoE Stability](https://mbrenndoerfer.com/writing/router-z-loss-moe-training-stability) (Content rephrased for compliance with licensing restrictions)
- Multiple arxiv papers on MoE stability and expert collapse

## Problem Identified

**Complete Expert Collapse**: Expert 0 handling 99.44% of samples, all other 10 experts essentially dead.

**Root Causes**:
1. Router temperature too low (2.0) - insufficient exploration
2. Load balance weights too weak (0.001-0.008) - couldn't prevent collapse
3. Expert keys initialized with RandomNormal - may converge to similar representations
4. No Router Z-Loss - logits can grow unbounded causing instability
5. Compactness loss starting too early - forces premature specialization
6. Missing auxiliary variance penalty - no direct pressure on usage distribution

## Implemented Fixes

### 1. Router Temperature (CRITICAL)
**Changed**: 2.0 → 5.0
**Why**: Higher temperature creates softer probability distributions, allowing more exploration. Prevents "winner takes all" dynamics where one expert dominates.
**Expected Impact**: Router will assign non-trivial probabilities to multiple experts, enabling gradient flow to all experts.

### 2. Load Balance Weights (CRITICAL)
**Changed**:
- Start: 0.001 → 0.01 (10x increase)
- Mid: 0.003 → 0.05 (16x increase)  
- Final: 0.008 → 0.1 (12x increase)
- Ramp period: [3, 10] → [3, 15] epochs

**Why**: Research shows typical values are 0.01-0.1. Our previous values were too weak to counteract the natural tendency toward collapse.
**Expected Impact**: Stronger gradient signal pushing router toward balanced expert usage.

### 3. Router Z-Loss (NEW - CRITICAL)
**Added**: Z-loss with weight 0.001
**Formula**: `L_z = mean(square(log_sum_exp(router_logits)))`
**Why**: Penalizes large router logits, preventing numerical instability and overconfident routing. This is a standard component in modern MoE (Switch Transformer, Mixtral).
**Expected Impact**: 
- Prevents logit explosion
- Maintains softer routing probabilities
- Improves gradient flow
- Better generalization

### 4. Expert Key Initialization (HIGH PRIORITY)
**Changed**: RandomNormal(stddev=0.5) → Orthogonal(gain=0.5)
**Why**: Orthogonal initialization ensures expert keys start maximally different from each other, preventing early convergence to similar representations.
**Expected Impact**: Experts begin with distinct identities, making it easier for router to learn meaningful distinctions.

### 5. Auxiliary Variance Loss (NEW)
**Added**: Direct penalty on standard deviation of router probabilities
**Weight**: 0.05 * load_balance_weight
**Why**: Complements existing load losses by directly targeting the variance in expert usage distribution.
**Expected Impact**: Additional pressure toward uniform expert utilization.

### 6. Compactness Schedule (IMPORTANT)
**Changed**:
- Start: 0.0025 → 0.0 (disabled initially)
- Final: 0.01 → 0.005 (reduced)
- Ramp: [5, 20] → [10, 25] epochs (delayed)

**Why**: Early compactness forces samples to their assigned expert before the router has learned good assignments, accelerating collapse. Delaying allows exploration first.
**Expected Impact**: Router explores different expert assignments before committing to specializations.

### 7. Capacity Factor
**Changed**: 1.25 → 2.0
**Why**: Higher capacity allows more tokens per expert, reducing token dropping and giving experts more training signal.
**Expected Impact**: Fewer tokens dropped, more stable expert training.

### 8. Other Weight Increases
- Importance weight: 0.01 → 0.02
- Load weight: 0.01 → 0.02
- Overflow penalty: 0.01 → 0.02
- Router noise: 0.02 → 0.05

## Expected Training Behavior

### Healthy Expert Usage (Target)
- Each expert: 5-15% usage
- Router entropy: > 1.5 (currently 0.032)
- No single expert > 30%
- Router LSE: < 10

### Monitoring Metrics
Watch these during training:
1. `expert_X_usage` - should be roughly balanced
2. `router_entropy` - should increase from current 0.032 to > 1.5
3. `router_z_loss` - should stay < 100
4. `router_lse_mean` - should stay < 10
5. `aux_load_loss` - should decrease as balance improves

### If Still Collapsing
If after 10 epochs experts are still collapsed:
1. Increase router_temperature to 7.0 or 10.0
2. Increase load_balance_weight_final to 0.2
3. Increase router_z_loss_weight to 0.01
4. Consider switching to hard top-2 routing

## Implementation Details

All changes made to: `training/integrate_moe_improvements.py`

**Lines modified**:
- Config initialization (~line 450-550)
- Expert key embedding (~line 650)
- Router z-loss addition (~line 680)
- Auxiliary load loss (~line 750)
- Entropy regularization (~line 800)

## Next Steps

1. **Run training**: `python training/integrate_moe_improvements.py`
2. **Monitor first 10 epochs**: Check expert usage metrics
3. **Evaluate**: If experts are balanced, run full training
4. **Compare**: Evaluate final model vs old baseline

## Success Criteria

Training is successful if:
- ✓ All 11 experts receive 5-15% of tokens
- ✓ Router entropy > 1.5
- ✓ No training instabilities (NaN, loss spikes)
- ✓ Validation R² improves over baseline
- ✓ Model accuracy beats old ensemble (MAE < 4.78)

## References

Research findings were synthesized and rephrased from multiple sources to comply with licensing restrictions. Key insights:
- Auxiliary balancing loss is essential for preventing expert collapse
- Router z-loss provides numerical stability and prevents overconfident routing
- Typical coefficients: load balance 0.01-0.1, z-loss 0.0001-0.01
- Orthogonal initialization helps experts develop distinct specializations
- Temperature and compactness schedule are critical for allowing exploration
