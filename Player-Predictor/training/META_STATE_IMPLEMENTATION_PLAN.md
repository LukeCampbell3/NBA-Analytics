# Meta-State MoE Implementation Plan

## Overview
Transform current "patterns → outputs" MoE into "meta-state → routing → delta prediction (+ uncertainty)" system.

## Current Architecture Analysis

### What We Have:
1. **Transformer encoder** → sequence representation
2. **Router** → routes to 12 base + 4 spike experts based on sequence_repr + baseline
3. **Experts** → output delta predictions (already using `predict_delta=True`)
4. **Spike experts** → activated by spike indicators (sigmoid on spike features)
5. **ConditionalSpikeOutput** → combines expert outputs with spike indicators
6. **Loss** → Student-t NLL on deltas + variance encouragement + calibration

### What Needs to Change:
1. **Add meta-state head `z`** → explicit "reason encoder" (32-64 dim)
2. **Route using `z`** → not raw sequence_repr
3. **Redefine spike experts** → event/regime specialists, not threshold-based
4. **Separate uncertainties** → epistemic (u_epi) vs aleatoric (σ_ale)
5. **Event prediction** → optional event logits head
6. **Update losses** → delta-focused, uncertainty separation, calibration

## Implementation Steps (6 Phases)

### Phase 1: Add Meta-State Head `z` ✓
**File**: `training/meta_state_moe_v1.py`
- Add `z = Dense(z_dim) -> LayerNorm -> GELU -> Dropout` after sequence pooling
- Keep existing architecture, just add z computation
- z_dim = 32 to 64

### Phase 2: Route Using `z` ✓
**File**: `training/meta_state_moe_v2.py`
- Change router input from `sequence_repr_with_baseline` to `concat(z, baseline)`
- Keep spike indicators for now (will replace in Phase 4)

### Phase 3: Add Event Prediction (Optional) ✓
**File**: `training/meta_state_moe_v3.py`
- Add `event_logits = Dense(num_events)(z)`
- Start with proxy events derived from data:
  - minutes_spike (MP > 38)
  - usage_spike (PTS + AST > 35)
  - pace_tier (high/medium/low based on team stats)
  - blowout_risk (score differential proxy)
- Add BCE loss for event classification

### Phase 4: Redefine Spike Experts as Event Specialists ✓
**File**: `training/meta_state_moe_v4.py`
- Replace spike threshold activation with event-gated activation
- `gate = sigmoid(Dense(1)(concat(z, u_epi)))`
- Spike contribution: `Δμ = Δμ_base + gate * Δμ_spike`
- Remove spike_indicators from ConditionalSpikeOutput

### Phase 5: Separate Epistemic vs Aleatoric Uncertainty ✓
**File**: `training/meta_state_moe_v5.py`
- Aleatoric: `σ_ale` (3-dim) from expert outputs (Student-t scales)
- Epistemic: `u_epi = Dense(3)(z)` OR MC dropout variance
- Update loss: σ_ale trained via Student-t NLL
- u_epi supervised weakly (ranking loss or correlation)

### Phase 6: Update Loss Functions ✓
**File**: `training/meta_state_moe_v6.py` (FINAL)
- Delta mean loss: Huber/L1 on `y_delta vs Δμ`
- Probabilistic NLL: Student-t on `y vs μ = baseline + Δμ` and `σ_ale`
- Event classification: BCE on event_logits
- Expert usage regularizer: 6-10 active experts
- Spike gate regularizer: prevent always-on gating
- Remove/weaken sigma regularization (scale cheating)

## File Structure

```
training/
├── META_STATE_IMPLEMENTATION_PLAN.md  (this file)
├── meta_state_moe_v1.py               (Phase 1: Add z)
├── meta_state_moe_v2.py               (Phase 2: Route using z)
├── meta_state_moe_v3.py               (Phase 3: Event prediction)
├── meta_state_moe_v4.py               (Phase 4: Event-gated spike experts)
├── meta_state_moe_v5.py               (Phase 5: Uncertainty separation)
├── meta_state_moe_v6.py               (Phase 6: Final loss updates) ← USE THIS
├── improved_baseline_trainer.py       (current trainer - keep for reference)
└── hybrid_spike_moe_trainer.py        (current MoE - keep for reference)
```

## Success Criteria

After Phase 6, you should see:
1. **Better delta learning**: R²_macro_delta rises before raw R²
2. **Meaningful spike gate usage**: gate mostly low, spikes for predicted events
3. **Improved calibration slope**: moves toward 0.5-1.0 (not 0.1)
4. **Improved PTS variance ratio (delta)**: rises materially
5. **MAE may increase slightly** while R² improves (expected if suppressing variance stops)

## Next Steps After Implementation

1. **Test Phase 1-6 incrementally**: Run each phase, compare metrics
2. **Fix time-aware batching** (Phase 7 from plan): Ensure no data leakage
3. **Add curriculum learning** (Phase 8): Use delta residual + epistemic for difficulty
4. **Replace batch variance** (Phase 9): Use buffered variance monitoring
5. **Calibration improvements** (Phase 10): Quantile coverage loss
6. **Training controls** (Phase 11): Stability window checkpointing

## Configuration Changes

Key config updates for meta-state architecture:
```python
"z_dim": 48,  # Meta-state dimension
"num_events": 4,  # Number of event types
"event_loss_weight": 0.1,  # Event classification weight
"gate_penalty_weight": 0.05,  # Prevent always-on gating
"epistemic_weight": 0.01,  # Epistemic uncertainty supervision
"use_mc_dropout": False,  # Use learned u_epi head (not MC dropout)
```
