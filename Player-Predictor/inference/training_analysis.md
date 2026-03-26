# Training Analysis - MoE Expert Collapse Issue

## Current Status

Training completed with 50 epochs, but **CRITICAL ISSUE DETECTED**: Complete expert collapse.

### Expert Usage (Final Epoch):
- **Expert 0**: 99.44% ← DOMINATES EVERYTHING
- Expert 1: 0.0011%
- Expert 2: 0.0018%
- Expert 3-10: < 0.0001% each
- **Spike experts (8-10)**: Essentially unused (< 0.000001%)

### Performance Metrics:
- Training R²: ~0.50
- Validation R²: ~0.45
- Router entropy: 0.032 (extremely low - should be ~2.4 for 11 experts)
- Max routing confidence: 99.44% (too high - indicates no exploration)

## Root Cause Analysis

Despite implementing all 4 phases of MoE improvements, the model collapsed to a single expert. This indicates:

1. **Router temperature too low** (2.0) - not enough exploration
2. **Load balance weights too weak** - started at 0.001, ramped to 0.008 (still too low)
3. **Expert keys initialized poorly** - all experts may have similar keys
4. **Training dynamics favor collapse** - once one expert starts winning, it gets all gradients

## Why This Matters

The current model is essentially:
- A single dense network (Expert 0)
- With 10 dead experts adding no value
- No specialization happening
- No spike detection working
- Wasting compute and parameters

**Baseline comparison**: The old ensemble had MAE=4.78, R²=-0.026. We need to evaluate if this new model beats that.

## Recommended Fixes (Priority Order)

### 1. IMMEDIATE: Increase Router Temperature
```python
"router_temperature": 5.0,  # Was 2.0, increase to 5.0 for more exploration
```

### 2. IMMEDIATE: Strengthen Load Balance
```python
"load_balance_weight_start": 0.01,   # Was 0.001
"load_balance_weight_mid": 0.05,     # Was 0.003  
"load_balance_weight_final": 0.1,    # Was 0.008
```

### 3. HIGH: Better Expert Key Initialization
```python
# In build_model_with_phase2(), change:
expert_key_table = Embedding(
    input_dim=total_experts,
    output_dim=key_dim,
    embeddings_initializer=tf.keras.initializers.Orthogonal(),  # Was RandomNormal
    name="expert_key_table",
)
```

### 4. HIGH: Add Entropy Loss (Force Diversity)
```python
"router_z_loss_weight": 0.01,  # Was 0.0, enable entropy regularization
```

### 5. MEDIUM: Reduce Compactness Early
```python
"compactness_start": 0.0,      # Was 0.0025, start at 0
"compactness_final": 0.005,    # Was 0.01, reduce final
```
Compactness forces samples to their assigned expert, which accelerates collapse.

### 6. MEDIUM: Increase Capacity Factor
```python
"capacity_factor": 2.0,  # Was 1.25, allow more samples per expert
```

### 7. LOW: Add Auxiliary Load Loss
Add a stronger auxiliary loss that directly penalizes usage imbalance:
```python
# In build_model_with_phase2(), add:
usage_std = tf.math.reduce_std(router_probs, axis=0)
aux_load_loss = usage_std * total_experts
model.add_loss(0.1 * aux_load_loss)
```

## Testing Strategy

1. Apply fixes 1-4 (immediate + high priority)
2. Retrain for 20 epochs
3. Check expert usage metrics:
   - Target: Each expert should have 5-15% usage
   - Router entropy should be > 1.5
   - No single expert > 30%
4. If still collapsed, apply fixes 5-7

## Alternative Approach: Hard Routing

If soft routing continues to collapse, consider switching to hard routing:
- Top-K routing (K=2): Each sample goes to exactly 2 experts
- Removes the "rich get richer" dynamic
- Forces all experts to receive gradients

## Evaluation Next Steps

Before making changes, we should:
1. Evaluate current model performance vs baseline
2. Check if despite collapse, it's still better than old ensemble
3. If it's worse, the collapse is definitely hurting performance
4. If it's better, collapse might be acceptable (but still suboptimal)

Run: `python inference/eval_h5.py` (after fixing the model loading issue)
