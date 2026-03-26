# Architecture Comparison: Original vs Meta-State MoE

## Side-by-Side Comparison

### Model Architecture

| Component | Original MoE | Meta-State MoE |
|-----------|-------------|----------------|
| **Sequence Encoder** | Transformer (4 layers) | ✅ Same |
| **Sequence Pooling** | GlobalAveragePooling1D | ✅ Same |
| **Meta-State Head** | ❌ None | ✅ z = Dense(48) → LayerNorm → GELU |
| **Event Prediction** | ❌ None | ✅ event_logits = Dense(4, sigmoid)(z) |
| **Epistemic Uncertainty** | ❌ None | ✅ u_epi = Dense(3, softplus)(z) |
| **Router Input** | sequence_repr + baseline | ✅ z + baseline |
| **Spike Activation** | Threshold-based (spike_indicators) | ✅ Event-gated (learned gate) |
| **Expert Outputs** | delta + sigma (6-dim) | ✅ Same |
| **Final Output** | [mu, sigma] (6-dim) | ✅ [mu, sigma_ale, u_epi, events, gate] (14-dim) |

### Loss Functions

| Loss Component | Original | Meta-State | Change |
|----------------|----------|------------|--------|
| **Delta Huber** | 0.35 weight | 0.5 weight | ⬆️ Increased |
| **Student-t NLL** | 1.0 weight | 1.0 weight | ✅ Same |
| **Mean Loss** | 0.08 weight | 0.05 weight | ⬇️ Reduced |
| **Sigma Regularization** | 0.001 weight | 0.0001 weight | ⬇️ Reduced 10x |
| **Variance Encouragement** | 3.5 weight | 3.5 weight | ✅ Same |
| **Cov Penalty** | 0.25 weight | 0.25 weight | ✅ Same |
| **Event Classification** | ❌ None | 0.1 weight | ✅ NEW |
| **Epistemic Supervision** | ❌ None | 0.01 weight | ✅ NEW |
| **Calibration (Quantile)** | ❌ None | 0.05 weight | ✅ NEW |
| **Gate Penalty** | ❌ None | 0.05 weight | ✅ NEW |
| **Expert Usage** | ❌ None | 0.1 weight | ✅ NEW |

### Training Configuration

| Setting | Original | Meta-State | Reason |
|---------|----------|------------|--------|
| **predict_delta** | True | True | ✅ Already doing delta prediction |
| **use_phased_training** | True (A→B→C) | True (A→B→C) | ✅ Keep phased approach |
| **use_curriculum** | True | True | ✅ Keep curriculum learning |
| **batch_size** | 32 | 32 | ✅ Same |
| **seq_len** | 10 | 10 | ✅ Same |
| **num_experts** | 12 | 12 | ✅ Same |
| **num_spike_experts** | 4 | 4 | ✅ Same |

### Output Structure

**Original MoE Output** (6-dim):
```python
[
    mu_PTS, mu_TRB, mu_AST,           # Reconstructed means (3)
    sigma_PTS, sigma_TRB, sigma_AST   # Aleatoric uncertainty (3)
]
```

**Meta-State MoE Output** (14-dim):
```python
[
    mu_PTS, mu_TRB, mu_AST,                    # Reconstructed means (3)
    sigma_ale_PTS, sigma_ale_TRB, sigma_ale_AST,  # Aleatoric uncertainty (3)
    u_epi_PTS, u_epi_TRB, u_epi_AST,           # Epistemic uncertainty (3)
    event_1, event_2, event_3, event_4,        # Event probabilities (4)
    gate                                        # Spike gate activation (1)
]
```

### Routing Mechanism

**Original**:
```python
sequence_repr = GlobalAveragePooling1D()(transformer_output)
router_input = Concatenate()([sequence_repr, baseline])
router_logits = Dense(16)(router_input)  # 12 base + 4 spike
router_probs = Softmax()(router_logits)

# Spike activation via threshold
spike_indicators = sigmoid(Dense(3)(spike_features))  # Based on features
```

**Meta-State**:
```python
sequence_repr = GlobalAveragePooling1D()(transformer_output)

# NEW: Meta-state head
z = Dense(48)(sequence_repr)
z = LayerNorm()(z)
z = GELU()(z)

# NEW: Event prediction
event_logits = Dense(4, sigmoid)(z)

# NEW: Epistemic uncertainty
u_epi = Dense(3, softplus)(z)

# Router uses z
router_input = Concatenate()([z, baseline])
router_logits = Dense(16)(router_input)
router_probs = Softmax()(router_logits)

# NEW: Event-gated spike activation
gate_input = Concatenate()([z, u_epi])
gate = Dense(1, sigmoid)(gate_input)  # Learned, not threshold-based
```

### Expert Combination

**Original**:
```python
# Weighted combination of all experts
expert_stack = tf.stack(expert_outputs, axis=1)
combined = tf.reduce_sum(expert_stack * router_probs_expanded, axis=1)

# Spike modulation via ConditionalSpikeOutput
final = ConditionalSpikeOutput()([combined, baseline, spike_indicators])
```

**Meta-State**:
```python
# Separate base and spike experts
base_experts = experts[:12]
spike_experts = experts[12:]

base_output = weighted_sum(base_experts, router_probs[:12])
spike_output = weighted_sum(spike_experts, router_probs[12:])

# Event-gated combination
delta_base = base_output[:, :3]
delta_spike = spike_output[:, :3]
delta_final = delta_base + gate * delta_spike

# Reconstruct
mu = baseline + delta_final
```

## Key Conceptual Differences

### 1. Routing Philosophy

**Original**: "What pattern does this look like?"
- Router sees raw sequence representation
- Pattern matching approach
- Implicit reasoning

**Meta-State**: "Why might this happen?"
- Router sees meta-state z (reason encoder)
- Causal reasoning approach
- Explicit reasoning via z

### 2. Spike Expert Activation

**Original**: "Is this value above threshold?"
- Threshold-based (PTS > 30, TRB > 12, AST > 10)
- Reactive to observed features
- Can't predict future spikes

**Meta-State**: "Is an event likely to occur?"
- Event-based (minutes_spike, usage_spike, etc.)
- Predictive of future events
- Learned gate (not hardcoded threshold)

### 3. Uncertainty Modeling

**Original**: "How uncertain am I?"
- Single uncertainty (sigma)
- Mixes data noise + model uncertainty
- Can "cheat" by adjusting sigma

**Meta-State**: "Why am I uncertain?"
- Separated uncertainties:
  - σ_ale: Data noise (aleatoric)
  - u_epi: Model uncertainty (epistemic)
- Honest uncertainty quantification
- Prevents scale cheating

### 4. Loss Focus

**Original**: "Predict accurately"
- Focus on absolute predictions
- Delta learning via loss convention
- Variance encouragement to prevent collapse

**Meta-State**: "Predict deltas accurately"
- Focus on delta predictions
- Explicit delta supervision
- Event classification for interpretability

## Migration Path

### Option 1: Minimal Changes (5 minutes)

Add to existing `hybrid_spike_moe_trainer.py`:

```python
# After sequence_repr = GlobalAveragePooling1D()(x)
z = Dense(48)(sequence_repr)
z = LayerNormalization()(z)
z = Activation("gelu")(z)

# Use z for routing
sequence_repr_with_baseline = Concatenate()([z, baseline_input])
```

### Option 2: Full Migration (Use New Trainer)

```bash
python training/meta_state_trainer_simple.py
```

## Expected Improvements

### Metrics That Should Improve

1. **R²_macro_delta**: 0.10-0.20 → **0.30-0.50**
   - Better delta learning via explicit supervision

2. **Calibration slope**: 0.10-0.20 → **0.40-0.80**
   - Separated uncertainties prevent scale cheating

3. **PTS variance ratio**: 0.50-0.70 → **0.70-1.00**
   - Event-gated spike experts add variance when needed

4. **Expert usage**: 1-3 active → **6-10 active**
   - Better regularization prevents collapse

5. **Gate activation**: N/A → **0.15-0.25**
   - Spike experts activate for events (not always)

### Metrics That May Stay Similar

1. **R²_macro**: May stay similar initially
   - Delta R² improves first, then raw R²

2. **MAE**: May increase slightly
   - Stopping variance suppression increases MAE
   - But predictions are more "honest"

3. **Training time**: ~10% slower
   - Extra heads (z, events, u_epi)
   - But early stopping may trigger earlier

## Backward Compatibility

### What's Compatible

✅ **Data format** - Same sequences, baselines, targets
✅ **Scaler** - Same StandardScaler on features
✅ **Metadata** - Same player_id, season_id, game_index
✅ **Batching** - Same TimeAwareBatchGenerator
✅ **Training loop** - Same phased training (A→B→C)

### What's Different

❌ **Model weights** - Cannot load original weights into meta-state model
❌ **Output shape** - 6-dim → 14-dim (need to update inference)
❌ **Loss function** - Different loss components
❌ **Config** - New hyperparameters (z_dim, num_events, etc.)

### Inference Changes

**Original**:
```python
preds = model.predict([X, baselines])
mu = preds[:, :3]
sigma = preds[:, 3:6]
```

**Meta-State**:
```python
preds = model.predict([X, baselines])
mu = preds[:, :3]
sigma_ale = preds[:, 3:6]
u_epi = preds[:, 6:9]
events = preds[:, 9:13]
gate = preds[:, 13]

# Total uncertainty
sigma_total = np.sqrt(sigma_ale**2 + u_epi**2)
```

## Recommendation

**Start with Meta-State Trainer** (`meta_state_trainer_simple.py`)

Reasons:
1. ✅ Complete implementation (no manual changes needed)
2. ✅ All improvements included
3. ✅ Easy to compare with original
4. ✅ Can always revert to original if needed

**Keep Original Trainer** for comparison:
1. Run both trainers
2. Compare R²_macro_delta, calibration slope
3. Analyze which architecture works better for your data

## Summary

The meta-state MoE is a **conceptual upgrade** that:

1. Makes routing **interpretable** (via meta-state z)
2. Makes spike experts **predictive** (via event gating)
3. Makes uncertainty **honest** (via separation)
4. Makes delta learning **explicit** (via supervision)

It's **not just a hyperparameter change** - it's a **different way of thinking** about the problem:

- Original: "Match patterns → predict"
- Meta-State: "Understand why → predict"

Both are valid approaches. The meta-state version should give you:
- Better interpretability (what is z learning?)
- Better calibration (separated uncertainties)
- Better delta learning (explicit supervision)
- Better spike expert usage (event-driven)

Try it and see! 🚀
