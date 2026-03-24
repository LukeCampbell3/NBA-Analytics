# Quick Start: Meta-State MoE Implementation

## TL;DR - What Changed

Your current architecture already does **delta prediction** (`predict_delta=True`), which is great! 

The meta-state upgrade adds:
1. **Explicit "why" reasoning** via meta-state `z`
2. **Event-aware routing** (not just pattern matching)
3. **Separated uncertainties** (epistemic vs aleatoric)
4. **Event-gated spike experts** (not threshold-based)

## Step-by-Step Implementation

### Step 1: Add Meta-State Head (5 minutes)

In `hybrid_spike_moe_trainer.py`, after line 536 (`sequence_repr = GlobalAveragePooling1D...`):

```python
# NEW: Meta-state head (reason encoder)
z = Dense(48, name="meta_state_projection")(sequence_repr)
z = LayerNormalization(name="meta_state_norm")(z)
z = Activation("gelu", name="meta_state_activation")(z)
z = Dropout(self.config["dropout"], name="meta_state_dropout")(z)

# Use z for routing (instead of sequence_repr)
sequence_repr_with_baseline = Concatenate(name="z_baseline_concat")([z, baseline_input])
```

**Test**: Run training, should work identically (z is just a projection of sequence_repr).

### Step 2: Add Event Prediction (10 minutes)

After the meta-state head:

```python
# NEW: Event prediction head
event_logits = Dense(4, activation="sigmoid", name="event_logits")(z)
# Events: [minutes_spike, usage_spike, pace_tier, blowout_risk]
```

Create event labels in `prepare_data()`:

```python
# In prepare_data(), after creating df
df["event_minutes_spike"] = (df["MP"] > 38).astype(float)
df["event_usage_spike"] = ((df["PTS"] + df["AST"]) > 35).astype(float)
df["event_pace_tier"] = (df["MP"] > df["MP"].median()).astype(float)  # Proxy
df["event_blowout"] = 0.0  # Placeholder (need score differential)

# Add to sequences
event_labels = df[["event_minutes_spike", "event_usage_spike", "event_pace_tier", "event_blowout"]].values
# Return event_labels alongside X, baselines, y
```

Add event loss in `create_enhanced_loss()`:

```python
# NEW: Event classification loss
def event_loss_fn(event_true, event_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(event_true, event_pred))

# In main loss function:
event_loss = event_loss_fn(event_true, event_logits) if event_true is not None else 0.0
total += 0.1 * event_loss
```

**Test**: Should see event accuracy metrics, routing should start adapting.

### Step 3: Replace Spike Threshold with Event Gate (15 minutes)

In `ConditionalSpikeOutput`, replace spike indicator logic:

```python
# OLD: spike_indicators from features
# NEW: event-gated spike activation

class EventGatedSpikeOutput(Layer):
    def call(self, inputs):
        delta_output, baseline_input, z, u_epi = inputs
        
        # Event gate (learned, not threshold-based)
        gate_input = Concatenate()([z, u_epi])
        gate = Dense(1, activation="sigmoid", name="spike_gate")(gate_input)
        
        # Split delta_output into base and spike components
        n = len(target_columns)
        delta_base = delta_output[:, :n]
        delta_spike = delta_output[:, n:2*n]  # Spike expert contribution
        
        # Gated combination
        delta_combined = delta_base + gate * delta_spike
        
        # Reconstruct final prediction
        mu = baseline_input + delta_combined
        sigma = delta_output[:, 2*n:3*n]  # Aleatoric uncertainty
        
        return Concatenate()([mu, sigma])
```

Add gate penalty to loss:

```python
# Prevent always-on gating
gate_mean = tf.reduce_mean(gate)
gate_penalty = tf.square(gate_mean - 0.2)  # Target 20% activation
total += 0.05 * gate_penalty
```

**Test**: Gate should activate ~20% of the time, spike experts should specialize.

### Step 4: Separate Epistemic Uncertainty (10 minutes)

Add epistemic head after meta-state:

```python
# NEW: Epistemic uncertainty head
u_epi = Dense(3, activation="softplus", name="epistemic_uncertainty")(z)
# 3-dim for PTS/TRB/AST
```

Update expert outputs to only produce aleatoric:

```python
# Experts output: [delta_mean (3), sigma_ale (3)]
# Total: 6 outputs per expert (not 12)
expert_out = Dense(6, name=f"expert_{i}_out")(expert)
```

Add epistemic supervision (weak):

```python
# Epistemic should correlate with ensemble disagreement or residual variance
# Weak ranking loss: higher epistemic for harder samples
def epistemic_loss_fn(y_true, y_pred, u_epi, baselines):
    residuals = tf.abs(y_true - (baselines + y_pred[:, :3]))
    residual_ranks = tf.argsort(tf.argsort(residuals, axis=0), axis=0)
    epi_ranks = tf.argsort(tf.argsort(u_epi, axis=0), axis=0)
    rank_corr = tf.reduce_mean(tf.cast(residual_ranks, tf.float32) * tf.cast(epi_ranks, tf.float32))
    return -rank_corr  # Maximize correlation

total += 0.01 * epistemic_loss_fn(y_true, y_pred, u_epi, baselines)
```

**Test**: u_epi should be higher for harder-to-predict samples.

### Step 5: Update Loss Functions (10 minutes)

Focus on delta learning:

```python
def create_meta_state_loss(self):
    def loss_fn(y_true, y_pred, baselines, event_true=None):
        n = 3  # PTS, TRB, AST
        
        # Extract outputs
        mu = y_pred[:, :n]  # Reconstructed mean
        sigma_ale = y_pred[:, n:2*n]  # Aleatoric uncertainty
        
        # Delta space
        delta_true = y_true - baselines
        delta_pred = mu - baselines
        
        # 1. Delta Huber loss (explicit delta learning)
        delta_loss = tf.keras.losses.huber(delta_true, delta_pred, delta=2.0)
        
        # 2. Student-t NLL on reconstructed μ (probabilistic)
        df = 4.0
        resid = (y_true - mu) / (sigma_ale + 1e-6)
        nll = -tf.reduce_mean(
            tf.math.lgamma((df + 1) / 2) - tf.math.lgamma(df / 2) -
            0.5 * tf.math.log(df * np.pi) - tf.math.log(sigma_ale + 1e-6) -
            ((df + 1) / 2) * tf.math.log(1 + tf.square(resid) / df)
        )
        
        # 3. Weak mean loss (backup)
        mean_loss = tf.reduce_mean(tf.abs(y_true - mu))
        
        # Combine
        total = (
            0.5 * delta_loss +  # Explicit delta learning
            1.0 * nll +  # Probabilistic loss
            0.05 * mean_loss  # Backup
        )
        
        return total
    
    return loss_fn
```

**Test**: Delta R² should improve faster than raw R².

## Expected Results

After all steps:
- **R²_macro_delta**: Should rise to 0.3-0.5 (was 0.1-0.2)
- **Calibration slope**: Should move toward 0.5-1.0 (was 0.1)
- **PTS variance ratio**: Should improve materially
- **Gate activation**: ~20% (event-driven, not always-on)
- **Expert usage**: 6-10 active experts (not collapsed to 1-2)

## Troubleshooting

**Problem**: Gate always on (>80%)
- Increase `GATE_PENALTY_WEIGHT` to 0.1
- Lower `GATE_MEAN_TARGET` to 0.15

**Problem**: Expert collapse (1-2 experts dominate)
- Increase `EXPERT_USAGE_WEIGHT` to 0.2
- Add expert diversity penalty

**Problem**: Calibration doesn't improve
- Reduce `SIGMA_REG_WEIGHT` to 0.0
- Add quantile coverage loss

**Problem**: Delta R² doesn't improve
- Increase `DELTA_HUBER_WEIGHT` to 1.0
- Check baseline correlation (should be near 0)

## Next Steps

1. **Run Phase 1-5 incrementally** to isolate issues
2. **Monitor new metrics**: gate_activation, epistemic_mean, event_accuracy
3. **Tune hyperparameters** based on results
4. **Add curriculum learning** using delta residual + epistemic
5. **Fix time-aware batching** to prevent data leakage

## Files to Modify

1. `hybrid_spike_moe_trainer.py` - Add meta-state head, event prediction
2. `improved_baseline_trainer.py` - Update loss functions
3. `config` dict - Add new hyperparameters

Or create new file `meta_state_moe_trainer.py` inheriting from `HybridSpikeMoETrainer`.
