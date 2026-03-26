# How to Run Meta-State MoE Trainer

## Quick Start (5 minutes)

### Option 1: Minimal Changes to Existing Code

Add these lines to `hybrid_spike_moe_trainer.py` in the `build_model()` method:

**After line 536** (`sequence_repr = GlobalAveragePooling1D...`):

```python
# ===== META-STATE UPGRADE START =====
# 1. Meta-state head
z_dim = self.config.get("z_dim", 48)
z = Dense(z_dim, name="meta_state")(sequence_repr)
z = LayerNormalization()(z)
z = Activation("gelu")(z)
z = Dropout(self.config["dropout"])(z)

# 2. Event prediction
event_logits = Dense(4, activation="sigmoid", name="events")(z)

# 3. Epistemic uncertainty
u_epi = Dense(len(self.target_columns), activation="softplus", name="epistemic")(z)

# 4. Use z for routing (replace sequence_repr_with_baseline)
sequence_repr_with_baseline = Concatenate()([z, baseline_input])
# ===== META-STATE UPGRADE END =====
```

**Update config** in `__init__`:

```python
self.config.update({
    "z_dim": 48,
    "use_meta_state": True,
    "event_loss_weight": 0.1,
    "epistemic_weight": 0.01,
    "gate_penalty_weight": 0.05,
})
```

**Update loss** in `improved_baseline_trainer.py`:

Add to `create_enhanced_loss()`:

```python
# Extract new outputs
n = len(self.target_columns)
mu = y_pred[:, :n]
sigma_ale = y_pred[:, n:2*n]
u_epi = y_pred[:, 2*n:3*n] if y_pred.shape[-1] > 2*n else None
event_logits = y_pred[:, 3*n:3*n+4] if y_pred.shape[-1] > 3*n else None

# Event loss
if event_logits is not None and event_true is not None:
    event_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(event_true, event_logits))
    total += self.config.get("event_loss_weight", 0.1) * event_loss

# Epistemic supervision (weak)
if u_epi is not None:
    abs_residuals = tf.abs(y_true - mu)
    residual_difficulty = tf.reduce_mean(abs_residuals, axis=1)
    epi_magnitude = tf.reduce_mean(u_epi, axis=1)
    
    # Rank correlation
    residual_ranks = tf.cast(tf.argsort(tf.argsort(residual_difficulty)), tf.float32)
    epi_ranks = tf.cast(tf.argsort(tf.argsort(epi_magnitude)), tf.float32)
    rank_corr = tf.reduce_mean(residual_ranks * epi_ranks)
    
    epistemic_loss = -rank_corr / (tf.cast(tf.shape(residual_ranks)[0], tf.float32) + 1e-6)
    total += self.config.get("epistemic_weight", 0.01) * epistemic_loss
```

**Run training**:

```bash
python training/improved_baseline_trainer.py
```

### Option 2: Use New Standalone Trainer

Create `training/meta_state_trainer_simple.py`:

```python
#!/usr/bin/env python3
"""Simple Meta-State MoE Trainer"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from improved_baseline_trainer import ImprovedBaselineTrainer
from meta_state_moe_trainer_core import build_meta_state_moe
from meta_state_losses import create_meta_state_loss


class MetaStateTrainer(ImprovedBaselineTrainer):
    """Meta-State MoE Trainer - inherits from ImprovedBaselineTrainer"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add meta-state config
        self.config.update({
            "z_dim": 48,
            "num_events": 4,
            "event_loss_weight": 0.1,
            "epistemic_weight": 0.01,
            "gate_penalty_weight": 0.05,
            "gate_mean_target": 0.2,
            "expert_usage_weight": 0.1,
            "delta_huber_weight": 0.5,
            "nll_weight": 1.0,
            "calibration_weight": 0.05,
        })
    
    def build_model(self):
        """Build meta-state MoE instead of original"""
        return build_meta_state_moe(
            config=self.config,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            player_mapping=self.player_mapping,
            team_mapping=self.team_mapping,
            opponent_mapping=self.opponent_mapping
        )
    
    def create_enhanced_loss(self):
        """Use meta-state loss functions"""
        return create_meta_state_loss(self.config, self.target_columns)


def main():
    trainer = MetaStateTrainer(ensemble_size=1)
    model, meta = trainer.train()
    print(f"\n✅ Training complete! R²_macro: {meta['final_performance']['r2_macro']:.3f}")


if __name__ == "__main__":
    main()
```

**Run**:

```bash
python training/meta_state_trainer_simple.py
```

## What to Expect

### New Metrics

You'll see these new metrics during training:

```
gate_activation: 0.18-0.25  (target: ~0.20)
active_experts: 6-10  (target: 6-10)
router_entropy: 1.3-1.8  (target: ~1.5)
epistemic_mean: 1.5-3.0  (higher for harder samples)
event_accuracy: 0.6-0.8  (proxy events)
```

### Performance Improvements

Expected improvements after meta-state upgrade:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| R²_macro_delta | 0.10-0.20 | 0.30-0.50 | >0.30 |
| Calibration slope | 0.10-0.20 | 0.40-0.80 | >0.50 |
| PTS variance ratio | 0.50-0.70 | 0.70-1.00 | 0.60-1.20 |
| Gate activation | N/A | 0.15-0.25 | 0.15-0.25 |
| Expert collapse | 1-3 active | 6-10 active | 6-10 |

### Training Time

- **First epoch**: ~10-15% slower (extra heads)
- **Overall**: Similar (early stopping may trigger earlier)

## Troubleshooting

### Problem: Gate always on (>80%)

```python
# Increase penalty
self.config["gate_penalty_weight"] = 0.1  # was 0.05
self.config["gate_mean_target"] = 0.15  # was 0.20
```

### Problem: Expert collapse (1-2 experts)

```python
# Increase expert usage penalty
self.config["expert_usage_weight"] = 0.2  # was 0.1

# Add diversity penalty
self.config["expert_diversity_weight"] = 0.05
self.config["min_expert_usage"] = 0.05
```

### Problem: Delta R² not improving

```python
# Increase delta learning
self.config["delta_huber_weight"] = 1.0  # was 0.5

# Reduce NLL weight temporarily
self.config["nll_weight"] = 0.5  # was 1.0

# Check baseline correlation
# Should be near 0, if >0.3 increase cov_penalty_weight
```

### Problem: Calibration not improving

```python
# Remove sigma regularization
self.config["sigma_regularization_weight"] = 0.0

# Increase calibration weight
self.config["calibration_weight"] = 0.1  # was 0.05
```

### Problem: NaN losses

```python
# Clip gradients more aggressively
optimizer = tf.keras.optimizers.Adam(lr=1e-4, clipnorm=0.5)

# Reduce learning rate
self.config["lr"] = 5e-5  # was 1e-4

# Check for extreme values in data
print(f"Max baseline: {np.max(baselines)}")
print(f"Max target: {np.max(y)}")
```

## Next Steps

After successful meta-state training:

1. **Analyze gate activation patterns**
   - Which samples trigger spike experts?
   - Are they actually "events" (high minutes, usage spikes)?

2. **Inspect epistemic uncertainty**
   - Does u_epi correlate with prediction errors?
   - Are hard samples getting higher u_epi?

3. **Check expert specialization**
   - Are different experts activating for different player types?
   - Use `expert_i_usage` metrics to see distribution

4. **Implement curriculum learning**
   - Use `delta_residual + u_epi` for difficulty scoring
   - Mine hard examples for Phase 2 training

5. **Add real event labels**
   - Replace proxy events with actual game context
   - Minutes played, rest days, opponent strength, etc.

## Files Created

```
training/
├── META_STATE_IMPLEMENTATION_PLAN.md  (overview)
├── QUICK_START_META_STATE.md          (step-by-step guide)
├── RUN_META_STATE_TRAINER.md          (this file)
├── meta_state_moe_trainer_core.py     (architecture)
├── meta_state_losses.py               (loss functions)
└── meta_state_trainer_simple.py       (complete trainer - create this)
```

## Support

If you encounter issues:

1. Check `training/QUICK_START_META_STATE.md` for detailed explanations
2. Review `training/META_STATE_IMPLEMENTATION_PLAN.md` for architecture overview
3. Compare with original `hybrid_spike_moe_trainer.py` to see what changed
4. Run with `verbose=2` to see detailed metrics per batch

Good luck! 🚀
