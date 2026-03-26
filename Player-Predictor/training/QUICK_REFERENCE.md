# Meta-State MoE - Quick Reference Card

## 🚀 Quick Start

```bash
# Run the new meta-state trainer
python training/meta_state_trainer_simple.py
```

## 📊 New Metrics to Watch

```
gate_activation: 0.18-0.25  ← Spike expert usage (target: ~20%)
active_experts: 6-10        ← Active experts (target: 6-10)
router_entropy: 1.3-1.8     ← Routing diversity (target: ~1.5)
epistemic_mean: 1.5-3.0     ← Model uncertainty
```

## 🎯 Expected Improvements

| Metric | Before | After |
|--------|--------|-------|
| R²_macro_delta | 0.10-0.20 | **0.30-0.50** |
| Calibration slope | 0.10-0.20 | **0.40-0.80** |
| PTS variance ratio | 0.50-0.70 | **0.70-1.00** |
| Active experts | 1-3 | **6-10** |

## 🔧 Key Hyperparameters

```python
{
    "z_dim": 48,                    # Meta-state dimension
    "delta_huber_weight": 0.5,      # Explicit delta learning
    "gate_penalty_weight": 0.05,    # Prevent always-on gate
    "epistemic_weight": 0.01,       # Epistemic supervision
    "calibration_weight": 0.05,     # Quantile coverage
}
```

## 🐛 Troubleshooting

### Gate always on (>80%)
```python
config["gate_penalty_weight"] = 0.1  # ⬆️ Increase
config["gate_mean_target"] = 0.15    # ⬇️ Lower
```

### Expert collapse (1-2 experts)
```python
config["expert_usage_weight"] = 0.2  # ⬆️ Increase
```

### Delta R² not improving
```python
config["delta_huber_weight"] = 1.0   # ⬆️ Increase
config["cov_penalty_weight"] = 0.5   # ⬆️ Increase
```

### Calibration not improving
```python
config["sigma_regularization_weight"] = 0.0  # ❌ Remove
config["calibration_weight"] = 0.1           # ⬆️ Increase
```

## 📁 Files Created

```
training/
├── META_STATE_SUMMARY.md              ← Start here!
├── QUICK_START_META_STATE.md          ← Step-by-step guide
├── RUN_META_STATE_TRAINER.md          ← How to run
├── COMPARE_ARCHITECTURES.md           ← Original vs Meta-State
├── QUICK_REFERENCE.md                 ← This file
├── meta_state_trainer_simple.py       ← RUN THIS!
├── meta_state_moe_trainer_core.py     ← Architecture
└── meta_state_losses.py               ← Loss functions
```

## 🏗️ Architecture Changes

### What Changed
1. ✅ Meta-state head `z` (reason encoder)
2. ✅ Event prediction (4 proxy events)
3. ✅ Event-gated spike experts (not threshold-based)
4. ✅ Separated epistemic/aleatoric uncertainty
5. ✅ Delta-focused loss functions

### What Stayed Same
1. ✅ Data preparation
2. ✅ Transformer encoder
3. ✅ Time-aware batching
4. ✅ Phased training (A→B→C)
5. ✅ Delta prediction (`predict_delta=True`)

## 📈 Output Structure

**Original** (6-dim):
```python
[mu_PTS, mu_TRB, mu_AST, sigma_PTS, sigma_TRB, sigma_AST]
```

**Meta-State** (14-dim):
```python
[
    mu_PTS, mu_TRB, mu_AST,              # Means (3)
    sigma_ale_PTS, sigma_ale_TRB, sigma_ale_AST,  # Aleatoric (3)
    u_epi_PTS, u_epi_TRB, u_epi_AST,     # Epistemic (3)
    event_1, event_2, event_3, event_4,  # Events (4)
    gate                                  # Gate (1)
]
```

## 🔍 Inference Example

```python
# Load model
model = tf.keras.models.load_model("model/improved_baseline_final.h5")

# Predict
preds = model.predict([X, baselines])

# Extract outputs
mu = preds[:, :3]                # Predictions
sigma_ale = preds[:, 3:6]        # Data noise
u_epi = preds[:, 6:9]            # Model uncertainty
events = preds[:, 9:13]          # Event probabilities
gate = preds[:, 13]              # Spike gate

# Total uncertainty
sigma_total = np.sqrt(sigma_ale**2 + u_epi**2)

# Confidence intervals
lower = mu - 1.96 * sigma_total
upper = mu + 1.96 * sigma_total
```

## 📚 Documentation

1. **Overview**: `META_STATE_SUMMARY.md`
2. **Step-by-step**: `QUICK_START_META_STATE.md`
3. **How to run**: `RUN_META_STATE_TRAINER.md`
4. **Comparison**: `COMPARE_ARCHITECTURES.md`
5. **Implementation plan**: `META_STATE_IMPLEMENTATION_PLAN.md`

## ✅ Success Criteria

After training, you should see:

- ✅ R²_macro_delta > 0.30 (was 0.10-0.20)
- ✅ Calibration slope > 0.50 (was 0.10-0.20)
- ✅ Gate activation ~0.20 (not always-on)
- ✅ 6-10 active experts (not collapsed)
- ✅ Epistemic correlates with errors

## 🎓 Key Concepts

### Meta-State `z`
- **What**: 48-dim representation of "why this might happen"
- **Purpose**: Interpretable routing, event prediction
- **Used by**: Router, event head, epistemic head, gate

### Event Gating
- **What**: Learned gate (not threshold-based)
- **Input**: z + u_epi
- **Output**: Probability of spike expert activation
- **Target**: ~20% activation (event-driven)

### Uncertainty Separation
- **Aleatoric (σ_ale)**: Data noise (irreducible)
- **Epistemic (u_epi)**: Model uncertainty (reducible)
- **Why**: Prevents "scale cheating", honest calibration

### Delta Focus
- **What**: Explicit supervision on Δy = y - baseline
- **Why**: Prevents collapse, improves variance
- **How**: Delta Huber loss + variance encouragement

## 🚦 Next Steps

1. **Run trainer**: `python training/meta_state_trainer_simple.py`
2. **Monitor metrics**: gate_activation, active_experts, epistemic_mean
3. **Compare results**: R²_macro_delta, calibration slope
4. **Tune hyperparameters**: Based on observed metrics
5. **Analyze interpretability**: What is z learning? When does gate activate?

## 💡 Pro Tips

1. **Start with defaults** - Don't tune hyperparameters until you see results
2. **Monitor gate activation** - Should be ~20%, not 80%
3. **Check expert usage** - Should be 6-10 active, not 1-2
4. **Analyze epistemic** - Should be higher for harder samples
5. **Compare with original** - Run both trainers to see difference

## 📞 Support

If stuck, read:
1. `META_STATE_SUMMARY.md` - Complete overview
2. `RUN_META_STATE_TRAINER.md` - Troubleshooting guide
3. `COMPARE_ARCHITECTURES.md` - What changed and why

Good luck! 🚀
