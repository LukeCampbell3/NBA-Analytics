# Unified MoE Trainer - Complete Guide

## 🎯 What This Is

A **single unified training file** that combines spike features and expert collapse prevention into one easy-to-use trainer.

## 🚀 Quick Start

```bash
# Train the model
python training/unified_moe_trainer.py

# Evaluate the model
python inference/evaluate_unified_moe.py
```

That's it! The model will train with all anti-collapse mechanisms enabled.

## 📁 Files Created

### Main Files
1. **training/unified_moe_trainer.py** - All-in-one training file (600+ lines)
2. **inference/evaluate_unified_moe.py** - Evaluation script

### Documentation
3. **QUICK_START.md** - Quick reference (read this first!)
4. **UNIFIED_MOE_GUIDE.md** - Detailed guide
5. **CHANGES_SUMMARY.md** - What was changed
6. **BEFORE_AFTER_COMPARISON.md** - Visual comparison
7. **README_UNIFIED_MOE.md** - This file

## 🔧 What Was Fixed

### The Problem
- Expert 0: 99.44% usage ❌
- Experts 1-10: ~0% usage ❌
- Router entropy: 0.032 ❌

### The Solution
8 anti-collapse mechanisms:
1. Router temperature: 5.0 (high exploration)
2. Load balance: 10x stronger (0.01 → 0.1)
3. Router Z-loss: NEW (prevents logit explosion)
4. Orthogonal expert keys: NEW (start different)
5. Delayed compactness (starts epoch 10)
6. Auxiliary variance penalty: NEW
7. Increased capacity (2.0)
8. Increased router noise (0.05)

### Expected Result
- All experts: 5-15% usage ✓
- Router entropy: > 1.5 ✓
- Balanced, healthy MoE ✓

## 📊 What to Watch During Training

```
📊 Expert Usage (Epoch 10):
   Expert 0 (regular): 12.34%  ← Should be 5-15%
   Expert 1 (regular): 10.56%  ← Should be 5-15%
   Expert 2 (regular): 9.87%   ← Should be 5-15%
   ...
   Expert 10 (spike): 10.45%   ← Should be 5-15%
   Router entropy: 2.145       ← Should be > 1.5
   Router Z-loss: 45.32        ← Should be < 100
```

## ⚙️ Configuration

All settings are in `UnifiedMoETrainer.__init__()` around line 200-300.

### Key Settings
```python
"router_temperature": 5.0,           # High exploration
"load_balance_weight_final": 0.1,    # Strong balancing
"router_z_loss_weight": 0.001,       # Logit stability
"compactness_ramp_epochs": [10, 25], # Delayed specialization
"capacity_factor": 2.0,              # More tokens per expert
```

### If Experts Still Collapse
Increase these values:
```python
"router_temperature": 7.0,  # or 10.0
"load_balance_weight_final": 0.2,  # or 0.3
```

## 📈 Expected Training Time

~30-60 minutes for 50 epochs (depends on hardware)

## ✅ Success Criteria

Training is successful if:
- ✓ All 11 experts at 5-15% usage
- ✓ Router entropy > 1.5
- ✓ No collapse warnings
- ✓ No training instabilities
- ✓ MAE < 4.795 (better than old ensemble)

## 🔍 Troubleshooting

### Problem: Experts still collapsing
**Solution**: Increase `router_temperature` to 7.0-10.0

### Problem: Training unstable (NaN loss)
**Solution**: Decrease `router_z_loss_weight` to 0.0001

### Problem: Low accuracy despite balanced experts
**Solution**: Increase `compactness_final` to 0.01 after epoch 15

### Problem: Import errors
**Solution**: Make sure you're in the project root directory

## 📚 Documentation Guide

1. **Start here**: QUICK_START.md
2. **Detailed info**: UNIFIED_MOE_GUIDE.md
3. **What changed**: CHANGES_SUMMARY.md
4. **Visual comparison**: BEFORE_AFTER_COMPARISON.md
5. **This file**: README_UNIFIED_MOE.md

## 🗂️ Old Files (Can Archive)

These are now replaced by `unified_moe_trainer.py`:
- `training/hybrid_spike_moe_trainer.py`
- `training/integrate_moe_improvements.py`

You can keep them for reference or delete them.

## 🎓 Technical Details

### Model Architecture
- 11 total experts (8 regular + 3 spike)
- Top-2 routing (each sample uses 2 experts)
- Transformer encoder (4 layers, 8 heads, 256 dim)
- Supervised outlier detection
- Delta-only training (predicts residuals)

### Anti-Collapse Mechanisms
1. **High Temperature**: Softer routing probabilities
2. **Strong Load Balance**: Forces balanced usage
3. **Router Z-Loss**: Prevents logit explosion
4. **Orthogonal Keys**: Experts start different
5. **Delayed Compactness**: Allows exploration first
6. **Variance Penalty**: Direct pressure on distribution
7. **High Capacity**: More tokens per expert
8. **Router Noise**: Exploration during training

### Loss Components
- Delta loss (Huber)
- Band loss (within-deviation)
- Outlier loss (focal BCE)
- Load balance loss
- Router Z-loss
- Compactness loss (delayed)
- Diversity loss

## 🔬 Research Sources

Based on latest 2024-2025 MoE research:
- Switch Transformer (Google)
- Mixtral 8x7B (Mistral AI)
- Auxiliary balancing loss techniques
- Router z-loss for stability

All content rephrased for licensing compliance.

## 💡 Key Insights

### Why Experts Collapsed Before
1. Low router temperature (2.0) → winner-takes-all
2. Weak load balance (0.001-0.008) → couldn't prevent collapse
3. No z-loss → logits grew unbounded
4. RandomNormal keys → experts started similar
5. Early compactness → forced premature specialization

### How We Fixed It
1. High temperature (5.0) → softer probabilities
2. Strong load balance (0.01-0.1) → forces balance
3. Z-loss (0.001) → prevents logit explosion
4. Orthogonal keys → experts start different
5. Delayed compactness → allows exploration

## 🎯 Bottom Line

You now have:
- ✅ Single unified training file
- ✅ All anti-collapse fixes integrated
- ✅ Easy to use and modify
- ✅ Well documented

Just run `python training/unified_moe_trainer.py` and watch the expert usage metrics. If all experts are balanced (5-15% each), the problem is solved!

## 📞 Next Steps

1. Read QUICK_START.md
2. Run training: `python training/unified_moe_trainer.py`
3. Monitor expert usage during training
4. Evaluate: `python inference/evaluate_unified_moe.py`
5. Compare results to old ensemble (MAE=4.795)

Good luck! 🚀
