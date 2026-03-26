# Quick Start - Unified MoE Trainer

## TL;DR

```bash
# Train the model (with anti-collapse fixes)
python training/unified_moe_trainer.py

# Evaluate the model
python inference/evaluate_unified_moe.py
```

## What's Different?

### Old Model (Collapsed)
- Expert 0: 99.44% usage ❌
- Experts 1-10: ~0% usage ❌
- Router entropy: 0.032 ❌
- MAE: 4.795

### New Model (Expected)
- All experts: 5-15% usage ✓
- Router entropy: > 1.5 ✓
- MAE: < 4.795 (hopefully) ✓

## Key Anti-Collapse Features

1. **Router Temperature: 5.0** (was 2.0)
   - Prevents winner-takes-all

2. **Load Balance: 10x Stronger** (0.01 → 0.1)
   - Forces balanced expert usage

3. **Router Z-Loss: NEW** (0.001)
   - Prevents logit explosion

4. **Orthogonal Expert Keys: NEW**
   - Experts start maximally different

5. **Delayed Compactness** (starts epoch 10)
   - Allows exploration first

## What to Watch During Training

```
📊 Expert Usage (Epoch 10):
   Expert 0: 12.34%  ← Should be 5-15%, not 99%
   Expert 1: 10.56%  ← Should be 5-15%, not 0%
   Expert 2: 9.87%   ← Should be 5-15%, not 0%
   ...
   Router entropy: 2.145  ← Should be > 1.5, not 0.032
```

## If Experts Still Collapse

Edit `training/unified_moe_trainer.py` line ~250:

```python
"router_temperature": 7.0,  # Increase from 5.0 to 7.0 or 10.0
"load_balance_weight_final": 0.2,  # Increase from 0.1 to 0.2
```

## Files

- **training/unified_moe_trainer.py** - All-in-one training file
- **inference/evaluate_unified_moe.py** - Evaluation script
- **UNIFIED_MOE_GUIDE.md** - Detailed guide
- **QUICK_START.md** - This file

## Expected Training Time

~30-60 minutes for 50 epochs (depends on hardware)

## Success Criteria

- ✓ All 11 experts at 5-15% usage
- ✓ Router entropy > 1.5
- ✓ No collapse warnings
- ✓ MAE < 4.795 (better than old ensemble)

## That's It!

Just run the training script and watch the expert usage metrics. If all experts are balanced (5-15% each), the anti-collapse mechanisms worked!
