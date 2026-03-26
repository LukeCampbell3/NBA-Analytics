# Unified Training and Inference Structure

## Overview

The codebase has been simplified into two main entry points:
- `train.py` - All training functionality
- `inference.py` - All inference functionality

## File Structure

```
Player-Predictor/
├── train.py              # ⭐ UNIFIED TRAINING ENTRY POINT
├── inference.py          # ⭐ UNIFIED INFERENCE ENTRY POINT
├── training/             # Original training modules (imported by train.py)
│   ├── hybrid_spike_moe_trainer.py
│   ├── improved_baseline_trainer.py
│   ├── moe_metrics.py
│   └── ...
├── inference/            # Original inference modules (imported by inference.py)
│   ├── ensemble_inference.py
│   ├── evaluate_simple.py
│   └── ...
├── model/                # Saved models and metadata
└── Data/                 # Training data
```

## Quick Start

### Training

```bash
# Train MoE model (default, recommended)
python train.py

# Train baseline model (no MoE)
python train.py --mode baseline

# Train ensemble of 3 models
python train.py --mode ensemble

# Custom parameters
python train.py --mode moe --epochs 30 --batch-size 64
```

### Inference

```bash
# Evaluate trained model
python inference.py

# With visualizations (coming soon)
python inference.py --visualize

# Predict next game (coming soon)
python inference.py --predict --player "LeBron_James"
```

## Training Modes

### 1. MoE Mode (Default)
```bash
python train.py --mode moe
```
- Trains Hybrid Spike-Driver MoE model
- 11 experts (8 regular + 3 spike)
- Emergency fixes applied:
  - Router logits scaled down 10x
  - Entropy weight increased 2000x
- Expected: Balanced expert usage, R² > 0.15

### 2. Baseline Mode
```bash
python train.py --mode baseline
```
- Simple baseline model without MoE
- Baseline + delta architecture
- Faster training, simpler model

### 3. Ensemble Mode
```bash
python train.py --mode ensemble
```
- Trains 3 MoE models with different strategies:
  1. high_confidence_spike_focused
  2. balanced_confidence_variance
  3. stable_confident_routing
- Aggregates predictions for better performance

## Command Line Arguments

### train.py
```
--mode {moe,baseline,ensemble}  Training mode (default: moe)
--epochs INT                    Number of epochs (default: 50)
--batch-size INT                Batch size (default: 32)
--seq-len INT                   Sequence length (default: 10)
```

### inference.py
```
--visualize                     Generate visualizations
--predict                       Predict next game
--player STR                    Player name for prediction
--games INT                     Number of recent games (default: 10)
```

## What Changed

### Before (Multiple Files)
```
training/hybrid_spike_moe_trainer.py      # MoE training
training/improved_baseline_trainer.py     # Baseline training
training/integrate_moe_improvements.py    # Incomplete improvements
inference/ensemble_inference.py           # Ensemble evaluation
inference/evaluate_simple.py              # Simple evaluation
inference/predict_game.py                 # Game prediction
```

### After (Unified)
```
train.py                                  # All training
inference.py                              # All inference
```

## Benefits

1. **Simpler**: One command for training, one for inference
2. **Clearer**: No confusion about which file to run
3. **Flexible**: Easy to switch between modes
4. **Maintainable**: Changes in one place
5. **Documented**: Clear help messages and examples

## Emergency Fixes Applied

The MoE trainer has emergency fixes for expert collapse:

1. **Router Logit Scaling** (line ~708 in hybrid_spike_moe_trainer.py):
   ```python
   router_logits = router_logits * 0.1  # Scale down 10x
   ```

2. **Massive Entropy Penalty** (line ~815 in hybrid_spike_moe_trainer.py):
   ```python
   entropy_weight = 2.0  # 2000x increase from 0.001
   ```

These fixes force expert diversity and prevent collapse.

## Monitoring Training

Watch these metrics during training:

### Success Indicators (First 5 Epochs)
- ✅ `router_entropy` > 1.0 (increasing toward 1.8)
- ✅ `expert_X_usage` spreading out (5-15% each)
- ✅ `avg_max_prob` < 0.7 (decreasing from 0.999)
- ✅ No single expert >40% usage

### Warning Signs
- ❌ `router_entropy` < 0.5 (still collapsed)
- ❌ Any expert >60% usage
- ❌ `avg_max_prob` > 0.9 (too confident)

## Expected Performance

### Current (With Collapse)
```
MAE: 4.795
R²:  -0.033 (negative!)
Expert 1: 99.96% usage
```

### Target (With Fixes)
```
MAE: 4.0-4.5
R²:  0.15-0.25 (positive)
Each expert: 5-15% usage
```

## Troubleshooting

### If Training Fails
1. Check data is in `Data/` directory
2. Verify Python 3.8+ and TensorFlow 2.x installed
3. Check GPU memory if using GPU

### If Expert Collapse Continues
1. Increase entropy weight in `training/hybrid_spike_moe_trainer.py`:
   ```python
   entropy_weight = 5.0  # or 10.0
   ```

2. Scale logits down more:
   ```python
   router_logits = router_logits * 0.05  # or 0.02
   ```

### If Performance is Poor
1. Try ensemble mode: `python train.py --mode ensemble`
2. Increase epochs: `python train.py --epochs 100`
3. Check evaluation results: `inference/evaluation_results.json`

## Next Steps

1. Run training: `python train.py`
2. Monitor first 5 epochs for expert diversity
3. Run evaluation: `python inference.py`
4. Check results: `inference/evaluation_results.json`

## Original Files

The original training and inference files are still in their directories:
- `training/` - All training modules
- `inference/` - All inference modules

The unified files (`train.py` and `inference.py`) import and orchestrate these modules.

## Support

For issues or questions:
1. Check `VALIDATION_REPORT.md` for current status
2. Check `FIXES_APPLIED_TO_CORRECT_FILE.md` for fix details
3. Check `training/EMERGENCY_FIX.md` for root cause analysis
