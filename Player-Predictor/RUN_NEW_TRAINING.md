# Run New Training with Expert Collapse Fixes

## IMPORTANT: You ran the WRONG training script!

The training that just completed used the OLD ensemble trainer (`hybrid_spike_moe_trainer.py`) which does NOT have the expert collapse fixes.

You need to run the NEW training script with all the fixes we just implemented.

## Run This Command:

```bash
python training/integrate_moe_improvements.py
```

## What This Will Do:

1. Train with the new MoE improvements:
   - Router temperature: 5.0 (was 2.0)
   - Load balance weights: 10x stronger
   - Router Z-Loss: NEW stability mechanism
   - Orthogonal expert key initialization
   - Delayed compactness schedule
   - Auxiliary variance penalty

2. Save model to: `model/improved_baseline_final.weights.h5`

3. You should see in the logs:
   - Router temperature: 5.0 (not 2.0)
   - Expert usage becoming more balanced over epochs
   - Router entropy increasing (target > 1.5)
   - All 11 experts being used (not just Expert 0)

## How to Monitor Success:

Watch these metrics during training:

```
expert_0_usage: should decrease from 99% to ~9-15%
expert_1_usage: should increase from 0% to ~9-15%
expert_2_usage: should increase from 0% to ~9-15%
... (all experts should be 5-15%)

router_entropy: should increase from 0.032 to > 1.5
router_z_loss: should stay < 100
router_lse_mean: should stay < 10
```

## After Training Completes:

The model will be saved, but you'll need to create a NEW inference script because:
- The old `ensemble_inference.py` loads the OLD ensemble models
- The new model is a single model, not an ensemble
- Use the evaluation scripts we created earlier:
  - `inference/evaluate_direct.py` 
  - `inference/eval_h5.py`

Or I can create a new inference script specifically for the improved model.

## Current Status:

❌ You just trained: `hybrid_spike_moe_trainer.py` (OLD, has expert collapse)
✅ You need to train: `integrate_moe_improvements.py` (NEW, has all fixes)

The evaluation results you're seeing (MAE=4.79) are from the OLD model without fixes.
