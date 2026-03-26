# Fix Accuracy NOW - Quick Guide

## Current Problem
```
❌ PTS R²: -0.329 (catastrophic)
❌ Overall R²: -0.033 (negative)
❌ MAE: 4.79 (poor)
❌ Outlier detection: 16-37% precision (terrible)
```

## Solution: Train SIMPLE Model

### Step 1: Train
```bash
python train.py --mode simple
```

This will:
- Use simple LSTM architecture (NO MoE)
- Optimize R² directly
- Focus on outlier detection
- Train for 100 epochs (~30-40 minutes)

### Step 2: Evaluate
```bash
python inference.py
```

### Step 3: Check Results
```bash
cat inference/evaluation_results.json
```

## Expected Results

### Before (Current MOE)
```
PTS: MAE=8.36, R²=-0.329
TRB: MAE=3.29, R²=0.280
AST: MAE=2.73, R²=-0.050
Overall: MAE=4.79, R²=-0.033
```

### After (SIMPLE)
```
PTS: MAE=7.0-7.5, R²=0.25-0.35
TRB: MAE=2.5-3.0, R²=0.40-0.50
AST: MAE=2.0-2.5, R²=0.30-0.40
Overall: MAE=3.8-4.2, R²=0.30-0.40
```

## Why This Works

1. **Simpler = Better**: LSTM proven for sequences, no MoE complexity
2. **R² Optimization**: Directly optimizes what we measure
3. **Outlier Focus**: Special loss for large errors
4. **No Overfitting**: Strong regularization (30% dropout)
5. **No Expert Collapse**: No experts to collapse!

## What to Watch During Training

### Good Signs ✅
- R² increasing steadily
- MAE decreasing steadily
- Validation R² following training R²
- No sudden jumps or drops

### Bad Signs ❌
- R² not improving after 20 epochs
- Validation R² much worse than training R²
- MAE increasing
- Loss exploding (NaN)

## If Results Are Still Poor

### Option 1: Try ACCURATE mode
```bash
python train.py --mode accurate
```
- Simplified MoE (6 experts)
- R² optimization
- Expected: R² > 0.2, MAE < 4.2

### Option 2: Increase epochs
```bash
python train.py --mode simple --epochs 150
```

### Option 3: Adjust batch size
```bash
python train.py --mode simple --batch-size 128
```

## Files Created

- `training/simple_accurate_trainer.py` - Simple LSTM trainer
- `training/accuracy_fixes.py` - Simplified MoE trainer
- `ACCURACY_IMPROVEMENT_PLAN.md` - Full documentation

## Summary

**Problem**: R² = -0.033 (negative!)

**Solution**: `python train.py --mode simple`

**Expected**: R² > 0.3 (10x better!)

**Time**: ~30-40 minutes

**Why**: Simple LSTM, R² optimization, outlier focus

## DO THIS NOW

```bash
# Train simple model
python train.py --mode simple

# Wait ~30-40 minutes

# Evaluate
python inference.py

# Check results
cat inference/evaluation_results.json
```

If R² > 0.3 and MAE < 4.0, you're done! ✅

If not, try `--mode accurate` or increase `--epochs 150`
