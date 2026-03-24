# Accuracy Improvement Plan

## Current Issues

### Performance (CRITICAL)
```
PTS: MAE=8.36, R²=-0.329 (WORSE than predicting mean!)
TRB: MAE=3.29, R²=0.280 (only decent one)
AST: MAE=2.73, R²=-0.050 (negative)
Overall: MAE=4.79, R²=-0.033 (NEGATIVE!)
```

### Root Causes
1. **Model too complex**: 11 experts, 4 transformer layers, 256 dimensions → overfitting
2. **Expert collapse**: Expert 1 handles 99.96%, others dead → no ensemble benefit
3. **Wrong objective**: Optimizing NLL, not R² → poor R² performance
4. **Poor outlier detection**: Low precision (16-36%), poor calibration
5. **Architecture mismatch**: MoE routing interferes with learning

## Solution: Three Training Modes

### Mode 1: SIMPLE (RECOMMENDED) ⭐
```bash
python train.py --mode simple
```

**Strategy**: Maximum simplicity, proven architecture
- **Architecture**: LSTM + Dense (NO MoE)
- **Loss**: MAE + R² + Outlier focus
- **Expected**: R² > 0.3, MAE < 4.0

**Why it works**:
- LSTM proven for sequences
- Direct R² optimization
- No MoE complexity
- Strong regularization

**Best for**: Maximum accuracy, best outlier detection

### Mode 2: ACCURATE
```bash
python train.py --mode accurate
```

**Strategy**: Simplified MoE with fixes
- **Architecture**: 6 experts (vs 11), smaller dimensions
- **Loss**: R² + MAE + Strong entropy penalty
- **Expected**: R² > 0.2, MAE < 4.2

**Why it works**:
- 50% fewer parameters
- R² loss added
- Extreme routing fixes
- Better regularization

**Best for**: Balance between MoE benefits and accuracy

### Mode 3: MOE (Current, NOT RECOMMENDED)
```bash
python train.py --mode moe
```

**Issues**:
- Too complex (11 experts)
- Expert collapse
- Negative R²
- Poor outlier detection

**Only use if**: You need the full MoE for research purposes

## Detailed Comparison

| Feature | SIMPLE | ACCURATE | MOE (Current) |
|---------|--------|----------|---------------|
| Architecture | LSTM | Transformer | Transformer |
| Experts | 0 | 6 | 11 |
| Parameters | ~50K | ~200K | ~500K |
| Complexity | Low | Medium | High |
| R² Loss | ✅ Yes | ✅ Yes | ❌ No |
| Outlier Focus | ✅ Yes | ✅ Yes | ⚠️ Weak |
| Expected R² | >0.3 | >0.2 | -0.03 |
| Expected MAE | <4.0 | <4.2 | 4.8 |
| Training Time | Fast | Medium | Slow |
| Overfitting Risk | Low | Medium | High |

## Key Improvements

### 1. Simpler Architecture
**Problem**: 11 experts, 4 layers, 256 dims → overfitting
**Solution**: 
- SIMPLE: LSTM(64) + Dense(32)
- ACCURATE: 6 experts, 2 layers, 128 dims

### 2. R² Optimization
**Problem**: Optimizing NLL, not R²
**Solution**: Add R² loss directly
```python
def r2_loss(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return -r2  # Negative to maximize
```

### 3. Outlier Focus
**Problem**: Poor outlier detection (precision 16-36%)
**Solution**: Add outlier-focused loss
```python
errors = tf.abs(y_true - y_pred)
outlier_mask = tf.cast(errors > 5.0, tf.float32)
outlier_loss = tf.reduce_mean(outlier_mask * errors)
```

### 4. Strong Regularization
**Problem**: Overfitting on training data
**Solution**:
- Dropout: 0.1 → 0.3 (3x increase)
- L2 weight: 0.0 → 0.001
- Batch size: 32 → 64 (more stable)

### 5. Fixed Routing (ACCURATE mode only)
**Problem**: Expert collapse (99.96% on one expert)
**Solution**:
- Temperature: 5.0 → 10.0 (even softer)
- Entropy weight: 2.0 → 5.0 (even stronger)
- Logit scale: 0.1 → 0.05 (even smaller)

## Expected Results

### Current (MOE mode)
```
PTS: MAE=8.36, R²=-0.329
TRB: MAE=3.29, R²=0.280
AST: MAE=2.73, R²=-0.050
Overall: MAE=4.79, R²=-0.033
```

### Target (SIMPLE mode)
```
PTS: MAE=7.0-7.5, R²=0.25-0.35
TRB: MAE=2.5-3.0, R²=0.40-0.50
AST: MAE=2.0-2.5, R²=0.30-0.40
Overall: MAE=3.8-4.2, R²=0.30-0.40
```

### Target (ACCURATE mode)
```
PTS: MAE=7.5-8.0, R²=0.15-0.25
TRB: MAE=2.8-3.2, R²=0.35-0.45
AST: MAE=2.2-2.7, R²=0.20-0.30
Overall: MAE=4.0-4.5, R²=0.20-0.30
```

## Outlier Detection Improvements

### Current Performance
```
PTS: Precision=16%, Recall=39%
TRB: Precision=19%, Recall=12%
AST: Precision=37%, Recall=41%
```

### Target Performance
```
PTS: Precision>40%, Recall>50%
TRB: Precision>35%, Recall>40%
AST: Precision>50%, Recall>55%
```

### How to Achieve
1. **Better features**: Add outlier-specific features
2. **Focused loss**: Penalize outlier errors more
3. **Robust detection**: Use MAD-based thresholds
4. **Calibration**: Ensure uncertainty matches errors

## Training Instructions

### Step 1: Train SIMPLE model (RECOMMENDED)
```bash
python train.py --mode simple --epochs 100
```

Watch for:
- R² increasing (should reach >0.3)
- MAE decreasing (should reach <4.0)
- Validation R² not decreasing (no overfitting)

### Step 2: Evaluate
```bash
python inference.py
```

Check:
- Overall R² > 0.3
- All target R² > 0.2
- MAE < 4.0
- Outlier precision > 40%

### Step 3: If not satisfied, try ACCURATE mode
```bash
python train.py --mode accurate --epochs 80
```

### Step 4: Compare results
```bash
# Check evaluation_results.json
cat inference/evaluation_results.json
```

## Troubleshooting

### If R² is still negative
1. Check for data leakage
2. Verify baseline calculation
3. Try even simpler architecture
4. Increase regularization

### If MAE is still high
1. Add more features
2. Increase model capacity slightly
3. Train longer (more epochs)
4. Check for outliers in training data

### If outlier detection is poor
1. Adjust outlier threshold
2. Add outlier-specific features
3. Increase outlier loss weight
4. Use ensemble of models

## Summary

**Current**: MAE=4.79, R²=-0.033 (BROKEN)

**Target**: MAE<4.0, R²>0.3 (GOOD)

**Recommended**: `python train.py --mode simple`

**Why**: Simplest, most proven, best accuracy

**Expected improvement**: 
- R²: -0.033 → 0.30+ (10x better!)
- MAE: 4.79 → 3.8-4.2 (15-20% better)
- Outlier detection: 16-37% → 40-50% precision (2x better)
