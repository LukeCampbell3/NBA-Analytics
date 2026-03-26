# Training Directory

## Current Training System

### Main Trainer
- **`improved_baseline_trainer.py`** - Current production trainer with all V2 fixes
  - Inherits from `hybrid_spike_moe_trainer.py`
  - Implements: time-aware batching, scale cheating prevention, curriculum learning
  - Includes all 5 critical fixes from detailed analysis
  - Run with: `python training/improved_baseline_trainer.py`

### Base Architecture
- **`hybrid_spike_moe_trainer.py`** - Base MoE trainer class
  - Mixture of Experts architecture with spike detection
  - Probabilistic predictions with uncertainty
  - Required by improved_baseline_trainer.py

### Testing & Validation
- **`test_spike_detection.py`** - Comprehensive spike detection testing
  - Tests normal, low, and high production detection
  - Validates uncertainty calibration
  - Run after training completes

- **`quick_spike_validation.py`** - Pre-training configuration validation
  - Validates spike thresholds and expert ratios
  - Run before training to verify setup

- **`simple_spike_test.py`** - Quick results viewer
  - Displays training results from metadata
  - No model loading required
  - Run with: `python training/simple_spike_test.py`

## Usage

### Training
```bash
python training/improved_baseline_trainer.py
```

### Validation
```bash
# Before training
python training/quick_spike_validation.py

# After training
python training/simple_spike_test.py
python training/test_spike_detection.py
```

## Archived Files

Old trainers and experimental files have been moved to `archive/old_trainers/`.
