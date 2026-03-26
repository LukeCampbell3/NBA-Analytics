#!/usr/bin/env python3
"""
Meta-State MoE Trainer - Phase 6 (FINAL)
Complete implementation of "meta-state → routing → delta prediction (+ uncertainty)"

Key Changes from Original:
1. Meta-state head z (reason encoder)
2. Routing using z (not raw sequence_repr)
3. Event prediction (optional proxy events)
4. Event-gated spike experts (not threshold-based)
5. Separated epistemic vs aleatoric uncertainty
6. Updated loss functions (delta-focused, uncertainty separation)

This is Part 1 of 3 - contains imports, config, and helper classes
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import json
import warnings
import sys

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import base trainer for data preparation
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_spike_moe_trainer import HybridSpikeMoETrainer


class MetaStateConfig:
    """Configuration for meta-state MoE architecture"""
    
    # Meta-state architecture
    Z_DIM = 48  # Meta-state dimension (32-64 recommended)
    USE_META_STATE = True  # Enable meta-state routing
    
    # Event prediction
    NUM_EVENTS = 4  # minutes_spike, usage_spike, pace_tier, blowout_risk
    USE_EVENT_PREDICTION = True
    EVENT_LOSS_WEIGHT = 0.1
    
    # Spike expert gating
    USE_EVENT_GATING = True  # Event-gated spike experts (not threshold-based)
    GATE_PENALTY_WEIGHT = 0.05  # Prevent always-on gating
    GATE_MEAN_TARGET = 0.2  # Target 20% spike gate activation
    
    # Uncertainty separation
    USE_EPISTEMIC_HEAD = True  # Learned epistemic uncertainty
    USE_MC_DROPOUT = False  # Alternative: MC dropout for epistemic
    EPISTEMIC_WEIGHT = 0.01  # Weak supervision for epistemic
    MC_DROPOUT_SAMPLES = 10  # If using MC dropout
    
    # Loss weights (delta-focused)
    DELTA_HUBER_WEIGHT = 0.5  # Explicit delta learning
    NLL_WEIGHT = 1.0  # Probabilistic loss on reconstructed μ
    MEAN_LOSS_WEIGHT = 0.05  # Reduced (NLL carries most signal)
    
    # Expert regularization
    EXPERT_USAGE_MIN = 6  # Minimum active experts
    EXPERT_USAGE_MAX = 10  # Maximum active experts
    EXPERT_USAGE_WEIGHT = 0.1
    
    # Calibration (reduced to avoid scale cheating)
    SIGMA_REG_WEIGHT = 0.0001  # Very weak sigma regularization
    CALIBRATION_WEIGHT = 0.05  # Quantile coverage loss


# Continue in part 2...
