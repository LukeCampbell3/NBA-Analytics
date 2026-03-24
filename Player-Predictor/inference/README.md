# Inference Module

Comprehensive inference and evaluation tools for the Hybrid Spike-Driver MoE ensemble.

## Overview

The trained ensemble consists of 3 models with different strategies:
1. **confident_spike_aware**: High confidence with spike focus
2. **balanced_performance**: Balanced approach
3. **stable_intelligent_routing**: Stable routing with diversity

Each model outputs 15 dimensions:
- Normal means (3): PTS, TRB, AST predictions
- Normal scales (3): Uncertainty estimates
- Spike means (3): Alternative predictions for spike scenarios
- Spike scales (3): Spike uncertainty
- Spike indicators (3): Probability of spike for each stat

## Scripts

### 1. `ensemble_inference.py`
Main inference engine with comprehensive evaluation.

**Features:**
- Load trained ensemble models
- Single model predictions
- Ensemble predictions (mean, median, weighted)
- Uncertainty quantification (epistemic + aleatoric)
- Spike detection analysis
- Calibration analysis
- Performance metrics (MAE, R², correlation)

**Usage:**
```python
from inference.ensemble_inference import EnsembleInference

# Initialize
inference = EnsembleInference(model_dir="model")

# Evaluate on dataset
results = inference.evaluate_on_dataset(X_test, baselines_test, y_test)
inference.print_evaluation_summary(results)
```

**Run evaluation:**
```bash
python inference/ensemble_inference.py
```

### 2. `predict_game.py`
Predict a single game for a player.

**Features:**
- Load player's recent games
- Generate prediction with uncertainty
- Show individual model predictions
- Display spike probabilities

**Usage:**
```python
from inference.predict_game import predict_next_game, print_prediction

# Predict
prediction = predict_next_game('Stephen_Curry', recent_games_csv='path/to/games.csv')
print_prediction('Stephen_Curry', prediction)
```

**Run example:**
```bash
python inference/predict_game.py
```

### 3. `visualize_predictions.py`
Create comprehensive visualizations.

**Plots:**
1. **Prediction Scatter**: Predicted vs actual with uncertainty coloring
2. **Calibration**: Uncertainty calibration analysis
3. **Spike Detection ROC**: ROC curves for spike detection
4. **Residual Distribution**: Distribution of prediction errors
5. **Uncertainty Decomposition**: Epistemic vs aleatoric uncertainty

**Usage:**
```python
from inference.visualize_predictions import create_all_visualizations

# Create all plots
create_all_visualizations(output_dir="inference/plots")
```

**Run visualization:**
```bash
python inference/visualize_predictions.py
```

## Output Files

### Evaluation Results
`inference/evaluation_results.json` - Comprehensive metrics including:
- Individual model performance
- Ensemble performance
- Spike detection metrics
- Calibration analysis

### Visualizations
`inference/plots/` directory contains:
- `prediction_scatter.png` - Prediction quality
- `calibration.png` - Uncertainty calibration
- `spike_detection_roc.png` - Spike detection performance
- `residual_distribution.png` - Error distributions
- `uncertainty_decomposition.png` - Uncertainty breakdown

## Key Metrics

### Performance Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² (R-squared)**: Proportion of variance explained
- **Calibration Correlation**: How well uncertainty matches actual errors

### Spike Detection Metrics
- **Precision**: Of predicted spikes, how many were real?
- **Recall**: Of real spikes, how many were detected?
- **F1 Score**: Harmonic mean of precision and recall

### Uncertainty Metrics
- **Epistemic Uncertainty**: Model disagreement (reducible with more data)
- **Aleatoric Uncertainty**: Data noise (irreducible)
- **Total Uncertainty**: Combined uncertainty estimate

## Expected Performance

Based on the training configuration:

### Overall Performance
- **MAE**: ~3-5 points (PTS), ~1-2 rebounds (TRB), ~1-2 assists (AST)
- **R²**: 0.10-0.20 (challenging due to high variance in NBA games)
- **Calibration**: 0.15-0.30 correlation between uncertainty and errors

### Spike Detection
- **Precision**: 30-50% (conservative to avoid false alarms)
- **Recall**: 40-60% (catches most major spikes)
- **F1**: 35-55%

### Uncertainty
- **Epistemic**: 1-2 points (model disagreement)
- **Aleatoric**: 3-5 points (data noise)
- **Total**: 3-6 points (combined)

## Model Architecture

Each model in the ensemble:
- **Transformer backbone**: 4 layers, 8 heads, 256 dimensions
- **MoE layer**: 8 regular experts + 3 spike experts
- **Routing**: Temperature-scaled softmax with load balancing
- **Output**: ConditionalSpikeOutput layer (15 dimensions)

## Ensemble Strategies

### Strategy 1: confident_spike_aware
- Higher spike loss weights (1.4, 1.3, 1.2)
- Lower temperature (1.0) for sharper routing
- Higher confidence threshold (0.55)
- Strong variance encouragement (1.0)

### Strategy 2: balanced_performance
- Balanced spike weights (1.3, 1.25, 1.15)
- Moderate temperature (1.1)
- Moderate confidence (0.52)
- Balanced variance (0.8)

### Strategy 3: stable_intelligent_routing
- Moderate spike weights (1.5, 1.4, 1.3)
- Higher temperature (1.2) for diversity
- Lower confidence (0.50) for exploration
- Controlled variance (0.6)

## Tips for Best Results

1. **Use ensemble predictions**: Ensemble typically outperforms individual models
2. **Check uncertainty**: High uncertainty indicates low confidence
3. **Monitor spike indicators**: Values >0.5 suggest spike potential
4. **Consider epistemic uncertainty**: High model disagreement suggests edge cases
5. **Validate calibration**: Well-calibrated models have uncertainty ≈ actual error

## Troubleshooting

### Low R² scores
- Expected for NBA prediction (high inherent variance)
- Focus on MAE and calibration instead
- Check if predictions beat baseline

### Poor spike detection
- Adjust spike probability threshold (default 0.5)
- Check spike_indicators distribution
- Verify spike thresholds are appropriate

### High uncertainty
- Normal for difficult predictions
- Check epistemic vs aleatoric split
- High epistemic suggests need for more training data

### Model disagreement
- Indicates ensemble diversity (good!)
- Use weighted aggregation to favor confident models
- Check if disagreement correlates with difficulty

## Next Steps

1. **Run evaluation**: `python inference/ensemble_inference.py`
2. **Create visualizations**: `python inference/visualize_predictions.py`
3. **Test predictions**: `python inference/predict_game.py`
4. **Analyze results**: Review `evaluation_results.json`
5. **Iterate**: Adjust thresholds, aggregation methods, or retrain if needed
