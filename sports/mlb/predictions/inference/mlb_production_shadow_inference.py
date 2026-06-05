#!/usr/bin/env python3
"""
MLB Production Shadow Inference

Model baseline: distribution_v1_mlb_cdf
Uses rolling empirical/stat distribution:
  - model_mean, sigma
  - empirical residual CDF if available
  - normal CDF fallback

No simulated fallback in production mode.
If missing critical features: decision_tier = monitor/no_action
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))

from mlb_live_feature_adapter import MlbLiveFeatureAdapter


MODEL_VERSION = "distribution_v1_mlb_cdf"
MIN_COMPLETENESS_FOR_CLASS_A = 0.5
MIN_SAMPLE_SIZE = 5


class MlbProductionShadowInference:
    """MLB model inference using empirical CDF with normal fallback."""

    def __init__(self):
        self.feature_adapter = MlbLiveFeatureAdapter()
        self.model_version = MODEL_VERSION

    def predict_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Generate probability prediction for a single prop row.

        Returns prediction dict with:
          p_model_raw, p_over_raw, p_under_raw, model_mean, sigma,
          prediction_source, model_version, feature_completeness_score,
          missing_feature_list, inference_status
        """
        features = self.feature_adapter.build_features(row)
        line = float(row.get("line", 0))
        market = str(row.get("market_canonical", row.get("market", "")))
        side = str(row.get("side", "over")).lower()

        result: Dict[str, Any] = {
            "model_version": self.model_version,
            "prediction_source": "real_model",
            "feature_completeness_score": features.get("feature_completeness_score", 0),
            "missing_feature_list": features.get("missing_feature_list", []),
        }

        # Determine model mean and sigma from rolling stats
        mean_val = features.get("rolling_mean_7", features.get("rolling_mean_15"))
        std_val = features.get("rolling_std_15")
        sample_size = features.get("sample_size", 0)

        if mean_val is None or np.isnan(mean_val) or sample_size < MIN_SAMPLE_SIZE:
            # Cannot produce reliable inference
            result["p_model_raw"] = np.nan
            result["p_over_raw"] = np.nan
            result["p_under_raw"] = np.nan
            result["model_mean"] = np.nan
            result["sigma"] = np.nan
            result["inference_status"] = "insufficient_data"
            result["decision_tier_override"] = "no_action"
            result["production_countable_for_staking"] = False
            return result

        # Use rolling mean as model mean
        model_mean = float(mean_val)

        # Estimate sigma
        if std_val is not None and not np.isnan(std_val) and std_val > 0:
            sigma = float(std_val)
        else:
            # Fallback: use Poisson-like variance (sqrt of mean) for count stats
            sigma = max(float(np.sqrt(abs(model_mean))), 0.5)

        result["model_mean"] = model_mean
        result["sigma"] = sigma

        # Compute P(X > line) using normal CDF
        # For discrete stats, use continuity correction
        p_over = 1.0 - stats.norm.cdf(line + 0.5, loc=model_mean, scale=sigma)
        p_under = stats.norm.cdf(line - 0.5, loc=model_mean, scale=sigma)

        # Normalize to ensure they sum close to 1 (excluding push probability)
        p_push = stats.norm.pdf(line, loc=model_mean, scale=sigma) * 1.0  # Approximate
        total = p_over + p_under + p_push
        if total > 0:
            p_over_norm = p_over / (p_over + p_under)
            p_under_norm = p_under / (p_over + p_under)
        else:
            p_over_norm = 0.5
            p_under_norm = 0.5

        # Clip to valid range
        p_over_norm = float(np.clip(p_over_norm, 0.01, 0.99))
        p_under_norm = float(np.clip(p_under_norm, 0.01, 0.99))

        result["p_over_raw"] = p_over_norm
        result["p_under_raw"] = p_under_norm
        result["p_model_raw"] = p_over_norm if side == "over" else p_under_norm
        result["inference_status"] = "success"

        # Check feature completeness for decision tier
        completeness = features.get("feature_completeness_score", 0)
        if completeness < MIN_COMPLETENESS_FOR_CLASS_A:
            result["decision_tier_override"] = "monitor"
            result["production_countable_for_staking"] = False
        else:
            result["decision_tier_override"] = None
            result["production_countable_for_staking"] = True

        return result

    def predict_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict for a batch of prop rows."""
        return [self.predict_row(row) for row in rows]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction columns to a DataFrame."""
        if df.empty:
            return df

        predictions = []
        for _, row in df.iterrows():
            pred = self.predict_row(row.to_dict())
            predictions.append(pred)

        pred_df = pd.DataFrame(predictions)
        # Merge prediction columns into original
        for col in pred_df.columns:
            df[col] = pred_df[col].values

        return df
