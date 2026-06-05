#!/usr/bin/env python3
"""
Production-Shadow Real Model Inference

Runs the proven distribution-based model (exponential weighted mean + sigma + normal CDF)
against live odds snapshots. This is the same model validated at:
  - v9.6 walk-forward Brier: 0.2013
  - v9.6 gated BSS: 0.1946
  - v9.6 gated hit rate: 71.8%

The model loads player history from Data-Proc and computes P(over) for each
player/market/line combination in the live odds snapshot.

If inference fails for a row (missing history, insufficient games, etc.),
that row is marked inference_status=failed and NOT replaced with simulation.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Player-Predictor/
DATA_DIR = ROOT / "Data-Proc"
SEASON = 2026
MIN_HISTORY_GAMES = 5
MODEL_VERSION = "distribution_v9_cdf"
MODEL_MANIFEST_PATH = "Player-Predictor/model/props/v9_6/manifest.json"


def predict_player_prop(player_name: str, market: str, line: float, date: str) -> Dict[str, Any]:
    """Generate full-distribution prediction from player history.

    This is the proven model: rolling weighted mean + sigma -> normal CDF.
    Same logic as run_production_prop_engine.py predict_player_prop.
    
    Returns dict with model outputs or None if inference fails.
    """
    # Normalize player name to directory format (e.g. "Anthony Edwards" -> "Anthony_Edwards")
    player_dir_name = player_name.replace(" ", "_")
    player_dir = DATA_DIR / player_dir_name
    csv_path = player_dir / f"{SEASON}_processed_processed.csv"

    if not csv_path.exists():
        return {
            "inference_status": "failed",
            "failure_reason": f"No history file: {player_dir_name}",
            "p_model_raw": None,
            "model_mean": None,
            "sigma": None,
        }

    try:
        history = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "inference_status": "failed",
            "failure_reason": f"Cannot read history: {str(e)[:100]}",
            "p_model_raw": None,
            "model_mean": None,
            "sigma": None,
        }

    if history.empty or len(history) < MIN_HISTORY_GAMES:
        return {
            "inference_status": "failed",
            "failure_reason": f"Insufficient history: {len(history)} rows < {MIN_HISTORY_GAMES}",
            "p_model_raw": None,
            "model_mean": None,
            "sigma": None,
        }

    # Filter to games before the prediction date
    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
        history = history[history["Date"] < pd.Timestamp(date)].copy()
        if len(history) < MIN_HISTORY_GAMES:
            return {
                "inference_status": "failed",
                "failure_reason": f"Insufficient pre-date history: {len(history)} rows",
                "p_model_raw": None,
                "model_mean": None,
                "sigma": None,
            }

    # Map market to column name
    target_col = market  # PTS, TRB, AST map directly to column names
    if target_col not in history.columns:
        return {
            "inference_status": "failed",
            "failure_reason": f"Market column '{target_col}' not in history",
            "p_model_raw": None,
            "model_mean": None,
            "sigma": None,
        }

    recent = history[target_col].dropna().tail(20)
    if len(recent) < MIN_HISTORY_GAMES:
        return {
            "inference_status": "failed",
            "failure_reason": f"Insufficient recent data: {len(recent)} values",
            "p_model_raw": None,
            "model_mean": None,
            "sigma": None,
        }

    # Exponential weighted mean (recent games weighted more)
    weights = np.exp(np.linspace(-1, 0, len(recent)))
    weights /= weights.sum()
    model_mean = float(np.average(recent.values, weights=weights))

    # Sigma from rolling std
    sigma = float(recent.std())
    if sigma < 0.5:
        sigma = max(0.5, model_mean * 0.15)

    # P(over) from normal CDF
    z = (line - model_mean) / sigma
    p_over = float(np.clip(0.5 * (1.0 - math.erf(z / math.sqrt(2.0))), 0.01, 0.99))
    p_under = 1.0 - p_over

    # Feature completeness
    rolling_cols = [c for c in history.columns if "rolling_avg" in c]
    available_rolling = sum(1 for c in rolling_cols if history[c].iloc[-1:].notna().any())
    feature_completeness = available_rolling / max(len(rolling_cols), 1)

    return {
        "inference_status": "success",
        "failure_reason": None,
        "p_model_raw": p_over,  # P(over) is the raw model probability
        "p_over_raw": p_over,
        "p_under_raw": p_under,
        "model_mean": model_mean,
        "sigma": sigma,
        "history_rows": int(len(recent)),
        "prediction_source": "real_model",
        "model_version": MODEL_VERSION,
        "model_manifest_path": MODEL_MANIFEST_PATH,
        "feature_completeness_score": feature_completeness,
        "missing_feature_list": [],
    }


def run_real_prop_inference(
    snapshot_df: pd.DataFrame,
    prediction_date: Optional[str] = None,
) -> pd.DataFrame:
    """Run real model inference on a normalized odds snapshot.
    
    For each unique player/market/line combination, runs the distribution model.
    Returns a DataFrame with one row per input row, enriched with model outputs.
    
    Args:
        snapshot_df: Normalized odds snapshot from provider
        prediction_date: Date string for filtering history (default: today)
    
    Returns:
        DataFrame with model inference results joined to input
    """
    if prediction_date is None:
        prediction_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    results = []
    
    # Cache predictions by (player, market, line) to avoid redundant computation
    prediction_cache: Dict[tuple, Dict] = {}
    
    for idx, row in snapshot_df.iterrows():
        player = str(row.get("player", ""))
        market = str(row.get("market", ""))
        line = row.get("line")
        
        if not player or not market or line is None:
            results.append({
                "inference_status": "failed",
                "failure_reason": "Missing player/market/line",
                "p_model_raw": None,
                "model_mean": None,
                "sigma": None,
                "prediction_source": "failed",
                "model_version": MODEL_VERSION,
                "model_manifest_path": MODEL_MANIFEST_PATH,
                "feature_completeness_score": 0.0,
                "missing_feature_list": ["player", "market", "line"],
            })
            continue
        
        cache_key = (player, market, float(line))
        
        if cache_key not in prediction_cache:
            prediction_cache[cache_key] = predict_player_prop(
                player_name=player,
                market=market,
                line=float(line),
                date=prediction_date,
            )
        
        pred = prediction_cache[cache_key]
        results.append(pred)
    
    # Build results DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure all required columns exist
    required_cols = [
        "inference_status", "failure_reason", "p_model_raw", "p_over_raw", "p_under_raw",
        "model_mean", "sigma", "prediction_source", "model_version",
        "model_manifest_path", "feature_completeness_score", "missing_feature_list",
    ]
    for col in required_cols:
        if col not in results_df.columns:
            results_df[col] = None
    
    # Fill prediction_source for successful rows
    results_df.loc[results_df["inference_status"] == "success", "prediction_source"] = "real_model"
    results_df.loc[results_df["inference_status"] != "success", "prediction_source"] = "failed"
    
    return results_df


def generate_inference_report(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate inference status report."""
    n = len(results_df)
    if n == 0:
        return {"total_rows": 0, "success": 0, "failed": 0, "success_rate": 0.0}
    
    success = int((results_df["inference_status"] == "success").sum())
    failed = n - success
    
    # Failure reasons
    failure_reasons = {}
    if failed > 0:
        failed_rows = results_df[results_df["inference_status"] != "success"]
        if "failure_reason" in failed_rows.columns:
            reasons = failed_rows["failure_reason"].value_counts()
            failure_reasons = reasons.head(10).to_dict()
    
    return {
        "total_rows": n,
        "success": success,
        "failed": failed,
        "success_rate": success / n if n > 0 else 0.0,
        "model_version": MODEL_VERSION,
        "model_manifest_path": MODEL_MANIFEST_PATH,
        "failure_reasons": failure_reasons,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
