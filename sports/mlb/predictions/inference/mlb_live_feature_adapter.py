#!/usr/bin/env python3
"""
MLB Live Feature Adapter

Builds feature vectors for MLB player prop inference.
Uses rolling empirical/stat distributions.

For pitcher K: recent Ks, innings, pitch count, opponent K rate, handedness
For batter H/TB/RBI/R: recent outcomes, batting order, opposing pitcher, park, team context
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
DATA_PROC_MLB_DIR = WORKSPACE / "Player-Predictor" / "Data-Proc-MLB"


def _read_player_history(player_name: str, market: str) -> pd.DataFrame:
    """Read processed MLB player history from Data-Proc-MLB."""
    player_dir = DATA_PROC_MLB_DIR / player_name.replace(" ", "_")
    if not player_dir.exists():
        return pd.DataFrame()

    files = sorted(player_dir.glob("*processed*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files[-3:]:  # Last 3 season files
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Market → column mapping in processed data
MARKET_COLUMN_MAP = {
    "K": "SO",  # Pitcher strikeouts
    "H": "H",
    "TB": "TB",
    "RBI": "RBI",
    "R": "R",
    "HR": "HR",
    "SO": "SO",  # Batter strikeouts
    "ER": "ER",
    "HA": "H",  # Hits allowed (pitcher)
    "BB": "BB",
    "OUTS": "IP",  # Approximate via innings pitched
}


class MlbLiveFeatureAdapter:
    """Builds feature vectors for MLB prop inference."""

    def __init__(self):
        self.history_cache: Dict[str, pd.DataFrame] = {}

    def build_features(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Build feature dict for a single prop row.

        Returns features + completeness metadata.
        """
        player = str(row.get("player", ""))
        market = str(row.get("market_canonical", row.get("market", "")))
        line = float(row.get("line", 0))
        team = str(row.get("team", ""))
        opponent = str(row.get("opponent", ""))

        features: Dict[str, Any] = {
            "player": player,
            "market": market,
            "line": line,
            "team": team,
            "opponent": opponent,
        }
        missing_features: List[str] = []

        # Load player history
        history = self._get_history(player)
        col = MARKET_COLUMN_MAP.get(market, market)

        if history.empty or col not in history.columns:
            missing_features.append("player_history")
            features["rolling_mean_3"] = np.nan
            features["rolling_mean_7"] = np.nan
            features["rolling_mean_15"] = np.nan
            features["rolling_std_15"] = np.nan
            features["sample_size"] = 0
        else:
            values = pd.to_numeric(history[col], errors="coerce").dropna()
            n = len(values)
            features["sample_size"] = n
            features["rolling_mean_3"] = float(values.tail(3).mean()) if n >= 3 else np.nan
            features["rolling_mean_7"] = float(values.tail(7).mean()) if n >= 7 else np.nan
            features["rolling_mean_15"] = float(values.tail(15).mean()) if n >= 15 else np.nan
            features["rolling_std_15"] = float(values.tail(15).std()) if n >= 15 else np.nan

            if n < 3:
                missing_features.append("insufficient_history")

        # Pitcher-specific features
        if market == "K":
            if not history.empty and "IP" in history.columns:
                ip = pd.to_numeric(history["IP"], errors="coerce").dropna()
                features["avg_innings_pitched"] = float(ip.tail(7).mean()) if len(ip) >= 7 else np.nan
            else:
                features["avg_innings_pitched"] = np.nan
                missing_features.append("innings_pitched")

            # Opponent K rate would require opponent data
            features["opponent_k_rate"] = np.nan
            missing_features.append("opponent_k_rate")

        # Batting order (not always available)
        features["batting_order"] = row.get("batting_order", np.nan)
        if pd.isna(features["batting_order"]):
            missing_features.append("batting_order")

        # Opposing pitcher
        features["opposing_pitcher"] = row.get("opposing_pitcher", "")
        if not features["opposing_pitcher"]:
            missing_features.append("opposing_pitcher")

        # Park factor
        features["park_factor"] = row.get("park_factor", np.nan)
        if pd.isna(features["park_factor"]):
            missing_features.append("park_factor")

        # Handedness
        features["handedness"] = row.get("handedness", "")
        features["opponent_hand"] = row.get("opponent_hand", "")

        # Rest days
        if not history.empty and "Date" in history.columns:
            dates = pd.to_datetime(history["Date"], errors="coerce").dropna()
            if len(dates) >= 2:
                last_game = dates.max()
                features["rest_days"] = (pd.Timestamp.now() - last_game).days
            else:
                features["rest_days"] = np.nan
                missing_features.append("rest_days")
        else:
            features["rest_days"] = np.nan
            missing_features.append("rest_days")

        # Completeness score
        total_features = 10
        present = total_features - len(missing_features)
        features["feature_completeness_score"] = present / total_features
        features["missing_feature_list"] = missing_features

        return features

    def _get_history(self, player: str) -> pd.DataFrame:
        if player in self.history_cache:
            return self.history_cache[player]
        df = _read_player_history(player, "")
        self.history_cache[player] = df
        return df
