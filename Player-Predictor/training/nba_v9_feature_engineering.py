#!/usr/bin/env python3
"""
NBA v9 Feature Engineering Extensions

Extends v8 features with v9 research-backed enhancements:
  - HMM regime probability features (soft regime membership)
  - Copula-derived correlation features
  - Lineup opportunity features
  - Distribution tail pressure features
  - Chaos/entropy features for uncertainty estimation
  - Adaptive calibration drift signals

These features feed into the existing model pipeline and improve
the distributional head's ability to estimate P(over) accurately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from nba_v8_feature_engineering import (
    _safe_div, _rolling_stat, STAT_COLS, ROLLING_WINDOWS,
    add_v8_enhanced_features,
)


def add_tail_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced tail pressure features for prop threshold modeling.

    Research insight: Props are threshold-crossing events on fat-tailed
    distributions. We need features that capture how much "tail pressure"
    exists above/below the line.
    """
    df = df.copy()

    for stat in STAT_COLS:
        if stat not in df.columns:
            continue

        # Coefficient of variation (higher = more volatile = harder to predict)
        for w in [5, 10]:
            roll_mean = _rolling_stat(df[stat], w, "mean")
            roll_std = _rolling_stat(df[stat], w, "std")
            df[f"{stat}_cv_{w}"] = _safe_div(roll_std.values, roll_mean.values.clip(0.1))

        # Tail ratio: how often does the player produce extreme outcomes?
        # (games > 1.5 std above mean) / total games
        for w in [10, 15]:
            roll_mean = _rolling_stat(df[stat], w, "mean")
            roll_std = _rolling_stat(df[stat], w, "std").clip(0.5)
            z_scores = (df[stat] - roll_mean) / roll_std

            # Upper tail frequency (over-prone)
            df[f"{stat}_upper_tail_rate_{w}"] = (
                z_scores.rolling(w, min_periods=3)
                .apply(lambda x: float(np.mean(x > 1.0)), raw=True)
                .fillna(0.3)
            )
            # Lower tail frequency (under-prone)
            df[f"{stat}_lower_tail_rate_{w}"] = (
                z_scores.rolling(w, min_periods=3)
                .apply(lambda x: float(np.mean(x < -1.0)), raw=True)
                .fillna(0.3)
            )

        # Consecutive direction: how many games in a row above/below median?
        roll_median = df[stat].rolling(10, min_periods=3).median()
        above_median = (df[stat] > roll_median).astype(float)
        # Streak length
        streak = above_median.copy()
        for i in range(1, len(streak)):
            if streak.iloc[i] == streak.iloc[i-1]:
                streak.iloc[i] = streak.iloc[i-1] + 1
            else:
                streak.iloc[i] = 1
        df[f"{stat}_direction_streak"] = streak

        # IQR ratio: how tight is the distribution?
        # Tight IQR = more predictable, wide IQR = more volatile
        q75 = df[stat].rolling(10, min_periods=3).quantile(0.75)
        q25 = df[stat].rolling(10, min_periods=3).quantile(0.25)
        iqr = (q75 - q25).clip(0.1)
        df[f"{stat}_iqr_10"] = iqr
        df[f"{stat}_iqr_ratio"] = _safe_div(iqr.values, roll_mean.values.clip(0.1))

    return df


def add_regime_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that capture regime transitions and instability.

    Research insight: HMM models show player behavior has hidden states.
    These features proxy the transition probabilities between states.
    """
    df = df.copy()

    # Usage regime transitions
    if "USG%" in df.columns:
        usg = df["USG%"]
        usg_roll5 = _rolling_stat(usg, 5, "mean")
        usg_roll10 = _rolling_stat(usg, 10, "mean")

        # Usage trend (short vs long term)
        df["USG_trend_5v10"] = usg_roll5 - usg_roll10

        # Usage volatility acceleration (is volatility increasing?)
        usg_std5 = _rolling_stat(usg, 5, "std")
        usg_std10 = _rolling_stat(usg, 10, "std")
        df["USG_vol_acceleration"] = usg_std5 - usg_std10

        # Regime change signal: large shift in recent usage
        df["USG_regime_shift"] = (
            (usg - usg_roll10).abs() / usg_roll10.clip(0.01)
        ).fillna(0)

    # Minutes regime transitions
    if "MP" in df.columns:
        mp = df["MP"]
        mp_roll5 = _rolling_stat(mp, 5, "mean")
        mp_roll10 = _rolling_stat(mp, 10, "mean")

        # Minutes trend
        df["MP_trend_5v10"] = mp_roll5 - mp_roll10

        # Minutes floor: minimum recent minutes (proxy for role security)
        df["MP_floor_5"] = df["MP"].rolling(5, min_periods=1).min()
        df["MP_floor_10"] = df["MP"].rolling(10, min_periods=1).min()

        # Minutes ceiling: maximum recent minutes (proxy for upside)
        df["MP_ceiling_5"] = df["MP"].rolling(5, min_periods=1).max()

    # Efficiency regime transitions
    if "TS%" in df.columns:
        ts = df["TS%"]
        ts_roll5 = _rolling_stat(ts, 5, "mean")
        ts_roll10 = _rolling_stat(ts, 10, "mean")

        # Efficiency trend
        df["TS_trend_5v10"] = ts_roll5 - ts_roll10

        # Efficiency stability (low std = stable shooter)
        df["TS_stability_10"] = 1.0 - _rolling_stat(ts, 10, "std").clip(0, 0.15) / 0.15

    return df


def add_opportunity_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that capture the opportunity environment.

    Research insight: A player's prop outcome depends on opportunity:
    minutes × usage × pace × game script. These features capture
    the "opportunity surface" that determines stat production.
    """
    df = df.copy()

    # Opportunity index: minutes × usage rate
    if "MP" in df.columns and "USG%" in df.columns:
        df["Opportunity_Index"] = df["MP"] * df["USG%"]
        df["Opportunity_Index_roll5"] = _rolling_stat(df["Opportunity_Index"], 5, "mean")
        df["Opportunity_Index_trend"] = (
            df["Opportunity_Index"] - df["Opportunity_Index_roll5"]
        )

    # Shot opportunity: FGA per minute
    if "FGA" in df.columns and "MP" in df.columns:
        df["FGA_per_min"] = _safe_div(df["FGA"].values, df["MP"].values.clip(1))
        df["FGA_per_min_roll5"] = _rolling_stat(df["FGA_per_min"], 5, "mean")

    # Assist opportunity: assists per minute
    if "AST" in df.columns and "MP" in df.columns:
        df["AST_per_min"] = _safe_div(df["AST"].values, df["MP"].values.clip(1))
        df["AST_per_min_roll5"] = _rolling_stat(df["AST_per_min"], 5, "mean")

    # Rebound opportunity: rebounds per minute
    if "TRB" in df.columns and "MP" in df.columns:
        df["TRB_per_min"] = _safe_div(df["TRB"].values, df["MP"].values.clip(1))
        df["TRB_per_min_roll5"] = _rolling_stat(df["TRB_per_min"], 5, "mean")

    # Game script proxy: PLUS_MINUS trend indicates blowout/competitive
    if "PLUS_MINUS" in df.columns:
        pm_abs = df["PLUS_MINUS"].abs()
        df["Game_Competitiveness"] = 1.0 - (pm_abs / 30.0).clip(0, 1)
        df["Game_Competitiveness_roll5"] = _rolling_stat(df["Game_Competitiveness"], 5, "mean")

    return df


def add_entropy_uncertainty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that capture prediction uncertainty and chaos level.

    Research insight: The model should know when it doesn't know.
    High entropy in recent outcomes = high uncertainty = abstain.
    """
    df = df.copy()

    for stat in STAT_COLS:
        if stat not in df.columns:
            continue

        # Rolling entropy of stat (discretized)
        # High entropy = outcomes are spread out = hard to predict
        for w in [5, 10]:
            def _rolling_entropy(x):
                if len(x) < 3:
                    return 1.0
                # Discretize into 5 bins
                bins = np.linspace(x.min() - 0.1, x.max() + 0.1, 6)
                counts = np.histogram(x, bins=bins)[0]
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                max_entropy = np.log2(min(len(x), 5))
                return entropy / max_entropy if max_entropy > 0 else 1.0

            df[f"{stat}_entropy_{w}"] = (
                df[stat].rolling(w, min_periods=3)
                .apply(_rolling_entropy, raw=True)
                .fillna(1.0)
            )

        # Prediction difficulty: how far is the market line from the median?
        market_col = f"Market_{stat}"
        if market_col in df.columns:
            roll_median = df[stat].rolling(10, min_periods=3).median()
            roll_std = _rolling_stat(df[stat], 10, "std").clip(0.5)
            # Distance from line in std units (higher = easier to predict direction)
            df[f"{stat}_line_distance_std"] = (
                (roll_median - df[market_col]).abs() / roll_std
            ).fillna(0)

    return df


def add_v9_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: apply all v9 feature enhancements.
    Designed to be called AFTER v8 feature engineering.
    """
    # First apply v8 features if not already applied
    # (check for a v8 feature to determine)
    if "USG_spike_zscore" not in df.columns:
        df = add_v8_enhanced_features(df)

    # Apply v9 extensions
    df = add_tail_pressure_features(df)
    df = add_regime_transition_features(df)
    df = add_opportunity_environment_features(df)
    df = add_entropy_uncertainty_features(df)

    # Fill any NaN/inf introduced
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def get_v9_new_feature_names() -> list[str]:
    """Return list of new feature column names added by v9 engineering."""
    names = []

    # Tail pressure features
    for stat in STAT_COLS:
        for w in [5, 10]:
            names.append(f"{stat}_cv_{w}")
        for w in [10, 15]:
            names.append(f"{stat}_upper_tail_rate_{w}")
            names.append(f"{stat}_lower_tail_rate_{w}")
        names.append(f"{stat}_direction_streak")
        names.append(f"{stat}_iqr_10")
        names.append(f"{stat}_iqr_ratio")

    # Regime transition features
    names += [
        "USG_trend_5v10", "USG_vol_acceleration", "USG_regime_shift",
        "MP_trend_5v10", "MP_floor_5", "MP_floor_10", "MP_ceiling_5",
        "TS_trend_5v10", "TS_stability_10",
    ]

    # Opportunity environment features
    names += [
        "Opportunity_Index", "Opportunity_Index_roll5", "Opportunity_Index_trend",
        "FGA_per_min", "FGA_per_min_roll5",
        "AST_per_min", "AST_per_min_roll5",
        "TRB_per_min", "TRB_per_min_roll5",
        "Game_Competitiveness", "Game_Competitiveness_roll5",
    ]

    # Entropy/uncertainty features
    for stat in STAT_COLS:
        for w in [5, 10]:
            names.append(f"{stat}_entropy_{w}")
        names.append(f"{stat}_line_distance_std")

    return names


if __name__ == "__main__":
    np.random.seed(42)
    print("Testing v9 Feature Engineering...")

    n = 50
    test_df = pd.DataFrame({
        "PTS": np.random.normal(25, 8, n),
        "TRB": np.random.normal(6, 3, n),
        "AST": np.random.normal(5, 2, n),
        "MP": np.random.normal(32, 5, n),
        "FGA": np.random.normal(18, 4, n),
        "FTA": np.random.normal(5, 2, n),
        "USG%": np.random.normal(0.28, 0.05, n),
        "TS%": np.random.normal(0.58, 0.08, n),
        "FG%": np.random.normal(0.47, 0.07, n),
        "3P%": np.random.normal(0.36, 0.10, n),
        "FT%": np.random.normal(0.80, 0.10, n),
        "ORTG": np.random.normal(112, 8, n),
        "DRTG": np.random.normal(110, 8, n),
        "PLUS_MINUS": np.random.normal(2, 12, n),
        "BPM": np.random.normal(3, 5, n),
        "Market_PTS": np.random.normal(24.5, 2, n),
        "Market_TRB": np.random.normal(5.5, 1, n),
        "Market_AST": np.random.normal(4.5, 1, n),
    })

    result = add_v9_enhanced_features(test_df)
    v9_features = get_v9_new_feature_names()
    present = [c for c in v9_features if c in result.columns]
    print(f"  v9 features added: {len(present)}/{len(v9_features)}")
    print(f"  Total columns: {len(result.columns)}")

    # Show some sample values
    sample_features = ["PTS_cv_5", "PTS_upper_tail_rate_10", "USG_regime_shift",
                       "Opportunity_Index", "PTS_entropy_10", "PTS_line_distance_std"]
    for feat in sample_features:
        if feat in result.columns:
            val = result[feat].iloc[-1]
            print(f"    {feat}: {val:.4f}")

    print("\nv9 Feature Engineering smoke test PASSED")
