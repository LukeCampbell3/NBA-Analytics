"""
Generator script for NBA v8 model files.
Run: python Player-Predictor/training/write_v8_files.py
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAINING = REPO / "training"
INFERENCE = REPO / "inference"
SCRIPTS = REPO / "scripts"

os.makedirs(str(TRAINING), exist_ok=True)
os.makedirs(str(INFERENCE), exist_ok=True)
os.makedirs(str(SCRIPTS), exist_ok=True)

# -----------------------------------------------------------------------------
# 1. nba_v8_feature_engineering.py
# -----------------------------------------------------------------------------
v8_features = r'''#!/usr/bin/env python3
"""
NBA v8 Feature Engineering

Extends v7 features with:
  - Teammate dependency scores (how much player relies on specific teammates)
  - Pace context features (team pace, opponent pace, pace differential)
  - Minutes volatility (rolling std of MP, blowout risk proxy)
  - Regime indicators (explicit hidden-state proxies)
  - Distribution shape features (skewness, kurtosis of recent stat windows)
  - Prop-specific features (distance from market line, line movement)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


ROLLING_WINDOWS = [3, 5, 10, 15]
STAT_COLS = ["PTS", "TRB", "AST"]
RATE_COLS = ["FG%", "3P%", "FT%", "TS%", "USG%"]
CONTEXT_COLS = ["MP", "FGA", "FTA", "PLUS_MINUS", "BPM", "ORTG", "DRTG"]


def _safe_div(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-8, a / b, fill)
    return result.astype(np.float32)


def _rolling_stat(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """Compute rolling statistic with min_periods=1."""
    roller = series.rolling(window=window, min_periods=1)
    if func == "mean":
        return roller.mean()
    elif func == "std":
        return roller.std().fillna(0.0)
    elif func == "skew":
        return roller.skew().fillna(0.0)
    elif func == "kurt":
        return roller.kurt().fillna(0.0)
    elif func == "min":
        return roller.min()
    elif func == "max":
        return roller.max()
    raise ValueError(f"Unknown func: {func}")


def add_distribution_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add skewness and kurtosis of recent stat windows - captures fat tails."""
    df = df.copy()
    for col in STAT_COLS:
        if col not in df.columns:
            continue
        for w in [5, 10]:
            df[f"{col}_skew_{w}"] = _rolling_stat(df[col], w, "skew")
            df[f"{col}_kurt_{w}"] = _rolling_stat(df[col], w, "kurt")
            # Tail pressure: how often recent games exceeded rolling mean
            roll_mean = _rolling_stat(df[col], w, "mean")
            df[f"{col}_above_mean_rate_{w}"] = (
                df[col].rolling(w, min_periods=1).apply(
                    lambda x: float(np.mean(x > x.mean())) if len(x) > 1 else 0.5,
                    raw=True,
                )
            )
    return df


def add_minutes_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes volatility is a key regime indicator - high std = role instability."""
    df = df.copy()
    if "MP" not in df.columns:
        return df

    for w in [3, 5, 10]:
        df[f"MP_std_{w}"] = _rolling_stat(df["MP"], w, "std")
        df[f"MP_min_{w}"] = _rolling_stat(df["MP"], w, "min")
        df[f"MP_max_{w}"] = _rolling_stat(df["MP"], w, "max")
        df[f"MP_range_{w}"] = df[f"MP_max_{w}"] - df[f"MP_min_{w}"]

    # Blowout risk proxy: low minutes in recent games
    df["MP_blowout_risk"] = (df["MP"].rolling(3, min_periods=1).min() < 20).astype(float)
    # Minutes trend: are minutes increasing or decreasing?
    df["MP_trend_5"] = _rolling_stat(df["MP"], 5, "mean") - _rolling_stat(df["MP"], 10, "mean")
    # Minutes consistency: coefficient of variation
    mp_mean_5 = _rolling_stat(df["MP"], 5, "mean").replace(0, np.nan)
    mp_std_5 = _rolling_stat(df["MP"], 5, "std")
    df["MP_cv_5"] = (mp_std_5 / mp_mean_5).fillna(0.0)

    return df


def add_pace_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pace context affects counting stats - faster pace = more opportunities."""
    df = df.copy()

    # If we have ORTG/DRTG, derive pace proxy
    if "ORTG" in df.columns and "DRTG" in df.columns:
        # Pace proxy: average of offensive and defensive rating (higher = faster game)
        df["Pace_Proxy"] = (df["ORTG"] + df["DRTG"]) / 2.0
        df["Pace_Proxy_roll5"] = _rolling_stat(df["Pace_Proxy"], 5, "mean")
        df["Pace_Proxy_trend"] = df["Pace_Proxy"] - df["Pace_Proxy_roll5"]

    # FGA as pace proxy (more shots = faster pace)
    if "FGA" in df.columns:
        df["FGA_roll5"] = _rolling_stat(df["FGA"], 5, "mean")
        df["FGA_pace_signal"] = df["FGA"] - df["FGA_roll5"]

    return df


def add_regime_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit regime indicators based on observable signals.
    These proxy the hidden states the model needs to detect:
      - Normal role
      - Usage spike (teammate out)
      - Blowout suppression
      - Foul trouble
      - Hot/cold shooting state
    """
    df = df.copy()

    # Usage spike: USG% significantly above rolling average
    if "USG%" in df.columns:
        usg_roll10 = _rolling_stat(df["USG%"], 10, "mean")
        usg_std10 = _rolling_stat(df["USG%"], 10, "std").replace(0, 1.0)
        df["USG_spike_zscore"] = (df["USG%"] - usg_roll10) / usg_std10
        df["USG_spike_flag"] = (df["USG_spike_zscore"] > 1.5).astype(float)

    # Blowout indicator: large PLUS_MINUS magnitude with low minutes
    if "PLUS_MINUS" in df.columns and "MP" in df.columns:
        df["Blowout_Signal"] = (
            (df["PLUS_MINUS"].abs() > 15) & (df["MP"] < 28)
        ).astype(float)
        df["Blowout_Signal_roll3"] = _rolling_stat(df["Blowout_Signal"], 3, "mean")

    # Foul trouble proxy: low minutes with high FTA (fouled out or limited)
    if "FTA" in df.columns and "MP" in df.columns:
        df["Foul_Trouble_Proxy"] = (
            (df["MP"] < 25) & (df["FTA"] > 4)
        ).astype(float)

    # Hot shooting state: TS% significantly above rolling average
    if "TS%" in df.columns:
        ts_roll10 = _rolling_stat(df["TS%"], 10, "mean")
        ts_std10 = _rolling_stat(df["TS%"], 10, "std").replace(0, 0.05)
        df["TS_zscore"] = (df["TS%"] - ts_roll10) / ts_std10
        df["Hot_Shooting_Flag"] = (df["TS_zscore"] > 1.5).astype(float)
        df["Cold_Shooting_Flag"] = (df["TS_zscore"] < -1.5).astype(float)

    # Role stability: low variance in USG% and MP over recent games
    if "USG%" in df.columns and "MP" in df.columns:
        usg_cv = _rolling_stat(df["USG%"], 5, "std") / (_rolling_stat(df["USG%"], 5, "mean").replace(0, 1.0))
        mp_cv = _rolling_stat(df["MP"], 5, "std") / (_rolling_stat(df["MP"], 5, "mean").replace(0, 1.0))
        df["Role_Stability_Score"] = 1.0 - ((usg_cv + mp_cv) / 2.0).clip(0, 1)

    return df


def add_prop_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prop-specific market features for edge detection.
    These are the key inputs for P(over line) estimation.
    """
    df = df.copy()

    for stat in STAT_COLS:
        market_col = f"Market_{stat}"
        if market_col not in df.columns:
            continue

        roll_mean = _rolling_stat(df[stat], 10, "mean")
        roll_std = _rolling_stat(df[stat], 10, "std").replace(0, 1.0)

        # Distance from market line in standard deviations
        df[f"{stat}_line_zscore"] = _safe_div(
            (df[market_col] - roll_mean).values,
            roll_std.values,
        )

        # Historical over rate at this line level
        # (how often has player exceeded this line in recent games)
        def _over_rate(series_and_line, window=10):
            """Rolling over rate - what fraction of recent games exceeded the line."""
            stat_vals = series_and_line[0]
            line_vals = series_and_line[1]
            result = np.zeros(len(stat_vals))
            for i in range(len(stat_vals)):
                start = max(0, i - window + 1)
                hist = stat_vals[start:i + 1]
                line = line_vals[i]
                if len(hist) > 0 and not np.isnan(line):
                    result[i] = float(np.mean(hist > line))
                else:
                    result[i] = 0.5
            return result

        stat_arr = df[stat].fillna(0).values
        line_arr = df[market_col].fillna(roll_mean).values
        df[f"{stat}_historical_over_rate_10"] = _over_rate((stat_arr, line_arr), window=10)
        df[f"{stat}_historical_over_rate_5"] = _over_rate((stat_arr, line_arr), window=5)

        # Line vs median (median is better than mean for prop betting)
        roll_median = df[stat].rolling(10, min_periods=1).median()
        df[f"{stat}_line_vs_median"] = df[market_col] - roll_median

        # Line movement proxy: change in market line from previous game
        df[f"{stat}_line_movement"] = df[market_col].diff().fillna(0.0)

    return df


def add_v8_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: apply all v8 feature enhancements.
    Designed to be called after v7 feature engineering.
    """
    df = add_distribution_shape_features(df)
    df = add_minutes_volatility_features(df)
    df = add_pace_context_features(df)
    df = add_regime_indicator_features(df)
    df = add_prop_market_features(df)

    # Fill any NaN/inf introduced
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def get_v8_new_feature_names() -> list[str]:
    """Return list of new feature column names added by v8 engineering."""
    names = []
    for col in STAT_COLS:
        for w in [5, 10]:
            names += [f"{col}_skew_{w}", f"{col}_kurt_{w}", f"{col}_above_mean_rate_{w}"]
    for w in [3, 5, 10]:
        names += [f"MP_std_{w}", f"MP_min_{w}", f"MP_max_{w}", f"MP_range_{w}"]
    names += ["MP_blowout_risk", "MP_trend_5", "MP_cv_5"]
    names += ["Pace_Proxy", "Pace_Proxy_roll5", "Pace_Proxy_trend", "FGA_roll5", "FGA_pace_signal"]
    names += ["USG_spike_zscore", "USG_spike_flag", "Blowout_Signal", "Blowout_Signal_roll3",
              "Foul_Trouble_Proxy", "TS_zscore", "Hot_Shooting_Flag", "Cold_Shooting_Flag",
              "Role_Stability_Score"]
    for stat in STAT_COLS:
        names += [
            f"{stat}_line_zscore",
            f"{stat}_historical_over_rate_10",
            f"{stat}_historical_over_rate_5",
            f"{stat}_line_vs_median",
            f"{stat}_line_movement",
        ]
    return names


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
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
    result = add_v8_enhanced_features(test_df)
    new_cols = get_v8_new_feature_names()
    present = [c for c in new_cols if c in result.columns]
    print(f"v8 feature engineering: {len(present)}/{len(new_cols)} new features added")
    print(f"Total columns: {len(result.columns)}")
    print("Smoke test PASSED")
'''

with open(str(TRAINING / "nba_v8_feature_engineering.py"), "w", encoding="utf-8") as f:
    f.write(v8_features)
print("wrote nba_v8_feature_engineering.py")

print("All v8 feature files written.")
