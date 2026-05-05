"""Streak regression detector.

Identifies players whose market lines have been inflated by recent hot
streaks or deflated by cold streaks.  Markets overreact to short-term
performance — regression to the mean creates systematic value:

  - Player on a hot streak → line gets bumped up → UNDER value
  - Player on a cold streak → line gets dropped → OVER value

This module computes a "streak signal" by comparing the player's recent
performance (last 3-5 games) to their season baseline.  When the gap is
large, the market is likely overreacting and regression is imminent.

Academic support:
  - Harvard (2025): "markets efficiently price streak-related information"
    in closing odds, but OPENING odds overreact to momentum
  - arxiv 1810.03383: "Illusion of persistence in NBA 1995-2018" — hot hand
    effect is largely illusory at the individual game level
  - The key: we bet AGAINST the streak, not with it
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class StreakRegressionConfig:
    """Configuration for streak regression detection."""
    enabled: bool = True
    min_history_rows: int = 15
    recent_window: int = 4          # last N games to measure "streak"
    baseline_window: int = 20       # games for baseline average
    hot_streak_threshold: float = 1.3   # recent/baseline ratio for "hot"
    cold_streak_threshold: float = 0.7  # recent/baseline ratio for "cold"
    regression_boost: float = 0.020     # win rate boost for regression picks
    anti_streak_penalty: float = 0.015  # penalty for betting WITH the streak


def _sf(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def detect_streak(
    player_history: pd.DataFrame,
    target: str,
    *,
    config: StreakRegressionConfig | None = None,
) -> dict[str, Any]:
    """Detect if a player is on a hot or cold streak for a given target.

    Parameters
    ----------
    player_history : pd.DataFrame
        Player's game log with Date and target columns, sorted chronologically.
    target : str
        The stat column (PTS, TRB, AST, H, TB, etc.)

    Returns
    -------
    Dict with streak_type (hot/cold/neutral), streak_ratio, regression_direction.
    """
    cfg = config or StreakRegressionConfig()

    if not cfg.enabled or player_history.empty:
        return {"streak_type": "neutral", "streak_ratio": 1.0, "regression_direction": ""}

    if target not in player_history.columns:
        return {"streak_type": "neutral", "streak_ratio": 1.0, "regression_direction": ""}

    values = pd.to_numeric(player_history[target], errors="coerce").dropna()
    if len(values) < cfg.min_history_rows:
        return {"streak_type": "neutral", "streak_ratio": 1.0, "regression_direction": ""}

    recent = values.tail(cfg.recent_window)
    baseline = values.tail(cfg.baseline_window)

    recent_mean = float(recent.mean())
    baseline_mean = float(baseline.mean())

    if baseline_mean <= 0:
        return {"streak_type": "neutral", "streak_ratio": 1.0, "regression_direction": ""}

    ratio = recent_mean / baseline_mean

    if ratio >= cfg.hot_streak_threshold:
        return {
            "streak_type": "hot",
            "streak_ratio": ratio,
            "regression_direction": "UNDER",  # expect regression down
            "recent_mean": recent_mean,
            "baseline_mean": baseline_mean,
        }
    elif ratio <= cfg.cold_streak_threshold:
        return {
            "streak_type": "cold",
            "streak_ratio": ratio,
            "regression_direction": "OVER",  # expect regression up
            "recent_mean": recent_mean,
            "baseline_mean": baseline_mean,
        }
    else:
        return {
            "streak_type": "neutral",
            "streak_ratio": ratio,
            "regression_direction": "",
            "recent_mean": recent_mean,
            "baseline_mean": baseline_mean,
        }


def compute_streak_adjustment(
    *,
    direction: str,
    streak_type: str,
    regression_direction: str,
    streak_ratio: float,
    config: StreakRegressionConfig | None = None,
) -> float:
    """Compute win rate adjustment based on streak regression.

    If our bet direction ALIGNS with regression (e.g., betting UNDER on a
    hot-streak player), we get a boost.  If it goes AGAINST regression
    (e.g., betting OVER on a hot-streak player), we get a penalty.
    """
    cfg = config or StreakRegressionConfig()

    if not cfg.enabled or streak_type == "neutral" or not regression_direction:
        return 0.0

    dir_upper = str(direction).upper().strip()
    reg_dir = str(regression_direction).upper().strip()

    # How extreme is the streak? Scale the adjustment
    if streak_type == "hot":
        intensity = float(np.clip((streak_ratio - cfg.hot_streak_threshold) / 0.5, 0.0, 1.0))
    else:
        intensity = float(np.clip((cfg.cold_streak_threshold - streak_ratio) / 0.3, 0.0, 1.0))

    if dir_upper == reg_dir:
        # Betting WITH regression — boost
        return cfg.regression_boost * (0.5 + 0.5 * intensity)
    else:
        # Betting AGAINST regression — penalty
        return -cfg.anti_streak_penalty * (0.5 + 0.5 * intensity)


def annotate_streak_regression(
    candidates: pd.DataFrame,
    *,
    data_proc_root: Path | str | None = None,
    config: StreakRegressionConfig | None = None,
) -> pd.DataFrame:
    """Annotate candidates with streak regression signals.

    Looks up each player's recent game history from Data-Proc and computes
    streak detection.  Adds columns:
      - streak_type: hot/cold/neutral
      - streak_ratio: recent/baseline performance ratio
      - streak_regression_adj: win rate adjustment
    """
    cfg = config or StreakRegressionConfig()
    out = candidates.copy()

    if out.empty or not cfg.enabled:
        out["streak_type"] = "neutral"
        out["streak_ratio"] = 1.0
        out["streak_regression_adj"] = 0.0
        return out

    # Resolve data proc root
    if data_proc_root is None:
        data_proc_root = Path(__file__).resolve().parents[1] / "Data-Proc"
    else:
        data_proc_root = Path(data_proc_root)

    streak_types = []
    streak_ratios = []
    streak_adjs = []

    for _, row in out.iterrows():
        # Find player's CSV
        csv_path = str(row.get("csv", ""))
        target = str(row.get("target", "")).upper()
        direction = str(row.get("direction", ""))

        player_df = pd.DataFrame()
        if csv_path and Path(csv_path).exists():
            try:
                player_df = pd.read_csv(csv_path)
            except Exception:
                pass
        elif data_proc_root.exists():
            # Try to find by player name
            player_name = str(row.get("player", "")).replace(" ", "_")
            player_dir = data_proc_root / player_name
            csv_candidate = player_dir / "2026_processed_processed.csv"
            if csv_candidate.exists():
                try:
                    player_df = pd.read_csv(csv_candidate)
                except Exception:
                    pass

        if player_df.empty or target not in player_df.columns:
            streak_types.append("neutral")
            streak_ratios.append(1.0)
            streak_adjs.append(0.0)
            continue

        # Filter to active games only
        if "Did_Not_Play" in player_df.columns:
            player_df = player_df[pd.to_numeric(player_df["Did_Not_Play"], errors="coerce").fillna(0) < 0.5]

        streak_info = detect_streak(player_df, target, config=cfg)
        adj = compute_streak_adjustment(
            direction=direction,
            streak_type=streak_info["streak_type"],
            regression_direction=streak_info.get("regression_direction", ""),
            streak_ratio=streak_info["streak_ratio"],
            config=cfg,
        )

        streak_types.append(streak_info["streak_type"])
        streak_ratios.append(streak_info["streak_ratio"])
        streak_adjs.append(adj)

    out["streak_type"] = streak_types
    out["streak_ratio"] = streak_ratios
    out["streak_regression_adj"] = streak_adjs
    return out
