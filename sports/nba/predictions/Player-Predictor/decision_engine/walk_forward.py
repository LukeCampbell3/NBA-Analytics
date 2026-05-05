"""Walk-forward validation framework.

Provides honest out-of-sample performance measurement by simulating
day-by-day prediction and grading.  For each day:
  1. Train/update models on all data BEFORE that day
  2. Generate predictions for that day
  3. Grade against actual outcomes
  4. Accumulate metrics

This prevents overfitting to the validation window and gives a realistic
estimate of live performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    """Results from a walk-forward backtest."""
    total_picks: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    pnl_units: float = 0.0
    roi_pct: float = 0.0
    days_tested: int = 0
    avg_picks_per_day: float = 0.0
    calibration_gap: float = 0.0
    brier_score: float = 0.0
    by_direction: dict[str, dict[str, float]] = field(default_factory=dict)
    by_target: dict[str, dict[str, float]] = field(default_factory=dict)
    daily_results: list[dict[str, Any]] = field(default_factory=list)


PAYOUT = 100.0 / 110.0  # -110 standard


def run_walk_forward(
    history_df: pd.DataFrame,
    *,
    min_train_days: int = 5,
    selection_fn=None,
    probability_col: str = "estimated_win_rate",
    max_picks_per_day: int = 0,
    min_probability: float = 0.0,
) -> WalkForwardResult:
    """Run walk-forward validation on historical data.

    Parameters
    ----------
    history_df : pd.DataFrame
        Full history with market_date, result, direction, target, and
        probability/edge columns.
    min_train_days : int
        Minimum days of history before starting predictions.
    selection_fn : callable, optional
        Function(day_df, train_df) -> selected_df that simulates the
        daily selection logic.  If None, uses all graded rows.
    probability_col : str
        Column name for the predicted probability.
    max_picks_per_day : int
        If > 0, limits picks per day to top N by probability.
    min_probability : float
        Minimum probability threshold for inclusion.

    Returns
    -------
    WalkForwardResult with accumulated metrics.
    """
    graded = history_df[history_df["result"].isin(["win", "loss"])].copy()
    graded["_date"] = pd.to_datetime(graded["market_date"], errors="coerce")
    graded = graded.dropna(subset=["_date"]).sort_values("_date")
    graded["_win"] = (graded["result"] == "win").astype(int)

    dates = sorted(graded["_date"].dt.date.unique())
    if len(dates) < min_train_days + 1:
        return WalkForwardResult()

    result = WalkForwardResult()
    all_predicted = []
    all_actual = []

    for i, test_date in enumerate(dates):
        if i < min_train_days:
            continue

        train_data = graded[graded["_date"].dt.date < test_date]
        test_data = graded[graded["_date"].dt.date == test_date].copy()

        if test_data.empty:
            continue

        # Apply selection function if provided
        if selection_fn is not None:
            try:
                test_data = selection_fn(test_data, train_data)
            except Exception:
                continue

        if test_data.empty:
            continue

        # Apply probability threshold
        prob = pd.to_numeric(test_data.get(probability_col), errors="coerce").fillna(0.5)
        if min_probability > 0:
            mask = prob >= min_probability
            test_data = test_data[mask]
            prob = prob[mask]

        if test_data.empty:
            continue

        # Apply max picks per day
        if max_picks_per_day > 0 and len(test_data) > max_picks_per_day:
            test_data = test_data.nlargest(max_picks_per_day, probability_col)
            prob = pd.to_numeric(test_data[probability_col], errors="coerce").fillna(0.5)

        # Grade
        wins_today = int(test_data["_win"].sum())
        losses_today = int(len(test_data) - wins_today)
        pnl_today = wins_today * PAYOUT - losses_today * 1.0

        result.total_picks += len(test_data)
        result.wins += wins_today
        result.losses += losses_today
        result.pnl_units += pnl_today
        result.days_tested += 1

        all_predicted.extend(prob.tolist())
        all_actual.extend(test_data["_win"].tolist())

        # Track by direction and target
        for direction in test_data["direction"].unique():
            d_mask = test_data["direction"] == direction
            d_wins = int(test_data.loc[d_mask, "_win"].sum())
            d_total = int(d_mask.sum())
            if direction not in result.by_direction:
                result.by_direction[direction] = {"wins": 0, "total": 0}
            result.by_direction[direction]["wins"] += d_wins
            result.by_direction[direction]["total"] += d_total

        for target in test_data["target"].unique():
            t_mask = test_data["target"] == target
            t_wins = int(test_data.loc[t_mask, "_win"].sum())
            t_total = int(t_mask.sum())
            if target not in result.by_target:
                result.by_target[target] = {"wins": 0, "total": 0}
            result.by_target[target]["wins"] += t_wins
            result.by_target[target]["total"] += t_total

        result.daily_results.append({
            "date": str(test_date),
            "picks": len(test_data),
            "wins": wins_today,
            "losses": losses_today,
            "pnl": pnl_today,
            "win_rate": wins_today / max(1, len(test_data)),
        })

    # Compute summary metrics
    if result.total_picks > 0:
        result.win_rate = result.wins / result.total_picks
        result.roi_pct = result.pnl_units / result.total_picks * 100
        result.avg_picks_per_day = result.total_picks / max(1, result.days_tested)

    if all_predicted and all_actual:
        predicted = np.array(all_predicted)
        actual = np.array(all_actual)
        result.calibration_gap = float(abs(predicted.mean() - actual.mean()))
        result.brier_score = float(((predicted - actual) ** 2).mean())

    # Compute win rates for direction/target breakdowns
    for d_stats in result.by_direction.values():
        d_stats["win_rate"] = d_stats["wins"] / max(1, d_stats["total"])
    for t_stats in result.by_target.values():
        t_stats["win_rate"] = t_stats["wins"] / max(1, t_stats["total"])

    return result


def format_walk_forward_result(result: WalkForwardResult) -> str:
    """Format walk-forward results as a readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("WALK-FORWARD VALIDATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"  Days tested: {result.days_tested}")
    lines.append(f"  Total picks: {result.total_picks}")
    lines.append(f"  Record: {result.wins}W-{result.losses}L ({result.win_rate:.1%})")
    lines.append(f"  PnL: {result.pnl_units:+.1f}u | ROI: {result.roi_pct:+.1f}%")
    lines.append(f"  Avg picks/day: {result.avg_picks_per_day:.1f}")
    lines.append(f"  Calibration gap: {result.calibration_gap:.4f}")
    lines.append(f"  Brier score: {result.brier_score:.4f}")

    if result.by_direction:
        lines.append(f"\n  By Direction:")
        for d, stats in sorted(result.by_direction.items()):
            lines.append(f"    {d}: {stats['wins']}W/{stats['total']} = {stats['win_rate']:.1%}")

    if result.by_target:
        lines.append(f"\n  By Target:")
        for t, stats in sorted(result.by_target.items()):
            lines.append(f"    {t}: {stats['wins']}W/{stats['total']} = {stats['win_rate']:.1%}")

    return "\n".join(lines)
