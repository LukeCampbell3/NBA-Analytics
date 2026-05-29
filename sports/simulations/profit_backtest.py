"""
Lightweight profit backtest utilities for MLB and NBA predictions.

Usage (CLI):
  python -m sports.simulations.profit_backtest --predictions predictions.csv \
      --settled settled.csv --stake 1.0

Predictions CSV should include columns: event_id,predicted_prob,american_odds,prediction_time
Settled CSV should include columns: event_id,actual_outcome (0/1)
"""
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def american_profit_per_unit(price: Optional[float]) -> Optional[float]:
    """Return profit (units) won per 1 unit wagered for American odds.

    Returns None for missing/invalid prices.
    """
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None

    if p == 0 or np.isnan(p):
        return None

    if p > 0:
        return p / 100.0
    return 100.0 / abs(p)


@dataclass
class BacktestMetrics:
    total_plays: int
    winning_plays: int
    hit_rate: float
    total_units_wagered: float
    total_units_won: float
    roi: float
    max_drawdown: float
    sharpe: float


def backtest_predictions(predictions: pd.DataFrame, settled: pd.DataFrame, *,
                         stake: float = 1.0, stake_col: Optional[str] = None,
                         time_col: str = "prediction_time") -> tuple[pd.DataFrame, BacktestMetrics]:
    """Backtest a flat (or per-row) staking strategy.

    predictions: DataFrame with at least `event_id`, `american_odds`, and optionally a stake_col.
    settled: DataFrame with `event_id` and `actual_outcome` (0/1 or bool).
    Returns: joined rows with per-row units_wagered/units_won and aggregated metrics.
    """
    if time_col in predictions.columns:
        predictions[time_col] = pd.to_datetime(predictions[time_col])

    # Join predictions to settled outcomes by event_id
    merged = pd.merge(predictions, settled[["event_id", "actual_outcome"]], on="event_id", how="left")
    merged = merged.copy()

    # Determine stake per row
    if stake_col and stake_col in merged.columns:
        merged["units_wagered"] = merged[stake_col].astype(float).fillna(float(stake))
    else:
        merged["units_wagered"] = float(stake)

    # Compute profit if win per unit
    merged["profit_per_unit"] = merged["american_odds"].apply(american_profit_per_unit)

    # Units won: profit_per_unit * units_wagered if actual_outcome truthy else 0
    merged["actual_outcome"] = merged["actual_outcome"].fillna(0).astype(float)
    merged["units_won"] = merged.apply(
        lambda r: (r["profit_per_unit"] * r["units_wagered"]) if bool(r["actual_outcome"]) and r["profit_per_unit"] is not None else 0.0,
        axis=1,
    )

    # Per-row pnl (won - wagered)
    merged["pnl"] = merged["units_won"] - merged["units_wagered"]

    # Sort by time if present, else keep original order
    if time_col in merged.columns:
        merged.sort_values(time_col, inplace=True)

    # Cumulative PnL
    merged["cum_pnl"] = merged["pnl"].cumsum()

    # Metrics
    total_plays = int(len(merged))
    winning_plays = int(merged[merged["actual_outcome"] == 1.0].shape[0])
    hit_rate = winning_plays / total_plays if total_plays > 0 else 0.0
    total_units_wagered = float(merged["units_wagered"].sum())
    total_units_won = float(merged["units_won"].sum())
    roi = (total_units_won - total_units_wagered) / total_units_wagered if total_units_wagered > 0 else 0.0

    # Max drawdown
    cum = merged["cum_pnl"].fillna(0).to_numpy(dtype=float)
    peak = np.maximum.accumulate(cum)
    drawdowns = peak - cum
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe-like: mean(pnl) / std(pnl)
    pnl_seq = pd.to_numeric(merged["pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sharpe = float(np.mean(pnl_seq) / (np.std(pnl_seq) + 1e-9)) if len(pnl_seq) > 1 else 0.0

    metrics = BacktestMetrics(
        total_plays=total_plays,
        winning_plays=winning_plays,
        hit_rate=hit_rate,
        total_units_wagered=total_units_wagered,
        total_units_won=total_units_won,
        roi=roi,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
    )

    return merged, metrics


def print_metrics(metrics: BacktestMetrics) -> None:
    print("BACKTEST SUMMARY")
    print("Total plays:", metrics.total_plays)
    print("Winning plays:", metrics.winning_plays)
    print(f"Hit rate: {metrics.hit_rate:.2%}")
    print(f"Total units wagered: {metrics.total_units_wagered:.2f}")
    print(f"Total units won: {metrics.total_units_won:.2f}")
    print(f"ROI: {metrics.roi:.2%}")
    print(f"Max drawdown: {metrics.max_drawdown:.2f}")
    print(f"Sharpe-like: {metrics.sharpe:.3f}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Profit backtest for predictions vs settled outcomes")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with predictions (event_id,american_odds,...)" )
    parser.add_argument("--settled", type=Path, required=True, help="CSV with settled outcomes (event_id,actual_outcome)")
    parser.add_argument("--stake", type=float, default=1.0, help="Units wagered per play (flat)")
    parser.add_argument("--stake-col", type=str, default=None, help="Optional column in predictions providing per-row stake")
    parser.add_argument("--out-csv", type=Path, default=None, help="Write per-row PnL CSV")
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions)
    settled = pd.read_csv(args.settled)

    rows, metrics = backtest_predictions(preds, settled, stake=args.stake, stake_col=args.stake_col)

    print_metrics(metrics)

    if args.out_csv:
        rows.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    _cli()
