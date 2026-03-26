from __future__ import annotations

import numpy as np
import pandas as pd

from .gating import StrategyConfig


def american_profit_per_unit(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def compute_kelly_fraction(expected_win_rate: float, payout_per_unit: float, expected_push_rate: float = 0.0) -> float:
    p_win = float(np.clip(expected_win_rate, 0.0, 1.0))
    p_push = float(np.clip(expected_push_rate, 0.0, 1.0 - p_win))
    p_loss = max(0.0, 1.0 - p_win - p_push)
    if payout_per_unit <= 0.0:
        return 0.0
    full_kelly = (payout_per_unit * p_win - p_loss) / payout_per_unit
    return max(0.0, float(full_kelly))


def percentile_size_multiplier(gap_percentile: pd.Series, config: StrategyConfig) -> pd.Series:
    pct = pd.to_numeric(gap_percentile, errors="coerce").fillna(0.0)
    start = float(config.edge_scale_start_percentile)
    span = max(float(config.edge_scale_span), 1e-9)
    lift = float(config.edge_scale_lift)
    max_multiplier = float(config.edge_scale_max_multiplier)
    scaled = 1.0 + lift * ((pct - start) / span).clip(lower=0.0)
    return scaled.clip(lower=1.0, upper=max_multiplier)


def size_bets(selected_df: pd.DataFrame, bankroll: float, config: StrategyConfig) -> pd.DataFrame:
    if selected_df.empty:
        out = selected_df.copy()
        out["stake"] = 0.0
        out["bet_fraction"] = 0.0
        out["raw_kelly_fraction"] = 0.0
        out["payout_per_unit"] = american_profit_per_unit(config.american_odds)
        out["size_multiplier"] = 1.0
        return out

    payout = american_profit_per_unit(config.american_odds)
    out = selected_df.copy()
    out["payout_per_unit"] = payout

    raw_kelly = out.apply(
        lambda row: compute_kelly_fraction(
            expected_win_rate=float(row.get("expected_win_rate", 0.0)),
            payout_per_unit=payout,
            expected_push_rate=float(row.get("expected_push_rate", 0.0)),
        ),
        axis=1,
    ).astype(float)
    out["raw_kelly_fraction"] = raw_kelly

    if config.sizing_method == "flat":
        out["bet_fraction"] = 0.0
        out["stake"] = float(config.flat_stake)
        out["size_multiplier"] = 1.0
        return out

    size_multiplier = percentile_size_multiplier(out.get("gap_percentile"), config)
    out["size_multiplier"] = size_multiplier

    if config.sizing_method == "base_fraction":
        scaled_fraction = pd.Series(float(config.base_bet_fraction), index=out.index, dtype="float64") * size_multiplier
    else:
        scaled_fraction = raw_kelly * float(config.kelly_fraction) * size_multiplier

    scaled_fraction = scaled_fraction.clip(lower=float(config.min_bet_fraction), upper=float(config.max_bet_fraction))
    out["bet_fraction"] = scaled_fraction
    out["stake"] = bankroll * scaled_fraction
    return out


def settle_bets(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df.empty:
        out = selected_df.copy()
        out["expected_profit"] = 0.0
        out["profit"] = 0.0
        out["roi"] = 0.0
        return out

    out = selected_df.copy()
    out["expected_profit"] = out["stake"] * pd.to_numeric(out["ev"], errors="coerce").fillna(0.0)
    result = out["result"].astype(str).str.lower()
    payout = pd.to_numeric(out["payout_per_unit"], errors="coerce").fillna(0.0)
    out["profit"] = np.select(
        [result.eq("win"), result.eq("push")],
        [out["stake"] * payout, 0.0],
        default=-out["stake"],
    )
    out["roi"] = np.where(out["stake"] > 0.0, out["profit"] / out["stake"], 0.0)
    return out
