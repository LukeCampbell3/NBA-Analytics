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


def percentile_tier_from_series(
    gap_percentile: pd.Series,
    medium_tier_percentile: float,
    strong_tier_percentile: float,
    elite_tier_percentile: float,
) -> pd.Series:
    pct = pd.to_numeric(gap_percentile, errors="coerce").fillna(0.0)
    return pd.Series(
        np.select(
            [
                pct >= float(elite_tier_percentile),
                pct >= float(strong_tier_percentile),
                pct >= float(medium_tier_percentile),
            ],
            ["elite", "strong", "medium"],
            default="weak",
        ),
        index=pct.index,
        dtype="object",
    )


def tier_action_level_from_series(percentile_tier: pd.Series) -> pd.Series:
    return percentile_tier.map({"weak": 0, "medium": 1, "strong": 2, "elite": 3}).fillna(0).astype(int)


def probability_action_level_from_series(
    expected_win_rate: pd.Series,
    min_bet_win_rate: float,
    medium_bet_win_rate: float,
    full_bet_win_rate: float,
) -> pd.Series:
    rate = pd.to_numeric(expected_win_rate, errors="coerce").fillna(0.0)
    return pd.Series(
        np.select(
            [
                rate >= float(full_bet_win_rate),
                rate >= float(medium_bet_win_rate),
                rate >= float(min_bet_win_rate),
            ],
            [3, 2, 1],
            default=0,
        ),
        index=rate.index,
        dtype="int64",
    )


def action_label_from_level(level: pd.Series) -> pd.Series:
    return level.map({0: "no_bet", 1: "small", 2: "medium", 3: "full"}).fillna("no_bet")


def bet_fraction_from_action_level(
    action_level: pd.Series,
    small_bet_fraction: float,
    medium_bet_fraction: float,
    full_bet_fraction: float,
    max_bet_fraction: float,
) -> pd.Series:
    fractions = action_level.map(
        {
            0: 0.0,
            1: float(small_bet_fraction),
            2: float(medium_bet_fraction),
            3: float(full_bet_fraction),
        }
    ).fillna(0.0)
    return pd.to_numeric(fractions, errors="coerce").fillna(0.0).clip(lower=0.0, upper=float(max_bet_fraction))


def apply_tiered_bet_sizing(
    df: pd.DataFrame,
    expected_win_rate_col: str,
    gap_percentile_col: str,
    min_bet_win_rate: float,
    medium_bet_win_rate: float,
    full_bet_win_rate: float,
    medium_tier_percentile: float,
    strong_tier_percentile: float,
    elite_tier_percentile: float,
    small_bet_fraction: float,
    medium_bet_fraction: float,
    full_bet_fraction: float,
    max_bet_fraction: float,
    max_total_bet_fraction: float,
) -> pd.DataFrame:
    out = df.copy()
    out["allocation_tier"] = percentile_tier_from_series(
        out.get(gap_percentile_col),
        medium_tier_percentile=medium_tier_percentile,
        strong_tier_percentile=strong_tier_percentile,
        elite_tier_percentile=elite_tier_percentile,
    )
    tier_levels = tier_action_level_from_series(out["allocation_tier"])
    probability_levels = probability_action_level_from_series(
        out.get(expected_win_rate_col),
        min_bet_win_rate=min_bet_win_rate,
        medium_bet_win_rate=medium_bet_win_rate,
        full_bet_win_rate=full_bet_win_rate,
    )
    out["allocation_action_from_tier"] = action_label_from_level(tier_levels)
    out["allocation_action_from_probability"] = action_label_from_level(probability_levels)
    out["allocation_action_level"] = np.minimum(tier_levels, probability_levels)
    out["allocation_action"] = action_label_from_level(out["allocation_action_level"])
    out["bet_fraction_raw"] = bet_fraction_from_action_level(
        out["allocation_action_level"],
        small_bet_fraction=small_bet_fraction,
        medium_bet_fraction=medium_bet_fraction,
        full_bet_fraction=full_bet_fraction,
        max_bet_fraction=max_bet_fraction,
    )

    total_fraction = float(pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0).sum())
    scale = 1.0
    if total_fraction > 0.0 and float(max_total_bet_fraction) > 0.0:
        scale = min(1.0, float(max_total_bet_fraction) / total_fraction)
    out["bet_fraction_scale"] = scale
    out["bet_fraction"] = pd.to_numeric(out["bet_fraction_raw"], errors="coerce").fillna(0.0) * scale
    return out


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

    if config.sizing_method == "tiered_probability":
        out = apply_tiered_bet_sizing(
            out,
            expected_win_rate_col="expected_win_rate",
            gap_percentile_col="gap_percentile",
            min_bet_win_rate=config.min_bet_win_rate,
            medium_bet_win_rate=config.medium_bet_win_rate,
            full_bet_win_rate=config.full_bet_win_rate,
            medium_tier_percentile=config.medium_tier_percentile,
            strong_tier_percentile=config.strong_tier_percentile,
            elite_tier_percentile=config.elite_tier_percentile,
            small_bet_fraction=config.small_bet_fraction,
            medium_bet_fraction=config.medium_bet_fraction,
            full_bet_fraction=config.full_bet_fraction,
            max_bet_fraction=config.max_bet_fraction,
            max_total_bet_fraction=config.max_total_bet_fraction,
        )
        out["size_multiplier"] = 1.0
        out["stake"] = bankroll * pd.to_numeric(out["bet_fraction"], errors="coerce").fillna(0.0)
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
