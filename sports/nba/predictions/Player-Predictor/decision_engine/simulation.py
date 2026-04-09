from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .gating import StrategyConfig, build_history_lookup, prepare_historical_decisions, score_candidates
from .selection import apply_policy
from .sizing import settle_bets, size_bets
from .validation import summarize_simulation


@dataclass
class SimulationResult:
    config: StrategyConfig
    decisions: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, Any]


def simulate_strategy(data: str | pd.DataFrame, config: StrategyConfig) -> SimulationResult:
    historical_df = prepare_historical_decisions(data)
    bankroll = float(config.starting_bankroll)
    decision_frames: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []

    for target_date in sorted(historical_df["target_date"].dropna().unique()):
        current_df = historical_df.loc[historical_df["target_date"] == target_date].copy()
        history_df = historical_df.loc[historical_df["target_date"] < target_date].copy()
        if len(history_df) < int(config.min_history_rows):
            continue

        history_lookup = build_history_lookup(history_df, config)
        if not history_lookup:
            continue

        scored = score_candidates(current_df, history_lookup, config)
        if scored.empty:
            continue

        annotated = apply_policy(scored, config).reset_index(drop=True)
        annotated["bankroll_start"] = bankroll
        annotated["bankroll_end"] = bankroll
        annotated["stake"] = 0.0
        annotated["bet_fraction"] = 0.0
        annotated["raw_kelly_fraction"] = 0.0
        annotated["size_multiplier"] = 1.0
        annotated["payout_per_unit"] = 0.0
        annotated["expected_profit"] = 0.0
        annotated["profit"] = 0.0
        annotated["roi"] = 0.0

        selected_mask = annotated["selected"]
        selected = settle_bets(size_bets(annotated.loc[selected_mask].copy(), bankroll, config))
        day_profit = float(selected["profit"].sum()) if not selected.empty else 0.0
        bankroll_end = bankroll + day_profit

        if not selected.empty:
            annotated.loc[selected.index, ["stake", "bet_fraction", "raw_kelly_fraction", "size_multiplier", "payout_per_unit", "expected_profit", "profit", "roi"]] = selected[
                ["stake", "bet_fraction", "raw_kelly_fraction", "size_multiplier", "payout_per_unit", "expected_profit", "profit", "roi"]
            ]

        annotated["bankroll_end"] = bankroll_end
        decision_frames.append(annotated)
        daily_rows.append(
            {
                "strategy": config.name,
                "target_date": pd.Timestamp(target_date),
                "bankroll_start": bankroll,
                "bankroll_end": bankroll_end,
                "day_profit": day_profit,
                "opportunities": int(len(annotated)),
                "gated_opportunities": int(annotated["gating_passed"].sum()),
                "selected_plays": int(annotated["selected"].sum()),
                "expected_profit": float(annotated["expected_profit"].sum()),
            }
        )
        bankroll = bankroll_end

    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else historical_df.iloc[0:0].copy()
    daily = pd.DataFrame.from_records(daily_rows)
    summary = summarize_simulation(decisions, daily, config)
    return SimulationResult(config=config, decisions=decisions, daily=daily, summary=summary)
