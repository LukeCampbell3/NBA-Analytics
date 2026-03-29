from __future__ import annotations

from itertools import product
from typing import Any, Iterable

import pandas as pd

from .gating import StrategyConfig
from .simulation import SimulationResult, simulate_strategy


def build_default_shadow_strategies() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            name="production_calibrated",
            probability_shrink_factor=0.80,
            ranking_mode="ev_adjusted",
            min_ev=0.05,
            min_final_confidence=0.03,
            min_recommendation="consider",
            max_plays_per_player=1,
            max_pts_plays=6,
            max_trb_plays=2,
            max_ast_plays=2,
            max_total_plays=10,
            non_pts_min_gap_percentile=0.90,
            sizing_method="base_fraction",
            base_bet_fraction=0.015,
            max_bet_fraction=0.02,
            edge_scale_start_percentile=0.75,
            edge_scale_span=0.25,
            edge_scale_lift=0.15,
            edge_scale_max_multiplier=1.25,
            target_thresholds={
                "PTS": {"consider_pct": 0.80, "strong_pct": 0.90, "elite_pct": 0.95},
                "TRB": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
                "AST": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
            },
        ),
        StrategyConfig(
            name="holdout_xgb_ltr",
            probability_shrink_factor=0.80,
            ranking_mode="xgb_ltr",
            xgb_ltr_min_train_rows=4000,
            xgb_ltr_num_pair_per_sample=12,
            min_ev=0.05,
            min_final_confidence=0.03,
            min_recommendation="consider",
            max_plays_per_player=1,
            max_pts_plays=6,
            max_trb_plays=2,
            max_ast_plays=2,
            max_total_plays=10,
            non_pts_min_gap_percentile=0.90,
            sizing_method="base_fraction",
            base_bet_fraction=0.015,
            max_bet_fraction=0.02,
            edge_scale_start_percentile=0.75,
            edge_scale_span=0.25,
            edge_scale_lift=0.15,
            edge_scale_max_multiplier=1.25,
            target_thresholds={
                "PTS": {"consider_pct": 0.80, "strong_pct": 0.90, "elite_pct": 0.95},
                "TRB": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
                "AST": {"consider_pct": 0.90, "strong_pct": 0.95, "elite_pct": 0.975},
            },
        ),
        StrategyConfig(
            name="strict_current_plus",
            probability_shrink_factor=0.80,
            ranking_mode="ev_adjusted",
            min_ev=0.01,
            min_final_confidence=0.04,
            max_pts_plays=5,
            max_trb_plays=3,
            max_ast_plays=2,
            max_total_plays=8,
            non_pts_min_gap_percentile=0.93,
            kelly_fraction=0.20,
            max_bet_fraction=0.04,
        ),
        StrategyConfig(
            name="balanced_current",
            probability_shrink_factor=0.80,
            ranking_mode="ev_adjusted",
            min_ev=0.0,
            min_final_confidence=0.03,
            max_pts_plays=6,
            max_trb_plays=4,
            max_ast_plays=2,
            max_total_plays=12,
            non_pts_min_gap_percentile=0.90,
            kelly_fraction=0.25,
            max_bet_fraction=0.05,
        ),
        StrategyConfig(
            name="aggressive_shadow",
            probability_shrink_factor=0.80,
            ranking_mode="ev_adjusted",
            min_ev=-0.01,
            min_final_confidence=0.02,
            max_pts_plays=8,
            max_trb_plays=5,
            max_ast_plays=3,
            max_total_plays=16,
            non_pts_min_gap_percentile=0.85,
            kelly_fraction=0.35,
            max_bet_fraction=0.06,
        ),
    ]


def build_grid_strategies(
    min_final_confidences: Iterable[float],
    min_evs: Iterable[float],
    max_total_plays: Iterable[int],
    non_pts_percentiles: Iterable[float],
    kelly_fractions: Iterable[float],
    base_config: StrategyConfig | None = None,
) -> list[StrategyConfig]:
    base = base_config or StrategyConfig(name="grid_base")
    configs: list[StrategyConfig] = []
    for idx, (min_conf, min_ev, max_total, non_pts_pct, kelly_fraction) in enumerate(
        product(min_final_confidences, min_evs, max_total_plays, non_pts_percentiles, kelly_fractions),
        start=1,
    ):
        payload = base.to_dict()
        payload.update(
            {
                "name": f"grid_{idx:03d}",
                "min_final_confidence": float(min_conf),
                "min_ev": float(min_ev),
                "max_total_plays": int(max_total),
                "non_pts_min_gap_percentile": float(non_pts_pct),
                "kelly_fraction": float(kelly_fraction),
            }
        )
        configs.append(StrategyConfig(**payload))
    return configs


def summarize_results(results: list[SimulationResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        overall = result.summary["overall"]
        selected = result.summary["gating_effectiveness"]["selected_candidates"]
        expected_profit = overall["expected_profit"]
        total_profit = overall["total_profit"]
        rows.append(
            {
                "strategy": result.config.name,
                "total_profit": total_profit,
                "ending_bankroll": overall["ending_bankroll"],
                "roi": overall["roi"],
                "profit_per_opportunity": overall["profit_per_opportunity"],
                "log_bankroll_growth": overall["log_bankroll_growth"],
                "max_drawdown": overall["max_drawdown"],
                "selected_plays": overall["selected_plays"],
                "selected_win_rate": selected["win_rate"],
                "gated_opportunities": overall["gated_opportunities"],
                "opportunities": overall["opportunities"],
                "expected_profit": expected_profit,
                "ev_capture_ratio": (total_profit / expected_profit) if expected_profit else None,
                "ev_realization_gap": overall["ev_realization_gap"],
            }
        )
    return pd.DataFrame.from_records(rows)


def run_policy_tuning(
    data: str | pd.DataFrame,
    configs: list[StrategyConfig],
    objective: str = "ending_bankroll",
) -> tuple[list[SimulationResult], pd.DataFrame]:
    results = [simulate_strategy(data, config) for config in configs]
    summary_df = summarize_results(results)
    if not summary_df.empty and objective in summary_df.columns:
        summary_df = summary_df.sort_values([objective, "total_profit"], ascending=[False, False]).reset_index(drop=True)
    return results, summary_df
