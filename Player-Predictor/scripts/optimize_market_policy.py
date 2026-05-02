#!/usr/bin/env python3
"""
Walk-forward search for market policy parameters on historical backtest rows.

This is a research tool for tuning the core hard-coded policy values that affect:
- board size
- hit rate
- EV gating
- confidence gating
- belief-uncertainty scaling

It converts the row-level backtest CSV into long-form decision rows, replays each
candidate policy chronologically, and reports train/holdout metrics so we can
avoid tuning entirely by feel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine.gating import StrategyConfig
from decision_engine.historical_backtest_adapter import backtest_rows_to_decisions
from decision_engine.policy_tuning import build_default_shadow_strategies
from decision_engine.simulation import simulate_strategy


@dataclass
class CandidateResult:
    name: str
    config: dict
    train: dict
    holdout: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize core market policy parameters with walk-forward replay.")
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "refreshed_market_comparison_strict_rows.csv",
        help="Row-level backtest CSV from backtest_inference_accuracy.py",
    )
    parser.add_argument(
        "--base-profile",
        type=str,
        default="production_high_precision",
        help="Base policy profile to perturb during search.",
    )
    parser.add_argument("--iterations", type=int, default=30, help="Random-search iterations around the base policy.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.80,
        help="Fraction of chronological dates used for search/tuning before holdout evaluation.",
    )
    parser.add_argument("--min-win-rate", type=float, default=0.70, help="Minimum acceptable win rate for the optimization score.")
    parser.add_argument("--min-plays", type=int, default=75, help="Minimum acceptable selected plays for the optimization score.")
    parser.add_argument("--loss-penalty", type=float, default=0.45, help="Penalty applied to each loss inside the optimization score.")
    parser.add_argument("--play-bonus", type=float, default=0.05, help="Small reward for maintaining usable board size.")
    parser.add_argument(
        "--holdout-top-k",
        type=int,
        default=12,
        help="After ranking by train score, re-rank the top K candidates by holdout score for reporting.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "policy_search",
        help="Directory for JSON/CSV outputs.",
    )
    return parser.parse_args()


def _policy_map() -> dict[str, StrategyConfig]:
    return {config.name: config for config in build_default_shadow_strategies()}


def _window_metrics(decisions: pd.DataFrame, target_dates: set[pd.Timestamp]) -> dict:
    window = decisions.loc[decisions["target_date"].isin(target_dates)].copy()
    if "selected" in window.columns:
        selected = window.loc[window["selected"].fillna(False)].copy()
    else:
        selected = window.iloc[0:0].copy()
    plays = int(len(selected))
    wins = int((selected["result"] == "win").sum()) if not selected.empty else 0
    losses = int((selected["result"] == "loss").sum()) if not selected.empty else 0
    pushes = int((selected["result"] == "push").sum()) if not selected.empty else 0
    win_rate = (wins / (wins + losses)) if (wins + losses) else None
    total_profit = float(pd.to_numeric(selected.get("profit"), errors="coerce").fillna(0.0).sum()) if not selected.empty else 0.0
    total_stake = float(pd.to_numeric(selected.get("stake"), errors="coerce").fillna(0.0).sum()) if not selected.empty else 0.0
    roi = (total_profit / total_stake) if total_stake > 0.0 else None
    avg_plays_per_day = (plays / max(1, len(target_dates))) if target_dates else 0.0
    return {
        "dates": int(len(target_dates)),
        "selected_plays": plays,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "profit": total_profit,
        "stake": total_stake,
        "roi": roi,
        "avg_plays_per_day": avg_plays_per_day,
    }


def _score_window(metrics: dict, *, min_win_rate: float, min_plays: int, loss_penalty: float, play_bonus: float) -> float:
    plays = int(metrics.get("selected_plays") or 0)
    wins = int(metrics.get("wins") or 0)
    losses = int(metrics.get("losses") or 0)
    win_rate = float(metrics.get("win_rate") or 0.0)

    score = float(wins) - float(loss_penalty) * float(losses) + float(play_bonus) * float(plays)
    if plays < int(min_plays):
        score -= float(min_plays - plays) * (1.0 + float(loss_penalty))
    if win_rate < float(min_win_rate):
        score -= float(min_win_rate - win_rate) * max(float(plays), float(min_plays))
    return float(score)


def _candidate_payload(config: StrategyConfig) -> dict:
    payload = config.to_dict()
    payload["target_thresholds"] = {
        target: {key: float(value) for key, value in values.items()}
        for target, values in payload.get("target_thresholds", {}).items()
    }
    return payload


def _sample_config(base: StrategyConfig, rng: np.random.Generator, idx: int) -> StrategyConfig:
    payload = base.to_dict()
    payload["name"] = f"{base.name}_search_{idx:03d}"
    payload["probability_shrink_factor"] = float(rng.uniform(0.68, 0.82))
    payload["min_ev"] = float(rng.uniform(0.02, 0.08))
    payload["min_final_confidence"] = float(rng.uniform(0.02, 0.07))
    payload["non_pts_min_gap_percentile"] = float(rng.uniform(0.86, 0.96))
    payload["edge_adjust_k"] = float(rng.uniform(0.15, 0.45))
    payload["max_total_plays"] = int(rng.integers(4, 11))
    payload["max_pts_plays"] = int(rng.integers(4, 9))
    payload["max_trb_plays"] = int(rng.integers(1, 5))
    payload["max_ast_plays"] = int(rng.integers(1, 4))
    payload["min_bet_win_rate"] = float(rng.uniform(0.55, 0.60))
    payload["medium_bet_win_rate"] = float(payload["min_bet_win_rate"] + rng.uniform(0.015, 0.04))
    payload["full_bet_win_rate"] = float(payload["medium_bet_win_rate"] + rng.uniform(0.025, 0.06))
    payload["medium_tier_percentile"] = float(rng.uniform(0.78, 0.86))
    payload["strong_tier_percentile"] = float(rng.uniform(0.88, 0.93))
    payload["elite_tier_percentile"] = float(rng.uniform(0.94, 0.98))
    payload["belief_uncertainty_lower"] = float(rng.uniform(0.70, 0.85))
    payload["belief_uncertainty_upper"] = float(rng.uniform(1.05, 1.20))
    if payload["belief_uncertainty_upper"] < payload["belief_uncertainty_lower"] + 0.18:
        payload["belief_uncertainty_upper"] = float(payload["belief_uncertainty_lower"] + 0.18)
    payload["medium_bet_win_rate"] = float(min(payload["medium_bet_win_rate"], 0.67))
    payload["full_bet_win_rate"] = float(min(max(payload["full_bet_win_rate"], payload["medium_bet_win_rate"] + 0.01), 0.72))
    payload["strong_tier_percentile"] = float(max(payload["strong_tier_percentile"], payload["medium_tier_percentile"] + 0.04))
    payload["elite_tier_percentile"] = float(max(payload["elite_tier_percentile"], payload["strong_tier_percentile"] + 0.03))
    return StrategyConfig(**payload)


def _evaluate_candidate(
    decisions_df: pd.DataFrame,
    train_dates: set[pd.Timestamp],
    holdout_dates: set[pd.Timestamp],
    config: StrategyConfig,
    *,
    min_win_rate: float,
    min_plays: int,
    loss_penalty: float,
    play_bonus: float,
) -> CandidateResult:
    simulation = simulate_strategy(decisions_df, config)
    train = _window_metrics(simulation.decisions, train_dates)
    holdout = _window_metrics(simulation.decisions, holdout_dates)
    train["score"] = _score_window(
        train,
        min_win_rate=min_win_rate,
        min_plays=min_plays,
        loss_penalty=loss_penalty,
        play_bonus=play_bonus,
    )
    holdout["score"] = _score_window(
        holdout,
        min_win_rate=min_win_rate,
        min_plays=max(10, int(min_plays * 0.2)),
        loss_penalty=loss_penalty,
        play_bonus=play_bonus,
    )
    return CandidateResult(name=config.name, config=_candidate_payload(config), train=train, holdout=holdout)


def main() -> None:
    args = parse_args()
    policy_map = _policy_map()
    if args.base_profile not in policy_map:
        raise ValueError(f"Unknown base profile: {args.base_profile}")

    decisions_df = backtest_rows_to_decisions(args.history_csv.resolve())
    if decisions_df.empty:
        raise RuntimeError(f"No decision rows were derived from {args.history_csv}")

    unique_dates = sorted(pd.Timestamp(value).normalize() for value in decisions_df["target_date"].dropna().unique())
    if len(unique_dates) < 20:
        raise RuntimeError(f"Need at least 20 historical dates for a meaningful split, found {len(unique_dates)}")

    split_idx = min(max(int(len(unique_dates) * float(args.train_fraction)), 1), len(unique_dates) - 1)
    train_dates = set(unique_dates[:split_idx])
    holdout_dates = set(unique_dates[split_idx:])

    rng = np.random.default_rng(int(args.seed))
    base = policy_map[args.base_profile]

    candidates: list[StrategyConfig] = list(policy_map.values())
    for idx in range(1, int(args.iterations) + 1):
        candidates.append(_sample_config(base, rng, idx))

    results: list[CandidateResult] = []
    for config in candidates:
        results.append(
            _evaluate_candidate(
                decisions_df,
                train_dates,
                holdout_dates,
                config,
                min_win_rate=float(args.min_win_rate),
                min_plays=int(args.min_plays),
                loss_penalty=float(args.loss_penalty),
                play_bonus=float(args.play_bonus),
            )
        )

    results_by_train = sorted(
        results,
        key=lambda item: (
            -(item.train.get("score") or 0.0),
            -(item.train.get("win_rate") or 0.0),
            -(item.train.get("selected_plays") or 0),
            -(item.holdout.get("score") or 0.0),
        ),
    )
    top_k = max(1, min(int(args.holdout_top_k), len(results_by_train)))
    results_by_holdout = sorted(
        results_by_train[:top_k],
        key=lambda item: (
            -(item.holdout.get("score") or 0.0),
            -(item.holdout.get("win_rate") or 0.0),
            -(item.holdout.get("selected_plays") or 0),
            -(item.train.get("score") or 0.0),
        ),
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "policy_search_summary.json"
    csv_path = out_dir / "policy_search_results.csv"

    csv_rows = []
    for item in results_by_train:
        csv_rows.append(
            {
                "name": item.name,
                "train_score": item.train.get("score"),
                "train_win_rate": item.train.get("win_rate"),
                "train_selected_plays": item.train.get("selected_plays"),
                "train_wins": item.train.get("wins"),
                "train_losses": item.train.get("losses"),
                "holdout_score": item.holdout.get("score"),
                "holdout_win_rate": item.holdout.get("win_rate"),
                "holdout_selected_plays": item.holdout.get("selected_plays"),
                "holdout_wins": item.holdout.get("wins"),
                "holdout_losses": item.holdout.get("losses"),
                "probability_shrink_factor": item.config.get("probability_shrink_factor"),
                "min_ev": item.config.get("min_ev"),
                "min_final_confidence": item.config.get("min_final_confidence"),
                "non_pts_min_gap_percentile": item.config.get("non_pts_min_gap_percentile"),
                "edge_adjust_k": item.config.get("edge_adjust_k"),
                "max_total_plays": item.config.get("max_total_plays"),
                "max_pts_plays": item.config.get("max_pts_plays"),
                "max_trb_plays": item.config.get("max_trb_plays"),
                "max_ast_plays": item.config.get("max_ast_plays"),
                "belief_uncertainty_lower": item.config.get("belief_uncertainty_lower"),
                "belief_uncertainty_upper": item.config.get("belief_uncertainty_upper"),
            }
        )
    pd.DataFrame.from_records(csv_rows).to_csv(csv_path, index=False)

    payload = {
        "history_csv": str(args.history_csv.resolve()),
        "base_profile": args.base_profile,
        "dates_total": len(unique_dates),
        "train_dates": [str(value.date()) for value in sorted(train_dates)],
        "holdout_dates": [str(value.date()) for value in sorted(holdout_dates)],
        "search": {
            "iterations": int(args.iterations),
            "seed": int(args.seed),
            "min_win_rate": float(args.min_win_rate),
            "min_plays": int(args.min_plays),
            "loss_penalty": float(args.loss_penalty),
            "play_bonus": float(args.play_bonus),
            "holdout_top_k": int(top_k),
        },
        "best_train": {
            "name": results_by_train[0].name,
            "train": results_by_train[0].train,
            "holdout": results_by_train[0].holdout,
            "config": results_by_train[0].config,
        } if results_by_train else None,
        "best_holdout_within_top_k_train": {
            "name": results_by_holdout[0].name,
            "train": results_by_holdout[0].train,
            "holdout": results_by_holdout[0].holdout,
            "config": results_by_holdout[0].config,
        } if results_by_holdout else None,
        "top_train_results": [
            {
                "name": item.name,
                "train": item.train,
                "holdout": item.holdout,
                "config": item.config,
            }
            for item in results_by_train[:top_k]
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best_train = results_by_train[0]
    best_holdout = results_by_holdout[0]
    print("\n" + "=" * 90)
    print("POLICY SEARCH")
    print("=" * 90)
    print(f"History rows: {len(decisions_df):,}")
    print(f"Dates:        {len(unique_dates)} ({len(train_dates)} train / {len(holdout_dates)} holdout)")
    print(
        "Best train:   "
        f"{best_train.name} | score={best_train.train['score']:.2f} | "
        f"wins={best_train.train['wins']} losses={best_train.train['losses']} "
        f"plays={best_train.train['selected_plays']} win_rate={best_train.train['win_rate']}"
    )
    print(
        "Best holdout: "
        f"{best_holdout.name} | score={best_holdout.holdout['score']:.2f} | "
        f"wins={best_holdout.holdout['wins']} losses={best_holdout.holdout['losses']} "
        f"plays={best_holdout.holdout['selected_plays']} win_rate={best_holdout.holdout['win_rate']}"
    )
    print(f"CSV:          {csv_path}")
    print(f"Summary:      {summary_path}")


if __name__ == "__main__":
    main()
