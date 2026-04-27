#!/usr/bin/env python3
"""
Replay a recent historical window from Data-Proc and compare pool-selection strategies.

This script is intentionally self-contained so we can evaluate fallback pool selection
even when the full daily run archive or production model artifacts are unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "inference"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from structured_stack_inference import StructuredStackInference
from export_daily_predictions_web import apply_adaptive_board_sizing, apply_variance_aware_reexpand


DEFAULT_START_DATE = "2026-04-06"
DEFAULT_END_DATE = "2026-04-12"
DEFAULT_TOP_K = 4
PAYOUT_MINUS_110 = 100.0 / 110.0
TARGETS = ("PTS", "TRB", "AST")
BACKTEST_REEXPAND_TOP2_AVG_PROBABILITY_MAX = 0.76
BACKTEST_REEXPAND_THIRD_PROBABILITY_MIN = 0.72
BACKTEST_REEXPAND_THIRD_CONFIDENCE_MIN = 0.75


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    min_prob: float
    min_ev: float
    min_confidence: float
    min_edge_to_sigma: float
    sort_columns: tuple[str, ...]


STRATEGIES: tuple[StrategyConfig, ...] = (
    StrategyConfig(
        name="win_rate_priority",
        min_prob=0.505,
        min_ev=0.0,
        min_confidence=0.45,
        min_edge_to_sigma=0.12,
        sort_columns=("estimated_win_rate", "estimated_ev", "selection_confidence", "abs_edge"),
    ),
    StrategyConfig(
        name="robust_pool_score",
        min_prob=0.503,
        min_ev=0.0,
        min_confidence=0.42,
        min_edge_to_sigma=0.10,
        sort_columns=("robust_pool_score", "estimated_win_rate", "estimated_ev", "selection_confidence"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest recent NBA pool-selection strategies.")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Max plays per slate.")
    parser.add_argument("--min-history-rows", type=int, default=12, help="Minimum prior rows required before making a prediction.")
    parser.add_argument("--recent-history-rows", type=int, default=18, help="Recent rows passed into inference for each prediction.")
    parser.add_argument(
        "--data-proc-root",
        type=Path,
        default=REPO_ROOT / "Data-Proc",
        help="Processed player history root.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "model",
        help="Model directory. Missing artifacts automatically fall back to heuristic mode.",
    )
    parser.add_argument("--rows-csv-out", type=Path, default=None, help="Optional candidate rows CSV output.")
    parser.add_argument("--summary-json-out", type=Path, default=None, help="Optional strategy summary JSON output.")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _normalize_game_key(row: pd.Series) -> str:
    date_key = str(row.get("Date", ""))
    team_id = str(row.get("Team_ID", "")).strip()
    opp_id = str(row.get("Opponent_ID", "")).strip()
    matchup = str(row.get("MATCHUP", "")).strip()
    teams = sorted(part for part in [team_id, opp_id] if part)
    if teams:
        return f"{date_key}|{'|'.join(teams)}"
    if matchup:
        return f"{date_key}|{matchup}"
    return date_key


def _resolve_result(direction: str, line: float, actual: float, tol: float = 1e-9) -> str:
    if not math.isfinite(line) or not math.isfinite(actual):
        return "missing"
    if abs(actual - line) <= tol:
        return "push"
    if str(direction).upper() == "OVER":
        return "win" if actual > line else "loss"
    return "win" if actual < line else "loss"


def _build_confidence(
    *,
    edge_to_sigma: float,
    history_rows: int,
    spike_probability: float,
) -> float:
    history_component = min(1.0, max(0.0, float(history_rows)) / 40.0)
    signal_component = min(1.0, max(0.0, float(edge_to_sigma)) / 0.75)
    stability_component = max(0.0, min(1.0, 1.0 - float(spike_probability)))
    return float(0.45 * signal_component + 0.35 * history_component + 0.20 * stability_component)


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
    mean = float(numeric.mean())
    return (numeric - mean) / std


def build_candidate_frame(
    predictor: StructuredStackInference,
    *,
    data_proc_root: Path,
    start_date: str,
    end_date: str,
    min_history_rows: int,
    recent_history_rows: int,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    records: list[dict[str, Any]] = []

    for csv_path in sorted(data_proc_root.glob("*/*_processed_processed.csv")):
        try:
            history_df = pd.read_csv(csv_path)
        except Exception:
            continue
        if history_df.empty or "Date" not in history_df.columns:
            continue

        history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
        history_df = history_df.loc[history_df["Date"].notna()].sort_values("Date").reset_index(drop=True)
        if history_df.empty:
            continue

        for index, row in history_df.iterrows():
            game_date = row["Date"]
            if game_date < start_ts or game_date > end_ts:
                continue
            if _safe_float(row.get("Did_Not_Play"), default=0.0) >= 0.5:
                continue

            prior = history_df.iloc[:index].copy()
            if len(prior.index) < int(min_history_rows):
                continue
            recent = prior.tail(int(recent_history_rows)).copy()
            explanation = predictor.predict(recent, assume_prepared=True, return_debug=True)

            active_history = prior.loc[pd.to_numeric(prior.get("Did_Not_Play"), errors="coerce").fillna(0.0) < 0.5].copy()
            history_rows = int(len(active_history.index))
            player_name = str(row.get("Player") or csv_path.parent.name)
            game_key = _normalize_game_key(row)

            for target in TARGETS:
                market_line = pd.to_numeric(pd.Series([row.get(f"Market_{target}")]), errors="coerce").iloc[0]
                actual = pd.to_numeric(pd.Series([row.get(target)]), errors="coerce").iloc[0]
                if pd.isna(market_line) or pd.isna(actual):
                    continue

                predicted = _safe_float((explanation.get("predicted") or {}).get(target), default=float("nan"))
                if not math.isfinite(predicted):
                    continue
                target_factors = (explanation.get("target_factors") or {}).get(target, {})
                sigma = max(_safe_float(target_factors.get("uncertainty_sigma"), default=0.0), 0.25)
                spike_probability = float(np.clip(_safe_float(target_factors.get("spike_probability"), default=0.5), 0.0, 1.0))
                edge = float(predicted - float(market_line))
                direction = "OVER" if edge >= 0.0 else "UNDER"
                abs_edge = abs(edge)
                edge_to_sigma = abs_edge / sigma if sigma > 0 else 0.0
                estimated_win_rate = float(np.clip(_normal_cdf(edge_to_sigma), 0.5, 0.92))
                estimated_ev = float(estimated_win_rate * PAYOUT_MINUS_110 - (1.0 - estimated_win_rate))
                confidence = _build_confidence(
                    edge_to_sigma=edge_to_sigma,
                    history_rows=history_rows,
                    spike_probability=spike_probability,
                )
                result = _resolve_result(direction, float(market_line), float(actual))

                records.append(
                    {
                        "market_date": game_date.strftime("%Y-%m-%d"),
                        "player": player_name,
                        "target": target,
                        "direction": direction,
                        "prediction": float(predicted),
                        "market_line": float(market_line),
                        "actual": float(actual),
                        "result": result,
                        "edge": edge,
                        "abs_edge": abs_edge,
                        "uncertainty_sigma": sigma,
                        "edge_to_sigma": edge_to_sigma,
                        "estimated_win_rate": estimated_win_rate,
                        "estimated_ev": estimated_ev,
                        "selection_confidence": confidence,
                        "history_rows": history_rows,
                        "spike_probability": spike_probability,
                        "game_key": game_key,
                    }
                )

    return pd.DataFrame.from_records(records)


def add_day_level_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    scored_parts: list[pd.DataFrame] = []
    for market_date, part in candidates.groupby("market_date", dropna=False):
        scored = part.copy()
        scored["z_prob"] = _zscore(scored["estimated_win_rate"])
        scored["z_ev"] = _zscore(scored["estimated_ev"])
        scored["z_confidence"] = _zscore(scored["selection_confidence"])
        scored["z_edge_to_sigma"] = _zscore(scored["edge_to_sigma"])
        scored["z_history"] = _zscore(scored["history_rows"])
        scored["z_uncertainty"] = _zscore(scored["uncertainty_sigma"])
        scored["z_spike"] = _zscore(scored["spike_probability"])
        scored["robust_pool_score"] = (
            0.30 * scored["z_prob"]
            + 0.24 * scored["z_ev"]
            + 0.18 * scored["z_confidence"]
            + 0.16 * scored["z_edge_to_sigma"]
            + 0.08 * scored["z_history"]
            - 0.02 * scored["z_uncertainty"]
            - 0.04 * scored["z_spike"]
        )
        scored_parts.append(scored)
    return pd.concat(scored_parts, ignore_index=True)


def select_daily_pool(candidates: pd.DataFrame, *, strategy: StrategyConfig, top_k: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    eligible = candidates.loc[
        candidates["estimated_win_rate"].ge(float(strategy.min_prob))
        & candidates["estimated_ev"].ge(float(strategy.min_ev))
        & candidates["selection_confidence"].ge(float(strategy.min_confidence))
        & candidates["edge_to_sigma"].ge(float(strategy.min_edge_to_sigma))
    ].copy()
    if eligible.empty:
        return eligible

    ascending = [False] * len(strategy.sort_columns)
    eligible = eligible.sort_values(list(strategy.sort_columns), ascending=ascending, na_position="last").reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    seen_players: set[str] = set()
    game_counts: dict[str, int] = {}
    for _, row in eligible.iterrows():
        player = str(row.get("player", "")).strip().lower()
        game_key = str(row.get("game_key", "")).strip().lower()
        if player and player in seen_players:
            continue
        if game_key and game_counts.get(game_key, 0) >= 2:
            continue
        selected_rows.append(row)
        if player:
            seen_players.add(player)
        if game_key:
            game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if len(selected_rows) >= int(top_k):
            break

    if not selected_rows:
        return eligible.head(0).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def summarize_selection(selection: pd.DataFrame) -> dict[str, Any]:
    if selection.empty:
        return {
            "plays": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": None,
            "roi_per_play": None,
            "avg_estimated_win_rate": None,
            "avg_estimated_ev": None,
            "avg_selection_confidence": None,
        }

    wins = int(selection["result"].eq("win").sum())
    losses = int(selection["result"].eq("loss").sum())
    pushes = int(selection["result"].eq("push").sum())
    resolved = wins + losses
    roi = ((wins * PAYOUT_MINUS_110) - losses) / resolved if resolved > 0 else None
    return {
        "plays": int(len(selection)),
        "resolved": int(resolved),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": float(wins / resolved) if resolved > 0 else None,
        "roi_per_play": float(roi) if roi is not None else None,
        "avg_estimated_win_rate": _safe_float(selection["estimated_win_rate"].mean(), default=float("nan")),
        "avg_estimated_ev": _safe_float(selection["estimated_ev"].mean(), default=float("nan")),
        "avg_selection_confidence": _safe_float(selection["selection_confidence"].mean(), default=float("nan")),
    }


def build_strategy_selection(
    candidates: pd.DataFrame,
    *,
    strategy: StrategyConfig,
    top_k: int,
    adaptive_sizing: bool = False,
    variance_aware_reexpand: bool = False,
) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for _, part in candidates.groupby("market_date", dropna=False):
        picked = select_daily_pool(part, strategy=strategy, top_k=int(top_k))
        if adaptive_sizing and not picked.empty:
            adaptive_ready = picked.copy()
            adaptive_ready["pool_selection_score"] = pd.to_numeric(adaptive_ready.get("robust_pool_score"), errors="coerce")
            picked = apply_adaptive_board_sizing(adaptive_ready)
            if variance_aware_reexpand and not picked.empty:
                picked = apply_variance_aware_reexpand(
                    picked,
                    adaptive_ready,
                    probability_field="estimated_win_rate",
                    confidence_field="selection_confidence",
                    ev_field="estimated_ev",
                    max_top2_avg_probability=float(BACKTEST_REEXPAND_TOP2_AVG_PROBABILITY_MAX),
                    min_third_probability=float(BACKTEST_REEXPAND_THIRD_PROBABILITY_MIN),
                    min_third_confidence=float(BACKTEST_REEXPAND_THIRD_CONFIDENCE_MIN),
                    min_third_ev=0.0,
                )
        if picked.empty:
            continue
        picked = picked.copy()
        suffix = ""
        if adaptive_sizing:
            suffix = "_adaptive_reexpand" if variance_aware_reexpand else "_adaptive"
        picked["strategy"] = strategy.name + suffix
        selections.append(picked)
    if not selections:
        return candidates.head(0).copy()
    return pd.concat(selections, ignore_index=True)


def main() -> None:
    args = parse_args()
    predictor = StructuredStackInference(model_dir=str(args.model_dir.resolve()))

    candidates = build_candidate_frame(
        predictor,
        data_proc_root=args.data_proc_root.resolve(),
        start_date=args.start_date,
        end_date=args.end_date,
        min_history_rows=int(args.min_history_rows),
        recent_history_rows=int(args.recent_history_rows),
    )
    candidates = add_day_level_scores(candidates)

    summary: dict[str, Any] = {
        "window": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "top_k": int(args.top_k),
            "candidate_rows": int(len(candidates)),
            "market_dates": sorted(candidates["market_date"].dropna().astype(str).unique().tolist()) if not candidates.empty else [],
        },
        "predictor": {
            "artifact_free": bool(getattr(predictor, "artifact_free", False)),
            "artifact_free_reason": getattr(predictor, "artifact_free_reason", None),
        },
        "strategies": {},
    }

    for strategy in STRATEGIES:
        selected_df = build_strategy_selection(candidates, strategy=strategy, top_k=int(args.top_k), adaptive_sizing=False)
        summary["strategies"][strategy.name] = summarize_selection(selected_df)
        if strategy.name == "robust_pool_score":
            adaptive_df = build_strategy_selection(candidates, strategy=strategy, top_k=int(args.top_k), adaptive_sizing=True)
            summary["strategies"][f"{strategy.name}_adaptive"] = summarize_selection(adaptive_df)
            adaptive_reexpand_df = build_strategy_selection(
                candidates,
                strategy=strategy,
                top_k=int(args.top_k),
                adaptive_sizing=True,
                variance_aware_reexpand=True,
            )
            summary["strategies"][f"{strategy.name}_adaptive_reexpand"] = summarize_selection(adaptive_reexpand_df)

    if args.rows_csv_out is not None:
        args.rows_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(args.rows_csv_out.resolve(), index=False)
    if args.summary_json_out is not None:
        args.summary_json_out.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json_out.resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
