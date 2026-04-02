#!/usr/bin/env python3
"""
Walk-forward lambda tuning for board-objective selection mode.

The tuner evaluates lambda grids over historical daily selector artifacts and
selects lambdas fold-by-fold using only prior data.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_engine.policy_tuning import build_default_shadow_strategies
from post_process_market_plays import compute_final_board
from validate_board_objective_mode import (
    _build_actual_lookup,
    _build_data_proc_actual_lookup,
    _build_rows_actual_lookup,
    _iter_run_dates,
    _normalize_player_key,
    _resolve_result,
)


POLICY_PROFILES = {config.name: config for config in build_default_shadow_strategies()}


def _binary_log_loss(prob: np.ndarray, label: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype="float64"), 1e-6, 1.0 - 1e-6)
    y = np.asarray(label, dtype="float64")
    if len(p) == 0:
        return np.nan
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _ece_10(prob: np.ndarray, label: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype="float64"), 0.0, 1.0)
    y = np.asarray(label, dtype="float64")
    if len(p) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(p, edges[1:-1], right=False)
    n = max(1, len(p))
    ece = 0.0
    for b in range(10):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (float(np.sum(mask)) / n) * abs(acc - conf)
    return float(ece)


def _parse_grid(text: str) -> list[float]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"Grid has no values: '{text}'")
    return sorted(set(values))


def _load_selected_board_calibrator(path: Path, disabled: bool) -> dict | None:
    if disabled:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None


def _policy_kwargs(payload: dict, lambda_corr: float, lambda_conc: float, lambda_unc: float) -> dict:
    return {
        "american_odds": payload["american_odds"],
        "min_ev": payload["min_ev"],
        "min_final_confidence": payload["min_final_confidence"],
        "min_recommendation": payload["min_recommendation"],
        "selection_mode": "board_objective",
        "ranking_mode": "board_objective",
        "max_plays_per_player": payload["max_plays_per_player"],
        "max_plays_per_target": payload["max_plays_per_target"],
        "max_total_plays": payload["max_total_plays"],
        "min_board_plays": payload.get("min_board_plays", 0),
        "max_target_plays": {"PTS": payload["max_pts_plays"], "TRB": payload["max_trb_plays"], "AST": payload["max_ast_plays"]},
        "max_plays_per_game": payload.get("max_plays_per_game", 2),
        "max_plays_per_script_cluster": payload.get("max_plays_per_script_cluster", 2),
        "non_pts_min_gap_percentile": payload["non_pts_min_gap_percentile"],
        "edge_adjust_k": payload["edge_adjust_k"],
        "thompson_temperature": payload.get("thompson_temperature", 1.0),
        "thompson_seed": payload.get("thompson_seed", 17),
        "min_bet_win_rate": payload.get("min_bet_win_rate", 0.49),
        "medium_bet_win_rate": payload.get("medium_bet_win_rate", 0.52),
        "full_bet_win_rate": payload.get("full_bet_win_rate", 0.56),
        "medium_tier_percentile": payload.get("medium_tier_percentile", 0.0),
        "strong_tier_percentile": payload.get("strong_tier_percentile", 0.0),
        "elite_tier_percentile": payload.get("elite_tier_percentile", 0.0),
        "small_bet_fraction": payload.get("small_bet_fraction", 0.005),
        "medium_bet_fraction": payload.get("medium_bet_fraction", 0.010),
        "full_bet_fraction": payload.get("full_bet_fraction", 0.015),
        "max_bet_fraction": payload.get("max_bet_fraction", 0.02),
        "max_total_bet_fraction": payload.get("max_total_bet_fraction", 0.06),
        "belief_uncertainty_lower": payload.get("belief_uncertainty_lower", 0.75),
        "belief_uncertainty_upper": payload.get("belief_uncertainty_upper", 1.15),
        "append_agreement_min": payload.get("append_agreement_min", 3),
        "append_edge_percentile_min": payload.get("append_edge_percentile_min", 0.90),
        "append_max_extra_plays": payload.get("append_max_extra_plays", 3),
        "board_objective_overfetch": payload.get("board_objective_overfetch", 4.0),
        "board_objective_candidate_limit": payload.get("board_objective_candidate_limit", 36),
        "board_objective_max_search_nodes": payload.get("board_objective_max_search_nodes", 750000),
        "board_objective_lambda_corr": float(lambda_corr),
        "board_objective_lambda_conc": float(lambda_conc),
        "board_objective_lambda_unc": float(lambda_unc),
        "board_objective_corr_same_game": payload.get("board_objective_corr_same_game", 0.65),
        "board_objective_corr_same_player": payload.get("board_objective_corr_same_player", 1.0),
        "board_objective_corr_same_target": payload.get("board_objective_corr_same_target", 0.15),
        "board_objective_corr_same_direction": payload.get("board_objective_corr_same_direction", 0.05),
        "board_objective_corr_same_script_cluster": payload.get("board_objective_corr_same_script_cluster", 0.30),
        "board_objective_swap_candidates": payload.get("board_objective_swap_candidates", 18),
        "board_objective_swap_rounds": payload.get("board_objective_swap_rounds", 2),
        "max_history_staleness_days": payload.get("max_history_staleness_days", 0),
        "min_recency_factor": payload.get("min_recency_factor", 0.0),
    }


def _evaluate_one_day(
    selector_df: pd.DataFrame,
    run_date_token: str,
    kwargs: dict,
    data_proc_lookup: dict,
    history_lookup: dict,
    rows_actual_lookup: dict,
    threshold_clear_count: int,
    selected_board_calibrator: dict | None,
) -> dict:
    local_kwargs = dict(kwargs)
    if selected_board_calibrator is not None:
        run_month = pd.to_datetime(run_date_token, format="%Y%m%d", errors="coerce")
        month_hint = run_month.strftime("%Y-%m") if pd.notna(run_month) else datetime.utcnow().strftime("%Y-%m")
        local_kwargs["selected_board_calibrator"] = selected_board_calibrator
        local_kwargs["selected_board_calibration_month"] = month_hint
    board = compute_final_board(selector_df.copy(), **local_kwargs)
    requested_size = int(local_kwargs.get("max_total_plays", 0))
    deficit_flag = bool(requested_size > 0 and len(board) < requested_size)
    if board.empty:
        return {
            "rows": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "missing": 0,
            "hit_rate": np.nan,
            "avg_ev": np.nan,
            "avg_expected_win_rate": np.nan,
            "avg_expected_resolved": np.nan,
            "calibration_gap_pp": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "ece_10": np.nan,
            "under_share": np.nan,
            "swap_rate": np.nan,
            "solver_truncated_rate": np.nan,
            "threshold_clear": np.nan,
            "deficit_fill_flag": float(deficit_flag),
            "requested_board_size": requested_size,
            "actual_board_size": 0,
        }

    run_date_key = pd.to_datetime(run_date_token, format="%Y%m%d", errors="coerce")
    run_date_key = run_date_key.strftime("%Y-%m-%d") if pd.notna(run_date_key) else str(run_date_token)

    resolved_probs: list[float] = []
    resolved_labels: list[float] = []
    results = []
    for _, row in board.iterrows():
        market_date = pd.to_datetime(row.get("market_date"), errors="coerce")
        market_date_key = market_date.strftime("%Y-%m-%d") if pd.notna(market_date) else ""
        player = str(row.get("player", "")).strip()
        player_norm = _normalize_player_key(player)
        target = str(row.get("target", "")).strip().upper()
        direction = str(row.get("direction", "")).strip().upper()
        line = pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").iloc[0]
        rounded_line = float(np.round(line, 6)) if pd.notna(line) else np.nan

        actual = np.nan
        for player_key in (player, player_norm):
            if not player_key:
                continue
            lookup_key = (market_date_key, str(player_key), target)
            if lookup_key in data_proc_lookup:
                actual = data_proc_lookup[lookup_key]
                break
            if lookup_key in history_lookup:
                actual = history_lookup[lookup_key]
                break
        if pd.isna(actual) and rounded_line == rounded_line:
            fallback_key = (run_date_key, player, target, direction, rounded_line)
            actual = rows_actual_lookup.get(fallback_key, np.nan)

        result = _resolve_result(direction, float(line) if pd.notna(line) else np.nan, actual)
        results.append(result)
        if result in {"win", "loss"}:
            resolved_labels.append(1.0 if result == "win" else 0.0)
            resolved_probs.append(
                float(
                    np.clip(
                        pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").fillna(0.5).iloc[0],
                        0.0,
                        1.0,
                    )
                )
            )

    result_series = pd.Series(results, dtype=str)
    resolved_mask = result_series.isin(["win", "loss"])
    resolved_count = int(resolved_mask.sum())
    wins = int((result_series == "win").sum())
    losses = int((result_series == "loss").sum())
    pushes = int((result_series == "push").sum())
    missing = int((result_series == "missing").sum())

    threshold_hits = int((result_series == "win").sum())
    threshold_clear = float(threshold_hits >= max(1, int(threshold_clear_count)))
    if resolved_probs:
        rp = np.asarray(resolved_probs, dtype="float64")
        rl = np.asarray(resolved_labels, dtype="float64")
        avg_expected_resolved = float(np.mean(rp))
        calibration_gap_pp = float((float(np.mean(rl)) - avg_expected_resolved) * 100.0)
        brier = float(np.mean((rp - rl) ** 2))
        log_loss = _binary_log_loss(rp, rl)
        ece_10 = _ece_10(rp, rl)
    else:
        avg_expected_resolved = np.nan
        calibration_gap_pp = np.nan
        brier = np.nan
        log_loss = np.nan
        ece_10 = np.nan

    return {
        "rows": int(len(board)),
        "resolved": resolved_count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "missing": missing,
        "hit_rate": float(wins / resolved_count) if resolved_count > 0 else np.nan,
        "avg_ev": float(pd.to_numeric(board.get("ev"), errors="coerce").mean()),
        "avg_expected_win_rate": float(pd.to_numeric(board.get("expected_win_rate"), errors="coerce").mean()),
        "avg_expected_resolved": avg_expected_resolved,
        "calibration_gap_pp": calibration_gap_pp,
        "brier": brier,
        "log_loss": log_loss,
        "ece_10": ece_10,
        "under_share": float(board["direction"].astype(str).str.upper().eq("UNDER").mean()),
        "swap_rate": float(pd.to_numeric(board.get("board_objective_swap_applied"), errors="coerce").fillna(0).astype(bool).mean())
        if "board_objective_swap_applied" in board.columns
        else np.nan,
        "solver_truncated_rate": float(pd.to_numeric(board.get("board_objective_search_truncated"), errors="coerce").fillna(0).astype(bool).mean())
        if "board_objective_search_truncated" in board.columns
        else np.nan,
        "threshold_clear": threshold_clear,
        "deficit_fill_flag": float(deficit_flag),
        "requested_board_size": requested_size,
        "actual_board_size": int(len(board)),
    }


def _aggregate_days(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "days": 0,
            "days_with_resolved": 0,
            "rows": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "missing": 0,
            "hit_rate": np.nan,
            "mean_daily_hit_rate": np.nan,
            "avg_ev": np.nan,
            "avg_expected_win_rate": np.nan,
            "avg_expected_resolved": np.nan,
            "mean_calibration_gap_pp": np.nan,
            "mean_abs_calibration_gap_pp": np.nan,
            "mean_brier": np.nan,
            "mean_log_loss": np.nan,
            "mean_ece_10": np.nan,
            "mean_under_share": np.nan,
            "mean_swap_rate": np.nan,
            "mean_solver_truncated_rate": np.nan,
            "threshold_clear_rate": np.nan,
            "deficit_fill_rate": np.nan,
            "mean_board_size": np.nan,
        }
    resolved = int(df["resolved"].sum())
    wins = int(df["wins"].sum())
    losses = int(df["losses"].sum())
    pushes = int(df["pushes"].sum())
    missing = int(df["missing"].sum())
    return {
        "days": int(df["run_date"].nunique()),
        "days_with_resolved": int(df.loc[df["resolved"] > 0, "run_date"].nunique()),
        "rows": int(df["rows"].sum()),
        "resolved": resolved,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "missing": missing,
        "hit_rate": float(wins / resolved) if resolved > 0 else np.nan,
        "mean_daily_hit_rate": float(df.loc[df["resolved"] > 0, "hit_rate"].mean()),
        "avg_ev": float(df["avg_ev"].mean()),
        "avg_expected_win_rate": float(df["avg_expected_win_rate"].mean()),
        "avg_expected_resolved": float(df.loc[df["resolved"] > 0, "avg_expected_resolved"].mean()),
        "mean_calibration_gap_pp": float(df.loc[df["resolved"] > 0, "calibration_gap_pp"].mean()),
        "mean_abs_calibration_gap_pp": float(np.abs(df.loc[df["resolved"] > 0, "calibration_gap_pp"]).mean()),
        "mean_brier": float(df.loc[df["resolved"] > 0, "brier"].mean()),
        "mean_log_loss": float(df.loc[df["resolved"] > 0, "log_loss"].mean()),
        "mean_ece_10": float(df.loc[df["resolved"] > 0, "ece_10"].mean()),
        "mean_under_share": float(df["under_share"].mean()),
        "mean_swap_rate": float(df["swap_rate"].mean()),
        "mean_solver_truncated_rate": float(df["solver_truncated_rate"].mean()),
        "threshold_clear_rate": float(df["threshold_clear"].mean()),
        "deficit_fill_rate": float(df["deficit_fill_flag"].mean()),
        "mean_board_size": float(df["actual_board_size"].mean()),
    }


def _passes_guardrails(agg: dict, args: argparse.Namespace) -> bool:
    checks = [
        float(agg.get("mean_abs_calibration_gap_pp", np.inf)) <= float(args.guardrail_max_abs_calibration_gap_pp),
        float(agg.get("mean_ece_10", np.inf)) <= float(args.guardrail_max_mean_ece_10),
        float(agg.get("deficit_fill_rate", np.inf)) <= float(args.guardrail_max_deficit_fill_rate),
        float(agg.get("threshold_clear_rate", -np.inf)) >= float(args.guardrail_min_threshold_clear_rate),
    ]
    return bool(all(checks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward tune lambdas for board-objective mode.")
    parser.add_argument("--start-run-date", type=str, default="20260101", help="Inclusive run-date start (YYYYMMDD).")
    parser.add_argument("--end-run-date", type=str, default="20260331", help="Inclusive run-date end (YYYYMMDD).")
    parser.add_argument("--policy-profile", type=str, default="production_board_objective_b12", choices=sorted(POLICY_PROFILES.keys()))
    parser.add_argument("--daily-runs-dir", type=Path, default=REPO_ROOT / "model" / "analysis" / "daily_runs")
    parser.add_argument("--history-csv", type=Path, default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv")
    parser.add_argument("--data-proc-root", type=Path, default=REPO_ROOT / "Data-Proc")
    parser.add_argument("--actual-rows-csv", type=Path, default=None, help="Optional long-format fallback rows CSV for actuals.")
    parser.add_argument("--max-days", type=int, default=0, help="Optional cap on replayed run dates (0 disables).")
    parser.add_argument("--train-days", type=int, default=35, help="Walk-forward train window length in days.")
    parser.add_argument("--test-days", type=int, default=14, help="Walk-forward test window length in days.")
    parser.add_argument("--step-days", type=int, default=7, help="Walk-forward step size in days.")
    parser.add_argument(
        "--threshold-clear-count",
        type=int,
        default=None,
        help="Board-level success threshold used for diagnostics (defaults to policy min_board_plays when >0, else board size).",
    )
    parser.add_argument("--lambda-corr-grid", type=str, default="0.10,0.14")
    parser.add_argument("--lambda-conc-grid", type=str, default="0.05,0.09")
    parser.add_argument("--lambda-unc-grid", type=str, default="0.05")
    parser.add_argument("--board-objective-overfetch", type=float, default=None)
    parser.add_argument("--board-objective-candidate-limit", type=int, default=None)
    parser.add_argument("--board-objective-max-search-nodes", type=int, default=None)
    parser.add_argument(
        "--selected-board-calibrator-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator.json",
        help="Optional selected-board calibrator payload JSON used during replay scoring.",
    )
    parser.add_argument(
        "--disable-selected-board-calibration",
        action="store_true",
        help="Disable selected-board calibration while tuning lambdas.",
    )
    parser.add_argument("--guardrail-max-abs-calibration-gap-pp", type=float, default=5.0)
    parser.add_argument("--guardrail-max-mean-ece-10", type=float, default=0.15)
    parser.add_argument("--guardrail-max-deficit-fill-rate", type=float, default=0.35)
    parser.add_argument("--guardrail-min-threshold-clear-rate", type=float, default=0.55)
    parser.add_argument("--rows-csv-out", type=Path, default=None, help="Per-day per-combo output CSV.")
    parser.add_argument("--combo-summary-csv-out", type=Path, default=None, help="Combo aggregate summary CSV.")
    parser.add_argument("--fold-summary-csv-out", type=Path, default=None, help="Walk-forward fold summary CSV.")
    parser.add_argument("--summary-json-out", type=Path, default=None, help="Overall JSON summary output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_policy = POLICY_PROFILES[args.policy_profile].to_dict()
    if args.board_objective_overfetch is not None:
        base_policy["board_objective_overfetch"] = float(args.board_objective_overfetch)
    if args.board_objective_candidate_limit is not None:
        base_policy["board_objective_candidate_limit"] = int(args.board_objective_candidate_limit)
    if args.board_objective_max_search_nodes is not None:
        base_policy["board_objective_max_search_nodes"] = int(args.board_objective_max_search_nodes)
    selected_board_calibrator = _load_selected_board_calibrator(
        args.selected_board_calibrator_json,
        disabled=bool(args.disable_selected_board_calibration),
    )

    corr_grid = _parse_grid(args.lambda_corr_grid)
    conc_grid = _parse_grid(args.lambda_conc_grid)
    unc_grid = _parse_grid(args.lambda_unc_grid)
    combos = list(itertools.product(corr_grid, conc_grid, unc_grid))
    if not combos:
        raise RuntimeError("No lambda combos configured.")

    run_dates = _iter_run_dates(args.daily_runs_dir.resolve(), args.start_run_date, args.end_run_date, args.max_days)
    if not run_dates:
        raise RuntimeError("No run dates found in the requested window.")

    selector_map: dict[str, pd.DataFrame] = {}
    for run_date in run_dates:
        selector_csv = args.daily_runs_dir / run_date / f"upcoming_market_play_selector_{run_date}.csv"
        selector_map[run_date] = pd.read_csv(selector_csv)

    data_proc_lookup = _build_data_proc_actual_lookup(args.data_proc_root, args.start_run_date, args.end_run_date)
    history_lookup = _build_actual_lookup(args.history_csv.resolve())
    fallback_rows_csv = args.actual_rows_csv
    if fallback_rows_csv is None:
        candidate = args.daily_runs_dir / f"mode_compare_edge_absedge_evadj_daily_rows_{args.start_run_date}_{args.end_run_date}.csv"
        fallback_rows_csv = candidate if candidate.exists() else None
    rows_actual_lookup = _build_rows_actual_lookup(fallback_rows_csv)

    threshold_clear_count = args.threshold_clear_count
    if threshold_clear_count is None:
        policy_min_board = int(base_policy.get("min_board_plays", 0))
        policy_max_board = int(base_policy.get("max_total_plays", 0))
        threshold_clear_count = policy_min_board if policy_min_board > 0 else max(policy_max_board, 1)

    mode_rows = []
    for corr, conc, unc in combos:
        kwargs = _policy_kwargs(base_policy, lambda_corr=corr, lambda_conc=conc, lambda_unc=unc)
        combo_id = f"corr={corr:.4f}|conc={conc:.4f}|unc={unc:.4f}"
        print(f"Evaluating combo {combo_id} across {len(run_dates)} run dates...")
        for run_date in run_dates:
            day_metrics = _evaluate_one_day(
                selector_map[run_date],
                run_date_token=run_date,
                kwargs=kwargs,
                data_proc_lookup=data_proc_lookup,
                history_lookup=history_lookup,
                rows_actual_lookup=rows_actual_lookup,
                threshold_clear_count=threshold_clear_count,
                selected_board_calibrator=selected_board_calibrator,
            )
            mode_rows.append(
                {
                    "run_date": run_date,
                    "combo_id": combo_id,
                    "lambda_corr": float(corr),
                    "lambda_conc": float(conc),
                    "lambda_unc": float(unc),
                    **day_metrics,
                }
            )

    day_df = pd.DataFrame.from_records(mode_rows)
    if day_df.empty:
        raise RuntimeError("No day-level rows produced.")

    combo_summary_rows = []
    for combo_id, part in day_df.groupby("combo_id"):
        agg = _aggregate_days(part)
        first = part.iloc[0]
        guardrail_pass = _passes_guardrails(agg, args)
        combo_summary_rows.append(
            {
                "combo_id": combo_id,
                "lambda_corr": float(first["lambda_corr"]),
                "lambda_conc": float(first["lambda_conc"]),
                "lambda_unc": float(first["lambda_unc"]),
                "guardrail_pass": bool(guardrail_pass),
                **agg,
            }
        )
    combo_summary_df = pd.DataFrame.from_records(combo_summary_rows).sort_values(
        ["guardrail_pass", "mean_daily_hit_rate", "hit_rate", "avg_ev"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    fold_rows = []
    train_days = max(1, int(args.train_days))
    test_days = max(1, int(args.test_days))
    step_days = max(1, int(args.step_days))
    n = len(run_dates)
    fold_id = 0
    for start in range(train_days, n - test_days + 1, step_days):
        fold_id += 1
        train_set = set(run_dates[start - train_days : start])
        test_set = set(run_dates[start : start + test_days])

        if not train_set or not test_set:
            continue

        best_row = None
        guardrail_candidates: list[dict] = []
        fallback_candidates: list[dict] = []
        for combo_id, part in day_df.groupby("combo_id"):
            train_part = part.loc[part["run_date"].isin(train_set)].copy()
            train_agg = _aggregate_days(train_part)
            score = train_agg["mean_daily_hit_rate"]
            ev = train_agg["avg_ev"]
            first = train_part.iloc[0]
            candidate_row = {
                "combo_id": combo_id,
                "lambda_corr": float(first["lambda_corr"]),
                "lambda_conc": float(first["lambda_conc"]),
                "lambda_unc": float(first["lambda_unc"]),
                "train_mean_daily_hit_rate": float(score),
                "train_hit_rate": float(train_agg["hit_rate"]),
                "train_avg_ev": float(ev),
                "train_mean_abs_calibration_gap_pp": float(train_agg["mean_abs_calibration_gap_pp"]),
                "train_mean_ece_10": float(train_agg["mean_ece_10"]),
                "train_deficit_fill_rate": float(train_agg["deficit_fill_rate"]),
                "train_threshold_clear_rate": float(train_agg["threshold_clear_rate"]),
                "train_guardrail_pass": bool(_passes_guardrails(train_agg, args)),
            }
            fallback_candidates.append(candidate_row)
            if candidate_row["train_guardrail_pass"]:
                guardrail_candidates.append(candidate_row)
        pool = guardrail_candidates if guardrail_candidates else fallback_candidates
        if pool:
            pool_sorted = sorted(
                pool,
                key=lambda r: (
                    float(r["train_mean_daily_hit_rate"]),
                    float(r["train_hit_rate"]),
                    float(r["train_avg_ev"]),
                ),
                reverse=True,
            )
            best_row = pool_sorted[0]
        if best_row is None:
            continue

        combo_day = day_df.loc[day_df["combo_id"] == best_row["combo_id"]].copy()
        test_part = combo_day.loc[combo_day["run_date"].isin(test_set)].copy()
        test_agg = _aggregate_days(test_part)
        fold_rows.append(
            {
                "fold": int(fold_id),
                "train_start": run_dates[start - train_days],
                "train_end": run_dates[start - 1],
                "test_start": run_dates[start],
                "test_end": run_dates[start + test_days - 1],
                **best_row,
                "test_mean_daily_hit_rate": float(test_agg["mean_daily_hit_rate"]),
                "test_hit_rate": float(test_agg["hit_rate"]),
                "test_avg_ev": float(test_agg["avg_ev"]),
                "test_threshold_clear_rate": float(test_agg["threshold_clear_rate"]),
                "test_mean_abs_calibration_gap_pp": float(test_agg["mean_abs_calibration_gap_pp"]),
                "test_mean_ece_10": float(test_agg["mean_ece_10"]),
                "test_deficit_fill_rate": float(test_agg["deficit_fill_rate"]),
                "test_guardrail_pass": bool(_passes_guardrails(test_agg, args)),
                "test_days": int(test_agg["days"]),
                "test_resolved": int(test_agg["resolved"]),
                "test_wins": int(test_agg["wins"]),
                "test_losses": int(test_agg["losses"]),
                "test_pushes": int(test_agg["pushes"]),
                "test_missing": int(test_agg["missing"]),
            }
        )

    fold_df = pd.DataFrame.from_records(fold_rows)
    if fold_df.empty:
        raise RuntimeError(
            "No walk-forward folds produced. Increase window length or reduce train/test span."
        )

    latest_fold = fold_df.sort_values("fold").iloc[-1]
    wf_combo_summary = (
        fold_df.groupby(["combo_id", "lambda_corr", "lambda_conc", "lambda_unc"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_test_daily_hit_rate=("test_mean_daily_hit_rate", "mean"),
            mean_test_hit_rate=("test_hit_rate", "mean"),
            mean_test_avg_ev=("test_avg_ev", "mean"),
            mean_test_threshold_clear_rate=("test_threshold_clear_rate", "mean"),
            mean_test_abs_calibration_gap_pp=("test_mean_abs_calibration_gap_pp", "mean"),
            mean_test_ece_10=("test_mean_ece_10", "mean"),
            mean_test_deficit_fill_rate=("test_deficit_fill_rate", "mean"),
            fold_guardrail_pass_rate=("test_guardrail_pass", "mean"),
        )
        .sort_values(
            ["fold_guardrail_pass_rate", "mean_test_daily_hit_rate", "mean_test_hit_rate", "mean_test_avg_ev"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    best_wf = wf_combo_summary.iloc[0]

    token = f"{args.start_run_date}_{args.end_run_date}"
    rows_out = args.rows_csv_out or (args.daily_runs_dir / f"board_objective_lambda_tuning_daily_{token}.csv")
    combo_out = args.combo_summary_csv_out or (args.daily_runs_dir / f"board_objective_lambda_tuning_combo_summary_{token}.csv")
    fold_out = args.fold_summary_csv_out or (args.daily_runs_dir / f"board_objective_lambda_tuning_walk_forward_{token}.csv")
    summary_out = args.summary_json_out or (args.daily_runs_dir / f"board_objective_lambda_tuning_summary_{token}.json")

    rows_out.parent.mkdir(parents=True, exist_ok=True)
    combo_out.parent.mkdir(parents=True, exist_ok=True)
    fold_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    day_df.to_csv(rows_out, index=False)
    combo_summary_df.to_csv(combo_out, index=False)
    fold_df.to_csv(fold_out, index=False)

    payload = {
        "window": {"start_run_date": args.start_run_date, "end_run_date": args.end_run_date},
        "policy_profile_base": args.policy_profile,
        "run_dates": int(len(run_dates)),
        "combos_evaluated": int(len(combos)),
        "grids": {
            "lambda_corr": corr_grid,
            "lambda_conc": conc_grid,
            "lambda_unc": unc_grid,
        },
        "selected_board_calibrator": {
            "enabled": bool(selected_board_calibrator is not None),
            "path": str(args.selected_board_calibrator_json.resolve()),
        },
        "guardrails": {
            "max_abs_calibration_gap_pp": float(args.guardrail_max_abs_calibration_gap_pp),
            "max_mean_ece_10": float(args.guardrail_max_mean_ece_10),
            "max_deficit_fill_rate": float(args.guardrail_max_deficit_fill_rate),
            "min_threshold_clear_rate": float(args.guardrail_min_threshold_clear_rate),
        },
        "walk_forward": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "folds": int(len(fold_df)),
            "best_combo_by_mean_test_daily_hit_rate": {
                "combo_id": str(best_wf["combo_id"]),
                "lambda_corr": float(best_wf["lambda_corr"]),
                "lambda_conc": float(best_wf["lambda_conc"]),
                "lambda_unc": float(best_wf["lambda_unc"]),
                "mean_test_daily_hit_rate": float(best_wf["mean_test_daily_hit_rate"]),
                "mean_test_hit_rate": float(best_wf["mean_test_hit_rate"]),
                "mean_test_avg_ev": float(best_wf["mean_test_avg_ev"]),
                "mean_test_threshold_clear_rate": float(best_wf["mean_test_threshold_clear_rate"]),
                "mean_test_abs_calibration_gap_pp": float(best_wf["mean_test_abs_calibration_gap_pp"]),
                "mean_test_ece_10": float(best_wf["mean_test_ece_10"]),
                "mean_test_deficit_fill_rate": float(best_wf["mean_test_deficit_fill_rate"]),
                "fold_guardrail_pass_rate": float(best_wf["fold_guardrail_pass_rate"]),
            },
            "latest_fold_selected_combo": {
                "fold": int(latest_fold["fold"]),
                "train_start": str(latest_fold["train_start"]),
                "train_end": str(latest_fold["train_end"]),
                "test_start": str(latest_fold["test_start"]),
                "test_end": str(latest_fold["test_end"]),
                "combo_id": str(latest_fold["combo_id"]),
                "lambda_corr": float(latest_fold["lambda_corr"]),
                "lambda_conc": float(latest_fold["lambda_conc"]),
                "lambda_unc": float(latest_fold["lambda_unc"]),
                "test_mean_daily_hit_rate": float(latest_fold["test_mean_daily_hit_rate"]),
                "test_hit_rate": float(latest_fold["test_hit_rate"]),
                "test_avg_ev": float(latest_fold["test_avg_ev"]),
                "test_threshold_clear_rate": float(latest_fold["test_threshold_clear_rate"]),
                "test_mean_abs_calibration_gap_pp": float(latest_fold["test_mean_abs_calibration_gap_pp"]),
                "test_mean_ece_10": float(latest_fold["test_mean_ece_10"]),
                "test_deficit_fill_rate": float(latest_fold["test_deficit_fill_rate"]),
                "test_guardrail_pass": bool(latest_fold["test_guardrail_pass"]),
            },
        },
        "outputs": {
            "daily_rows_csv": str(rows_out),
            "combo_summary_csv": str(combo_out),
            "walk_forward_csv": str(fold_out),
        },
    }
    summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Daily rows CSV:        {rows_out}")
    print(f"Combo summary CSV:     {combo_out}")
    print(f"Walk-forward CSV:      {fold_out}")
    print(f"Summary JSON:          {summary_out}")
    print("\nTop combos (overall):")
    print(combo_summary_df.head(10).to_string(index=False))
    print("\nWalk-forward best by mean test daily hit rate:")
    print(wf_combo_summary.head(10).to_string(index=False))
    print("\nLatest fold selected combo:")
    print(fold_df.sort_values('fold').tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
