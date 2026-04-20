#!/usr/bin/env python3
"""
Paired replay backtest for the line-decision sidecar.

This script replays saved daily run slates twice:
1. baseline selector logic with the sidecar disabled
2. treatment selector logic with the sidecar enabled

It then rebuilds final boards, resolves outcomes from historical actuals,
and reports whether the sidecar improved decision-layer accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_engine.conditional_promotion import apply_conditional_promotion
from decision_engine.line_decision import LineDecisionConfig, build_line_decision_lookup
from post_process_market_plays import compute_final_board
from run_market_pipeline import (
    apply_live_policy_calibration,
    load_accepted_pick_gate,
    load_learned_pool_gate,
    load_selected_board_calibrator,
    load_staking_bucket_model,
    maybe_apply_accept_rejector,
    maybe_apply_robust_reranker,
    maybe_apply_xgb_ltr_reranker,
)
from select_market_plays import build_history_lookup, build_play_rows
from validate_board_objective_mode import (
    _build_actual_lookup,
    _build_data_proc_actual_lookup,
    _iter_run_dates,
    _lookup_actual_with_date_fallback,
    _player_key_variants,
    _resolve_result,
)


DEFAULT_START = "20260301"
DEFAULT_END = "20260331"
PAYOUT_MINUS_110 = 100.0 / 110.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired replay backtest for the line-decision sidecar.")
    parser.add_argument("--start-run-date", type=str, default=DEFAULT_START, help="Inclusive start run date (YYYYMMDD).")
    parser.add_argument("--end-run-date", type=str, default=DEFAULT_END, help="Inclusive end run date (YYYYMMDD).")
    parser.add_argument(
        "--daily-runs-dir",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs",
        help="Directory containing dated daily run folders.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv",
        help="Historical selector/backtest CSV used for calibration features.",
    )
    parser.add_argument(
        "--data-proc-root",
        type=Path,
        default=REPO_ROOT / "Data-Proc",
        help="Processed player data root used for actual outcome lookup.",
    )
    parser.add_argument("--max-days", type=int, default=0, help="Optional cap on replayed days (0 disables).")
    parser.add_argument(
        "--selected-board-calibrator-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator.json",
    )
    parser.add_argument(
        "--learned-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "learned_pool_gate.json",
    )
    parser.add_argument(
        "--accepted-pick-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "accepted_pick_gate" / "candidates" / "accepted_pick_gate_candidate.json",
    )
    parser.add_argument(
        "--staking-bucket-model-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "staking_bucket_model_v2.json",
    )
    parser.add_argument("--rows-csv-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "line_decision_sidecar_backtest_rows.csv")
    parser.add_argument("--summary-json-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "line_decision_sidecar_backtest_summary.json")
    parser.add_argument("--summary-csv-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "line_decision_sidecar_backtest_summary.csv")
    parser.add_argument("--no-trade-threshold", type=float, default=LineDecisionConfig().no_trade_threshold)
    parser.add_argument("--min-trade-prob", type=float, default=LineDecisionConfig().min_trade_prob)
    parser.add_argument("--min-trade-prob-gap", type=float, default=LineDecisionConfig().min_trade_prob_gap)
    return parser.parse_args()


def _canonical_policy_payload(raw_policy: dict[str, Any]) -> dict[str, Any]:
    policy = dict(raw_policy or {})
    if "ranking_mode" not in policy and "selection_mode" in policy:
        policy["ranking_mode"] = policy["selection_mode"]
    if "selection_mode" not in policy and "ranking_mode" in policy:
        policy["selection_mode"] = policy["ranking_mode"]
    return policy


def _load_daily_policy(run_dir: Path) -> dict[str, Any]:
    token = run_dir.name
    final_json = run_dir / f"final_market_plays_{token}.json"
    if not final_json.exists():
        raise FileNotFoundError(f"Missing final board JSON for {token}: {final_json}")
    payload = json.loads(final_json.read_text(encoding="utf-8"))
    return _canonical_policy_payload(payload.get("policy", {}))


def _prepare_selector(
    slate_df: pd.DataFrame,
    history_df: pd.DataFrame,
    history_lookup: dict[str, dict],
    line_decision_lookup: dict[str, dict],
    policy_payload: dict[str, Any],
    *,
    line_decision_enabled: bool,
    line_decision_config: LineDecisionConfig,
) -> pd.DataFrame:
    selector_df = build_play_rows(
        slate_df,
        history_lookup,
        line_decision_lookup=line_decision_lookup,
        belief_uncertainty_lower=float(policy_payload.get("belief_uncertainty_lower", 0.75)),
        belief_uncertainty_upper=float(policy_payload.get("belief_uncertainty_upper", 1.15)),
        market_regression_floor=float(policy_payload.get("market_regression_floor", 0.25)),
        market_regression_ceiling=float(policy_payload.get("market_regression_ceiling", 0.95)),
        line_decision_enabled=bool(line_decision_enabled),
        line_decision_config=line_decision_config,
    )
    selector_df = apply_live_policy_calibration(selector_df, policy_payload)
    selector_df, _ = maybe_apply_xgb_ltr_reranker(selector_df, history_df, policy_payload)
    selector_df, _ = maybe_apply_accept_rejector(selector_df, history_df, policy_payload)
    selector_df, _ = maybe_apply_robust_reranker(selector_df, history_df, policy_payload)
    try:
        selector_df, _ = apply_conditional_promotion(
            selector_df=selector_df,
            policy_payload=policy_payload,
            history_df=history_df,
            american_odds=int(policy_payload.get("american_odds", -110)),
        )
    except Exception:
        selector_df = selector_df.copy()
        selector_df["conditional_eligible_for_board"] = True
        selector_df["conditional_promoted"] = False
        selector_df["decision_tier"] = "Tier A - Baseline"
        selector_df["weak_bucket"] = "K"
        selector_df["conditional_audit_summary"] = "Conditional layer failed; fell back to baseline-only mode."
    return selector_df


def _compute_board(
    selector_df: pd.DataFrame,
    policy_payload: dict[str, Any],
    selected_board_calibrator: dict | None,
    learned_pool_gate: dict | None,
    accepted_pick_gate: dict | None,
    staking_bucket_model: dict | None,
    run_month: str | None,
) -> pd.DataFrame:
    return compute_final_board(
        selector_df,
        american_odds=int(policy_payload.get("american_odds", -110)),
        min_ev=float(policy_payload.get("min_ev", 0.0)),
        min_final_confidence=float(policy_payload.get("min_final_confidence", 0.02)),
        min_recommendation=str(policy_payload.get("min_recommendation", "consider")),
        selection_mode=str(policy_payload.get("selection_mode", policy_payload.get("ranking_mode", "ev_adjusted"))),
        ranking_mode=str(policy_payload.get("ranking_mode", "ev_adjusted")),
        max_plays_per_player=int(policy_payload.get("max_plays_per_player", 1)),
        max_plays_per_target=int(policy_payload.get("max_plays_per_target", 8)),
        max_total_plays=int(policy_payload.get("max_total_plays", 20)),
        min_board_plays=int(policy_payload.get("min_board_plays", 0)),
        max_target_plays={
            "PTS": int(policy_payload.get("max_pts_plays", 8)),
            "TRB": int(policy_payload.get("max_trb_plays", 4)),
            "AST": int(policy_payload.get("max_ast_plays", 4)),
        },
        max_plays_per_game=int(policy_payload.get("max_plays_per_game", 2)),
        max_plays_per_script_cluster=int(policy_payload.get("max_plays_per_script_cluster", 2)),
        non_pts_min_gap_percentile=float(policy_payload.get("non_pts_min_gap_percentile", 0.90)),
        exclude_micro_lines_enabled=bool(policy_payload.get("exclude_micro_lines_enabled", True)),
        exclude_micro_line_targets=tuple(
            str(token).upper().strip()
            for token in policy_payload.get("exclude_micro_line_targets", ("PTS", "TRB", "AST"))
            if str(token).strip()
        ),
        exclude_micro_line_min=float(policy_payload.get("exclude_micro_line_min", 0.5)),
        exclude_micro_line_max=float(policy_payload.get("exclude_micro_line_max", 1.5)),
        edge_adjust_k=float(policy_payload.get("edge_adjust_k", 0.30)),
        thompson_temperature=float(policy_payload.get("thompson_temperature", 1.0)),
        thompson_seed=int(policy_payload.get("thompson_seed", 17)),
        min_bet_win_rate=float(policy_payload.get("min_bet_win_rate", 0.57)),
        medium_bet_win_rate=float(policy_payload.get("medium_bet_win_rate", 0.60)),
        full_bet_win_rate=float(policy_payload.get("full_bet_win_rate", 0.65)),
        medium_tier_percentile=float(policy_payload.get("medium_tier_percentile", 0.80)),
        strong_tier_percentile=float(policy_payload.get("strong_tier_percentile", 0.90)),
        elite_tier_percentile=float(policy_payload.get("elite_tier_percentile", policy_payload.get("elite_pct", 0.95))),
        small_bet_fraction=float(policy_payload.get("small_bet_fraction", 0.005)),
        medium_bet_fraction=float(policy_payload.get("medium_bet_fraction", 0.010)),
        full_bet_fraction=float(policy_payload.get("full_bet_fraction", 0.015)),
        max_bet_fraction=float(policy_payload.get("max_bet_fraction", 0.02)),
        max_total_bet_fraction=float(policy_payload.get("max_total_bet_fraction", 0.05)),
        sizing_method=str(policy_payload.get("sizing_method", "tiered_probability")),
        flat_bet_fraction=float(policy_payload.get("flat_bet_fraction", policy_payload.get("base_bet_fraction", policy_payload.get("small_bet_fraction", 0.005)))),
        coarse_low_bet_fraction=float(policy_payload.get("coarse_low_bet_fraction", 0.003)),
        coarse_mid_bet_fraction=float(policy_payload.get("coarse_mid_bet_fraction", 0.005)),
        coarse_high_bet_fraction=float(policy_payload.get("coarse_high_bet_fraction", 0.007)),
        coarse_high_max_share=float(policy_payload.get("coarse_high_max_share", 0.30)),
        coarse_mid_max_share=float(policy_payload.get("coarse_mid_max_share", 0.50)),
        coarse_high_max_plays=int(policy_payload.get("coarse_high_max_plays", 0)),
        coarse_mid_max_plays=int(policy_payload.get("coarse_mid_max_plays", 0)),
        coarse_score_alpha_uncertainty=float(policy_payload.get("coarse_score_alpha_uncertainty", 0.18)),
        coarse_score_beta_dependency=float(policy_payload.get("coarse_score_beta_dependency", 0.12)),
        coarse_score_gamma_support=float(policy_payload.get("coarse_score_gamma_support", 0.08)),
        coarse_score_model=str(policy_payload.get("coarse_score_model", "legacy")),
        coarse_score_delta_prob_weight=float(policy_payload.get("coarse_score_delta_prob_weight", 0.0)),
        coarse_score_ev_weight=float(policy_payload.get("coarse_score_ev_weight", 0.0)),
        coarse_score_risk_weight=float(policy_payload.get("coarse_score_risk_weight", 0.0)),
        coarse_score_recency_weight=float(policy_payload.get("coarse_score_recency_weight", 0.0)),
        staking_bucket_model_payload=staking_bucket_model,
        staking_bucket_model_month=run_month,
        staking_bucket_model_min_rows=int(policy_payload.get("staking_bucket_model_min_rows", 0)),
        belief_uncertainty_lower=float(policy_payload.get("belief_uncertainty_lower", 0.75)),
        belief_uncertainty_upper=float(policy_payload.get("belief_uncertainty_upper", 1.15)),
        append_agreement_min=int(policy_payload.get("append_agreement_min", 3)),
        append_edge_percentile_min=float(policy_payload.get("append_edge_percentile_min", 0.90)),
        append_max_extra_plays=int(policy_payload.get("append_max_extra_plays", 3)),
        board_objective_overfetch=float(policy_payload.get("board_objective_overfetch", 4.0)),
        board_objective_candidate_limit=int(policy_payload.get("board_objective_candidate_limit", 36)),
        board_objective_max_search_nodes=int(policy_payload.get("board_objective_max_search_nodes", 750000)),
        board_objective_lambda_corr=float(policy_payload.get("board_objective_lambda_corr", 0.12)),
        board_objective_lambda_conc=float(policy_payload.get("board_objective_lambda_conc", 0.07)),
        board_objective_lambda_unc=float(policy_payload.get("board_objective_lambda_unc", 0.06)),
        board_objective_corr_same_game=float(policy_payload.get("board_objective_corr_same_game", 0.65)),
        board_objective_corr_same_player=float(policy_payload.get("board_objective_corr_same_player", 1.0)),
        board_objective_corr_same_target=float(policy_payload.get("board_objective_corr_same_target", 0.15)),
        board_objective_corr_same_direction=float(policy_payload.get("board_objective_corr_same_direction", 0.05)),
        board_objective_corr_same_script_cluster=float(policy_payload.get("board_objective_corr_same_script_cluster", 0.30)),
        board_objective_swap_candidates=int(policy_payload.get("board_objective_swap_candidates", 18)),
        board_objective_swap_rounds=int(policy_payload.get("board_objective_swap_rounds", 2)),
        board_objective_instability_enabled=bool(policy_payload.get("board_objective_instability_enabled", False)),
        board_objective_lambda_shadow_disagreement=float(policy_payload.get("board_objective_lambda_shadow_disagreement", 0.0)),
        board_objective_lambda_segment_weakness=float(policy_payload.get("board_objective_lambda_segment_weakness", 0.0)),
        board_objective_instability_near_cutoff_window=int(policy_payload.get("board_objective_instability_near_cutoff_window", 3)),
        board_objective_instability_top_protected=int(policy_payload.get("board_objective_instability_top_protected", 3)),
        board_objective_instability_veto_enabled=bool(policy_payload.get("board_objective_instability_veto_enabled", False)),
        board_objective_instability_veto_quantile=float(policy_payload.get("board_objective_instability_veto_quantile", 0.85)),
        board_objective_dynamic_size_enabled=bool(policy_payload.get("board_objective_dynamic_size_enabled", False)),
        board_objective_dynamic_size_max_shrink=int(policy_payload.get("board_objective_dynamic_size_max_shrink", 0)),
        board_objective_dynamic_size_trigger=float(policy_payload.get("board_objective_dynamic_size_trigger", 0.62)),
        board_objective_fp_veto_enabled=bool(policy_payload.get("board_objective_fp_veto_enabled", False)),
        board_objective_fp_veto_live=bool(policy_payload.get("board_objective_fp_veto_live", False)),
        board_objective_fp_veto_tail_slots=int(policy_payload.get("board_objective_fp_veto_tail_slots", 2)),
        board_objective_fp_veto_top_protected=int(policy_payload.get("board_objective_fp_veto_top_protected", 6)),
        board_objective_fp_veto_threshold=float(policy_payload.get("board_objective_fp_veto_threshold", 0.80)),
        board_objective_fp_veto_max_drops=int(policy_payload.get("board_objective_fp_veto_max_drops", 1)),
        board_objective_fp_veto_quantile=float(policy_payload.get("board_objective_fp_veto_quantile", 0.70)),
        board_objective_fp_veto_max_swaps=int(policy_payload.get("board_objective_fp_veto_max_swaps", 1)),
        board_objective_fp_veto_swap_candidates=int(policy_payload.get("board_objective_fp_veto_swap_candidates", 24)),
        board_objective_fp_veto_min_swap_gain=float(policy_payload.get("board_objective_fp_veto_min_swap_gain", 0.0025)),
        board_objective_fp_veto_risk_lambda=float(policy_payload.get("board_objective_fp_veto_risk_lambda", 0.18)),
        board_objective_fp_veto_ml_weight=float(policy_payload.get("board_objective_fp_veto_ml_weight", 0.45)),
        max_history_staleness_days=int(policy_payload.get("max_history_staleness_days", 0)),
        min_recency_factor=float(policy_payload.get("min_recency_factor", 0.0)),
        selected_board_calibrator=selected_board_calibrator,
        selected_board_calibration_month=run_month,
        learned_gate_payload=learned_pool_gate,
        learned_gate_month=run_month,
        learned_gate_min_rows=int(policy_payload.get("learned_gate_min_rows", 0)),
        learned_gate_rescue_enabled=bool(policy_payload.get("learned_gate_rescue_enabled", True)),
        learned_gate_rescue_target_share=float(policy_payload.get("learned_gate_rescue_target_share", 0.35)),
        learned_gate_rescue_floor_quantile=float(policy_payload.get("learned_gate_rescue_floor_quantile", 0.35)),
        learned_gate_rescue_max_rows=int(policy_payload.get("learned_gate_rescue_max_rows", 0)),
        initial_pool_gate_enabled=bool(policy_payload.get("initial_pool_gate_enabled", True)),
        initial_pool_gate_drop_fraction=float(policy_payload.get("initial_pool_gate_drop_fraction", 0.10)),
        initial_pool_gate_score_col=str(policy_payload.get("initial_pool_gate_score_col", "selector_expected_win_rate")),
        initial_pool_gate_min_keep_rows=int(policy_payload.get("initial_pool_gate_min_keep_rows", 20)),
        accepted_pick_gate_payload=accepted_pick_gate,
        accepted_pick_gate_month=run_month,
        accepted_pick_gate_enabled=bool(policy_payload.get("accepted_pick_gate_enabled", False)),
        accepted_pick_gate_live=bool(policy_payload.get("accepted_pick_gate_live", False)),
        accepted_pick_gate_min_rows=int(policy_payload.get("accepted_pick_gate_min_rows", 0)),
    )


def _resolve_board_rows(
    board_df: pd.DataFrame,
    history_actual_lookup: dict[tuple[str, str, str], float],
    data_proc_actual_lookup: dict[tuple[str, str, str], float],
    *,
    variant: str,
    run_date_token: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if board_df.empty:
        return rows
    payout = PAYOUT_MINUS_110
    for _, row in board_df.iterrows():
        market_date_key = pd.to_datetime(row.get("market_date"), errors="coerce")
        market_date_key = market_date_key.strftime("%Y-%m-%d") if pd.notna(market_date_key) else pd.to_datetime(run_date_token, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")
        player_keys = _player_key_variants(str(row.get("player", "")), str(row.get("market_player_raw", "")))
        target = str(row.get("target", "")).upper().strip()
        actual, actual_source, matched_date = _lookup_actual_with_date_fallback(
            market_date_key,
            player_keys,
            target,
            data_proc_actual_lookup=data_proc_actual_lookup,
            history_actual_lookup=history_actual_lookup,
            near_date_days=1,
        )
        line = pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").iloc[0]
        result = _resolve_result(str(row.get("direction", "")), line=float(line) if pd.notna(line) else np.nan, actual=actual)
        units = np.nan
        if result == "win":
            units = payout
        elif result == "loss":
            units = -1.0
        elif result == "push":
            units = 0.0
        rows.append(
            {
                "variant": str(variant),
                "run_date": str(pd.to_datetime(run_date_token, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")),
                "player": str(row.get("player", "")),
                "target": target,
                "direction": str(row.get("direction", "")).upper().strip(),
                "market_line": float(line) if pd.notna(line) else np.nan,
                "prediction": float(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]) else np.nan,
                "expected_win_rate": float(pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").iloc[0]) else np.nan,
                "board_play_win_prob": float(pd.to_numeric(pd.Series([row.get("board_play_win_prob")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("board_play_win_prob")]), errors="coerce").iloc[0]) else np.nan,
                "line_decision_trade_eligible": bool(pd.to_numeric(pd.Series([row.get("line_decision_trade_eligible")]), errors="coerce").fillna(1).iloc[0]),
                "line_decision_action": str(row.get("line_decision_action", "")),
                "actual": float(actual) if pd.notna(actual) else np.nan,
                "actual_source": str(actual_source),
                "actual_matched_date": str(matched_date),
                "result": str(result),
                "units": float(units) if pd.notna(units) else np.nan,
            }
        )
    return rows


def _summarize_variant(rows_df: pd.DataFrame) -> dict[str, Any]:
    if rows_df.empty or "result" not in rows_df.columns:
        return {
            "rows": int(len(rows_df)),
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "missing": int(len(rows_df)),
            "hit_rate": np.nan,
            "units": 0.0,
            "units_per_resolved": np.nan,
            "avg_board_play_win_prob_resolved": np.nan,
        }
    resolved = rows_df.loc[rows_df["result"].isin(["win", "loss"])].copy()
    wins = int((resolved["result"] == "win").sum())
    losses = int((resolved["result"] == "loss").sum())
    pushes = int((rows_df["result"] == "push").sum())
    missing = int((rows_df["result"] == "missing").sum())
    total_rows = int(len(rows_df))
    resolved_count = int(len(resolved))
    units = float(pd.to_numeric(rows_df["units"], errors="coerce").fillna(0.0).sum()) if not rows_df.empty else 0.0
    avg_prob = float(pd.to_numeric(resolved["board_play_win_prob"], errors="coerce").mean()) if resolved_count > 0 else np.nan
    hit_rate = float(wins / resolved_count) if resolved_count > 0 else np.nan
    units_per_resolved = float(units / resolved_count) if resolved_count > 0 else np.nan
    return {
        "rows": total_rows,
        "resolved": resolved_count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "missing": missing,
        "hit_rate": hit_rate,
        "units": units,
        "units_per_resolved": units_per_resolved,
        "avg_board_play_win_prob_resolved": avg_prob,
    }


def main() -> None:
    args = parse_args()
    run_dates = _iter_run_dates(args.daily_runs_dir.resolve(), args.start_run_date, args.end_run_date, args.max_days)
    if not run_dates:
        raise RuntimeError("No run-date folders with selector artifacts were found in the requested window.")

    history_csv = args.history_csv.resolve()
    history_df = pd.read_csv(history_csv)
    history_lookup = build_history_lookup(history_df)
    line_decision_lookup = build_line_decision_lookup(history_df)

    selected_board_calibrator, _ = load_selected_board_calibrator(args.selected_board_calibrator_json, disabled=False)
    history_actual_lookup = _build_actual_lookup(history_csv)
    data_proc_actual_lookup = _build_data_proc_actual_lookup(
        args.data_proc_root.resolve(),
        args.start_run_date,
        args.end_run_date,
    )
    line_decision_config = LineDecisionConfig(
        no_trade_threshold=float(args.no_trade_threshold),
        min_trade_prob=float(args.min_trade_prob),
        min_trade_prob_gap=float(args.min_trade_prob_gap),
    )

    row_records: list[dict[str, Any]] = []
    day_records: list[dict[str, Any]] = []

    for run_date in run_dates:
        run_dir = args.daily_runs_dir.resolve() / run_date
        slate_csv = run_dir / f"upcoming_market_slate_{run_date}.csv"
        if not slate_csv.exists():
            continue
        slate_df = pd.read_csv(slate_csv)
        if slate_df.empty:
            continue
        policy_payload = _load_daily_policy(run_dir)
        if not policy_payload:
            continue
        run_month = pd.to_datetime(run_date, format="%Y%m%d", errors="coerce")
        run_month_token = run_month.strftime("%Y-%m") if pd.notna(run_month) else None
        learned_pool_gate, _ = load_learned_pool_gate(
            args.learned_gate_json,
            disabled=bool(not policy_payload.get("learned_gate_enabled", False)),
        )
        accepted_pick_gate, _ = load_accepted_pick_gate(
            args.accepted_pick_gate_json,
            disabled=bool(not policy_payload.get("accepted_pick_gate_enabled", False)),
        )
        staking_bucket_model, _ = load_staking_bucket_model(
            args.staking_bucket_model_json,
            disabled=bool(not policy_payload.get("staking_bucket_model_enabled", False)),
        )

        per_variant_metrics: dict[str, dict[str, Any]] = {}
        for variant, sidecar_enabled in (("baseline", False), ("sidecar", True)):
            selector_df = _prepare_selector(
                slate_df,
                history_df,
                history_lookup,
                line_decision_lookup,
                policy_payload,
                line_decision_enabled=sidecar_enabled,
                line_decision_config=line_decision_config,
            )
            board_df = _compute_board(
                selector_df,
                policy_payload,
                selected_board_calibrator=selected_board_calibrator,
                learned_pool_gate=learned_pool_gate,
                accepted_pick_gate=accepted_pick_gate,
                staking_bucket_model=staking_bucket_model,
                run_month=run_month_token,
            )
            resolved_rows = _resolve_board_rows(
                board_df,
                history_actual_lookup=history_actual_lookup,
                data_proc_actual_lookup=data_proc_actual_lookup,
                variant=variant,
                run_date_token=run_date,
            )
            row_records.extend(resolved_rows)
            variant_rows_df = pd.DataFrame.from_records(resolved_rows)
            per_variant_metrics[variant] = _summarize_variant(variant_rows_df)

        base_metrics = per_variant_metrics.get("baseline", {})
        side_metrics = per_variant_metrics.get("sidecar", {})
        day_records.append(
            {
                "run_date": pd.to_datetime(run_date, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d"),
                "baseline_rows": int(base_metrics.get("rows", 0)),
                "sidecar_rows": int(side_metrics.get("rows", 0)),
                "baseline_resolved": int(base_metrics.get("resolved", 0)),
                "sidecar_resolved": int(side_metrics.get("resolved", 0)),
                "baseline_hit_rate": base_metrics.get("hit_rate", np.nan),
                "sidecar_hit_rate": side_metrics.get("hit_rate", np.nan),
                "delta_hit_rate_pp": (float(side_metrics["hit_rate"]) - float(base_metrics["hit_rate"])) * 100.0
                if pd.notna(side_metrics.get("hit_rate")) and pd.notna(base_metrics.get("hit_rate"))
                else np.nan,
                "baseline_units": float(base_metrics.get("units", 0.0)),
                "sidecar_units": float(side_metrics.get("units", 0.0)),
                "delta_units": float(side_metrics.get("units", 0.0)) - float(base_metrics.get("units", 0.0)),
            }
        )

    rows_df = pd.DataFrame.from_records(row_records)
    daily_df = pd.DataFrame.from_records(day_records)
    if rows_df.empty:
        raise RuntimeError("Replay produced no resolved rows.")

    summary_by_variant = {
        variant: _summarize_variant(rows_df.loc[rows_df["variant"] == variant].copy())
        for variant in ("baseline", "sidecar")
    }
    baseline = summary_by_variant["baseline"]
    sidecar = summary_by_variant["sidecar"]
    delta_hit_rate_pp = (
        (float(sidecar["hit_rate"]) - float(baseline["hit_rate"])) * 100.0
        if pd.notna(sidecar.get("hit_rate")) and pd.notna(baseline.get("hit_rate"))
        else np.nan
    )
    delta_units = float(sidecar["units"]) - float(baseline["units"])
    delta_rows = int(sidecar["rows"]) - int(baseline["rows"])
    daily_positive_hit_days = int((pd.to_numeric(daily_df.get("delta_hit_rate_pp"), errors="coerce") > 0).sum()) if not daily_df.empty else 0
    daily_positive_unit_days = int((pd.to_numeric(daily_df.get("delta_units"), errors="coerce") > 0).sum()) if not daily_df.empty else 0

    summary_payload = {
        "window": {
            "start_run_date": str(args.start_run_date),
            "end_run_date": str(args.end_run_date),
            "days_replayed": int(len(daily_df)),
        },
        "line_decision_config": {
            "no_trade_threshold": float(args.no_trade_threshold),
            "min_trade_prob": float(args.min_trade_prob),
            "min_trade_prob_gap": float(args.min_trade_prob_gap),
        },
        "baseline": baseline,
        "sidecar": sidecar,
        "delta": {
            "hit_rate_pp": delta_hit_rate_pp,
            "units": delta_units,
            "rows": delta_rows,
            "positive_hit_rate_delta": bool(pd.notna(delta_hit_rate_pp) and delta_hit_rate_pp > 0.0),
            "positive_units_delta": bool(delta_units > 0.0),
            "daily_positive_hit_rate_days": daily_positive_hit_days,
            "daily_positive_unit_days": daily_positive_unit_days,
        },
    }

    args.rows_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv_out.parent.mkdir(parents=True, exist_ok=True)

    rows_df.to_csv(args.rows_csv_out, index=False)
    daily_df.to_csv(args.summary_csv_out, index=False)
    args.summary_json_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("LINE-DECISION SIDECAR BACKTEST")
    print("=" * 88)
    print(f"Days replayed:          {len(daily_df)}")
    print(f"Baseline hit rate:      {baseline.get('hit_rate', np.nan):.4f}  ({baseline.get('wins', 0)}-{baseline.get('losses', 0)} resolved)")
    print(f"Sidecar hit rate:       {sidecar.get('hit_rate', np.nan):.4f}  ({sidecar.get('wins', 0)}-{sidecar.get('losses', 0)} resolved)")
    print(f"Hit-rate delta (pp):    {delta_hit_rate_pp:+.2f}")
    print(f"Baseline units:         {baseline.get('units', 0.0):+.3f}")
    print(f"Sidecar units:          {sidecar.get('units', 0.0):+.3f}")
    print(f"Units delta:            {delta_units:+.3f}")
    print(f"Row-count delta:        {delta_rows:+d}")
    print(f"Daily +hit-rate days:   {daily_positive_hit_days}")
    print(f"Daily +units days:      {daily_positive_unit_days}")
    print(f"Rows CSV:               {args.rows_csv_out}")
    print(f"Daily summary CSV:      {args.summary_csv_out}")
    print(f"Summary JSON:           {args.summary_json_out}")


if __name__ == "__main__":
    main()
