#!/usr/bin/env python3
"""
End-to-end market decision pipeline.

This script intentionally orchestrates the pipeline as functions, not shell calls:
1. build upcoming slate from the latest market snapshot
2. score/rank plays using historical edge behavior
3. apply production-safe calibration defaults
4. post-process into a final de-correlated, EV-ranked board
5. validate market-line and prior-history source coverage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "inference"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_upcoming_slate import (
    DEFAULT_TARGET_PREDICTION_CALIBRATOR,
    MODEL_DIR,
    build_records,
    load_market_wide,
    resolve_manifest_path,
)
from decision_engine.conditional_promotion import apply_conditional_promotion
from decision_engine.policy_tuning import build_default_shadow_strategies
from post_process_market_plays import compute_final_board
from select_market_plays import build_history_lookup, build_play_rows
from structured_stack_inference import StructuredStackInference

try:
    from decision_engine.xgb_ltr_reranker import score_selector_with_xgb_ltr

    HAS_XGB_LTR = True
except Exception:
    HAS_XGB_LTR = False

try:
    from decision_engine.accept_reject_model import apply_acceptor_to_selector

    HAS_ACCEPT_REJECT = True
except Exception:
    HAS_ACCEPT_REJECT = False

try:
    from decision_engine.robust_reranker import score_selector_with_robust_reranker

    HAS_ROBUST_RERANKER = True
except Exception:
    HAS_ROBUST_RERANKER = False


POLICY_PROFILES = {config.name: config for config in build_default_shadow_strategies()}
DEFAULT_POLICY = POLICY_PROFILES["production_board_objective_b12"]
TARGETS = ["PTS", "TRB", "AST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the market slate -> selector -> final board pipeline.")
    parser.add_argument("--season", type=int, required=True, help="Season end year, e.g. 2026.")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production.")
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="production_board_objective_b12",
        choices=sorted(POLICY_PROFILES.keys()),
        help="Policy profile used for live play selection defaults.",
    )
    parser.add_argument(
        "--market-wide-path",
        type=Path,
        default=REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "latest_player_props_wide.parquet",
        help="Current normalized market snapshot.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv",
        help="Historical row-level backtest CSV for edge calibration.",
    )
    parser.add_argument(
        "--slate-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "upcoming_market_slate.csv",
        help="Intermediate slate CSV output.",
    )
    parser.add_argument(
        "--selector-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "upcoming_market_play_selector.csv",
        help="Intermediate selector CSV output.",
    )
    parser.add_argument(
        "--final-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "final_market_plays.csv",
        help="Final play board CSV output.",
    )
    parser.add_argument(
        "--final-json-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "final_market_plays.json",
        help="Final play board JSON summary.",
    )
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow pipeline to continue with heuristic-only predictions when model load fails.",
    )
    parser.add_argument(
        "--target-prediction-calibrator-json",
        type=Path,
        default=DEFAULT_TARGET_PREDICTION_CALIBRATOR,
        help="Optional target-level short-term prediction calibrator JSON applied before selector ranking.",
    )
    parser.add_argument(
        "--disable-target-prediction-calibration",
        action="store_true",
        help="Disable target-level short-term prediction calibration and keep raw predictor outputs.",
    )
    parser.add_argument("--american-odds", type=int, default=None, help="Assumed American odds for EV.")
    parser.add_argument("--probability-shrink-factor", type=float, default=None, help="Shrink expected win rate toward 50%%.")
    parser.add_argument("--elite-pct", type=float, default=None, help="Percentile cutoff for elite priority plays.")
    parser.add_argument("--min-ev", type=float, default=None, help="Minimum EV to keep a play.")
    parser.add_argument("--min-final-confidence", type=float, default=None, help="Minimum final confidence to keep a play.")
    parser.add_argument(
        "--max-history-staleness-days",
        type=int,
        default=None,
        help="Optional max days between market_date and last_history_date (0 disables staleness filtering).",
    )
    parser.add_argument(
        "--min-recency-factor",
        type=float,
        default=None,
        help="Optional minimum recency_factor required to keep a play (0 disables).",
    )
    parser.add_argument(
        "--min-recommendation",
        type=str,
        default=None,
        choices=["pass", "consider", "strong", "elite"],
        help="Lowest selector recommendation allowed into the final board.",
    )
    parser.add_argument("--max-plays-per-player", type=int, default=None, help="Maximum final plays to keep per player.")
    parser.add_argument("--max-plays-per-target", type=int, default=None, help="Maximum final plays to keep per target when target-specific caps are not supplied.")
    parser.add_argument("--max-pts-plays", type=int, default=None, help="Maximum final PTS plays to keep.")
    parser.add_argument("--max-trb-plays", type=int, default=None, help="Maximum final TRB plays to keep.")
    parser.add_argument("--max-ast-plays", type=int, default=None, help="Maximum final AST plays to keep.")
    parser.add_argument("--max-total-plays", type=int, default=None, help="Maximum total plays to keep.")
    parser.add_argument("--min-board-plays", type=int, default=None, help="Minimum final-board rows to target by adaptive EV relaxation.")
    parser.add_argument("--max-plays-per-game", type=int, default=None, help="Maximum selected plays to keep from the same game/event.")
    parser.add_argument(
        "--max-plays-per-script-cluster",
        type=int,
        default=None,
        help="Maximum selected plays to keep from the same inferred script cluster.",
    )
    parser.add_argument("--non-pts-min-gap-percentile", type=float, default=None, help="Minimum disagreement percentile for TRB/AST plays.")
    parser.add_argument("--edge-adjust-k", type=float, default=None, help="Weight for edge-adjusted EV ranking.")
    parser.add_argument(
        "--selection-mode",
        type=str,
        default=None,
        choices=["ev_adjusted", "edge", "abs_edge", "xgb_ltr", "robust_reranker", "thompson_ev", "set_theory", "edge_append_shadow", "board_objective"],
        help="Final board ranking mode before portfolio constraints.",
    )
    parser.add_argument("--thompson-temperature", type=float, default=None, help="Temperature used for Thompson sampling.")
    parser.add_argument("--thompson-seed", type=int, default=None, help="Seed salt used for deterministic Thompson sampling.")
    parser.add_argument(
        "--sizing-method",
        type=str,
        default=None,
        choices=["tiered_probability", "flat_fraction", "coarse_bucket"],
        help="Stake allocation method applied on the final board.",
    )
    parser.add_argument(
        "--flat-bet-fraction",
        type=float,
        default=None,
        help="Per-play bankroll fraction when sizing-method=flat_fraction.",
    )
    parser.add_argument("--coarse-low-bet-fraction", type=float, default=None, help="When sizing-method=coarse_bucket, low-tier bankroll fraction.")
    parser.add_argument("--coarse-mid-bet-fraction", type=float, default=None, help="When sizing-method=coarse_bucket, mid-tier bankroll fraction.")
    parser.add_argument("--coarse-high-bet-fraction", type=float, default=None, help="When sizing-method=coarse_bucket, high-tier bankroll fraction.")
    parser.add_argument("--coarse-high-max-share", type=float, default=None, help="When sizing-method=coarse_bucket, hard cap on high-tier share.")
    parser.add_argument("--coarse-mid-max-share", type=float, default=None, help="When sizing-method=coarse_bucket, hard cap on mid-tier share.")
    parser.add_argument("--coarse-high-max-plays", type=int, default=None, help="When sizing-method=coarse_bucket, optional hard cap on high-tier play count.")
    parser.add_argument("--coarse-mid-max-plays", type=int, default=None, help="When sizing-method=coarse_bucket, optional hard cap on mid-tier play count.")
    parser.add_argument("--coarse-score-alpha-uncertainty", type=float, default=None, help="Weight on uncertainty penalty in coarse bucket stake score.")
    parser.add_argument("--coarse-score-beta-dependency", type=float, default=None, help="Weight on dependency burden penalty in coarse bucket stake score.")
    parser.add_argument("--coarse-score-gamma-support", type=float, default=None, help="Weight on support-strength lift in coarse bucket stake score.")
    parser.add_argument(
        "--coarse-score-model",
        type=str,
        default=None,
        choices=["legacy", "stake_score_v1", "stake_model_v2"],
        help="Coarse bucket stake-score model variant.",
    )
    parser.add_argument("--coarse-score-delta-prob-weight", type=float, default=None, help="Additional weight on calibrated non-push win-edge strength in coarse stake score.")
    parser.add_argument("--coarse-score-ev-weight", type=float, default=None, help="Additional weight on within-board EV strength in coarse stake score.")
    parser.add_argument("--coarse-score-risk-weight", type=float, default=None, help="Penalty weight on risk composite in coarse stake score.")
    parser.add_argument("--coarse-score-recency-weight", type=float, default=None, help="Additional weight on recency support in coarse stake score.")
    parser.add_argument("--market-regression-floor", type=float, default=None, help="Minimum market-regression lambda for prediction shrinkage.")
    parser.add_argument("--market-regression-ceiling", type=float, default=None, help="Maximum market-regression lambda for prediction shrinkage.")
    parser.add_argument("--belief-uncertainty-lower", type=float, default=None, help="Lower anchor for belief uncertainty confidence scaling.")
    parser.add_argument("--belief-uncertainty-upper", type=float, default=None, help="Upper anchor for belief uncertainty confidence scaling.")
    parser.add_argument("--append-agreement-min", type=int, default=None, help="Minimum E/T/V agreement count required for append-only shadow candidates.")
    parser.add_argument("--append-edge-percentile-min", type=float, default=None, help="Minimum abs-edge percentile required for append-only shadow candidates.")
    parser.add_argument("--append-max-extra-plays", type=int, default=None, help="Maximum append-only shadow plays added beyond the edge base board.")
    parser.add_argument("--board-objective-overfetch", type=float, default=None, help="Candidate overfetch multiplier for board-objective mode.")
    parser.add_argument("--board-objective-candidate-limit", type=int, default=None, help="Candidate universe cap for board-objective mode (0 disables).")
    parser.add_argument("--board-objective-max-search-nodes", type=int, default=None, help="Branch-and-bound node cap for board-objective exact solve.")
    parser.add_argument("--board-objective-lambda-corr", type=float, default=None, help="Correlation penalty weight for board-objective mode.")
    parser.add_argument("--board-objective-lambda-conc", type=float, default=None, help="Concentration penalty weight for board-objective mode.")
    parser.add_argument("--board-objective-lambda-unc", type=float, default=None, help="Uncertainty penalty weight for board-objective mode.")
    parser.add_argument("--board-objective-corr-same-game", type=float, default=None, help="Pairwise dependency contribution for same-game candidates.")
    parser.add_argument("--board-objective-corr-same-player", type=float, default=None, help="Pairwise dependency contribution for same-player candidates.")
    parser.add_argument("--board-objective-corr-same-target", type=float, default=None, help="Pairwise dependency contribution for same-target candidates.")
    parser.add_argument("--board-objective-corr-same-direction", type=float, default=None, help="Pairwise dependency contribution for same-direction candidates.")
    parser.add_argument("--board-objective-corr-same-script-cluster", type=float, default=None, help="Pairwise dependency contribution for same-script-cluster candidates.")
    parser.add_argument("--board-objective-swap-candidates", type=int, default=None, help="Out-of-universe candidates considered for swap optimization.")
    parser.add_argument("--board-objective-swap-rounds", type=int, default=None, help="Max improving swap rounds for board-objective mode.")
    parser.add_argument(
        "--board-objective-instability-enabled",
        action="store_true",
        help="Enable near-cutoff instability penalties/vetoes in board-objective mode.",
    )
    parser.add_argument(
        "--board-objective-instability-disabled",
        action="store_true",
        help="Disable near-cutoff instability penalties/vetoes in board-objective mode.",
    )
    parser.add_argument(
        "--board-objective-lambda-shadow-disagreement",
        type=float,
        default=None,
        help="Penalty weight on shadow-model disagreement near the board cutoff.",
    )
    parser.add_argument(
        "--board-objective-lambda-segment-weakness",
        type=float,
        default=None,
        help="Penalty weight on segment recent weakness near the board cutoff.",
    )
    parser.add_argument(
        "--board-objective-instability-near-cutoff-window",
        type=int,
        default=None,
        help="Rank-distance window around cutoff where instability penalties apply.",
    )
    parser.add_argument(
        "--board-objective-instability-top-protected",
        type=int,
        default=None,
        help="Top ranked rows protected from instability penalties/veto.",
    )
    parser.add_argument(
        "--board-objective-instability-veto-enabled",
        action="store_true",
        help="Enable near-cutoff veto of highest-instability inclusion candidates.",
    )
    parser.add_argument(
        "--board-objective-instability-veto-disabled",
        action="store_true",
        help="Disable near-cutoff instability veto even when policy enables it.",
    )
    parser.add_argument(
        "--board-objective-instability-veto-quantile",
        type=float,
        default=None,
        help="Quantile threshold for near-cutoff instability veto.",
    )
    parser.add_argument(
        "--board-objective-dynamic-size-enabled",
        action="store_true",
        help="Enable dynamic board-size shrink on unstable slates in board-objective mode.",
    )
    parser.add_argument(
        "--board-objective-dynamic-size-disabled",
        action="store_true",
        help="Disable dynamic board-size shrink on unstable slates in board-objective mode.",
    )
    parser.add_argument(
        "--board-objective-dynamic-size-max-shrink",
        type=int,
        default=None,
        help="Maximum rows dynamic board size may shrink below max-total-plays.",
    )
    parser.add_argument(
        "--board-objective-dynamic-size-trigger",
        type=float,
        default=None,
        help="Composite instability trigger for dynamic board-size shrink.",
    )
    parser.add_argument(
        "--disable-conditional-framework",
        action="store_true",
        help="Disable structured conditional promotion and run baseline-only selection.",
    )
    parser.add_argument(
        "--conditional-framework-mode",
        type=str,
        default=None,
        choices=["auto", "full", "reduced", "baseline_only", "safe_shutdown"],
        help="Override conditional framework mode.",
    )
    parser.add_argument(
        "--conditional-failure-memory-path",
        type=Path,
        default=None,
        help="Optional path for persisted conditional failure memory.",
    )
    parser.add_argument(
        "--conditional-max-promotions-per-slate",
        type=int,
        default=None,
        help="Optional override for total conditional promotions per slate.",
    )
    parser.add_argument(
        "--conditional-max-promotions-per-game",
        type=int,
        default=None,
        help="Optional override for per-game conditional promotions.",
    )
    parser.add_argument(
        "--conditional-max-promoted-share",
        type=float,
        default=None,
        help="Optional override for promoted share cap within recoverable weak plays.",
    )
    parser.add_argument(
        "--selected-board-calibrator-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator.json",
        help="Optional selected-board calibrator payload JSON (monthly walk-forward).",
    )
    parser.add_argument(
        "--disable-selected-board-calibration",
        action="store_true",
        help="Disable selected-board calibration and keep identity probabilities.",
    )
    parser.add_argument(
        "--selected-board-calibration-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month hint when applying selected-board calibration.",
    )
    parser.add_argument(
        "--learned-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "learned_pool_gate.json",
        help="Optional learned pool-gate payload JSON.",
    )
    parser.add_argument(
        "--disable-learned-gate",
        action="store_true",
        help="Disable learned pool-gate filtering even if policy enables it.",
    )
    parser.add_argument(
        "--enable-learned-gate",
        action="store_true",
        help="Enable learned pool-gate filtering even if policy disables it.",
    )
    parser.add_argument(
        "--learned-gate-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month hint when applying learned pool-gate thresholds.",
    )
    parser.add_argument(
        "--learned-gate-min-rows",
        type=int,
        default=None,
        help="Minimum rows that must pass learned gate before enforcement (0 delegates to payload/policy).",
    )
    parser.add_argument(
        "--disable-initial-pool-gate",
        action="store_true",
        help="Disable pre-board initial pool pruning before learned-gate and board-objective selection.",
    )
    parser.add_argument(
        "--initial-pool-gate-drop-fraction",
        type=float,
        default=None,
        help="Drop fraction of lowest-scoring initial pool rows in board-objective mode.",
    )
    parser.add_argument(
        "--initial-pool-gate-score-col",
        type=str,
        default=None,
        help="Primary numeric selector column used to rank rows for initial pool pruning.",
    )
    parser.add_argument(
        "--initial-pool-gate-min-keep-rows",
        type=int,
        default=None,
        help="Minimum rows preserved after initial pool pruning.",
    )
    parser.add_argument(
        "--accepted-pick-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "accepted_pick_gate" / "candidates" / "accepted_pick_gate_candidate.json",
        help="Optional accepted-pick keep/drop gate payload JSON.",
    )
    parser.add_argument(
        "--enable-accepted-pick-gate",
        action="store_true",
        help="Enable accepted-pick gate in final board filtering.",
    )
    parser.add_argument(
        "--disable-accepted-pick-gate",
        action="store_true",
        help="Disable accepted-pick gate even if policy enables it.",
    )
    parser.add_argument(
        "--accepted-pick-gate-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month hint when applying accepted-pick gate payloads.",
    )
    parser.add_argument(
        "--accepted-pick-gate-min-rows",
        type=int,
        default=None,
        help="Minimum rows required before enforcing accepted-pick gate (0 disables row floor).",
    )
    parser.add_argument(
        "--accepted-pick-gate-live",
        action="store_true",
        help="Apply accepted-pick gate vetoes live (drop vetoed rows).",
    )
    parser.add_argument(
        "--accepted-pick-gate-shadow",
        action="store_true",
        help="Force accepted-pick gate to shadow-only tagging (no live drops).",
    )
    parser.add_argument(
        "--staking-bucket-model-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "staking_bucket_model_v2.json",
        help="Optional walk-forward staking bucket model payload JSON.",
    )
    parser.add_argument(
        "--disable-staking-bucket-model",
        action="store_true",
        help="Disable walk-forward staking bucket model usage even if policy enables it.",
    )
    parser.add_argument(
        "--enable-staking-bucket-model",
        action="store_true",
        help="Enable walk-forward staking bucket model usage even if policy disables it.",
    )
    parser.add_argument(
        "--staking-bucket-model-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month hint for walk-forward staking bucket model lookup.",
    )
    parser.add_argument(
        "--staking-bucket-model-min-rows",
        type=int,
        default=None,
        help="Minimum monthly train rows required before applying walk-forward staking bucket model (0 disables this guard).",
    )
    return parser.parse_args()


def resolve_policy(args: argparse.Namespace):
    base = POLICY_PROFILES[args.policy_profile]
    payload = base.to_dict()
    override_fields = {
        "american_odds": args.american_odds,
        "probability_shrink_factor": args.probability_shrink_factor,
        "elite_pct": args.elite_pct,
        "min_ev": args.min_ev,
        "min_final_confidence": args.min_final_confidence,
        "max_history_staleness_days": args.max_history_staleness_days,
        "min_recency_factor": args.min_recency_factor,
        "min_recommendation": args.min_recommendation,
        "max_plays_per_player": args.max_plays_per_player,
        "max_plays_per_target": args.max_plays_per_target,
        "max_pts_plays": args.max_pts_plays,
        "max_trb_plays": args.max_trb_plays,
        "max_ast_plays": args.max_ast_plays,
        "max_total_plays": args.max_total_plays,
        "min_board_plays": args.min_board_plays,
        "max_plays_per_game": args.max_plays_per_game,
        "max_plays_per_script_cluster": args.max_plays_per_script_cluster,
        "non_pts_min_gap_percentile": args.non_pts_min_gap_percentile,
        "edge_adjust_k": args.edge_adjust_k,
        "selection_mode": args.selection_mode,
        "thompson_temperature": args.thompson_temperature,
        "thompson_seed": args.thompson_seed,
        "sizing_method": args.sizing_method,
        "flat_bet_fraction": args.flat_bet_fraction,
        "coarse_low_bet_fraction": args.coarse_low_bet_fraction,
        "coarse_mid_bet_fraction": args.coarse_mid_bet_fraction,
        "coarse_high_bet_fraction": args.coarse_high_bet_fraction,
        "coarse_high_max_share": args.coarse_high_max_share,
        "coarse_mid_max_share": args.coarse_mid_max_share,
        "coarse_high_max_plays": args.coarse_high_max_plays,
        "coarse_mid_max_plays": args.coarse_mid_max_plays,
        "coarse_score_alpha_uncertainty": args.coarse_score_alpha_uncertainty,
        "coarse_score_beta_dependency": args.coarse_score_beta_dependency,
        "coarse_score_gamma_support": args.coarse_score_gamma_support,
        "coarse_score_model": args.coarse_score_model,
        "coarse_score_delta_prob_weight": args.coarse_score_delta_prob_weight,
        "coarse_score_ev_weight": args.coarse_score_ev_weight,
        "coarse_score_risk_weight": args.coarse_score_risk_weight,
        "coarse_score_recency_weight": args.coarse_score_recency_weight,
        "staking_bucket_model_enabled": True if args.enable_staking_bucket_model else (False if args.disable_staking_bucket_model else None),
        "staking_bucket_model_min_rows": args.staking_bucket_model_min_rows,
        "market_regression_floor": args.market_regression_floor,
        "market_regression_ceiling": args.market_regression_ceiling,
        "belief_uncertainty_lower": args.belief_uncertainty_lower,
        "belief_uncertainty_upper": args.belief_uncertainty_upper,
        "append_agreement_min": args.append_agreement_min,
        "append_edge_percentile_min": args.append_edge_percentile_min,
        "append_max_extra_plays": args.append_max_extra_plays,
        "board_objective_overfetch": args.board_objective_overfetch,
        "board_objective_candidate_limit": args.board_objective_candidate_limit,
        "board_objective_max_search_nodes": args.board_objective_max_search_nodes,
        "board_objective_lambda_corr": args.board_objective_lambda_corr,
        "board_objective_lambda_conc": args.board_objective_lambda_conc,
        "board_objective_lambda_unc": args.board_objective_lambda_unc,
        "board_objective_corr_same_game": args.board_objective_corr_same_game,
        "board_objective_corr_same_player": args.board_objective_corr_same_player,
        "board_objective_corr_same_target": args.board_objective_corr_same_target,
        "board_objective_corr_same_direction": args.board_objective_corr_same_direction,
        "board_objective_corr_same_script_cluster": args.board_objective_corr_same_script_cluster,
        "board_objective_swap_candidates": args.board_objective_swap_candidates,
        "board_objective_swap_rounds": args.board_objective_swap_rounds,
        "board_objective_instability_enabled": True
        if args.board_objective_instability_enabled
        else (False if args.board_objective_instability_disabled else None),
        "board_objective_lambda_shadow_disagreement": args.board_objective_lambda_shadow_disagreement,
        "board_objective_lambda_segment_weakness": args.board_objective_lambda_segment_weakness,
        "board_objective_instability_near_cutoff_window": args.board_objective_instability_near_cutoff_window,
        "board_objective_instability_top_protected": args.board_objective_instability_top_protected,
        "board_objective_instability_veto_enabled": True
        if args.board_objective_instability_veto_enabled
        else (False if args.board_objective_instability_veto_disabled else None),
        "board_objective_instability_veto_quantile": args.board_objective_instability_veto_quantile,
        "board_objective_dynamic_size_enabled": True
        if args.board_objective_dynamic_size_enabled
        else (False if args.board_objective_dynamic_size_disabled else None),
        "board_objective_dynamic_size_max_shrink": args.board_objective_dynamic_size_max_shrink,
        "board_objective_dynamic_size_trigger": args.board_objective_dynamic_size_trigger,
        "conditional_framework_enabled": None if not args.disable_conditional_framework else False,
        "conditional_framework_mode": args.conditional_framework_mode,
        "conditional_failure_memory_path": str(args.conditional_failure_memory_path) if args.conditional_failure_memory_path else None,
        "conditional_max_promotions_per_slate": args.conditional_max_promotions_per_slate,
        "conditional_max_promotions_per_game": args.conditional_max_promotions_per_game,
        "conditional_max_promoted_share_of_recoverable": args.conditional_max_promoted_share,
        "learned_gate_enabled": True if args.enable_learned_gate else (False if args.disable_learned_gate else None),
        "learned_gate_min_rows": args.learned_gate_min_rows,
        "initial_pool_gate_enabled": False if args.disable_initial_pool_gate else None,
        "initial_pool_gate_drop_fraction": args.initial_pool_gate_drop_fraction,
        "initial_pool_gate_score_col": args.initial_pool_gate_score_col,
        "initial_pool_gate_min_keep_rows": args.initial_pool_gate_min_keep_rows,
        "accepted_pick_gate_enabled": True
        if args.enable_accepted_pick_gate
        else (False if args.disable_accepted_pick_gate else None),
        "accepted_pick_gate_live": True
        if args.accepted_pick_gate_live
        else (False if args.accepted_pick_gate_shadow else None),
        "accepted_pick_gate_min_rows": args.accepted_pick_gate_min_rows,
    }
    for key, value in override_fields.items():
        if value is not None:
            payload[key] = value
    return payload


def apply_heuristic_policy_overrides(policy_payload: dict) -> dict:
    """
    When historical calibration rows are unavailable, switch to a deterministic
    abs-edge fallback policy so boards remain actionable.
    """
    out = dict(policy_payload)
    out["selection_mode"] = "abs_edge"
    out["ranking_mode"] = "abs_edge"
    out["min_recommendation"] = "pass"
    out["min_ev"] = min(float(out.get("min_ev", 0.0)), -0.20)
    out["min_final_confidence"] = min(float(out.get("min_final_confidence", 0.0)), 0.0)
    out["non_pts_min_gap_percentile"] = min(float(out.get("non_pts_min_gap_percentile", 0.90)), 0.0)
    out["max_total_plays"] = max(int(out.get("max_total_plays", 0)), 8)
    out["max_pts_plays"] = max(int(out.get("max_pts_plays", 0)), 8)
    out["max_plays_per_game"] = 0
    out["min_bet_win_rate"] = min(float(out.get("min_bet_win_rate", 0.57)), 0.49)
    out["medium_bet_win_rate"] = min(float(out.get("medium_bet_win_rate", 0.60)), 0.52)
    out["full_bet_win_rate"] = min(float(out.get("full_bet_win_rate", 0.65)), 0.56)
    out["medium_tier_percentile"] = min(float(out.get("medium_tier_percentile", 0.80)), 0.00)
    out["strong_tier_percentile"] = min(float(out.get("strong_tier_percentile", 0.90)), 0.00)
    out["elite_tier_percentile"] = min(float(out.get("elite_tier_percentile", out.get("elite_pct", 0.95))), 0.00)
    out["learned_gate_enabled"] = False
    out["accepted_pick_gate_enabled"] = False
    out["accepted_pick_gate_live"] = False
    out["board_objective_instability_enabled"] = False
    out["board_objective_dynamic_size_enabled"] = False
    out["heuristic_overrides_applied"] = True
    return out


def apply_weak_play_capacity_overrides(policy_payload: dict) -> dict:
    """
    If the policy intentionally allows weak/pass recommendations, expand board
    capacity so the slate has more optional plays to choose from.
    """
    out = dict(policy_payload)
    min_recommendation = str(out.get("min_recommendation", "consider")).lower()
    min_ev = float(out.get("min_ev", 0.0))
    min_final_confidence = float(out.get("min_final_confidence", 0.0))
    intentionally_weak_gate = (min_recommendation == "pass") and (min_ev < 0.0) and (min_final_confidence <= 0.0)
    if not intentionally_weak_gate:
        out["weak_play_expanded_board"] = False
        return out

    out["max_total_plays"] = max(int(out.get("max_total_plays", 0)), 12)
    out["max_pts_plays"] = max(int(out.get("max_pts_plays", 0)), 10)
    out["max_trb_plays"] = max(int(out.get("max_trb_plays", 0)), 4)
    out["max_ast_plays"] = max(int(out.get("max_ast_plays", 0)), 4)
    if int(out.get("max_plays_per_target", 0)) > 0:
        out["max_plays_per_target"] = max(int(out.get("max_plays_per_target", 0)), 4)
    if int(out.get("max_plays_per_game", 0)) > 0:
        out["max_plays_per_game"] = max(int(out.get("max_plays_per_game", 0)), 3)
    out["weak_play_expanded_board"] = True
    return out


def shrink_expected_win_rate(raw_rate: pd.Series, shrink_factor: float) -> pd.Series:
    raw = pd.to_numeric(raw_rate, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    shrink = float(np.clip(shrink_factor, 0.0, 1.0))
    return (0.5 + shrink * (raw - 0.5)).clip(lower=0.0, upper=1.0)


def apply_live_policy_calibration(selector_df: pd.DataFrame, policy_payload: dict) -> pd.DataFrame:
    if selector_df.empty:
        return selector_df.copy()

    out = selector_df.copy()
    out["raw_expected_win_rate"] = pd.to_numeric(out["expected_win_rate"], errors="coerce").fillna(0.5)
    out["expected_push_rate"] = pd.to_numeric(out.get("expected_push_rate"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    non_push = np.clip(1.0 - out["expected_push_rate"], 0.0, 1.0)

    # Single-stage probability policy: keep selector output as the canonical
    # calibrated probability and avoid additional generic shrink-to-0.5 passes.
    out["p_selector"] = out["raw_expected_win_rate"].clip(lower=0.0, upper=1.0)
    out["p_calibrated"] = out["p_selector"].clip(lower=0.0, upper=non_push)
    out["policy_calibration_weight"] = 1.0
    out["policy_calibration_source"] = "single_stage_selector"
    out["expected_win_rate"] = out["p_calibrated"]
    out["expected_loss_rate"] = np.clip(non_push - out["expected_win_rate"], 0.0, 1.0)
    elite_pct = float(policy_payload["elite_pct"])
    out["recommendation"] = np.where(
        pd.to_numeric(out["gap_percentile"], errors="coerce").fillna(0.0) >= elite_pct,
        "elite",
        out["recommendation"],
    )
    return out


def maybe_apply_xgb_ltr_reranker(selector_df: pd.DataFrame, history_df: pd.DataFrame, policy_payload: dict) -> tuple[pd.DataFrame, dict | None]:
    if selector_df.empty:
        return selector_df.copy(), None
    if str(policy_payload.get("ranking_mode", "ev_adjusted")) != "xgb_ltr":
        return selector_df.copy(), None
    if not HAS_XGB_LTR:
        out = selector_df.copy()
        out["xgb_ltr_score"] = np.nan
        out["xgb_ltr_enabled"] = False
        return out, {"enabled": False, "reason": "module_missing"}
    if history_df.empty:
        out = selector_df.copy()
        out["xgb_ltr_score"] = np.nan
        out["xgb_ltr_enabled"] = False
        return out, {"enabled": False, "reason": "empty_history"}
    return score_selector_with_xgb_ltr(
        selector_df,
        history_df,
        min_train_rows=int(policy_payload.get("xgb_ltr_min_train_rows", 4000)),
        num_pair_per_sample=int(policy_payload.get("xgb_ltr_num_pair_per_sample", 12)),
    )


def maybe_apply_accept_rejector(selector_df: pd.DataFrame, history_df: pd.DataFrame, policy_payload: dict) -> tuple[pd.DataFrame, dict | None]:
    if selector_df.empty:
        return selector_df.copy(), None
    if not bool(policy_payload.get("accept_reject_enabled", False)):
        return selector_df.copy(), None
    if not HAS_ACCEPT_REJECT:
        out = selector_df.copy()
        out["accept_reject_probability"] = np.nan
        out["accept_reject_enabled"] = False
        return out, {"enabled": False, "reason": "module_missing"}
    return apply_acceptor_to_selector(
        selector_df,
        history_df,
        probability_shrink_factor=float(policy_payload.get("probability_shrink_factor", 0.75)),
        elite_pct=float(policy_payload.get("elite_pct", 0.95)),
        min_train_rows=int(policy_payload.get("accept_reject_min_train_rows", 3000)),
        holdout_days=int(policy_payload.get("accept_reject_holdout_days", 45)),
        min_holdout_rows=int(policy_payload.get("accept_reject_min_holdout_rows", 250)),
        min_accept_rate=float(policy_payload.get("accept_reject_min_accept_rate", 0.05)),
        threshold_floor=float(policy_payload.get("accept_reject_threshold_floor", 0.0)),
    )


def maybe_apply_robust_reranker(selector_df: pd.DataFrame, history_df: pd.DataFrame, policy_payload: dict) -> tuple[pd.DataFrame, dict | None]:
    if selector_df.empty:
        return selector_df.copy(), None
    if str(policy_payload.get("ranking_mode", "ev_adjusted")) != "robust_reranker":
        return selector_df.copy(), None
    if not HAS_ROBUST_RERANKER:
        out = selector_df.copy()
        out["robust_reranker_prob"] = np.nan
        out["robust_reranker_enabled"] = False
        return out, {"enabled": False, "reason": "module_missing"}
    return score_selector_with_robust_reranker(
        selector_df,
        history_df,
        probability_shrink_factor=float(policy_payload.get("probability_shrink_factor", 0.75)),
        elite_pct=float(policy_payload.get("elite_pct", 0.95)),
        min_train_rows=int(policy_payload.get("robust_reranker_min_train_rows", 4000)),
        holdout_days=int(policy_payload.get("robust_reranker_holdout_days", 45)),
        min_holdout_rows=int(policy_payload.get("robust_reranker_min_holdout_rows", 250)),
        num_pair_per_sample=int(policy_payload.get("robust_reranker_num_pair_per_sample", 12)),
        min_candidate_expected_win_rate=float(policy_payload.get("robust_reranker_min_candidate_win_rate", 0.55)),
        min_candidate_final_confidence=float(policy_payload.get("robust_reranker_min_candidate_final_confidence", 0.03)),
        min_candidate_recommendation=str(policy_payload.get("robust_reranker_min_candidate_recommendation", "consider")),
    )


def summarize_skip_reasons(skipped_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped_rows:
        reason = str(item.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_selected_board_calibrator(path: Path, disabled: bool) -> tuple[dict | None, dict]:
    if disabled:
        return None, {"enabled": False, "reason": "disabled_flag"}
    resolved = path.resolve()
    if not resolved.exists():
        return None, {"enabled": False, "reason": "missing_file", "path": str(resolved)}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        months = payload.get("months", {}) if isinstance(payload, dict) else {}
        return payload, {
            "enabled": True,
            "path": str(resolved),
            "months": int(len(months)) if isinstance(months, dict) else 0,
            "version": int(payload.get("version", 0)) if isinstance(payload, dict) else 0,
        }
    except Exception as exc:
        return None, {
            "enabled": False,
            "reason": "load_error",
            "path": str(resolved),
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_learned_pool_gate(path: Path, disabled: bool) -> tuple[dict | None, dict]:
    if disabled:
        return None, {"enabled": False, "reason": "disabled_flag"}
    resolved = path.resolve()
    if not resolved.exists():
        return None, {"enabled": False, "reason": "missing_file", "path": str(resolved)}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        months = payload.get("months", {}) if isinstance(payload, dict) else {}
        return payload, {
            "enabled": True,
            "path": str(resolved),
            "months": int(len(months)) if isinstance(months, dict) else 0,
            "version": int(payload.get("version", 0)) if isinstance(payload, dict) else 0,
            "recommended_min_rows": int(payload.get("recommended_min_rows", 0)) if isinstance(payload, dict) else 0,
        }
    except Exception as exc:
        return None, {
            "enabled": False,
            "reason": "load_error",
            "path": str(resolved),
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_accepted_pick_gate(path: Path, disabled: bool) -> tuple[dict | None, dict]:
    if disabled:
        return None, {"enabled": False, "reason": "disabled_flag"}
    resolved = path.resolve()
    if not resolved.exists():
        return None, {"enabled": False, "reason": "missing_file", "path": str(resolved)}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        months = payload.get("months", {}) if isinstance(payload, dict) else {}
        return payload, {
            "enabled": True,
            "path": str(resolved),
            "months": int(len(months)) if isinstance(months, dict) else 0,
            "version": int(payload.get("version", 0)) if isinstance(payload, dict) else 0,
            "threshold": float(payload.get("threshold", np.nan)) if isinstance(payload, dict) else np.nan,
        }
    except Exception as exc:
        return None, {
            "enabled": False,
            "reason": "load_error",
            "path": str(resolved),
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_staking_bucket_model(path: Path, disabled: bool) -> tuple[dict | None, dict]:
    if disabled:
        return None, {"enabled": False, "reason": "disabled_flag"}
    resolved = path.resolve()
    if not resolved.exists():
        return None, {"enabled": False, "reason": "missing_file", "path": str(resolved)}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        months = payload.get("months", {}) if isinstance(payload, dict) else {}
        return payload, {
            "enabled": True,
            "path": str(resolved),
            "months": int(len(months)) if isinstance(months, dict) else 0,
            "version": int(payload.get("version", 0)) if isinstance(payload, dict) else 0,
        }
    except Exception as exc:
        return None, {
            "enabled": False,
            "reason": "load_error",
            "path": str(resolved),
            "error": f"{type(exc).__name__}: {exc}",
        }


def validate_pipeline_inputs(market_df: pd.DataFrame, slate_df: pd.DataFrame, skipped_rows: list[dict]) -> dict:
    market_validation = {
        "market_rows": int(len(market_df)),
        "market_dates_non_null": int(market_df["Market_Date"].notna().sum()) if "Market_Date" in market_df.columns else 0,
        "unique_market_players": int(market_df["Player"].nunique()) if "Player" in market_df.columns else 0,
    }
    for target in TARGETS:
        market_validation[f"market_{target.lower()}_lines"] = int(pd.to_numeric(market_df.get(f"Market_{target}"), errors="coerce").notna().sum())
        market_validation[f"market_{target.lower()}_books"] = int(pd.to_numeric(market_df.get(f"Market_{target}_books"), errors="coerce").notna().sum())

    if slate_df.empty:
        prior_history_validation = {
            "slate_rows": 0,
            "history_rows_min": 0,
            "history_rows_median": 0.0,
            "history_rows_max": 0,
            "last_history_date_non_null": 0,
            "history_before_market_ok_rows": 0,
            "history_before_market_violations": 0,
            "csv_exists_rows": 0,
        }
    else:
        history_rows = pd.to_numeric(slate_df.get("history_rows"), errors="coerce").fillna(0)
        last_history = pd.to_datetime(slate_df.get("last_history_date"), errors="coerce")
        market_dates = pd.to_datetime(slate_df.get("market_date"), errors="coerce")
        history_before_market = (last_history < market_dates)
        csv_exists = slate_df.get("csv", pd.Series("", index=slate_df.index)).astype(str).map(lambda item: Path(item).exists())
        prior_history_validation = {
            "slate_rows": int(len(slate_df)),
            "history_rows_min": int(history_rows.min()) if len(history_rows) else 0,
            "history_rows_median": float(history_rows.median()) if len(history_rows) else 0.0,
            "history_rows_max": int(history_rows.max()) if len(history_rows) else 0,
            "last_history_date_non_null": int(last_history.notna().sum()),
            "history_before_market_ok_rows": int(history_before_market.fillna(False).sum()),
            "history_before_market_violations": int((~history_before_market.fillna(False)).sum()),
            "csv_exists_rows": int(csv_exists.sum()),
        }

    return {
        "market_lines": market_validation,
        "prior_game_data": prior_history_validation,
        "skipped_rows": {
            "count": int(len(skipped_rows)),
            "reasons": summarize_skip_reasons(skipped_rows),
            "sample": skipped_rows[:10],
        },
    }


def main() -> None:
    args = parse_args()
    if args.enable_learned_gate and args.disable_learned_gate:
        raise ValueError("Cannot pass both --enable-learned-gate and --disable-learned-gate.")
    if args.enable_accepted_pick_gate and args.disable_accepted_pick_gate:
        raise ValueError("Cannot pass both --enable-accepted-pick-gate and --disable-accepted-pick-gate.")
    if args.accepted_pick_gate_live and args.accepted_pick_gate_shadow:
        raise ValueError("Cannot pass both --accepted-pick-gate-live and --accepted-pick-gate-shadow.")
    if args.enable_staking_bucket_model and args.disable_staking_bucket_model:
        raise ValueError("Cannot pass both --enable-staking-bucket-model and --disable-staking-bucket-model.")
    if args.board_objective_instability_enabled and args.board_objective_instability_disabled:
        raise ValueError("Cannot pass both --board-objective-instability-enabled and --board-objective-instability-disabled.")
    if args.board_objective_instability_veto_enabled and args.board_objective_instability_veto_disabled:
        raise ValueError("Cannot pass both --board-objective-instability-veto-enabled and --board-objective-instability-veto-disabled.")
    if args.board_objective_dynamic_size_enabled and args.board_objective_dynamic_size_disabled:
        raise ValueError("Cannot pass both --board-objective-dynamic-size-enabled and --board-objective-dynamic-size-disabled.")
    policy_payload = resolve_policy(args)

    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor: StructuredStackInference | None = None
    predictor_error = None
    try:
        predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    except Exception as exc:
        predictor_error = f"{type(exc).__name__}: {exc}"
        if not args.allow_heuristic_fallback:
            raise RuntimeError(
                "Model inference failed while heuristic fallback is disabled. "
                "Pass --allow-heuristic-fallback to continue anyway. "
                f"Root cause: {predictor_error}"
            ) from exc
        print(f"Warning: model inference unavailable, using heuristic fallback only ({predictor_error})")

    market_df = load_market_wide(args.market_wide_path)
    calibrator_path = None if args.disable_target_prediction_calibration else args.target_prediction_calibrator_json
    slate_records, slate_skipped = build_records(
        predictor,
        market_df,
        args.season,
        target_prediction_calibrator_path=calibrator_path,
    )
    if not slate_records:
        raise RuntimeError(f"No upcoming slate rows built. Skipped={len(slate_skipped)} sample={slate_skipped[:5]}")

    slate_df = pd.DataFrame.from_records(slate_records).sort_values(["market_date", "player"]).reset_index(drop=True)
    input_validation = validate_pipeline_inputs(market_df, slate_df, slate_skipped)

    args.slate_csv_out.parent.mkdir(parents=True, exist_ok=True)
    slate_df.to_csv(args.slate_csv_out, index=False)

    history_path = args.history_csv.resolve()
    history_df = pd.DataFrame()
    history_lookup: dict[str, dict] = {}
    history_mode = "historical_backtest"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        if history_df.empty:
            history_mode = "heuristic_fallback_empty_history"
            print(f"Warning: history CSV is empty ({history_path}); using heuristic edge calibration.")
            policy_payload = apply_heuristic_policy_overrides(policy_payload)
        else:
            history_lookup = build_history_lookup(history_df)
            if not history_lookup:
                history_mode = "heuristic_fallback_unusable_history"
                print(f"Warning: history CSV has no usable calibration rows ({history_path}); using heuristic edge calibration.")
                policy_payload = apply_heuristic_policy_overrides(policy_payload)
    else:
        history_mode = "heuristic_fallback"
        print(f"Warning: history CSV not found ({history_path}); using heuristic edge calibration.")
        policy_payload = apply_heuristic_policy_overrides(policy_payload)
    policy_payload = apply_weak_play_capacity_overrides(policy_payload)
    selector_df = build_play_rows(
        slate_df,
        history_lookup,
        belief_uncertainty_lower=float(policy_payload.get("belief_uncertainty_lower", 0.75)),
        belief_uncertainty_upper=float(policy_payload.get("belief_uncertainty_upper", 1.15)),
        market_regression_floor=float(policy_payload.get("market_regression_floor", 0.25)),
        market_regression_ceiling=float(policy_payload.get("market_regression_ceiling", 0.95)),
    )
    selector_df = apply_live_policy_calibration(selector_df, policy_payload)
    selector_df, xgb_ltr_summary = maybe_apply_xgb_ltr_reranker(selector_df, history_df, policy_payload)
    selector_df, accept_reject_summary = maybe_apply_accept_rejector(selector_df, history_df, policy_payload)
    selector_df, robust_reranker_summary = maybe_apply_robust_reranker(selector_df, history_df, policy_payload)
    conditional_summary: dict | None = None
    try:
        selector_df, conditional_summary = apply_conditional_promotion(
            selector_df=selector_df,
            policy_payload=policy_payload,
            history_df=history_df,
            american_odds=int(policy_payload.get("american_odds", -110)),
        )
    except Exception as exc:
        conditional_summary = {
            "fallback_mode": "C_baseline_only",
            "fallback_reasons": ["conditional_framework_exception"],
            "error": f"{type(exc).__name__}: {exc}",
        }
        selector_df = selector_df.copy()
        selector_df["conditional_eligible_for_board"] = True
        selector_df["conditional_promoted"] = False
        selector_df["decision_tier"] = "Tier A - Baseline"
        selector_df["weak_bucket"] = "K"
        selector_df["conditional_audit_summary"] = "Conditional layer failed; fell back to baseline-only mode."
    if selector_df.empty:
        raise RuntimeError("Selector produced no rows from the current slate.")
    args.selector_csv_out.parent.mkdir(parents=True, exist_ok=True)
    selector_df.to_csv(args.selector_csv_out, index=False)

    selected_board_calibrator, selected_board_calibrator_summary = load_selected_board_calibrator(
        args.selected_board_calibrator_json,
        disabled=bool(args.disable_selected_board_calibration),
    )
    learned_gate_disabled = bool(not policy_payload.get("learned_gate_enabled", False))
    learned_pool_gate, learned_pool_gate_summary = load_learned_pool_gate(
        args.learned_gate_json,
        disabled=learned_gate_disabled,
    )
    accepted_pick_gate_disabled = bool(not policy_payload.get("accepted_pick_gate_enabled", False))
    accepted_pick_gate, accepted_pick_gate_summary = load_accepted_pick_gate(
        args.accepted_pick_gate_json,
        disabled=accepted_pick_gate_disabled,
    )
    staking_bucket_model_disabled = bool(not policy_payload.get("staking_bucket_model_enabled", False))
    staking_bucket_model, staking_bucket_model_summary = load_staking_bucket_model(
        args.staking_bucket_model_json,
        disabled=staking_bucket_model_disabled,
    )

    requested_policy_payload = dict(policy_payload)

    def _compute_final_board_for_policy(active_policy: dict) -> pd.DataFrame:
        return compute_final_board(
            selector_df,
            american_odds=active_policy["american_odds"],
            min_ev=active_policy["min_ev"],
            min_final_confidence=active_policy["min_final_confidence"],
            min_recommendation=active_policy["min_recommendation"],
            selection_mode=active_policy.get("selection_mode", active_policy.get("ranking_mode", "ev_adjusted")),
            ranking_mode=active_policy.get("ranking_mode", "ev_adjusted"),
            max_plays_per_player=active_policy["max_plays_per_player"],
            max_plays_per_target=active_policy["max_plays_per_target"],
            max_total_plays=active_policy["max_total_plays"],
            min_board_plays=active_policy.get("min_board_plays", 0),
            max_target_plays={"PTS": active_policy["max_pts_plays"], "TRB": active_policy["max_trb_plays"], "AST": active_policy["max_ast_plays"]},
            max_plays_per_game=active_policy.get("max_plays_per_game", 2),
            max_plays_per_script_cluster=active_policy.get("max_plays_per_script_cluster", 2),
            non_pts_min_gap_percentile=active_policy["non_pts_min_gap_percentile"],
            exclude_micro_lines_enabled=bool(active_policy.get("exclude_micro_lines_enabled", True)),
            exclude_micro_line_targets=tuple(
                str(token).upper().strip()
                for token in active_policy.get("exclude_micro_line_targets", ("PTS", "TRB", "AST"))
                if str(token).strip()
            ),
            exclude_micro_line_min=float(active_policy.get("exclude_micro_line_min", 0.5)),
            exclude_micro_line_max=float(active_policy.get("exclude_micro_line_max", 1.5)),
            edge_adjust_k=active_policy["edge_adjust_k"],
            thompson_temperature=active_policy.get("thompson_temperature", 1.0),
            thompson_seed=active_policy.get("thompson_seed", 17),
            min_bet_win_rate=active_policy.get("min_bet_win_rate", 0.57),
            medium_bet_win_rate=active_policy.get("medium_bet_win_rate", 0.60),
            full_bet_win_rate=active_policy.get("full_bet_win_rate", 0.65),
            medium_tier_percentile=active_policy.get("medium_tier_percentile", 0.80),
            strong_tier_percentile=active_policy.get("strong_tier_percentile", 0.90),
            elite_tier_percentile=active_policy.get("elite_tier_percentile", active_policy.get("elite_pct", 0.95)),
            small_bet_fraction=active_policy.get("small_bet_fraction", 0.005),
            medium_bet_fraction=active_policy.get("medium_bet_fraction", 0.010),
            full_bet_fraction=active_policy.get("full_bet_fraction", 0.015),
            max_bet_fraction=active_policy.get("max_bet_fraction", 0.02),
            max_total_bet_fraction=active_policy.get("max_total_bet_fraction", 0.05),
            sizing_method=active_policy.get("sizing_method", "tiered_probability"),
            flat_bet_fraction=active_policy.get(
                "flat_bet_fraction",
                active_policy.get("base_bet_fraction", active_policy.get("small_bet_fraction", 0.005)),
            ),
            coarse_low_bet_fraction=active_policy.get("coarse_low_bet_fraction", 0.003),
            coarse_mid_bet_fraction=active_policy.get("coarse_mid_bet_fraction", 0.005),
            coarse_high_bet_fraction=active_policy.get("coarse_high_bet_fraction", 0.007),
            coarse_high_max_share=active_policy.get("coarse_high_max_share", 0.30),
            coarse_mid_max_share=active_policy.get("coarse_mid_max_share", 0.50),
            coarse_high_max_plays=active_policy.get("coarse_high_max_plays", 0),
            coarse_mid_max_plays=active_policy.get("coarse_mid_max_plays", 0),
            coarse_score_alpha_uncertainty=active_policy.get("coarse_score_alpha_uncertainty", 0.18),
            coarse_score_beta_dependency=active_policy.get("coarse_score_beta_dependency", 0.12),
            coarse_score_gamma_support=active_policy.get("coarse_score_gamma_support", 0.08),
            coarse_score_model=active_policy.get("coarse_score_model", "legacy"),
            coarse_score_delta_prob_weight=active_policy.get("coarse_score_delta_prob_weight", 0.0),
            coarse_score_ev_weight=active_policy.get("coarse_score_ev_weight", 0.0),
            coarse_score_risk_weight=active_policy.get("coarse_score_risk_weight", 0.0),
            coarse_score_recency_weight=active_policy.get("coarse_score_recency_weight", 0.0),
            staking_bucket_model_payload=staking_bucket_model,
            staking_bucket_model_month=args.staking_bucket_model_month,
            staking_bucket_model_min_rows=int(active_policy.get("staking_bucket_model_min_rows", 0)),
            belief_uncertainty_lower=active_policy.get("belief_uncertainty_lower", 0.75),
            belief_uncertainty_upper=active_policy.get("belief_uncertainty_upper", 1.15),
            append_agreement_min=active_policy.get("append_agreement_min", 3),
            append_edge_percentile_min=active_policy.get("append_edge_percentile_min", 0.90),
            append_max_extra_plays=active_policy.get("append_max_extra_plays", 3),
            board_objective_overfetch=active_policy.get("board_objective_overfetch", 4.0),
            board_objective_candidate_limit=active_policy.get("board_objective_candidate_limit", 36),
            board_objective_max_search_nodes=active_policy.get("board_objective_max_search_nodes", 750000),
            board_objective_lambda_corr=active_policy.get("board_objective_lambda_corr", 0.12),
            board_objective_lambda_conc=active_policy.get("board_objective_lambda_conc", 0.07),
            board_objective_lambda_unc=active_policy.get("board_objective_lambda_unc", 0.06),
            board_objective_corr_same_game=active_policy.get("board_objective_corr_same_game", 0.65),
            board_objective_corr_same_player=active_policy.get("board_objective_corr_same_player", 1.0),
            board_objective_corr_same_target=active_policy.get("board_objective_corr_same_target", 0.15),
            board_objective_corr_same_direction=active_policy.get("board_objective_corr_same_direction", 0.05),
            board_objective_corr_same_script_cluster=active_policy.get("board_objective_corr_same_script_cluster", 0.30),
            board_objective_swap_candidates=active_policy.get("board_objective_swap_candidates", 18),
            board_objective_swap_rounds=active_policy.get("board_objective_swap_rounds", 2),
            board_objective_instability_enabled=bool(active_policy.get("board_objective_instability_enabled", False)),
            board_objective_lambda_shadow_disagreement=active_policy.get("board_objective_lambda_shadow_disagreement", 0.0),
            board_objective_lambda_segment_weakness=active_policy.get("board_objective_lambda_segment_weakness", 0.0),
            board_objective_instability_near_cutoff_window=active_policy.get("board_objective_instability_near_cutoff_window", 3),
            board_objective_instability_top_protected=active_policy.get("board_objective_instability_top_protected", 3),
            board_objective_instability_veto_enabled=bool(active_policy.get("board_objective_instability_veto_enabled", False)),
            board_objective_instability_veto_quantile=active_policy.get("board_objective_instability_veto_quantile", 0.85),
            board_objective_dynamic_size_enabled=bool(active_policy.get("board_objective_dynamic_size_enabled", False)),
            board_objective_dynamic_size_max_shrink=active_policy.get("board_objective_dynamic_size_max_shrink", 0),
            board_objective_dynamic_size_trigger=active_policy.get("board_objective_dynamic_size_trigger", 0.62),
            board_objective_fp_veto_enabled=bool(active_policy.get("board_objective_fp_veto_enabled", False)),
            board_objective_fp_veto_live=bool(active_policy.get("board_objective_fp_veto_live", False)),
            board_objective_fp_veto_tail_slots=active_policy.get("board_objective_fp_veto_tail_slots", 2),
            board_objective_fp_veto_top_protected=active_policy.get("board_objective_fp_veto_top_protected", 6),
            board_objective_fp_veto_threshold=active_policy.get("board_objective_fp_veto_threshold", 0.80),
            board_objective_fp_veto_max_drops=active_policy.get("board_objective_fp_veto_max_drops", 1),
            board_objective_fp_veto_quantile=active_policy.get("board_objective_fp_veto_quantile", 0.70),
            board_objective_fp_veto_max_swaps=active_policy.get("board_objective_fp_veto_max_swaps", 1),
            board_objective_fp_veto_swap_candidates=active_policy.get("board_objective_fp_veto_swap_candidates", 24),
            board_objective_fp_veto_min_swap_gain=active_policy.get("board_objective_fp_veto_min_swap_gain", 0.0025),
            board_objective_fp_veto_risk_lambda=active_policy.get("board_objective_fp_veto_risk_lambda", 0.18),
            board_objective_fp_veto_ml_weight=active_policy.get("board_objective_fp_veto_ml_weight", 0.45),
            max_history_staleness_days=active_policy.get("max_history_staleness_days", 0),
            min_recency_factor=active_policy.get("min_recency_factor", 0.0),
            selected_board_calibrator=selected_board_calibrator,
            selected_board_calibration_month=args.selected_board_calibration_month,
            learned_gate_payload=learned_pool_gate,
            learned_gate_month=args.learned_gate_month,
            learned_gate_min_rows=int(active_policy.get("learned_gate_min_rows", 0)),
            initial_pool_gate_enabled=bool(active_policy.get("initial_pool_gate_enabled", True)),
            initial_pool_gate_drop_fraction=float(active_policy.get("initial_pool_gate_drop_fraction", 0.10)),
            initial_pool_gate_score_col=str(active_policy.get("initial_pool_gate_score_col", "selector_expected_win_rate")),
            initial_pool_gate_min_keep_rows=int(active_policy.get("initial_pool_gate_min_keep_rows", 20)),
            accepted_pick_gate_payload=accepted_pick_gate,
            accepted_pick_gate_month=args.accepted_pick_gate_month,
            accepted_pick_gate_enabled=bool(active_policy.get("accepted_pick_gate_enabled", False)),
            accepted_pick_gate_live=bool(active_policy.get("accepted_pick_gate_live", False)),
            accepted_pick_gate_min_rows=int(active_policy.get("accepted_pick_gate_min_rows", 0)),
            selector_pool_append_max_rows=int(active_policy.get("selector_pool_append_max_rows", 0)),
            selector_pool_append_rank_window=int(active_policy.get("selector_pool_append_rank_window", 24)),
        )

    effective_policy_payload = dict(requested_policy_payload)
    board_fallback_reason = None
    final_board = _compute_final_board_for_policy(effective_policy_payload)
    if final_board.empty and not selector_df.empty:
        fallback_policy_payload = apply_weak_play_capacity_overrides(apply_heuristic_policy_overrides(requested_policy_payload))
        fallback_board = _compute_final_board_for_policy(fallback_policy_payload)
        if not fallback_board.empty:
            final_board = fallback_board
            effective_policy_payload = fallback_policy_payload
            board_fallback_reason = "empty_final_board_relaxed_to_selector_friendly_policy"
            print(
                "[warning] Final NBA board was empty under the requested policy; "
                "publishing a relaxed fallback board from the selector pool instead."
            )
    policy_payload = effective_policy_payload
    board_stage_counts_raw = {
        str(key): int(value)
        for key, value in getattr(final_board, "attrs", {}).get("stage_counts", {}).items()
    }
    pipeline_stage_counts = {
        "raw_market_rows": int(input_validation["market_lines"]["market_rows"]),
        "slate_rows": int(len(slate_df)),
        "selector_rows": int(len(selector_df)),
        "after_initial_pool_gate": int(board_stage_counts_raw.get("after_initial_pool_gate", len(selector_df))),
        "after_recency": int(board_stage_counts_raw.get("after_recency", len(selector_df))),
        "after_confidence": int(board_stage_counts_raw.get("after_confidence", len(selector_df))),
        "after_min_ev": int(board_stage_counts_raw.get("after_min_ev", len(selector_df))),
        "after_learned_gate": int(board_stage_counts_raw.get("after_learned_gate", len(selector_df))),
        "candidate_universe": int(board_stage_counts_raw.get("candidate_universe", board_stage_counts_raw.get("after_learned_gate", len(selector_df)))),
        "after_accepted_pick_gate": int(board_stage_counts_raw.get("after_accepted_pick_gate", len(final_board))),
        "final_board_rows": int(board_stage_counts_raw.get("final_board_rows", len(final_board))),
    }
    args.final_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.final_json_out.parent.mkdir(parents=True, exist_ok=True)
    final_board.to_csv(args.final_csv_out, index=False)
    promoted_mask = (
        pd.to_numeric(final_board["conditional_promoted"], errors="coerce").fillna(0).astype(bool)
        if "conditional_promoted" in final_board.columns
        else pd.Series(False, index=final_board.index)
    )
    tier_a_board = final_board.loc[~promoted_mask].copy()
    tier_b_board = final_board.loc[promoted_mask].copy()
    tier_a_csv = args.final_csv_out.with_name(f"{args.final_csv_out.stem}_tier_a{args.final_csv_out.suffix}")
    tier_b_csv = args.final_csv_out.with_name(f"{args.final_csv_out.stem}_tier_b{args.final_csv_out.suffix}")
    tier_a_board.to_csv(tier_a_csv, index=False)
    tier_b_board.to_csv(tier_b_csv, index=False)

    requested_board_size = int(policy_payload.get("max_total_plays", 0))
    final_board_size = int(len(final_board))
    board_diagnostics = {
        "requested_board_size": requested_board_size,
        "selector_pool_size": int(len(selector_df)),
        "final_board_size": final_board_size,
        "size_deficit": int(max(0, requested_board_size - final_board_size)) if requested_board_size > 0 else 0,
        "deficit_fill_active": bool(requested_board_size > 0 and final_board_size < requested_board_size),
        "avg_expected_wins_per_board": float(pd.to_numeric(final_board.get("expected_win_rate"), errors="coerce").fillna(0.0).sum()),
        "instability_enabled": bool(pd.to_numeric(final_board.get("board_objective_instability_enabled"), errors="coerce").fillna(0).astype(bool).any())
        if "board_objective_instability_enabled" in final_board.columns
        else False,
        "instability_mean_score": float(pd.to_numeric(final_board.get("board_instability_score"), errors="coerce").fillna(np.nan).mean())
        if "board_instability_score" in final_board.columns
        else np.nan,
        "dynamic_size_shrink": int(pd.to_numeric(final_board.get("board_objective_dynamic_shrink"), errors="coerce").fillna(0).max())
        if "board_objective_dynamic_shrink" in final_board.columns and not final_board.empty
        else 0,
        "dynamic_size_target": int(pd.to_numeric(final_board.get("board_objective_dynamic_target_size"), errors="coerce").fillna(final_board_size).max())
        if "board_objective_dynamic_target_size" in final_board.columns and not final_board.empty
        else final_board_size,
        "initial_pool_gate_enabled": bool(pd.to_numeric(final_board.get("initial_pool_gate_enabled"), errors="coerce").fillna(0).astype(bool).any())
        if "initial_pool_gate_enabled" in final_board.columns and not final_board.empty
        else bool(policy_payload.get("initial_pool_gate_enabled", True)),
        "initial_pool_gate_applied": bool(pd.to_numeric(final_board.get("initial_pool_gate_applied"), errors="coerce").fillna(0).astype(bool).any())
        if "initial_pool_gate_applied" in final_board.columns and not final_board.empty
        else False,
        "initial_pool_gate_drop_fraction": float(pd.to_numeric(final_board.get("initial_pool_gate_drop_fraction"), errors="coerce").fillna(np.nan).max())
        if "initial_pool_gate_drop_fraction" in final_board.columns and not final_board.empty
        else float(policy_payload.get("initial_pool_gate_drop_fraction", 0.10)),
        "initial_pool_gate_rows_before": int(pd.to_numeric(final_board.get("initial_pool_gate_rows_before"), errors="coerce").fillna(np.nan).max())
        if "initial_pool_gate_rows_before" in final_board.columns and not final_board.empty
        else int(len(selector_df)),
        "initial_pool_gate_rows_after": int(pd.to_numeric(final_board.get("initial_pool_gate_rows_after"), errors="coerce").fillna(np.nan).max())
        if "initial_pool_gate_rows_after" in final_board.columns and not final_board.empty
        else int(len(selector_df)),
        "initial_pool_gate_dropped_rows": int(pd.to_numeric(final_board.get("initial_pool_gate_dropped_rows"), errors="coerce").fillna(0).max())
        if "initial_pool_gate_dropped_rows" in final_board.columns and not final_board.empty
        else 0,
        "fp_veto_enabled": bool(pd.to_numeric(final_board.get("board_objective_fp_veto_enabled"), errors="coerce").fillna(0).astype(bool).any())
        if "board_objective_fp_veto_enabled" in final_board.columns and not final_board.empty
        else False,
        "fp_veto_live": bool(pd.to_numeric(final_board.get("board_objective_fp_veto_live"), errors="coerce").fillna(0).astype(bool).any())
        if "board_objective_fp_veto_live" in final_board.columns and not final_board.empty
        else False,
        "fp_veto_drop_count": int(pd.to_numeric(final_board.get("board_objective_fp_veto_drop_count"), errors="coerce").fillna(0).max())
        if "board_objective_fp_veto_drop_count" in final_board.columns and not final_board.empty
        else 0,
        "fp_veto_swap_count": int(pd.to_numeric(final_board.get("board_objective_fp_veto_swap_count"), errors="coerce").fillna(0).max())
        if "board_objective_fp_veto_swap_count" in final_board.columns and not final_board.empty
        else 0,
        "fp_veto_flagged_share": float(pd.to_numeric(final_board.get("board_objective_fp_veto_flagged"), errors="coerce").fillna(0).mean())
        if "board_objective_fp_veto_flagged" in final_board.columns and not final_board.empty
        else 0.0,
        "fp_veto_swap_selected_share": float(pd.to_numeric(final_board.get("board_objective_fp_veto_swap_selected"), errors="coerce").fillna(0).mean())
        if "board_objective_fp_veto_swap_selected" in final_board.columns and not final_board.empty
        else 0.0,
        "fp_veto_threshold_effective": float(pd.to_numeric(final_board.get("board_objective_fp_veto_threshold_effective"), errors="coerce").fillna(np.nan).mean())
        if "board_objective_fp_veto_threshold_effective" in final_board.columns and not final_board.empty
        else np.nan,
        "accepted_pick_gate_enabled": bool(pd.to_numeric(final_board.get("accepted_pick_gate_enabled"), errors="coerce").fillna(0).astype(bool).any())
        if "accepted_pick_gate_enabled" in final_board.columns and not final_board.empty
        else False,
        "accepted_pick_gate_enforced": bool(pd.to_numeric(final_board.get("accepted_pick_gate_enforced"), errors="coerce").fillna(0).astype(bool).any())
        if "accepted_pick_gate_enforced" in final_board.columns and not final_board.empty
        else False,
        "accepted_pick_gate_live": bool(pd.to_numeric(final_board.get("accepted_pick_gate_live"), errors="coerce").fillna(0).astype(bool).any())
        if "accepted_pick_gate_live" in final_board.columns and not final_board.empty
        else False,
        "accepted_pick_gate_veto_share": float(pd.to_numeric(final_board.get("accepted_pick_gate_veto"), errors="coerce").fillna(0).mean())
        if "accepted_pick_gate_veto" in final_board.columns and not final_board.empty
        else 0.0,
        "accepted_pick_gate_drop_count": int(pd.to_numeric(final_board.get("accepted_pick_gate_drop_count"), errors="coerce").fillna(0).max())
        if "accepted_pick_gate_drop_count" in final_board.columns and not final_board.empty
        else 0,
        "accepted_pick_gate_threshold": float(pd.to_numeric(final_board.get("accepted_pick_gate_threshold"), errors="coerce").fillna(np.nan).mean())
        if "accepted_pick_gate_threshold" in final_board.columns and not final_board.empty
        else np.nan,
        "stage_counts": pipeline_stage_counts,
    }

    payload = {
        "manifest_path": str(manifest_path),
        "run_id": predictor.metadata.get("run_id") if predictor is not None else None,
        "predictor_error": predictor_error,
        "market_snapshot": str(args.market_wide_path),
        "history_csv": str(history_path),
        "history_mode": history_mode,
        "board_fallback_reason": board_fallback_reason,
        "target_prediction_calibrator_json": str(calibrator_path) if calibrator_path is not None else None,
        "target_prediction_calibration_enabled": bool(not args.disable_target_prediction_calibration),
        "season": args.season,
        "policy_profile": args.policy_profile,
        "policy": policy_payload,
        "requested_policy": requested_policy_payload,
        "xgb_ltr": xgb_ltr_summary,
        "accept_reject": accept_reject_summary,
        "robust_reranker": robust_reranker_summary,
        "conditional_framework": conditional_summary,
        "selected_board_calibrator": selected_board_calibrator_summary,
        "learned_pool_gate": learned_pool_gate_summary,
        "accepted_pick_gate": accepted_pick_gate_summary,
        "staking_bucket_model": staking_bucket_model_summary,
        "slate_rows": int(len(slate_df)),
        "selector_rows": int(len(selector_df)),
        "final_rows": int(len(final_board)),
        "tier_a_rows": int(len(tier_a_board)),
        "tier_b_rows": int(len(tier_b_board)),
        "tier_a_csv": str(tier_a_csv),
        "tier_b_csv": str(tier_b_csv),
        "board_diagnostics": board_diagnostics,
        "pipeline_stage_counts": pipeline_stage_counts,
        "input_validation": input_validation,
        "top_plays": final_board.head(20).to_dict(orient="records"),
        "tier_a_top_plays": tier_a_board.head(20).to_dict(orient="records"),
        "tier_b_top_plays": tier_b_board.head(20).to_dict(orient="records"),
    }
    args.final_json_out.write_text(json.dumps(sanitize_for_json(payload), indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("MARKET PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Run id:        {predictor.metadata.get('run_id') if predictor is not None else 'n/a'}")
    print(f"Policy:        {args.policy_profile}")
    print(f"Slate rows:    {len(slate_df)}")
    print(f"Selector rows: {len(selector_df)}")
    print(f"Final rows:    {len(final_board)}")
    print(f"Tier A rows:   {len(tier_a_board)}")
    print(f"Tier B rows:   {len(tier_b_board)}")
    print(f"Slate CSV:     {args.slate_csv_out}")
    print(f"Selector CSV:  {args.selector_csv_out}")
    print(f"Final CSV:     {args.final_csv_out}")
    print(f"Tier A CSV:    {tier_a_csv}")
    print(f"Tier B CSV:    {tier_b_csv}")
    print(f"Final JSON:    {args.final_json_out}")
    print(f"Selected-board calibrator: {selected_board_calibrator_summary}")
    print(f"Learned pool gate: {learned_pool_gate_summary}")
    print(f"Accepted-pick gate: {accepted_pick_gate_summary}")
    print(f"Board diagnostics: {board_diagnostics}")
    print("Stage counts:")
    for stage_name in [
        "raw_market_rows",
        "slate_rows",
        "selector_rows",
        "after_initial_pool_gate",
        "after_recency",
        "after_confidence",
        "after_min_ev",
        "after_learned_gate",
        "candidate_universe",
        "after_accepted_pick_gate",
        "final_board_rows",
    ]:
        print(f"  {stage_name}: {pipeline_stage_counts[stage_name]}")
    print("Input validation:")
    print(f"  Market rows:        {input_validation['market_lines']['market_rows']}")
    print(f"  Prior-history rows: {input_validation['prior_game_data']['slate_rows']}")
    print(f"  History violations: {input_validation['prior_game_data']['history_before_market_violations']}")
    print(f"  Skipped rows:       {input_validation['skipped_rows']['count']}")
    if not final_board.empty:
        show_cols = [
            "player",
            "target",
            "direction",
            "prediction",
            "market_line",
            "abs_edge",
            "raw_expected_win_rate",
            "expected_win_rate",
            "ev",
            "final_confidence",
            "allocation_tier",
            "allocation_action",
            "bet_fraction",
            "recommendation",
        ]
        present_cols = [column for column in show_cols if column in final_board.columns]
        print("\nFinal plays:")
        print(final_board[present_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
