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

from build_upcoming_slate import MODEL_DIR, build_records, load_market_wide, resolve_manifest_path
from decision_engine.accept_reject_model import apply_acceptor_to_selector
from decision_engine.policy_tuning import build_default_shadow_strategies
from decision_engine.robust_reranker import score_selector_with_robust_reranker
from decision_engine.xgb_ltr_reranker import score_selector_with_xgb_ltr
from post_process_market_plays import compute_final_board
from select_market_plays import build_history_lookup, build_play_rows
from structured_stack_inference import StructuredStackInference


POLICY_PROFILES = {config.name: config for config in build_default_shadow_strategies()}
DEFAULT_POLICY = POLICY_PROFILES["production_calibrated"]
TARGETS = ["PTS", "TRB", "AST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the market slate -> selector -> final board pipeline.")
    parser.add_argument("--season", type=int, required=True, help="Season end year, e.g. 2026.")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production.")
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="production_calibrated",
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
    parser.add_argument("--american-odds", type=int, default=None, help="Assumed American odds for EV.")
    parser.add_argument("--probability-shrink-factor", type=float, default=None, help="Shrink expected win rate toward 50%%.")
    parser.add_argument("--elite-pct", type=float, default=None, help="Percentile cutoff for elite priority plays.")
    parser.add_argument("--min-ev", type=float, default=None, help="Minimum EV to keep a play.")
    parser.add_argument("--min-final-confidence", type=float, default=None, help="Minimum final confidence to keep a play.")
    parser.add_argument(
        "--min-recommendation",
        type=str,
        default=None,
        choices=["consider", "strong", "elite"],
        help="Lowest selector recommendation allowed into the final board.",
    )
    parser.add_argument("--max-plays-per-player", type=int, default=None, help="Maximum final plays to keep per player.")
    parser.add_argument("--max-plays-per-target", type=int, default=None, help="Maximum final plays to keep per target when target-specific caps are not supplied.")
    parser.add_argument("--max-pts-plays", type=int, default=None, help="Maximum final PTS plays to keep.")
    parser.add_argument("--max-trb-plays", type=int, default=None, help="Maximum final TRB plays to keep.")
    parser.add_argument("--max-ast-plays", type=int, default=None, help="Maximum final AST plays to keep.")
    parser.add_argument("--max-total-plays", type=int, default=None, help="Maximum total plays to keep.")
    parser.add_argument("--non-pts-min-gap-percentile", type=float, default=None, help="Minimum disagreement percentile for TRB/AST plays.")
    parser.add_argument("--edge-adjust-k", type=float, default=None, help="Weight for edge-adjusted EV ranking.")
    parser.add_argument("--belief-uncertainty-lower", type=float, default=None, help="Lower anchor for belief uncertainty confidence scaling.")
    parser.add_argument("--belief-uncertainty-upper", type=float, default=None, help="Upper anchor for belief uncertainty confidence scaling.")
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
        "min_recommendation": args.min_recommendation,
        "max_plays_per_player": args.max_plays_per_player,
        "max_plays_per_target": args.max_plays_per_target,
        "max_pts_plays": args.max_pts_plays,
        "max_trb_plays": args.max_trb_plays,
        "max_ast_plays": args.max_ast_plays,
        "max_total_plays": args.max_total_plays,
        "non_pts_min_gap_percentile": args.non_pts_min_gap_percentile,
        "edge_adjust_k": args.edge_adjust_k,
        "belief_uncertainty_lower": args.belief_uncertainty_lower,
        "belief_uncertainty_upper": args.belief_uncertainty_upper,
    }
    for key, value in override_fields.items():
        if value is not None:
            payload[key] = value
    return payload


def shrink_expected_win_rate(raw_rate: pd.Series, shrink_factor: float) -> pd.Series:
    raw = pd.to_numeric(raw_rate, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    shrink = float(np.clip(shrink_factor, 0.0, 1.0))
    return (0.5 + shrink * (raw - 0.5)).clip(lower=0.0, upper=1.0)


def apply_live_policy_calibration(selector_df: pd.DataFrame, policy_payload: dict) -> pd.DataFrame:
    if selector_df.empty:
        return selector_df.copy()

    out = selector_df.copy()
    out["raw_expected_win_rate"] = pd.to_numeric(out["expected_win_rate"], errors="coerce").fillna(0.5)
    out["expected_win_rate"] = shrink_expected_win_rate(out["raw_expected_win_rate"], policy_payload["probability_shrink_factor"])
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
    slate_records, slate_skipped = build_records(predictor, market_df, args.season)
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
        history_lookup = build_history_lookup(history_df)
    else:
        history_mode = "heuristic_fallback"
        print(f"Warning: history CSV not found ({history_path}); using heuristic edge calibration.")
    selector_df = build_play_rows(
        slate_df,
        history_lookup,
        belief_uncertainty_lower=float(policy_payload.get("belief_uncertainty_lower", 0.75)),
        belief_uncertainty_upper=float(policy_payload.get("belief_uncertainty_upper", 1.15)),
    )
    selector_df = apply_live_policy_calibration(selector_df, policy_payload)
    selector_df, xgb_ltr_summary = maybe_apply_xgb_ltr_reranker(selector_df, history_df, policy_payload)
    selector_df, accept_reject_summary = maybe_apply_accept_rejector(selector_df, history_df, policy_payload)
    selector_df, robust_reranker_summary = maybe_apply_robust_reranker(selector_df, history_df, policy_payload)
    if selector_df.empty:
        raise RuntimeError("Selector produced no rows from the current slate.")
    args.selector_csv_out.parent.mkdir(parents=True, exist_ok=True)
    selector_df.to_csv(args.selector_csv_out, index=False)

    final_board = compute_final_board(
        selector_df,
        american_odds=policy_payload["american_odds"],
        min_ev=policy_payload["min_ev"],
        min_final_confidence=policy_payload["min_final_confidence"],
        min_recommendation=policy_payload["min_recommendation"],
        ranking_mode=policy_payload.get("ranking_mode", "ev_adjusted"),
        max_plays_per_player=policy_payload["max_plays_per_player"],
        max_plays_per_target=policy_payload["max_plays_per_target"],
        max_total_plays=policy_payload["max_total_plays"],
        max_target_plays={"PTS": policy_payload["max_pts_plays"], "TRB": policy_payload["max_trb_plays"], "AST": policy_payload["max_ast_plays"]},
        non_pts_min_gap_percentile=policy_payload["non_pts_min_gap_percentile"],
        edge_adjust_k=policy_payload["edge_adjust_k"],
        min_bet_win_rate=policy_payload.get("min_bet_win_rate", 0.57),
        medium_bet_win_rate=policy_payload.get("medium_bet_win_rate", 0.60),
        full_bet_win_rate=policy_payload.get("full_bet_win_rate", 0.65),
        medium_tier_percentile=policy_payload.get("medium_tier_percentile", 0.80),
        strong_tier_percentile=policy_payload.get("strong_tier_percentile", 0.90),
        elite_tier_percentile=policy_payload.get("elite_tier_percentile", policy_payload.get("elite_pct", 0.95)),
        small_bet_fraction=policy_payload.get("small_bet_fraction", 0.005),
        medium_bet_fraction=policy_payload.get("medium_bet_fraction", 0.010),
        full_bet_fraction=policy_payload.get("full_bet_fraction", 0.015),
        max_bet_fraction=policy_payload.get("max_bet_fraction", 0.02),
        max_total_bet_fraction=policy_payload.get("max_total_bet_fraction", 0.05),
        belief_uncertainty_lower=policy_payload.get("belief_uncertainty_lower", 0.75),
        belief_uncertainty_upper=policy_payload.get("belief_uncertainty_upper", 1.15),
    )
    args.final_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.final_json_out.parent.mkdir(parents=True, exist_ok=True)
    final_board.to_csv(args.final_csv_out, index=False)

    payload = {
        "manifest_path": str(manifest_path),
        "run_id": predictor.metadata.get("run_id") if predictor is not None else None,
        "predictor_error": predictor_error,
        "market_snapshot": str(args.market_wide_path),
        "history_csv": str(history_path),
        "history_mode": history_mode,
        "season": args.season,
        "policy_profile": args.policy_profile,
        "policy": policy_payload,
        "xgb_ltr": xgb_ltr_summary,
        "accept_reject": accept_reject_summary,
        "robust_reranker": robust_reranker_summary,
        "slate_rows": int(len(slate_df)),
        "selector_rows": int(len(selector_df)),
        "final_rows": int(len(final_board)),
        "input_validation": input_validation,
        "top_plays": final_board.head(20).to_dict(orient="records"),
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
    print(f"Slate CSV:     {args.slate_csv_out}")
    print(f"Selector CSV:  {args.selector_csv_out}")
    print(f"Final CSV:     {args.final_csv_out}")
    print(f"Final JSON:    {args.final_json_out}")
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
