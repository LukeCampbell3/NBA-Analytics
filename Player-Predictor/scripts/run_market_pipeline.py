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
from decision_engine.policy_tuning import build_default_shadow_strategies
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
    parser.add_argument("--american-odds", type=int, default=DEFAULT_POLICY.american_odds, help="Assumed American odds for EV.")
    parser.add_argument("--probability-shrink-factor", type=float, default=DEFAULT_POLICY.probability_shrink_factor, help="Shrink expected win rate toward 50%.")
    parser.add_argument("--elite-pct", type=float, default=DEFAULT_POLICY.elite_pct, help="Percentile cutoff for elite priority plays.")
    parser.add_argument("--min-ev", type=float, default=DEFAULT_POLICY.min_ev, help="Minimum EV to keep a play.")
    parser.add_argument("--min-final-confidence", type=float, default=DEFAULT_POLICY.min_final_confidence, help="Minimum final confidence to keep a play.")
    parser.add_argument(
        "--min-recommendation",
        type=str,
        default=DEFAULT_POLICY.min_recommendation,
        choices=["consider", "strong", "elite"],
        help="Lowest selector recommendation allowed into the final board.",
    )
    parser.add_argument("--max-plays-per-player", type=int, default=DEFAULT_POLICY.max_plays_per_player, help="Maximum final plays to keep per player.")
    parser.add_argument("--max-plays-per-target", type=int, default=DEFAULT_POLICY.max_plays_per_target, help="Maximum final plays to keep per target when target-specific caps are not supplied.")
    parser.add_argument("--max-pts-plays", type=int, default=DEFAULT_POLICY.max_pts_plays, help="Maximum final PTS plays to keep.")
    parser.add_argument("--max-trb-plays", type=int, default=DEFAULT_POLICY.max_trb_plays, help="Maximum final TRB plays to keep.")
    parser.add_argument("--max-ast-plays", type=int, default=DEFAULT_POLICY.max_ast_plays, help="Maximum final AST plays to keep.")
    parser.add_argument("--max-total-plays", type=int, default=DEFAULT_POLICY.max_total_plays, help="Maximum total plays to keep.")
    parser.add_argument("--non-pts-min-gap-percentile", type=float, default=DEFAULT_POLICY.non_pts_min_gap_percentile, help="Minimum disagreement percentile for TRB/AST plays.")
    parser.add_argument("--edge-adjust-k", type=float, default=DEFAULT_POLICY.edge_adjust_k, help="Weight for edge-adjusted EV ranking.")
    return parser.parse_args()


def resolve_policy(args: argparse.Namespace):
    base = POLICY_PROFILES[args.policy_profile]
    payload = base.to_dict()
    payload.update(
        {
            "american_odds": int(args.american_odds),
            "probability_shrink_factor": float(args.probability_shrink_factor),
            "elite_pct": float(args.elite_pct),
            "min_ev": float(args.min_ev),
            "min_final_confidence": float(args.min_final_confidence),
            "min_recommendation": str(args.min_recommendation),
            "max_plays_per_player": int(args.max_plays_per_player),
            "max_plays_per_target": int(args.max_plays_per_target),
            "max_pts_plays": int(args.max_pts_plays),
            "max_trb_plays": int(args.max_trb_plays),
            "max_ast_plays": int(args.max_ast_plays),
            "max_total_plays": int(args.max_total_plays),
            "non_pts_min_gap_percentile": float(args.non_pts_min_gap_percentile),
            "edge_adjust_k": float(args.edge_adjust_k),
        }
    )
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


def summarize_skip_reasons(skipped_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped_rows:
        reason = str(item.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


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
    selector_df = build_play_rows(slate_df, history_lookup)
    selector_df = apply_live_policy_calibration(selector_df, policy_payload)
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
        max_plays_per_player=policy_payload["max_plays_per_player"],
        max_plays_per_target=policy_payload["max_plays_per_target"],
        max_total_plays=policy_payload["max_total_plays"],
        max_target_plays={"PTS": policy_payload["max_pts_plays"], "TRB": policy_payload["max_trb_plays"], "AST": policy_payload["max_ast_plays"]},
        non_pts_min_gap_percentile=policy_payload["non_pts_min_gap_percentile"],
        edge_adjust_k=policy_payload["edge_adjust_k"],
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
        "slate_rows": int(len(slate_df)),
        "selector_rows": int(len(selector_df)),
        "final_rows": int(len(final_board)),
        "input_validation": input_validation,
        "top_plays": final_board.head(20).to_dict(orient="records"),
    }
    args.final_json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
            "recommendation",
        ]
        present_cols = [column for column in show_cols if column in final_board.columns]
        print("\nFinal plays:")
        print(final_board[present_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
