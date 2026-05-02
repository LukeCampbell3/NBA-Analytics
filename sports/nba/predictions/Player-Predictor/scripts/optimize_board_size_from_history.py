#!/usr/bin/env python3
"""
Replay historical selector slates across candidate board sizes and recommend a
production board size from resolved results.

This creates a longer board-history artifact by expanding each replay day into
multiple candidate board sizes, which is useful even when the calendar window
itself is still short.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_engine.board_size_optimization import recommend_board_size, summarize_board_size_history
from decision_engine.policy_tuning import build_default_shadow_strategies
from post_process_market_plays import compute_final_board
from validate_board_objective_mode import (
    _build_actual_lookup,
    _build_data_proc_actual_lookup,
    _build_rows_actual_lookup,
    _iter_run_dates,
    _load_accepted_pick_gate,
    _load_learned_pool_gate,
    _load_selected_board_calibrator,
    _load_staking_bucket_model,
    _lookup_actual_with_date_fallback,
    _normalize_player_key,
    _player_key_variants,
    _policy_kwargs,
    _resolve_result,
)


POLICY_PROFILES = {config.name: config for config in build_default_shadow_strategies()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize board size from historical selector replays.")
    parser.add_argument("--start-run-date", type=str, default="20260424", help="Inclusive run-date start (YYYYMMDD).")
    parser.add_argument("--end-run-date", type=str, default="20260430", help="Inclusive run-date end (YYYYMMDD).")
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="production_board_objective_b12",
        choices=sorted(POLICY_PROFILES.keys()),
        help="Base policy profile to replay.",
    )
    parser.add_argument("--min-board-size", type=int, default=3, help="Minimum requested board size to evaluate.")
    parser.add_argument("--max-board-size", type=int, default=12, help="Maximum requested board size to evaluate.")
    parser.add_argument(
        "--board-sizes",
        nargs="*",
        type=int,
        default=None,
        help="Explicit board sizes to evaluate. Overrides min/max when provided.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv",
        help="Wide historical outcomes table.",
    )
    parser.add_argument(
        "--actual-rows-csv",
        type=Path,
        default=None,
        help="Optional long-format resolved rows fallback.",
    )
    parser.add_argument(
        "--daily-runs-dir",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs",
        help="Directory containing dated daily run folders.",
    )
    parser.add_argument(
        "--data-proc-root",
        type=Path,
        default=REPO_ROOT / "Data-Proc",
        help="Player processed data root used to derive actual outcomes.",
    )
    parser.add_argument("--max-days", type=int, default=0, help="Optional cap on replayed days (0 disables).")
    parser.add_argument(
        "--rows-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "board_size_history_rows.csv",
        help="Row-level board-history output CSV.",
    )
    parser.add_argument(
        "--daily-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "board_size_history_daily.csv",
        help="Daily board-size summary output CSV.",
    )
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "board_size_history_summary.csv",
        help="Board-size summary output CSV.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "board_size_history_summary.json",
        help="Board-size summary output JSON.",
    )
    parser.add_argument(
        "--selected-board-calibrator-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator.json",
        help="Optional selected-board calibrator payload JSON.",
    )
    parser.add_argument("--disable-selected-board-calibration", action="store_true", help="Disable selected-board calibration.")
    parser.add_argument(
        "--learned-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "learned_pool_gate.json",
        help="Optional learned pool-gate payload JSON.",
    )
    parser.add_argument("--enable-learned-gate", action="store_true", help="Enable learned pool-gate during replay.")
    parser.add_argument("--learned-gate-min-rows", type=int, default=0, help="Minimum rows required before learned gate enforcement.")
    parser.add_argument(
        "--accepted-pick-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "accepted_pick_gate" / "candidates" / "accepted_pick_gate_candidate.json",
        help="Optional accepted-pick gate payload JSON.",
    )
    parser.add_argument("--enable-accepted-pick-gate", action="store_true", help="Enable accepted-pick gate during replay.")
    parser.add_argument("--accepted-pick-gate-live", action="store_true", help="Apply accepted-pick gate live during replay.")
    parser.add_argument("--accepted-pick-gate-min-rows", type=int, default=None, help="Minimum rows required before accepted-pick gate enforcement.")
    parser.add_argument(
        "--staking-bucket-model-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "staking_bucket_model_v2.json",
        help="Optional walk-forward staking bucket model payload JSON.",
    )
    parser.add_argument("--enable-staking-bucket-model", action="store_true", help="Enable walk-forward staking bucket model.")
    parser.add_argument("--staking-bucket-model-min-rows", type=int, default=None, help="Override minimum monthly rows required by staking bucket model.")
    return parser.parse_args()


def _resolve_board_sizes(args: argparse.Namespace) -> list[int]:
    if args.board_sizes:
        sizes = sorted({int(size) for size in args.board_sizes if int(size) > 0})
    else:
        lower = max(1, int(args.min_board_size))
        upper = max(lower, int(args.max_board_size))
        sizes = list(range(lower, upper + 1))
    if not sizes:
        raise ValueError("At least one positive board size is required.")
    return sizes


def _mean_or_nan(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())
    return float("nan")


def main() -> None:
    args = parse_args()
    board_sizes = _resolve_board_sizes(args)

    base_profile = POLICY_PROFILES[args.policy_profile].to_dict()
    if args.enable_accepted_pick_gate:
        base_profile["accepted_pick_gate_enabled"] = True
    if args.accepted_pick_gate_live:
        base_profile["accepted_pick_gate_live"] = True
    if args.accepted_pick_gate_min_rows is not None:
        base_profile["accepted_pick_gate_min_rows"] = int(args.accepted_pick_gate_min_rows)

    selected_board_calibrator = _load_selected_board_calibrator(
        args.selected_board_calibrator_json,
        disabled=bool(args.disable_selected_board_calibration),
    )
    learned_pool_gate = _load_learned_pool_gate(
        args.learned_gate_json,
        enabled=bool(args.enable_learned_gate),
    )
    accepted_pick_gate = _load_accepted_pick_gate(
        args.accepted_pick_gate_json,
        enabled=bool(base_profile.get("accepted_pick_gate_enabled", False)),
    )
    model_enabled_default = bool(base_profile.get("staking_bucket_model_enabled", False)) or bool(args.enable_staking_bucket_model)
    staking_bucket_model = _load_staking_bucket_model(
        args.staking_bucket_model_json,
        enabled=model_enabled_default,
    )

    data_proc_actual_lookup = _build_data_proc_actual_lookup(args.data_proc_root, args.start_run_date, args.end_run_date)
    history_actual_lookup = _build_actual_lookup(args.history_csv.resolve())
    rows_actual_lookup = _build_rows_actual_lookup(args.actual_rows_csv)
    run_dates = _iter_run_dates(args.daily_runs_dir.resolve(), args.start_run_date, args.end_run_date, args.max_days)
    if not run_dates:
        raise RuntimeError("No run-date folders with selector CSVs found in the requested window.")

    row_records: list[dict] = []
    daily_records: list[dict] = []

    for run_date in run_dates:
        selector_csv = args.daily_runs_dir / run_date / f"upcoming_market_play_selector_{run_date}.csv"
        selector_df = pd.read_csv(selector_csv)
        if selector_df.empty:
            continue

        run_month = pd.to_datetime(run_date, format="%Y%m%d", errors="coerce")
        run_date_key = run_month.strftime("%Y-%m-%d") if pd.notna(run_month) else run_date

        for board_size in board_sizes:
            local_profile = dict(base_profile)
            local_profile["max_total_plays"] = int(board_size)
            local_profile["min_board_plays"] = int(board_size)

            kwargs = _policy_kwargs(local_profile, mode="board_objective")
            kwargs["selected_board_calibrator"] = selected_board_calibrator
            kwargs["selected_board_calibration_month"] = run_month.strftime("%Y-%m") if pd.notna(run_month) else None
            kwargs["learned_gate_payload"] = learned_pool_gate
            kwargs["learned_gate_month"] = run_month.strftime("%Y-%m") if pd.notna(run_month) else None
            kwargs["learned_gate_min_rows"] = int(args.learned_gate_min_rows)
            kwargs["accepted_pick_gate_payload"] = accepted_pick_gate
            kwargs["accepted_pick_gate_month"] = run_month.strftime("%Y-%m") if pd.notna(run_month) else None
            kwargs["accepted_pick_gate_enabled"] = bool(local_profile.get("accepted_pick_gate_enabled", False))
            kwargs["accepted_pick_gate_live"] = bool(local_profile.get("accepted_pick_gate_live", False))
            kwargs["accepted_pick_gate_min_rows"] = int(local_profile.get("accepted_pick_gate_min_rows", 0))
            kwargs["staking_bucket_model_payload"] = staking_bucket_model if model_enabled_default else None
            kwargs["staking_bucket_model_month"] = run_month.strftime("%Y-%m") if pd.notna(run_month) else None
            if args.staking_bucket_model_min_rows is not None:
                kwargs["staking_bucket_model_min_rows"] = int(args.staking_bucket_model_min_rows)

            board = compute_final_board(selector_df.copy(), **kwargs)
            board_realized = int(len(board))
            wins = 0
            losses = 0
            pushes = 0
            missing = 0

            if not board.empty:
                for _, row in board.iterrows():
                    market_date = pd.to_datetime(row.get("market_date"), errors="coerce")
                    market_date_key = market_date.strftime("%Y-%m-%d") if pd.notna(market_date) else ""
                    player = str(row.get("player", "")).strip()
                    player_norm = _normalize_player_key(player)
                    target = str(row.get("target", "")).strip().upper()
                    direction = str(row.get("direction", "")).strip().upper()
                    line = pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").iloc[0]
                    rounded_line = float(np.round(line, 6)) if pd.notna(line) else np.nan
                    player_keys = _player_key_variants(player, player_norm)

                    actual, actual_source, actual_match_date = _lookup_actual_with_date_fallback(
                        market_date_key=market_date_key,
                        player_keys=player_keys,
                        target=target,
                        data_proc_actual_lookup=data_proc_actual_lookup,
                        history_actual_lookup=history_actual_lookup,
                        near_date_days=1,
                    )
                    fallback_key = (run_date_key, player, target, direction, rounded_line)
                    if pd.isna(actual) and rounded_line == rounded_line:
                        actual = rows_actual_lookup.get(fallback_key, np.nan)
                        if pd.notna(actual):
                            actual_source = "rows_fallback"
                            actual_match_date = run_date_key

                    result = _resolve_result(direction=direction, line=float(line) if pd.notna(line) else np.nan, actual=actual)
                    if result == "win":
                        wins += 1
                    elif result == "loss":
                        losses += 1
                    elif result == "push":
                        pushes += 1
                    else:
                        missing += 1

                    row_records.append(
                        {
                            "run_date": run_date,
                            "run_date_iso": run_date_key,
                            "board_size_requested": int(board_size),
                            "board_size_realized": int(board_realized),
                            "board_size_shortfall": int(max(0, int(board_size) - int(board_realized))),
                            "player": player,
                            "target": target,
                            "direction": direction,
                            "market_date": market_date_key,
                            "market_line": float(line) if pd.notna(line) else np.nan,
                            "prediction": float(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]),
                            "expected_win_rate": float(pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").fillna(np.nan).iloc[0]),
                            "board_play_win_prob": float(pd.to_numeric(pd.Series([row.get("board_play_win_prob")]), errors="coerce").fillna(np.nan).iloc[0]),
                            "p_calibrated": float(pd.to_numeric(pd.Series([row.get("p_calibrated")]), errors="coerce").fillna(np.nan).iloc[0]),
                            "ev": float(pd.to_numeric(pd.Series([row.get("ev")]), errors="coerce").fillna(np.nan).iloc[0]),
                            "final_confidence": float(pd.to_numeric(pd.Series([row.get("final_confidence")]), errors="coerce").fillna(np.nan).iloc[0]),
                            "recommendation": str(row.get("recommendation", "")),
                            "board_objective_dynamic_target_size": float(pd.to_numeric(pd.Series([row.get("board_objective_dynamic_target_size")]), errors="coerce").fillna(np.nan).iloc[0]),
                            "board_target_cap_effective": float(pd.to_numeric(pd.Series([row.get("board_target_cap_effective")]), errors="coerce").fillna(np.nan).iloc[0]),
                            "actual": float(actual) if pd.notna(actual) else np.nan,
                            "actual_source": str(actual_source),
                            "actual_match_date": str(actual_match_date or ""),
                            "result": result,
                        }
                    )

            resolved = int(wins + losses)
            units = float(wins * (100.0 / 110.0) - losses)
            daily_records.append(
                {
                    "run_date": run_date,
                    "run_date_iso": run_date_key,
                    "board_size_requested": int(board_size),
                    "board_size_realized": int(board_realized),
                    "board_size_shortfall": int(max(0, int(board_size) - int(board_realized))),
                    "resolved": resolved,
                    "wins": int(wins),
                    "losses": int(losses),
                    "pushes": int(pushes),
                    "missing": int(missing),
                    "units": units,
                    "hit_rate_day": float(wins / resolved) if resolved > 0 else np.nan,
                    "expected_win_rate_mean": _mean_or_nan(board.get("expected_win_rate", pd.Series(dtype="float64"))) if not board.empty else np.nan,
                    "board_play_win_prob_mean": _mean_or_nan(board.get("board_play_win_prob", pd.Series(dtype="float64"))) if not board.empty else np.nan,
                    "final_confidence_mean": _mean_or_nan(board.get("final_confidence", pd.Series(dtype="float64"))) if not board.empty else np.nan,
                }
            )

    rows_df = pd.DataFrame(row_records)
    daily_df = pd.DataFrame(daily_records)
    summary_df = summarize_board_size_history(daily_df)
    recommendation = recommend_board_size(summary_df)

    payload = {
        "window": {
            "start_run_date": str(args.start_run_date),
            "end_run_date": str(args.end_run_date),
            "days_replayed": int(len(sorted(set(daily_df["run_date"].astype(str).tolist())))) if not daily_df.empty else 0,
            "board_sizes": [int(size) for size in board_sizes],
        },
        "policy_profile": str(args.policy_profile),
        "recommendation": recommendation,
        "summary": summary_df.to_dict(orient="records"),
    }

    for path in [args.rows_csv_out, args.daily_csv_out, args.summary_csv_out, args.summary_json_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(args.rows_csv_out, index=False)
    daily_df.to_csv(args.daily_csv_out, index=False)
    summary_df.to_csv(args.summary_csv_out, index=False)
    args.summary_json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Rows CSV:    {args.rows_csv_out.resolve()}")
    print(f"Daily CSV:   {args.daily_csv_out.resolve()}")
    print(f"Summary CSV: {args.summary_csv_out.resolve()}")
    print(f"Summary JSON:{args.summary_json_out.resolve()}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print("Recommendation:")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
