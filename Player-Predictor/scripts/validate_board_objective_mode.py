#!/usr/bin/env python3
"""
Replay daily selector artifacts over a date window and compare board modes.

This script is intentionally lightweight:
- reads saved `upcoming_market_play_selector_<date>.csv` files
- runs `compute_final_board` for each requested mode
- resolves outcomes from `latest_market_comparison_strict_rows.csv`
- writes mode-level and row-level validation artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_engine.policy_tuning import build_default_shadow_strategies
from post_process_market_plays import compute_final_board


POLICY_PROFILES = {config.name: config for config in build_default_shadow_strategies()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate board objective mode against historical resolved outcomes.")
    parser.add_argument("--start-run-date", type=str, default="20260101", help="Inclusive run-date start (YYYYMMDD).")
    parser.add_argument("--end-run-date", type=str, default="20260331", help="Inclusive run-date end (YYYYMMDD).")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["abs_edge", "board_objective"],
        choices=["edge", "abs_edge", "ev_adjusted", "thompson_ev", "board_objective"],
        help="Selection modes to compare.",
    )
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="production_abs_edge_b12",
        choices=sorted(POLICY_PROFILES.keys()),
        help="Base policy profile used for all mode replays.",
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
        help=(
            "Optional long-format resolved rows fallback (for example prior mode-compare rows). "
            "Used when history-csv does not cover the full replay window."
        ),
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
        help="Player processed data root used to derive actual PTS/TRB/AST outcomes.",
    )
    parser.add_argument("--max-days", type=int, default=0, help="Optional cap on replayed days (0 disables).")
    parser.add_argument("--rows-csv-out", type=Path, default=None, help="Row-level output CSV.")
    parser.add_argument("--summary-csv-out", type=Path, default=None, help="Mode summary output CSV.")
    parser.add_argument("--summary-json-out", type=Path, default=None, help="Mode summary output JSON.")
    parser.add_argument(
        "--selected-board-calibrator-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator.json",
        help="Optional selected-board calibrator payload JSON.",
    )
    parser.add_argument(
        "--disable-selected-board-calibration",
        action="store_true",
        help="Disable selected-board calibration during replay validation.",
    )
    parser.add_argument(
        "--learned-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "learned_pool_gate.json",
        help="Optional learned pool-gate payload JSON.",
    )
    parser.add_argument(
        "--enable-learned-gate",
        action="store_true",
        help="Enable learned pool-gate filtering during replay validation.",
    )
    parser.add_argument(
        "--learned-gate-min-rows",
        type=int,
        default=0,
        help="Minimum rows that must pass learned gate before enforcement (0 uses payload/default policy).",
    )
    return parser.parse_args()


def _resolve_result(direction: str, line: float, actual: float, tol: float = 1e-9) -> str:
    if pd.isna(actual) or pd.isna(line):
        return "missing"
    if abs(float(actual) - float(line)) <= tol:
        return "push"
    d = str(direction).upper()
    if d == "OVER":
        return "win" if float(actual) > float(line) else "loss"
    if d == "UNDER":
        return "win" if float(actual) < float(line) else "loss"
    return "missing"


def _normalize_player_key(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace(" ", "_")
    for old, new in [(".", ""), ("'", ""), ("`", ""), ("’", ""), (",", ""), ("/", "-"), ("\\", "-"), (":", "")]:
        text = text.replace(old, new)
    text = "_".join(part for part in text.split("_") if part)
    folded = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return ascii_text


def _build_data_proc_actual_lookup(data_proc_root: Path, start_token: str, end_token: str) -> dict[tuple[str, str, str], float]:
    """
    Build (market_date, player_key, target) -> actual from per-player processed logs.
    """
    root = data_proc_root.resolve()
    if not root.exists():
        return {}

    start_date = pd.to_datetime(start_token, format="%Y%m%d", errors="coerce")
    end_date = pd.to_datetime(end_token, format="%Y%m%d", errors="coerce")
    if pd.isna(start_date) or pd.isna(end_date):
        return {}
    # Include near-forward market dates from end-of-window run folders.
    end_date = end_date + pd.Timedelta(days=3)

    lookup: dict[tuple[str, str, str], float] = {}
    use_cols = {"Date", "Player", "PTS", "TRB", "AST"}
    for csv_path in root.glob("*/*_processed_processed.csv"):
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in use_cols)
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue
        dates = pd.to_datetime(df["Date"], errors="coerce")
        mask = (dates >= start_date) & (dates <= end_date)
        if not mask.any():
            continue
        subset = df.loc[mask].copy()
        if subset.empty:
            continue
        subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        subset = subset.loc[subset["Date"].notna()].copy()
        if subset.empty:
            continue

        folder_player = str(csv_path.parent.name).strip()
        folder_norm = _normalize_player_key(folder_player)

        for _, row in subset.iterrows():
            market_date = str(row.get("Date", ""))
            player_raw = str(row.get("Player", folder_player)).strip() if pd.notna(row.get("Player")) else folder_player
            player_raw_norm = _normalize_player_key(player_raw)
            player_keys = {folder_player, folder_norm, player_raw, player_raw_norm}
            for target in ("PTS", "TRB", "AST"):
                value = pd.to_numeric(pd.Series([row.get(target)]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                for player_key in player_keys:
                    if not player_key:
                        continue
                    lookup[(market_date, str(player_key), target)] = float(value)
    return lookup


def _build_actual_lookup(history_csv: Path) -> dict[tuple[str, str, str], float]:
    if not history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {history_csv}")
    df = pd.read_csv(history_csv)
    required = ["player", "market_date", "market_PTS", "actual_PTS", "market_TRB", "actual_TRB", "market_AST", "actual_AST"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"History CSV missing columns: {missing}")

    lookup: dict[tuple[str, str, str], float] = {}
    working = df.copy()
    working["player"] = working["player"].astype(str).str.strip()
    working["market_date"] = pd.to_datetime(working["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for target in ("PTS", "TRB", "AST"):
        actual_col = f"actual_{target}"
        market_col = f"market_{target}"
        part = working[["player", "market_date", market_col, actual_col]].copy()
        part = part.loc[part["market_date"].notna() & part[market_col].notna()].copy()
        if part.empty:
            continue
        for _, row in part.iterrows():
            key = (str(row["market_date"]), str(row["player"]), target)
            lookup[key] = float(row[actual_col]) if pd.notna(row[actual_col]) else np.nan
    return lookup


def _iter_run_dates(daily_runs_dir: Path, start_token: str, end_token: str, max_days: int) -> list[str]:
    folders = []
    for child in sorted(daily_runs_dir.iterdir()):
        if not child.is_dir():
            continue
        token = child.name
        if len(token) != 8 or not token.isdigit():
            continue
        if token < start_token or token > end_token:
            continue
        selector_csv = child / f"upcoming_market_play_selector_{token}.csv"
        if selector_csv.exists():
            folders.append(token)
    if max_days > 0:
        folders = folders[: int(max_days)]
    return folders


def _build_rows_actual_lookup(rows_csv: Path | None) -> dict[tuple[str, str, str, str, float], float]:
    if rows_csv is None:
        return {}
    rows_csv = rows_csv.resolve()
    if not rows_csv.exists():
        return {}
    df = pd.read_csv(rows_csv)
    required = {"run_date", "player", "target", "direction", "market_line", "actual"}
    if not required.issubset(df.columns):
        return {}

    out: dict[tuple[str, str, str, str, float], float] = {}
    working = df.copy()
    working["run_date"] = pd.to_datetime(working["run_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["player"] = working["player"].astype(str).str.strip()
    working["target"] = working["target"].astype(str).str.upper().str.strip()
    working["direction"] = working["direction"].astype(str).str.upper().str.strip()
    working["market_line"] = pd.to_numeric(working["market_line"], errors="coerce").round(6)
    working["actual"] = pd.to_numeric(working["actual"], errors="coerce")
    working = working.loc[working["run_date"].notna() & working["player"].ne("") & working["target"].isin(["PTS", "TRB", "AST"])].copy()
    for _, row in working.iterrows():
        key = (str(row["run_date"]), str(row["player"]), str(row["target"]), str(row["direction"]), float(row["market_line"]))
        out[key] = float(row["actual"]) if pd.notna(row["actual"]) else np.nan
    return out


def _policy_kwargs(payload: dict, mode: str) -> dict:
    return {
        "american_odds": payload["american_odds"],
        "min_ev": payload["min_ev"],
        "min_final_confidence": payload["min_final_confidence"],
        "min_recommendation": payload["min_recommendation"],
        "selection_mode": mode,
        "ranking_mode": mode,
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
        "board_objective_lambda_corr": payload.get("board_objective_lambda_corr", 0.12),
        "board_objective_lambda_conc": payload.get("board_objective_lambda_conc", 0.07),
        "board_objective_lambda_unc": payload.get("board_objective_lambda_unc", 0.06),
        "board_objective_corr_same_game": payload.get("board_objective_corr_same_game", 0.65),
        "board_objective_corr_same_player": payload.get("board_objective_corr_same_player", 1.0),
        "board_objective_corr_same_target": payload.get("board_objective_corr_same_target", 0.15),
        "board_objective_corr_same_direction": payload.get("board_objective_corr_same_direction", 0.05),
        "board_objective_corr_same_script_cluster": payload.get("board_objective_corr_same_script_cluster", 0.30),
        "board_objective_swap_candidates": payload.get("board_objective_swap_candidates", 18),
        "board_objective_swap_rounds": payload.get("board_objective_swap_rounds", 2),
        "max_history_staleness_days": payload.get("max_history_staleness_days", 0),
        "min_recency_factor": payload.get("min_recency_factor", 0.0),
        "learned_gate_min_rows": payload.get("learned_gate_min_rows", 0),
    }


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


def _load_learned_pool_gate(path: Path, enabled: bool) -> dict | None:
    if not enabled:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    base_profile = POLICY_PROFILES[args.policy_profile].to_dict()
    selected_board_calibrator = _load_selected_board_calibrator(
        args.selected_board_calibrator_json,
        disabled=bool(args.disable_selected_board_calibration),
    )
    learned_pool_gate = _load_learned_pool_gate(
        args.learned_gate_json,
        enabled=bool(args.enable_learned_gate),
    )

    mode_token = "_".join(args.modes)
    token = f"{args.start_run_date}_{args.end_run_date}"
    rows_csv = args.rows_csv_out or (args.daily_runs_dir / f"mode_compare_{mode_token}_{token}_rows.csv")
    summary_csv = args.summary_csv_out or (args.daily_runs_dir / f"mode_compare_{mode_token}_{token}.csv")
    summary_json = args.summary_json_out or (args.daily_runs_dir / f"mode_compare_{mode_token}_{token}.json")

    data_proc_actual_lookup = _build_data_proc_actual_lookup(args.data_proc_root, args.start_run_date, args.end_run_date)
    actual_lookup = _build_actual_lookup(args.history_csv.resolve())
    fallback_rows_csv = args.actual_rows_csv
    if fallback_rows_csv is None:
        candidate = args.daily_runs_dir / f"mode_compare_edge_absedge_evadj_daily_rows_{token}.csv"
        fallback_rows_csv = candidate if candidate.exists() else None
    rows_actual_lookup = _build_rows_actual_lookup(fallback_rows_csv)
    run_dates = _iter_run_dates(args.daily_runs_dir.resolve(), args.start_run_date, args.end_run_date, args.max_days)
    if not run_dates:
        raise RuntimeError("No run-date folders with selector CSVs found in the requested window.")

    rows: list[dict] = []
    for run_date in run_dates:
        selector_csv = args.daily_runs_dir / run_date / f"upcoming_market_play_selector_{run_date}.csv"
        selector_df = pd.read_csv(selector_csv)
        if selector_df.empty:
            continue
        for mode in args.modes:
            kwargs = _policy_kwargs(base_profile, mode=mode)
            kwargs["selected_board_calibrator"] = selected_board_calibrator
            run_month = pd.to_datetime(run_date, format="%Y%m%d", errors="coerce")
            kwargs["selected_board_calibration_month"] = run_month.strftime("%Y-%m") if pd.notna(run_month) else None
            kwargs["learned_gate_payload"] = learned_pool_gate
            kwargs["learned_gate_month"] = run_month.strftime("%Y-%m") if pd.notna(run_month) else None
            kwargs["learned_gate_min_rows"] = int(args.learned_gate_min_rows)
            board = compute_final_board(selector_df.copy(), **kwargs)
            if board.empty:
                continue
            for _, row in board.iterrows():
                market_date = pd.to_datetime(row.get("market_date"), errors="coerce")
                market_date_key = market_date.strftime("%Y-%m-%d") if pd.notna(market_date) else ""
                run_date_key = pd.to_datetime(run_date, format="%Y%m%d", errors="coerce")
                run_date_key = run_date_key.strftime("%Y-%m-%d") if pd.notna(run_date_key) else run_date
                player = str(row.get("player", "")).strip()
                player_norm = _normalize_player_key(player)
                target = str(row.get("target", "")).strip().upper()
                line = pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").iloc[0]
                rounded_line = float(np.round(line, 6)) if pd.notna(line) else np.nan
                fallback_key = (run_date_key, player, target, str(row.get("direction", "")).upper(), rounded_line)
                actual = np.nan
                for player_key in (player, player_norm):
                    if not player_key:
                        continue
                    lookup_key = (market_date_key, str(player_key), target)
                    if lookup_key in data_proc_actual_lookup:
                        actual = data_proc_actual_lookup[lookup_key]
                        break
                    if lookup_key in actual_lookup:
                        actual = actual_lookup[lookup_key]
                        break
                if pd.isna(actual) and rounded_line == rounded_line:
                    actual = rows_actual_lookup.get(fallback_key, np.nan)
                result = _resolve_result(str(row.get("direction", "")), line=float(line) if pd.notna(line) else np.nan, actual=actual)

                rows.append(
                    {
                        "run_date": run_date,
                        "mode": mode,
                        "player": player,
                        "target": target,
                        "direction": str(row.get("direction", "")),
                        "market_date": market_date_key,
                        "market_line": float(line) if pd.notna(line) else np.nan,
                        "prediction": float(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]),
                        "expected_win_rate": float(pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").fillna(np.nan).iloc[0]),
                        "board_play_win_prob": float(pd.to_numeric(pd.Series([row.get("board_play_win_prob")]), errors="coerce").fillna(np.nan).iloc[0]),
                        "p_calibrated": float(pd.to_numeric(pd.Series([row.get("p_calibrated")]), errors="coerce").fillna(np.nan).iloc[0]),
                        "ev": float(pd.to_numeric(pd.Series([row.get("ev")]), errors="coerce").fillna(np.nan).iloc[0]),
                        "learned_gate_enabled": bool(pd.to_numeric(pd.Series([row.get("learned_gate_enabled")]), errors="coerce").fillna(0).iloc[0]),
                        "learned_gate_enforced": bool(pd.to_numeric(pd.Series([row.get("learned_gate_enforced")]), errors="coerce").fillna(0).iloc[0]),
                        "learned_gate_pass": bool(pd.to_numeric(pd.Series([row.get("learned_gate_pass")]), errors="coerce").fillna(1).iloc[0]),
                        "learned_gate_threshold": float(pd.to_numeric(pd.Series([row.get("learned_gate_threshold")]), errors="coerce").fillna(np.nan).iloc[0]),
                        "actual": float(actual) if pd.notna(actual) else np.nan,
                        "result": result,
                    }
                )

    rows_df = pd.DataFrame.from_records(rows)
    if rows_df.empty:
        raise RuntimeError("No replay rows were generated; check selector artifacts and date window.")

    summary_rows: list[dict] = []
    for mode in args.modes:
        part = rows_df.loc[rows_df["mode"] == mode].copy()
        if part.empty:
            continue
        resolved = part.loc[part["result"].isin(["win", "loss"])].copy()
        wins = int((resolved["result"] == "win").sum())
        losses = int((resolved["result"] == "loss").sum())
        pushes = int((part["result"] == "push").sum())
        missing = int((part["result"] == "missing").sum())
        resolved_count = int(len(resolved))
        resolved_probs = pd.to_numeric(
            resolved["p_calibrated"] if "p_calibrated" in resolved.columns else resolved["expected_win_rate"],
            errors="coerce",
        ).dropna()
        avg_expected_resolved = float(resolved_probs.mean()) if len(resolved_probs) else np.nan
        hit_rate = float(wins / resolved_count) if resolved_count > 0 else np.nan
        summary_rows.append(
            {
                "mode": mode,
                "days": int(part["run_date"].nunique()),
                "rows": int(len(part)),
                "under": int((part["direction"].astype(str).str.upper() == "UNDER").sum()),
                "over": int((part["direction"].astype(str).str.upper() == "OVER").sum()),
                "under_share": float((part["direction"].astype(str).str.upper() == "UNDER").mean()) if len(part) else np.nan,
                "resolved": resolved_count,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "missing": missing,
                "hit_rate": hit_rate,
                "avg_expected_win_rate": float(pd.to_numeric(part["expected_win_rate"], errors="coerce").mean()),
                "avg_board_play_win_prob": float(pd.to_numeric(part["board_play_win_prob"], errors="coerce").mean()),
                "avg_p_calibrated": float(pd.to_numeric(part["p_calibrated"], errors="coerce").mean()) if "p_calibrated" in part.columns else np.nan,
                "avg_expected_resolved": avg_expected_resolved,
                "calibration_gap_pp": float((hit_rate - avg_expected_resolved) * 100.0) if resolved_count > 0 and avg_expected_resolved == avg_expected_resolved else np.nan,
                "avg_ev": float(pd.to_numeric(part["ev"], errors="coerce").mean()),
                "learned_gate_enforced_rate": float(pd.to_numeric(part.get("learned_gate_enforced"), errors="coerce").fillna(0).mean()) if "learned_gate_enforced" in part.columns else np.nan,
                "learned_gate_pass_rate": float(pd.to_numeric(part.get("learned_gate_pass"), errors="coerce").fillna(1).mean()) if "learned_gate_pass" in part.columns else np.nan,
            }
        )

    summary_df = pd.DataFrame.from_records(summary_rows).sort_values("mode").reset_index(drop=True)
    rows_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(rows_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    payload = {
        "window": {
            "start_run_date": args.start_run_date,
            "end_run_date": args.end_run_date,
        },
        "policy_profile": args.policy_profile,
        "modes": args.modes,
        "days_replayed": int(len(run_dates)),
        "summary": summary_rows,
        "rows_csv": str(rows_csv),
        "summary_csv": str(summary_csv),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Rows CSV:    {rows_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON:{summary_json}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
