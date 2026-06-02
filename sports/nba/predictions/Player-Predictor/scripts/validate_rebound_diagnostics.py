#!/usr/bin/env python3
"""
Replay saved daily market slates across rebound-diagnostic variants and compare
board outcomes, calibration, and TRB-specific loss-removal behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from backtest_line_decision_sidecar import _compute_board, _load_daily_policy, _prepare_selector
from decision_engine.line_decision import LineDecisionConfig, build_line_decision_lookup
from post_process_market_plays import american_profit_per_unit
from run_market_pipeline import (
    load_accepted_pick_gate,
    load_learned_pool_gate,
    load_selected_board_calibrator,
    load_staking_bucket_model,
)
from select_market_plays import DEFAULT_REBOUND_DIAGNOSTICS_CONFIG, build_history_lookup
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
SEGMENTS = [
    "TRB_OVER_UPPER_BAND",
    "TRB_OVER_LOW_LINE_ROLE_VOLATILE",
    "TRB_OVER_SUPPLY_DEPENDENT",
    "TRB_OVER_SHARE_COMPETITION",
    "TRB_OVER_STABLE",
    "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY",
]
VARIANT_ORDER = [
    "baseline_no_rebound_diagnostics",
    "upper_band_only",
    "full_rebound_diagnostics",
    "full_rebound_diagnostics_plus_opposite_under",
]
BASELINE_VARIANT = "baseline_no_rebound_diagnostics"
FULL_DIAGNOSTICS_VARIANT = "full_rebound_diagnostics"
PROMOTION_TARGET_VARIANT = "full_rebound_diagnostics_plus_opposite_under"
NO_OP_WINDOW = "NO_OP_NARROWNESS_WINDOW"
ACTIVE_WINDOW = "ACTIVE_REBOUND_RISK_WINDOW"
MIXED_WINDOW = "MIXED_WINDOW"
NO_OP_DAY = "NO_OP_NARROWNESS_DAY"
ACTIVE_DAY = "ACTIVE_REBOUND_RISK_DAY"
DEFAULT_COVERAGE_THRESHOLD = 0.95
DEFAULT_MIN_RESOLVED_PICKS = 8
DEFAULT_NO_OP_BOARD_CHANGE_TOLERANCE = 0
DEFAULT_WIN_PRESERVATION_FLOOR = 0.67
DEFAULT_RESULT_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay rebound-diagnostic variants across saved daily run slates.")
    parser.add_argument("--start-run-date", type=str, default=DEFAULT_START, help="Inclusive start run date (YYYYMMDD).")
    parser.add_argument("--end-run-date", type=str, default=DEFAULT_END, help="Inclusive end run date (YYYYMMDD).")
    parser.add_argument(
        "--daily-runs-dir",
        type=Path,
        action="append",
        default=[],
        help="Daily run directory. Pass multiple times to include both heuristic and trained-bundle replays.",
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
    parser.add_argument("--max-days", type=int, default=0, help="Optional cap on replayed days per directory (0 disables).")
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
    parser.add_argument(
        "--rows-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_validation_rows.csv",
    )
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_validation_summary.csv",
    )
    parser.add_argument(
        "--segments-csv-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_validation_segments.csv",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "rebound_diagnostics_validation_summary.json",
    )
    parser.add_argument("--no-trade-threshold", type=float, default=LineDecisionConfig().no_trade_threshold)
    parser.add_argument("--min-trade-prob", type=float, default=LineDecisionConfig().min_trade_prob)
    parser.add_argument("--min-trade-prob-gap", type=float, default=LineDecisionConfig().min_trade_prob_gap)
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument("--min-resolved-picks", type=int, default=DEFAULT_MIN_RESOLVED_PICKS)
    parser.add_argument("--no-op-board-change-tolerance", type=int, default=DEFAULT_NO_OP_BOARD_CHANGE_TOLERANCE)
    parser.add_argument("--win-preservation-floor", type=float, default=DEFAULT_WIN_PRESERVATION_FLOOR)
    return parser.parse_args()


def _merge_nested_dict(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    override_payload = overrides if isinstance(overrides, dict) else {}
    for key, default_value in defaults.items():
        override_value = override_payload.get(key)
        if isinstance(default_value, dict):
            merged[key] = _merge_nested_dict(default_value, override_value if isinstance(override_value, dict) else {})
        else:
            merged[key] = default_value if key not in override_payload else override_value
    for key, value in override_payload.items():
        if key not in merged:
            merged[key] = value
    return merged


def _variant_rebound_config(variant: str) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_REBOUND_DIAGNOSTICS_CONFIG)
    if variant == "baseline_no_rebound_diagnostics":
        config["enabled"] = False
        for key in ("upper_band", "low_line_role_volatility", "rebound_supply", "rebound_share", "opposite_side_discovery"):
            config[key]["enabled"] = False
    elif variant == "upper_band_only":
        config["enabled"] = True
        config["upper_band"]["enabled"] = True
        config["low_line_role_volatility"]["enabled"] = False
        config["rebound_supply"]["enabled"] = False
        config["rebound_share"]["enabled"] = False
        config["opposite_side_discovery"]["enabled"] = False
    elif variant == "full_rebound_diagnostics":
        config["enabled"] = True
        config["opposite_side_discovery"]["enabled"] = False
    elif variant == "full_rebound_diagnostics_plus_opposite_under":
        config["enabled"] = True
        config["opposite_side_discovery"]["enabled"] = True
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return config


def _infer_validation_mode(run_dir: Path) -> str:
    token = run_dir.name
    final_json = run_dir / f"final_market_plays_{token}.json"
    if not final_json.exists():
        return "unknown"
    payload = json.loads(final_json.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", "")).strip()
    return "artifact_free_heuristic" if run_id == "artifact_free_heuristic" else "trained_bundle"


def _variant_policy_payload(base_policy: dict[str, Any], variant: str) -> dict[str, Any]:
    policy = copy.deepcopy(base_policy)
    base_rebound = policy.get("rebound_diagnostics")
    base_config = _merge_nested_dict(DEFAULT_REBOUND_DIAGNOSTICS_CONFIG, base_rebound if isinstance(base_rebound, dict) else {})
    policy["rebound_diagnostics"] = _merge_nested_dict(base_config, _variant_rebound_config(variant))
    return policy


def _profit_per_unit_from_price(odds: float | int | None, fallback_odds: int = -110) -> float:
    price = pd.to_numeric(pd.Series([odds]), errors="coerce").fillna(np.nan).iloc[0]
    if pd.notna(price) and abs(float(price)) >= 1.0:
        if float(price) > 0:
            return float(price) / 100.0
        return float(100.0 / max(abs(float(price)), 1.0))
    return float(american_profit_per_unit(int(fallback_odds)))


def _ece_10(prob: np.ndarray, label: np.ndarray) -> float:
    if prob.size == 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    n = max(prob.size, 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not mask.any():
            continue
        acc = float(label[mask].mean())
        conf = float(prob[mask].mean())
        ece += (float(mask.sum()) / n) * abs(acc - conf)
    return float(ece)


def _pick_key(frame: pd.DataFrame) -> pd.Series:
    line = pd.to_numeric(frame.get("market_line"), errors="coerce").round(4).astype(str)
    return (
        frame.get("run_date", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + frame.get("player", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + frame.get("target", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + frame.get("direction", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + line
    )


def _resolve_board_rows(
    board_df: pd.DataFrame,
    history_actual_lookup: dict[tuple[str, str, str], float],
    data_proc_actual_lookup: dict[tuple[str, str, str], float],
    *,
    variant: str,
    validation_mode: str,
    run_date_token: str,
    fallback_odds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if board_df.empty:
        return rows
    for _, row in board_df.iterrows():
        market_date_key = pd.to_datetime(row.get("market_date"), errors="coerce")
        market_date_key = (
            market_date_key.strftime("%Y-%m-%d")
            if pd.notna(market_date_key)
            else pd.to_datetime(run_date_token, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")
        )
        player_keys = _player_key_variants(str(row.get("player", "")), str(row.get("market_player_raw", "")))
        target = str(row.get("target", "")).upper().strip()
        direction = str(row.get("direction", "")).upper().strip()
        actual, actual_source, matched_date = _lookup_actual_with_date_fallback(
            market_date_key,
            player_keys,
            target,
            data_proc_actual_lookup=data_proc_actual_lookup,
            history_actual_lookup=history_actual_lookup,
            near_date_days=1,
        )
        line = pd.to_numeric(pd.Series([row.get("market_line")]), errors="coerce").iloc[0]
        result = _resolve_result(direction, line=float(line) if pd.notna(line) else np.nan, actual=actual)
        market_price = pd.to_numeric(pd.Series([row.get("market_side_price")]), errors="coerce").fillna(np.nan).iloc[0]
        units = np.nan
        if result == "win":
            units = _profit_per_unit_from_price(market_price, fallback_odds=fallback_odds)
        elif result == "loss":
            units = -1.0
        elif result == "push":
            units = 0.0
        rows.append(
            {
                "variant": variant,
                "validation_mode": validation_mode,
                "run_date": str(pd.to_datetime(run_date_token, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")),
                "player": str(row.get("player", "")),
                "target": target,
                "direction": direction,
                "market_id": f"{target}_{direction}",
                "market_line": float(line) if pd.notna(line) else np.nan,
                "market_side_price": float(market_price) if pd.notna(market_price) else np.nan,
                "prediction": float(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]) else np.nan,
                "expected_win_rate": float(pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("expected_win_rate")]), errors="coerce").iloc[0]) else np.nan,
                "board_play_win_prob": float(pd.to_numeric(pd.Series([row.get("board_play_win_prob")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("board_play_win_prob")]), errors="coerce").iloc[0]) else np.nan,
                "recommendation": str(row.get("recommendation", "")),
                "result": str(result),
                "units": float(units) if pd.notna(units) else np.nan,
                "actual": float(actual) if pd.notna(actual) else np.nan,
                "actual_source": str(actual_source),
                "actual_matched_date": str(matched_date),
                "pick_key": "",
                "rebound_diagnostic_segment": str(row.get("rebound_diagnostic_segment", "NOT_APPLICABLE")),
                "trb_over_bucket": str(row.get("trb_over_bucket", "")),
                "opposite_side_decision": str(row.get("opposite_side_decision", "")),
            }
        )
    resolved = pd.DataFrame.from_records(rows)
    if not resolved.empty:
        resolved["pick_key"] = _pick_key(resolved)
        return resolved.to_dict(orient="records")
    return rows


def _segment_mask(frame: pd.DataFrame, segment: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    if segment == "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY":
        return frame.get("rebound_diagnostic_segment", pd.Series("", index=frame.index)).astype(str).eq(segment)
    return (
        frame.get("market_id", pd.Series("", index=frame.index)).astype(str).eq("TRB_OVER")
        & frame.get("trb_over_bucket", pd.Series("", index=frame.index)).astype(str).str.contains(re.escape(segment), regex=True)
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _run_date_label(run_date_token: str) -> str:
    parsed = pd.to_datetime(run_date_token, format="%Y%m%d", errors="coerce")
    if pd.notna(parsed):
        return str(parsed.strftime("%Y-%m-%d"))
    parsed = pd.to_datetime(run_date_token, errors="coerce")
    if pd.notna(parsed):
        return str(parsed.strftime("%Y-%m-%d"))
    return str(run_date_token)


def _market_id_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    if "market_id" in frame.columns:
        return frame["market_id"].astype(str)
    target = frame.get("target", pd.Series("", index=frame.index)).astype(str).str.upper().str.strip()
    direction = frame.get("direction", pd.Series("", index=frame.index)).astype(str).str.upper().str.strip()
    return target + "_" + direction


def _is_trb_over_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return _market_id_series(frame).eq("TRB_OVER")


def _is_trb_under_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return _market_id_series(frame).eq("TRB_UNDER")


def _is_non_rebound_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return ~_market_id_series(frame).str.startswith("TRB_")


def _risky_trb_over_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    total_penalty = pd.to_numeric(frame.get("total_rebound_penalty", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    bucket = frame.get("trb_over_bucket", pd.Series("", index=frame.index)).astype(str)
    return _is_trb_over_mask(frame) & ((total_penalty > 0.0) | bucket.str.contains("TRB_OVER_(?!STABLE)", regex=True))


def _float_or_none(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _bool_or_none(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    return bool(value)


def _listify_reasons(*reasons: tuple[bool, str]) -> list[str]:
    return [message for condition, message in reasons if condition]


def _join_reason(messages: list[str], success_message: str) -> str:
    if not messages:
        return success_message
    return "; ".join(messages)


def _near_zero(value: Any, tolerance: float = DEFAULT_RESULT_TOLERANCE) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return True
    return bool(abs(float(numeric)) <= tolerance)


def _safe_delta(current: Any, baseline: Any) -> float:
    current_num = pd.to_numeric(pd.Series([current]), errors="coerce").iloc[0]
    baseline_num = pd.to_numeric(pd.Series([baseline]), errors="coerce").iloc[0]
    if pd.isna(current_num) or pd.isna(baseline_num):
        return np.nan
    return float(current_num - baseline_num)


def _selected_board_keys(board_df: pd.DataFrame, run_date_token: str) -> set[str]:
    if board_df.empty:
        return set()
    tagged = board_df.copy()
    tagged["run_date"] = _run_date_label(run_date_token)
    return set(_pick_key(tagged).tolist())


def _capture_selector_rows(
    selector_df: pd.DataFrame,
    board_df: pd.DataFrame,
    *,
    variant: str,
    validation_mode: str,
    run_date_token: str,
) -> list[dict[str, Any]]:
    if selector_df.empty:
        return []
    run_date = _run_date_label(run_date_token)
    selected_keys = _selected_board_keys(board_df, run_date_token)
    tagged = selector_df.copy()
    tagged["run_date"] = run_date
    tagged["market_id"] = _market_id_series(tagged)
    tagged["pick_key"] = _pick_key(tagged)
    tagged["selected_on_board"] = tagged["pick_key"].isin(selected_keys)
    records: list[dict[str, Any]] = []
    for _, row in tagged.iterrows():
        records.append(
            {
                "variant": variant,
                "validation_mode": validation_mode,
                "run_date": run_date,
                "player": str(row.get("player", "")),
                "target": str(row.get("target", "")).upper().strip(),
                "direction": str(row.get("direction", "")).upper().strip(),
                "market_id": str(row.get("market_id", "")),
                "market_line": _float_or_none(row.get("market_line")),
                "recommendation": str(row.get("recommendation", "")),
                "raw_recommendation": str(row.get("raw_recommendation", "")),
                "expected_win_rate": _float_or_none(row.get("expected_win_rate")),
                "board_play_win_prob": _float_or_none(row.get("board_play_win_prob")),
                "market_side_price": _float_or_none(row.get("market_side_price")),
                "market_side_break_even": _float_or_none(row.get("market_side_break_even")),
                "rebound_diagnostic_segment": str(row.get("rebound_diagnostic_segment", "NOT_APPLICABLE")),
                "trb_over_bucket": str(row.get("trb_over_bucket", "NOT_APPLICABLE")),
                "trb_over_bucket_reasons": str(row.get("trb_over_bucket_reasons", "")),
                "total_rebound_penalty": _float_or_none(row.get("total_rebound_penalty")),
                "adjusted_abs_edge": _float_or_none(row.get("adjusted_abs_edge")),
                "adjusted_stress_prob": _float_or_none(row.get("adjusted_stress_prob")),
                "adjusted_lcb_edge": _float_or_none(row.get("adjusted_lcb_edge")),
                "upper_band_line_penalty": _float_or_none(row.get("upper_band_line_penalty")),
                "low_line_role_volatility_penalty": _float_or_none(row.get("low_line_role_volatility_penalty")),
                "rebound_supply_penalty": _float_or_none(row.get("rebound_supply_penalty")),
                "rebound_share_competition_penalty": _float_or_none(row.get("rebound_share_competition_penalty")),
                "opposite_side_candidate_flag": bool(row.get("opposite_side_candidate_flag", False)),
                "opposite_side_reason": str(row.get("opposite_side_reason", "")),
                "opposite_side_market_type": str(row.get("opposite_side_market_type", "")),
                "opposite_side_line": _float_or_none(row.get("opposite_side_line")),
                "opposite_side_odds": _float_or_none(row.get("opposite_side_odds")),
                "opposite_side_break_even": _float_or_none(row.get("opposite_side_break_even")),
                "opposite_side_stress_prob": _float_or_none(row.get("opposite_side_stress_prob")),
                "opposite_side_lcb_edge": _float_or_none(row.get("opposite_side_lcb_edge")),
                "opposite_side_decision": str(row.get("opposite_side_decision", "not_evaluated")),
                "selected_on_board": bool(row.get("selected_on_board", False)),
                "pick_key": str(row.get("pick_key", "")),
            }
        )
    return records


def _rows_metrics(part: pd.DataFrame) -> dict[str, Any]:
    resolved = part.loc[part["result"].isin(["win", "loss"])].copy()
    wins = int((resolved["result"] == "win").sum())
    losses = int((resolved["result"] == "loss").sum())
    pushes = int((part["result"] == "push").sum())
    labels = (resolved["result"] == "win").astype(float).to_numpy(dtype="float64", copy=False) if not resolved.empty else np.array([], dtype="float64")
    if resolved.empty:
        probs = np.array([], dtype="float64")
    else:
        base_prob = pd.to_numeric(
            resolved.get("board_play_win_prob", pd.Series(np.nan, index=resolved.index)),
            errors="coerce",
        )
        fallback_prob = pd.to_numeric(
            resolved.get("expected_win_rate", pd.Series(np.nan, index=resolved.index)),
            errors="coerce",
        )
        probs = base_prob.fillna(fallback_prob).fillna(0.5).to_numpy(dtype="float64", copy=False)
    brier = float(np.mean((probs - labels) ** 2)) if resolved.shape[0] else np.nan
    ece = _ece_10(probs, labels) if resolved.shape[0] else np.nan
    calibration_gap = float(np.mean(probs - labels)) if resolved.shape[0] else np.nan
    return {
        "total_picks": int(len(part)),
        "resolved_picks": int(len(resolved)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": _safe_rate(wins, len(resolved)),
        "profit_units": float(
            pd.to_numeric(part.get("units", pd.Series(0.0, index=part.index)), errors="coerce").fillna(0.0).sum()
        )
        if not part.empty
        else 0.0,
        "roi": _safe_rate(
            float(pd.to_numeric(part.get("units", pd.Series(0.0, index=part.index)), errors="coerce").fillna(0.0).sum()),
            len(resolved),
        ),
        "brier": brier,
        "ece": ece,
        "calibration_gap": calibration_gap,
    }


def _market_hit_rate(part: pd.DataFrame, market_id: str) -> float:
    market_rows = part.loc[part["market_id"].astype(str) == market_id].copy()
    resolved = market_rows.loc[market_rows["result"].isin(["win", "loss"])].copy()
    if resolved.empty:
        return np.nan
    return float((resolved["result"] == "win").mean())


def _compare_to_baseline(part: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    if baseline.empty:
        return {
            "coverage_retained": np.nan,
            "removed_trb_over_wins": 0,
            "removed_trb_over_losses": 0,
            "kept_trb_over_wins": 0,
            "kept_trb_over_losses": 0,
            "loss_removal_rate": np.nan,
            "win_preservation_rate": np.nan,
            "board_change_count": 0,
            "non_rebound_board_change_count": 0,
            "trb_board_change_count": 0,
            "net_precision_delta": np.nan,
            "non_rebound_hit_rate": np.nan,
            "non_rebound_hit_rate_delta": np.nan,
            "trb_under_added_count": 0,
            "opposite_under_added_count": 0,
        }

    baseline_keys = set(_pick_key(baseline).tolist())
    variant_keys = set(_pick_key(part).tolist())
    baseline_market_id = _market_id_series(baseline)
    variant_market_id = _market_id_series(part)
    baseline_trb_over = baseline.loc[baseline_market_id == "TRB_OVER"].copy()
    baseline_trb_over = baseline_trb_over.assign(pick_key=_pick_key(baseline_trb_over))
    removed = baseline_trb_over.loc[~baseline_trb_over["pick_key"].isin(variant_keys)].copy()
    kept = baseline_trb_over.loc[baseline_trb_over["pick_key"].isin(variant_keys)].copy()

    removed_wins = int((removed["result"] == "win").sum())
    removed_losses = int((removed["result"] == "loss").sum())
    kept_wins = int((kept["result"] == "win").sum())
    kept_losses = int((kept["result"] == "loss").sum())
    baseline_losses = int((baseline_trb_over["result"] == "loss").sum())
    baseline_wins = int((baseline_trb_over["result"] == "win").sum())

    baseline_trb_hit = _market_hit_rate(baseline.loc[baseline_market_id.str.startswith("TRB_")].copy(), "TRB_OVER")
    variant_trb_hit = _market_hit_rate(part.loc[variant_market_id.str.startswith("TRB_")].copy(), "TRB_OVER")
    baseline_non_rebound = baseline.loc[_is_non_rebound_mask(baseline)].copy()
    variant_non_rebound = part.loc[_is_non_rebound_mask(part)].copy()
    baseline_non_rebound_hit = _rows_metrics(baseline_non_rebound)["hit_rate"]
    variant_non_rebound_hit = _rows_metrics(variant_non_rebound)["hit_rate"]
    baseline_non_rebound_keys = set(_pick_key(baseline_non_rebound).tolist())
    variant_non_rebound_keys = set(_pick_key(variant_non_rebound).tolist())
    variant_trb_under = part.loc[variant_market_id == "TRB_UNDER"].copy()
    variant_trb_under = variant_trb_under.assign(pick_key=_pick_key(variant_trb_under))
    opposite_added = variant_trb_under.loc[
        variant_trb_under.get("rebound_diagnostic_segment", pd.Series("", index=variant_trb_under.index))
        .astype(str)
        .eq("TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY")
    ].copy()
    return {
        "coverage_retained": _safe_rate(len(part), len(baseline)),
        "removed_trb_over_wins": removed_wins,
        "removed_trb_over_losses": removed_losses,
        "kept_trb_over_wins": kept_wins,
        "kept_trb_over_losses": kept_losses,
        "loss_removal_rate": _safe_rate(removed_losses, baseline_losses),
        "win_preservation_rate": _safe_rate(kept_wins, baseline_wins),
        "board_change_count": int(len(baseline_keys.symmetric_difference(variant_keys))),
        "non_rebound_board_change_count": int(len(baseline_non_rebound_keys.symmetric_difference(variant_non_rebound_keys))),
        "trb_board_change_count": int(len(baseline_keys.symmetric_difference(variant_keys)) - len(baseline_non_rebound_keys.symmetric_difference(variant_non_rebound_keys))),
        "net_precision_delta": float(variant_trb_hit - baseline_trb_hit) if pd.notna(variant_trb_hit) and pd.notna(baseline_trb_hit) else np.nan,
        "non_rebound_hit_rate": variant_non_rebound_hit,
        "non_rebound_hit_rate_delta": float(variant_non_rebound_hit - baseline_non_rebound_hit) if pd.notna(variant_non_rebound_hit) and pd.notna(baseline_non_rebound_hit) else np.nan,
        "trb_under_added_count": int((~variant_trb_under["pick_key"].isin(baseline_keys)).sum()),
        "opposite_under_added_count": int((~opposite_added["pick_key"].isin(baseline_keys)).sum()),
    }


def _classify_validation_window(day_df: pd.DataFrame) -> str:
    if day_df.empty:
        return NO_OP_WINDOW
    day_types = set(day_df.get("day_type", pd.Series(dtype=str)).dropna().astype(str).tolist())
    if not day_types or day_types == {NO_OP_DAY}:
        return NO_OP_WINDOW
    if day_types == {ACTIVE_DAY}:
        return ACTIVE_WINDOW
    return MIXED_WINDOW


def _window_day_summary(
    selector_df: pd.DataFrame,
    board_df: pd.DataFrame,
    *,
    validation_mode: str,
    variant: str,
) -> pd.DataFrame:
    baseline_selector = selector_df.loc[
        (selector_df["validation_mode"] == validation_mode) & (selector_df["variant"] == BASELINE_VARIANT)
    ].copy()
    variant_selector = selector_df.loc[
        (selector_df["validation_mode"] == validation_mode) & (selector_df["variant"] == variant)
    ].copy()
    baseline_board = board_df.loc[
        (board_df["validation_mode"] == validation_mode) & (board_df["variant"] == BASELINE_VARIANT)
    ].copy()
    full_diag_board = board_df.loc[
        (board_df["validation_mode"] == validation_mode) & (board_df["variant"] == FULL_DIAGNOSTICS_VARIANT)
    ].copy()

    run_dates = sorted(
        set(baseline_selector.get("run_date", pd.Series(dtype=str)).astype(str).tolist())
        | set(variant_selector.get("run_date", pd.Series(dtype=str)).astype(str).tolist())
        | set(baseline_board.get("run_date", pd.Series(dtype=str)).astype(str).tolist())
    )
    records: list[dict[str, Any]] = []
    for run_date in run_dates:
        baseline_selector_day = baseline_selector.loc[baseline_selector["run_date"] == run_date].copy()
        variant_selector_day = variant_selector.loc[variant_selector["run_date"] == run_date].copy()
        baseline_board_day = baseline_board.loc[baseline_board["run_date"] == run_date].copy()
        full_diag_board_day = full_diag_board.loc[full_diag_board["run_date"] == run_date].copy()
        candidate_pool_trb_over_count = int(_is_trb_over_mask(baseline_selector_day).sum())
        risky_trb_over_candidate_count = int(_risky_trb_over_mask(variant_selector_day).sum())
        diagnostics_trigger_count = risky_trb_over_candidate_count
        baseline_board_trb_over_count = int(_is_trb_over_mask(baseline_board_day).sum())
        full_diag_board_trb_over_count = int(_is_trb_over_mask(full_diag_board_day).sum())
        active_rebound_risk_present = bool(baseline_board_trb_over_count > 0)
        records.append(
            {
                "validation_mode": validation_mode,
                "variant": variant,
                "run_date": run_date,
                "candidate_pool_trb_over_count": candidate_pool_trb_over_count,
                "risky_trb_over_candidate_count": risky_trb_over_candidate_count,
                "diagnostics_trigger_count": diagnostics_trigger_count,
                "final_board_trb_over_count_baseline": baseline_board_trb_over_count,
                "final_board_trb_over_count_full_diagnostics": full_diag_board_trb_over_count,
                "active_rebound_risk_present": active_rebound_risk_present,
                "day_type": ACTIVE_DAY if active_rebound_risk_present else NO_OP_DAY,
            }
        )
    return pd.DataFrame.from_records(records)


def _evaluate_no_op_narrowness(
    variant_part: pd.DataFrame,
    baseline_part: pd.DataFrame,
    selector_part: pd.DataFrame,
    *,
    no_op_dates: set[str],
    coverage_threshold: float,
    board_change_tolerance: int,
) -> dict[str, Any]:
    empty_payload = {
        "passed": False,
        "reason": "no_no_op_narrowness_days_present",
        "board_change_count": 0,
        "non_rebound_board_change_count": 0,
        "non_rebound_hit_rate_delta": np.nan,
        "coverage_retained": np.nan,
        "final_board_trb_over_count": 0,
        "diagnostics_trigger_count": 0,
        "synthetic_under_added_count": 0,
        "synthetic_under_price_valid_count": 0,
        "overtrigger_warning": False,
    }
    if not no_op_dates:
        return empty_payload

    variant_subset = variant_part.loc[variant_part["run_date"].isin(no_op_dates)].copy()
    baseline_subset = baseline_part.loc[baseline_part["run_date"].isin(no_op_dates)].copy()
    selector_subset = selector_part.loc[selector_part["run_date"].isin(no_op_dates)].copy()
    compare = _compare_to_baseline(variant_subset, baseline_subset)
    diagnostics_trigger_count = int(_risky_trb_over_mask(selector_subset).sum())
    synthetic_under_rows = variant_subset.loc[
        variant_subset.get("rebound_diagnostic_segment", pd.Series("", index=variant_subset.index))
        .astype(str)
        .eq("TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY")
    ].copy()
    synthetic_under_price_valid_count = int(
        pd.to_numeric(synthetic_under_rows.get("market_side_price", pd.Series(np.nan, index=synthetic_under_rows.index)), errors="coerce")
        .notna()
        .sum()
    )
    pass_reasons = _listify_reasons(
        (int(compare.get("board_change_count", 0)) > int(board_change_tolerance), f"board_change_count={int(compare.get('board_change_count', 0))}>{int(board_change_tolerance)}"),
        (int(compare.get("non_rebound_board_change_count", 0)) != 0, f"non_rebound_board_change_count={int(compare.get('non_rebound_board_change_count', 0))}"),
        (
            not (pd.isna(compare.get("coverage_retained", np.nan)) or float(compare.get("coverage_retained", 0.0)) >= coverage_threshold),
            f"coverage_retained={float(compare.get('coverage_retained', np.nan)):.3f}<{coverage_threshold:.3f}",
        ),
        (
            not _near_zero(compare.get("non_rebound_hit_rate_delta", np.nan)),
            f"non_rebound_hit_rate_delta={float(compare.get('non_rebound_hit_rate_delta', 0.0)):+.4f}",
        ),
        (
            len(synthetic_under_rows) > 0 and synthetic_under_price_valid_count < len(synthetic_under_rows),
            "synthetic_under_added_without_price_validation",
        ),
    )
    passed = not pass_reasons
    return {
        "passed": passed,
        "reason": _join_reason(pass_reasons, "no_op_window_preserved_narrowness"),
        "board_change_count": int(compare.get("board_change_count", 0)),
        "non_rebound_board_change_count": int(compare.get("non_rebound_board_change_count", 0)),
        "non_rebound_hit_rate_delta": compare.get("non_rebound_hit_rate_delta", np.nan),
        "coverage_retained": compare.get("coverage_retained", np.nan),
        "final_board_trb_over_count": int(_is_trb_over_mask(variant_subset).sum()),
        "diagnostics_trigger_count": diagnostics_trigger_count,
        "synthetic_under_added_count": int(len(synthetic_under_rows)),
        "synthetic_under_price_valid_count": synthetic_under_price_valid_count,
        "overtrigger_warning": bool(int(compare.get("board_change_count", 0)) > int(board_change_tolerance) or int(compare.get("non_rebound_board_change_count", 0)) > 0),
    }


def _evaluate_active_improvement(
    variant_part: pd.DataFrame,
    baseline_part: pd.DataFrame,
    *,
    active_dates: set[str],
    full_variant_part: pd.DataFrame | None = None,
    full_baseline_part: pd.DataFrame | None = None,
    coverage_threshold: float,
    win_preservation_floor: float,
) -> dict[str, Any]:
    empty_payload = {
        "passed": False,
        "reason": "no_active_rebound_risk_days_present",
        "removed_trb_over_wins": 0,
        "removed_trb_over_losses": 0,
        "kept_trb_over_wins": 0,
        "kept_trb_over_losses": 0,
        "win_preservation_rate": np.nan,
        "loss_removal_rate": np.nan,
        "board_change_count": 0,
        "non_rebound_board_change_count": 0,
        "coverage_retained": np.nan,
        "roi_delta": np.nan,
        "brier_delta": np.nan,
        "ece_delta": np.nan,
        "hit_rate_delta": np.nan,
        "profit_units_delta": np.nan,
        "non_rebound_hit_rate_delta": np.nan,
    }
    if not active_dates:
        return empty_payload

    variant_subset = variant_part.loc[variant_part["run_date"].isin(active_dates)].copy()
    baseline_subset = baseline_part.loc[baseline_part["run_date"].isin(active_dates)].copy()
    removal_compare = _compare_to_baseline(variant_subset, baseline_subset)
    eval_variant = full_variant_part.copy() if isinstance(full_variant_part, pd.DataFrame) else variant_subset
    eval_baseline = full_baseline_part.copy() if isinstance(full_baseline_part, pd.DataFrame) else baseline_subset
    compare = _compare_to_baseline(eval_variant, eval_baseline)
    variant_metrics = _rows_metrics(eval_variant)
    baseline_metrics = _rows_metrics(eval_baseline)
    roi_delta = _safe_delta(variant_metrics.get("roi"), baseline_metrics.get("roi"))
    brier_delta = _safe_delta(variant_metrics.get("brier"), baseline_metrics.get("brier"))
    ece_delta = _safe_delta(variant_metrics.get("ece"), baseline_metrics.get("ece"))
    hit_rate_delta = _safe_delta(variant_metrics.get("hit_rate"), baseline_metrics.get("hit_rate"))
    profit_units_delta = _safe_delta(variant_metrics.get("profit_units"), baseline_metrics.get("profit_units"))

    removed_losses = int(removal_compare.get("removed_trb_over_losses", 0))
    removed_wins = int(removal_compare.get("removed_trb_over_wins", 0))
    win_preservation_rate = removal_compare.get("win_preservation_rate", np.nan)
    win_preservation_ok = (
        removed_wins == 0
        or pd.isna(win_preservation_rate)
        or float(win_preservation_rate) >= float(win_preservation_floor)
    )
    fail_reasons = _listify_reasons(
        (removed_losses <= removed_wins, f"removed_losses={removed_losses}<=removed_wins={removed_wins}"),
        (not win_preservation_ok, f"win_preservation_rate={float(win_preservation_rate):.3f}<{float(win_preservation_floor):.3f}"),
        (
            not (pd.isna(compare.get("coverage_retained", np.nan)) or float(compare.get("coverage_retained", 0.0)) >= coverage_threshold),
            f"coverage_retained={float(compare.get('coverage_retained', np.nan)):.3f}<{coverage_threshold:.3f}",
        ),
        (pd.notna(roi_delta) and float(roi_delta) < -DEFAULT_RESULT_TOLERANCE, f"roi_delta={float(roi_delta):+.4f}"),
        (pd.notna(brier_delta) and float(brier_delta) > DEFAULT_RESULT_TOLERANCE, f"brier_delta={float(brier_delta):+.4f}"),
        (pd.notna(ece_delta) and float(ece_delta) > DEFAULT_RESULT_TOLERANCE, f"ece_delta={float(ece_delta):+.4f}"),
        (int(compare.get("non_rebound_board_change_count", 0)) != 0, f"non_rebound_board_change_count={int(compare.get('non_rebound_board_change_count', 0))}"),
        (not _near_zero(compare.get("non_rebound_hit_rate_delta", np.nan)), f"non_rebound_hit_rate_delta={float(compare.get('non_rebound_hit_rate_delta', 0.0)):+.4f}"),
    )
    passed = not fail_reasons
    return {
        "passed": passed,
        "reason": _join_reason(fail_reasons, "active_window_removed_more_losses_than_wins"),
        "removed_trb_over_wins": removed_wins,
        "removed_trb_over_losses": removed_losses,
        "kept_trb_over_wins": int(removal_compare.get("kept_trb_over_wins", 0)),
        "kept_trb_over_losses": int(removal_compare.get("kept_trb_over_losses", 0)),
        "win_preservation_rate": win_preservation_rate,
        "loss_removal_rate": removal_compare.get("loss_removal_rate", np.nan),
        "board_change_count": int(compare.get("board_change_count", 0)),
        "non_rebound_board_change_count": int(compare.get("non_rebound_board_change_count", 0)),
        "coverage_retained": compare.get("coverage_retained", np.nan),
        "roi_delta": roi_delta,
        "brier_delta": brier_delta,
        "ece_delta": ece_delta,
        "hit_rate_delta": hit_rate_delta,
        "profit_units_delta": profit_units_delta,
        "non_rebound_hit_rate_delta": compare.get("non_rebound_hit_rate_delta", np.nan),
    }


def _build_opposite_under_audit(selector_part: pd.DataFrame, board_part: pd.DataFrame) -> dict[str, Any]:
    trb_over_selector = selector_part.loc[_is_trb_over_mask(selector_part)].copy()
    flagged_over = trb_over_selector.loc[
        pd.to_numeric(trb_over_selector.get("total_rebound_penalty", pd.Series(0.0, index=trb_over_selector.index)), errors="coerce").fillna(0.0) > 0.0
    ].copy()
    synthetic_selector = selector_part.loc[
        selector_part.get("rebound_diagnostic_segment", pd.Series("", index=selector_part.index))
        .astype(str)
        .eq("TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY")
    ].copy()
    added_under_rows = board_part.loc[
        board_part.get("rebound_diagnostic_segment", pd.Series("", index=board_part.index))
        .astype(str)
        .eq("TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY")
    ].copy()

    valid_price_mask = (
        pd.to_numeric(flagged_over.get("opposite_side_odds", pd.Series(np.nan, index=flagged_over.index)), errors="coerce").notna()
        & pd.to_numeric(flagged_over.get("opposite_side_break_even", pd.Series(np.nan, index=flagged_over.index)), errors="coerce").notna()
    )
    break_even_mask = (
        pd.to_numeric(flagged_over.get("opposite_side_stress_prob", pd.Series(np.nan, index=flagged_over.index)), errors="coerce")
        > pd.to_numeric(flagged_over.get("opposite_side_break_even", pd.Series(np.nan, index=flagged_over.index)), errors="coerce")
    )
    decision_series = flagged_over.get("opposite_side_decision", pd.Series("", index=flagged_over.index)).astype(str)
    added_metrics = _rows_metrics(added_under_rows)
    return {
        "enabled": bool(not selector_part.empty and selector_part["variant"].astype(str).eq(PROMOTION_TARGET_VARIANT).any()),
        "flagged_over_count": int(len(flagged_over)),
        "synthetic_under_candidates_created": int(len(synthetic_selector)),
        "under_candidates_with_valid_price": int(valid_price_mask.sum()),
        "under_candidates_passing_break_even": int(break_even_mask.sum()),
        "under_candidates_added_to_board": int(len(added_under_rows)),
        "under_candidates_rejected_price": int(decision_series.eq("reject_price_unavailable").sum()),
        "under_candidates_rejected_forecastability": int(decision_series.eq("reject_forecastability").sum()),
        "under_candidates_rejected_stress": int(decision_series.isin(["reject_break_even", "reject_lcb_edge"]).sum()),
        "under_candidate_results": {
            "resolved_picks": int(added_metrics.get("resolved_picks", 0)),
            "wins": int(added_metrics.get("wins", 0)),
            "losses": int(added_metrics.get("losses", 0)),
            "pushes": int(added_metrics.get("pushes", 0)),
            "hit_rate": added_metrics.get("hit_rate", np.nan),
            "profit_units": added_metrics.get("profit_units", 0.0),
            "roi": added_metrics.get("roi", np.nan),
        },
        "added_under_rows": [
            {
                "player": str(row.get("player", "")),
                "game_date": str(row.get("run_date", "")),
                "line": _float_or_none(row.get("market_line")),
                "odds": _float_or_none(row.get("market_side_price")),
                "reason": str(row.get("trb_over_bucket_reasons", row.get("opposite_side_reason", ""))),
                "result": str(row.get("result", "")),
            }
            for _, row in added_under_rows.iterrows()
        ],
    }


def _segment_report(
    selector_part: pd.DataFrame,
    board_part: pd.DataFrame,
    baseline_part: pd.DataFrame,
    *,
    validation_mode: str,
    variant: str,
    segment: str,
) -> dict[str, Any]:
    seg_candidates = selector_part.loc[_segment_mask(selector_part, segment)].copy()
    seg_board = board_part.loc[_segment_mask(board_part, segment)].copy()
    seg_metrics = _rows_metrics(seg_board)
    baseline_trb_over = baseline_part.loc[_is_trb_over_mask(baseline_part)].copy()
    baseline_trb_over = baseline_trb_over.assign(pick_key=_pick_key(baseline_trb_over))
    seg_candidates = seg_candidates.assign(pick_key=_pick_key(seg_candidates)) if not seg_candidates.empty else seg_candidates
    variant_board_keys = set(_pick_key(board_part).tolist()) if not board_part.empty else set()

    removed = pd.DataFrame()
    kept = pd.DataFrame()
    if segment != "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY" and not seg_candidates.empty and not baseline_trb_over.empty:
        seg_candidate_keys = set(seg_candidates["pick_key"].tolist())
        removed = baseline_trb_over.loc[
            baseline_trb_over["pick_key"].isin(seg_candidate_keys) & ~baseline_trb_over["pick_key"].isin(variant_board_keys)
        ].copy()
        kept = baseline_trb_over.loc[
            baseline_trb_over["pick_key"].isin(seg_candidate_keys) & baseline_trb_over["pick_key"].isin(variant_board_keys)
        ].copy()

    return {
        "validation_mode": validation_mode,
        "variant": variant,
        "segment": segment,
        "candidate_count": int(len(seg_candidates)),
        "final_board_count": int(len(seg_board)),
        "removed_count": int(len(removed)),
        "kept_count": int(len(kept)),
        "wins_removed": int((removed.get("result", pd.Series(dtype=str)) == "win").sum()),
        "losses_removed": int((removed.get("result", pd.Series(dtype=str)) == "loss").sum()),
        "wins_kept": int((kept.get("result", pd.Series(dtype=str)) == "win").sum()),
        "losses_kept": int((kept.get("result", pd.Series(dtype=str)) == "loss").sum()),
        "total_picks": int(seg_metrics.get("total_picks", 0)),
        "resolved_picks": int(seg_metrics.get("resolved_picks", 0)),
        "wins": int(seg_metrics.get("wins", 0)),
        "losses": int(seg_metrics.get("losses", 0)),
        "pushes": int(seg_metrics.get("pushes", 0)),
        "hit_rate": seg_metrics.get("hit_rate", np.nan),
        "profit_units": seg_metrics.get("profit_units", 0.0),
        "roi": seg_metrics.get("roi", np.nan),
        "brier": seg_metrics.get("brier", np.nan),
        "ece": seg_metrics.get("ece", np.nan),
        "calibration_gap": seg_metrics.get("calibration_gap", np.nan),
        "avg_penalty": float(pd.to_numeric(seg_candidates.get("total_rebound_penalty", pd.Series(np.nan, index=seg_candidates.index)), errors="coerce").mean()) if not seg_candidates.empty else np.nan,
        "avg_adjusted_edge": float(pd.to_numeric(seg_candidates.get("adjusted_abs_edge", pd.Series(np.nan, index=seg_candidates.index)), errors="coerce").mean()) if not seg_candidates.empty else np.nan,
        "avg_lcb_edge": float(pd.to_numeric(seg_candidates.get("adjusted_lcb_edge", pd.Series(np.nan, index=seg_candidates.index)), errors="coerce").mean()) if not seg_candidates.empty else np.nan,
    }


def _status_label(
    *,
    variant: str,
    validation_window_type: str,
    no_op_validation: dict[str, Any],
    active_validation: dict[str, Any],
    resolved_picks: int,
    min_resolved_picks: int,
) -> str:
    if variant == BASELINE_VARIANT:
        return "baseline"
    if int(resolved_picks) < int(min_resolved_picks):
        return "needs_more_sample"
    if validation_window_type == NO_OP_WINDOW:
        return "no_op_narrowness_pass" if bool(no_op_validation.get("passed", False)) else "rejected_overfit"
    if validation_window_type == ACTIVE_WINDOW:
        return "logic_improvement_pass" if bool(active_validation.get("passed", False)) else "rejected_overfit"
    no_op_pass = bool(no_op_validation.get("passed", False))
    active_pass = bool(active_validation.get("passed", False))
    if no_op_pass and active_pass:
        return "logic_improvement_pass"
    if no_op_pass and not active_pass:
        return "rejected_overfit"
    if active_pass and not no_op_pass:
        return "rejected_overfit"
    return "rejected_overfit"


def main() -> None:
    args = parse_args()
    daily_run_dirs = [path.resolve() for path in args.daily_runs_dir] if args.daily_runs_dir else [REPO_ROOT / "model" / "analysis" / "daily_runs"]
    history_csv = args.history_csv.resolve()
    history_df = pd.read_csv(history_csv)
    history_lookup = build_history_lookup(history_df)
    line_decision_lookup = build_line_decision_lookup(history_df)
    history_actual_lookup = _build_actual_lookup(history_csv)
    data_proc_actual_lookup = _build_data_proc_actual_lookup(
        args.data_proc_root.resolve(),
        args.start_run_date,
        args.end_run_date,
    )
    selected_board_calibrator, _ = load_selected_board_calibrator(args.selected_board_calibrator_json, disabled=False)
    row_records: list[dict[str, Any]] = []
    selector_records: list[dict[str, Any]] = []
    line_decision_config = LineDecisionConfig(
        no_trade_threshold=float(args.no_trade_threshold),
        min_trade_prob=float(args.min_trade_prob),
        min_trade_prob_gap=float(args.min_trade_prob_gap),
    )

    for daily_runs_dir in daily_run_dirs:
        run_dates = _iter_run_dates(daily_runs_dir, args.start_run_date, args.end_run_date, args.max_days)
        for run_date in run_dates:
            run_dir = daily_runs_dir / run_date
            slate_csv = run_dir / f"upcoming_market_slate_{run_date}.csv"
            if not slate_csv.exists():
                continue
            slate_df = pd.read_csv(slate_csv)
            if slate_df.empty:
                continue
            policy_payload = _load_daily_policy(run_dir)
            if not policy_payload:
                continue
            validation_mode = _infer_validation_mode(run_dir)
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

            for variant in VARIANT_ORDER:
                variant_policy = _variant_policy_payload(policy_payload, variant)
                selector_df = _prepare_selector(
                    slate_df,
                    history_df,
                    history_lookup,
                    line_decision_lookup,
                    variant_policy,
                    line_decision_enabled=True,
                    line_decision_config=line_decision_config,
                )
                board_df = _compute_board(
                    selector_df,
                    variant_policy,
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
                    validation_mode=validation_mode,
                    run_date_token=run_date,
                    fallback_odds=int(variant_policy.get("american_odds", -110)),
                )
                selector_records.extend(
                    _capture_selector_rows(
                        selector_df,
                        board_df,
                        variant=variant,
                        validation_mode=validation_mode,
                        run_date_token=run_date,
                    )
                )
                row_records.extend(resolved_rows)

    rows_df = pd.DataFrame.from_records(row_records)
    if rows_df.empty:
        raise RuntimeError("Replay produced no resolved rebound-diagnostic rows.")
    selector_df = pd.DataFrame.from_records(selector_records)
    if not selector_df.empty:
        selected_results = rows_df[
            [
                "variant",
                "validation_mode",
                "run_date",
                "pick_key",
                "result",
                "units",
                "actual",
                "actual_source",
                "actual_matched_date",
            ]
        ].drop_duplicates(subset=["variant", "validation_mode", "run_date", "pick_key"])
        selector_df = selector_df.merge(
            selected_results.rename(
                columns={
                    "result": "selected_result",
                    "units": "selected_units",
                    "actual": "selected_actual",
                    "actual_source": "selected_actual_source",
                    "actual_matched_date": "selected_actual_matched_date",
                }
            ),
            on=["variant", "validation_mode", "run_date", "pick_key"],
            how="left",
        )

    summary_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    window_reports: list[dict[str, Any]] = []
    for validation_mode, mode_part in rows_df.groupby("validation_mode", sort=False):
        baseline_part = mode_part.loc[mode_part["variant"] == BASELINE_VARIANT].copy()
        baseline_metrics = _rows_metrics(baseline_part)
        for variant in VARIANT_ORDER:
            variant_part = mode_part.loc[mode_part["variant"] == variant].copy()
            selector_part = selector_df.loc[
                (selector_df["validation_mode"] == validation_mode) & (selector_df["variant"] == variant)
            ].copy()
            metrics = _rows_metrics(variant_part)
            metrics.update(
                {
                    "validation_mode": validation_mode,
                    "variant": variant,
                    "trb_over_hit_rate": _market_hit_rate(variant_part, "TRB_OVER"),
                    "trb_under_hit_rate": _market_hit_rate(variant_part, "TRB_UNDER"),
                    "final_board_trb_over_count": int(_is_trb_over_mask(variant_part).sum()),
                    "final_board_trb_under_count": int(_is_trb_under_mask(variant_part).sum()),
                }
            )
            metrics.update(_compare_to_baseline(variant_part, baseline_part))

            day_df = _window_day_summary(selector_df, rows_df, validation_mode=validation_mode, variant=variant)
            validation_window_type = _classify_validation_window(day_df)
            no_op_dates = set(day_df.loc[day_df["day_type"] == NO_OP_DAY, "run_date"].astype(str).tolist()) if not day_df.empty else set()
            active_dates = set(day_df.loc[day_df["day_type"] == ACTIVE_DAY, "run_date"].astype(str).tolist()) if not day_df.empty else set()
            no_op_validation = _evaluate_no_op_narrowness(
                variant_part,
                baseline_part,
                selector_part,
                no_op_dates=no_op_dates,
                coverage_threshold=float(args.coverage_threshold),
                board_change_tolerance=int(args.no_op_board_change_tolerance),
            )
            active_validation = _evaluate_active_improvement(
                variant_part,
                baseline_part,
                active_dates=active_dates,
                full_variant_part=variant_part,
                full_baseline_part=baseline_part,
                coverage_threshold=float(args.coverage_threshold),
                win_preservation_floor=float(args.win_preservation_floor),
            )
            opposite_under_audit = _build_opposite_under_audit(selector_part, variant_part)
            status_label = _status_label(
                variant=variant,
                validation_window_type=validation_window_type,
                no_op_validation=no_op_validation,
                active_validation=active_validation,
                resolved_picks=int(metrics.get("resolved_picks", 0)),
                min_resolved_picks=int(args.min_resolved_picks),
            )

            window_report = {
                "validation_mode": validation_mode,
                "variant": variant,
                "validation_window_type": validation_window_type,
                "final_board_trb_over_count_baseline": int(day_df.get("final_board_trb_over_count_baseline", pd.Series(dtype=int)).sum()) if not day_df.empty else 0,
                "final_board_trb_over_count_full_diagnostics": int(day_df.get("final_board_trb_over_count_full_diagnostics", pd.Series(dtype=int)).sum()) if not day_df.empty else 0,
                "candidate_pool_trb_over_count": int(day_df.get("candidate_pool_trb_over_count", pd.Series(dtype=int)).sum()) if not day_df.empty else 0,
                "risky_trb_over_candidate_count": int(day_df.get("risky_trb_over_candidate_count", pd.Series(dtype=int)).sum()) if not day_df.empty else 0,
                "active_rebound_risk_present": bool(day_df.get("active_rebound_risk_present", pd.Series(dtype=bool)).any()) if not day_df.empty else False,
                "no_op_day_count": int((day_df.get("day_type", pd.Series(dtype=str)) == NO_OP_DAY).sum()) if not day_df.empty else 0,
                "active_day_count": int((day_df.get("day_type", pd.Series(dtype=str)) == ACTIVE_DAY).sum()) if not day_df.empty else 0,
                "no_op_narrowness_validation": no_op_validation,
                "active_improvement_validation": active_validation,
                "opposite_under_discovery": opposite_under_audit,
                "status_label": status_label,
            }
            window_reports.append(window_report)

            metrics.update(
                {
                    "validation_window_type": validation_window_type,
                    "final_board_trb_over_count_baseline": window_report["final_board_trb_over_count_baseline"],
                    "final_board_trb_over_count_full_diagnostics": window_report["final_board_trb_over_count_full_diagnostics"],
                    "candidate_pool_trb_over_count": window_report["candidate_pool_trb_over_count"],
                    "risky_trb_over_candidate_count": window_report["risky_trb_over_candidate_count"],
                    "active_rebound_risk_present": window_report["active_rebound_risk_present"],
                    "no_op_day_count": window_report["no_op_day_count"],
                    "active_day_count": window_report["active_day_count"],
                    "no_op_narrowness_passed": bool(no_op_validation.get("passed", False)),
                    "no_op_narrowness_reason": str(no_op_validation.get("reason", "")),
                    "active_improvement_passed": bool(active_validation.get("passed", False)),
                    "active_improvement_reason": str(active_validation.get("reason", "")),
                    "opposite_under_flagged_over_count": int(opposite_under_audit.get("flagged_over_count", 0)),
                    "opposite_under_added_to_board": int(opposite_under_audit.get("under_candidates_added_to_board", 0)),
                    "status_label": status_label,
                }
            )
            summary_records.append(metrics)

            for segment in SEGMENTS:
                segment_records.append(
                    _segment_report(
                        selector_part,
                        variant_part,
                        baseline_part,
                        validation_mode=validation_mode,
                        variant=variant,
                        segment=segment,
                    )
                )

    summary_df = pd.DataFrame.from_records(summary_records)
    segments_df = pd.DataFrame.from_records(segment_records)
    heuristic_target = [
        report
        for report in window_reports
        if report["validation_mode"] == "artifact_free_heuristic" and report["variant"] == PROMOTION_TARGET_VARIANT
    ]
    trained_target = [
        report
        for report in window_reports
        if report["validation_mode"] == "trained_bundle" and report["variant"] == PROMOTION_TARGET_VARIANT
    ]
    shadow_validated_logic = bool(
        any(bool(report["no_op_narrowness_validation"].get("passed", False)) for report in heuristic_target)
        and any(bool(report["active_improvement_validation"].get("passed", False)) for report in heuristic_target)
    )
    trained_bundle_validated = bool(
        any(bool(report["no_op_narrowness_validation"].get("passed", False)) for report in trained_target)
        and any(bool(report["active_improvement_validation"].get("passed", False)) for report in trained_target)
    )
    overall_status = "needs_more_sample"
    target_variant_rows = summary_df.loc[summary_df["variant"] == PROMOTION_TARGET_VARIANT].copy()
    if not target_variant_rows.empty:
        labels = target_variant_rows["status_label"].astype(str).tolist()
        if "logic_improvement_pass" in labels:
            overall_status = "logic_improvement_pass"
        elif "no_op_narrowness_pass" in labels:
            overall_status = "no_op_narrowness_pass"
        elif "rejected_overfit" in labels:
            overall_status = "rejected_overfit"
    payload = {
        "window": {
            "start_run_date": str(args.start_run_date),
            "end_run_date": str(args.end_run_date),
        },
        "modes_present": sorted(summary_df["validation_mode"].dropna().astype(str).unique().tolist()),
        "overall_status": overall_status,
        "shadow_only": True,
        "shadow_validated_logic": shadow_validated_logic,
        "trained_bundle_validated": trained_bundle_validated,
        "selector_rows_count": int(len(selector_df)),
        "summary": summary_df.to_dict(orient="records"),
        "segments": segments_df.to_dict(orient="records"),
        "window_reports": window_reports,
    }

    args.rows_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.segments_csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(args.rows_csv_out, index=False)
    summary_df.to_csv(args.summary_csv_out, index=False)
    segments_df.to_csv(args.segments_csv_out, index=False)
    args.summary_json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("REBOUND DIAGNOSTICS VALIDATION")
    print("=" * 92)
    print(f"Overall status:        {overall_status}")
    print(f"Modes present:         {', '.join(payload['modes_present'])}")
    for validation_mode, mode_part in summary_df.groupby("validation_mode", sort=False):
        print(f"\nMode: {validation_mode}")
        for variant in VARIANT_ORDER:
            row = mode_part.loc[mode_part["variant"] == variant]
            if row.empty:
                continue
            metrics = row.iloc[0]
            print(
                f"  {variant}: hit_rate={metrics['hit_rate']:.4f} "
                f"roi={metrics['roi']:+.3f} picks={int(metrics['total_picks'])} "
                f"trb_over={metrics['trb_over_hit_rate'] if pd.notna(metrics['trb_over_hit_rate']) else float('nan'):.4f} "
                f"removed_losses={int(metrics['removed_trb_over_losses'])} "
                f"removed_wins={int(metrics['removed_trb_over_wins'])} "
                f"window={metrics['validation_window_type']} "
                f"status={metrics['status_label']}"
            )
    print(f"Rows CSV:              {args.rows_csv_out}")
    print(f"Summary CSV:           {args.summary_csv_out}")
    print(f"Segments CSV:          {args.segments_csv_out}")
    print(f"Summary JSON:          {args.summary_json_out}")


if __name__ == "__main__":
    main()
