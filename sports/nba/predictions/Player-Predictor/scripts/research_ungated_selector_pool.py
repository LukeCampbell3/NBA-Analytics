#!/usr/bin/env python3
"""
Research which ungated selector-pool metadata best separates NBA wins from losses.

This script builds a labeled history from historical selector CSVs under
``model/analysis/daily_runs`` and joins each row to:
1. actual outcome from the processed player history file
2. whether the row made the published final board for that run date

The outputs are meant to support two production improvements:
- vetoing historically weak accepted picks without collapsing coverage
- appending strong near-miss rows from just outside the board cutoff
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_RUNS_ROOT = REPO_ROOT / "model" / "analysis" / "daily_runs"
OUTPUT_ROOT = REPO_ROOT / "model" / "analysis" / "selector_pool_research"
PAYOUT_MINUS_110 = 100.0 / 110.0

NUMERIC_FEATURE_CANDIDATES = [
    "pool_universe_rank",
    "selected_rank",
    "market_line",
    "edge",
    "abs_edge",
    "gap_percentile",
    "raw_gap_percentile",
    "expected_win_rate",
    "selector_expected_win_rate",
    "raw_expected_win_rate",
    "p_selector",
    "p_calibrated",
    "board_play_win_prob",
    "selected_board_prob_raw",
    "ev",
    "ev_adjusted",
    "final_confidence",
    "confidence_score",
    "market_books",
    "market_books_norm",
    "history_rows",
    "recency_factor",
    "belief_uncertainty",
    "belief_uncertainty_normalized",
    "belief_confidence_factor",
    "posterior_alpha",
    "posterior_beta",
    "posterior_variance",
    "calibration_blend_weight",
    "risk_penalty",
    "uncertainty_sigma",
    "spike_probability",
    "volatility_score",
    "tail_imbalance",
    "line_chosen_direction_prob",
    "line_opposite_direction_prob",
    "line_conditional_prob_gap",
    "line_decision_support_rows",
    "line_decision_support_strength",
    "line_decision_sigma_pressure",
    "line_decision_instability_score",
    "line_decision_fragility_score",
    "line_decision_empirical_blend_weight",
    "script_compatibility",
    "recoverability_score",
    "contradiction_score",
    "noise_score",
    "conditional_support",
    "conditional_score",
    "snapshot_selector_rows",
    "snapshot_board_size_requested",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "target",
    "direction",
    "recommendation",
    "raw_recommendation",
    "decision_tier",
    "line_decision_source",
    "calibration_source",
    "policy_calibration_source",
    "selected_board_calibration_source",
    "weak_bucket",
    "script_cluster_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research metadata signals inside the ungated selector pool.")
    parser.add_argument("--daily-runs-dir", type=Path, default=DAILY_RUNS_ROOT, help="Historical NBA daily_runs root.")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive YYYY-MM-DD start date.")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive YYYY-MM-DD end date.")
    parser.add_argument("--board-size", type=int, default=12, help="Reference published board size used for near-miss bands.")
    parser.add_argument(
        "--near-miss-window",
        type=int,
        default=12,
        help="How many ranks beyond the board cutoff count as append-style near misses.",
    )
    parser.add_argument(
        "--min-resolved-rows",
        type=int,
        default=24,
        help="Minimum resolved rows required before a feature bucket/cohort is reported.",
    )
    parser.add_argument(
        "--history-out-csv",
        type=Path,
        default=OUTPUT_ROOT / "ungated_selector_history.csv",
        help="Labeled historical selector-pool rows.",
    )
    parser.add_argument(
        "--feature-out-csv",
        type=Path,
        default=OUTPUT_ROOT / "ungated_selector_feature_report.csv",
        help="Bucket/cohort-level feature report.",
    )
    parser.add_argument(
        "--summary-out-json",
        type=Path,
        default=OUTPUT_ROOT / "ungated_selector_summary.json",
        help="High-level research summary JSON.",
    )
    return parser.parse_args()


def normalize_player_name(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_pick_key(
    frame: pd.DataFrame,
    *,
    player_col: str = "market_player_raw",
    fallback_player_col: str = "player",
    market_date_col: str = "market_date",
    target_col: str = "target",
    direction_col: str = "direction",
    line_col: str = "market_line",
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="object")
    player_a = frame.get(player_col, pd.Series("", index=frame.index)).map(normalize_player_name)
    player_b = frame.get(fallback_player_col, pd.Series("", index=frame.index)).map(normalize_player_name)
    player = player_a.where(player_a.ne(""), player_b)
    market_date = pd.to_datetime(frame.get(market_date_col), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    target = frame.get(target_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    direction = frame.get(direction_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    line = pd.to_numeric(frame.get(line_col), errors="coerce").round(4)
    line_text = line.map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return (market_date + "|" + player + "|" + target + "|" + direction + "|" + line_text).astype("object")


def resolve_processed_csv(raw_path: str) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    fallback = REPO_ROOT / "Data-Proc" / candidate.parent.name / candidate.name
    if fallback.exists():
        return fallback
    return None


def load_actual_lookup():
    cache: dict[str, pd.DataFrame] = {}

    def resolve(csv_path: str, market_date: str, target: str) -> float | None:
        resolved_path = resolve_processed_csv(csv_path)
        cache_key = str(resolved_path) if resolved_path is not None else str(csv_path)
        if cache_key not in cache:
            if resolved_path is None or not resolved_path.exists():
                cache[cache_key] = pd.DataFrame(columns=["Date", "PTS", "TRB", "AST"])
            else:
                try:
                    frame = pd.read_csv(resolved_path, usecols=["Date", "PTS", "TRB", "AST"])
                except Exception:
                    frame = pd.DataFrame(columns=["Date", "PTS", "TRB", "AST"])
                frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                cache[cache_key] = frame

        frame = cache[cache_key]
        target_key = str(target).upper().strip()
        if frame.empty or target_key not in {"PTS", "TRB", "AST"}:
            return None
        rows = frame.loc[frame["Date"] == str(market_date)]
        if rows.empty:
            return None
        actual = pd.to_numeric(rows.iloc[-1][target_key], errors="coerce")
        if pd.isna(actual):
            return None
        return float(actual)

    return resolve


def classify_result(direction: str, line: float | None, actual: float | None) -> str:
    if actual is None or line is None:
        return "missing"
    token = str(direction).upper().strip()
    if token == "OVER":
        if actual > line:
            return "win"
        if actual < line:
            return "loss"
        return "push"
    if token == "UNDER":
        if actual < line:
            return "win"
        if actual > line:
            return "loss"
        return "push"
    return "missing"


def expected_utility_from_result(result: str) -> float:
    token = str(result).strip().lower()
    if token == "win":
        return float(PAYOUT_MINUS_110)
    if token == "loss":
        return -1.0
    if token == "push":
        return 0.0
    return float("nan")


def summarize_slice(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "missing": 0,
            "resolved_hit_rate": None,
            "ev_per_resolved": None,
        }
    result = frame.get("result", pd.Series("missing", index=frame.index)).astype(str).str.lower().str.strip()
    wins = int((result == "win").sum())
    losses = int((result == "loss").sum())
    pushes = int((result == "push").sum())
    resolved = wins + losses
    missing = int(len(frame) - resolved - pushes)
    utility = pd.to_numeric(frame.get("utility_units"), errors="coerce")
    ev_per_resolved = None
    if resolved > 0 and utility.notna().any():
        ev_per_resolved = float(utility.loc[result.isin(["win", "loss"])].sum() / resolved)
    return {
        "rows": int(len(frame)),
        "resolved": int(resolved),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "missing": int(missing),
        "resolved_hit_rate": float(wins / resolved) if resolved > 0 else None,
        "ev_per_resolved": ev_per_resolved,
    }


def band_name(rank: int, board_size: int, near_miss_window: int) -> str:
    if rank <= int(board_size):
        return f"r1_{board_size}"
    if rank <= int(board_size + near_miss_window):
        return f"r{board_size + 1}_{board_size + near_miss_window}"
    if rank <= int(board_size + (2 * near_miss_window)):
        upper = board_size + (2 * near_miss_window)
        return f"r{board_size + near_miss_window + 1}_{upper}"
    return f"r{board_size + (2 * near_miss_window) + 1}_plus"


def iter_run_dirs(root: Path, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None) -> list[Path]:
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.isdigit() or len(child.name) != 8:
            continue
        run_date = pd.to_datetime(child.name, format="%Y%m%d", errors="coerce")
        if pd.isna(run_date):
            continue
        if start_date is not None and run_date < start_date:
            continue
        if end_date is not None and run_date > end_date:
            continue
        out.append(child)
    return out


def collect_history(
    daily_runs_dir: Path,
    *,
    board_size: int,
    near_miss_window: int,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup_actual = load_actual_lookup()
    rows: list[pd.DataFrame] = []
    diagnostics = {
        "run_dirs_seen": 0,
        "run_dirs_loaded": 0,
        "selector_csv_missing": 0,
        "final_csv_missing": 0,
    }

    for run_dir in iter_run_dirs(daily_runs_dir, start_date, end_date):
        diagnostics["run_dirs_seen"] += 1
        selector_csv = run_dir / f"upcoming_market_play_selector_{run_dir.name}.csv"
        if not selector_csv.exists():
            diagnostics["selector_csv_missing"] += 1
            continue

        try:
            selector = pd.read_csv(selector_csv)
        except Exception:
            continue
        if selector.empty:
            continue

        final_csv = run_dir / f"final_market_plays_{run_dir.name}.csv"
        final_frame = pd.DataFrame()
        if final_csv.exists():
            try:
                final_frame = pd.read_csv(final_csv)
            except Exception:
                final_frame = pd.DataFrame()
        else:
            diagnostics["final_csv_missing"] += 1

        selector = selector.copy().reset_index(drop=True)
        selector["pool_universe_rank"] = np.arange(1, len(selector) + 1, dtype=int)
        selector["run_stamp"] = run_dir.name
        selector["run_date"] = pd.to_datetime(run_dir.name, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")
        selector["pick_key"] = build_pick_key(selector)
        selector["pool_rank_band"] = selector["pool_universe_rank"].map(
            lambda value: band_name(int(value), board_size=board_size, near_miss_window=near_miss_window)
        )

        if not final_frame.empty:
            final_frame = final_frame.copy()
            final_frame["pick_key"] = build_pick_key(final_frame)
            selected_keys = set(final_frame["pick_key"].astype(str))
        else:
            selected_keys = set()
        selector["selected_for_board"] = selector["pick_key"].astype(str).isin(selected_keys)
        selector["near_miss_candidate"] = (
            (~selector["selected_for_board"])
            & selector["pool_universe_rank"].between(board_size + 1, board_size + near_miss_window, inclusive="both")
        )

        dates = pd.to_datetime(selector.get("market_date"), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        targets = selector.get("target", pd.Series("", index=selector.index)).fillna("").astype(str).str.upper().str.strip()
        directions = selector.get("direction", pd.Series("", index=selector.index)).fillna("").astype(str).str.upper().str.strip()
        line_vals = pd.to_numeric(selector.get("market_line"), errors="coerce")
        csv_paths = selector.get("csv", pd.Series("", index=selector.index)).fillna("").astype(str)

        actual_values: list[float] = []
        results: list[str] = []
        utility_units: list[float] = []
        for csv_path, market_date, target, direction, line in zip(csv_paths, dates, targets, directions, line_vals):
            actual = lookup_actual(str(csv_path), str(market_date), str(target))
            line_value = None if pd.isna(line) else float(line)
            result = classify_result(str(direction), line_value, actual)
            actual_values.append(float(actual) if actual is not None else float("nan"))
            results.append(str(result))
            utility_units.append(float(expected_utility_from_result(result)))

        selector["actual_value"] = pd.Series(actual_values, index=selector.index, dtype="float64")
        selector["result"] = pd.Series(results, index=selector.index, dtype="object")
        selector["utility_units"] = pd.Series(utility_units, index=selector.index, dtype="float64")
        selector["resolved"] = selector["result"].isin(["win", "loss"]).astype(bool)

        rows.append(selector)
        diagnostics["run_dirs_loaded"] += 1

    history = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return history, diagnostics


def feature_bucket_rows(
    frame: pd.DataFrame,
    *,
    group_name: str,
    min_resolved_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    spreads: list[dict[str, Any]] = []
    resolved = frame.loc[frame["result"].isin(["win", "loss"])].copy()
    if resolved.empty:
        return records, spreads
    resolved["win_label"] = (resolved["result"] == "win").astype(int)
    overall_hit_rate = float(resolved["win_label"].mean())

    for feature in NUMERIC_FEATURE_CANDIDATES:
        if feature not in resolved.columns:
            continue
        values = pd.to_numeric(resolved[feature], errors="coerce")
        valid = resolved.loc[values.notna()].copy()
        valid[feature] = pd.to_numeric(valid[feature], errors="coerce")
        if len(valid) < int(min_resolved_rows):
            continue
        unique_count = int(valid[feature].nunique(dropna=True))
        if unique_count < 4:
            continue
        try:
            bucket = pd.qcut(valid[feature], q=4, duplicates="drop")
        except Exception:
            continue
        valid["_bucket"] = bucket.astype(str)
        grouped = (
            valid.groupby("_bucket", observed=False)
            .agg(
                rows=("result", "size"),
                wins=("win_label", "sum"),
                losses=("win_label", lambda s: int(len(s) - s.sum())),
                feature_min=(feature, "min"),
                feature_max=(feature, "max"),
                feature_mean=(feature, "mean"),
            )
            .reset_index()
        )
        if len(grouped) < 2:
            continue
        grouped["resolved"] = grouped["rows"].astype(int)
        grouped["hit_rate"] = grouped["wins"] / grouped["resolved"]
        grouped["delta_vs_group_hit_rate_pp"] = (grouped["hit_rate"] - overall_hit_rate) * 100.0

        top_hit = float(grouped["hit_rate"].max())
        bottom_hit = float(grouped["hit_rate"].min())
        best_bucket = grouped.loc[grouped["hit_rate"].idxmax(), "_bucket"]
        worst_bucket = grouped.loc[grouped["hit_rate"].idxmin(), "_bucket"]
        best_mean = float(grouped.loc[grouped["hit_rate"].idxmax(), "feature_mean"])
        worst_mean = float(grouped.loc[grouped["hit_rate"].idxmin(), "feature_mean"])
        preferred_direction = "higher_better" if best_mean >= worst_mean else "lower_better"

        spreads.append(
            {
                "analysis_group": group_name,
                "feature_type": "numeric",
                "feature": feature,
                "resolved_rows": int(grouped["resolved"].sum()),
                "bucket_count": int(len(grouped)),
                "overall_hit_rate": overall_hit_rate,
                "best_bucket": str(best_bucket),
                "worst_bucket": str(worst_bucket),
                "best_hit_rate": top_hit,
                "worst_hit_rate": bottom_hit,
                "spread_pp": float((top_hit - bottom_hit) * 100.0),
                "preferred_direction": preferred_direction,
            }
        )

        for _, row in grouped.iterrows():
            records.append(
                {
                    "analysis_group": group_name,
                    "feature_type": "numeric",
                    "feature": feature,
                    "bucket_or_level": str(row["_bucket"]),
                    "rows": int(row["rows"]),
                    "resolved": int(row["resolved"]),
                    "wins": int(row["wins"]),
                    "losses": int(row["losses"]),
                    "hit_rate": float(row["hit_rate"]),
                    "delta_vs_group_hit_rate_pp": float(row["delta_vs_group_hit_rate_pp"]),
                    "feature_min": float(row["feature_min"]),
                    "feature_max": float(row["feature_max"]),
                    "feature_mean": float(row["feature_mean"]),
                }
            )

    for feature in CATEGORICAL_FEATURE_CANDIDATES:
        if feature not in resolved.columns:
            continue
        values = resolved[feature].fillna("").astype(str).str.strip()
        valid = resolved.loc[values.ne("")].copy()
        if valid.empty:
            continue
        valid[feature] = values.loc[valid.index]
        grouped = (
            valid.groupby(feature, dropna=False)
            .agg(
                rows=("result", "size"),
                wins=("win_label", "sum"),
            )
            .reset_index()
            .rename(columns={feature: "bucket_or_level"})
        )
        grouped["resolved"] = grouped["rows"].astype(int)
        grouped["losses"] = grouped["resolved"] - grouped["wins"]
        grouped = grouped.loc[grouped["resolved"] >= int(min_resolved_rows)].copy()
        if grouped.empty:
            continue
        grouped["hit_rate"] = grouped["wins"] / grouped["resolved"]
        grouped["delta_vs_group_hit_rate_pp"] = (grouped["hit_rate"] - overall_hit_rate) * 100.0
        best_hit = float(grouped["hit_rate"].max())
        worst_hit = float(grouped["hit_rate"].min())
        best_level = grouped.loc[grouped["hit_rate"].idxmax(), "bucket_or_level"]
        worst_level = grouped.loc[grouped["hit_rate"].idxmin(), "bucket_or_level"]
        spreads.append(
            {
                "analysis_group": group_name,
                "feature_type": "categorical",
                "feature": feature,
                "resolved_rows": int(grouped["resolved"].sum()),
                "bucket_count": int(len(grouped)),
                "overall_hit_rate": overall_hit_rate,
                "best_bucket": str(best_level),
                "worst_bucket": str(worst_level),
                "best_hit_rate": best_hit,
                "worst_hit_rate": worst_hit,
                "spread_pp": float((best_hit - worst_hit) * 100.0),
                "preferred_direction": "level_specific",
            }
        )
        for _, row in grouped.iterrows():
            records.append(
                {
                    "analysis_group": group_name,
                    "feature_type": "categorical",
                    "feature": feature,
                    "bucket_or_level": str(row["bucket_or_level"]),
                    "rows": int(row["rows"]),
                    "resolved": int(row["resolved"]),
                    "wins": int(row["wins"]),
                    "losses": int(row["losses"]),
                    "hit_rate": float(row["hit_rate"]),
                    "delta_vs_group_hit_rate_pp": float(row["delta_vs_group_hit_rate_pp"]),
                    "feature_min": np.nan,
                    "feature_max": np.nan,
                    "feature_mean": np.nan,
                }
            )

    return records, spreads


def build_summary(
    history: pd.DataFrame,
    *,
    board_size: int,
    near_miss_window: int,
    feature_spreads: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    if history.empty:
        return {
            "window": {"start_date": start_date, "end_date": end_date},
            "diagnostics": diagnostics,
            "overall": {},
        }

    history = history.copy()
    resolved = history.loc[history["result"].isin(["win", "loss"])].copy()
    rank_band_summary = {
        name: summarize_slice(part)
        for name, part in history.groupby("pool_rank_band", dropna=False)
    }

    selected = history.loc[history["selected_for_board"]].copy()
    near_miss = history.loc[history["near_miss_candidate"]].copy()
    tail = history.loc[~history["selected_for_board"] & ~history["near_miss_candidate"]].copy()

    selected_resolved = selected.loc[selected["result"].isin(["win", "loss"])].copy()
    near_miss_resolved = near_miss.loc[near_miss["result"].isin(["win", "loss"])].copy()
    tail_resolved = tail.loc[tail["result"].isin(["win", "loss"])].copy()

    append_opportunity = {
        "selected_board": summarize_slice(selected),
        "near_miss_pool": summarize_slice(near_miss),
        "tail_pool": summarize_slice(tail),
        "near_miss_vs_selected_hit_rate_pp": (
            (float(near_miss_resolved["result"].eq("win").mean()) - float(selected_resolved["result"].eq("win").mean())) * 100.0
            if not selected_resolved.empty and not near_miss_resolved.empty
            else None
        ),
        "near_miss_vs_tail_hit_rate_pp": (
            (float(near_miss_resolved["result"].eq("win").mean()) - float(tail_resolved["result"].eq("win").mean())) * 100.0
            if not near_miss_resolved.empty and not tail_resolved.empty
            else None
        ),
    }

    top_spreads = (
        pd.DataFrame(feature_spreads)
        .sort_values(["analysis_group", "spread_pp", "resolved_rows"], ascending=[True, False, False])
        .groupby("analysis_group", dropna=False)
        .head(8)
    )

    selected_hit_rate = float(selected_resolved["result"].eq("win").mean()) if not selected_resolved.empty else None
    near_miss_hit_rate = float(near_miss_resolved["result"].eq("win").mean()) if not near_miss_resolved.empty else None

    return {
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "board_size": int(board_size),
            "near_miss_window": int(near_miss_window),
            "run_dates": sorted(history["run_date"].dropna().astype(str).unique().tolist()),
            "run_date_count": int(history["run_date"].dropna().astype(str).nunique()),
        },
        "diagnostics": diagnostics,
        "overall": summarize_slice(history),
        "selected_board": summarize_slice(selected),
        "near_miss_pool": summarize_slice(near_miss),
        "tail_pool": summarize_slice(tail),
        "rank_bands": rank_band_summary,
        "append_opportunity": append_opportunity,
        "feature_spread_leaders": {
            group: top_spreads.loc[top_spreads["analysis_group"] == group].to_dict(orient="records")
            for group in sorted(top_spreads["analysis_group"].dropna().astype(str).unique().tolist())
        },
        "recommended_method": {
            "primary_problem": (
                "Use metadata-aware scoring on the ungated selector pool because the current selector probability "
                "alone is too compressed to separate true hits from misses."
            ),
            "veto_leg": (
                "Train a keep/drop model on historically settled selector rows to veto weak accepted picks using "
                "metadata such as line-decision probabilities, volatility, recency, script compatibility, and "
                "conditional recoverability."
            ),
            "append_leg": (
                "Train an append-only ranker on unselected near-miss rows ranked "
                f"{board_size + 1} through {board_size + near_miss_window} and only add rows whose metadata profile "
                "matches historically winning near-miss cohorts."
            ),
            "success_condition": {
                "selected_hit_rate": selected_hit_rate,
                "near_miss_hit_rate": near_miss_hit_rate,
                "near_miss_must_outperform_tail": True,
            },
        },
    }


def main() -> None:
    args = parse_args()
    start_date = pd.to_datetime(args.start_date, errors="coerce") if args.start_date else None
    end_date = pd.to_datetime(args.end_date, errors="coerce") if args.end_date else None

    history, diagnostics = collect_history(
        args.daily_runs_dir.resolve(),
        board_size=int(args.board_size),
        near_miss_window=int(args.near_miss_window),
        start_date=start_date,
        end_date=end_date,
    )
    if history.empty:
        raise RuntimeError(f"No selector history could be built from {args.daily_runs_dir}")

    feature_rows: list[dict[str, Any]] = []
    feature_spreads: list[dict[str, Any]] = []
    groups = {
        "full_pool": history,
        "selected_board": history.loc[history["selected_for_board"]].copy(),
        "near_miss_unselected": history.loc[history["near_miss_candidate"]].copy(),
    }
    for group_name, frame in groups.items():
        rows, spreads = feature_bucket_rows(
            frame,
            group_name=group_name,
            min_resolved_rows=int(args.min_resolved_rows),
        )
        feature_rows.extend(rows)
        feature_spreads.extend(spreads)

    feature_report = pd.DataFrame(feature_rows)
    summary = build_summary(
        history,
        board_size=int(args.board_size),
        near_miss_window=int(args.near_miss_window),
        feature_spreads=feature_spreads,
        diagnostics=diagnostics,
        start_date=str(args.start_date) if args.start_date else None,
        end_date=str(args.end_date) if args.end_date else None,
    )

    args.history_out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.feature_out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out_json.parent.mkdir(parents=True, exist_ok=True)

    history.to_csv(args.history_out_csv, index=False)
    feature_report.to_csv(args.feature_out_csv, index=False)
    args.summary_out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("UNGATED SELECTOR POOL RESEARCH COMPLETE")
    print(f"History rows:      {len(history)}")
    print(f"Resolved rows:     {int(history['resolved'].sum())}")
    print(f"Run dates:         {int(history['run_date'].astype(str).nunique())}")
    print(f"History CSV:       {args.history_out_csv}")
    print(f"Feature report:    {args.feature_out_csv}")
    print(f"Summary JSON:      {args.summary_out_json}")


if __name__ == "__main__":
    main()
