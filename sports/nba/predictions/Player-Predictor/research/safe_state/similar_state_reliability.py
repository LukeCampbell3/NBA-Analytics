from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _coalesce(frame: pd.DataFrame, columns: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            out = out.where(out.notna(), values)
    return out.fillna(default)


def _target(row: pd.Series) -> str:
    target = str(row.get("target", "") or "").upper().strip()
    if target:
        return target
    market_type = str(row.get("market_type", "") or "").upper()
    for candidate in ["PTS", "TRB", "AST", "PRA", "PR", "PA", "RA", "3PM"]:
        if candidate in market_type:
            return candidate
    return market_type


def _side(row: pd.Series) -> str:
    side = str(row.get("side", "") or row.get("direction", "") or "").upper().strip()
    if side in {"OVER", "UNDER"}:
        return side
    market_type = str(row.get("market_type", "") or "").upper()
    if market_type.endswith("_OVER"):
        return "OVER"
    if market_type.endswith("_UNDER"):
        return "UNDER"
    return side


def _date_series(frame: pd.DataFrame) -> pd.Series:
    for column in ["market_date", "game_date", "run_date", "date"]:
        if column in frame.columns:
            return pd.to_datetime(frame[column], errors="coerce")
    return pd.Series(pd.NaT, index=frame.index)


def _actual_stat(frame: pd.DataFrame, target: str) -> pd.Series:
    candidates = [
        "actual_stat",
        f"actual_{target.lower()}",
        f"{target}_actual",
        target,
    ]
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _win_label(frame: pd.DataFrame, side: str, target: str, line: float) -> pd.Series:
    for column in ["actual_result", "result", "settled_result"]:
        if column in frame.columns:
            raw = frame[column].fillna("").astype(str).str.lower()
            label = pd.Series(np.nan, index=frame.index, dtype="float64")
            label = label.mask(raw.str.contains("win|hit"), 1.0)
            label = label.mask(raw.str.contains("loss|miss"), 0.0)
            label = label.mask(raw.str.contains("push|void"), np.nan)
            if label.notna().any():
                return label
    actual = _actual_stat(frame, target)
    if not np.isfinite(line):
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    if side == "UNDER":
        return (actual < float(line)).astype(float).where(actual.notna(), np.nan)
    return (actual > float(line)).astype(float).where(actual.notna(), np.nan)


def _similar_pool(candidate: pd.Series, history: pd.DataFrame, *, min_count: int, line_tolerance: float) -> tuple[pd.DataFrame, str]:
    if history.empty:
        return history.copy(), "missing_history"

    target = _target(candidate)
    player = str(candidate.get("player_name", "") or candidate.get("player", "") or "").strip().lower()
    archetype = str(candidate.get("player_archetype", "") or "").strip().lower()
    line = pd.to_numeric(pd.Series([candidate.get("line", candidate.get("market_line", np.nan))]), errors="coerce").iloc[0]
    candidate_date = pd.to_datetime(candidate.get("market_date", candidate.get("game_date", pd.NaT)), errors="coerce")

    hist = history.copy()
    hist_target = hist.apply(_target, axis=1).astype(str)
    pool = hist.loc[hist_target.eq(target)].copy()
    hist_dates = _date_series(pool)
    if pd.notna(candidate_date) and hist_dates.notna().any():
        pool = pool.loc[hist_dates < candidate_date].copy()

    if "candidate_id" in pool.columns and "candidate_id" in candidate.index:
        pool = pool.loc[pool["candidate_id"].astype(str) != str(candidate.get("candidate_id", ""))].copy()
    if "game_id" in pool.columns and "game_id" in candidate.index:
        pool = pool.loc[pool["game_id"].astype(str) != str(candidate.get("game_id", ""))].copy()

    same_player = pd.Series(False, index=pool.index)
    if player:
        names = _text(pool, "player_name").str.lower().where(_text(pool, "player_name").str.strip().ne(""), _text(pool, "player").str.lower())
        same_player = names.eq(player)
    same_player_pool = pool.loc[same_player].copy()

    if np.isfinite(line):
        candidate_line = float(line)
        tolerance = max(float(line_tolerance), abs(candidate_line) * 0.15)
        line_values = _coalesce(same_player_pool, ["line", "market_line"], default=np.nan)
        same_player_pool = same_player_pool.loc[(line_values - candidate_line).abs().le(tolerance) | line_values.isna()].copy()

    if len(same_player_pool) >= min_count:
        return same_player_pool, "same_player_target_line"

    fallback = pool.copy()
    if archetype and "player_archetype" in fallback.columns:
        archetype_mask = _text(fallback, "player_archetype").str.lower().eq(archetype)
        archetype_pool = fallback.loc[archetype_mask].copy()
        if len(archetype_pool) >= min_count:
            fallback = archetype_pool
    if np.isfinite(line):
        candidate_line = float(line)
        tolerance = max(float(line_tolerance) * 1.5, abs(candidate_line) * 0.25)
        line_values = _coalesce(fallback, ["line", "market_line"], default=np.nan)
        narrowed = fallback.loc[(line_values - candidate_line).abs().le(tolerance) | line_values.isna()].copy()
        if len(narrowed) >= min_count:
            fallback = narrowed
    return fallback, "fallback_target_archetype_line"


def _summarize_examples(pool: pd.DataFrame, *, target: str, max_examples: int = 5) -> str:
    columns = [c for c in ["game_date", "market_date", "player", "player_name", "target", "side", "line", "actual_stat", "actual_result", "result"] if c in pool.columns]
    if not columns:
        return "[]"
    examples = []
    for _, row in pool.head(max_examples).iterrows():
        payload: dict[str, Any] = {}
        for column in columns:
            value = row.get(column)
            if pd.isna(value):
                continue
            payload[column] = value.item() if hasattr(value, "item") else value
        if "actual_stat" not in payload:
            actual = pd.to_numeric(pd.Series([row.get(target, np.nan)]), errors="coerce").iloc[0]
            if pd.notna(actual):
                payload["actual_stat"] = float(actual)
        examples.append(payload)
    return json.dumps(examples, default=str)


def annotate_similar_state_reliability(
    candidates: pd.DataFrame,
    historical_rows: pd.DataFrame | None = None,
    *,
    min_count: int = 5,
    line_tolerance: float = 2.5,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    out = candidates.copy()
    history = pd.DataFrame() if historical_rows is None else historical_rows.copy()

    counts: list[int] = []
    win_rates: list[float] = []
    avg_residuals: list[float] = []
    median_abs_errors: list[float] = []
    p75_abs_errors: list[float] = []
    p90_abs_errors: list[float] = []
    interval_failure_rates: list[float] = []
    directional_failure_rates: list[float] = []
    iqrs: list[float] = []
    tightness_scores: list[float] = []
    tiers: list[str] = []
    warnings: list[str] = []
    examples_json: list[str] = []

    for _, candidate in out.iterrows():
        target = _target(candidate)
        side = _side(candidate)
        line = pd.to_numeric(pd.Series([candidate.get("line", candidate.get("market_line", np.nan))]), errors="coerce").iloc[0]
        pool, source = _similar_pool(candidate, history, min_count=min_count, line_tolerance=line_tolerance)
        pool_count = int(len(pool))
        actual = _actual_stat(pool, target)
        line_values = _coalesce(pool, ["line", "market_line"], default=line)
        residual = actual - line_values
        abs_error = residual.abs()
        if not np.isfinite(line):
            residual = pd.Series(np.nan, index=pool.index)
            abs_error = pd.Series(np.nan, index=pool.index)

        wins = _win_label(pool, side, target, float(line) if np.isfinite(line) else np.nan)
        win_rate = float(wins.mean()) if wins.notna().any() else np.nan
        avg_residual = float(residual.mean()) if residual.notna().any() else np.nan
        median_abs = float(abs_error.median()) if abs_error.notna().any() else np.nan
        p75_abs = float(abs_error.quantile(0.75)) if abs_error.notna().any() else np.nan
        p90_abs = float(abs_error.quantile(0.90)) if abs_error.notna().any() else np.nan
        iqr = float(actual.quantile(0.75) - actual.quantile(0.25)) if actual.notna().sum() >= 2 else np.nan
        directional_failure = float((wins == 0.0).mean()) if wins.notna().any() else np.nan
        interval_failure = float(abs_error.gt(max(2.0, abs(float(line)) * 0.20)).mean()) if abs_error.notna().any() and np.isfinite(line) else np.nan

        scale = max(2.0, abs(float(line)) * 0.30) if np.isfinite(line) else 6.0
        raw_tightness = 1.0 - ((p75_abs if np.isfinite(p75_abs) else scale) / scale)
        sample_weight = min(1.0, pool_count / max(float(min_count), 1.0))
        tightness = float(np.clip(raw_tightness, 0.0, 1.0) * (0.50 + 0.50 * sample_weight))

        if pool_count < min_count:
            tier = "INSUFFICIENT_SAMPLE"
            warning = f"only_{pool_count}_similar_states"
        elif tightness >= 0.75:
            tier = "TIGHT"
            warning = source
        elif tightness >= 0.55:
            tier = "ACCEPTABLE"
            warning = source
        else:
            tier = "SCATTERED"
            warning = f"{source};outcomes_scattered"

        counts.append(pool_count)
        win_rates.append(win_rate)
        avg_residuals.append(avg_residual)
        median_abs_errors.append(median_abs)
        p75_abs_errors.append(p75_abs)
        p90_abs_errors.append(p90_abs)
        interval_failure_rates.append(interval_failure)
        directional_failure_rates.append(directional_failure)
        iqrs.append(iqr)
        tightness_scores.append(tightness)
        tiers.append(tier)
        warnings.append(warning)
        examples_json.append(_summarize_examples(pool, target=target))

    out["similar_state_count"] = counts
    out["similar_state_win_rate"] = win_rates
    out["similar_state_avg_residual"] = avg_residuals
    out["similar_state_median_abs_error"] = median_abs_errors
    out["similar_state_p75_abs_error"] = p75_abs_errors
    out["similar_state_p90_abs_error"] = p90_abs_errors
    out["similar_state_interval_failure_rate"] = interval_failure_rates
    out["similar_state_directional_failure_rate"] = directional_failure_rates
    out["similar_state_outcome_iqr"] = iqrs
    out["similar_state_tightness_score"] = tightness_scores
    out["similar_state_reliability_tier"] = tiers
    out["similar_state_warning"] = warnings
    out["comparable_state_examples_json"] = examples_json
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate comparable-state reliability for candidate rows.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--historical-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--min-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    history = pd.read_csv(args.historical_csv) if args.historical_csv and args.historical_csv.exists() else pd.DataFrame()
    annotated = annotate_similar_state_reliability(candidates, history, min_count=int(args.min_count))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_csv, index=False)
    if args.summary_json:
        payload = {
            "rows": int(len(annotated)),
            "similar_state_tier_counts": annotated["similar_state_reliability_tier"].value_counts(dropna=False).to_dict(),
            "shadow_only": True,
            "production_behavior_changed": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
