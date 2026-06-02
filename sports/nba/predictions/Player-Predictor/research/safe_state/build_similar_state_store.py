from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.backfill_distribution_quantiles import _line, _line_zone, _target
from research.safe_state.backfill_minutes_state import _candidate_date, _candidate_player, _candidate_player_id, _load_player_logs


MIN_SAMPLE = 5
LOOKBACK_GAMES = 40


def _side(row: pd.Series) -> str:
    side = str(row.get("side", "") or row.get("direction", "") or "").upper().strip()
    if side in {"OVER", "UNDER"}:
        return side
    market_type = str(row.get("market_type", "") or "").upper()
    if market_type.endswith("_UNDER"):
        return "UNDER"
    return "OVER"


def _result(actual: pd.Series, line: float, side: str) -> pd.Series:
    if not np.isfinite(line):
        return pd.Series(np.nan, index=actual.index, dtype="float64")
    if side == "UNDER":
        return actual.lt(line).astype(float).where(actual.notna(), np.nan)
    return actual.gt(line).astype(float).where(actual.notna(), np.nan)


def _score_tightness(abs_residual: pd.Series, line: float, count: int) -> float:
    if abs_residual.dropna().empty:
        return 0.0
    p75 = float(abs_residual.dropna().quantile(0.75))
    scale = max(2.5, abs(float(line)) * 0.30) if np.isfinite(line) else 6.0
    sample_weight = min(1.0, count / float(MIN_SAMPLE))
    return float(np.clip(1.0 - (p75 / scale), 0.0, 1.0) * (0.50 + 0.50 * sample_weight))


def _summarize_pool(pool: pd.DataFrame, *, target: str, line: float, side: str) -> dict[str, Any]:
    actual = pd.to_numeric(pool.get(target, pd.Series(np.nan, index=pool.index)), errors="coerce")
    residual = actual - float(line) if np.isfinite(line) else pd.Series(np.nan, index=pool.index)
    abs_residual = residual.abs()
    wins = _result(actual, line, side)
    count = int(actual.notna().sum())
    tightness = _score_tightness(abs_residual, line, count)
    if count < MIN_SAMPLE:
        tier = "INSUFFICIENT_SAMPLE"
        gap = "FORECASTABILITY_GAP_SIMILAR_STATE_SAMPLE"
        fix = "NEEDS_MORE_SAMPLE"
        warning = f"similar_state_count={count}"
    elif tightness >= 0.75:
        tier = "TIGHT"
        gap = ""
        fix = "FIXABLE_WITH_EXISTING_LOGS"
        warning = "similar_states_tight"
    elif tightness >= 0.55:
        tier = "ACCEPTABLE"
        gap = ""
        fix = "FIXABLE_WITH_EXISTING_LOGS"
        warning = "similar_states_acceptable"
    else:
        tier = "SCATTERED"
        gap = "FORECASTABILITY_GAP_SIMILAR_STATE_SCATTER"
        fix = "TRUE_UNSTABLE_STATE"
        warning = "similar_states_scattered"
    return {
        "similar_state_count": count,
        "similar_state_win_rate": float(wins.mean()) if wins.notna().any() else np.nan,
        "similar_state_avg_residual": float(residual.mean()) if residual.notna().any() else np.nan,
        "similar_state_median_abs_error": float(abs_residual.median()) if abs_residual.notna().any() else np.nan,
        "similar_state_p75_abs_error": float(abs_residual.quantile(0.75)) if abs_residual.notna().any() else np.nan,
        "similar_state_p90_abs_error": float(abs_residual.quantile(0.90)) if abs_residual.notna().any() else np.nan,
        "similar_state_outcome_iqr": float(actual.quantile(0.75) - actual.quantile(0.25)) if actual.notna().sum() >= 2 else np.nan,
        "similar_state_tightness_score": tightness,
        "similar_state_reliability_tier": tier,
        "similar_state_gap_type": gap,
        "similar_state_gap_reason": warning,
        "similar_state_gap_fixability": fix,
    }


def build_similar_state_features(
    candidates: pd.DataFrame,
    *,
    data_proc_dir: Path | None,
    lookback_games: int = LOOKBACK_GAMES,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        player = _candidate_player(row)
        player_id = _candidate_player_id(row)
        target = _target(row)
        side = _side(row)
        market_date = _candidate_date(row)
        line = _line(row)
        cache_key = player_id or player
        if cache_key not in cache:
            cache[cache_key] = _load_player_logs(data_proc_dir, player, player_id=player_id)
        logs = cache[cache_key]
        if logs.empty or target not in logs.columns or pd.isna(market_date):
            rows.append(_empty_summary("missing_similar_state_history"))
            continue
        prior = logs.loc[logs["Date"] < market_date].tail(int(lookback_games)).copy()
        if prior.empty:
            rows.append(_empty_summary("no_prior_games_before_market_date"))
            continue
        # Same-player + same-target is the safest first store. Line-zone/minutes narrowing can be added
        # once those fields are fully populated across historical candidates.
        summary = _summarize_pool(prior, target=target, line=line, side=side)
        summary["similar_state_source"] = "same_player_target_pre_event_logs"
        rows.append(summary)
    features = pd.DataFrame(rows, index=out.index)
    for col in features.columns:
        out[col] = features[col]
    return out


def _empty_summary(reason: str) -> dict[str, Any]:
    return {
        "similar_state_count": 0,
        "similar_state_win_rate": np.nan,
        "similar_state_avg_residual": np.nan,
        "similar_state_median_abs_error": np.nan,
        "similar_state_p75_abs_error": np.nan,
        "similar_state_p90_abs_error": np.nan,
        "similar_state_outcome_iqr": np.nan,
        "similar_state_tightness_score": 0.0,
        "similar_state_reliability_tier": "INSUFFICIENT_SAMPLE",
        "similar_state_gap_type": "FORECASTABILITY_GAP_SIMILAR_STATE_SAMPLE",
        "similar_state_gap_reason": reason,
        "similar_state_gap_fixability": "NEEDS_MORE_SAMPLE",
        "similar_state_source": "missing",
    }


def build_similar_state_store(candidates: pd.DataFrame, *, data_proc_dir: Path | None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    for _, row in candidates.iterrows():
        player = _candidate_player(row)
        player_id = _candidate_player_id(row)
        target = _target(row)
        side = _side(row)
        market_date = _candidate_date(row)
        line = _line(row)
        cache_key = player_id or player
        if cache_key not in cache:
            cache[cache_key] = _load_player_logs(data_proc_dir, player, player_id=player_id)
        logs = cache[cache_key]
        if logs.empty or target not in logs.columns or pd.isna(market_date):
            continue
        prior = logs.loc[logs["Date"] < market_date].tail(LOOKBACK_GAMES).copy()
        for _, hist in prior.iterrows():
            actual = pd.to_numeric(pd.Series([hist.get(target)]), errors="coerce").iloc[0]
            residual = actual - line if np.isfinite(line) and pd.notna(actual) else np.nan
            side_win = np.nan
            if pd.notna(actual) and np.isfinite(line):
                side_win = float(actual < line) if side == "UNDER" else float(actual > line)
            q25 = pd.to_numeric(pd.Series([row.get("target_q25_recent", np.nan)]), errors="coerce").iloc[0]
            q50 = pd.to_numeric(pd.Series([row.get("target_q50_recent", np.nan)]), errors="coerce").iloc[0]
            q75 = pd.to_numeric(pd.Series([row.get("target_q75_recent", np.nan)]), errors="coerce").iloc[0]
            q90 = pd.to_numeric(pd.Series([row.get("target_q90_recent", np.nan)]), errors="coerce").iloc[0]
            zone, _ = _line_zone(line, q25, q50, q75, q90)
            records.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "player": row.get("player", row.get("player_name", "")),
                    "player_id": row.get("player_id", row.get("Player_ID", "")),
                    "team": row.get("team", ""),
                    "opponent": hist.get("Opponent", row.get("opponent", "")),
                    "game_date": hist.get("Date"),
                    "target": target,
                    "side": side,
                    "line": line,
                    "line_zone": zone,
                    "minutes_band_low": row.get("expected_minutes_band_low", np.nan),
                    "minutes_band_high": row.get("expected_minutes_band_high", np.nan),
                    "recent_target_median": row.get("target_recent_median", np.nan),
                    "recent_target_std": row.get("target_recent_std", np.nan),
                    "actual_target_value": actual,
                    "side_result": side_win,
                    "residual_vs_line": residual,
                    "abs_residual": abs(residual) if pd.notna(residual) else np.nan,
                    "interval_failure_flag": bool(abs(residual) > max(2.5, abs(line) * 0.30)) if pd.notna(residual) and np.isfinite(line) else False,
                    "directional_failure_flag": bool(side_win == 0.0) if pd.notna(side_win) else False,
                }
            )
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safe-state similar-state store and candidate features.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--store-csv", type=Path)
    parser.add_argument("--data-proc-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Data-Proc")
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    out = build_similar_state_features(candidates, data_proc_dir=args.data_proc_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    store = build_similar_state_store(out, data_proc_dir=args.data_proc_dir)
    if args.store_csv:
        args.store_csv.parent.mkdir(parents=True, exist_ok=True)
        store.to_csv(args.store_csv, index=False)
    if args.summary_json:
        payload = {
            "rows": int(len(out)),
            "store_rows": int(len(store)),
            "similar_state_tier_counts": out["similar_state_reliability_tier"].fillna("").astype(str).value_counts().to_dict(),
            "similar_state_gap_counts": out["similar_state_gap_type"].fillna("").astype(str).value_counts().to_dict(),
            "production_behavior_changed": False,
            "promotion_claim": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
