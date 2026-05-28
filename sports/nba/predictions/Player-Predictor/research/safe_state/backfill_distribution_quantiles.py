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

from research.safe_state.backfill_minutes_state import _candidate_date, _candidate_player, _candidate_player_id, _load_player_logs


DEFAULT_LOOKBACK = 12
MIN_SAMPLE = 4


def _target(row: pd.Series) -> str:
    target = str(row.get("target", "") or "").upper().strip()
    if target:
        return target
    market_type = str(row.get("market_type", "") or "").upper()
    for candidate in ["PTS", "TRB", "AST"]:
        if candidate in market_type:
            return candidate
    return target


def _line(row: pd.Series) -> float:
    return pd.to_numeric(pd.Series([row.get("line", row.get("market_line", np.nan))]), errors="coerce").iloc[0]


def _line_zone(line: float, q25: float, q50: float, q75: float, q90: float) -> tuple[str, float]:
    if not np.isfinite(line):
        return "UNKNOWN", np.nan
    if np.isfinite(q90) and line > q90:
        return "EXTREME_TAIL", 0.95
    if np.isfinite(q75) and line > q75:
        return "ABOVE_Q75", 0.80
    if np.isfinite(q25) and line < q25:
        return "BELOW_Q25", 0.20
    if np.isfinite(q50):
        return "NEAR_MEDIAN", 0.50
    return "UNKNOWN", np.nan


def _score_distribution(values: pd.Series, line: float) -> tuple[float, str, str, str]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < MIN_SAMPLE:
        return 0.0, "FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA", f"quantile_sample_count={len(values)}", "NEEDS_MORE_SAMPLE"
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    q90 = float(values.quantile(0.90))
    q10 = float(values.quantile(0.10))
    width = q75 - q25
    tail_width = q90 - q10
    scale = max(2.5, abs(float(line)) * 0.35) if np.isfinite(line) else 6.0
    score = float(np.clip(1.0 - (width / scale), 0.0, 1.0))
    if width > scale or tail_width > scale * 1.8:
        return score, "FORECASTABILITY_GAP_DISTRIBUTION_WIDTH", f"target_iqr={width:.2f};target_p10_p90={tail_width:.2f}", "TRUE_UNSTABLE_STATE"
    return score, "", "distribution_band_tight", "FIXABLE_WITH_EXISTING_LOGS"


def backfill_distribution_quantiles(
    candidates: pd.DataFrame,
    *,
    data_proc_dir: Path | None,
    lookback_games: int = DEFAULT_LOOKBACK,
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
        market_date = _candidate_date(row)
        line = _line(row)
        cache_key = player_id or player
        if cache_key not in cache:
            cache[cache_key] = _load_player_logs(data_proc_dir, player, player_id=player_id)
        logs = cache[cache_key]
        if logs.empty or target not in logs.columns or pd.isna(market_date):
            rows.append(_empty_row("missing_target_history", "FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA"))
            continue
        prior = logs.loc[logs["Date"] < market_date].tail(int(lookback_games)).copy()
        values = pd.to_numeric(prior[target], errors="coerce").dropna()
        score, gap_type, reason, fixability = _score_distribution(values, line)
        if values.empty:
            mean = median = std = q10 = q25 = q50 = q75 = q90 = np.nan
        else:
            mean = float(values.mean())
            median = float(values.median())
            std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            q10 = float(values.quantile(0.10))
            q25 = float(values.quantile(0.25))
            q50 = float(values.quantile(0.50))
            q75 = float(values.quantile(0.75))
            q90 = float(values.quantile(0.90))
        zone, percentile = _line_zone(line, q25, q50, q75, q90)
        rows.append(
            {
                "target_recent_mean": mean,
                "target_recent_median": median,
                "target_recent_std": std,
                "target_q10_recent": q10,
                "target_q25_recent": q25,
                "target_q50_recent": q50,
                "target_q75_recent": q75,
                "target_q90_recent": q90,
                "q10": q10,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": q90,
                "line_percentile_recent": percentile,
                "line_percentile": percentile,
                "line_zone": zone,
                "conservative_expected_stat": mean - 0.25 * std if pd.notna(mean) and pd.notna(std) else np.nan,
                "model_mean": row.get("model_mean", mean),
                "distribution_width": float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else np.nan,
                "distribution_stability_score": score,
                "quantile_source": "Data-Proc_pre_event_logs",
                "quantile_sample_count": int(len(values)),
                "quantile_warning": "" if not gap_type else reason,
                "distribution_gap_type": gap_type,
                "distribution_gap_reason": reason,
                "distribution_gap_fixability": fixability,
            }
        )
    backfill = pd.DataFrame(rows, index=out.index)
    for col in backfill.columns:
        out[col] = backfill[col]
    return out


def _empty_row(reason: str, gap_type: str) -> dict[str, Any]:
    return {
        "target_recent_mean": np.nan,
        "target_recent_median": np.nan,
        "target_recent_std": np.nan,
        "target_q10_recent": np.nan,
        "target_q25_recent": np.nan,
        "target_q50_recent": np.nan,
        "target_q75_recent": np.nan,
        "target_q90_recent": np.nan,
        "q10": np.nan,
        "q25": np.nan,
        "q50": np.nan,
        "q75": np.nan,
        "q90": np.nan,
        "line_percentile_recent": np.nan,
        "line_percentile": np.nan,
        "line_zone": "UNKNOWN",
        "conservative_expected_stat": np.nan,
        "model_mean": np.nan,
        "distribution_width": np.nan,
        "distribution_stability_score": 0.0,
        "quantile_source": "missing_target_history",
        "quantile_sample_count": 0,
        "quantile_warning": reason,
        "distribution_gap_type": gap_type,
        "distribution_gap_reason": reason,
        "distribution_gap_fixability": "NEEDS_MORE_SAMPLE",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill target distribution quantiles for safe-state research.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--data-proc-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Data-Proc")
    parser.add_argument("--lookback-games", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    out = backfill_distribution_quantiles(candidates, data_proc_dir=args.data_proc_dir, lookback_games=int(args.lookback_games))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    if args.summary_json:
        payload = {
            "rows": int(len(out)),
            "distribution_gap_counts": out["distribution_gap_type"].fillna("").astype(str).value_counts().to_dict(),
            "line_zone_counts": out["line_zone"].fillna("").astype(str).value_counts().to_dict(),
            "production_behavior_changed": False,
            "promotion_claim": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
