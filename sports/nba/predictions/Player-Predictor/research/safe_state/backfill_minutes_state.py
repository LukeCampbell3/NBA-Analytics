from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MIN_SAMPLE = 3
DEFAULT_LOOKBACK = 10


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _player_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _candidate_player(row: pd.Series) -> str:
    for col in ["player", "player_name", "market_player_raw"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return _player_key(value)
    return ""


def _candidate_player_id(row: pd.Series) -> str:
    for col in ["player_id", "Player_ID"]:
        if col in row.index:
            value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return str(int(value))
    return ""


def _candidate_date(row: pd.Series) -> pd.Timestamp:
    for col in ["market_date", "game_date", "run_date", "date"]:
        if col in row.index:
            value = pd.to_datetime(row.get(col), errors="coerce")
            if pd.notna(value):
                return value.normalize()
    return pd.NaT


def _load_player_logs(data_proc_dir: Path | None, player: str, player_id: str = "") -> pd.DataFrame:
    if data_proc_dir is None or not player:
        return pd.DataFrame()
    player_dir = data_proc_dir / player
    if not player_dir.exists():
        matches = list(data_proc_dir.glob(f"{player}*"))
        player_dir = matches[0] if matches else player_dir
    if not player_dir.exists() and player_id:
        for path in data_proc_dir.glob("*/2026_processed_processed.csv"):
            try:
                head = pd.read_csv(path, nrows=3, usecols=lambda col: col in {"Player_ID"})
            except Exception:
                continue
            ids = pd.to_numeric(head.get("Player_ID", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).astype(str)
            if player_id in set(ids.tolist()):
                player_dir = path.parent
                break
    if not player_dir.exists():
        return pd.DataFrame()
    files = sorted(player_dir.glob("*_processed_processed.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        frame = _read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "Date" not in out.columns:
        return pd.DataFrame()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out.dropna(subset=["Date"]).sort_values("Date")


def _minutes_series(logs: pd.DataFrame) -> pd.Series:
    if logs.empty:
        return pd.Series(dtype="float64")
    for col in ["MP", "MIN", "minutes", "Minutes"]:
        if col in logs.columns:
            values = pd.to_numeric(logs[col], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(dtype="float64")


def _score_minutes(values: pd.Series) -> tuple[float, str, str, str]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < MIN_SAMPLE:
        return 0.0, "FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA", f"minutes_sample_count={len(values)}", "NEEDS_MORE_SAMPLE"
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    cv = float(std / mean) if mean > 0 else 1.0
    p25 = float(values.quantile(0.25))
    p75 = float(values.quantile(0.75))
    floor = float(values.min())
    width = float(p75 - p25)
    band_score = (1.0 - (width / 14.0))
    floor_score = (floor - 12.0) / 24.0
    cv_score = 1.0 - (cv / 0.35)
    score = float(np.clip(0.40 * band_score + 0.35 * floor_score + 0.25 * cv_score, 0.0, 1.0))
    if floor < 18 or width > 9 or cv > 0.28:
        return score, "FORECASTABILITY_GAP_MINUTES_STATE", f"minutes_floor={floor:.1f};minutes_iqr={width:.1f};minutes_cv={cv:.3f}", "TRUE_UNSTABLE_STATE"
    return score, "", "minutes_state_stable", "FIXABLE_WITH_EXISTING_LOGS"


def backfill_minutes_state(
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
        market_date = _candidate_date(row)
        cache_key = player_id or player
        if cache_key not in cache:
            cache[cache_key] = _load_player_logs(data_proc_dir, player, player_id=player_id)
        logs = cache[cache_key]
        if logs.empty or pd.isna(market_date):
            rows.append(
                {
                    "minutes_recent_mean": np.nan,
                    "minutes_recent_median": np.nan,
                    "minutes_recent_std": np.nan,
                    "minutes_recent_cv": np.nan,
                    "minutes_floor_recent": np.nan,
                    "minutes_p25_recent": np.nan,
                    "minutes_p75_recent": np.nan,
                    "minutes_ceiling_recent": np.nan,
                    "expected_minutes_band_low": np.nan,
                    "expected_minutes_band_high": np.nan,
                    "expected_minutes_band_width": np.nan,
                    "minutes_band_stability_score": 0.0,
                    "minutes_forecastability_score": 0.0,
                    "minutes_state_source": "missing_player_logs",
                    "minutes_state_sample_count": 0,
                    "minutes_state_warning": "missing_pre_event_player_logs",
                    "minutes_state_gap_type": "FORECASTABILITY_GAP_INSUFFICIENT_PRE_EVENT_DATA",
                    "minutes_state_gap_reason": "missing_pre_event_player_logs",
                    "minutes_state_fixability": "FIXABLE_WITH_EXISTING_LOGS",
                }
            )
            continue
        prior = logs.loc[logs["Date"] < market_date].tail(int(lookback_games)).copy()
        minutes = _minutes_series(prior).dropna()
        score, gap_type, reason, fixability = _score_minutes(minutes)
        if minutes.empty:
            mean = median = std = cv = floor = p25 = p75 = ceiling = np.nan
        else:
            mean = float(minutes.mean())
            median = float(minutes.median())
            std = float(minutes.std(ddof=0)) if len(minutes) > 1 else 0.0
            cv = float(std / mean) if mean > 0 else np.nan
            floor = float(minutes.min())
            p25 = float(minutes.quantile(0.25))
            p75 = float(minutes.quantile(0.75))
            ceiling = float(minutes.max())
        band_low = p25 if pd.notna(p25) else np.nan
        band_high = p75 if pd.notna(p75) else np.nan
        rows.append(
            {
                "minutes_recent_mean": mean,
                "minutes_recent_median": median,
                "minutes_recent_std": std,
                "minutes_recent_cv": cv,
                "minutes_floor_recent": floor,
                "minutes_p25_recent": p25,
                "minutes_p75_recent": p75,
                "minutes_ceiling_recent": ceiling,
                "expected_minutes_band_low": band_low,
                "expected_minutes_band_high": band_high,
                "expected_minutes_band_width": float(band_high - band_low) if pd.notna(band_low) and pd.notna(band_high) else np.nan,
                "minutes_band_stability_score": score,
                "minutes_forecastability_score": score,
                "minutes_state_source": "Data-Proc_pre_event_logs",
                "minutes_state_sample_count": int(len(minutes)),
                "minutes_state_warning": "" if not gap_type else reason,
                "minutes_state_gap_type": gap_type,
                "minutes_state_gap_reason": reason,
                "minutes_state_fixability": fixability,
            }
        )
    backfill = pd.DataFrame(rows, index=out.index)
    for col in backfill.columns:
        out[col] = backfill[col]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill pre-event minutes-state diagnostics for safe-state research.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--data-proc-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Data-Proc")
    parser.add_argument("--lookback-games", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    out = backfill_minutes_state(candidates, data_proc_dir=args.data_proc_dir, lookback_games=int(args.lookback_games))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    if args.summary_json:
        payload = {
            "rows": int(len(out)),
            "minutes_state_gap_counts": out["minutes_state_gap_type"].fillna("").astype(str).value_counts().to_dict(),
            "production_behavior_changed": False,
            "promotion_claim": False,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
