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

from research.market_quality.common import candidate_identity_columns
from research.safe_state.backfill_distribution_quantiles import _line, _line_zone, _target
from research.safe_state.backfill_minutes_state import _candidate_date, _candidate_player, _candidate_player_id, _load_player_logs
from research.safe_state.build_similar_state_store import MIN_SAMPLE, _result, _score_tightness, _side


FALLBACK_LEVELS = [
    "same_player_target_line_zone_minutes_band",
    "same_player_target_line_zone",
    "same_player_target",
    "archetype_target_minutes_band_line_zone",
    "archetype_target_line_zone",
    "opponent_adjusted_archetype_target_line_zone",
]


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _ensure_direction(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "direction" not in out.columns and "side" in out.columns:
        out["direction"] = out["side"]
    return out


def _num(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return np.nan


def _text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return ""


def _candidate_line_zone(row: pd.Series) -> str:
    existing = _text(row, "line_zone")
    if existing:
        return existing
    line = _line(row)
    q25 = _num(row, "target_q25_recent", "q25")
    q50 = _num(row, "target_q50_recent", "q50")
    q75 = _num(row, "target_q75_recent", "q75")
    q90 = _num(row, "target_q90_recent", "q90")
    zone, _ = _line_zone(line, q25, q50, q75, q90)
    return zone


def _minutes_bucket(value: float) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    if value < 20:
        return "LOW"
    if value < 28:
        return "MEDIUM"
    if value < 34:
        return "HIGH"
    return "MAX"


def _target_archetype(row: pd.Series) -> str:
    minutes = _num(row, "minutes_recent_median", "expected_minutes_band_high", "expected_minutes_band_low")
    target = _target(row)
    return f"{target}_{_minutes_bucket(minutes)}_MINUTES"


def _load_all_logs(data_proc_dir: Path | None, market_date: pd.Timestamp, target: str, limit: int | None = None) -> pd.DataFrame:
    if data_proc_dir is None or pd.isna(market_date):
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    paths = sorted(data_proc_dir.glob("*/2026_processed_processed.csv"))
    if limit is not None:
        paths = paths[: int(limit)]
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "Date" not in frame.columns or target not in frame.columns:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.loc[frame["Date"].notna() & frame["Date"].lt(market_date)].copy()
        if frame.empty:
            continue
        frame["player"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pool_for_level(
    *,
    row: pd.Series,
    level: int,
    same_player_logs: pd.DataFrame,
    all_logs: pd.DataFrame,
    target: str,
    line_zone: str,
) -> pd.DataFrame:
    if level in {1, 2, 3}:
        pool = same_player_logs.copy()
    else:
        pool = all_logs.copy()
    if pool.empty or target not in pool.columns:
        return pd.DataFrame()
    if level in {1, 2, 4, 5, 6} and line_zone not in {"UNKNOWN", ""}:
        actual = pd.to_numeric(pool[target], errors="coerce")
        q25, q50, q75, q90 = actual.quantile(0.25), actual.quantile(0.50), actual.quantile(0.75), actual.quantile(0.90)
        line = _line(row)
        zones = []
        for _ in pool.index:
            zone, _ = _line_zone(line, q25, q50, q75, q90)
            zones.append(zone)
        pool = pool.loc[pd.Series(zones, index=pool.index).eq(line_zone)].copy()
    if level in {1, 4}:
        low = _num(row, "expected_minutes_band_low", "minutes_p25_recent")
        high = _num(row, "expected_minutes_band_high", "minutes_p75_recent")
        if pd.notna(low) and pd.notna(high) and "MP" in pool.columns:
            minutes = pd.to_numeric(pool["MP"], errors="coerce")
            pool = pool.loc[minutes.ge(low - 3.0) & minutes.le(high + 3.0)].copy()
    if level == 6:
        opponent = _text(row, "opponent")
        if opponent and "Opponent" in pool.columns:
            opponent_mask = pool["Opponent"].fillna("").astype(str).str.upper().eq(opponent.upper())
            if opponent_mask.any():
                pool = pool.loc[opponent_mask].copy()
    return pool


def _summarize(pool: pd.DataFrame, *, target: str, line: float, side: str, fallback_level: int) -> dict[str, Any]:
    actual = pd.to_numeric(pool.get(target, pd.Series(np.nan, index=pool.index)), errors="coerce")
    residual = actual - float(line) if np.isfinite(line) else pd.Series(np.nan, index=pool.index)
    abs_residual = residual.abs()
    wins = _result(actual, line, side)
    count = int(actual.notna().sum())
    base_tightness = _score_tightness(abs_residual, line, count)
    uncertainty_penalty = max(0.0, (fallback_level - 1) * 0.04)
    tightness = float(np.clip(base_tightness - uncertainty_penalty, 0.0, 1.0))
    if count < MIN_SAMPLE:
        tier = "INSUFFICIENT_SAMPLE"
        status = "INSUFFICIENT_SAMPLE"
    elif tightness >= 0.75:
        tier = "TIGHT"
        status = "SUFFICIENT_TIGHT"
    elif tightness >= 0.55:
        tier = "ACCEPTABLE"
        status = "SUFFICIENT_TIGHT"
    else:
        tier = "SCATTERED"
        status = "SUFFICIENT_SCATTERED"
    interval_threshold = max(2.5, abs(float(line)) * 0.30) if np.isfinite(line) else 6.0
    return {
        "match_count": count,
        "win_rate": float(wins.mean()) if wins.notna().any() else np.nan,
        "avg_residual": float(residual.mean()) if residual.notna().any() else np.nan,
        "median_abs_error": float(abs_residual.median()) if abs_residual.notna().any() else np.nan,
        "p75_abs_error": float(abs_residual.quantile(0.75)) if abs_residual.notna().any() else np.nan,
        "p90_abs_error": float(abs_residual.quantile(0.90)) if abs_residual.notna().any() else np.nan,
        "outcome_iqr": float(actual.quantile(0.75) - actual.quantile(0.25)) if actual.notna().sum() >= 2 else np.nan,
        "tightness_score": tightness,
        "interval_failure_rate": float(abs_residual.gt(interval_threshold).mean()) if abs_residual.notna().any() else np.nan,
        "directional_failure_rate": float(wins.eq(0.0).mean()) if wins.notna().any() else np.nan,
        "comparable_state_reliability_tier": tier,
        "expansion_status": status,
        "uncertainty_penalty": uncertainty_penalty,
    }


def expand_comparable_state_sampling(
    *,
    output_dir: Path,
    needs_more_sample_queue_csv: Path,
    annotated_candidates_csv: Path,
    data_proc_dir: Path | None = None,
    max_archetype_players: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    queue = candidate_identity_columns(_ensure_direction(_read_csv(needs_more_sample_queue_csv)))
    annotated = candidate_identity_columns(_read_csv(annotated_candidates_csv))
    if not annotated.empty and not queue.empty:
        queue = queue.merge(annotated, on="candidate_id", how="left", suffixes=("", "_annotated"))
        for col in [c for c in queue.columns if c.endswith("_annotated")]:
            original = col.removesuffix("_annotated")
            if original in queue.columns:
                queue[original] = queue[original].where(queue[original].notna() & queue[original].astype(str).str.strip().ne(""), queue[col])
                queue = queue.drop(columns=[col])

    records: list[dict[str, Any]] = []
    all_logs_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for _, row in queue.iterrows():
        market_date = _candidate_date(row)
        player = _candidate_player(row)
        player_id = _candidate_player_id(row)
        target = _target(row)
        side = _side(row)
        line = _line(row)
        line_zone = _candidate_line_zone(row)
        same_player_logs = _load_player_logs(data_proc_dir, player, player_id=player_id)
        if not same_player_logs.empty and pd.notna(market_date):
            same_player_logs = same_player_logs.loc[same_player_logs["Date"].lt(market_date)].copy()
        archetype_key = (target, str(market_date.date()) if pd.notna(market_date) else "")
        if archetype_key not in all_logs_cache:
            all_logs_cache[archetype_key] = _load_all_logs(data_proc_dir, market_date, target, limit=max_archetype_players)
        all_logs = all_logs_cache[archetype_key]
        for level, label in enumerate(FALLBACK_LEVELS, start=1):
            pool = _pool_for_level(row=row, level=level, same_player_logs=same_player_logs, all_logs=all_logs, target=target, line_zone=line_zone)
            summary = _summarize(pool, target=target, line=line, side=side, fallback_level=level)
            if level >= 4 and pool.empty:
                summary["expansion_status"] = "NOT_AVAILABLE"
            if level == 6 and _text(row, "opponent") == "":
                summary["expansion_status"] = "CONTEXT_MISMATCH"
            records.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "player": row.get("player", row.get("player_name", "")),
                    "game_id": row.get("game_id", ""),
                    "market_date": row.get("market_date", row.get("game_date", "")),
                    "target": target,
                    "market_type": row.get("market_type", ""),
                    "side": side,
                    "line": line,
                    "line_zone": line_zone,
                    "player_archetype": _target_archetype(row),
                    "fallback_level": level,
                    "fallback_label": label,
                    **summary,
                }
            )

    rows = pd.DataFrame.from_records(records)
    rows_path = output_dir / "comparable_state_expansion_rows.csv"
    summary_path = output_dir / "comparable_state_expansion_summary.json"
    md_path = output_dir / "comparable_state_expansion_report.md"
    rows.to_csv(rows_path, index=False)
    report = {
        "input_paths": {
            "needs_more_sample_queue_csv": str(needs_more_sample_queue_csv),
            "annotated_candidates_csv": str(annotated_candidates_csv),
            "data_proc_dir": str(data_proc_dir) if data_proc_dir else "",
        },
        "output_paths": {"rows_csv": str(rows_path), "summary_json": str(summary_path), "markdown": str(md_path)},
        "queued_candidates": int(len(queue)),
        "expansion_rows": int(len(rows)),
        "expansion_status_counts": rows.get("expansion_status", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict() if not rows.empty else {},
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report, rows)
    return report


def _write_markdown(path: Path, report: dict[str, Any], rows: pd.DataFrame) -> None:
    lines = [
        "# Comparable-State Expansion Report",
        "",
        f"- Queued candidates: {report['queued_candidates']}",
        f"- Expansion rows: {report['expansion_rows']}",
        "- Production behavior changed: false",
        "- Promotion claim: false",
        "",
        "## Status Counts",
    ]
    for status, count in report["expansion_status_counts"].items():
        lines.append(f"- {status}: {count}")
    lines.extend(["", "Wider fallback levels carry explicit uncertainty penalties and cannot become production evidence by themselves."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand comparable-state sampling for NEEDS_MORE_SAMPLE rows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--needs-more-sample-queue-csv", type=Path, required=True)
    parser.add_argument("--annotated-candidates-csv", type=Path, required=True)
    parser.add_argument("--data-proc-dir", type=Path)
    parser.add_argument("--max-archetype-players", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = expand_comparable_state_sampling(
        output_dir=args.output_dir,
        needs_more_sample_queue_csv=args.needs_more_sample_queue_csv,
        annotated_candidates_csv=args.annotated_candidates_csv,
        data_proc_dir=args.data_proc_dir,
        max_archetype_players=args.max_archetype_players,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
