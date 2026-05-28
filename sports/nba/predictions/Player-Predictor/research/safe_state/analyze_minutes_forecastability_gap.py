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

from research.safe_state.backfill_minutes_state import (
    MIN_SAMPLE,
    _candidate_date,
    _candidate_player,
    _candidate_player_id,
    _load_player_logs,
    _minutes_series,
)


MINUTES_GAP = "FORECASTABILITY_GAP_MINUTES_STATE"
TRUE_UNSTABLE = "TRUE_UNSTABLE_STATE"
FIX_EXISTING_LOGS = "FIXABLE_WITH_EXISTING_LOGS"
FIX_NEW_PIPELINE = "FIXABLE_WITH_NEW_PIPELINE_DATA"
NEEDS_MORE_SAMPLE = "NEEDS_MORE_SAMPLE"
FEATURE_MISSING = "FEATURE_MISSING"

MINUTES_GAP_COLUMNS = [
    "minutes_gap_subtype",
    "minutes_gap_primary_driver",
    "minutes_gap_severity",
    "minutes_gap_fixability",
    "minutes_gap_reason",
    "minutes_gap_recommended_fix",
    "minutes_gap_blocks_safe_state_flag",
]


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


def _has_minutes_gap(row: pd.Series) -> bool:
    values = [
        _text(row, "forecastability_gap_primary"),
        _text(row, "forecastability_gap_secondary"),
        _text(row, "minutes_state_gap_type"),
        _text(row, "safe_state_blockers"),
        _text(row, "primary_blocker"),
    ]
    return any(MINUTES_GAP in value.upper() for value in values)


def _has_minutes_evidence(row: pd.Series) -> bool:
    columns = [
        "minutes_floor_recent",
        "minutes_p25_recent",
        "minutes_p75_recent",
        "minutes_recent_std",
        "minutes_recent_cv",
        "expected_minutes_band_width",
        "expected_minutes_band_low",
        "expected_minutes_band_high",
        "minutes_state_sample_count",
    ]
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return True
    return False


def _logs_available(row: pd.Series, data_proc_dir: Path | None) -> tuple[bool, int, str]:
    if data_proc_dir is None:
        return False, 0, "data_proc_dir_not_supplied"
    player = _candidate_player(row)
    player_id = _candidate_player_id(row)
    market_date = _candidate_date(row)
    logs = _load_player_logs(data_proc_dir, player, player_id=player_id)
    if logs.empty:
        return False, 0, "player_logs_not_found"
    if pd.isna(market_date):
        return True, 0, "market_date_missing"
    prior = logs.loc[logs["Date"] < market_date].copy()
    minutes = _minutes_series(prior).dropna()
    return True, int(len(minutes)), "pre_event_logs_found"


def _severity_for_numeric(kind: str, value: float) -> str:
    if kind == "floor":
        if pd.notna(value) and value < 14.0:
            return "CRITICAL"
        return "HIGH"
    if kind == "width":
        if pd.notna(value) and value > 14.0:
            return "CRITICAL"
        return "HIGH"
    if kind == "cv":
        if pd.notna(value) and value > 0.45:
            return "CRITICAL"
        return "HIGH"
    if kind == "role":
        if pd.notna(value) and value >= 3.0:
            return "CRITICAL"
        return "HIGH"
    if kind in {"blowout", "foul"}:
        if pd.notna(value) and value >= 0.80:
            return "HIGH"
        return "MEDIUM"
    return "MEDIUM"


def _candidate_driver(row: pd.Series, data_proc_dir: Path | None) -> dict[str, Any]:
    sample_count = _num(row, "minutes_state_sample_count", "recent_games_count", "minutes_sample_count")
    has_evidence = _has_minutes_evidence(row)
    logs_found, log_sample_count, log_reason = _logs_available(row, data_proc_dir)
    existing_fixability = _text(row, "forecastability_gap_fixability", "minutes_state_fixability").upper()

    if not has_evidence:
        if existing_fixability == TRUE_UNSTABLE:
            return {
                "subtype": "MINUTES_ROLE_UNSTABLE",
                "driver": "existing_true_unstable_forecastability_label",
                "severity": "CRITICAL",
                "fixability": TRUE_UNSTABLE,
                "reason": "existing forecastability evidence already labels minutes state as TRUE_UNSTABLE_STATE",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
                "blocks": True,
            }
        if logs_found:
            return {
                "subtype": "MINUTES_PIPELINE_MISSING",
                "driver": "missing_minutes_features_with_logs_available",
                "severity": "MEDIUM",
                "fixability": FIX_EXISTING_LOGS,
                "reason": f"{log_reason};pre_event_minutes_logs={log_sample_count}",
                "recommended_fix": "FIX_EXISTING_LOG_PARSE",
                "blocks": True,
            }
        return {
            "subtype": "MINUTES_PIPELINE_MISSING",
            "driver": "missing_minutes_features_no_verified_logs",
            "severity": "HIGH",
            "fixability": FEATURE_MISSING,
            "reason": log_reason,
            "recommended_fix": "REFRESH_PLAYER_DATA",
            "blocks": True,
        }

    if pd.notna(sample_count) and sample_count < MIN_SAMPLE:
        return {
            "subtype": "MINUTES_SAMPLE_INSUFFICIENT",
            "driver": "minutes_sample_count",
            "severity": "MEDIUM",
            "fixability": NEEDS_MORE_SAMPLE,
            "reason": f"minutes_sample_count={int(sample_count)}",
            "recommended_fix": "NEEDS_MORE_SAMPLE",
            "blocks": True,
        }

    floor = _num(row, "minutes_floor_recent", "expected_minutes_band_low")
    width = _num(row, "expected_minutes_band_width")
    if pd.isna(width):
        high = _num(row, "expected_minutes_band_high", "minutes_p75_recent")
        low = _num(row, "expected_minutes_band_low", "minutes_p25_recent")
        if pd.notna(high) and pd.notna(low):
            width = high - low
    std = _num(row, "minutes_recent_std", "minutes_std_recent")
    cv = _num(row, "minutes_recent_cv", "minutes_cv_recent")
    role_changes = _num(row, "starter_status_change_count", "role_change_count")
    rotation_volatility = _num(row, "rotation_volatility_score")
    blowout = _num(row, "blowout_minutes_sensitivity")
    foul = _num(row, "foul_rate_minutes_loss_risk")
    starter_status = _text(row, "starter_status_recent", "role_state", "rotation_role").lower()

    candidates: list[dict[str, Any]] = []
    if pd.notna(floor) and floor < 18.0:
        candidates.append(
            {
                "subtype": "MINUTES_LOW_FLOOR",
                "driver": "minutes_floor_recent",
                "severity": _severity_for_numeric("floor", floor),
                "fixability": TRUE_UNSTABLE,
                "reason": f"minutes_floor_recent={floor:.1f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if pd.notna(width) and width > 8.0:
        candidates.append(
            {
                "subtype": "MINUTES_WIDE_BAND",
                "driver": "expected_minutes_band_width",
                "severity": _severity_for_numeric("width", width),
                "fixability": TRUE_UNSTABLE,
                "reason": f"expected_minutes_band_width={width:.1f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if (pd.notna(cv) and cv > 0.28) or (pd.notna(std) and std > 6.0):
        value = cv if pd.notna(cv) else std
        driver = "minutes_recent_cv" if pd.notna(cv) else "minutes_recent_std"
        candidates.append(
            {
                "subtype": "MINUTES_HIGH_VOLATILITY",
                "driver": driver,
                "severity": _severity_for_numeric("cv", value),
                "fixability": TRUE_UNSTABLE,
                "reason": f"{driver}={value:.3f}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    role_unstable = (
        (pd.notna(role_changes) and role_changes > 0)
        or (pd.notna(rotation_volatility) and rotation_volatility > 0.65)
        or any(token in starter_status for token in ["bench", "uncertain", "changed", "volatile"])
    )
    if role_unstable:
        driver_value = role_changes if pd.notna(role_changes) else rotation_volatility
        candidates.append(
            {
                "subtype": "MINUTES_ROLE_UNSTABLE",
                "driver": "starter_status_change_count" if pd.notna(role_changes) else "rotation_volatility_score",
                "severity": _severity_for_numeric("role", driver_value),
                "fixability": TRUE_UNSTABLE,
                "reason": f"starter_status={starter_status or 'unknown'};role_changes={role_changes if pd.notna(role_changes) else 'na'};rotation_volatility={rotation_volatility if pd.notna(rotation_volatility) else 'na'}",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if pd.notna(blowout) and blowout > 0.60:
        candidates.append(
            {
                "subtype": "MINUTES_BLOWOUT_SENSITIVE",
                "driver": "blowout_minutes_sensitivity",
                "severity": _severity_for_numeric("blowout", blowout),
                "fixability": FIX_NEW_PIPELINE if blowout < 0.80 else TRUE_UNSTABLE,
                "reason": f"blowout_minutes_sensitivity={blowout:.2f}",
                "recommended_fix": "ADD_OPPONENT_CONTEXT_PIPELINE" if blowout < 0.80 else "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )
    if pd.notna(foul) and foul > 0.60:
        candidates.append(
            {
                "subtype": "MINUTES_FOUL_SENSITIVE",
                "driver": "foul_rate_minutes_loss_risk",
                "severity": _severity_for_numeric("foul", foul),
                "fixability": FIX_NEW_PIPELINE if foul < 0.80 else TRUE_UNSTABLE,
                "reason": f"foul_rate_minutes_loss_risk={foul:.2f}",
                "recommended_fix": "ADD_FOUL_RISK_CONTEXT" if foul < 0.80 else "KEEP_UNSAFE_TRUE_VOLATILITY",
            }
        )

    if not candidates:
        if existing_fixability == TRUE_UNSTABLE:
            return {
                "subtype": "MINUTES_ROLE_UNSTABLE",
                "driver": "existing_true_unstable_forecastability_label",
                "severity": "CRITICAL",
                "fixability": TRUE_UNSTABLE,
                "reason": "existing forecastability evidence already labels minutes state as TRUE_UNSTABLE_STATE",
                "recommended_fix": "KEEP_UNSAFE_TRUE_VOLATILITY",
                "blocks": True,
            }
        return {
            "subtype": "MINUTES_SAMPLE_INSUFFICIENT" if pd.isna(sample_count) else "MINUTES_PIPELINE_MISSING",
            "driver": "minutes_gap_without_specific_driver",
            "severity": "MEDIUM",
            "fixability": FIX_EXISTING_LOGS if logs_found else FEATURE_MISSING,
            "reason": "minutes_gap_present_but_specific_driver_missing",
            "recommended_fix": "FIX_EXISTING_LOG_PARSE" if logs_found else "REFRESH_PLAYER_DATA",
            "blocks": True,
        }

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
    subtype_order = {
        "MINUTES_LOW_FLOOR": 0,
        "MINUTES_WIDE_BAND": 1,
        "MINUTES_HIGH_VOLATILITY": 2,
        "MINUTES_ROLE_UNSTABLE": 3,
        "MINUTES_BLOWOUT_SENSITIVE": 4,
        "MINUTES_FOUL_SENSITIVE": 5,
    }
    winner = sorted(candidates, key=lambda item: (severity_order.get(item["severity"], 9), subtype_order.get(item["subtype"], 99)))[0]
    winner["blocks"] = True
    return winner


def annotate_minutes_gap_decomposition(
    candidates: pd.DataFrame,
    *,
    data_proc_dir: Path | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        if not _has_minutes_gap(row):
            rows.append(
                {
                    "minutes_gap_subtype": "",
                    "minutes_gap_primary_driver": "",
                    "minutes_gap_severity": "NONE",
                    "minutes_gap_fixability": "",
                    "minutes_gap_reason": "",
                    "minutes_gap_recommended_fix": "",
                    "minutes_gap_blocks_safe_state_flag": False,
                }
            )
            continue
        driver = _candidate_driver(row, data_proc_dir)
        rows.append(
            {
                "minutes_gap_subtype": driver["subtype"],
                "minutes_gap_primary_driver": driver["driver"],
                "minutes_gap_severity": driver["severity"],
                "minutes_gap_fixability": driver["fixability"],
                "minutes_gap_reason": driver["reason"],
                "minutes_gap_recommended_fix": driver["recommended_fix"],
                "minutes_gap_blocks_safe_state_flag": bool(driver["blocks"]),
            }
        )
    frame = pd.DataFrame(rows, index=out.index)
    for column in frame.columns:
        out[column] = frame[column]
    return out


def _write_summary(path: Path, out: pd.DataFrame) -> None:
    payload = {
        "rows": int(len(out)),
        "minutes_gap_rows": int(out["minutes_gap_subtype"].fillna("").astype(str).str.strip().ne("").sum()),
        "minutes_gap_subtype_counts": out["minutes_gap_subtype"].fillna("").astype(str).replace("", "NONE").value_counts().to_dict(),
        "minutes_gap_fixability_counts": out["minutes_gap_fixability"].fillna("").astype(str).replace("", "NONE").value_counts().to_dict(),
        "production_behavior_changed": False,
        "promotion_claim": False,
        "shadow_only": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose minutes-state forecastability blockers.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--data-proc-dir", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidate_csv)
    out = annotate_minutes_gap_decomposition(candidates, data_proc_dir=args.data_proc_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    if args.summary_json:
        _write_summary(args.summary_json, out)


if __name__ == "__main__":
    main()
