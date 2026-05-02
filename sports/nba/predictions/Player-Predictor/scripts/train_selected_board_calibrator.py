#!/usr/bin/env python3
"""
Train a walk-forward selected-board probability calibrator.

Input rows should contain at least:
- run_date (YYYYMMDD or parseable date)
- target
- direction
- expected_win_rate (or alternate --prob-col)
- result in {win, loss, push, missing}

The calibrator is fit month-by-month using only prior data within a rolling
lookback window, then can be applied live with a monthly freeze.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine.selected_board_calibration import (
    CalibratorFitConfig,
    apply_selected_board_calibration,
    evaluate_calibration,
    fit_selected_board_calibrator_payload,
)

ANALYSIS_ROOT = REPO_ROOT / "model" / "analysis"
WORKSPACE_ROOT = REPO_ROOT.parents[3]
SHARED_VALIDATION_ROOT = WORKSPACE_ROOT / "sports" / "validation"
DATE_COL_CANDIDATES = ("run_date", "market_date", "run_date_iso")
TARGET_COL_CANDIDATES = ("target",)
DIRECTION_COL_CANDIDATES = ("direction",)
PROB_COL_CANDIDATES = ("expected_win_rate", "estimated_win_rate", "board_play_win_prob", "p_calibrated")
RESULT_COL_CANDIDATES = ("result",)
PREFERRED_ROWS_PATTERNS: tuple[tuple[str, int], ...] = (
    ("validation_recent_pool_selector", 500),
    ("selector_replay_rows_rebuilt", 400),
    ("selector_replay_rows", 360),
    ("board_size_history_rows_current_prod", 320),
    ("validation_current_prod_hitrate_rows", 280),
    ("line_decision_sidecar_backtest_rows_rebuilt", 240),
)


@dataclass(frozen=True)
class ResolvedInputColumns:
    run_date_col: str
    target_col: str
    direction_col: str
    prob_col: str
    result_col: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected-board walk-forward calibrator.")
    parser.add_argument(
        "--rows-csv",
        type=Path,
        default=None,
        help="Resolved row-level CSV. When omitted, the trainer auto-discovers the best recent replay/validation CSV.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ANALYSIS_ROOT,
        help="Analysis directory searched when --rows-csv is omitted.",
    )
    parser.add_argument("--run-date-col", type=str, default="run_date")
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--direction-col", type=str, default="direction")
    parser.add_argument("--prob-col", type=str, default="expected_win_rate")
    parser.add_argument("--result-col", type=str, default="result")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--min-rows-global", type=int, default=250)
    parser.add_argument("--min-rows-segment", type=int, default=80)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator.json",
        help="Calibrator payload output JSON.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator_report.json",
        help="Training/evaluation report output JSON.",
    )
    return parser.parse_args()


def _parse_run_dates(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed_token = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    parsed_generic = pd.to_datetime(series, errors="coerce")
    return parsed_token.fillna(parsed_generic)


def _resolve_column(
    df: pd.DataFrame,
    preferred: str,
    candidates: tuple[str, ...],
) -> str:
    if preferred in df.columns:
        return str(preferred)
    for candidate in candidates:
        if candidate in df.columns:
            return str(candidate)
    raise ValueError(f"None of the candidate columns were found: {[preferred, *candidates]}")


def resolve_input_columns(df: pd.DataFrame, args: argparse.Namespace) -> ResolvedInputColumns:
    return ResolvedInputColumns(
        run_date_col=_resolve_column(df, str(args.run_date_col), DATE_COL_CANDIDATES),
        target_col=_resolve_column(df, str(args.target_col), TARGET_COL_CANDIDATES),
        direction_col=_resolve_column(df, str(args.direction_col), DIRECTION_COL_CANDIDATES),
        prob_col=_resolve_column(df, str(args.prob_col), PROB_COL_CANDIDATES),
        result_col=_resolve_column(df, str(args.result_col), RESULT_COL_CANDIDATES),
    )


def _candidate_name_score(path: Path) -> int:
    name = path.name.lower()
    for token, score in PREFERRED_ROWS_PATTERNS:
        if token in name:
            return int(score)
    return 0


def discover_rows_csv(analysis_dir: Path, args: argparse.Namespace) -> Path:
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found for auto-discovery: {analysis_dir}")

    best: tuple[tuple[int, int, int, int, str], Path] | None = None
    search_roots = [analysis_dir]
    if SHARED_VALIDATION_ROOT.exists() and SHARED_VALIDATION_ROOT.resolve() != analysis_dir.resolve():
        search_roots.append(SHARED_VALIDATION_ROOT.resolve())

    seen_paths: set[Path] = set()
    for root in search_roots:
        for path in sorted(root.glob("*.csv")):
            resolved_path = path.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            try:
                df = pd.read_csv(path)
                columns = resolve_input_columns(df, args)
            except Exception:
                continue

            parsed_dates = _parse_run_dates(df[columns.run_date_col])
            valid_targets = df[columns.target_col].astype(str).str.upper().str.strip().isin(["PTS", "TRB", "AST"])
            valid_directions = df[columns.direction_col].astype(str).str.upper().str.strip().isin(["OVER", "UNDER"])
            valid_results = df[columns.result_col].astype(str).str.lower().str.strip().isin(["win", "loss"])
            valid_probs = pd.to_numeric(df[columns.prob_col], errors="coerce").notna()
            valid_dates = parsed_dates.notna()
            resolved_mask = valid_targets & valid_directions & valid_results & valid_probs & valid_dates
            resolved_rows = int(resolved_mask.sum())
            if resolved_rows <= 0:
                continue

            resolved_dates = parsed_dates.loc[resolved_mask]
            unique_days = int(resolved_dates.dt.normalize().nunique())
            latest_day = resolved_dates.max()
            latest_ordinal = int(latest_day.toordinal()) if pd.notna(latest_day) else 0
            rank = (
                _candidate_name_score(path),
                unique_days,
                resolved_rows,
                latest_ordinal,
                str(path),
            )
            if best is None or rank > best[0]:
                best = (rank, path)

    if best is None:
        raise FileNotFoundError(
            f"No auto-discoverable selected-board calibrator rows CSV was found under {analysis_dir} or {SHARED_VALIDATION_ROOT}."
        )
    return best[1]


def _prepare_rows(
    df: pd.DataFrame,
    columns: ResolvedInputColumns,
) -> pd.DataFrame:
    missing = [value for value in columns.__dict__.values() if value not in df.columns]
    if missing:
        raise ValueError(f"Rows CSV missing required resolved columns: {missing}")

    out = df.copy()
    out["_run_date"] = _parse_run_dates(out[columns.run_date_col])
    out["_target"] = out[columns.target_col].astype(str).str.upper().str.strip()
    out["_direction"] = out[columns.direction_col].astype(str).str.upper().str.strip()
    out["_prob"] = pd.to_numeric(out[columns.prob_col], errors="coerce")
    out["_result"] = out[columns.result_col].astype(str).str.lower().str.strip()
    out = out.loc[out["_run_date"].notna() & out["_target"].isin(["PTS", "TRB", "AST"]) & out["_direction"].isin(["OVER", "UNDER"])].copy()
    out = out.loc[out["_result"].isin(["win", "loss"])].copy()
    if out.empty:
        raise RuntimeError("No resolved win/loss rows after preprocessing.")
    out["is_win"] = (out["_result"] == "win").astype("float64")
    out = out.sort_values("_run_date").reset_index(drop=True)
    return out


def _walkforward_apply(
    rows: pd.DataFrame,
    payload: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = rows.copy()
    calibrated_values = []
    calibration_sources = []
    calibration_months = []
    for _, row in out.iterrows():
        frame = pd.DataFrame(
            [
                {
                    "target": row["_target"],
                    "direction": row["_direction"],
                    "board_play_win_prob": float(np.clip(float(row["_prob"]), 0.01, 0.99)),
                    "market_date": row["_run_date"],
                }
            ]
        )
        calibrated, source, month = apply_selected_board_calibration(
            frame,
            payload=payload,
            run_date_hint=row["_run_date"].strftime("%Y-%m-%d"),
            prob_col="board_play_win_prob",
            target_col="target",
            direction_col="direction",
        )
        calibrated_values.append(float(pd.to_numeric(calibrated, errors="coerce").fillna(0.5).iloc[0]))
        calibration_sources.append(str(source.iloc[0] if len(source) else "identity"))
        calibration_months.append(str(month))
    out["p_raw"] = pd.to_numeric(out["_prob"], errors="coerce").fillna(0.5).clip(lower=0.01, upper=0.99)
    out["p_calibrated"] = pd.Series(calibrated_values, index=out.index, dtype="float64").clip(lower=0.01, upper=0.99)
    out["calibration_source"] = calibration_sources
    out["calibration_month"] = calibration_months
    return out


def _segment_metrics(df: pd.DataFrame, prob_col: str) -> list[dict]:
    rows: list[dict] = []
    for (target, direction), part in df.groupby(["_target", "_direction"], dropna=False):
        metrics = evaluate_calibration(
            probs=pd.to_numeric(part[prob_col], errors="coerce").fillna(0.5).to_numpy(dtype="float64"),
            labels=pd.to_numeric(part["is_win"], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
        )
        rows.append(
            {
                "segment": f"{str(target)}_{str(direction)}",
                "rows": int(len(part)),
                "mean_prob": metrics["mean_prob"],
                "mean_label": metrics["mean_label"],
                "gap": metrics["gap"],
                "brier": metrics["brier"],
                "log_loss": metrics["log_loss"],
                "ece_10": metrics["ece_10"],
            }
        )
    return sorted(rows, key=lambda x: x["segment"])


def main() -> None:
    args = parse_args()
    rows_csv = (args.rows_csv or discover_rows_csv(args.analysis_dir.resolve(), args)).resolve()
    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")

    raw = pd.read_csv(rows_csv)
    resolved_columns = resolve_input_columns(raw, args)
    rows = _prepare_rows(raw, resolved_columns)

    cfg = CalibratorFitConfig(
        lookback_days=int(args.lookback_days),
        min_rows_global=int(args.min_rows_global),
        min_rows_segment=int(args.min_rows_segment),
        n_bins=int(args.n_bins),
    )
    fit_df = pd.DataFrame(
        {
            "run_date": rows["_run_date"],
            "target": rows["_target"],
            "direction": rows["_direction"],
            "is_win": rows["is_win"],
            "probability": pd.to_numeric(rows["_prob"], errors="coerce").fillna(0.5),
        }
    )
    payload = fit_selected_board_calibrator_payload(
        rows_df=fit_df,
        run_date_col="run_date",
        prob_col="probability",
        label_col="is_win",
        target_col="target",
        direction_col="direction",
        config=cfg,
    )

    applied = _walkforward_apply(rows, payload=payload, args=args)
    raw_metrics = evaluate_calibration(
        probs=applied["p_raw"].to_numpy(dtype="float64"),
        labels=applied["is_win"].to_numpy(dtype="float64"),
    )
    cal_metrics = evaluate_calibration(
        probs=applied["p_calibrated"].to_numpy(dtype="float64"),
        labels=applied["is_win"].to_numpy(dtype="float64"),
    )

    report = {
        "rows_csv": str(rows_csv),
        "resolved_columns": resolved_columns.__dict__.copy(),
        "rows_resolved": int(len(rows)),
        "config": cfg.__dict__.copy(),
        "months_fitted": sorted((payload.get("months") or {}).keys()),
        "raw": raw_metrics,
        "calibrated": cal_metrics,
        "delta": {
            "gap_pp": float((cal_metrics["gap"] - raw_metrics["gap"]) * 100.0),
            "brier": float(cal_metrics["brier"] - raw_metrics["brier"]),
            "log_loss": float(cal_metrics["log_loss"] - raw_metrics["log_loss"]),
            "ece_10": float(cal_metrics["ece_10"] - raw_metrics["ece_10"]),
        },
        "segment_raw": _segment_metrics(applied, "p_raw"),
        "segment_calibrated": _segment_metrics(applied, "p_calibrated"),
        "calibration_source_counts": applied["calibration_source"].value_counts(dropna=False).to_dict(),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Calibrator JSON: {args.out_json}")
    print(f"Report JSON:     {args.report_json}")
    print("Raw metrics:")
    print(json.dumps(raw_metrics, indent=2))
    print("Calibrated metrics:")
    print(json.dumps(cal_metrics, indent=2))
    print("Delta (calibrated - raw):")
    print(json.dumps(report["delta"], indent=2))


if __name__ == "__main__":
    main()
