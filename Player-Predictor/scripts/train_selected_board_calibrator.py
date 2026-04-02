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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected-board walk-forward calibrator.")
    parser.add_argument(
        "--rows-csv",
        type=Path,
        required=True,
        help="Resolved row-level CSV (e.g., validate_board_objective_mode rows output).",
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


def _prepare_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {args.run_date_col, args.target_col, args.direction_col, args.prob_col, args.result_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Rows CSV missing required columns: {missing}")

    out = df.copy()
    raw_dates = out[args.run_date_col]
    parsed_token = pd.to_datetime(raw_dates.astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    parsed_generic = pd.to_datetime(raw_dates, errors="coerce")
    out["_run_date"] = parsed_token.fillna(parsed_generic)
    out["_target"] = out[args.target_col].astype(str).str.upper().str.strip()
    out["_direction"] = out[args.direction_col].astype(str).str.upper().str.strip()
    out["_prob"] = pd.to_numeric(out[args.prob_col], errors="coerce")
    out["_result"] = out[args.result_col].astype(str).str.lower().str.strip()
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
    rows_csv = args.rows_csv.resolve()
    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")

    raw = pd.read_csv(rows_csv)
    rows = _prepare_rows(raw, args)

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
