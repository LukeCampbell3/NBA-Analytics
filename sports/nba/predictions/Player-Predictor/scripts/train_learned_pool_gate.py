#!/usr/bin/env python3
"""
Train a walk-forward learned gate from resolved board rows.

The learned gate fits per-month probability thresholds (global + target/direction
segments) using only prior data and exports a payload that can be applied live.
"""

from __future__ import annotations

import argparse
import json
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train walk-forward learned pool-gate thresholds.")
    parser.add_argument(
        "--rows-csv",
        type=Path,
        required=True,
        help="Resolved row-level replay CSV (for example validate_board_objective_mode rows output).",
    )
    parser.add_argument("--run-date-col", type=str, default="run_date")
    parser.add_argument("--prob-col", type=str, default="expected_win_rate")
    parser.add_argument("--result-col", type=str, default="result")
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--direction-col", type=str, default="direction")
    parser.add_argument(
        "--train-lookback-days",
        type=int,
        default=120,
        help="Rolling lookback days for each monthly fit (0 uses all prior rows).",
    )
    parser.add_argument("--min-train-rows-global", type=int, default=200)
    parser.add_argument("--min-train-rows-segment", type=int, default=80)
    parser.add_argument(
        "--threshold-quantiles",
        type=str,
        default="0.00,0.10,0.20,0.30,0.40,0.50,0.60",
        help="Candidate quantiles over probability for threshold search.",
    )
    parser.add_argument(
        "--min-accept-rate",
        type=float,
        default=0.35,
        help="Minimum kept share constraint during threshold search.",
    )
    parser.add_argument(
        "--max-accept-rate",
        type=float,
        default=0.95,
        help="Maximum kept share constraint during threshold search.",
    )
    parser.add_argument(
        "--include-next-month",
        action="store_true",
        help="Also fit and emit threshold payload for the month after the latest observed month.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("model/analysis/calibration/learned_pool_gate.json"),
        help="Output learned-gate payload JSON.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("model/analysis/calibration/learned_pool_gate_report.json"),
        help="Output training report JSON.",
    )
    return parser.parse_args()


def _month_start(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=value.year, month=value.month, day=1)


def _normalize_run_date_token(value: object) -> pd.Timestamp | pd.NaT:
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def _fit_threshold(
    frame: pd.DataFrame,
    prob_col: str,
    label_col: str,
    quantiles: list[float],
    min_accept_rate: float,
    max_accept_rate: float,
) -> dict[str, Any] | None:
    if frame.empty:
        return None
    probs = pd.to_numeric(frame[prob_col], errors="coerce").dropna()
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    valid = probs.index.intersection(labels.dropna().index)
    if len(valid) == 0:
        return None
    probs = probs.loc[valid].clip(lower=0.0, upper=1.0)
    labels = labels.loc[valid].clip(lower=0.0, upper=1.0)
    total_rows = int(len(probs))
    if total_rows <= 0:
        return None

    q_values = sorted(set(float(np.clip(q, 0.0, 1.0)) for q in quantiles))
    candidates = sorted(set(float(probs.quantile(q)) for q in q_values))
    if not candidates:
        candidates = [0.0]

    baseline_hit = float(labels.mean())
    best: dict[str, Any] | None = None
    for threshold in candidates:
        keep_mask = probs >= float(threshold)
        keep_rows = int(keep_mask.sum())
        if keep_rows <= 0:
            continue
        accept_rate = keep_rows / total_rows
        if accept_rate < float(min_accept_rate) or accept_rate > float(max_accept_rate):
            continue
        keep_hit = float(labels.loc[keep_mask].mean())
        lift_pp = float((keep_hit - baseline_hit) * 100.0)
        candidate = {
            "threshold": float(threshold),
            "keep_rows": keep_rows,
            "total_rows": total_rows,
            "accept_rate": float(accept_rate),
            "hit_rate": keep_hit,
            "baseline_hit_rate": baseline_hit,
            "lift_pp": lift_pp,
        }
        if best is None:
            best = candidate
            continue
        if candidate["hit_rate"] > best["hit_rate"] + 1e-12:
            best = candidate
            continue
        if abs(candidate["hit_rate"] - best["hit_rate"]) <= 1e-12 and candidate["keep_rows"] > best["keep_rows"]:
            best = candidate
            continue
        if (
            abs(candidate["hit_rate"] - best["hit_rate"]) <= 1e-12
            and candidate["keep_rows"] == best["keep_rows"]
            and candidate["threshold"] < best["threshold"]
        ):
            best = candidate

    if best is None:
        # Fallback to identity/no-op threshold.
        best = {
            "threshold": 0.0,
            "keep_rows": total_rows,
            "total_rows": total_rows,
            "accept_rate": 1.0,
            "hit_rate": baseline_hit,
            "baseline_hit_rate": baseline_hit,
            "lift_pp": 0.0,
        }
    return best


def _next_month_token(month_token: str) -> str:
    dt = pd.to_datetime(f"{month_token}-01", errors="coerce")
    if pd.isna(dt):
        return month_token
    nxt = dt + pd.offsets.MonthBegin(1)
    return pd.Timestamp(nxt).strftime("%Y-%m")


def main() -> None:
    args = parse_args()
    rows_csv = args.rows_csv.resolve()
    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")

    df = pd.read_csv(rows_csv)
    required = {args.run_date_col, args.prob_col, args.result_col, args.target_col, args.direction_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Rows CSV missing required columns: {missing}")

    working = df.copy()
    working["run_date_ts"] = working[args.run_date_col].map(_normalize_run_date_token)
    working["label"] = np.where(working[args.result_col].astype(str).str.lower() == "win", 1.0, np.where(working[args.result_col].astype(str).str.lower() == "loss", 0.0, np.nan))
    working["prob"] = pd.to_numeric(working[args.prob_col], errors="coerce").clip(lower=0.0, upper=1.0)
    working["target"] = working[args.target_col].astype(str).str.upper().str.strip()
    working["direction"] = working[args.direction_col].astype(str).str.upper().str.strip()
    working = working.loc[working["run_date_ts"].notna() & working["label"].notna() & working["prob"].notna()].copy()
    if working.empty:
        raise RuntimeError("No resolved win/loss rows available after cleaning.")

    working["run_month"] = working["run_date_ts"].dt.strftime("%Y-%m")
    month_tokens = sorted(working["run_month"].dropna().unique().tolist())
    if args.include_next_month and month_tokens:
        month_tokens = sorted(set(month_tokens + [_next_month_token(month_tokens[-1])]))
    if not month_tokens:
        raise RuntimeError("No month tokens could be derived from run_date.")

    quantiles = [float(token.strip()) for token in str(args.threshold_quantiles).split(",") if token.strip()]
    if not quantiles:
        raise ValueError("threshold-quantiles must include at least one value.")

    months_payload: dict[str, dict[str, Any]] = {}
    report_months: list[dict[str, Any]] = []
    lookback_days = int(max(0, args.train_lookback_days))
    min_global = int(max(1, args.min_train_rows_global))
    min_segment = int(max(1, args.min_train_rows_segment))

    for month in month_tokens:
        month_start = pd.to_datetime(f"{month}-01", errors="coerce")
        if pd.isna(month_start):
            continue
        train = working.loc[working["run_date_ts"] < month_start].copy()
        if lookback_days > 0 and not train.empty:
            cutoff = month_start - pd.Timedelta(days=lookback_days)
            train = train.loc[train["run_date_ts"] >= cutoff].copy()
        if len(train) < min_global:
            report_months.append(
                {
                    "month": month,
                    "train_rows": int(len(train)),
                    "trained": False,
                    "reason": "insufficient_global_rows",
                }
            )
            continue

        global_fit = _fit_threshold(
            train,
            prob_col="prob",
            label_col="label",
            quantiles=quantiles,
            min_accept_rate=float(args.min_accept_rate),
            max_accept_rate=float(args.max_accept_rate),
        )
        if global_fit is None:
            report_months.append(
                {
                    "month": month,
                    "train_rows": int(len(train)),
                    "trained": False,
                    "reason": "global_fit_failed",
                }
            )
            continue

        segments: dict[str, dict[str, Any]] = {}
        grouped = train.groupby(["target", "direction"], dropna=False)
        for (target, direction), part in grouped:
            if len(part) < min_segment:
                continue
            seg_fit = _fit_threshold(
                part,
                prob_col="prob",
                label_col="label",
                quantiles=quantiles,
                min_accept_rate=float(args.min_accept_rate),
                max_accept_rate=float(args.max_accept_rate),
            )
            if seg_fit is None:
                continue
            key = f"{str(target).upper()}|{str(direction).upper()}"
            segments[key] = {
                "threshold": float(seg_fit["threshold"]),
                "train_rows": int(seg_fit["total_rows"]),
                "accept_rate": float(seg_fit["accept_rate"]),
                "hit_rate": float(seg_fit["hit_rate"]),
                "baseline_hit_rate": float(seg_fit["baseline_hit_rate"]),
                "lift_pp": float(seg_fit["lift_pp"]),
            }

        months_payload[month] = {
            "global": {
                "threshold": float(global_fit["threshold"]),
                "train_rows": int(global_fit["total_rows"]),
                "accept_rate": float(global_fit["accept_rate"]),
                "hit_rate": float(global_fit["hit_rate"]),
                "baseline_hit_rate": float(global_fit["baseline_hit_rate"]),
                "lift_pp": float(global_fit["lift_pp"]),
            },
            "segments": segments,
            "train_row_count": int(len(train)),
            "train_date_min": str(train["run_date_ts"].min().date()) if not train.empty else None,
            "train_date_max": str(train["run_date_ts"].max().date()) if not train.empty else None,
        }
        report_months.append(
            {
                "month": month,
                "train_rows": int(len(train)),
                "trained": True,
                "global_threshold": float(global_fit["threshold"]),
                "segment_count": int(len(segments)),
            }
        )

    board_size_median = int(
        np.round(
            pd.to_numeric(
                working.groupby("run_date_ts").size(),
                errors="coerce",
            ).median()
        )
    )
    board_size_median = max(1, board_size_median)

    payload = {
        "version": 1,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows_csv": str(rows_csv),
        "rows_used": int(len(working)),
        "prob_col": str(args.prob_col),
        "result_col": str(args.result_col),
        "target_col": str(args.target_col),
        "direction_col": str(args.direction_col),
        "run_date_col": str(args.run_date_col),
        "train_lookback_days": int(lookback_days),
        "min_train_rows_global": int(min_global),
        "min_train_rows_segment": int(min_segment),
        "min_accept_rate": float(args.min_accept_rate),
        "max_accept_rate": float(args.max_accept_rate),
        "recommended_min_rows": int(board_size_median),
        "months": months_payload,
    }

    report = {
        "payload_path": str(args.out_json.resolve()),
        "rows_csv": str(rows_csv),
        "rows_used": int(len(working)),
        "month_count": int(len(months_payload)),
        "months": report_months,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Learned gate payload: {args.out_json}")
    print(f"Training report:      {args.report_json}")
    print(f"Rows used:            {len(working)}")
    print(f"Months trained:       {len(months_payload)}")
    print(f"Recommended min rows: {board_size_median}")


if __name__ == "__main__":
    main()

