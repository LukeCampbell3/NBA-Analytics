#!/usr/bin/env python3
"""
Generate reliability diagnostics for selected-board probabilities.

Outputs:
- overall metrics
- segment metrics (target x direction)
- board-size regime metrics (short vs full)
- monthly metrics
- monthly segment metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from decision_engine.selected_board_calibration import evaluate_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report reliability diagnostics for selected-board probabilities.")
    parser.add_argument("--rows-csv", type=Path, required=True, help="Row-level replay/production CSV with resolved outcomes.")
    parser.add_argument("--run-date-col", type=str, default="run_date")
    parser.add_argument("--mode-col", type=str, default="mode")
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--direction-col", type=str, default="direction")
    parser.add_argument("--result-col", type=str, default="result")
    parser.add_argument("--prob-col", type=str, default="expected_win_rate")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "reliability_diagnostics",
        help="Output prefix path (without extension).",
    )
    return parser.parse_args()


def _fit_logistic_calibration(y: np.ndarray, p: np.ndarray, max_iter: int = 100, tol: float = 1e-7) -> tuple[float, float]:
    """
    Fit logistic calibration model:
      logit(P(Y=1)) = intercept + slope * logit(p)
    via Newton-Raphson on two parameters.
    """
    if len(y) < 25:
        return np.nan, np.nan
    y = np.asarray(y, dtype="float64")
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1.0 - 1e-6)
    x = np.log(p / (1.0 - p))
    X = np.column_stack([np.ones_like(x), x])

    beta = np.array([0.0, 1.0], dtype="float64")
    for _ in range(max_iter):
        z = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        w = np.clip(mu * (1.0 - mu), 1e-8, None)
        grad = X.T @ (y - mu)
        H = -(X.T @ (X * w[:, None]))
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def _compute_metrics(df: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "hit_rate": np.nan,
            "mean_prob": np.nan,
            "gap_pp": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "ece_10": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }
    probs = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0).to_numpy(dtype="float64")
    labels = (df["__is_win__"] == 1.0).astype("float64").to_numpy(dtype="float64")
    basic = evaluate_calibration(probs=probs, labels=labels)
    intercept, slope = _fit_logistic_calibration(labels, probs)
    return {
        "rows": int(len(df)),
        "hit_rate": float(np.mean(labels)),
        "mean_prob": float(np.mean(probs)),
        "gap_pp": float((np.mean(labels) - np.mean(probs)) * 100.0),
        "brier": float(basic["brier"]),
        "log_loss": float(basic["log_loss"]),
        "ece_10": float(basic["ece_10"]),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _group_rows(df: pd.DataFrame, keys: list[str], prob_col: str, group_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(keys, dropna=False) if keys else [((), df)]
    for key, part in grouped:
        metrics = _compute_metrics(part, prob_col=prob_col)
        row = {"group_type": group_name}
        if keys:
            if not isinstance(key, tuple):
                key = (key,)
            for k, v in zip(keys, key):
                row[k] = v
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rows_csv = args.rows_csv.resolve()
    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")

    df = pd.read_csv(rows_csv)
    required = {args.run_date_col, args.target_col, args.direction_col, args.result_col, args.prob_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    token_dates = pd.to_datetime(work[args.run_date_col].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    generic_dates = pd.to_datetime(work[args.run_date_col], errors="coerce")
    work["__run_date__"] = token_dates.fillna(generic_dates)
    work["__month__"] = work["__run_date__"].dt.strftime("%Y-%m")
    work["__target__"] = work[args.target_col].astype(str).str.upper().str.strip()
    work["__direction__"] = work[args.direction_col].astype(str).str.upper().str.strip()
    work["__segment__"] = work["__target__"] + "_" + work["__direction__"]
    work["__prob__"] = pd.to_numeric(work[args.prob_col], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    work["__result__"] = work[args.result_col].astype(str).str.lower().str.strip()
    work = work.loc[work["__result__"].isin(["win", "loss"])].copy()
    if work.empty:
        raise RuntimeError("No resolved win/loss rows available.")
    work["__is_win__"] = (work["__result__"] == "win").astype("float64")

    if args.mode_col in work.columns:
        work["__mode__"] = work[args.mode_col].astype(str).str.strip()
    else:
        work["__mode__"] = "all"

    # Board-size regime: full = max board size observed for the mode, else short.
    board_sizes = (
        work.groupby(["__mode__", "__run_date__"], dropna=False)
        .size()
        .rename("board_size")
        .reset_index()
    )
    max_size = board_sizes.groupby("__mode__", dropna=False)["board_size"].transform("max")
    board_sizes["board_regime"] = np.where(board_sizes["board_size"] >= max_size, "full", "short")
    work = work.merge(board_sizes, how="left", on=["__mode__", "__run_date__"])

    overall = _group_rows(work, keys=[], prob_col="__prob__", group_name="overall")
    by_segment = _group_rows(work, keys=["__segment__"], prob_col="__prob__", group_name="segment")
    by_board_regime = _group_rows(work, keys=["__mode__", "board_regime"], prob_col="__prob__", group_name="board_regime")
    by_month = _group_rows(work, keys=["__month__"], prob_col="__prob__", group_name="month")
    by_month_segment = _group_rows(
        work,
        keys=["__month__", "__segment__"],
        prob_col="__prob__",
        group_name="month_segment",
    )

    out_prefix = args.out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    overall_csv = out_prefix.with_name(f"{out_prefix.name}_overall.csv")
    segment_csv = out_prefix.with_name(f"{out_prefix.name}_segment.csv")
    board_csv = out_prefix.with_name(f"{out_prefix.name}_board_regime.csv")
    month_csv = out_prefix.with_name(f"{out_prefix.name}_month.csv")
    month_segment_csv = out_prefix.with_name(f"{out_prefix.name}_month_segment.csv")
    summary_json = out_prefix.with_name(f"{out_prefix.name}_summary.json")

    overall.to_csv(overall_csv, index=False)
    by_segment.to_csv(segment_csv, index=False)
    by_board_regime.to_csv(board_csv, index=False)
    by_month.to_csv(month_csv, index=False)
    by_month_segment.to_csv(month_segment_csv, index=False)

    payload = {
        "rows_csv": str(rows_csv),
        "resolved_rows": int(len(work)),
        "prob_col": args.prob_col,
        "outputs": {
            "overall_csv": str(overall_csv),
            "segment_csv": str(segment_csv),
            "board_regime_csv": str(board_csv),
            "month_csv": str(month_csv),
            "month_segment_csv": str(month_segment_csv),
        },
        "overall": overall.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Overall CSV:       {overall_csv}")
    print(f"Segment CSV:       {segment_csv}")
    print(f"Board Regime CSV:  {board_csv}")
    print(f"Month CSV:         {month_csv}")
    print(f"Month Segment CSV: {month_segment_csv}")
    print(f"Summary JSON:      {summary_json}")
    if not overall.empty:
        print("\nOverall metrics:")
        print(overall.to_string(index=False))


if __name__ == "__main__":
    main()

