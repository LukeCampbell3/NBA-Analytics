#!/usr/bin/env python3
"""
Fit a short-term target prediction calibrator from recent backtest rows.

The calibrator adjusts prediction-vs-market edges per target:
    calibrated_prediction = market_line + edge_bias + edge_multiplier * (prediction - market_line)

This is intentionally conservative and regularized toward the identity map so
we can tune short-term hit rate without fully retraining the model stack.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_upcoming_slate import DEFAULT_TARGET_PREDICTION_CALIBRATOR  # noqa: E402


TARGETS = ["PTS", "TRB", "AST"]
BIAS_BOUNDS = {
    "PTS": (-2.0, 1.0),
    "TRB": (-1.0, 0.5),
    "AST": (-1.0, 0.5),
}
MAX_ADJUSTMENT_ABS = {
    "PTS": 3.0,
    "TRB": 2.0,
    "AST": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit short-term target prediction calibration from recent backtest rows.")
    parser.add_argument("--rows-csv", type=Path, required=True, help="Recent long-format backtest rows CSV.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_TARGET_PREDICTION_CALIBRATOR,
        help="Output calibrator JSON.",
    )
    parser.add_argument(
        "--multiplier-penalty",
        type=float,
        default=0.02,
        help="Regularization penalty applied to |edge_multiplier - 1| during grid search.",
    )
    parser.add_argument(
        "--bias-penalty",
        type=float,
        default=0.015,
        help="Regularization penalty applied to |edge_bias| during grid search.",
    )
    parser.add_argument(
        "--support-scale",
        type=float,
        default=120.0,
        help="Rows scale used when converting target support into a calibration confidence weight.",
    )
    parser.add_argument(
        "--protect-target",
        nargs="*",
        default=[],
        help="Targets forced to identity calibration (for example: AST).",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_result(prediction: float, market_line: float, actual: float) -> str:
    if not np.isfinite(prediction) or not np.isfinite(market_line) or not np.isfinite(actual):
        return "missing"
    if abs(actual - market_line) <= 1e-9:
        return "push"
    if abs(prediction - market_line) <= 1e-9:
        return "push"
    if prediction > market_line:
        return "win" if actual > market_line else "loss"
    return "win" if actual < market_line else "loss"


def fit_target_params(
    frame: pd.DataFrame,
    target: str,
    *,
    multiplier_penalty: float,
    bias_penalty: float,
    support_scale: float,
    protected: bool,
) -> dict:
    resolved = frame.loc[frame["result"].isin(["win", "loss"])].copy()
    baseline_hit_rate = float((resolved["result"] == "win").mean()) if not resolved.empty else np.nan
    rows = int(len(resolved))
    if resolved.empty:
        return {
            "enabled": False,
            "edge_multiplier": 1.0,
            "edge_bias": 0.0,
            "support_rows": 0,
            "support_weight": 0.0,
            "train_hit_rate_raw": np.nan,
            "train_hit_rate_tuned": np.nan,
            "train_delta_hit_rate_pp": np.nan,
            "source": "empty_rows",
        }

    if protected:
        return {
            "enabled": True,
            "edge_multiplier": 1.0,
            "edge_bias": 0.0,
            "support_rows": rows,
            "support_weight": 0.0,
            "train_hit_rate_raw": baseline_hit_rate,
            "train_hit_rate_tuned": baseline_hit_rate,
            "train_delta_hit_rate_pp": 0.0,
            "source": "protected_identity",
        }

    multipliers = np.round(np.arange(0.40, 1.01, 0.05), 4)
    bias_low, bias_high = BIAS_BOUNDS[target]
    biases = np.round(np.arange(bias_low, bias_high + 0.001, 0.25), 4)

    best_payload: dict | None = None
    for multiplier in multipliers:
        for bias in biases:
            tuned_pred = resolved["market_line"] + float(bias) + float(multiplier) * (resolved["prediction"] - resolved["market_line"])
            tuned_pred = resolved["prediction"] + np.clip(
                tuned_pred - resolved["prediction"],
                -MAX_ADJUSTMENT_ABS[target],
                MAX_ADJUSTMENT_ABS[target],
            )
            tuned_result = [
                classify_result(prediction=float(pred), market_line=float(line), actual=float(actual))
                for pred, line, actual in zip(tuned_pred, resolved["market_line"], resolved["actual"])
            ]
            tuned_win_mask = pd.Series(tuned_result, index=resolved.index).eq("win")
            tuned_loss_mask = pd.Series(tuned_result, index=resolved.index).eq("loss")
            tuned_count = int((tuned_win_mask | tuned_loss_mask).sum())
            if tuned_count <= 0:
                continue
            tuned_hit_rate = float(tuned_win_mask.sum() / tuned_count)
            score = (
                tuned_hit_rate
                - float(multiplier_penalty) * abs(float(multiplier) - 1.0)
                - float(bias_penalty) * abs(float(bias))
            )
            candidate = {
                "score": score,
                "edge_multiplier": float(multiplier),
                "edge_bias": float(bias),
                "train_hit_rate_tuned": tuned_hit_rate,
            }
            if best_payload is None or float(candidate["score"]) > float(best_payload["score"]):
                best_payload = candidate

    assert best_payload is not None
    support_weight = float(np.clip(rows / max(1.0, rows + float(support_scale)), 0.0, 1.0))
    tuned_multiplier = float(1.0 + support_weight * (float(best_payload["edge_multiplier"]) - 1.0))
    tuned_bias = float(support_weight * float(best_payload["edge_bias"]))

    tuned_pred = resolved["market_line"] + tuned_bias + tuned_multiplier * (resolved["prediction"] - resolved["market_line"])
    tuned_pred = resolved["prediction"] + np.clip(
        tuned_pred - resolved["prediction"],
        -MAX_ADJUSTMENT_ABS[target],
        MAX_ADJUSTMENT_ABS[target],
    )
    tuned_result = [
        classify_result(prediction=float(pred), market_line=float(line), actual=float(actual))
        for pred, line, actual in zip(tuned_pred, resolved["market_line"], resolved["actual"])
    ]
    tuned_hit_rate = float(pd.Series(tuned_result).eq("win").mean())
    delta_pp = float((tuned_hit_rate - baseline_hit_rate) * 100.0) if baseline_hit_rate == baseline_hit_rate else np.nan

    return {
        "enabled": True,
        "edge_multiplier": tuned_multiplier,
        "edge_bias": tuned_bias,
        "support_rows": rows,
        "support_weight": support_weight,
        "train_hit_rate_raw": baseline_hit_rate,
        "train_hit_rate_tuned": tuned_hit_rate,
        "train_delta_hit_rate_pp": delta_pp,
        "max_adjustment_abs": float(MAX_ADJUSTMENT_ABS[target]),
        "source": "grid_search_regularized",
        "raw_best_multiplier": float(best_payload["edge_multiplier"]),
        "raw_best_bias": float(best_payload["edge_bias"]),
        "raw_best_score": float(best_payload["score"]),
    }


def main() -> None:
    args = parse_args()
    rows_path = args.rows_csv.resolve()
    if not rows_path.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_path}")
    rows_df = pd.read_csv(rows_path)
    required = {"target", "prediction", "market_line", "actual", "result"}
    missing = sorted(required - set(rows_df.columns))
    if missing:
        raise ValueError(f"Rows CSV is missing required columns: {missing}")

    rows_df["target"] = rows_df["target"].astype(str).str.upper().str.strip()
    rows_df = rows_df.loc[rows_df["target"].isin(TARGETS)].copy()
    protected_targets = {str(item).upper().strip() for item in args.protect_target}

    targets_payload: dict[str, dict] = {}
    summary_rows: list[dict] = []
    for target in TARGETS:
        target_frame = rows_df.loc[rows_df["target"] == target].copy()
        payload = fit_target_params(
            target_frame,
            target,
            multiplier_penalty=float(args.multiplier_penalty),
            bias_penalty=float(args.bias_penalty),
            support_scale=float(args.support_scale),
            protected=target in protected_targets,
        )
        targets_payload[target] = payload
        summary_rows.append(
            {
                "target": target,
                "support_rows": int(payload.get("support_rows", 0)),
                "edge_multiplier": float(payload.get("edge_multiplier", 1.0)),
                "edge_bias": float(payload.get("edge_bias", 0.0)),
                "train_hit_rate_raw": float(payload.get("train_hit_rate_raw", np.nan)),
                "train_hit_rate_tuned": float(payload.get("train_hit_rate_tuned", np.nan)),
                "train_delta_hit_rate_pp": float(payload.get("train_delta_hit_rate_pp", np.nan)),
                "source": str(payload.get("source", "")),
            }
        )

    summary_df = pd.DataFrame.from_records(summary_rows).sort_values("target").reset_index(drop=True)
    payload = {
        "artifact_type": "short_term_target_prediction_calibrator",
        "created_at_utc": utc_now_iso(),
        "rows_csv": str(rows_path),
        "multiplier_penalty": float(args.multiplier_penalty),
        "bias_penalty": float(args.bias_penalty),
        "support_scale": float(args.support_scale),
        "protect_target": sorted(protected_targets),
        "targets": targets_payload,
        "summary": summary_df.to_dict(orient="records"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("TARGET PREDICTION CALIBRATOR FIT")
    print("=" * 88)
    print(f"Rows CSV:  {rows_path}")
    print(f"Out JSON:  {args.out_json}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
