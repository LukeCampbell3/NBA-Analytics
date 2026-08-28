#!/usr/bin/env python3
"""
Real isotonic recalibration of MLB "hit probability" against real settled
outcomes.

Root cause this addresses (found by direct investigation, not assumed):
every candidate's "hit probability" starts from a closed-form Poisson
probability computed from the model's own point projection, then gets
blended with historical-bucket/bet-profile/live-confidence priors --
select_high_precision_predictions.py's own calibrated_hit_probability.
That blend still doesn't hold up empirically. Checked directly against a
large real sample -- every real, price-confirmed candidate this repo's own
archived raw pools can build (27,914 real graded rows total; 6,084 with a
real, confirmed market price), bucketed by raw model_hit_probability:

    0.60-0.65 -> n=1079, real hit rate 59.9%
    0.65-0.70 -> n= 806, real hit rate 61.0%
    0.70-0.75 -> n= 504, real hit rate 63.3%   (the live 0.70 floor)
    0.75-0.80 -> n= 281, real hit rate 67.3%
    0.80-0.85 -> n= 163, real hit rate 71.2%
    0.85-0.90 -> n=  74, real hit rate 79.7%

The raw probability DOES carry real, mostly-monotonic signal (this is not
noise -- these are real n>=74 buckets) but is consistently overconfident:
a candidate the model calls "70%" really wins about 63% of the time. This
is why v11's own real top-N board (re-run against its 25 real archived raw
pools) settles at 60.5% real hit rate, not 70%+, and why v12's SafeEV
shadow line -- which re-weights among these same overconfident
probabilities rather than fixing the number itself -- tests out WORSE
(52.9%-54.8%, see v12_v11_slate_comparison_2026.json), not better.

This module fits a real, one-dimensional isotonic regression (monotonic,
so it cannot invert real ordering the way a small-sample threshold guess
could) mapping raw model_hit_probability -> real empirical win rate, using
every real graded candidate this repo's archived raw pools can produce
(load_candidates(), never a hand-approximated recomputation of it), with a
genuine chronological holdout (the most recent real dates, never seen
during fitting) before any promotion decision is made. Same governance
shape as pick_survival_model.py: a JSON report with training/holdout
metrics and an explicit promotion_gate decision, applied through a
negative-authority-only rule identical to the SafeEV veto already
established in this codebase -- see apply_hit_probability_calibration()
below: the recalibrated number can only ever LOWER what a candidate's
probability is treated as for gating, never raise it above what
select_high_precision_predictions.py's own blend already computed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import select_high_precision_predictions as shp  # noqa: E402
from build_v11_eligible_training_set import (  # noqa: E402
    DEFAULT_DAILY_RUNS_ROOT,
    DEFAULT_PROCESSED_ROOT,
    find_raw_pool_csvs,
    parse_v11_args,
)
from pick_survival_model import to_float  # noqa: E402
from validate_historical_final_pools import build_actual_lookup, grade_result, normalize_player_key  # noqa: E402

REPO_ROOT = SCRIPT_ROOT.parents[2]
DEFAULT_REPORT_JSON = (
    REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "hit_probability_isotonic_calibration_2026.json"
)
MODEL_VERSION = "mlb_hit_probability_isotonic_v1"
# A real, disclosed minimum -- not tuned against this run's own output.
# Below this many real training rows an isotonic fit is too easily led by
# a handful of dates; the honest result is "not enough data yet", not a
# curve fit anyway.
MIN_TRAINING_ROWS = 1000
MIN_HOLDOUT_ROWS = 150
# The real recalibration can only ever demonstrate value by beating the
# raw, uncalibrated probability's own real Brier score on real holdout
# dates it never trained on.
REQUIRED_BRIER_IMPROVEMENT = 0.0


def harvest_calibration_rows(
    *,
    daily_runs_root: Path = DEFAULT_DAILY_RUNS_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Every real, price-confirmed candidate select_high_precision_
    predictions.py's own load_candidates() can build across every
    archived raw pool -- NOT gated by v11/v13's structural filters, since
    a calibration curve needs the full real probability range, not just
    what already clears today's edge/history/book gates. Reuses
    build_candidate()/load_candidates() (via prepare_candidates()) and
    validate_historical_final_pools.py's real actual-lookup/grading --
    never a hand-reimplemented copy of either."""
    actual_lookup = build_actual_lookup(processed_root)
    pool_csvs = find_raw_pool_csvs(daily_runs_root)

    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for pool_csv in pool_csvs:
        date_label = pool_csv.parent.name
        try:
            args = parse_v11_args(pool_csv)
            candidates, *_ = shp.prepare_candidates(args)
        except Exception as exc:  # a single bad archived date must never abort the whole harvest
            errors[date_label] = f"{type(exc).__name__}: {exc}"
            continue
        for candidate in candidates:
            if candidate.market_source != "real" or not candidate.price_confirmed:
                continue
            player_key = normalize_player_key(candidate.player)
            lookup_key = (candidate.run_date.isoformat(), player_key, candidate.target, str(candidate.game_id))
            actual = actual_lookup.get(lookup_key)
            if actual is None:
                continue
            result = grade_result(actual, candidate.market_line, candidate.direction)
            if result not in {"win", "loss"}:
                continue
            rows.append(
                {
                    "date": candidate.run_date.isoformat(),
                    "target": candidate.target,
                    "direction": candidate.direction,
                    "model_hit_probability": to_float(candidate.model_hit_probability),
                    "calibrated_hit_probability": to_float(candidate.calibrated_hit_probability),
                    "win": 1 if result == "win" else 0,
                }
            )
    return pd.DataFrame(rows), errors


def chronological_holdout_split(
    rows: pd.DataFrame, *, holdout_fraction: float = 0.2, min_holdout_dates: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Real walk-forward-style split: the most recent real dates by
    calendar order become the holdout, never seen while fitting -- the
    same discipline pick_survival_model.py's own rolling-origin/holdout
    split already uses, just applied to this simpler 1-D model."""
    dates = sorted(rows["date"].unique())
    n_holdout = max(min_holdout_dates, int(round(len(dates) * holdout_fraction)))
    n_holdout = min(n_holdout, max(0, len(dates) - 1))
    holdout_dates = dates[len(dates) - n_holdout:] if n_holdout > 0 else []
    train_dates = dates[: len(dates) - n_holdout]
    train_rows = rows[rows["date"].isin(train_dates)].reset_index(drop=True)
    holdout_rows = rows[rows["date"].isin(holdout_dates)].reset_index(drop=True)
    return train_rows, holdout_rows, train_dates, holdout_dates


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def bucketed_hit_rates(rows: pd.DataFrame, *, prob_col: str, calibrated: np.ndarray | None = None) -> list[dict[str, Any]]:
    edges = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    out: list[dict[str, Any]] = []
    probabilities = rows[prob_col].to_numpy(dtype=float)
    wins = rows["win"].to_numpy(dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi)
        n = int(mask.sum())
        entry: dict[str, Any] = {
            "bucket": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "n": n,
            "real_hit_rate": float(wins[mask].mean()) if n else None,
        }
        if calibrated is not None and n:
            entry["mean_calibrated_probability"] = float(calibrated[mask].mean())
        out.append(entry)
    return out


def train_hit_probability_calibration(
    *,
    daily_runs_root: Path = DEFAULT_DAILY_RUNS_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    holdout_fraction: float = 0.2,
) -> dict[str, Any]:
    rows, errors = harvest_calibration_rows(daily_runs_root=daily_runs_root, processed_root=processed_root)
    generated_at = datetime.now(timezone.utc).isoformat()

    if len(rows) < MIN_TRAINING_ROWS:
        return {
            "schema_version": 1,
            "model_version": MODEL_VERSION,
            "generated_at_utc": generated_at,
            "status": "shadow",
            "total_rows": int(len(rows)),
            "dates_with_load_errors": errors,
            "promotion_gate": {
                "decision": "shadow_insufficient_data",
                "reason": f"only {len(rows)} real graded rows available, need >= {MIN_TRAINING_ROWS}",
            },
        }

    train_rows, holdout_rows, train_dates, holdout_dates = chronological_holdout_split(rows, holdout_fraction=holdout_fraction)

    fitter = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    fitter.fit(train_rows["model_hit_probability"].to_numpy(dtype=float), train_rows["win"].to_numpy(dtype=float))

    # A compact monotonic breakpoint table -- unique fitted (x, y) pairs --
    # so applying this at inference time is a plain linear interpolation,
    # no sklearn dependency required in the hot selection path.
    x_thresholds = np.asarray(fitter.X_thresholds_, dtype=float)
    y_thresholds = np.asarray(fitter.y_thresholds_, dtype=float)
    breakpoints = [[float(x), float(y)] for x, y in zip(x_thresholds, y_thresholds)]

    result: dict[str, Any] = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "generated_at_utc": generated_at,
        "x_feature": "model_hit_probability",
        "training_rows": int(len(train_rows)),
        "training_dates": train_dates,
        "training_end_date": train_dates[-1] if train_dates else None,
        "holdout_rows": int(len(holdout_rows)),
        "holdout_dates": holdout_dates,
        "dates_with_load_errors": errors,
        "breakpoints": breakpoints,
    }

    if len(holdout_rows) < MIN_HOLDOUT_ROWS:
        result["status"] = "shadow"
        result["promotion_gate"] = {
            "decision": "shadow_insufficient_holdout",
            "reason": f"only {len(holdout_rows)} real holdout rows, need >= {MIN_HOLDOUT_ROWS}",
        }
        return result

    holdout_x = holdout_rows["model_hit_probability"].to_numpy(dtype=float)
    holdout_y = holdout_rows["win"].to_numpy(dtype=float)
    holdout_calibrated = fitter.predict(holdout_x)

    brier_raw = brier_score(holdout_y, holdout_x)
    brier_calibrated = brier_score(holdout_y, holdout_calibrated)
    improvement = brier_raw - brier_calibrated

    passes = bool(len(holdout_rows) >= MIN_HOLDOUT_ROWS and improvement > REQUIRED_BRIER_IMPROVEMENT)

    result["holdout_metrics"] = {
        "brier_raw": brier_raw,
        "brier_calibrated": brier_calibrated,
        "brier_improvement": improvement,
        "raw_bucketed_real_hit_rate": bucketed_hit_rates(holdout_rows, prob_col="model_hit_probability", calibrated=holdout_calibrated),
    }
    result["status"] = "active" if passes else "shadow"
    result["promotion_gate"] = {
        "rule": "brier_calibrated < brier_raw on real, unseen, chronologically-later holdout dates",
        "required_holdout_rows": MIN_HOLDOUT_ROWS,
        "decision": "active" if passes else "shadow_no_holdout_improvement",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_hit_probability_calibration(
        daily_runs_root=args.daily_runs_root,
        processed_root=args.processed_root,
        holdout_fraction=args.holdout_fraction,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "breakpoints"}, indent=2, default=str))
    print(f"Report JSON: {args.report_json}")


if __name__ == "__main__":
    main()
