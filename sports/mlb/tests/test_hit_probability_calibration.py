from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hit_probability_calibration as calib  # noqa: E402


def test_chronological_holdout_split_takes_the_most_recent_dates_only():
    rows = pd.DataFrame(
        {
            "date": ["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"],
            "model_hit_probability": [0.6] * 6,
            "win": [1, 0, 1, 0, 1, 0],
        }
    )

    train, holdout, train_dates, holdout_dates = calib.chronological_holdout_split(
        rows, holdout_fraction=0.4, min_holdout_dates=2
    )

    # 5 distinct dates * 0.4 -> 2 holdout dates, the two most recent.
    assert holdout_dates == ["2026-08-04", "2026-08-05"]
    assert train_dates == ["2026-08-01", "2026-08-02", "2026-08-03"]
    assert set(holdout["date"]) == {"2026-08-04", "2026-08-05"}
    assert set(train["date"]) == {"2026-08-01", "2026-08-02", "2026-08-03"}
    # The real point of a chronological split: no row leaks across the boundary.
    assert set(train["date"]).isdisjoint(set(holdout["date"]))


def test_chronological_holdout_split_respects_min_holdout_dates_floor():
    rows = pd.DataFrame(
        {
            "date": ["2026-08-01", "2026-08-02", "2026-08-03"],
            "model_hit_probability": [0.6, 0.6, 0.6],
            "win": [1, 0, 1],
        }
    )

    _, _, train_dates, holdout_dates = calib.chronological_holdout_split(rows, holdout_fraction=0.1, min_holdout_dates=2)

    assert len(holdout_dates) == 2
    assert len(train_dates) == 1


def test_brier_score_is_zero_for_perfect_predictions():
    assert calib.brier_score(np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_brier_score_matches_hand_computation():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.7, 0.3])
    # (1-0.7)^2 = 0.09, (0-0.3)^2 = 0.09 -> mean 0.09
    assert calib.brier_score(y_true, y_pred) == pytest.approx(0.09)


def test_bucketed_hit_rates_reports_real_win_rate_per_bucket():
    rows = pd.DataFrame(
        {
            "model_hit_probability": [0.55, 0.58, 0.72, 0.74, 0.91],
            "win": [1, 0, 1, 1, 1],
        }
    )

    buckets = {entry["bucket"]: entry for entry in calib.bucketed_hit_rates(rows, prob_col="model_hit_probability")}

    assert buckets["0.55-0.60"]["n"] == 2
    assert buckets["0.55-0.60"]["real_hit_rate"] == pytest.approx(0.5)
    assert buckets["0.70-0.75"]["n"] == 2
    assert buckets["0.70-0.75"]["real_hit_rate"] == pytest.approx(1.0)
    assert buckets["0.90-0.95"]["n"] == 1
    assert buckets["0.00-0.50"]["n"] == 0
    assert buckets["0.00-0.50"]["real_hit_rate"] is None


def _synthetic_calibration_rows(n_per_date: int = 200) -> pd.DataFrame:
    """A real, honest-by-construction synthetic corpus: model_hit_
    probability is deliberately overconfident (true win rate is always
    10 points below what the probability claims, clipped to [0,1]) so a
    correctly-implemented isotonic fit should measurably improve holdout
    Brier score -- this is the property the real production wiring
    depends on, not just that the code runs without raising."""
    rng = np.random.default_rng(20260828)
    dates = [f"2026-08-{day:02d}" for day in range(1, 11)]
    rows = []
    for day in dates:
        probabilities = rng.uniform(0.5, 0.95, size=n_per_date)
        true_rate = np.clip(probabilities - 0.10, 0.0, 1.0)
        wins = rng.binomial(1, true_rate)
        for probability, win in zip(probabilities, wins):
            rows.append({"date": day, "target": "TB", "direction": "OVER", "model_hit_probability": float(probability), "win": int(win)})
    return pd.DataFrame(rows)


def test_train_hit_probability_calibration_reports_insufficient_data_below_the_real_row_floor(monkeypatch):
    monkeypatch.setattr(calib, "harvest_calibration_rows", lambda **_: (_synthetic_calibration_rows(n_per_date=5), {}))

    report = calib.train_hit_probability_calibration()

    assert report["status"] == "shadow"
    assert report["promotion_gate"]["decision"] == "shadow_insufficient_data"
    assert report["total_rows"] < calib.MIN_TRAINING_ROWS


def test_train_hit_probability_calibration_promotes_active_on_a_real_synthetic_improvement(monkeypatch):
    monkeypatch.setattr(calib, "harvest_calibration_rows", lambda **_: (_synthetic_calibration_rows(n_per_date=250), {}))

    report = calib.train_hit_probability_calibration(holdout_fraction=0.3)

    assert report["training_rows"] >= calib.MIN_TRAINING_ROWS
    assert report["holdout_rows"] >= calib.MIN_HOLDOUT_ROWS
    # The holdout dates must be strictly later than every training date --
    # this is the real leakage guard, not just a row count check.
    assert max(report["training_dates"]) < min(report["holdout_dates"])
    assert report["holdout_metrics"]["brier_calibrated"] < report["holdout_metrics"]["brier_raw"]
    assert report["status"] == "active"
    assert report["promotion_gate"]["decision"] == "active"
    assert len(report["breakpoints"]) > 0
    # Breakpoints must be a real, monotonically non-decreasing step function.
    ys = [point[1] for point in report["breakpoints"]]
    assert ys == sorted(ys)


def test_train_hit_probability_calibration_stays_shadow_when_holdout_is_too_thin(monkeypatch):
    # Enough total rows to clear MIN_TRAINING_ROWS, but almost all of them
    # land on a single date so the real holdout (later dates only) stays
    # thin -- the honest result is shadow, not a forced promotion.
    rng = np.random.default_rng(1)
    bulk = pd.DataFrame(
        {
            "date": ["2026-08-01"] * 1200,
            "target": ["TB"] * 1200,
            "direction": ["OVER"] * 1200,
            "model_hit_probability": rng.uniform(0.5, 0.9, size=1200),
            "win": rng.binomial(1, 0.6, size=1200),
        }
    )
    thin_holdout = pd.DataFrame(
        {
            "date": ["2026-08-02"] * 10,
            "target": ["TB"] * 10,
            "direction": ["OVER"] * 10,
            "model_hit_probability": [0.7] * 10,
            "win": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    rows = pd.concat([bulk, thin_holdout], ignore_index=True)
    monkeypatch.setattr(calib, "harvest_calibration_rows", lambda **_: (rows, {}))

    report = calib.train_hit_probability_calibration()

    assert report["holdout_rows"] < calib.MIN_HOLDOUT_ROWS
    assert report["status"] == "shadow"
    assert report["promotion_gate"]["decision"] == "shadow_insufficient_holdout"
