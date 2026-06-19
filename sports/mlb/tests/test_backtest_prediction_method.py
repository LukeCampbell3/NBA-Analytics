from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import backtest_prediction_method as backtest


def test_grade_respects_selected_direction() -> None:
    assert backtest.grade(2.0, 1.5, "OVER") == "win"
    assert backtest.grade(2.0, 1.5, "UNDER") == "loss"
    assert backtest.grade(1.0, 1.0, "UNDER") == "push"


def test_history_before_excludes_evaluation_and_future_dates() -> None:
    universe = pd.DataFrame(
        [
            {
                "_date": pd.Timestamp("2026-04-01"),
                "Edge": 1.0,
                "Target": "TB",
                "Market_Line": 1.5,
                "Actual": 2.0,
                "Prediction": 2.5,
                "Market_Source": "synthetic",
                "Market_Books": 0,
            },
            {
                "_date": pd.Timestamp("2026-04-02"),
                "Edge": -1.0,
                "Target": "TB",
                "Market_Line": 1.5,
                "Actual": 0.0,
                "Prediction": 0.5,
                "Market_Source": "synthetic",
                "Market_Books": 0,
            },
        ]
    )
    history = backtest.history_before(universe, pd.Timestamp("2026-04-02"))
    assert history["_date"].max() < pd.Timestamp("2026-04-02")
    assert len(history) == 1

    calibration, _ = backtest.prior_payload(history, pd.Timestamp("2026-04-02"))
    assert calibration["line_buckets"]["TB|OVER|1.5"]["wins"] == 1


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = backtest.wilson_interval(20, 16)
    assert low is not None and high is not None
    assert low < (20 / 36) < high


def test_raw_calibration_audit_exposes_top_bucket_overconfidence() -> None:
    frame = pd.DataFrame(
        [
            {"Edge": 4.0, "Actual": 0.0, "Market_Line": 0.5, "Prediction": 4.5},
            {"Edge": 4.0, "Actual": 2.0, "Market_Line": 0.5, "Prediction": 4.5},
        ]
    )
    audit = backtest.raw_calibration_audit(frame)
    assert audit["top_bucket"]["graded"] == 2
    assert audit["top_bucket"]["actual_hit_rate"] == 0.5
    assert audit["top_bucket"]["calibration_gap"] > 0.4
