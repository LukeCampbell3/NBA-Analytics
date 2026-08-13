from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "governance"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import backtest_real_market_priors as backtest


def test_walk_forward_does_not_use_same_day_outcomes() -> None:
    rows = pd.DataFrame(
        [
            {"date": "2026-01-01", "target": "TB", "direction": "UNDER", "line": 1.5, "actual": 0, "model_probability": 0.76, "result": "win", "real_priced": False},
            {"date": "2026-01-02", "target": "TB", "direction": "UNDER", "line": 1.5, "actual": 2, "model_probability": 0.76, "result": "loss", "real_priced": True},
        ]
    )

    report = backtest.run_backtest(rows, minimum_rows=1, threshold=0.75)

    assert report["old_mixed_prior"]["selections"] == 1
    assert report["new_real_market_prior"]["selections"] == 1
    assert report["old_mixed_prior"]["mean_probability"] > report["new_real_market_prior"]["mean_probability"]
