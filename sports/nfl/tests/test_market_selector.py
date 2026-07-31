from __future__ import annotations

import numpy as np
import pandas as pd

from sports.nfl.predictions.market_selector import (
    score_probabilities,
    summarize_market_rows,
    target_promotion_gate,
)


def test_probability_selector_uses_one_side_and_price_aware_abstention() -> None:
    frame = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "actual": [280.0, 210.0, 250.0],
            "line": [250.5, 250.5, 250.5],
            "over_price": [-110, -110, -110],
            "under_price": [-110, -110, -110],
        }
    )
    scored = score_probabilities(frame, np.array([0.62, 0.35, 0.54]))
    assert scored["side"].tolist() == ["over", "under", "over"]
    assert scored["eligible"].tolist() == [True, True, False]
    assert scored["result"].tolist() == ["win", "win", "loss"]


def test_target_gate_requires_precision_volume_weeks_and_roi() -> None:
    rows = pd.DataFrame(
        {
            "eligible": [True] * 160,
            "result": ["win"] * 96 + ["loss"] * 64,
            "profit_units": [100 / 110] * 96 + [-1.0] * 64,
            "season": [2024] * 160,
            "week": np.tile(np.arange(1, 17), 10),
            "side": ["under"] * 160,
        }
    )
    summary = summarize_market_rows(rows)
    assert summary["hit_rate"] == 0.6
    assert summary["roi"] > 0
    assert target_promotion_gate(summary)["status"] == "passed"
