from __future__ import annotations

import numpy as np
import pandas as pd

from sports.nfl.predictions.market_selector import (
    build_prediction_pool,
    build_weekly_validation,
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


def test_prediction_pool_and_weekly_validation_show_pass_fail_and_warmup() -> None:
    rows = pd.DataFrame(
        {
            "season": [2021, 2021],
            "week": [11, 11],
            "player_id": ["p1", "p2"],
            "player_display_name": ["Winner", "Loser"],
            "target": ["passing", "passing"],
            "eligible": [True, True],
            "side": ["under", "over"],
            "line": [250.5, 250.5],
            "selected_price": [-110, -110],
            "estimated_side_probability": [0.61, 0.59],
            "no_vig_side_probability": [0.5, 0.5],
            "probability_advantage": [0.11, 0.09],
            "actual": [200.0, 200.0],
            "result": ["win", "loss"],
            "profit_units": [100 / 110, -1.0],
        }
    )
    pool = build_prediction_pool(
        rows,
        evaluation_split="development_walk_forward",
        architecture_by_target={"passing": "regularized_logistic_raw"},
        promotion_by_target={"passing": "passed"},
    )
    assert pool["pick_validation"].tolist() == ["fail", "pass"]
    weekly = build_weekly_validation(
        [pool],
        season_weeks={2021: [1, 11]},
        promotion_by_target={"passing": "passed"},
        development_season=2021,
    )
    overall = weekly.loc[weekly["target"].eq("overall")].set_index("week")
    assert overall.loc[1, "pool_status"] == "calibration_warmup"
    assert overall.loc[11, "picks"] == 2
    assert overall.loc[11, "wins"] == 1
    assert overall.loc[11, "losses"] == 1
