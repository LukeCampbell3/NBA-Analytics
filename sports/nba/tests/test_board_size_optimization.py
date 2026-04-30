from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from decision_engine.board_size_optimization import recommend_board_size, summarize_board_size_history, wilson_lower_bound


def test_wilson_lower_bound_respects_sample_size() -> None:
    small = wilson_lower_bound(6, 8)
    large = wilson_lower_bound(60, 80)
    assert large > small


def test_summarize_board_size_history_prefers_conservative_reliable_size() -> None:
    daily = pd.DataFrame(
        [
            {"run_date": "20260424", "board_size_requested": 4, "board_size_realized": 4, "resolved": 4, "wins": 3, "losses": 1, "units": 1.7272727, "expected_win_rate_mean": 0.57},
            {"run_date": "20260425", "board_size_requested": 4, "board_size_realized": 4, "resolved": 4, "wins": 3, "losses": 1, "units": 1.7272727, "expected_win_rate_mean": 0.57},
            {"run_date": "20260426", "board_size_requested": 4, "board_size_realized": 4, "resolved": 4, "wins": 2, "losses": 2, "units": -0.1818182, "expected_win_rate_mean": 0.56},
            {"run_date": "20260424", "board_size_requested": 8, "board_size_realized": 8, "resolved": 8, "wins": 4, "losses": 4, "units": -0.3636364, "expected_win_rate_mean": 0.55},
            {"run_date": "20260425", "board_size_requested": 8, "board_size_realized": 8, "resolved": 8, "wins": 5, "losses": 3, "units": 1.5454545, "expected_win_rate_mean": 0.55},
            {"run_date": "20260426", "board_size_requested": 8, "board_size_realized": 7, "resolved": 7, "wins": 3, "losses": 4, "units": -1.2727273, "expected_win_rate_mean": 0.54},
        ]
    )

    summary = summarize_board_size_history(daily, min_resolved_for_full_weight=12)
    recommendation = recommend_board_size(summary)

    assert set(summary["board_size_requested"].tolist()) == {4, 8}
    assert recommendation["recommended_board_size"] == 4
    size4 = summary.loc[summary["board_size_requested"] == 4].iloc[0]
    size8 = summary.loc[summary["board_size_requested"] == 8].iloc[0]
    assert float(size4["objective_score"]) > float(size8["objective_score"])
