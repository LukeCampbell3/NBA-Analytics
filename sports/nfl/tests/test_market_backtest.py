from __future__ import annotations

import pandas as pd

from sports.nfl.predictions.market_backtest import (
    evaluate_market_backtest,
    normalize_market_archive,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_display_name": ["A.J. Runner", "Quarter Back"],
            "season": [2024, 2024],
            "week": [1, 1],
            "target": ["rushing", "passing"],
            "position": ["RB", "QB"],
            "prediction": [70.0, 220.0],
            "actual": [80.0, 210.0],
        }
    )


def test_synthetic_and_post_start_lines_are_rejected() -> None:
    markets = pd.DataFrame(
        {
            "player": ["A J Runner", "A J Runner", "A J Runner"],
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "market": ["player_rush_yds"] * 3,
            "line": [65.5, 66.5, 67.5],
            "bookmaker": ["draftkings"] * 3,
            "source": ["synthetic_baseline", "the_odds_api", "the_odds_api"],
            "snapshot_time_utc": [
                "2024-09-08T15:00:00Z",
                "2024-09-08T18:00:00Z",
                "2024-09-08T15:00:00Z",
            ],
            "commence_time_utc": ["2024-09-08T17:00:00Z"] * 3,
        }
    )
    accepted, audit = normalize_market_archive(markets)
    assert len(accepted) == 1
    assert accepted.iloc[0]["line"] == 67.5
    assert audit["rejected_synthetic_rows"] == 1
    assert audit["rejected_at_or_after_start_rows"] == 1


def test_market_backtest_grades_hit_rate_pushes_and_real_prices() -> None:
    markets = pd.DataFrame(
        {
            "player": ["AJ Runner", "Quarter Back"],
            "season": [2024, 2024],
            "week": [1, 1],
            "market": ["player_rush_yds", "player_pass_yds"],
            "line": [65.5, 225.0],
            "over_price": [-110, -105],
            "under_price": [-110, -115],
            "bookmaker": ["draftkings", "fanduel"],
            "source": ["the_odds_api", "the_odds_api"],
            "snapshot_time_utc": ["2024-09-08T15:00:00Z"] * 2,
            "commence_time_utc": ["2024-09-08T17:00:00Z"] * 2,
        }
    )
    report, rows = evaluate_market_backtest(_predictions(), markets)
    assert report["overall"]["wins"] == 2
    assert report["overall"]["hit_rate"] == 1.0
    assert report["overall"]["priced_bets"] == 2
    assert set(rows["side"]) == {"over", "under"}
    assert report["promotion_gate"]["status"] == "failed"  # sample is intentionally tiny
