from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports.mlb.governance.backtest_policy_thesis import (
    load_quotes,
    score_policy,
    select_daily,
    settle_two_leg_parlay,
)


def test_load_quotes_keeps_latest_supported_pregame_price(tmp_path: Path) -> None:
    path = tmp_path / "lines.csv"
    pd.DataFrame(
        [
            {
                "fetched_at_utc": "2026-06-20T12:00:00Z",
                "event_id": "event-1",
                "commence_time_utc": "2026-06-20T23:00:00Z",
                "event_date_et": "2026-06-20",
                "bookmaker_key": "draftkings",
                "market_key": "batter_total_bases",
                "player_name_norm": "Test_Player",
                "line": 1.5,
                "over_price": 110,
            },
            {
                "fetched_at_utc": "2026-06-20T18:00:00Z",
                "event_id": "event-1",
                "commence_time_utc": "2026-06-20T23:00:00Z",
                "event_date_et": "2026-06-20",
                "bookmaker_key": "draftkings",
                "market_key": "batter_total_bases",
                "player_name_norm": "Test_Player",
                "line": 1.5,
                "over_price": 120,
            },
            {
                "fetched_at_utc": "2026-06-21T01:00:00Z",
                "event_id": "event-1",
                "commence_time_utc": "2026-06-20T23:00:00Z",
                "event_date_et": "2026-06-20",
                "bookmaker_key": "fanduel",
                "market_key": "batter_total_bases",
                "player_name_norm": "Test_Player",
                "line": 1.5,
                "over_price": 130,
            },
        ]
    ).to_csv(path, index=False)

    quotes, summary, diagnostics = load_quotes(path)

    assert len(quotes) == 1
    assert summary.iloc[0]["best_american_price"] == 120
    assert diagnostics["raw_rows"] == 3
    assert diagnostics["single_acquisition_snapshots"] == 1


def test_daily_selector_enforces_distinct_players_and_games() -> None:
    pool = pd.DataFrame(
        [
            {"date": "2026-06-20", "player_id": "a", "game_id": "g1", "target": "TB", "line": 1.5, "policy_score": 0.9, "best_decimal_price": 2.0},
            {"date": "2026-06-20", "player_id": "b", "game_id": "g1", "target": "R", "line": 0.5, "policy_score": 0.8, "best_decimal_price": 2.0},
            {"date": "2026-06-20", "player_id": "a", "game_id": "g2", "target": "R", "line": 0.5, "policy_score": 0.7, "best_decimal_price": 2.0},
            {"date": "2026-06-20", "player_id": "c", "game_id": "g3", "target": "R", "line": 0.5, "policy_score": 0.6, "best_decimal_price": 2.0},
        ]
    )

    selected = select_daily(pool)

    assert selected["player_id"].tolist() == ["a", "c"]
    assert selected["game_id"].nunique() == len(selected)


def test_policy_score_counts_abstention_as_zero_calendar_return() -> None:
    rows = pd.DataFrame(
        [
            {"date": "2026-06-20", "result": "win", "unit_return": 1.0},
            {"date": "2026-06-21", "result": "loss", "unit_return": -1.0},
        ]
    )
    eligible = pd.DataFrame(
        [
            {"date": "2026-06-20"},
            {"date": "2026-06-21"},
            {"date": "2026-06-22"},
        ]
    )

    result = score_policy(rows, eligible, ["2026-06-20", "2026-06-21", "2026-06-22"], alpha=0.05)

    assert result["calendar_slate_return"] == 0.0
    assert result["slate_coverage"] == 2 / 3
    assert result["selected_candidates"] == 2


def test_parlay_push_reduces_to_surviving_leg() -> None:
    assert settle_two_leg_parlay("win", 2.1, "push", 1.9) == ("win", 1.1)
    assert settle_two_leg_parlay("loss", 2.0, "push", 1.9) == ("loss", -1.0)
    assert settle_two_leg_parlay("push", 2.1, "push", 1.9) == ("push", 0.0)
