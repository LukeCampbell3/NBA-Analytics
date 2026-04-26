from __future__ import annotations

import pandas as pd

from sports.parlay_analysis import annotate_parlay_board, evaluate_historical_parlays


def test_annotate_parlay_board_tags_best_disjoint_pair() -> None:
    plays = [
        {
            "player": "Alpha Guard",
            "player_display_name": "Alpha Guard",
            "team": "A",
            "target": "PTS",
            "direction": "OVER",
            "game_id": "game-1",
            "expected_win_rate": 0.66,
        },
        {
            "player": "Beta Wing",
            "player_display_name": "Beta Wing",
            "team": "B",
            "target": "AST",
            "direction": "OVER",
            "game_id": "game-2",
            "expected_win_rate": 0.64,
        },
        {
            "player": "Gamma Big",
            "player_display_name": "Gamma Big",
            "team": "A",
            "target": "TRB",
            "direction": "UNDER",
            "game_id": "game-1",
            "expected_win_rate": 0.58,
        },
    ]

    payload = annotate_parlay_board(plays, sport="nba", probability_field="expected_win_rate", max_pairs=1)

    tagged = [play for play in payload["plays"] if play["parlay_candidate"]]
    assert len(tagged) == 2
    assert {play["player"] for play in tagged} == {"Alpha Guard", "Beta Wing"}
    assert payload["summary"]["selected_pair_count"] == 1
    assert payload["pairs"][0]["projected_probability"] > 0.35


def test_evaluate_historical_parlays_reports_pair_hit_rate() -> None:
    history = pd.DataFrame(
        [
            {"market_date": "2026-04-01", "player": "Alpha", "target": "PTS", "direction": "OVER", "game_id": "g1", "estimated_win_rate": 0.68, "result": "win"},
            {"market_date": "2026-04-01", "player": "Beta", "target": "AST", "direction": "OVER", "game_id": "g2", "estimated_win_rate": 0.64, "result": "win"},
            {"market_date": "2026-04-01", "player": "Gamma", "target": "TRB", "direction": "UNDER", "game_id": "g1", "estimated_win_rate": 0.56, "result": "loss"},
            {"market_date": "2026-04-02", "player": "Alpha", "target": "PTS", "direction": "OVER", "game_id": "g3", "estimated_win_rate": 0.67, "result": "loss"},
            {"market_date": "2026-04-02", "player": "Beta", "target": "AST", "direction": "OVER", "game_id": "g4", "estimated_win_rate": 0.63, "result": "win"},
            {"market_date": "2026-04-02", "player": "Gamma", "target": "TRB", "direction": "UNDER", "game_id": "g3", "estimated_win_rate": 0.55, "result": "loss"},
        ]
    )

    summary = evaluate_historical_parlays(
        history,
        sport="nba",
        date_col="market_date",
        probability_col="estimated_win_rate",
        result_col="result",
        max_pairs_per_day=1,
    )

    assert summary["available"] is True
    assert summary["sample_dates"] == 2
    assert summary["selected"]["graded_pair_count"] == 2
    assert summary["selected"]["hit_pair_count"] == 1
    assert summary["selected"]["pair_hit_rate"] == 0.5


def test_annotate_parlay_board_uses_fallback_for_weak_nba_slate() -> None:
    plays = [
        {
            "player": "Austin Reaves",
            "player_display_name": "Austin Reaves",
            "team": "LAL",
            "target": "PTS",
            "direction": "OVER",
            "game_id": "g1",
            "expected_win_rate": 0.5093,
            "ev": 0.008,
        },
        {
            "player": "Stephon Castle",
            "player_display_name": "Stephon Castle",
            "team": "SAS",
            "target": "AST",
            "direction": "OVER",
            "game_id": "g2",
            "expected_win_rate": 0.5071,
            "ev": 0.004,
        },
        {
            "player": "Toumani Camara",
            "player_display_name": "Toumani Camara",
            "team": "POR",
            "target": "PTS",
            "direction": "OVER",
            "game_id": "g3",
            "expected_win_rate": 0.5024,
            "ev": -0.002,
        },
    ]

    payload = annotate_parlay_board(plays, sport="nba", probability_field="expected_win_rate")

    assert payload["summary"]["selection_mode"] == "fallback"
    assert payload["summary"]["selected_pair_count"] == 1
    tagged = [play for play in payload["plays"] if play["parlay_candidate"]]
    assert len(tagged) == 2
    assert {play["player"] for play in tagged} == {"Austin Reaves", "Stephon Castle"}
