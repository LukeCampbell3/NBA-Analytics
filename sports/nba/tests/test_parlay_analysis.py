from __future__ import annotations

import pandas as pd

from sports.parlay_analysis import annotate_parlay_board, evaluate_historical_parlays, score_candidate_parlays


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


def test_annotate_parlay_board_respects_explicit_leg_eligibility() -> None:
    plays = [
        {
            "player": "Alpha Guard",
            "player_display_name": "Alpha Guard",
            "team": "A",
            "target": "PTS",
            "direction": "UNDER",
            "game_id": "game-1",
            "expected_win_rate": 0.60,
            "parlay_precision_eligible": True,
        },
        {
            "player": "Beta Wing",
            "player_display_name": "Beta Wing",
            "team": "B",
            "target": "AST",
            "direction": "OVER",
            "game_id": "game-2",
            "expected_win_rate": 0.64,
            "parlay_precision_eligible": False,
        },
        {
            "player": "Gamma Big",
            "player_display_name": "Gamma Big",
            "team": "C",
            "target": "TRB",
            "direction": "UNDER",
            "game_id": "game-3",
            "expected_win_rate": 0.59,
            "parlay_precision_eligible": True,
        },
    ]

    payload = annotate_parlay_board(
        plays,
        sport="nba",
        probability_field="expected_win_rate",
        eligibility_field="parlay_precision_eligible",
        allow_fallback=False,
        max_pairs=1,
    )

    tagged = [play for play in payload["plays"] if play["parlay_candidate"]]
    assert len(tagged) == 2
    assert {play["player"] for play in tagged} == {"Alpha Guard", "Gamma Big"}


def test_annotate_parlay_board_can_build_three_leg_ticket() -> None:
    plays = [
        {
            "player": "Alpha Guard",
            "player_display_name": "Alpha Guard",
            "team": "A",
            "target": "PTS",
            "direction": "UNDER",
            "game_id": "game-1",
            "expected_win_rate": 0.72,
        },
        {
            "player": "Beta Wing",
            "player_display_name": "Beta Wing",
            "team": "B",
            "target": "AST",
            "direction": "UNDER",
            "game_id": "game-2",
            "expected_win_rate": 0.71,
        },
        {
            "player": "Gamma Big",
            "player_display_name": "Gamma Big",
            "team": "C",
            "target": "TRB",
            "direction": "UNDER",
            "game_id": "game-3",
            "expected_win_rate": 0.70,
        },
    ]

    payload = annotate_parlay_board(
        plays,
        sport="nba",
        probability_field="expected_win_rate",
        allow_fallback=False,
        max_pairs=1,
        min_legs_per_parlay=3,
        max_legs_per_parlay=3,
    )

    tagged = [play for play in payload["plays"] if play["parlay_candidate"]]
    assert len(tagged) == 3
    assert payload["summary"]["selected_pair_count"] == 1
    assert payload["summary"]["selected_parlay_count"] == 1
    assert payload["pairs"][0]["leg_count"] == 3
    assert payload["pairs"][0]["ticket_rank"] == 1
    assert all(play["parlay_leg_count"] == 3 for play in tagged)


def test_annotate_parlay_board_ignores_unknown_script_cluster_penalty() -> None:
    plays = [
        {
            "player": "Alpha Guard",
            "player_display_name": "Alpha Guard",
            "team": "A",
            "target": "AST",
            "direction": "UNDER",
            "game_id": "game-1",
            "expected_win_rate": 0.5560091784765044,
            "script_cluster_id": "script=unknown",
        },
        {
            "player": "Beta Wing",
            "player_display_name": "Beta Wing",
            "team": "B",
            "target": "AST",
            "direction": "UNDER",
            "game_id": "game-2",
            "expected_win_rate": 0.5560091784765044,
            "script_cluster_id": "script=unknown",
        },
    ]

    payload = annotate_parlay_board(
        plays,
        sport="nba",
        probability_field="expected_win_rate",
        allow_fallback=False,
        min_leg_probability=0.555,
        min_pair_probability=0.309,
        max_pairs=1,
        min_legs_per_parlay=2,
        max_legs_per_parlay=2,
    )

    assert payload["summary"]["selected_parlay_count"] == 1
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["projected_probability"] > 0.309


def test_score_candidate_parlays_caps_projected_probability_to_independent_joint_rate() -> None:
    plays = [
        {
            "player": "Alpha Guard",
            "player_display_name": "Alpha Guard",
            "team": "A",
            "target": "PTS",
            "direction": "OVER",
            "game_id": "game-1",
            "expected_win_rate": 0.80,
        },
        {
            "player": "Beta Wing",
            "player_display_name": "Beta Wing",
            "team": "B",
            "target": "AST",
            "direction": "OVER",
            "game_id": "game-2",
            "expected_win_rate": 0.75,
        },
    ]

    parlays = score_candidate_parlays(
        plays,
        sport="nba",
        probability_field="expected_win_rate",
        min_leg_probability=0.70,
        min_pair_probability=0.59,
        min_legs_per_parlay=2,
        max_legs_per_parlay=2,
    )

    assert len(parlays) == 1
    assert abs(parlays[0]["projected_probability"] - (0.80 * 0.75)) < 1e-9


def test_mlb_parlays_avoid_same_market_bucket_stacking() -> None:
    plays = [
        {
            "player": "Alpha Bat",
            "player_display_name": "Alpha Bat",
            "team": "A",
            "target": "TB",
            "direction": "OVER",
            "game_id": "game-1",
            "market_bucket": "TB|OVER|0.5",
            "estimated_graded_hit_rate": 0.84,
            "parlay_precision_eligible": True,
            "final_pool_quality_score": 0.92,
        },
        {
            "player": "Beta Bat",
            "player_display_name": "Beta Bat",
            "team": "B",
            "target": "TB",
            "direction": "OVER",
            "game_id": "game-2",
            "market_bucket": "TB|OVER|0.5",
            "estimated_graded_hit_rate": 0.83,
            "parlay_precision_eligible": True,
            "final_pool_quality_score": 0.91,
        },
        {
            "player": "Gamma Runner",
            "player_display_name": "Gamma Runner",
            "team": "C",
            "target": "R",
            "direction": "OVER",
            "game_id": "game-3",
            "market_bucket": "R|OVER|0.5",
            "estimated_graded_hit_rate": 0.79,
            "parlay_precision_eligible": True,
            "final_pool_quality_score": 0.88,
        },
        {
            "player": "Delta Hitter",
            "player_display_name": "Delta Hitter",
            "team": "D",
            "target": "H",
            "direction": "OVER",
            "game_id": "game-4",
            "market_bucket": "H|OVER|0.5",
            "estimated_graded_hit_rate": 0.78,
            "parlay_precision_eligible": True,
            "final_pool_quality_score": 0.87,
        },
    ]

    payload = annotate_parlay_board(
        plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
        allow_fallback=False,
        max_pairs=2,
        min_leg_probability=0.75,
        min_pair_probability=0.60,
    )

    assert payload["summary"]["selected_parlay_count"] == 1
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["same_market_bucket"] is False
    assert {leg["player"] for leg in payload["pairs"][0]["legs"]} == {"Alpha Bat", "Gamma Runner"}


def test_mlb_parlays_require_different_players_and_games() -> None:
    common = {
        "team": "A",
        "direction": "UNDER",
        "estimated_graded_hit_rate": 0.80,
        "parlay_precision_eligible": True,
        "final_pool_quality_score": 0.90,
    }
    plays = [
        {**common, "player": "Alpha", "target": "TB", "game_id": "game-1", "market_bucket": "TB|UNDER|1.5"},
        {**common, "player": "Beta", "target": "R", "game_id": "game-1", "market_bucket": "R|UNDER|0.5"},
        {**common, "player": "Alpha", "target": "H", "game_id": "game-2", "market_bucket": "H|UNDER|0.5"},
    ]

    tickets = score_candidate_parlays(
        plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
        min_leg_probability=0.75,
        min_pair_probability=0.60,
    )

    assert len(tickets) == 1
    assert tickets[0]["same_player"] is False
    assert tickets[0]["same_game"] is False
    assert set(tickets[0]["leg_indices"]) == {1, 2}
