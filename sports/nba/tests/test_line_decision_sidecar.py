from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from decision_engine.line_decision import LineDecisionConfig, build_line_decision_lookup, estimate_line_decision
from post_process_market_plays import compute_final_board
from select_market_plays import build_history_lookup, build_play_rows


def _synthetic_history() -> pd.DataFrame:
    rows: list[dict] = []
    residual_pattern = [1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, -0.1, 0.4, 0.7]
    for idx in range(80):
        pred_pts = 21.0 + 0.15 * (idx % 4)
        market_pts = 19.5 + float(idx % 2)
        residual = residual_pattern[idx % len(residual_pattern)]
        actual_pts = pred_pts + residual
        rows.append(
            {
                "player": f"Hist Player {idx}",
                "market_date": f"2026-03-{(idx % 28) + 1:02d}",
                "pred_PTS": pred_pts,
                "market_PTS": market_pts,
                "actual_PTS": actual_pts,
                "pred_TRB": None,
                "market_TRB": None,
                "actual_TRB": None,
                "pred_AST": None,
                "market_AST": None,
                "actual_AST": None,
                "did_not_play": 0,
                "minutes": 28.0,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_slate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Strong Sidecar",
                "market_date": "2026-04-01",
                "market_player_raw": "Strong Sidecar",
                "market_event_id": "game_a",
                "market_commence_time_utc": "2026-04-01T23:00:00Z",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "history_rows": 120,
                "last_history_date": "2026-03-31",
                "csv": "strong.csv",
                "belief_uncertainty": 0.42,
                "feasibility": 0.93,
                "fallback_blend": 0.0,
                "pred_PTS": 22.6,
                "baseline_PTS": 21.0,
                "market_PTS": 19.5,
                "baseline_edge_PTS": 1.5,
                "PTS_uncertainty_sigma": 0.75,
                "PTS_spike_probability": 0.32,
                "market_books_PTS": 6,
                "pred_TRB": None,
                "baseline_TRB": None,
                "market_TRB": None,
                "baseline_edge_TRB": None,
                "TRB_uncertainty_sigma": None,
                "TRB_spike_probability": None,
                "market_books_TRB": None,
                "pred_AST": None,
                "baseline_AST": None,
                "market_AST": None,
                "baseline_edge_AST": None,
                "AST_uncertainty_sigma": None,
                "AST_spike_probability": None,
                "market_books_AST": None,
            },
            {
                "player": "Fragile Sidecar",
                "market_date": "2026-04-01",
                "market_player_raw": "Fragile Sidecar",
                "market_event_id": "game_b",
                "market_commence_time_utc": "2026-04-01T23:30:00Z",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "history_rows": 18,
                "last_history_date": "2026-03-31",
                "csv": "fragile.csv",
                "belief_uncertainty": 0.86,
                "feasibility": 0.63,
                "fallback_blend": 0.35,
                "pred_PTS": 19.9,
                "baseline_PTS": 19.7,
                "market_PTS": 19.5,
                "baseline_edge_PTS": 0.2,
                "PTS_uncertainty_sigma": 4.2,
                "PTS_spike_probability": 0.58,
                "market_books_PTS": 2,
                "pred_TRB": None,
                "baseline_TRB": None,
                "market_TRB": None,
                "baseline_edge_TRB": None,
                "TRB_uncertainty_sigma": None,
                "TRB_spike_probability": None,
                "market_books_TRB": None,
                "pred_AST": None,
                "baseline_AST": None,
                "market_AST": None,
                "baseline_edge_AST": None,
                "AST_uncertainty_sigma": None,
                "AST_spike_probability": None,
                "market_books_AST": None,
            },
        ]
    )


def _board_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Keep Me",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 24.0,
                "market_line": 21.5,
                "abs_edge": 2.5,
                "edge": 2.5,
                "expected_win_rate": 0.68,
                "expected_push_rate": 0.05,
                "posterior_variance": 0.03,
                "belief_confidence_factor": 0.88,
                "feasibility": 0.91,
                "recommendation": "strong",
                "history_rows": 140,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_keep",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "risk_penalty": 0.20,
                "market_books": 6,
                "gap_percentile": 0.95,
                "confidence_score": 0.20,
                "line_decision_trade_eligible": True,
                "line_decision_action": "OVER",
            },
            {
                "player": "Drop Me",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 20.0,
                "market_line": 19.5,
                "abs_edge": 0.5,
                "edge": 0.5,
                "expected_win_rate": 0.56,
                "expected_push_rate": 0.38,
                "posterior_variance": 0.05,
                "belief_confidence_factor": 0.62,
                "feasibility": 0.66,
                "recommendation": "pass",
                "history_rows": 18,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_drop",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "risk_penalty": 0.41,
                "market_books": 2,
                "gap_percentile": 0.58,
                "confidence_score": 0.04,
                "line_decision_trade_eligible": False,
                "line_decision_action": "NO_TRADE",
            },
        ]
    )


def test_line_decision_prefers_trade_for_strong_edge() -> None:
    lookup = build_line_decision_lookup(_synthetic_history())
    decision = estimate_line_decision(
        lookup=lookup,
        target="PTS",
        prediction=22.6,
        market_line=19.5,
        direction="OVER",
        gap_percentile=0.96,
        uncertainty_sigma=0.75,
        belief_confidence_factor=0.90,
        feasibility=0.93,
        history_rows=120,
        market_books=6,
        fallback_blend=0.0,
        prior_direction_win_rate=0.66,
        prior_neutral_rate=0.03,
        config=LineDecisionConfig(),
    )
    assert decision["action"] == "OVER"
    assert decision["trade_eligible"] is True
    assert decision["chosen_direction_prob"] > decision["opposite_direction_prob"]
    assert decision["chosen_direction_conditional_prob"] >= 0.57
    assert decision["no_trade_prob"] < 0.36


def test_line_decision_gate_uses_conditional_trade_confidence() -> None:
    decision = estimate_line_decision(
        lookup={},
        target="PTS",
        prediction=20.2,
        market_line=19.5,
        direction="OVER",
        gap_percentile=0.80,
        uncertainty_sigma=0.0,
        belief_confidence_factor=1.0,
        feasibility=1.0,
        history_rows=100,
        market_books=5,
        fallback_blend=0.0,
        prior_direction_win_rate=0.55,
        prior_neutral_rate=0.20,
        config=LineDecisionConfig(no_trade_threshold=0.45, min_trade_prob=0.57, min_trade_prob_gap=0.06),
    )
    assert decision["chosen_direction_prob"] < 0.57
    assert decision["chosen_direction_conditional_prob"] > 0.57
    assert decision["trade_eligible"] is True


def test_build_play_rows_marks_fragile_near_line_case_as_no_trade() -> None:
    history = _synthetic_history()
    slate = _synthetic_slate()
    plays = build_play_rows(
        slate,
        build_history_lookup(history),
        line_decision_lookup=build_line_decision_lookup(history),
        line_decision_enabled=True,
        line_decision_config=LineDecisionConfig(),
    )
    assert {"line_decision_action", "line_no_trade_prob", "line_decision_trade_eligible"}.issubset(set(plays.columns))
    by_player = plays.set_index("player")
    assert by_player.loc["Strong Sidecar", "line_decision_action"] == "OVER"
    assert bool(by_player.loc["Strong Sidecar", "line_decision_trade_eligible"]) is True
    assert by_player.loc["Strong Sidecar", "expected_push_rate"] == by_player.loc["Strong Sidecar", "historical_push_rate"]
    assert by_player.loc["Strong Sidecar", "recommendation"] in {"consider", "strong", "elite"}
    assert by_player.loc["Fragile Sidecar", "line_decision_action"] == "NO_TRADE"
    assert bool(by_player.loc["Fragile Sidecar", "line_decision_trade_eligible"]) is False
    assert by_player.loc["Fragile Sidecar", "recommendation"] == "pass"
    assert by_player.loc["Fragile Sidecar", "line_no_trade_prob"] > 0.36


def test_compute_final_board_filters_line_decision_no_trade_rows() -> None:
    board = compute_final_board(
        _board_input(),
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        max_plays_per_player=5,
        max_plays_per_target=0,
        max_total_plays=5,
        max_target_plays={"PTS": 5, "TRB": 5, "AST": 5},
        max_plays_per_game=5,
        max_plays_per_script_cluster=5,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.52,
        full_bet_win_rate=0.56,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
    )
    assert not board.empty
    assert board["player"].tolist() == ["Keep Me"]
