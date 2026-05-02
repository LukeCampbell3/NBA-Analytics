from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from decision_engine.selected_board_calibration import apply_selected_board_calibration, fit_selected_board_calibrator_payload
from post_process_market_plays import compute_final_board


def test_selected_board_calibration_safety_profile_shrinks_extreme_probabilities() -> None:
    rows: list[dict] = []
    for idx in range(40):
        rows.append(
            {
                "run_date": "2026-04-24",
                "target": "PTS",
                "direction": "OVER",
                "probability": 0.61,
                "is_win": 1.0 if idx < 26 else 0.0,
            }
        )
    for idx in range(24):
        rows.append(
            {
                "run_date": "2026-04-24",
                "target": "PTS",
                "direction": "OVER",
                "probability": 0.82,
                "is_win": 1.0 if idx < 10 else 0.0,
            }
        )

    payload = fit_selected_board_calibrator_payload(
        pd.DataFrame(rows),
        run_date_col="run_date",
        prob_col="probability",
        label_col="is_win",
        target_col="target",
        direction_col="direction",
    )

    frame = pd.DataFrame(
        [
            {"target": "PTS", "direction": "OVER", "board_play_win_prob": 0.82, "market_date": "2026-05-01"},
            {"target": "PTS", "direction": "OVER", "board_play_win_prob": 0.61, "market_date": "2026-05-01"},
        ]
    )
    calibrated, source, month = apply_selected_board_calibration(
        frame,
        payload=payload,
        run_date_hint="2026-05-01",
        prob_col="board_play_win_prob",
        target_col="target",
        direction_col="direction",
    )

    assert month == "2026-05"
    assert source.iloc[0] == "safety"
    assert float(calibrated.iloc[0]) < 0.70
    assert float(calibrated.iloc[1]) < 0.61


def test_selected_board_calibration_applies_recent_segment_regime_adjustment() -> None:
    frame = pd.DataFrame(
        [
            {"target": "PTS", "direction": "OVER", "board_play_win_prob": 0.58, "market_date": "2026-05-01"},
            {"target": "AST", "direction": "UNDER", "board_play_win_prob": 0.58, "market_date": "2026-05-01"},
        ]
    )
    payload = {
        "version": 1,
        "config": {
            "recent_strength": 20.0,
            "recent_max_adjustment": 0.08,
            "recent_min_rows_segment": 18,
            "recent_min_rows_global": 40,
        },
        "months": {
            "2026-05": {
                "train_rows": 240,
                "train_start": "2026-01-01",
                "train_end": "2026-04-30",
                "global": None,
                "segments": {
                    "PTS_OVER": {"rows": 120, "mean_label": 0.54, "mean_raw_prob": 0.58},
                    "AST_UNDER": {"rows": 90, "mean_label": 0.60, "mean_raw_prob": 0.58},
                },
                "recent_global": {"rows": 60, "mean_label": 0.56, "mean_raw_prob": 0.58},
                "recent_segments": {
                    "PTS_OVER": {"rows": 28, "mean_label": 0.72, "mean_raw_prob": 0.58},
                    "AST_UNDER": {"rows": 24, "mean_label": 0.46, "mean_raw_prob": 0.58},
                },
                "safety_profile": None,
            }
        },
    }

    calibrated, source, month = apply_selected_board_calibration(
        frame,
        payload=payload,
        run_date_hint="2026-05-01",
        prob_col="board_play_win_prob",
        target_col="target",
        direction_col="direction",
    )

    assert month == "2026-05"
    assert "recent:segment:PTS_OVER" in str(source.iloc[0])
    assert "recent:segment:AST_UNDER" in str(source.iloc[1])
    assert float(calibrated.iloc[0]) > 0.58
    assert float(calibrated.iloc[1]) < 0.58


def test_compute_final_board_applies_confidence_haircut_from_selected_board_calibration() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "High Variance Guard",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 29.0,
                "market_line": 24.5,
                "abs_edge": 4.5,
                "edge": 4.5,
                "expected_win_rate": 0.95,
                "expected_push_rate": 0.0,
                "posterior_alpha": 10.0,
                "posterior_beta": 2.0,
                "posterior_variance": 0.02,
                "gap_percentile": 0.95,
                "belief_uncertainty": 0.80,
                "belief_confidence_factor": 0.92,
                "feasibility": 0.95,
                "recommendation": "elite",
                "history_rows": 120,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g1",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "market_player_raw": "High Variance Guard",
                "game_key": "g1",
            }
        ]
    )
    calibrator_payload = {
        "version": 1,
        "months": {
            "2026-05": {
                "train_rows": 64,
                "train_start": "2026-04-01",
                "train_end": "2026-04-30",
                "global": None,
                "segments": {},
                "safety_profile": {
                    "rows": 64,
                    "mean_prob": 0.70,
                    "mean_label": 0.56,
                    "global_shrink_factor": 0.70,
                    "high_prob_buckets": [
                        {
                            "lower": 0.80,
                            "upper": 1.00,
                            "rows": 20,
                            "wins": 8,
                            "avg_prob": 0.82,
                            "smoothed_hit_rate": 0.42,
                            "gap": 0.40,
                            "cap": 0.58,
                        }
                    ],
                },
            }
        },
    }

    board = compute_final_board(
        plays,
        max_total_plays=1,
        max_plays_per_player=1,
        max_plays_per_game=1,
        max_plays_per_script_cluster=1,
        selected_board_calibrator=calibrator_payload,
        selected_board_calibration_month="2026-05-01",
    )

    assert len(board) == 1
    assert float(board.iloc[0]["selected_board_prob_raw"]) > 0.70
    assert float(board.iloc[0]["expected_win_rate"]) < 0.60
    assert float(board.iloc[0]["selected_board_confidence_haircut"]) < 0.80
    assert float(board.iloc[0]["final_confidence"]) < float(board.iloc[0]["final_confidence_pre_selected_board_calibration"])


def test_compute_final_board_uses_empirical_category_priors_in_board_probability() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Weak PTS Over",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 25.1,
                "market_line": 23.5,
                "abs_edge": 1.6,
                "edge": 1.6,
                "expected_win_rate": 0.58,
                "expected_push_rate": 0.0,
                "posterior_alpha": 8.0,
                "posterior_beta": 4.0,
                "posterior_variance": 0.03,
                "gap_percentile": 0.88,
                "belief_uncertainty": 0.82,
                "belief_confidence_factor": 0.88,
                "feasibility": 0.90,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g1",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "market_player_raw": "Weak PTS Over",
                "game_key": "g1",
            },
            {
                "player": "Strong AST Under",
                "target": "AST",
                "direction": "UNDER",
                "prediction": 6.2,
                "market_line": 7.5,
                "abs_edge": 1.3,
                "edge": -1.3,
                "expected_win_rate": 0.58,
                "expected_push_rate": 0.0,
                "posterior_alpha": 8.0,
                "posterior_beta": 4.0,
                "posterior_variance": 0.03,
                "gap_percentile": 0.88,
                "belief_uncertainty": 0.82,
                "belief_confidence_factor": 0.88,
                "feasibility": 0.90,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g2",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "market_player_raw": "Strong AST Under",
                "game_key": "g2",
            },
        ]
    )
    calibrator_payload = {
        "version": 1,
        "months": {
            "2026-05": {
                "train_rows": 654,
                "train_start": "2026-01-01",
                "train_end": "2026-04-30",
                "global": None,
                "segments": {
                    "PTS_OVER": {"rows": 133, "mean_label": 0.37, "mean_raw_prob": 0.62},
                    "PTS_UNDER": {"rows": 87, "mean_label": 0.68, "mean_raw_prob": 0.60},
                    "AST_OVER": {"rows": 121, "mean_label": 0.38, "mean_raw_prob": 0.63},
                    "AST_UNDER": {"rows": 91, "mean_label": 0.81, "mean_raw_prob": 0.59},
                    "TRB_OVER": {"rows": 130, "mean_label": 0.48, "mean_raw_prob": 0.62},
                    "TRB_UNDER": {"rows": 92, "mean_label": 0.71, "mean_raw_prob": 0.61},
                },
                "safety_profile": {
                    "rows": 654,
                    "mean_prob": 0.61,
                    "mean_label": 0.56,
                    "global_shrink_factor": 0.85,
                    "high_prob_buckets": [],
                },
            }
        },
    }

    board = compute_final_board(
        plays,
        selection_mode="board_objective",
        ranking_mode="board_objective",
        max_total_plays=2,
        min_board_plays=2,
        max_plays_per_player=1,
        max_plays_per_game=1,
        max_plays_per_script_cluster=1,
        max_target_plays={"PTS": 2, "AST": 2, "TRB": 2},
        min_recommendation="pass",
        min_final_confidence=0.0,
        min_ev=-1.0,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.50,
        full_bet_win_rate=0.51,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
        selected_board_calibrator=calibrator_payload,
        selected_board_calibration_month="2026-05-01",
    ).set_index("player")

    weak = board.loc["Weak PTS Over"]
    strong = board.loc["Strong AST Under"]

    assert float(weak["board_category_prob_adjustment"]) < 0.0
    assert float(strong["board_category_prob_adjustment"]) > 0.0
    assert float(weak["board_segment_prior_anchor"]) < float(weak["selected_board_prob_raw"])
    assert float(strong["board_segment_prior_anchor"]) > float(strong["selected_board_prob_raw"])


def test_compute_final_board_prefers_strong_empirical_category_when_scores_are_close() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "PTS Over Candidate",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 25.2,
                "market_line": 23.5,
                "abs_edge": 1.7,
                "edge": 1.7,
                "expected_win_rate": 0.59,
                "expected_push_rate": 0.0,
                "posterior_alpha": 8.0,
                "posterior_beta": 4.0,
                "posterior_variance": 0.03,
                "gap_percentile": 0.89,
                "belief_uncertainty": 0.83,
                "belief_confidence_factor": 0.89,
                "feasibility": 0.90,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g1",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "market_player_raw": "PTS Over Candidate",
                "game_key": "g1",
            },
            {
                "player": "AST Under Candidate",
                "target": "AST",
                "direction": "UNDER",
                "prediction": 6.3,
                "market_line": 7.5,
                "abs_edge": 1.5,
                "edge": -1.2,
                "expected_win_rate": 0.585,
                "expected_push_rate": 0.0,
                "posterior_alpha": 8.0,
                "posterior_beta": 4.0,
                "posterior_variance": 0.03,
                "gap_percentile": 0.88,
                "belief_uncertainty": 0.83,
                "belief_confidence_factor": 0.89,
                "feasibility": 0.90,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g2",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "market_player_raw": "AST Under Candidate",
                "game_key": "g2",
            },
        ]
    )
    calibrator_payload = {
        "version": 1,
        "months": {
            "2026-05": {
                "train_rows": 654,
                "train_start": "2026-01-01",
                "train_end": "2026-04-30",
                "global": None,
                "segments": {
                    "PTS_OVER": {"rows": 133, "mean_label": 0.37, "mean_raw_prob": 0.62},
                    "AST_UNDER": {"rows": 91, "mean_label": 0.81, "mean_raw_prob": 0.59},
                },
                "safety_profile": {
                    "rows": 654,
                    "mean_prob": 0.61,
                    "mean_label": 0.56,
                    "global_shrink_factor": 0.85,
                    "high_prob_buckets": [],
                },
            }
        },
    }

    board = compute_final_board(
        plays,
        selection_mode="board_objective",
        ranking_mode="board_objective",
        max_total_plays=1,
        min_board_plays=1,
        max_plays_per_player=1,
        max_plays_per_game=1,
        max_plays_per_script_cluster=1,
        max_target_plays={"PTS": 1, "AST": 1, "TRB": 1},
        min_recommendation="pass",
        min_final_confidence=0.0,
        min_ev=-1.0,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.50,
        full_bet_win_rate=0.51,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
        selected_board_calibrator=calibrator_payload,
        selected_board_calibration_month="2026-05-01",
    )

    assert len(board) == 1
    assert board.iloc[0]["player"] == "AST Under Candidate"
