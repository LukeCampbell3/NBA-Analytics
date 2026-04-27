from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from decision_engine.gating import StrategyConfig
from decision_engine.board_optimizer import optimize_board
from decision_engine.policy_tuning import build_default_shadow_strategies
from decision_engine.selection import apply_policy


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Alpha Edge",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.67,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.95,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 2.0,
                "edge": 2.0,
                "recommendation": "strong",
                "game_key": "game_alpha",
                "script_cluster_id": "cluster_alpha",
            },
            {
                "player": "Beta Correlated",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.66,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.93,
                "belief_uncertainty": 0.79,
                "feasibility": 0.88,
                "abs_edge": 1.9,
                "edge": 1.9,
                "recommendation": "strong",
                "game_key": "game_alpha",
                "script_cluster_id": "cluster_alpha",
            },
            {
                "player": "Gamma Diversified",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.64,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.92,
                "belief_uncertainty": 0.76,
                "feasibility": 0.89,
                "abs_edge": 1.6,
                "edge": -1.6,
                "recommendation": "consider",
                "game_key": "game_gamma",
                "script_cluster_id": "cluster_gamma",
            },
            {
                "player": "Delta Filler",
                "target": "TRB",
                "direction": "UNDER",
                "expected_win_rate": 0.61,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.90,
                "belief_uncertainty": 0.77,
                "feasibility": 0.87,
                "abs_edge": 1.4,
                "edge": -1.4,
                "recommendation": "consider",
                "game_key": "game_delta",
                "script_cluster_id": "cluster_delta",
            },
        ]
    )


def test_edge_mode_uses_edge_ranking() -> None:
    frame = pd.DataFrame(
        [
            {
                "player": "EV Leader",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.68,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.91,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 1.2,
                "edge": 1.2,
                "recommendation": "pass",
            },
            {
                "player": "Edge Leader",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.60,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.91,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 2.4,
                "edge": 2.4,
                "recommendation": "pass",
            },
        ]
    )
    config = StrategyConfig(
        name="edge_mode_test",
        selection_mode="edge",
        ranking_mode="edge",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=1,
        non_pts_min_gap_percentile=0.0,
    )

    out = apply_policy(frame, config)

    selected = out.loc[out["selected"], "player"].tolist()
    assert selected == ["Edge Leader"]


def test_greedy_selection_applies_game_cap() -> None:
    frame = pd.DataFrame(
        [
            {
                "player": "First Game Play",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.65,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.94,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 2.0,
                "edge": 2.0,
                "recommendation": "pass",
                "game_key": "same_game",
            },
            {
                "player": "Second Same Game",
                "target": "AST",
                "direction": "OVER",
                "expected_win_rate": 0.64,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.93,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 1.9,
                "edge": 1.9,
                "recommendation": "pass",
                "game_key": "same_game",
            },
            {
                "player": "Other Game",
                "target": "TRB",
                "direction": "UNDER",
                "expected_win_rate": 0.63,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.92,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 1.8,
                "edge": -1.8,
                "recommendation": "pass",
                "game_key": "other_game",
            },
        ]
    )
    config = StrategyConfig(
        name="game_cap_test",
        selection_mode="ev_adjusted",
        ranking_mode="ev_adjusted",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=2,
        max_plays_per_game=1,
        non_pts_min_gap_percentile=0.0,
    )

    out = apply_policy(frame, config)

    assert out.loc[out["player"] == "Second Same Game", "decision_stage"].iloc[0] == "capped_game"
    assert set(out.loc[out["selected"], "player"].tolist()) == {"First Game Play", "Other Game"}


def test_board_objective_mode_optimizes_a_diverse_board() -> None:
    frame = _candidate_rows()
    config = StrategyConfig(
        name="board_objective_test",
        selection_mode="board_objective",
        ranking_mode="board_objective",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=2,
        max_plays_per_player=1,
        max_plays_per_game=2,
        max_plays_per_script_cluster=2,
        non_pts_min_gap_percentile=0.0,
        board_objective_candidate_limit=10,
        board_objective_max_search_nodes=5000,
        board_objective_lambda_corr=1.0,
        board_objective_lambda_conc=0.75,
        board_objective_lambda_unc=0.05,
        board_objective_corr_same_game=1.0,
        board_objective_corr_same_target=0.8,
        board_objective_corr_same_direction=0.4,
        board_objective_corr_same_script_cluster=0.6,
    )

    out = apply_policy(frame, config)

    selected = set(out.loc[out["selected"], "player"].tolist())
    assert selected == {"Alpha Edge", "Gamma Diversified"}
    assert out.loc[out["player"] == "Beta Correlated", "decision_stage"].iloc[0] == "portfolio_excluded"


def test_board_objective_uses_segment_specific_pool_gate() -> None:
    frame = pd.DataFrame(
        [
            {
                "player": "Safe Under",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.492,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.93,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 1.8,
                "edge": -1.8,
                "recommendation": "pass",
                "game_key": "g1",
            },
            {
                "player": "Weak Over",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.492,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.91,
                "belief_uncertainty": 0.79,
                "feasibility": 0.89,
                "abs_edge": 1.7,
                "edge": 1.7,
                "recommendation": "pass",
                "game_key": "g2",
            },
            {
                "player": "Anchor Ast",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.505,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.92,
                "belief_uncertainty": 0.77,
                "feasibility": 0.90,
                "abs_edge": 1.6,
                "edge": -1.6,
                "recommendation": "pass",
                "game_key": "g3",
            },
        ]
    )
    payload = {
        "months": {
            "2026-04": {
                "global": {"threshold": 0.495, "hit_rate": 0.60},
                "segments": {
                    "PTS|UNDER": {"threshold": 0.491, "hit_rate": 0.68, "lift_pp": 6.0},
                    "PTS|OVER": {"threshold": 0.495, "hit_rate": 0.56, "lift_pp": 1.0},
                    "AST|UNDER": {"threshold": 0.500, "hit_rate": 0.66, "lift_pp": 4.0},
                },
            }
        }
    }
    config = StrategyConfig(
        name="segment_pool_gate_test",
        selection_mode="board_objective",
        ranking_mode="board_objective",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=2,
        non_pts_min_gap_percentile=0.0,
        board_objective_candidate_limit=10,
        board_objective_max_search_nodes=5000,
        learned_gate_enabled=True,
        learned_gate_payload=payload,
        learned_gate_rescue_enabled=False,
        initial_pool_gate_drop_fraction=0.0,
    )

    out = apply_policy(frame, config)

    assert set(out.loc[out["selected"], "player"].tolist()) == {"Safe Under", "Anchor Ast"}
    assert bool(out.loc[out["player"] == "Safe Under", "learned_gate_pass"].iloc[0]) is True
    assert bool(out.loc[out["player"] == "Weak Over", "learned_gate_pass"].iloc[0]) is False
    assert out.loc[out["player"] == "Weak Over", "decision_stage"].iloc[0] == "final_pool_gate_filtered"


def test_board_objective_pool_gate_rescues_high_quality_near_miss() -> None:
    frame = pd.DataFrame(
        [
            {
                "player": "Anchor Pass",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.505,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.90,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 1.4,
                "edge": -1.4,
                "recommendation": "pass",
                "game_key": "g1",
                "noise_score": 0.10,
                "contradiction_score": 0.05,
                "recency_factor": 0.90,
                "recoverability_score": 0.75,
            },
            {
                "player": "Rescue Under",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.494,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.97,
                "belief_uncertainty": 0.76,
                "feasibility": 0.95,
                "abs_edge": 2.1,
                "edge": -2.1,
                "recommendation": "pass",
                "game_key": "g2",
                "noise_score": 0.02,
                "contradiction_score": 0.02,
                "recency_factor": 0.96,
                "recoverability_score": 0.90,
            },
            {
                "player": "Filtered Over",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.494,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.30,
                "belief_uncertainty": 0.95,
                "feasibility": 0.70,
                "abs_edge": 0.7,
                "edge": 0.7,
                "recommendation": "pass",
                "game_key": "g3",
                "noise_score": 0.90,
                "contradiction_score": 0.80,
                "recency_factor": 0.45,
                "recoverability_score": 0.20,
            },
        ]
    )
    payload = {
        "months": {
            "2026-04": {
                "global": {"threshold": 0.500, "hit_rate": 0.60},
                "segments": {
                    "PTS|UNDER": {"threshold": 0.495, "hit_rate": 0.69, "lift_pp": 7.0},
                    "PTS|OVER": {"threshold": 0.495, "hit_rate": 0.55, "lift_pp": 1.0},
                    "AST|UNDER": {"threshold": 0.500, "hit_rate": 0.66, "lift_pp": 4.0},
                },
            }
        }
    }
    config = StrategyConfig(
        name="rescue_pool_gate_test",
        selection_mode="board_objective",
        ranking_mode="board_objective",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=2,
        non_pts_min_gap_percentile=0.0,
        board_objective_candidate_limit=10,
        board_objective_max_search_nodes=5000,
        learned_gate_enabled=True,
        learned_gate_payload=payload,
        learned_gate_rescue_enabled=True,
        learned_gate_rescue_max_rows=1,
        learned_gate_near_miss_margin=0.003,
        initial_pool_gate_drop_fraction=0.0,
    )

    out = apply_policy(frame, config)

    assert set(out.loc[out["selected"], "player"].tolist()) == {"Anchor Pass", "Rescue Under"}
    assert out.loc[out["player"] == "Rescue Under", "learned_gate_fill_source"].iloc[0] == "rescue"
    assert bool(out.loc[out["player"] == "Rescue Under", "final_pool_gate_rescue_selected"].iloc[0]) is True
    assert out.loc[out["player"] == "Filtered Over", "decision_stage"].iloc[0] == "final_pool_gate_filtered"
    assert float(out.loc[out["player"] == "Rescue Under", "final_pool_quality_score"].iloc[0]) > float(
        out.loc[out["player"] == "Filtered Over", "final_pool_quality_score"].iloc[0]
    )


def test_board_objective_uses_game_key_not_legacy_game_id() -> None:
    frame = pd.DataFrame(
        [
            {
                "player": "Game Key One",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.66,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.94,
                "belief_uncertainty": 0.78,
                "feasibility": 0.90,
                "abs_edge": 1.8,
                "edge": 1.8,
                "recommendation": "pass",
                "game_key": "game_key_one",
                "game_id": "legacy_same_game_id",
            },
            {
                "player": "Game Key Two",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.65,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.93,
                "belief_uncertainty": 0.78,
                "feasibility": 0.89,
                "abs_edge": 1.7,
                "edge": -1.7,
                "recommendation": "pass",
                "game_key": "game_key_two",
                "game_id": "legacy_same_game_id",
            },
        ]
    )
    config = StrategyConfig(
        name="board_objective_game_key_test",
        selection_mode="board_objective",
        ranking_mode="board_objective",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=2,
        max_plays_per_game=1,
        non_pts_min_gap_percentile=0.0,
        board_objective_candidate_limit=10,
        board_objective_max_search_nodes=5000,
    )

    out = apply_policy(frame, config)

    assert set(out.loc[out["selected"], "player"].tolist()) == {"Game Key One", "Game Key Two"}


def test_board_objective_candidate_pool_respects_configured_cap() -> None:
    rows: list[dict[str, object]] = []
    for idx in range(50):
        rows.append(
            {
                "player": f"Candidate {idx}",
                "target": "PTS" if idx % 3 == 0 else ("TRB" if idx % 3 == 1 else "AST"),
                "direction": "OVER" if idx % 2 == 0 else "UNDER",
                "expected_win_rate": 0.70 - 0.002 * idx,
                "expected_push_rate": 0.02,
                "gap_percentile": 0.95 - 0.003 * idx,
                "belief_uncertainty": 0.78 + 0.001 * idx,
                "feasibility": 0.92 - 0.001 * idx,
                "abs_edge": 2.5 - 0.02 * idx,
                "edge": 2.5 - 0.02 * idx,
                "recommendation": "pass",
                "game_key": f"game_{idx}",
                "script_cluster_id": f"cluster_{idx % 5}",
            }
        )
    frame = pd.DataFrame(rows)
    config = StrategyConfig(
        name="board_objective_candidate_cap_test",
        selection_mode="board_objective",
        ranking_mode="board_objective",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=12,
        max_plays_per_game=0,
        max_plays_per_script_cluster=0,
        non_pts_min_gap_percentile=0.0,
        board_objective_overfetch=10.0,
        board_objective_candidate_limit=36,
        board_objective_max_search_nodes=10000,
    )

    result = optimize_board(frame, config)

    assert len(result.candidate_pool) == 36
    assert len(result.selected_board) == 12


def test_shadow_strategies_include_production_board_objective_v1() -> None:
    names = {config.name for config in build_default_shadow_strategies()}
    assert "production_board_objective_v1" in names
