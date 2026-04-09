from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from decision_engine.conditional_promotion import MODE_SAFE_SHUTDOWN, apply_conditional_promotion
from post_process_market_plays import compute_final_board


def _sample_selector_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Alpha One",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 24.2,
                "market_line": 22.5,
                "abs_edge": 1.7,
                "expected_win_rate": 0.70,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_confidence_factor": 0.85,
                "feasibility": 0.86,
                "recommendation": "elite",
                "history_rows": 140,
                "market_date": "2026-03-30",
                "last_history_date": "2026-03-29",
                "market_event_id": "game_1",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "risk_penalty": 0.22,
                "market_books": 5,
                "gap_percentile": 0.95,
                "confidence_score": 0.20,
                "edge": 1.7,
            },
            {
                "player": "Beta Two",
                "target": "AST",
                "direction": "OVER",
                "prediction": 7.1,
                "market_line": 6.5,
                "abs_edge": 0.6,
                "expected_win_rate": 0.63,
                "expected_push_rate": 0.04,
                "posterior_variance": 0.05,
                "belief_confidence_factor": 0.78,
                "feasibility": 0.81,
                "recommendation": "consider",
                "history_rows": 120,
                "market_date": "2026-03-30",
                "last_history_date": "2026-03-29",
                "market_event_id": "game_1",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "risk_penalty": 0.30,
                "market_books": 4,
                "gap_percentile": 0.86,
                "confidence_score": 0.12,
                "edge": 0.6,
            },
        ]
    )


def test_conditional_safe_shutdown_when_required_columns_missing() -> None:
    frame = pd.DataFrame([{"player": "A"}])
    out, summary = apply_conditional_promotion(frame, policy_payload={"conditional_framework_enabled": True})
    assert summary["fallback_mode"] == MODE_SAFE_SHUTDOWN
    assert (out["conditional_eligible_for_board"] == False).all()  # noqa: E712


def test_full_mode_blocks_non_anchor_rows_when_not_promoted() -> None:
    frame = _sample_selector_frame()
    policy = {
        "conditional_framework_enabled": True,
        "conditional_framework_mode": "full",
        "conditional_anchor_min_probability": 0.50,
        "conditional_anchor_min_confidence": 0.0,
        "conditional_anchor_max_risk_penalty": 1.0,
        "conditional_min_anchor_count": 1,
        "conditional_recoverability_threshold": 0.0,
        "conditional_noise_threshold": 1.0,
        "conditional_min_pair_count": 1,
        "conditional_min_recent_pair_count": 0,
        "conditional_min_regime_pair_count": 0,
        "conditional_min_support": 0.0,
        "conditional_promotion_min_probability": 0.99,
        "conditional_promotion_min_ev": 1.0,
    }
    out, summary = apply_conditional_promotion(frame, policy_payload=policy)
    assert summary["fallback_mode"].startswith("A_")
    assert int(out["conditional_promoted"].sum()) == 0
    assert (out.loc[~out["is_anchor"], "conditional_eligible_for_board"] == False).all()  # noqa: E712


def test_final_board_respects_conditional_eligibility_gate() -> None:
    frame = _sample_selector_frame()
    frame["conditional_eligible_for_board"] = [True, False]
    frame["conditional_promoted"] = [False, False]
    frame["decision_tier"] = ["Tier A - Baseline", "Tier C - Recoverable Rejected"]
    frame["script_cluster_id"] = ["cluster_a", "cluster_a"]
    board = compute_final_board(
        frame,
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
    )
    assert not board.empty
    assert (board["conditional_eligible_for_board"] == True).all()  # noqa: E712


def test_script_cluster_cap_ignores_unknown_cluster() -> None:
    frame = pd.concat([_sample_selector_frame()] * 2, ignore_index=True)
    frame.loc[:, "player"] = ["Alpha One", "Beta Two", "Gamma Three", "Delta Four"]
    frame.loc[:, "target"] = ["PTS", "PTS", "PTS", "PTS"]
    frame.loc[:, "expected_win_rate"] = [0.72, 0.71, 0.70, 0.69]
    frame.loc[:, "expected_push_rate"] = [0.02, 0.02, 0.02, 0.02]
    frame.loc[:, "recommendation"] = "pass"
    frame.loc[:, "script_cluster_id"] = "script=unknown"
    frame.loc[:, "conditional_eligible_for_board"] = True
    frame.loc[:, "market_event_id"] = ""
    frame.loc[:, "market_home_team"] = ""
    frame.loc[:, "market_away_team"] = ""
    frame.loc[:, "gap_percentile"] = 0.9
    frame.loc[:, "belief_confidence_factor"] = 0.9
    frame.loc[:, "feasibility"] = 0.9
    frame.loc[:, "posterior_variance"] = 0.02
    frame.loc[:, "edge"] = [2.0, 1.8, 1.6, 1.4]
    frame.loc[:, "abs_edge"] = [2.0, 1.8, 1.6, 1.4]
    frame.loc[:, "market_line"] = [10.5, 11.5, 12.5, 13.5]
    frame.loc[:, "prediction"] = [12.5, 13.3, 14.1, 14.9]

    board = compute_final_board(
        frame,
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        selection_mode="edge",
        max_plays_per_player=5,
        max_plays_per_target=0,
        max_total_plays=4,
        max_target_plays={"PTS": 4, "TRB": 4, "AST": 4},
        max_plays_per_game=0,
        max_plays_per_script_cluster=2,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.52,
        full_bet_win_rate=0.56,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
    )
    assert len(board) == 4


def test_set_theory_mode_emits_consensus_columns() -> None:
    frame = pd.concat([_sample_selector_frame()] * 5, ignore_index=True)
    frame.loc[:, "player"] = [f"Player {idx}" for idx in range(len(frame))]
    frame.loc[:, "target"] = "PTS"
    frame.loc[:, "direction"] = ["OVER" if idx % 2 == 0 else "UNDER" for idx in range(len(frame))]
    frame.loc[:, "expected_win_rate"] = [0.72 - 0.01 * idx for idx in range(len(frame))]
    frame.loc[:, "expected_push_rate"] = 0.02
    frame.loc[:, "posterior_alpha"] = [22 - idx for idx in range(len(frame))]
    frame.loc[:, "posterior_beta"] = [10 + idx for idx in range(len(frame))]
    frame.loc[:, "posterior_variance"] = 0.03
    frame.loc[:, "recommendation"] = "pass"
    frame.loc[:, "conditional_eligible_for_board"] = True
    frame.loc[:, "script_cluster_id"] = [f"cluster_{idx % 3}" for idx in range(len(frame))]
    frame.loc[:, "market_event_id"] = ""
    frame.loc[:, "market_home_team"] = ""
    frame.loc[:, "market_away_team"] = ""
    frame.loc[:, "gap_percentile"] = 0.88
    frame.loc[:, "belief_confidence_factor"] = 0.90
    frame.loc[:, "feasibility"] = 0.90
    frame.loc[:, "market_books"] = 6
    frame.loc[:, "history_rows"] = 80
    frame.loc[:, "edge"] = [2.4 - 0.2 * idx for idx in range(len(frame))]
    frame.loc[:, "abs_edge"] = pd.Series(frame["edge"]).abs()
    frame.loc[:, "market_line"] = [10.0 + idx for idx in range(len(frame))]
    frame.loc[:, "prediction"] = frame["market_line"] + frame["edge"]

    board = compute_final_board(
        frame,
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        selection_mode="set_theory",
        ranking_mode="set_theory",
        max_plays_per_player=1,
        max_plays_per_target=0,
        max_total_plays=6,
        max_target_plays={"PTS": 6, "TRB": 6, "AST": 6},
        max_plays_per_game=0,
        max_plays_per_script_cluster=6,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.52,
        full_bet_win_rate=0.56,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
    )
    assert not board.empty
    assert len(board) <= 6
    required_cols = {"set_group", "agreement_count", "set_sources", "consensus_score", "set_strength"}
    assert required_cols.issubset(set(board.columns))
    assert board["set_group"].isin({"core", "strong_expansion", "anchor_fallback"}).all()
    assert board["agreement_count"].between(1, 3).all()


def test_edge_append_shadow_mode_is_append_only() -> None:
    frame = pd.concat([_sample_selector_frame()] * 6, ignore_index=True)
    frame.loc[:, "player"] = [f"Append Player {idx}" for idx in range(len(frame))]
    frame.loc[:, "target"] = "PTS"
    frame.loc[:, "direction"] = ["OVER" if idx % 2 == 0 else "UNDER" for idx in range(len(frame))]
    frame.loc[:, "expected_win_rate"] = [0.74 - 0.01 * idx for idx in range(len(frame))]
    frame.loc[:, "expected_push_rate"] = 0.02
    frame.loc[:, "posterior_alpha"] = [24 - idx for idx in range(len(frame))]
    frame.loc[:, "posterior_beta"] = [9 + idx for idx in range(len(frame))]
    frame.loc[:, "posterior_variance"] = 0.03
    frame.loc[:, "recommendation"] = "pass"
    frame.loc[:, "conditional_eligible_for_board"] = True
    frame.loc[:, "script_cluster_id"] = [f"cluster_{idx % 3}" for idx in range(len(frame))]
    frame.loc[:, "market_event_id"] = ""
    frame.loc[:, "market_home_team"] = ""
    frame.loc[:, "market_away_team"] = ""
    frame.loc[:, "gap_percentile"] = 0.88
    frame.loc[:, "belief_confidence_factor"] = 0.90
    frame.loc[:, "feasibility"] = 0.90
    frame.loc[:, "market_books"] = 6
    frame.loc[:, "history_rows"] = 80
    frame.loc[:, "edge"] = [2.4 - 0.1 * idx for idx in range(len(frame))]
    frame.loc[:, "abs_edge"] = pd.Series(frame["edge"]).abs()
    frame.loc[:, "market_line"] = [10.0 + idx for idx in range(len(frame))]
    frame.loc[:, "prediction"] = frame["market_line"] + frame["edge"]

    base_board = compute_final_board(
        frame,
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        selection_mode="edge",
        ranking_mode="edge",
        max_plays_per_player=1,
        max_plays_per_target=0,
        max_total_plays=4,
        max_target_plays={"PTS": 10, "TRB": 4, "AST": 4},
        max_plays_per_game=0,
        max_plays_per_script_cluster=0,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.52,
        full_bet_win_rate=0.56,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
    )
    append_board = compute_final_board(
        frame,
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        selection_mode="edge_append_shadow",
        ranking_mode="edge_append_shadow",
        max_plays_per_player=1,
        max_plays_per_target=0,
        max_total_plays=4,
        max_target_plays={"PTS": 10, "TRB": 4, "AST": 4},
        max_plays_per_game=0,
        max_plays_per_script_cluster=0,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.52,
        full_bet_win_rate=0.56,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
        append_agreement_min=1,
        append_edge_percentile_min=0.0,
        append_max_extra_plays=2,
    )
    assert not base_board.empty
    assert not append_board.empty
    assert len(append_board) <= len(base_board) + 2
    assert {"append_shadow_added", "append_anchor_member"}.issubset(set(append_board.columns))

    key_cols = ["player", "target", "direction", "market_line"]
    base_keys = set(tuple(row) for row in base_board.loc[:, key_cols].to_numpy())
    append_keys = set(tuple(row) for row in append_board.loc[:, key_cols].to_numpy())
    assert base_keys.issubset(append_keys)
    assert int(append_board["append_shadow_added"].sum()) >= 1
    assert int(append_board["append_shadow_added"].sum()) <= 2
