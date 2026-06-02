from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from post_process_market_plays import compute_final_board
from research.common import build_candidate_id


def _selector_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Penalty Favorite",
                "market_player_raw": "Penalty Favorite",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 24.8,
                "market_line": 21.5,
                "abs_edge": 2.40,
                "edge": 2.40,
                "expected_win_rate": 0.680,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.76,
                "feasibility": 0.92,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_pts_a",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "gap_percentile": 0.96,
            },
            {
                "player": "Safe Backup",
                "market_player_raw": "Safe Backup",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 23.9,
                "market_line": 21.5,
                "abs_edge": 2.10,
                "edge": 2.10,
                "expected_win_rate": 0.664,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.74,
                "feasibility": 0.91,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_pts_b",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "gap_percentile": 0.94,
            },
            {
                "player": "Downgrade Me",
                "market_player_raw": "Downgrade Me",
                "target": "TRB",
                "direction": "OVER",
                "prediction": 9.8,
                "market_line": 8.5,
                "abs_edge": 1.30,
                "edge": 1.30,
                "expected_win_rate": 0.646,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.79,
                "feasibility": 0.88,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_trb_over",
                "market_home_team": "EEE",
                "market_away_team": "FFF",
                "gap_percentile": 0.92,
            },
            {
                "player": "Opposite Under Audit",
                "market_player_raw": "Opposite Under Audit",
                "target": "TRB",
                "direction": "UNDER",
                "prediction": 7.3,
                "market_line": 8.5,
                "abs_edge": 1.20,
                "edge": -1.20,
                "expected_win_rate": 0.638,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.77,
                "feasibility": 0.87,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_trb_under",
                "market_home_team": "EEE",
                "market_away_team": "FFF",
                "gap_percentile": 0.91,
            },
            {
                "player": "Alt Line Watch",
                "market_player_raw": "Alt Line Watch",
                "target": "AST",
                "direction": "OVER",
                "prediction": 6.9,
                "market_line": 5.5,
                "abs_edge": 1.40,
                "edge": 1.40,
                "expected_win_rate": 0.632,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.80,
                "feasibility": 0.86,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_ast_a",
                "market_home_team": "GGG",
                "market_away_team": "HHH",
                "gap_percentile": 0.90,
            },
            {
                "player": "Unrelated AST",
                "market_player_raw": "Unrelated AST",
                "target": "AST",
                "direction": "UNDER",
                "prediction": 4.7,
                "market_line": 5.5,
                "abs_edge": 0.80,
                "edge": -0.80,
                "expected_win_rate": 0.604,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.78,
                "feasibility": 0.84,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_ast_b",
                "market_home_team": "III",
                "market_away_team": "JJJ",
                "gap_percentile": 0.88,
            },
        ]
    )


def _candidate_id_map(frame: pd.DataFrame) -> dict[str, str]:
    working = frame.copy()
    working["candidate_id"] = build_candidate_id(working)
    return dict(zip(working["player"], working["candidate_id"]))


def _run_board(
    plays: pd.DataFrame,
    *,
    adjustments: pd.DataFrame | None = None,
    min_recommendation: str = "consider",
    max_total_plays: int = 10,
) -> pd.DataFrame:
    return compute_final_board(
        plays,
        selection_mode="ev_adjusted",
        ranking_mode="ev_adjusted",
        min_recommendation=min_recommendation,
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=max_total_plays,
        max_plays_per_game=0,
        max_plays_per_script_cluster=0,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.40,
        medium_bet_win_rate=0.50,
        full_bet_win_rate=0.60,
        failure_mode_adjustments=adjustments,
    )


def test_penalty_lowers_candidate_ranking() -> None:
    plays = _selector_rows()
    candidate_ids = _candidate_id_map(plays)
    baseline = _run_board(plays)
    adjusted = _run_board(
        plays,
        adjustments=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_ids["Penalty Favorite"],
                    "failure_mode_id": "MARKET_PRICE_MISPLACEMENT",
                    "penalty": 0.18,
                    "downgrade_tier": "",
                    "veto_flag": False,
                    "opposite_side_candidate_flag": False,
                    "alt_line_candidate_flag": False,
                    "explanation": "penalty_ranking_test",
                }
            ]
        ),
    )
    baseline_rank = int(baseline.loc[baseline["player"] == "Penalty Favorite", "selected_rank"].iloc[0])
    adjusted_rank = int(adjusted.loc[adjusted["player"] == "Penalty Favorite", "selected_rank"].iloc[0])
    safe_rank_after = int(adjusted.loc[adjusted["player"] == "Safe Backup", "selected_rank"].iloc[0])
    assert baseline_rank == 1
    assert adjusted_rank > baseline_rank
    assert safe_rank_after < adjusted_rank


def test_downgrade_tier_changes_final_candidate_status() -> None:
    plays = _selector_rows()
    candidate_ids = _candidate_id_map(plays)
    adjusted = _run_board(
        plays,
        adjustments=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_ids["Downgrade Me"],
                    "failure_mode_id": "REBOUND_LOW_LINE_ROLE_VOLATILITY",
                    "penalty": 0.0,
                    "downgrade_tier": "pass",
                    "veto_flag": False,
                    "opposite_side_candidate_flag": False,
                    "alt_line_candidate_flag": False,
                    "explanation": "downgrade_to_pass",
                }
            ]
        ),
        min_recommendation="consider",
    )
    assert "Downgrade Me" not in adjusted["player"].tolist()


def test_veto_flag_prevents_board_inclusion() -> None:
    plays = _selector_rows()
    candidate_ids = _candidate_id_map(plays)
    adjusted = _run_board(
        plays,
        adjustments=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_ids["Penalty Favorite"],
                    "failure_mode_id": "MINUTES_BAND_FAILURE",
                    "penalty": 0.10,
                    "downgrade_tier": "pass",
                    "veto_flag": True,
                    "opposite_side_candidate_flag": False,
                    "alt_line_candidate_flag": False,
                    "explanation": "veto_test",
                }
            ]
        ),
    )
    assert "Penalty Favorite" not in adjusted["player"].tolist()


def test_opposite_side_candidate_flag_adds_an_auditable_candidate() -> None:
    plays = _selector_rows()
    candidate_ids = _candidate_id_map(plays)
    adjusted = _run_board(
        plays,
        adjustments=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_ids["Opposite Under Audit"],
                    "failure_mode_id": "OPPOSITE_SIDE_SIGNAL",
                    "penalty": 0.0,
                    "downgrade_tier": "",
                    "veto_flag": False,
                    "opposite_side_candidate_flag": True,
                    "alt_line_candidate_flag": False,
                    "explanation": "opposite_side_audit",
                }
            ]
        ),
    )
    row = adjusted.loc[adjusted["player"] == "Opposite Under Audit"].iloc[0]
    assert bool(row["failure_mode_opposite_side_candidate_flag"]) is True
    assert row["failure_mode_ids"] == "OPPOSITE_SIDE_SIGNAL"
    assert "opposite_side_audit" in str(row["failure_mode_explanation"])


def test_alt_line_candidate_flag_does_not_mutate_unrelated_rows() -> None:
    plays = _selector_rows()
    candidate_ids = _candidate_id_map(plays)
    baseline = _run_board(plays)
    adjusted = _run_board(
        plays,
        adjustments=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_ids["Alt Line Watch"],
                    "failure_mode_id": "ALT_LINE_MISFRAMING",
                    "penalty": 0.0,
                    "downgrade_tier": "",
                    "veto_flag": False,
                    "opposite_side_candidate_flag": False,
                    "alt_line_candidate_flag": True,
                    "explanation": "alt_line_audit",
                }
            ]
        ),
    )
    alt_row = adjusted.loc[adjusted["player"] == "Alt Line Watch"].iloc[0]
    unrelated_adjusted = adjusted.loc[adjusted["player"] == "Unrelated AST"].iloc[0]
    unrelated_baseline = baseline.loc[baseline["player"] == "Unrelated AST"].iloc[0]
    assert bool(alt_row["failure_mode_alt_line_candidate_flag"]) is True
    assert bool(unrelated_adjusted["failure_mode_alt_line_candidate_flag"]) is False
    pdt.assert_series_equal(
        unrelated_adjusted[["expected_win_rate", "ev_adjusted", "recommendation"]],
        unrelated_baseline[["expected_win_rate", "ev_adjusted", "recommendation"]],
        check_names=False,
    )


def test_after_failure_mode_adjustments_stage_is_recorded() -> None:
    plays = _selector_rows()
    board = _run_board(plays)
    assert "after_failure_mode_adjustments" in board.attrs["stage_counts"]
    assert int(board.attrs["stage_counts"]["after_failure_mode_adjustments"]) == len(board)


def test_no_adjustments_preserves_exact_baseline_board() -> None:
    plays = _selector_rows()
    baseline = _run_board(plays)
    empty_adjustments = pd.DataFrame(
        columns=[
            "candidate_id",
            "failure_mode_id",
            "penalty",
            "downgrade_tier",
            "veto_flag",
            "opposite_side_candidate_flag",
            "alt_line_candidate_flag",
            "explanation",
        ]
    )
    adjusted = _run_board(plays, adjustments=empty_adjustments)
    pdt.assert_frame_equal(baseline.reset_index(drop=True), adjusted.reset_index(drop=True), check_dtype=False)
    assert baseline.attrs["stage_counts"] == adjusted.attrs["stage_counts"]


def test_non_target_markets_remain_unchanged() -> None:
    plays = _selector_rows()
    candidate_ids = _candidate_id_map(plays)
    baseline = _run_board(plays)
    adjusted = _run_board(
        plays,
        adjustments=pd.DataFrame(
            [
                {
                    "candidate_id": candidate_ids["Downgrade Me"],
                    "failure_mode_id": "REBOUND_SHARE_COMPETITION",
                    "penalty": 0.12,
                    "downgrade_tier": "",
                    "veto_flag": False,
                    "opposite_side_candidate_flag": False,
                    "alt_line_candidate_flag": False,
                    "explanation": "trb_only_penalty",
                }
            ]
        ),
    )
    baseline_row = baseline.loc[baseline["player"] == "Unrelated AST"].iloc[0]
    adjusted_row = adjusted.loc[adjusted["player"] == "Unrelated AST"].iloc[0]
    pdt.assert_series_equal(
        adjusted_row[["expected_win_rate", "ev_adjusted", "recommendation", "failure_mode_total_penalty"]],
        pd.Series(
            [
                baseline_row["expected_win_rate"],
                baseline_row["ev_adjusted"],
                baseline_row["recommendation"],
                0.0,
            ],
            index=["expected_win_rate", "ev_adjusted", "recommendation", "failure_mode_total_penalty"],
        ),
        check_names=False,
    )
