from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from decision_engine.robust_reranker import score_selector_with_robust_reranker
import decision_engine.robust_reranker as robust_reranker
from post_process_market_plays import compute_final_board


def test_robust_reranker_learns_from_prior_rows_only(tmp_path, monkeypatch) -> None:
    history_rows: list[dict] = []
    for idx in range(12):
        history_rows.append(
            {
                "market_date": "2026-04-20",
                "target": "PTS",
                "direction": "OVER",
                "estimated_win_rate": 0.63,
                "estimated_ev": 0.07,
                "selection_confidence": 0.46,
                "abs_edge": 1.4,
                "uncertainty_sigma": 4.2,
                "history_rows": 110,
                "spike_probability": 0.48,
                "market_line": 24.5,
                "result": "loss" if idx < 9 else "win",
            }
        )
        history_rows.append(
            {
                "market_date": "2026-04-20",
                "target": "AST",
                "direction": "UNDER",
                "estimated_win_rate": 0.58,
                "estimated_ev": 0.03,
                "selection_confidence": 0.43,
                "abs_edge": 0.9,
                "uncertainty_sigma": 1.2,
                "history_rows": 105,
                "spike_probability": 0.28,
                "market_line": 5.5,
                "result": "win" if idx < 9 else "loss",
            }
        )
    for idx in range(8):
        history_rows.append(
            {
                "market_date": "2026-05-03",
                "target": "PTS",
                "direction": "OVER",
                "estimated_win_rate": 0.63,
                "estimated_ev": 0.07,
                "selection_confidence": 0.46,
                "abs_edge": 1.4,
                "uncertainty_sigma": 4.2,
                "history_rows": 110,
                "spike_probability": 0.48,
                "market_line": 24.5,
                "result": "win" if idx < 7 else "loss",
            }
        )

    selector = pd.DataFrame(
        [
            {
                "player": "Volume Scorer",
                "market_date": "2026-05-01",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.63,
                "ev": 0.07,
                "final_confidence": 0.46,
                "recommendation": "strong",
                "abs_edge": 1.4,
                "uncertainty_sigma": 4.2,
                "history_rows": 110,
                "spike_probability": 0.48,
                "market_line": 24.5,
            },
            {
                "player": "Steady Facilitator",
                "market_date": "2026-05-01",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.58,
                "ev": 0.03,
                "final_confidence": 0.43,
                "recommendation": "consider",
                "abs_edge": 0.9,
                "uncertainty_sigma": 1.2,
                "history_rows": 105,
                "spike_probability": 0.28,
                "market_line": 5.5,
            },
        ]
    )

    empty_root = tmp_path / "empty_history_roots"
    empty_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(robust_reranker, "ANALYSIS_ROOT", empty_root)
    monkeypatch.setattr(robust_reranker, "SHARED_VALIDATION_ROOT", empty_root)

    scored, summary = score_selector_with_robust_reranker(
        selector,
        pd.DataFrame(history_rows),
        probability_shrink_factor=0.70,
        min_train_rows=12,
        holdout_days=0,
        min_holdout_rows=0,
        min_candidate_expected_win_rate=0.55,
        min_candidate_final_confidence=0.03,
        min_candidate_recommendation="consider",
    )

    assert bool(summary["enabled"]) is True
    assert int(summary["pre_cutoff_rows"]) == 24
    assert float(scored.loc[scored["player"] == "Steady Facilitator", "robust_reranker_prob"].iloc[0]) > float(
        scored.loc[scored["player"] == "Volume Scorer", "robust_reranker_prob"].iloc[0]
    )


def test_compute_final_board_prefers_robust_reranker_score_when_requested() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "High EV Legacy",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 29.0,
                "market_line": 24.5,
                "abs_edge": 4.5,
                "edge": 4.5,
                "expected_win_rate": 0.67,
                "expected_push_rate": 0.0,
                "posterior_alpha": 9.0,
                "posterior_beta": 4.0,
                "posterior_variance": 0.03,
                "gap_percentile": 0.96,
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
                "market_player_raw": "High EV Legacy",
                "game_key": "g1",
                "robust_reranker_prob": 0.31,
                "robust_reranker_blend_raw": 0.30,
            },
            {
                "player": "Meta Selector Favorite",
                "target": "AST",
                "direction": "UNDER",
                "prediction": 5.2,
                "market_line": 6.0,
                "abs_edge": 0.8,
                "edge": -0.8,
                "expected_win_rate": 0.60,
                "expected_push_rate": 0.0,
                "posterior_alpha": 8.0,
                "posterior_beta": 5.0,
                "posterior_variance": 0.03,
                "gap_percentile": 0.90,
                "belief_uncertainty": 0.80,
                "belief_confidence_factor": 0.90,
                "feasibility": 0.92,
                "recommendation": "strong",
                "history_rows": 110,
                "market_date": "2026-05-01",
                "last_history_date": "2026-04-30",
                "market_event_id": "g2",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "market_player_raw": "Meta Selector Favorite",
                "game_key": "g2",
                "robust_reranker_prob": 0.84,
                "robust_reranker_blend_raw": 0.82,
            },
        ]
    )

    board = compute_final_board(
        plays,
        max_total_plays=1,
        max_plays_per_player=1,
        max_plays_per_game=1,
        max_plays_per_script_cluster=1,
        selection_mode="robust_reranker",
        ranking_mode="robust_reranker",
        min_recommendation="pass",
        min_ev=-1.0,
        min_final_confidence=0.0,
    )

    assert len(board) == 1
    assert str(board.iloc[0]["player"]) == "Meta Selector Favorite"


def test_robust_reranker_discovers_richer_history_source(tmp_path, monkeypatch) -> None:
    analysis_root = tmp_path / "analysis"
    validation_root = tmp_path / "validation"
    analysis_root.mkdir(parents=True, exist_ok=True)
    validation_root.mkdir(parents=True, exist_ok=True)

    richer_rows: list[dict] = []
    for day in ("2026-04-20", "2026-04-21", "2026-04-22"):
        for idx in range(6):
            richer_rows.append(
                {
                    "market_date": day,
                    "target": "AST",
                    "direction": "UNDER",
                    "estimated_win_rate": 0.59,
                    "estimated_ev": 0.04,
                    "selection_confidence": 0.45,
                    "abs_edge": 1.0,
                    "uncertainty_sigma": 1.1,
                    "history_rows": 95,
                    "spike_probability": 0.25,
                    "market_line": 5.5,
                    "result": "win" if idx < 5 else "loss",
                }
            )
            richer_rows.append(
                {
                    "market_date": day,
                    "target": "PTS",
                    "direction": "OVER",
                    "estimated_win_rate": 0.62,
                    "estimated_ev": 0.06,
                    "selection_confidence": 0.44,
                    "abs_edge": 1.2,
                    "uncertainty_sigma": 4.0,
                    "history_rows": 90,
                    "spike_probability": 0.45,
                    "market_line": 24.5,
                    "result": "loss" if idx < 5 else "win",
                }
            )
    pd.DataFrame(richer_rows).to_csv(validation_root / "validation_recent_pool_selector_20260420_20260422_rows.csv", index=False)

    tiny_history = pd.DataFrame(
        [
            {
                "market_date": "2026-04-22",
                "target": "PTS",
                "direction": "OVER",
                "estimated_win_rate": 0.62,
                "estimated_ev": 0.06,
                "selection_confidence": 0.44,
                "abs_edge": 1.2,
                "uncertainty_sigma": 4.0,
                "history_rows": 90,
                "spike_probability": 0.45,
                "market_line": 24.5,
                "result": "win",
            },
            {
                "market_date": "2026-04-22",
                "target": "AST",
                "direction": "UNDER",
                "estimated_win_rate": 0.59,
                "estimated_ev": 0.04,
                "selection_confidence": 0.45,
                "abs_edge": 1.0,
                "uncertainty_sigma": 1.1,
                "history_rows": 95,
                "spike_probability": 0.25,
                "market_line": 5.5,
                "result": "loss",
            },
        ]
    )

    selector = pd.DataFrame(
        [
            {
                "player": "Discover Target",
                "market_date": "2026-04-23",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.58,
                "ev": 0.03,
                "final_confidence": 0.42,
                "recommendation": "consider",
                "abs_edge": 0.9,
                "uncertainty_sigma": 1.0,
                "history_rows": 80,
                "spike_probability": 0.20,
                "market_line": 5.5,
            }
        ]
    )

    monkeypatch.setattr(robust_reranker, "ANALYSIS_ROOT", analysis_root)
    monkeypatch.setattr(robust_reranker, "SHARED_VALIDATION_ROOT", validation_root)

    scored, summary = score_selector_with_robust_reranker(
        selector,
        tiny_history,
        min_train_rows=12,
        holdout_days=0,
        min_holdout_rows=0,
        min_candidate_expected_win_rate=0.50,
        min_candidate_final_confidence=0.0,
        min_candidate_recommendation="pass",
    )

    assert bool(summary["enabled"]) is True
    assert "validation_recent_pool_selector_20260420_20260422_rows.csv" in str(summary["history_source"]["path"])
    assert int(summary["pre_cutoff_rows"]) >= 30
    assert bool(scored["robust_reranker_enabled"].iloc[0]) is True
