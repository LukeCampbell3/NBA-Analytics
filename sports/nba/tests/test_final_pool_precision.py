from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from decision_engine.final_pool_precision import annotate_precision_pool, choose_precision_board_size
from post_process_market_plays import compute_final_board
import post_process_market_plays as post_process


def test_precision_pool_learns_reliable_segment_over_raw_probability() -> None:
    history_rows = []
    for idx in range(24):
        history_rows.append(
            {
                "market_date": "2026-04-20",
                "player": f"A{idx}",
                "target": "AST",
                "direction": "UNDER",
                "estimated_win_rate": 0.56,
                "selection_confidence": 0.30,
                "result": "win" if idx < 21 else "loss",
            }
        )
        history_rows.append(
            {
                "market_date": "2026-04-20",
                "player": f"P{idx}",
                "target": "PTS",
                "direction": "OVER",
                "estimated_win_rate": 0.66,
                "selection_confidence": 0.45,
                "result": "loss" if idx < 18 else "win",
            }
        )
    candidates = pd.DataFrame(
        [
            {
                "market_date": "2026-04-23",
                "player": "Raw Favorite",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.66,
                "final_confidence": 0.45,
                "final_pool_quality_score": 0.60,
            },
            {
                "market_date": "2026-04-23",
                "player": "Reliable Segment",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.57,
                "final_confidence": 0.35,
                "final_pool_quality_score": 0.58,
            },
        ]
    )

    scored, summary = annotate_precision_pool(candidates, history_frame=pd.DataFrame(history_rows))

    assert summary["enabled"] is True
    reliable = scored.loc[scored["player"] == "Reliable Segment"].iloc[0]
    raw = scored.loc[scored["player"] == "Raw Favorite"].iloc[0]
    assert float(reliable["precision_pool_score"]) > float(raw["precision_pool_score"])
    assert float(reliable["precision_pool_prob"]) > float(raw["precision_pool_prob"])


def test_precision_pool_discounts_unsupported_recent_spike_against_stable_history() -> None:
    history_rows = []
    dates = pd.date_range("2026-04-01", periods=30, freq="D")
    for day_index, day in enumerate(dates):
        for row_index in range(2):
            history_rows.append(
                {
                    "market_date": day.strftime("%Y-%m-%d"),
                    "player": f"Stable{day_index}_{row_index}",
                    "target": "AST",
                    "direction": "UNDER",
                    "estimated_win_rate": 0.59,
                    "selection_confidence": 0.35,
                    "result": "loss" if (day_index + row_index) % 4 == 0 else "win",
                }
            )
            history_rows.append(
                {
                    "market_date": day.strftime("%Y-%m-%d"),
                    "player": f"Spike{day_index}_{row_index}",
                    "target": "PTS",
                    "direction": "OVER",
                    "estimated_win_rate": 0.67,
                    "selection_confidence": 0.45,
                    "result": "win" if day_index >= 26 else "loss",
                }
            )
    candidates = pd.DataFrame(
        [
            {
                "market_date": "2026-05-01",
                "player": "Short Spike",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.68,
                "final_confidence": 0.45,
                "final_pool_quality_score": 0.60,
            },
            {
                "market_date": "2026-05-01",
                "player": "Stable Backed",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.60,
                "final_confidence": 0.35,
                "final_pool_quality_score": 0.58,
            },
        ]
    )

    scored, _ = annotate_precision_pool(candidates, history_frame=pd.DataFrame(history_rows))

    spike = scored.loc[scored["player"] == "Short Spike"].iloc[0]
    stable = scored.loc[scored["player"] == "Stable Backed"].iloc[0]
    assert float(spike["precision_pool_regime_delta"]) > 0.0
    assert float(spike["precision_pool_consistency_score"]) < float(stable["precision_pool_consistency_score"])
    assert float(stable["precision_pool_score"]) > float(spike["precision_pool_score"])


def test_precision_pool_allows_supported_recent_regime_to_lift_probability() -> None:
    history_rows = []
    dates = pd.date_range("2026-04-01", periods=30, freq="D")
    for day_index, day in enumerate(dates):
        for row_index in range(3):
            is_recent = day_index >= 16
            history_rows.append(
                {
                    "market_date": day.strftime("%Y-%m-%d"),
                    "player": f"Trend{day_index}_{row_index}",
                    "target": "TRB",
                    "direction": "UNDER",
                    "estimated_win_rate": 0.61,
                    "selection_confidence": 0.40,
                    "result": "win" if (is_recent or row_index == 0) else "loss",
                }
            )
    candidates = pd.DataFrame(
        [
            {
                "market_date": "2026-05-01",
                "player": "Supported Trend",
                "target": "TRB",
                "direction": "UNDER",
                "expected_win_rate": 0.62,
                "final_confidence": 0.40,
                "final_pool_quality_score": 0.62,
            }
        ]
    )

    scored, _ = annotate_precision_pool(candidates, history_frame=pd.DataFrame(history_rows))
    trend = scored.iloc[0]

    assert float(trend["precision_pool_regime_delta"]) > 0.0
    assert float(trend["precision_pool_regime_trust"]) > 0.50
    assert float(trend["precision_pool_prob"]) > float(trend["precision_pool_long_prob"])


def test_precision_pool_final_board_feedback_penalizes_bad_selected_segment() -> None:
    history_rows = []
    for idx in range(80):
        history_rows.append(
            {
                "market_date": "2026-04-20",
                "player": f"CandidateAst{idx}",
                "target": "AST",
                "direction": "UNDER",
                "estimated_win_rate": 0.66,
                "selection_confidence": 0.45,
                "result": "win" if idx < 60 else "loss",
            }
        )
        history_rows.append(
            {
                "market_date": "2026-04-20",
                "player": f"CandidatePts{idx}",
                "target": "PTS",
                "direction": "UNDER",
                "estimated_win_rate": 0.61,
                "selection_confidence": 0.35,
                "result": "win" if idx < 52 else "loss",
            }
        )
    feedback_rows = []
    for idx in range(16):
        feedback_rows.append(
            {
                "run_date": "2026-04-24",
                "mode": "precision_pool",
                "target": "AST",
                "direction": "UNDER",
                "result": "win" if idx < 5 else "loss",
            }
        )
    for idx in range(12):
        feedback_rows.append(
            {
                "run_date": "2026-04-24",
                "mode": "precision_pool",
                "target": "PTS",
                "direction": "UNDER",
                "result": "win" if idx < 9 else "loss",
            }
        )
    candidates = pd.DataFrame(
        [
            {
                "market_date": "2026-04-26",
                "player": "Bad Final Segment",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.66,
                "final_confidence": 0.45,
                "final_pool_quality_score": 0.60,
            },
            {
                "market_date": "2026-04-26",
                "player": "Good Final Segment",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.61,
                "final_confidence": 0.35,
                "final_pool_quality_score": 0.58,
            },
        ]
    )

    scored, summary = annotate_precision_pool(
        candidates,
        history_frame=pd.DataFrame(history_rows),
        feedback_frame=pd.DataFrame(feedback_rows),
    )

    bad = scored.loc[scored["player"] == "Bad Final Segment"].iloc[0]
    good = scored.loc[scored["player"] == "Good Final Segment"].iloc[0]
    assert summary["feedback_rows"] == 28
    assert float(bad["precision_pool_feedback_adjustment"]) < 0.0
    assert float(good["precision_pool_feedback_adjustment"]) > float(bad["precision_pool_feedback_adjustment"])
    assert float(good["precision_pool_score"]) > float(bad["precision_pool_score"])


def test_choose_precision_board_size_prefers_smaller_board_when_tail_dilutes_precision() -> None:
    ranked = pd.DataFrame(
        {
            "precision_pool_prob": [0.88, 0.70, 0.60, 0.58],
            "precision_pool_lcb": [0.84, 0.65, 0.55, 0.52],
        }
    )

    size, summary = choose_precision_board_size(ranked, max_total_plays=4, min_board_plays=1, target_accuracy=0.83)

    assert size == 1
    assert summary["target_attainable"] is True


def test_compute_final_board_precision_pool_uses_dynamic_precision_ranker(monkeypatch) -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Legacy Favorite",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 27.0,
                "market_line": 22.5,
                "expected_win_rate": 0.67,
                "expected_push_rate": 0.0,
                "gap_percentile": 0.95,
                "belief_uncertainty": 1.0,
                "feasibility": 0.95,
                "posterior_alpha": 7.0,
                "posterior_beta": 4.0,
                "posterior_variance": 0.03,
                "abs_edge": 4.5,
                "edge": 4.5,
                "recommendation": "elite",
                "market_date": "2026-04-23",
            },
            {
                "player": "Precision Favorite",
                "target": "AST",
                "direction": "UNDER",
                "prediction": 5.0,
                "market_line": 6.0,
                "expected_win_rate": 0.58,
                "expected_push_rate": 0.0,
                "gap_percentile": 0.95,
                "belief_uncertainty": 1.0,
                "feasibility": 0.95,
                "posterior_alpha": 7.0,
                "posterior_beta": 4.0,
                "posterior_variance": 0.03,
                "abs_edge": 1.0,
                "edge": -1.0,
                "recommendation": "strong",
                "market_date": "2026-04-23",
            },
        ]
    )

    def fake_precision(frame, **kwargs):
        out = frame.copy()
        out["precision_pool_enabled"] = True
        out["precision_pool_prob"] = out["player"].map({"Legacy Favorite": 0.54, "Precision Favorite": 0.89}).astype(float)
        out["precision_pool_lcb"] = out["player"].map({"Legacy Favorite": 0.50, "Precision Favorite": 0.84}).astype(float)
        out["precision_pool_score"] = out["precision_pool_lcb"]
        return out, {"enabled": True, "history_rows": 48, "recent_rows": 48}

    monkeypatch.setattr(post_process, "annotate_precision_pool_fn", fake_precision)

    board = compute_final_board(
        plays,
        selection_mode="precision_pool",
        ranking_mode="precision_pool",
        max_total_plays=2,
        min_board_plays=1,
        min_final_confidence=0.0,
        min_recommendation="pass",
        min_ev=-1.0,
        non_pts_min_gap_percentile=0.0,
    )

    assert board["player"].tolist() == ["Precision Favorite"]
    assert bool(board.iloc[0]["precision_pool_target_attainable"]) is True
