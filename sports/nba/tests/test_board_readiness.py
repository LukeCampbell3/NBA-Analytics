from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from decision_engine.board_readiness import annotate_board_readiness
from post_process_market_plays import compute_final_board
from report_board_readiness import build_board_readiness_report


def _selector_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "High Risk One",
                "market_player_raw": "High Risk One",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 25.0,
                "market_line": 21.5,
                "abs_edge": 2.8,
                "edge": 2.8,
                "expected_win_rate": 0.68,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 1.12,
                "feasibility": 0.90,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_a",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "script_cluster_id": "script_a",
                "gap_percentile": 0.96,
                "line_decision_fragility_score": 0.62,
                "line_decision_instability_score": 0.48,
                "final_pool_quality_score": 0.49,
                "recency_factor": 0.62,
                "noise_score": 0.40,
                "contradiction_score": 0.33,
                "price_validity_status": "MISSING_PRICE",
                "timestamp_safe_flag": False,
                "market_side_price": None,
                "market_side_break_even": None,
            },
            {
                "player": "High Risk Two",
                "market_player_raw": "High Risk Two",
                "target": "AST",
                "direction": "OVER",
                "prediction": 7.0,
                "market_line": 5.5,
                "abs_edge": 1.5,
                "edge": 1.5,
                "expected_win_rate": 0.63,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 1.05,
                "feasibility": 0.88,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_a",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "script_cluster_id": "script_a",
                "gap_percentile": 0.91,
                "line_decision_fragility_score": 0.58,
                "line_decision_instability_score": 0.46,
                "final_pool_quality_score": 0.53,
                "recency_factor": 0.68,
                "noise_score": 0.32,
                "contradiction_score": 0.15,
                "price_validity_status": "STALE_PRICE",
                "timestamp_safe_flag": False,
                "market_side_price": -110,
                "market_side_break_even": 0.5238,
            },
            {
                "player": "Stable Row",
                "market_player_raw": "Stable Row",
                "target": "TRB",
                "direction": "UNDER",
                "prediction": 7.2,
                "market_line": 8.5,
                "abs_edge": 1.3,
                "edge": -1.3,
                "expected_win_rate": 0.61,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.80,
                "feasibility": 0.89,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_b",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "script_cluster_id": "script_b",
                "gap_percentile": 0.89,
                "line_decision_fragility_score": 0.20,
                "line_decision_instability_score": 0.10,
                "final_pool_quality_score": 0.71,
                "recency_factor": 0.92,
                "noise_score": 0.05,
                "contradiction_score": 0.02,
                "price_validity_status": "PRICE_VALID",
                "timestamp_safe_flag": True,
                "market_side_price": -108,
                "market_side_break_even": 0.5192,
            },
        ]
    )


def test_annotate_board_readiness_flags_fragility_and_price_risk() -> None:
    annotated, summary = annotate_board_readiness(_selector_rows())
    assert "board_readiness_risk_score" in annotated.columns
    assert "board_readiness_reasons" in annotated.columns
    assert summary["board_readiness_status"] == "BLOCKER"
    assert "timestamp_safe_price_evidence_incomplete" in summary["blocked_reasons"]
    assert "same_game_concentration_elevated" in summary["blocked_reasons"]
    row = annotated.loc[annotated["player"] == "High Risk One"].iloc[0]
    assert bool(row["board_readiness_price_untrusted_flag"])
    assert bool(row["board_readiness_high_uncertainty_flag"])
    assert "price_untrusted" in str(row["board_readiness_reasons"])


def test_timestamp_safe_price_rows_can_clear_readiness() -> None:
    rows = _selector_rows().head(1).copy()
    rows["player"] = "Single Stable"
    rows["market_player_raw"] = "Single Stable"
    rows["belief_uncertainty"] = 0.80
    rows["line_decision_fragility_score"] = 0.10
    rows["line_decision_instability_score"] = 0.08
    rows["final_pool_quality_score"] = 0.79
    rows["recency_factor"] = 0.95
    rows["noise_score"] = 0.05
    rows["contradiction_score"] = 0.02
    rows["price_validity_status"] = "PRICE_VALID"
    rows["timestamp_safe_flag"] = True
    rows["market_side_price"] = -110
    rows["market_side_break_even"] = 110.0 / 210.0
    annotated, summary = annotate_board_readiness(rows)
    assert summary["production_readiness_clear"] is True
    assert summary["blocked_reasons"] == []
    assert annotated.iloc[0]["board_readiness_status"] == "STABLE"


def test_compute_final_board_adds_readiness_without_changing_board_shape() -> None:
    result = compute_final_board(
        _selector_rows(),
        selection_mode="ev_adjusted",
        ranking_mode="ev_adjusted",
        min_recommendation="consider",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=10,
        max_plays_per_game=0,
        max_plays_per_script_cluster=0,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.40,
        medium_bet_win_rate=0.50,
        full_bet_win_rate=0.60,
    )
    assert len(result) == 3
    assert "board_readiness_risk_score" in result.columns
    summary = result.attrs["board_readiness_summary"]
    assert summary["row_count"] == 3
    assert summary["board_readiness_status"] == "BLOCKER"


def test_report_board_readiness_writes_outputs(tmp_path: Path) -> None:
    board_csv = tmp_path / "board.csv"
    _selector_rows().to_csv(board_csv, index=False)
    outputs = build_board_readiness_report(pd.read_csv(board_csv), tmp_path / "audit" / "board_readiness")
    assert Path(outputs["rows_csv"]).exists()
    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["summary_md"]).exists()
    payload = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["board_readiness_status"] == "BLOCKER"
