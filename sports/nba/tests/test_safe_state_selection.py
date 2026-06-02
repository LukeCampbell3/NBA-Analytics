from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.evaluate_safe_state_shadow_results import evaluate_safe_state_shadow_results
from research.safe_state.player_state_forecastability import annotate_player_state_forecastability
from research.safe_state.safe_state_classifier import annotate_safe_state
from research.safe_state.safe_state_evidence_gap_report import build_safe_state_evidence_gap_report
from research.safe_state.safe_state_shadow_boards import build_safe_state_shadow_boards
from research.safe_state.similar_state_reliability import annotate_similar_state_reliability
from research.safe_state.structural_line_mispricing import annotate_structural_line_mispricing


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::safe",
        "game_id": "game_1",
        "game_date": "2026-05-26",
        "market_date": "2026-05-26",
        "player": "Safe Player",
        "player_name": "Safe Player",
        "team": "OKC",
        "opponent": "SAS",
        "target": "PTS",
        "side": "UNDER",
        "direction": "UNDER",
        "market_type": "PTS_UNDER",
        "line": 24.5,
        "market_line": 24.5,
        "market_side_price": -110.0,
        "market_side_break_even": 0.5238,
        "price_validity_status": "PRICE_VALID",
        "edge_defendability_tier": "EDGE_DEFENDABLE",
        "stress_probability": 0.60,
        "lcb_probability": 0.56,
        "stress_edge": 0.0762,
        "lcb_edge": 0.0362,
        "expected_minutes_band_low": 32.0,
        "expected_minutes_band_high": 36.0,
        "expected_minutes_band_width": 4.0,
        "minutes_floor_recent": 31.0,
        "minutes_p25_recent": 33.0,
        "minutes_median_recent": 35.0,
        "minutes_range_recent": 5.0,
        "starter_status_recent": "stable_starter",
        "starter_status_change_count": 0,
        "rotation_volatility_score": 0.10,
        "blowout_minutes_sensitivity": 0.10,
        "foul_rate_minutes_loss_risk": 0.05,
        "usage_volatility": 0.10,
        "fga_volatility": 0.10,
        "assist_opportunity_volatility": 0.15,
        "rebound_chance_volatility": 0.15,
        "opponent_context_similarity": 0.80,
        "teammate_context_score": 0.80,
        "opponent_context_score": 0.80,
        "scenario_agreement": 0.80,
        "chaos_score": 0.20,
        "similar_state_count": 8,
        "similar_state_tightness_score": 0.82,
        "similar_state_reliability_tier": "TIGHT",
        "similar_state_win_rate": 0.75,
        "model_mean": 20.5,
        "uncertainty_sigma": 1.5,
        "q25": 19.0,
        "q50": 20.5,
        "q75": 22.0,
        "q90": 23.5,
        "structural_pathway_score": 0.80,
        "market_role_mismatch_score": 0.65,
    }
    row.update(overrides)
    return row


def _history(count: int = 6) -> pd.DataFrame:
    values = [21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8][:count]
    return pd.DataFrame(
        [
            {
                "candidate_id": f"hist::{idx}",
                "game_date": f"2026-05-{10 + idx:02d}",
                "market_date": f"2026-05-{10 + idx:02d}",
                "player": "Safe Player",
                "player_name": "Safe Player",
                "target": "PTS",
                "side": "OVER",
                "line": 20.5,
                "actual_stat": value,
                "actual_result": "WIN",
            }
            for idx, value in enumerate(values)
        ]
    )


def test_forecastability_high_when_minutes_usage_role_are_stable() -> None:
    annotated = annotate_player_state_forecastability(pd.DataFrame([_row()]))

    assert annotated.iloc[0]["forecastability_tier"] == "HIGH_FORECASTABILITY"
    assert annotated.iloc[0]["overall_player_forecastability_score"] >= 0.75


def test_forecastability_low_when_minutes_band_wide_or_role_unstable() -> None:
    annotated = annotate_player_state_forecastability(
        pd.DataFrame(
            [
                _row(
                    expected_minutes_band_low=12,
                    expected_minutes_band_high=28,
                    expected_minutes_band_width=16,
                    minutes_floor_recent=12,
                    starter_status_recent="bench_uncertain",
                    starter_status_change_count=3,
                    rotation_volatility_score=0.90,
                    chaos_score=0.70,
                )
            ]
        )
    )

    assert annotated.iloc[0]["forecastability_tier"] in {"LOW_FORECASTABILITY", "UNFORECASTABLE"}
    assert bool(annotated.iloc[0]["forecastability_blocks_safe_state_flag"]) is True


def test_similar_state_reliability_tight_when_residuals_cluster() -> None:
    annotated = annotate_similar_state_reliability(pd.DataFrame([_row(side="OVER", line=20.5, market_line=20.5)]), _history(6), min_count=5)

    assert annotated.iloc[0]["similar_state_reliability_tier"] == "TIGHT"
    assert annotated.iloc[0]["similar_state_tightness_score"] >= 0.75


def test_similar_state_reliability_insufficient_when_sample_small() -> None:
    annotated = annotate_similar_state_reliability(pd.DataFrame([_row(side="OVER", line=20.5, market_line=20.5)]), _history(2), min_count=5)

    assert annotated.iloc[0]["similar_state_reliability_tier"] == "INSUFFICIENT_SAMPLE"


def test_structural_mispricing_not_triggered_by_price_edge_alone() -> None:
    annotated = annotate_structural_line_mispricing(
        pd.DataFrame(
            [
                _row(
                    model_mean=pd.NA,
                    q25=pd.NA,
                    q50=pd.NA,
                    q75=pd.NA,
                    q90=pd.NA,
                    structural_pathway_score=0.50,
                    similar_state_count=0,
                    similar_state_win_rate=pd.NA,
                    market_role_mismatch_score=0.0,
                )
            ]
        )
    )

    assert annotated.iloc[0]["structural_mispricing_tier"] in {"PRICE_ONLY_EDGE", "UNKNOWN"}
    assert not str(annotated.iloc[0]["structural_mispricing_tier"]).startswith("STRUCTURAL_MISPRICE")


def test_structural_mispricing_requires_distribution_and_pathway_support() -> None:
    annotated = annotate_structural_line_mispricing(pd.DataFrame([_row()]))

    assert annotated.iloc[0]["structural_mispricing_tier"] in {"STRUCTURAL_MISPRICE_STRONG", "STRUCTURAL_MISPRICE_ACCEPTABLE"}
    assert annotated.iloc[0]["overall_structural_mispricing_score"] >= 0.58


def test_safe_state_core_requires_price_forecastability_and_structural_mispricing() -> None:
    annotated = annotate_safe_state(annotate_structural_line_mispricing(annotate_player_state_forecastability(pd.DataFrame([_row()]))))

    assert annotated.iloc[0]["safe_state_tier"] == "SAFE_STATE_CORE"
    assert "price_defendable" in annotated.iloc[0]["safe_state_reasons"]


def test_edge_defendable_with_low_forecastability_becomes_unstable() -> None:
    base = _row(forecastability_tier="LOW_FORECASTABILITY", overall_player_forecastability_score=0.40)
    annotated = annotate_safe_state(pd.DataFrame([base]))

    assert annotated.iloc[0]["safe_state_tier"] == "SAFE_STATE_UNSTABLE"


def test_edge_defendable_with_weak_structural_logic_becomes_structurally_weak() -> None:
    base = _row(
        forecastability_tier="HIGH_FORECASTABILITY",
        overall_player_forecastability_score=0.85,
        structural_mispricing_tier="PRICE_ONLY_EDGE",
        overall_structural_mispricing_score=0.30,
    )
    annotated = annotate_safe_state(pd.DataFrame([base]))

    assert annotated.iloc[0]["safe_state_tier"] == "SAFE_STATE_STRUCTURALLY_WEAK"


def test_missing_similar_state_data_becomes_insufficient_evidence() -> None:
    base = _row(
        forecastability_tier="HIGH_FORECASTABILITY",
        overall_player_forecastability_score=0.85,
        structural_mispricing_tier="STRUCTURAL_MISPRICE_STRONG",
        overall_structural_mispricing_score=0.80,
        similar_state_reliability_tier="INSUFFICIENT_SAMPLE",
        similar_state_tightness_score=0.20,
    )
    annotated = annotate_safe_state(pd.DataFrame([base]))

    assert annotated.iloc[0]["safe_state_tier"] == "SAFE_STATE_INSUFFICIENT_EVIDENCE"


def test_safe_state_boards_never_mutate_production_output(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    production_csv = tmp_path / "production.csv"
    history_csv = tmp_path / "history.csv"
    output_dir = tmp_path / "safe_state_boards"
    production = pd.DataFrame([_row(candidate_id="candidate::prod")])
    candidates = pd.DataFrame([_row(candidate_id="candidate::prod"), _row(candidate_id="candidate::shadow", player="Shadow Player")])

    candidates.to_csv(candidate_csv, index=False)
    production.to_csv(production_csv, index=False)
    _history(6).to_csv(history_csv, index=False)

    report = build_safe_state_shadow_boards(
        output_dir=output_dir,
        candidate_pool_csv=candidate_csv,
        production_board_csv=production_csv,
        historical_csv=history_csv,
        board_size=1,
    )

    reloaded_production = pd.read_csv(output_dir / "production_board_as_is.csv")
    assert report["production_behavior_changed"] is False
    assert reloaded_production["candidate_id"].tolist() == ["candidate::prod"]
    assert (output_dir / "safe_state_core_board.csv").exists()


def test_settlement_evaluator_refuses_promotion_claims_from_one_slate(tmp_path: Path) -> None:
    board_dir = tmp_path / "boards"
    board_dir.mkdir()
    frame = pd.DataFrame([_row(actual_result="WIN")])
    for name in [
        "production_board_as_is",
        "price_defense_only_board",
        "forecastable_price_board",
        "structural_misprice_board",
        "safe_state_core_board",
        "safe_state_expanded_board",
    ]:
        frame.to_csv(board_dir / f"{name}.csv", index=False)

    report = evaluate_safe_state_shadow_results(board_dir=board_dir)

    assert report["promotion_ready"] is False
    assert report["promotion_claim"] is False
    assert report["status"] == "NEEDS_MORE_SAMPLE"
    assert "single_slate_or_insufficient_windows" in report["blocked_reasons"]
    assert json.loads((board_dir / "safe_state_shadow_settlement_report.json").read_text(encoding="utf-8"))["promotion_ready"] is False


def test_evidence_gap_report_identifies_near_core_and_feature_rankings(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    production_csv = tmp_path / "production.csv"
    safe_state_dir = tmp_path / "safe_state"
    safe_state_dir.mkdir()

    near_core = _row(
        candidate_id="candidate::near-core",
        safe_state_tier="SAFE_STATE_STRUCTURALLY_WEAK",
        forecastability_tier="HIGH_FORECASTABILITY",
        overall_player_forecastability_score=0.82,
        similar_state_reliability_tier="TIGHT",
        similar_state_tightness_score=0.82,
        structural_mispricing_tier="PRICE_ONLY_EDGE",
        overall_structural_mispricing_score=0.30,
    )
    unsafe = _row(
        candidate_id="candidate::unsafe",
        edge_defendability_tier="EDGE_UNTRUSTED_PRICE",
        price_validity_status="MISSING_PRICE",
        safe_state_tier="SAFE_STATE_REJECT",
        forecastability_tier="UNFORECASTABLE",
        similar_state_reliability_tier="INSUFFICIENT_SAMPLE",
        structural_mispricing_tier="UNKNOWN",
    )
    annotated = pd.DataFrame([near_core, unsafe])
    annotated.to_csv(candidate_csv, index=False)
    annotated.iloc[[0]].to_csv(production_csv, index=False)
    annotated.to_csv(safe_state_dir / "safe_state_annotated_candidates.csv", index=False)
    annotated.iloc[[0]].to_csv(safe_state_dir / "price_defense_only_board.csv", index=False)

    report = build_safe_state_evidence_gap_report(
        output_dir=safe_state_dir,
        candidate_pool_csv=candidate_csv,
        production_board_csv=production_csv,
        safe_state_dir=safe_state_dir,
    )

    blockers = pd.read_csv(safe_state_dir / "safe_state_candidate_blockers.csv")
    rankings = pd.read_csv(safe_state_dir / "safe_state_feature_gap_rankings.csv")

    assert report["safe_state_core_count"] == 0
    assert report["near_core_count"] == 1
    assert report["price_defense_only_board_rows"] == 1
    assert blockers.loc[blockers["candidate_id"] == "candidate::near-core", "safe_state_gap_tier"].iloc[0] == "SAFE_STATE_NEAR_CORE"
    assert "STRUCTURAL_MISPRICING_GAP" in blockers.loc[blockers["candidate_id"] == "candidate::near-core", "primary_blocker"].iloc[0]
    assert not rankings.empty
    assert (safe_state_dir / "safe_state_evidence_gap_report.json").exists()
    assert (safe_state_dir / "safe_state_evidence_gap_report.md").exists()
    assert report["production_behavior_changed"] is False
    assert report["promotion_claim"] is False
