from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from report_rebound_promotion_gate import build_promotion_gate
from validate_rebound_diagnostics import (
    ACTIVE_WINDOW,
    MIXED_WINDOW,
    NO_OP_WINDOW,
    _build_opposite_under_audit,
    _compare_to_baseline,
    _evaluate_active_improvement,
    _evaluate_no_op_narrowness,
    _status_label,
)


def _window_report_row(**overrides: object) -> dict[str, object]:
    row = {
        "window_key": "20260324:20260406:artifact_free_heuristic:full_rebound_diagnostics_plus_opposite_under:summary.json",
        "window_start_run_date": "20260324",
        "window_end_run_date": "20260406",
        "validation_mode": "artifact_free_heuristic",
        "variant": "full_rebound_diagnostics_plus_opposite_under",
        "validation_window_type": NO_OP_WINDOW,
        "status_label": "no_op_narrowness_pass",
        "no_op_narrowness_passed": True,
        "active_improvement_passed": False,
        "active_rebound_risk_present": False,
        "removed_trb_over_wins": 0,
        "removed_trb_over_losses": 0,
        "kept_trb_over_wins": 0,
        "kept_trb_over_losses": 0,
        "win_preservation_rate": np.nan,
        "loss_removal_rate": np.nan,
        "active_board_change_count": 0,
        "active_non_rebound_board_change_count": 0,
        "active_coverage_retained": 1.0,
        "roi_delta": 0.0,
        "brier_delta": 0.0,
        "ece_delta": 0.0,
        "hit_rate_delta": 0.0,
        "profit_units_delta": 0.0,
        "active_non_rebound_hit_rate_delta": 0.0,
        "no_op_board_change_count": 0,
        "no_op_non_rebound_board_change_count": 0,
        "no_op_non_rebound_hit_rate_delta": 0.0,
        "no_op_coverage_retained": 1.0,
        "no_op_final_board_trb_over_count": 0,
        "no_op_diagnostics_trigger_count": 0,
        "no_op_overtrigger_warning": False,
        "opposite_under_enabled": True,
        "opposite_under_flagged_over_count": 0,
        "synthetic_under_candidates_created": 0,
        "under_candidates_with_valid_price": 0,
        "under_candidates_passing_break_even": 0,
        "under_candidates_added_to_board": 0,
        "under_candidates_rejected_price": 0,
        "under_candidates_rejected_forecastability": 0,
        "under_candidates_rejected_stress": 0,
        "under_candidate_resolved_picks": 0,
        "under_candidate_wins": 0,
        "under_candidate_losses": 0,
        "under_candidate_pushes": 0,
        "under_candidate_profit_units": 0.0,
        "under_candidate_roi": np.nan,
        "added_under_rows": [],
    }
    row.update(overrides)
    return row


def test_no_op_window_passes_when_board_is_unchanged_and_coverage_holds() -> None:
    baseline = pd.DataFrame(
        [
            {"run_date": "2026-04-01", "player": "Safe Under", "target": "TRB", "direction": "UNDER", "market_line": 7.5, "market_id": "TRB_UNDER", "result": "win", "units": 0.9},
            {"run_date": "2026-04-01", "player": "Points", "target": "PTS", "direction": "OVER", "market_line": 19.5, "market_id": "PTS_OVER", "result": "win", "units": 0.9},
        ]
    )
    variant = baseline.copy()
    selector = pd.DataFrame(
        [
            {
                "run_date": "2026-04-01",
                "player": "Stable Rebound",
                "target": "TRB",
                "direction": "OVER",
                "market_id": "TRB_OVER",
                "market_line": 8.5,
                "total_rebound_penalty": 0.0,
                "trb_over_bucket": "TRB_OVER_STABLE",
            }
        ]
    )

    result = _evaluate_no_op_narrowness(
        variant,
        baseline,
        selector,
        no_op_dates={"2026-04-01"},
        coverage_threshold=0.95,
        board_change_tolerance=0,
    )

    assert result["passed"] is True
    assert result["board_change_count"] == 0
    assert result["non_rebound_board_change_count"] == 0
    assert result["coverage_retained"] == 1.0


def test_no_op_window_does_not_count_as_improvement_proof() -> None:
    window_reports = pd.DataFrame([_window_report_row()])

    gate = build_promotion_gate(window_reports, broader_window_min_count=4)

    assert gate["shadow_validated_logic"] is False
    assert gate["promotion_ready"] is False
    assert "active_rebound_improvement_window_required" in gate["blocked_reason"]


def test_active_window_passes_when_losses_removed_without_non_rebound_changes() -> None:
    baseline = pd.DataFrame(
        [
            {"run_date": "2026-04-01", "player": "Loss Removed", "target": "TRB", "direction": "OVER", "market_line": 11.5, "market_id": "TRB_OVER", "result": "loss", "units": -1.0, "board_play_win_prob": 0.40, "expected_win_rate": 0.40},
            {"run_date": "2026-04-01", "player": "Win Kept", "target": "TRB", "direction": "OVER", "market_line": 10.5, "market_id": "TRB_OVER", "result": "win", "units": 0.9, "board_play_win_prob": 0.70, "expected_win_rate": 0.70},
            {"run_date": "2026-04-01", "player": "Non Rebound", "target": "PTS", "direction": "OVER", "market_line": 19.5, "market_id": "PTS_OVER", "result": "win", "units": 0.9, "board_play_win_prob": 0.70, "expected_win_rate": 0.70},
        ]
    )
    variant = pd.DataFrame(
        [
            {"run_date": "2026-04-01", "player": "Win Kept", "target": "TRB", "direction": "OVER", "market_line": 10.5, "market_id": "TRB_OVER", "result": "win", "units": 0.9, "board_play_win_prob": 0.70, "expected_win_rate": 0.70},
            {"run_date": "2026-04-01", "player": "Non Rebound", "target": "PTS", "direction": "OVER", "market_line": 19.5, "market_id": "PTS_OVER", "result": "win", "units": 0.9, "board_play_win_prob": 0.70, "expected_win_rate": 0.70},
            {"run_date": "2026-04-01", "player": "Added Under", "target": "TRB", "direction": "UNDER", "market_line": 11.5, "market_id": "TRB_UNDER", "result": "win", "units": 0.9, "board_play_win_prob": 0.70, "expected_win_rate": 0.70},
        ]
    )

    result = _evaluate_active_improvement(
        variant,
        baseline,
        active_dates={"2026-04-01"},
        coverage_threshold=0.95,
        win_preservation_floor=0.67,
    )

    assert result["passed"] is True
    assert result["removed_trb_over_losses"] == 1
    assert result["removed_trb_over_wins"] == 0
    assert result["non_rebound_board_change_count"] == 0
    assert result["roi_delta"] >= 0.0
    assert result["brier_delta"] <= 0.0
    assert result["ece_delta"] <= 0.0


def test_promotion_blocked_when_validation_is_only_artifact_free_heuristic() -> None:
    window_reports = pd.DataFrame(
        [
            _window_report_row(),
            _window_report_row(
                window_key="20260418:20260426:artifact_free_heuristic:full_rebound_diagnostics_plus_opposite_under:summary.json",
                window_start_run_date="20260418",
                window_end_run_date="20260426",
                validation_window_type=ACTIVE_WINDOW,
                status_label="logic_improvement_pass",
                no_op_narrowness_passed=False,
                active_improvement_passed=True,
                active_rebound_risk_present=True,
                removed_trb_over_losses=2,
                roi_delta=0.15,
                brier_delta=-0.05,
                ece_delta=-0.03,
            ),
        ]
    )

    gate = build_promotion_gate(window_reports, broader_window_min_count=4)

    assert gate["shadow_validated_logic"] is True
    assert gate["trained_bundle_validated"] is False
    assert gate["promotion_ready"] is False
    assert gate["promotion_status_label"] == "trained_bundle_required"
    assert "trained_bundle_replay_required" in gate["blocked_reason"]


def test_promotion_blocked_when_trained_bundle_is_missing() -> None:
    window_reports = pd.DataFrame(
        [
            _window_report_row(),
            _window_report_row(
                window_key="20260418:20260426:artifact_free_heuristic:full_rebound_diagnostics_plus_opposite_under:summary.json",
                window_start_run_date="20260418",
                window_end_run_date="20260426",
                validation_window_type=ACTIVE_WINDOW,
                status_label="logic_improvement_pass",
                no_op_narrowness_passed=False,
                active_improvement_passed=True,
                active_rebound_risk_present=True,
                removed_trb_over_losses=2,
                roi_delta=0.15,
                brier_delta=-0.05,
                ece_delta=-0.03,
            ),
        ]
    )

    gate = build_promotion_gate(window_reports, broader_window_min_count=2)

    assert gate["trained_bundle_validated"] is False
    assert "trained_bundle_replay_required" in gate["blocked_reason"]


def test_opposite_under_audit_counts_flagged_and_added_candidates() -> None:
    selector = pd.DataFrame(
        [
            {
                "variant": "full_rebound_diagnostics_plus_opposite_under",
                "run_date": "2026-04-01",
                "player": "Flagged Rejected",
                "target": "TRB",
                "direction": "OVER",
                "market_id": "TRB_OVER",
                "total_rebound_penalty": 0.18,
                "opposite_side_odds": np.nan,
                "opposite_side_break_even": np.nan,
                "opposite_side_stress_prob": np.nan,
                "opposite_side_decision": "reject_price_unavailable",
                "rebound_diagnostic_segment": "TRB_OVER_LOW_LINE_ROLE_VOLATILE",
            },
            {
                "variant": "full_rebound_diagnostics_plus_opposite_under",
                "run_date": "2026-04-01",
                "player": "Flagged Promoted",
                "target": "TRB",
                "direction": "OVER",
                "market_id": "TRB_OVER",
                "total_rebound_penalty": 0.22,
                "opposite_side_odds": 120.0,
                "opposite_side_break_even": 0.4545,
                "opposite_side_stress_prob": 0.62,
                "opposite_side_decision": "promote_under_candidate",
                "rebound_diagnostic_segment": "TRB_OVER_SHARE_COMPETITION",
            },
            {
                "variant": "full_rebound_diagnostics_plus_opposite_under",
                "run_date": "2026-04-01",
                "player": "Flagged Promoted",
                "target": "TRB",
                "direction": "UNDER",
                "market_id": "TRB_UNDER",
                "market_side_price": 120.0,
                "rebound_diagnostic_segment": "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY",
                "trb_over_bucket_reasons": "under_prob=0.620;break_even=0.455;penalty=0.220",
            },
        ]
    )
    board = pd.DataFrame(
        [
            {
                "run_date": "2026-04-01",
                "player": "Flagged Promoted",
                "target": "TRB",
                "direction": "UNDER",
                "market_id": "TRB_UNDER",
                "market_line": 7.5,
                "market_side_price": 120.0,
                "rebound_diagnostic_segment": "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY",
                "result": "win",
                "units": 1.2,
            }
        ]
    )

    audit = _build_opposite_under_audit(selector, board)

    assert audit["flagged_over_count"] == 2
    assert audit["synthetic_under_candidates_created"] == 1
    assert audit["under_candidates_with_valid_price"] == 1
    assert audit["under_candidates_passing_break_even"] == 1
    assert audit["under_candidates_added_to_board"] == 1
    assert audit["under_candidates_rejected_price"] == 1
    assert audit["under_candidate_results"]["wins"] == 1


def test_non_rebound_board_changes_are_reported_separately() -> None:
    baseline = pd.DataFrame(
        [
            {"run_date": "2026-04-01", "player": "TRB Loss", "target": "TRB", "direction": "OVER", "market_line": 9.5, "market_id": "TRB_OVER", "result": "loss"},
            {"run_date": "2026-04-01", "player": "PTS Old", "target": "PTS", "direction": "OVER", "market_line": 19.5, "market_id": "PTS_OVER", "result": "win"},
        ]
    )
    variant = pd.DataFrame(
        [
            {"run_date": "2026-04-01", "player": "PTS New", "target": "PTS", "direction": "OVER", "market_line": 20.5, "market_id": "PTS_OVER", "result": "win"},
        ]
    )

    metrics = _compare_to_baseline(variant, baseline)

    assert metrics["board_change_count"] == 3
    assert metrics["non_rebound_board_change_count"] == 2


def test_status_labels_are_assigned_deterministically() -> None:
    no_op_validation = {"passed": True}
    active_validation = {"passed": True}

    assert _status_label(
        variant="baseline_no_rebound_diagnostics",
        validation_window_type=NO_OP_WINDOW,
        no_op_validation=no_op_validation,
        active_validation=active_validation,
        resolved_picks=20,
        min_resolved_picks=8,
    ) == "baseline"
    assert _status_label(
        variant="full_rebound_diagnostics_plus_opposite_under",
        validation_window_type=NO_OP_WINDOW,
        no_op_validation=no_op_validation,
        active_validation={"passed": False},
        resolved_picks=20,
        min_resolved_picks=8,
    ) == "no_op_narrowness_pass"
    assert _status_label(
        variant="full_rebound_diagnostics_plus_opposite_under",
        validation_window_type=ACTIVE_WINDOW,
        no_op_validation={"passed": False},
        active_validation=active_validation,
        resolved_picks=20,
        min_resolved_picks=8,
    ) == "logic_improvement_pass"
    assert _status_label(
        variant="full_rebound_diagnostics_plus_opposite_under",
        validation_window_type=MIXED_WINDOW,
        no_op_validation={"passed": True},
        active_validation={"passed": False},
        resolved_picks=20,
        min_resolved_picks=8,
    ) == "rejected_overfit"
    assert _status_label(
        variant="full_rebound_diagnostics_plus_opposite_under",
        validation_window_type=ACTIVE_WINDOW,
        no_op_validation={"passed": False},
        active_validation=active_validation,
        resolved_picks=4,
        min_resolved_picks=8,
    ) == "needs_more_sample"
