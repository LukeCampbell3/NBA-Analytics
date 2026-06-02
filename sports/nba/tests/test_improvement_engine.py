from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from post_process_market_plays import compute_final_board
from research.failure_modes.attribute_pick_failures import attribute_pick_failures
from research.failure_modes.failure_mode_registry import load_failure_mode_registry_payload, load_failure_mode_schema, validate_failure_mode_registry
from research.failure_modes.failure_mode_scoreboard import build_failure_mode_scoreboard
from research.improvement_ledger.ledger import append_improvement_entry, load_improvement_ledger
from research.interventions.failure_mode_adjustments import apply_failure_mode_adjustments
from research.interventions.propose_interventions import propose_interventions
from research.validation.report_intervention_promotion_gate import build_intervention_promotion_gate
from research.validation.validate_intervention import build_intervention_validation_payload


def _window_row(**overrides: object) -> dict[str, object]:
    row = {
        "window_key": "2026-04-01:2026-04-03:artifact_free_heuristic:target:summary.json",
        "validation_mode": "artifact_free_heuristic",
        "variant": "target",
        "validation_window_type": "NO_OP_NARROWNESS_WINDOW",
        "status_label": "no_op_narrowness_pass",
        "no_op_narrowness_passed": True,
        "active_improvement_passed": False,
        "active_rebound_risk_present": False,
        "no_op_board_change_count": 0,
        "no_op_non_target_board_change_count": 0,
        "no_op_coverage_retained": 1.0,
        "no_op_non_target_hit_rate_delta": 0.0,
        "active_board_change_count": 0,
        "active_non_target_board_change_count": 0,
        "active_coverage_retained": 1.0,
        "removed_wins": 0,
        "removed_losses": 0,
        "kept_wins": 0,
        "kept_losses": 0,
        "win_preservation_rate": 1.0,
        "loss_removal_rate": 0.0,
        "roi_delta": 0.0,
        "brier_delta": 0.0,
        "ece_delta": 0.0,
        "calibration_gap_delta": 0.0,
        "active_non_target_hit_rate_delta": 0.0,
        "under_candidates_added_to_board": 0,
        "under_candidates_with_valid_price": 0,
        "under_candidates_passing_break_even": 0,
    }
    row.update(overrides)
    return row


def _base_pick_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Risky Rebound",
                "market_player_raw": "Risky Rebound",
                "market_date": "2026-04-01",
                "target": "TRB",
                "direction": "OVER",
                "market_line": 4.5,
                "market_id": "TRB_OVER",
                "result": "loss",
                "actual_TRB": 2.0,
                "actual_minutes": 14.0,
                "predicted_probability": 0.63,
                "stress_probability": 0.60,
                "expected_win_rate": 0.60,
                "market_side_break_even": 0.52,
                "low_line_role_volatility_penalty": 0.11,
                "low_line_role_volatility_flag": True,
                "expected_minutes_band_low": 24.0,
                "expected_minutes_band_high": 32.0,
                "minutes_floor_recent": 12.0,
                "bench_role_flag": True,
                "rotation_volatility_score": 0.72,
                "units": -1.0,
            }
        ]
    )


def _board_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Penalized",
                "market_player_raw": "Penalized",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 24.0,
                "market_line": 21.5,
                "abs_edge": 2.6,
                "edge": 2.6,
                "expected_win_rate": 0.69,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.82,
                "feasibility": 0.90,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_a",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "gap_percentile": 0.96,
            },
            {
                "player": "Safe",
                "market_player_raw": "Safe",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 23.0,
                "market_line": 21.5,
                "abs_edge": 1.9,
                "edge": 1.9,
                "expected_win_rate": 0.64,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.74,
                "feasibility": 0.89,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_b",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "gap_percentile": 0.93,
            },
        ]
    )


def test_failure_mode_registry_schema_is_valid() -> None:
    payload = load_failure_mode_registry_payload()
    schema = load_failure_mode_schema()
    validate_failure_mode_registry(payload, schema)
    assert any(row["failure_mode_id"] == "REBOUND_SUPPLY_COLLAPSE" for row in payload["failure_modes"])


def test_postgame_failure_attribution_assigns_deterministic_labels() -> None:
    attributed = attribute_pick_failures(_base_pick_rows())
    assert len(attributed) == 1
    row = attributed.iloc[0]
    assert row["primary_failure_mode"] == "REBOUND_LOW_LINE_ROLE_VOLATILITY"
    assert "MINUTES_BAND_FAILURE" in row["failure_modes"]
    assert row["recoverability_class"] == "RECOVERABLE_PRE_EVENT"
    assert row["recommended_intervention_type"] in {"soft_downgrade", "hard_gate"}


def test_random_misses_are_not_converted_into_interventions() -> None:
    rows = pd.DataFrame(
        [
            {
                "player": "Random Miss",
                "market_player_raw": "Random Miss",
                "market_date": "2026-04-01",
                "target": "PTS",
                "direction": "OVER",
                "market_line": 21.5,
                "market_id": "PTS_OVER",
                "result": "loss",
                "actual_PTS": 20.0,
                "predicted_probability": 0.54,
                "stress_probability": 0.53,
                "expected_win_rate": 0.53,
                "market_side_break_even": 0.52,
                "units": -1.0,
            }
        ]
    )
    attributed = attribute_pick_failures(rows)
    scoreboard = build_failure_mode_scoreboard(attributed)
    proposals = propose_interventions(scoreboard)
    assert scoreboard.empty
    assert proposals.empty


def test_failure_mode_scoreboard_prioritizes_recurring_detectable_losses() -> None:
    attributed = pd.DataFrame(
        [
            {"failure_modes": ["REBOUND_SUPPLY_COLLAPSE"], "result": "loss", "predicted_probability": 0.62, "was_failure_pre_event_detectable": True, "miss_distance": -2.0, "units": -1.0},
            {"failure_modes": ["REBOUND_SUPPLY_COLLAPSE"], "result": "loss", "predicted_probability": 0.61, "was_failure_pre_event_detectable": True, "miss_distance": -1.5, "units": -1.0},
            {"failure_modes": ["REBOUND_SUPPLY_COLLAPSE"], "result": "win", "predicted_probability": 0.60, "was_failure_pre_event_detectable": True, "miss_distance": 1.0, "units": 0.91},
            {"failure_modes": ["MARKET_PRICE_MISPLACEMENT"], "result": "loss", "predicted_probability": 0.56, "was_failure_pre_event_detectable": True, "miss_distance": -0.5, "units": -1.0},
        ]
    )
    scoreboard = build_failure_mode_scoreboard(attributed)
    assert scoreboard.iloc[0]["failure_mode_id"] == "REBOUND_SUPPLY_COLLAPSE"
    assert float(scoreboard.iloc[0]["priority_score"]) > 0.0


def test_intervention_proposals_remain_shadow_only() -> None:
    scoreboard = pd.DataFrame(
        [
            {
                "failure_mode_id": "REBOUND_SUPPLY_COLLAPSE",
                "resolved_count": 8,
                "priority_score": 0.05,
                "pre_event_detectability_rate": 0.85,
                "coverage_loss_if_gated": 0.08,
                "losses": 5,
                "wins": 1,
                "intervention_available": True,
                "expected_improvement_if_gated": 0.30,
                "estimated_loss_removal_rate": 0.30,
                "estimated_win_removal_rate": 0.05,
            }
        ]
    )
    proposals = propose_interventions(scoreboard)
    assert not proposals.empty
    assert proposals["shadow_only"].astype(bool).all()
    assert proposals["failure_mode_id"].eq("REBOUND_SUPPLY_COLLAPSE").all()


def test_no_op_windows_do_not_count_as_improvement() -> None:
    reports = pd.DataFrame([_window_row()])
    gate = build_intervention_promotion_gate(reports, target_variant="target", broader_window_min_count=4)
    assert gate["promotion_ready"] is False
    assert "active_risk_improvement_window_required" in gate["blocked_reasons"]


def test_artifact_free_heuristic_blocks_promotion() -> None:
    reports = pd.DataFrame(
        [
            _window_row(),
            _window_row(
                window_key="2026-04-10:2026-04-12:artifact_free_heuristic:target:summary.json",
                validation_window_type="ACTIVE_REBOUND_RISK_WINDOW",
                status_label="logic_improvement_pass",
                no_op_narrowness_passed=False,
                active_improvement_passed=True,
                active_rebound_risk_present=True,
                removed_losses=2,
                removed_wins=0,
                roi_delta=0.10,
                brier_delta=-0.02,
                ece_delta=-0.01,
            ),
        ]
    )
    gate = build_intervention_promotion_gate(reports, target_variant="target", broader_window_min_count=2)
    assert gate["shadow_validated_logic"] is True
    assert gate["trained_bundle_validated"] is False
    assert gate["promotion_status_label"] == "trained_bundle_required"


def test_non_target_market_damage_blocks_promotion() -> None:
    reports = pd.DataFrame(
        [
            _window_row(window_key="a", validation_mode="artifact_free_heuristic"),
            _window_row(window_key="b", validation_mode="trained_bundle"),
            _window_row(
                window_key="c",
                validation_mode="artifact_free_heuristic",
                validation_window_type="ACTIVE_REBOUND_RISK_WINDOW",
                status_label="logic_improvement_pass",
                no_op_narrowness_passed=False,
                active_improvement_passed=True,
                active_rebound_risk_present=True,
                removed_losses=2,
                removed_wins=0,
                roi_delta=0.10,
                brier_delta=-0.02,
                ece_delta=-0.01,
            ),
            _window_row(
                window_key="d",
                validation_mode="trained_bundle",
                validation_window_type="ACTIVE_REBOUND_RISK_WINDOW",
                status_label="logic_improvement_pass",
                no_op_narrowness_passed=False,
                active_improvement_passed=True,
                active_rebound_risk_present=True,
                removed_losses=2,
                removed_wins=0,
                roi_delta=0.10,
                brier_delta=-0.02,
                ece_delta=-0.01,
                active_non_target_board_change_count=1,
            ),
        ]
    )
    gate = build_intervention_promotion_gate(reports, target_variant="target", broader_window_min_count=4)
    assert gate["promotion_ready"] is False
    assert "unexpected_non_target_market_damage" in gate["blocked_reasons"]


def test_removed_wins_and_losses_are_tracked_separately_in_generic_validation_payload() -> None:
    source_payload = {
        "window": {"start_run_date": "20260401", "end_run_date": "20260403"},
        "summary": [],
        "segments": [],
        "window_reports": [
            {
                "validation_mode": "artifact_free_heuristic",
                "variant": "target",
                "validation_window_type": "ACTIVE_REBOUND_RISK_WINDOW",
                "status_label": "logic_improvement_pass",
                "active_improvement_validation": {
                    "passed": True,
                    "removed_trb_over_wins": 1,
                    "removed_trb_over_losses": 3,
                    "kept_trb_over_wins": 2,
                    "kept_trb_over_losses": 1,
                },
            }
        ],
    }
    payload = build_intervention_validation_payload(
        intervention_family="rebound_diagnostics",
        intervention_id="rebound",
        failure_mode_id="REBOUND_SUPPLY_COLLAPSE",
        summary_payloads=[(Path("summary.json"), source_payload)],
    )
    row = payload["window_reports"][0]
    assert row["removed_wins"] == 1
    assert row["removed_losses"] == 3
    assert row["kept_wins"] == 2
    assert row["kept_losses"] == 1


def test_board_objective_consumes_generic_adjustment_sidecar() -> None:
    plays = _board_rows()
    adjustments = pd.DataFrame(
        [
            {
                "candidate_id": apply_failure_mode_adjustments(plays.iloc[[0]].copy(), None).iloc[0]["candidate_id"],
                "failure_mode_id": "REBOUND_SUPPLY_COLLAPSE",
                "penalty": 0.25,
                "downgrade_tier": "consider",
                "veto_flag": True,
                "opposite_side_candidate_flag": False,
                "alt_line_candidate_flag": False,
                "explanation": "synthetic_test_veto",
            }
        ]
    )
    board = compute_final_board(
        plays,
        selection_mode="board_objective",
        ranking_mode="board_objective",
        min_recommendation="consider",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=1,
        max_plays_per_game=0,
        max_plays_per_script_cluster=0,
        max_plays_per_target=0,
        non_pts_min_gap_percentile=0.0,
        failure_mode_adjustments=adjustments,
    )
    assert len(board) == 1
    assert board.iloc[0]["player"] == "Safe"
    assert int(board.attrs["stage_counts"]["after_failure_mode_adjustments"]) == 1


def test_improvement_ledger_appends_reproducible_records(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    append_improvement_entry(
        {
            "improvement_id": "imp-1",
            "failure_mode_id": "REBOUND_SUPPLY_COLLAPSE",
            "intervention_id": "rebound_supply_penalty",
            "author_or_run_id": "pytest",
            "hypothesis": "Test ledger append.",
            "implementation_files": ["a.py"],
            "validation_windows": ["20260401-20260403"],
            "metrics_before": {"roi": 0.10},
            "metrics_after": {"roi": 0.12},
            "segment_results": {"segment": "TRB_OVER_SUPPLY_DEPENDENT"},
            "promotion_status": "shadow_only_candidate",
            "blocked_reasons": ["trained_bundle_replay_required"],
            "rollback_rule": "disable flag",
            "final_decision": "logged",
        },
        ledger_path=ledger_path,
    )
    loaded = load_improvement_ledger(ledger_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["improvement_id"] == "imp-1"
    assert loaded.iloc[0]["rollback_rule"] == "disable flag"


def test_rollback_plan_exists_for_promotion_candidate() -> None:
    reports = pd.DataFrame(
        [
            _window_row(window_key="a", validation_mode="artifact_free_heuristic"),
            _window_row(window_key="b", validation_mode="artifact_free_heuristic", window_start_run_date="20260405", window_end_run_date="20260407"),
            _window_row(window_key="c", validation_mode="trained_bundle"),
            _window_row(window_key="d", validation_mode="trained_bundle", window_start_run_date="20260405", window_end_run_date="20260407"),
            _window_row(
                window_key="e",
                validation_mode="artifact_free_heuristic",
                validation_window_type="ACTIVE_REBOUND_RISK_WINDOW",
                status_label="logic_improvement_pass",
                no_op_narrowness_passed=False,
                active_improvement_passed=True,
                active_rebound_risk_present=True,
                removed_losses=2,
                removed_wins=0,
                roi_delta=0.10,
                brier_delta=-0.02,
                ece_delta=-0.01,
            ),
            _window_row(
                window_key="f",
                validation_mode="trained_bundle",
                validation_window_type="ACTIVE_REBOUND_RISK_WINDOW",
                status_label="logic_improvement_pass",
                no_op_narrowness_passed=False,
                active_improvement_passed=True,
                active_rebound_risk_present=True,
                removed_losses=2,
                removed_wins=0,
                roi_delta=0.10,
                brier_delta=-0.02,
                ece_delta=-0.01,
            ),
        ]
    )
    gate = build_intervention_promotion_gate(reports, target_variant="target", broader_window_min_count=4)
    assert gate["rollback_plan"]
