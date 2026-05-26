from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.failure_modes.attribute_pick_failures import attribute_pick_failures
from research.failure_modes.discover_unknown_failures import discover_unknown_failures
from research.failure_modes.failure_mode_scoreboard import build_failure_mode_scoreboard
from research.improvement_ledger.ledger import load_improvement_ledger
from research.interventions.propose_interventions import propose_interventions
from research.run_improvement_discovery import DEFAULT_TARGET_FAILURE_MODES, expand_failure_mode_exclusions, run_improvement_discovery


def _selected_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Scottie_Barnes",
                "market_player_raw": "Scottie_Barnes",
                "run_date": "2026-04-18",
                "actual_matched_date": "2026-04-18",
                "target": "AST",
                "direction": "OVER",
                "market_line": 6.5,
                "market_id": "AST_OVER",
                "result": "loss",
                "actual": 4.0,
                "prediction": 7.2,
                "expected_win_rate": 0.66,
                "board_play_win_prob": 0.67,
                "units": -1.0,
            },
            {
                "player": "Austin_Reaves",
                "market_player_raw": "Austin_Reaves",
                "run_date": "2026-04-19",
                "actual_matched_date": "2026-04-19",
                "target": "AST",
                "direction": "OVER",
                "market_line": 7.5,
                "market_id": "AST_OVER",
                "result": "loss",
                "actual": 5.0,
                "prediction": 8.0,
                "expected_win_rate": 0.67,
                "board_play_win_prob": 0.68,
                "units": -1.0,
            },
            {
                "player": "Kevin_Durant",
                "market_player_raw": "Kevin_Durant",
                "run_date": "2026-04-20",
                "actual_matched_date": "2026-04-20",
                "target": "AST",
                "direction": "OVER",
                "market_line": 5.5,
                "market_id": "AST_OVER",
                "result": "loss",
                "actual": 3.0,
                "prediction": 6.1,
                "expected_win_rate": 0.65,
                "board_play_win_prob": 0.66,
                "units": -1.0,
            },
            {
                "player": "Jaylen_Brown",
                "market_player_raw": "Jaylen_Brown",
                "run_date": "2026-04-21",
                "actual_matched_date": "2026-04-21",
                "target": "AST",
                "direction": "OVER",
                "market_line": 4.5,
                "market_id": "AST_OVER",
                "result": "win",
                "actual": 6.0,
                "prediction": 5.5,
                "expected_win_rate": 0.62,
                "board_play_win_prob": 0.63,
                "units": 0.91,
            },
            {
                "player": "Andre_Drummond",
                "market_player_raw": "Andre_Drummond",
                "run_date": "2026-04-22",
                "actual_matched_date": "2026-04-22",
                "target": "TRB",
                "direction": "OVER",
                "market_line": 4.5,
                "market_id": "TRB_OVER",
                "result": "loss",
                "actual": 2.0,
                "prediction": 5.2,
                "expected_win_rate": 0.61,
                "board_play_win_prob": 0.62,
                "units": -1.0,
            },
        ]
    )


def _candidate_pool_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Scottie_Barnes",
                "market_player_raw": "Scottie_Barnes",
                "market_date": "2026-04-18",
                "target": "AST",
                "direction": "OVER",
                "market_line": 6.5,
                "market_id": "AST_OVER",
                "market_side_price": -110,
                "market_side_break_even": 0.524,
                "prediction": 7.2,
                "expected_win_rate": 0.66,
                "stress_probability": 0.64,
                "predicted_probability": 0.66,
                "projected_team_fg_pct": 0.456,
                "line_decision_fragility_score": 0.62,
                "line_decision_instability_score": 0.54,
                "belief_uncertainty": 0.88,
                "posterior_variance": 0.09,
                "calibration_bucket_rows": 18,
            },
            {
                "player": "Austin_Reaves",
                "market_player_raw": "Austin_Reaves",
                "market_date": "2026-04-19",
                "target": "AST",
                "direction": "OVER",
                "market_line": 7.5,
                "market_id": "AST_OVER",
                "market_side_price": -112,
                "market_side_break_even": 0.528,
                "prediction": 8.0,
                "expected_win_rate": 0.67,
                "stress_probability": 0.65,
                "predicted_probability": 0.67,
                "projected_team_fg_pct": 0.459,
                "line_decision_fragility_score": 0.60,
                "line_decision_instability_score": 0.50,
                "belief_uncertainty": 0.86,
                "posterior_variance": 0.08,
                "calibration_bucket_rows": 20,
            },
            {
                "player": "Kevin_Durant",
                "market_player_raw": "Kevin_Durant",
                "market_date": "2026-04-20",
                "target": "AST",
                "direction": "OVER",
                "market_line": 5.5,
                "market_id": "AST_OVER",
                "market_side_price": -108,
                "market_side_break_even": 0.519,
                "prediction": 6.1,
                "expected_win_rate": 0.65,
                "stress_probability": 0.63,
                "predicted_probability": 0.65,
                "projected_team_fg_pct": 0.460,
                "line_decision_fragility_score": 0.58,
                "line_decision_instability_score": 0.48,
                "belief_uncertainty": 0.84,
                "posterior_variance": 0.08,
                "calibration_bucket_rows": 22,
            },
            {
                "player": "Jaylen_Brown",
                "market_player_raw": "Jaylen_Brown",
                "market_date": "2026-04-21",
                "target": "AST",
                "direction": "OVER",
                "market_line": 4.5,
                "market_id": "AST_OVER",
                "market_side_price": -109,
                "market_side_break_even": 0.521,
                "prediction": 5.5,
                "expected_win_rate": 0.62,
                "stress_probability": 0.60,
                "predicted_probability": 0.62,
                "projected_team_fg_pct": 0.478,
                "line_decision_fragility_score": 0.43,
                "line_decision_instability_score": 0.36,
                "belief_uncertainty": 0.73,
                "posterior_variance": 0.04,
                "calibration_bucket_rows": 64,
            },
            {
                "player": "Andre_Drummond",
                "market_player_raw": "Andre_Drummond",
                "market_date": "2026-04-22",
                "target": "TRB",
                "direction": "OVER",
                "market_line": 4.5,
                "market_id": "TRB_OVER",
                "market_side_price": -115,
                "market_side_break_even": 0.535,
                "prediction": 5.2,
                "expected_win_rate": 0.61,
                "stress_probability": 0.60,
                "predicted_probability": 0.61,
                "upper_band_line_penalty": 0.0,
                "low_line_role_volatility_penalty": 0.10,
                "low_line_role_volatility_flag": True,
                "projected_team_fg_pct": 0.481,
                "line_decision_fragility_score": 0.40,
                "line_decision_instability_score": 0.35,
                "belief_uncertainty": 0.70,
                "posterior_variance": 0.04,
                "calibration_bucket_rows": 60,
                "expected_minutes_band_width": 10.0,
                "minutes_floor_recent": 12.0,
                "bench_role_flag": True,
                "rotation_volatility_score": 0.72,
                "blowout_minutes_sensitivity": 0.40,
                "foul_rate_minutes_loss_risk": 0.30,
            },
        ]
    )


def test_rebound_failures_can_be_excluded() -> None:
    excluded_modes, excluded_markets = expand_failure_mode_exclusions(["rebound"])
    assert "REBOUND_LOW_LINE_ROLE_VOLATILITY" in excluded_modes
    assert "TRB" in excluded_markets


def test_target_failure_families_are_respected_and_rebound_is_not_top_result() -> None:
    selected = _selected_rows()
    pool = _candidate_pool_rows()
    excluded_modes, _ = expand_failure_mode_exclusions(["rebound"])
    attributed = attribute_pick_failures(
        selected,
        pool,
        allowed_failure_modes=set(DEFAULT_TARGET_FAILURE_MODES),
        excluded_failure_modes=excluded_modes,
    )
    scoreboard = build_failure_mode_scoreboard(
        attributed,
        candidate_pool_rows=pool,
        target_failure_modes=DEFAULT_TARGET_FAILURE_MODES,
        excluded_failure_modes=excluded_modes,
    )
    assert not scoreboard.empty
    assert not scoreboard["failure_mode_id"].astype(str).str.startswith("REBOUND_").any()
    assert set(scoreboard["failure_mode_id"].astype(str)).issubset(set(DEFAULT_TARGET_FAILURE_MODES))
    assert scoreboard.iloc[0]["failure_mode_id"] in {"LOW_TEAM_ASSIST_ENVIRONMENT", "TEAM_OFFENSE_COLLAPSE", "CALIBRATION_OVERCONFIDENCE"}


def test_unknown_failure_discovery_does_not_register_one_off_loss() -> None:
    attributed = pd.DataFrame(
        [
            {
                "player": "One_Off",
                "market_type": "PTS_OVER",
                "target": "PTS",
                "direction": "OVER",
                "market_line": 18.5,
                "predicted_probability": 0.67,
                "stress_probability": 0.65,
                "market_side_break_even": 0.52,
                "belief_uncertainty": 0.83,
                "line_decision_fragility_score": 0.58,
                "projected_team_fg_pct": 0.470,
                "result": "loss",
                "failure_modes": [],
                "recoverability_class": "ALEATORIC_OR_RANDOM",
                "game_date": "2026-04-18",
                "team": "AAA",
            },
            {
                "player": "Nearby_Win",
                "market_type": "PTS_OVER",
                "target": "PTS",
                "direction": "OVER",
                "market_line": 18.5,
                "predicted_probability": 0.66,
                "stress_probability": 0.64,
                "market_side_break_even": 0.52,
                "belief_uncertainty": 0.82,
                "line_decision_fragility_score": 0.57,
                "projected_team_fg_pct": 0.471,
                "result": "win",
                "failure_modes": [],
                "recoverability_class": "",
                "game_date": "2026-04-18",
                "team": "BBB",
            },
        ]
    )
    clusters, _, _ = discover_unknown_failures(attributed, min_cluster_losses=3, target_market_families=["PTS"])
    assert not clusters.empty
    assert set(clusters["recommendation"].astype(str)).issubset({"NEEDS_MORE_SAMPLE", "REJECT_RANDOM"})


def test_scoreboard_prioritizes_recurring_detectable_non_rebound_losses() -> None:
    attributed = pd.DataFrame(
        [
            {"failure_modes": ["LOW_TEAM_ASSIST_ENVIRONMENT"], "pre_event_failure_modes": ["LOW_TEAM_ASSIST_ENVIRONMENT"], "result": "loss", "predicted_probability": 0.66, "was_pre_event_detectable": True, "miss_distance": -2.0, "units": -1.0},
            {"failure_modes": ["LOW_TEAM_ASSIST_ENVIRONMENT"], "pre_event_failure_modes": ["LOW_TEAM_ASSIST_ENVIRONMENT"], "result": "loss", "predicted_probability": 0.65, "was_pre_event_detectable": True, "miss_distance": -1.6, "units": -1.0},
            {"failure_modes": ["LOW_TEAM_ASSIST_ENVIRONMENT"], "pre_event_failure_modes": ["LOW_TEAM_ASSIST_ENVIRONMENT"], "result": "win", "predicted_probability": 0.62, "was_pre_event_detectable": True, "miss_distance": 1.2, "units": 0.91},
            {"failure_modes": ["MARKET_PRICE_MISPLACEMENT"], "pre_event_failure_modes": ["MARKET_PRICE_MISPLACEMENT"], "result": "loss", "predicted_probability": 0.56, "was_pre_event_detectable": True, "miss_distance": -0.4, "units": -1.0},
        ]
    )
    scoreboard = build_failure_mode_scoreboard(attributed, target_failure_modes=["LOW_TEAM_ASSIST_ENVIRONMENT", "MARKET_PRICE_MISPLACEMENT"])
    assert scoreboard.iloc[0]["failure_mode_id"] == "LOW_TEAM_ASSIST_ENVIRONMENT"
    assert float(scoreboard.iloc[0]["priority_score"]) > float(scoreboard.iloc[1]["priority_score"])


def test_intervention_candidates_are_shadow_only_for_non_rebound_modes() -> None:
    scoreboard = pd.DataFrame(
        [
            {
                "failure_mode_id": "LOW_TEAM_ASSIST_ENVIRONMENT",
                "candidate_count": 4,
                "resolved_count": 4,
                "losses": 3,
                "wins": 1,
                "priority_score": 0.05,
                "pre_event_detectability_rate": 0.85,
                "estimated_coverage_cost": 0.10,
                "non_target_damage_risk": 0.02,
                "intervention_available": True,
                "expected_improvement_if_gated": 0.35,
                "estimated_loss_removal_rate": 0.35,
                "estimated_win_removal_rate": 0.08,
            }
        ]
    )
    proposals = propose_interventions(scoreboard)
    assert not proposals.empty
    assert proposals["shadow_only"].astype(bool).all()
    assert proposals["recommended_next_action"].eq("VALIDATE_SHADOW").any()
    assert not proposals.astype(str).apply(lambda column: column.str.contains("promotion_candidate|live_ready", regex=True)).any().any()


def test_discovery_run_writes_required_outputs_and_ledger(tmp_path: Path) -> None:
    selected = _selected_rows()
    pool = _candidate_pool_rows()
    outputs_dir = tmp_path / "discovery"
    ledger_path = tmp_path / "ledger.jsonl"
    excluded_modes, excluded_markets = expand_failure_mode_exclusions(["rebound"])
    report = run_improvement_discovery(
        selected_rows=selected,
        candidate_pool_rows=pool,
        outputs_dir=outputs_dir,
        ledger_path=ledger_path,
        input_manifest={"selected_board_csvs": ["synthetic_selected.csv"], "candidate_pool_csvs": ["synthetic_pool.csv"]},
        data_proc_root=tmp_path / "missing_data_proc",
        target_failure_modes=DEFAULT_TARGET_FAILURE_MODES,
        excluded_failure_modes=excluded_modes,
        excluded_market_families=excluded_markets,
        min_loss_count=3,
        min_pre_event_detectability=0.60,
        priority_floor=0.005,
        min_cluster_losses=3,
        discovery_only=True,
        shadow_only=True,
    )
    assert report["mode"] == "discovery_only"
    assert report["shadow_only"] is True
    assert report["status_label"] in {"discovery_only", "needs_more_sample", "feature_gap_blocked", "rejected_random", "validate_shadow_next"}
    assert "promotion_ready" not in report
    required_files = [
        "failure_mode_scoreboard.csv",
        "unknown_failure_clusters.csv",
        "intervention_candidates.csv",
        "improvement_discovery_report.md",
        "improvement_discovery_report.json",
    ]
    for name in required_files:
        assert (outputs_dir / name).exists(), name
    report_text = (outputs_dir / "improvement_discovery_report.json").read_text(encoding="utf-8")
    assert "promotion_candidate" not in report_text
    assert "live_ready" not in report_text
    ledger = load_improvement_ledger(ledger_path)
    assert len(ledger) == 1
    assert ledger.iloc[0]["mode"] == "discovery_only"
    assert ledger.iloc[0]["promotion_status"] == "not_applicable"

