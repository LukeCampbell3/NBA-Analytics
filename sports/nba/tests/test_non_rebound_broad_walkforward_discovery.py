from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.failure_modes.discover_unknown_failures import discover_unknown_failures
from research.failure_modes.failure_subregions import build_failure_subregion_scoreboard
from research.improvement_ledger.ledger import load_improvement_ledger
from research.run_improvement_discovery import run_improvement_discovery


def _base_row(index: int, **overrides: object) -> dict[str, object]:
    game_date = f"2026-01-{(index % 20) + 1:02d}"
    row: dict[str, object] = {
        "player": f"Player_{index}",
        "market_player_raw": f"Player_{index}",
        "run_date": game_date,
        "market_date": game_date,
        "actual_matched_date": game_date,
        "source_selected_board_csv": f"window_{index % 4}",
        "actual_team": f"T{index % 8}",
        "team": f"T{index % 8}",
        "opponent": f"O{index % 8}",
        "target": "AST",
        "direction": "OVER",
        "market_id": "AST_OVER",
        "market_line": 5.5,
        "prediction": 6.4,
        "expected_win_rate": 0.63,
        "predicted_probability": 0.63,
        "stress_probability": 0.61,
        "market_side_break_even": 0.52,
        "units": 0.91,
        "result": "win",
        "actual": 7.0,
        "actual_AST": 7.0,
        "actual_minutes": 33.0,
        "projected_team_fg_pct": 0.482,
        "line_decision_fragility_score": 0.40,
        "line_decision_instability_score": 0.32,
        "same_team_selected_over_count": 1,
        "belief_uncertainty": 0.78,
        "posterior_variance": 0.04,
        "calibration_bucket_rows": 80,
        "minutes_floor_recent": 28.0,
        "expected_minutes_band_low": 30.0,
        "expected_minutes_band_high": 34.0,
        "expected_minutes_band_width": 4.0,
        "bench_role_flag": False,
        "rotation_volatility_score": 0.20,
        "foul_rate_minutes_loss_risk": 0.20,
        "blowout_minutes_sensitivity": 0.20,
        "volatility_score": 0.30,
        "role_pathway_shift_score": 0.20,
        "prediction_shrink_lambda": 0.50,
        "feasibility": 0.90,
        "ev": 0.08,
        "market_books": 4,
    }
    row.update(overrides)
    return row


def _rows_with_minutes_subregion(losses: int, wins: int, total: int = 40, same_player: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(total):
        row = _base_row(idx)
        if idx < losses + wins:
            row.update(
                {
                    "target": "PTS",
                    "market_id": "PTS_OVER",
                    "market_line": 20.5,
                    "prediction": 21.1,
                    "minutes_floor_recent": 14.0,
                    "expected_minutes_band_low": 22.0,
                    "expected_minutes_band_width": 9.0,
                    "bench_role_flag": True,
                    "rotation_volatility_score": 0.70,
                    "foul_rate_minutes_loss_risk": 0.35,
                    "blowout_minutes_sensitivity": 0.35,
                    "actual": 18.0 if idx < losses else 24.0,
                    "actual_PTS": 18.0 if idx < losses else 24.0,
                    "result": "loss" if idx < losses else "win",
                    "units": -1.0 if idx < losses else 0.91,
                }
            )
            if same_player:
                row["player"] = "One_Player"
                row["market_player_raw"] = "One_Player"
                row["actual_team"] = "ONE"
                row["team"] = "ONE"
                row["run_date"] = "2026-01-01"
                row["market_date"] = "2026-01-01"
                row["actual_matched_date"] = "2026-01-01"
                row["source_selected_board_csv"] = "window_same"
        rows.append(row)
    return pd.DataFrame(rows)


def _rows_with_team_offense_subregion(losses: int, wins: int, total: int = 40) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(total):
        row = _base_row(idx)
        if idx < losses + wins:
            row.update(
                {
                    "projected_team_fg_pct": 0.455,
                    "line_decision_fragility_score": 0.60,
                    "line_decision_instability_score": 0.48,
                    "same_team_selected_over_count": 2,
                    "result": "loss" if idx < losses else "win",
                    "units": -1.0 if idx < losses else 0.91,
                    "actual": 4.0 if idx < losses else 7.0,
                    "actual_AST": 4.0 if idx < losses else 7.0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_broad_families_are_split_into_subregions() -> None:
    rows = _rows_with_team_offense_subregion(losses=4, wins=2, total=12)
    scoreboard = build_failure_subregion_scoreboard(rows, target_failure_modes=["TEAM_OFFENSE_COLLAPSE", "LOW_TEAM_ASSIST_ENVIRONMENT"])
    active = scoreboard.loc[scoreboard["selected_count"] > 0, "subregion_id"].astype(str).tolist()
    assert "TEAM_OFFENSE_COLLAPSE__same_team_over_stack" in active
    assert "TEAM_OFFENSE_COLLAPSE__poor_projected_team_fg_environment" in active
    assert "LOW_TEAM_ASSIST_ENVIRONMENT__AST_OVER_with_low_projected_fg_support" in active


def test_actionability_filter_rejects_high_coverage_cost_subregion() -> None:
    rows = _rows_with_team_offense_subregion(losses=15, wins=5, total=30)
    scoreboard = build_failure_subregion_scoreboard(rows, target_failure_modes=["TEAM_OFFENSE_COLLAPSE"])
    row = scoreboard.loc[scoreboard["subregion_id"] == "TEAM_OFFENSE_COLLAPSE__same_team_over_stack"].iloc[0]
    assert row["coverage_cost"] > 0.25
    assert row["recommended_next_action"] == "REJECT_RANDOM"


def test_actionability_filter_rejects_high_win_removal_subregion() -> None:
    rows = _rows_with_minutes_subregion(losses=4, wins=6, total=40)
    scoreboard = build_failure_subregion_scoreboard(rows, target_failure_modes=["MINUTES_BAND_FAILURE"])
    row = scoreboard.loc[scoreboard["subregion_id"] == "MINUTES_BAND_FAILURE__low_minutes_floor"].iloc[0]
    assert row["estimated_win_removal_rate"] >= row["estimated_loss_removal_rate"]
    assert row["recommended_next_action"] == "REJECT_RANDOM"


def test_subregion_can_become_validate_shadow_when_losses_concentrate_safely() -> None:
    rows = _rows_with_minutes_subregion(losses=8, wins=2, total=40)
    scoreboard = build_failure_subregion_scoreboard(rows, target_failure_modes=["MINUTES_BAND_FAILURE"])
    row = scoreboard.loc[scoreboard["subregion_id"] == "MINUTES_BAND_FAILURE__low_minutes_floor"].iloc[0]
    assert row["estimated_loss_removal_rate"] > row["estimated_win_removal_rate"]
    assert row["coverage_cost"] <= 0.25
    assert row["recommended_next_action"] == "VALIDATE_SHADOW"


def test_no_intervention_is_proposed_from_one_player_team_day_cluster() -> None:
    rows = _rows_with_minutes_subregion(losses=8, wins=2, total=40, same_player=True)
    scoreboard = build_failure_subregion_scoreboard(rows, target_failure_modes=["MINUTES_BAND_FAILURE"])
    row = scoreboard.loc[scoreboard["subregion_id"] == "MINUTES_BAND_FAILURE__low_minutes_floor"].iloc[0]
    assert row["recommended_next_action"] == "NEEDS_MORE_SAMPLE"


def test_unknown_clusters_compare_against_nearby_wins() -> None:
    attributed = pd.DataFrame(
        [
            {
                "player": "One_Off_Loss",
                "market_type": "PTS_OVER",
                "target": "PTS",
                "direction": "OVER",
                "market_line": 18.5,
                "predicted_probability": 0.67,
                "stress_probability": 0.65,
                "market_side_break_even": 0.52,
                "belief_uncertainty": 0.84,
                "line_decision_fragility_score": 0.58,
                "projected_team_fg_pct": 0.470,
                "result": "loss",
                "failure_modes": [],
                "recoverability_class": "ALEATORIC_OR_RANDOM",
                "game_date": "2026-02-01",
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
                "belief_uncertainty": 0.83,
                "line_decision_fragility_score": 0.57,
                "projected_team_fg_pct": 0.471,
                "result": "win",
                "failure_modes": [],
                "recoverability_class": "",
                "game_date": "2026-02-01",
                "team": "BBB",
            },
        ]
    )
    clusters, _, _ = discover_unknown_failures(attributed, min_cluster_losses=3, target_market_families=["PTS"])
    assert set(clusters["recommendation"].astype(str)).issubset({"NEEDS_MORE_SAMPLE", "REJECT_RANDOM"})


def test_broad_unsafe_families_are_not_proposed_as_interventions(tmp_path: Path) -> None:
    rows = _rows_with_team_offense_subregion(losses=15, wins=5, total=30)
    outputs_dir = tmp_path / "broad_unsafe"
    ledger_path = tmp_path / "ledger.jsonl"
    report = run_improvement_discovery(
        selected_rows=rows,
        candidate_pool_rows=rows.copy(),
        outputs_dir=outputs_dir,
        ledger_path=ledger_path,
        input_manifest={"selected_board_csvs": ["synthetic.csv"]},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["TEAM_OFFENSE_COLLAPSE", "LOW_TEAM_ASSIST_ENVIRONMENT", "MINUTES_BAND_FAILURE"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        broad_walk_forward=True,
        min_loss_count=3,
        min_resolved_count=8,
        min_pre_event_detectability=0.60,
        max_coverage_cost=0.25,
        max_non_target_damage=0.15,
        max_win_removal_rate=0.35,
        discovery_only=True,
        shadow_only=True,
    )
    assert report["status_label"] == "broad_signal_unsafe_to_act"
    assert report["candidate_interventions"] == []


def test_feature_gaps_are_reported_in_broad_walkforward_run(tmp_path: Path) -> None:
    rows = _rows_with_team_offense_subregion(losses=6, wins=2, total=20)
    report = run_improvement_discovery(
        selected_rows=rows,
        candidate_pool_rows=rows.copy(),
        outputs_dir=tmp_path / "feature_gaps",
        ledger_path=tmp_path / "ledger.jsonl",
        input_manifest={"selected_board_csvs": ["synthetic.csv"]},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["TEAM_OFFENSE_COLLAPSE", "LOW_TEAM_ASSIST_ENVIRONMENT", "USAGE_SUPPRESSION"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        broad_walk_forward=True,
        discovery_only=True,
        shadow_only=True,
    )
    gaps = {row["feature_name"]: row for row in report["feature_gap_report"]}
    assert "team_total" in gaps
    assert "usage_proxy" in gaps
    assert gaps["team_total"]["blocks_discovery"] is True


def test_discovery_only_mode_writes_required_outputs(tmp_path: Path) -> None:
    rows = _rows_with_minutes_subregion(losses=8, wins=2, total=40)
    outputs_dir = tmp_path / "outputs"
    report = run_improvement_discovery(
        selected_rows=rows,
        candidate_pool_rows=rows.copy(),
        outputs_dir=outputs_dir,
        ledger_path=tmp_path / "ledger.jsonl",
        input_manifest={"selected_board_csvs": ["synthetic.csv"]},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["MINUTES_BAND_FAILURE"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        broad_walk_forward=True,
        discovery_only=True,
        shadow_only=True,
    )
    assert report["mode"] == "broader_walkforward_discovery_only"
    for name in [
        "failure_mode_scoreboard.csv",
        "failure_subregion_scoreboard.csv",
        "unknown_failure_clusters.csv",
        "intervention_candidates.csv",
        "improvement_discovery_report.md",
        "improvement_discovery_report.json",
        "discovery_manifest.json",
    ]:
        assert (outputs_dir / name).exists(), name


def test_discovery_only_run_does_not_materialize_sidecars(tmp_path: Path) -> None:
    rows = _rows_with_minutes_subregion(losses=8, wins=2, total=40)
    outputs_dir = tmp_path / "no_sidecar"
    ledger_path = tmp_path / "ledger.jsonl"
    run_improvement_discovery(
        selected_rows=rows,
        candidate_pool_rows=rows.copy(),
        outputs_dir=outputs_dir,
        ledger_path=ledger_path,
        input_manifest={"selected_board_csvs": ["synthetic.csv"]},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["MINUTES_BAND_FAILURE"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        broad_walk_forward=True,
        discovery_only=True,
        shadow_only=True,
    )
    assert not (outputs_dir / "failure_mode_adjustments.csv").exists()
    ledger = load_improvement_ledger(ledger_path)
    assert len(ledger) == 1
    assert ledger.iloc[0]["promotion_status"] == "not_applicable"

