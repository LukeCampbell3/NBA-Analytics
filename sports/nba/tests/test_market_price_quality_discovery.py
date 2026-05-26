from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.backfill_break_even_fields import backfill_break_even_fields
from research.market_quality.common import american_odds_to_break_even, american_odds_to_decimal
from research.market_quality.stale_price_dependency import annotate_stale_price_dependency_rows
from research.run_improvement_discovery import run_improvement_discovery


def _audit_row(index: int, **overrides: object) -> dict[str, object]:
    game_date = f"2026-02-{(index % 8) + 1:02d}"
    row: dict[str, object] = {
        "candidate_id": f"candidate::{index}",
        "record_scope": "selected",
        "selected_on_board": True,
        "player": f"Player_{index}",
        "market_player_raw": f"Player_{index}",
        "game_date": game_date,
        "market_date": game_date,
        "run_date": game_date,
        "actual_matched_date": game_date,
        "source_selected_board_csv": f"window_{index % 2}",
        "source_candidate_pool_csv": f"selector_{index % 2}.csv",
        "actual_team": f"T{index % 4}",
        "team": f"T{index % 4}",
        "opponent": f"O{index % 4}",
        "target": "PTS",
        "direction": "OVER",
        "market_id": "PTS_OVER",
        "market_type": "PTS_OVER",
        "market_line": 20.5,
        "prediction": 21.4,
        "recommendation": "consider",
        "selected_rank": 1,
        "expected_win_rate": 0.56,
        "stress_probability": 0.54,
        "model_probability": 0.56,
        "result": "loss",
        "units": -1.0,
        "existing_market_side_price": np.nan,
        "existing_market_side_break_even": np.nan,
        "snapshot_market_side_price": -115.0,
        "snapshot_market_side_break_even": american_odds_to_break_even(-115.0),
        "snapshot_over_price": -115.0,
        "snapshot_under_price": -105.0,
        "snapshot_market_line": 20.5,
        "snapshot_source": "current_market_snapshot",
        "price_source_hint": "",
        "odds_snapshot_time": f"{game_date}T16:00:00+00:00",
        "prediction_snapshot_time": f"{game_date}T16:20:00+00:00",
        "stale_price_flag": False,
        "line_moved_since_prediction": 0.0,
        "odds_moved_since_prediction": 0.0,
    }
    row.update(overrides)
    return row


def _focused_price_quality_rows(losses: int, wins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = losses + wins
    for idx in range(total):
        rows.append(
            _audit_row(
                idx,
                result="loss" if idx < losses else "win",
                units=-1.0 if idx < losses else 0.91,
                stale_price_flag=True,
                snapshot_market_side_price=-118.0,
                snapshot_market_side_break_even=american_odds_to_break_even(-118.0),
            )
        )
    audit = pd.DataFrame(rows)
    backfilled, _ = backfill_break_even_fields(audit)
    return annotate_stale_price_dependency_rows(backfilled)


def test_american_odds_convert_to_break_even_correctly() -> None:
    assert round(american_odds_to_decimal(-110), 6) == round(1.9090909090909092, 6)
    assert round(american_odds_to_break_even(-110), 6) == round(110.0 / 210.0, 6)
    assert round(american_odds_to_decimal(150), 6) == round(2.5, 6)
    assert round(american_odds_to_break_even(150), 6) == round(100.0 / 250.0, 6)


def test_missing_price_creates_missing_price_status() -> None:
    audit = pd.DataFrame(
        [
            _audit_row(
                0,
                existing_market_side_price=np.nan,
                snapshot_market_side_price=np.nan,
                snapshot_market_side_break_even=np.nan,
                snapshot_over_price=np.nan,
                snapshot_under_price=np.nan,
            )
        ]
    )
    backfilled, _ = backfill_break_even_fields(audit)
    assert backfilled.iloc[0]["price_validity_status"] == "MISSING_PRICE"


def test_invalid_price_creates_invalid_price_status() -> None:
    audit = pd.DataFrame([_audit_row(0, snapshot_market_side_price=-10.0, snapshot_market_side_break_even=np.nan)])
    backfilled, _ = backfill_break_even_fields(audit)
    assert backfilled.iloc[0]["price_validity_status"] == "INVALID_PRICE"


def test_stale_price_creates_stale_price_dependency_flag() -> None:
    audit = pd.DataFrame([_audit_row(0, stale_price_flag=True)])
    backfilled, _ = backfill_break_even_fields(audit)
    annotated = annotate_stale_price_dependency_rows(backfilled)
    assert annotated.iloc[0]["stale_price_subregion"] == "STALE_PRICE_DEPENDENCY"


def test_corrected_price_can_move_candidate_to_price_dependent() -> None:
    audit = pd.DataFrame(
        [
            _audit_row(
                0,
                stale_price_flag=True,
                stress_probability=0.54,
                existing_market_side_break_even=american_odds_to_break_even(-110.0),
                snapshot_market_side_price=-117.0,
                snapshot_market_side_break_even=american_odds_to_break_even(-117.0),
            )
        ]
    )
    backfilled, _ = backfill_break_even_fields(audit)
    annotated = annotate_stale_price_dependency_rows(backfilled)
    assert annotated.iloc[0]["proposed_decision_after_price_fix"] == "PRICE_DEPENDENT"


def test_close_only_odds_are_marked_diagnostic_only() -> None:
    audit = pd.DataFrame([_audit_row(0, price_source_hint="close_only_snapshot")])
    backfilled, _ = backfill_break_even_fields(audit)
    annotated = annotate_stale_price_dependency_rows(backfilled)
    assert backfilled.iloc[0]["price_validity_status"] == "DIAGNOSTIC_ONLY"
    assert backfilled.iloc[0]["price_source_type"] == "CLOSE_ONLY_DIAGNOSTIC"
    assert annotated.iloc[0]["proposed_decision_after_price_fix"] == "DIAGNOSTIC_ONLY"


def test_backfill_never_uses_postgame_data() -> None:
    audit = pd.DataFrame(
        [
            _audit_row(
                0,
                game_date="2026-02-01",
                market_date="2026-02-01",
                odds_snapshot_time="2026-02-02T01:00:00+00:00",
                snapshot_market_side_price=-118.0,
                snapshot_market_side_break_even=american_odds_to_break_even(-118.0),
            )
        ]
    )
    backfilled, _ = backfill_break_even_fields(audit)
    assert backfilled.iloc[0]["price_validity_status"] == "DIAGNOSTIC_ONLY"
    assert pd.isna(backfilled.iloc[0]["corrected_price"])


def test_missing_price_rows_stay_feature_gap_blocked_for_stale_price_target(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            _audit_row(
                idx,
                result="loss",
                existing_market_side_price=np.nan,
                existing_market_side_break_even=np.nan,
                snapshot_market_side_price=np.nan,
                snapshot_market_side_break_even=np.nan,
                snapshot_over_price=np.nan,
                snapshot_under_price=np.nan,
            )
            for idx in range(10)
        ]
    )
    backfilled, _ = backfill_break_even_fields(rows)
    annotated = annotate_stale_price_dependency_rows(backfilled)
    report = run_improvement_discovery(
        selected_rows=annotated.copy(),
        candidate_pool_rows=annotated.copy(),
        outputs_dir=tmp_path / "feature_gap",
        ledger_path=tmp_path / "ledger.jsonl",
        input_manifest={"price_quality_rows": "synthetic_missing.csv"},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["MARKET_PRICE_MISPLACEMENT"],
        target_subregions=["MARKET_PRICE_MISPLACEMENT__stale_price_dependency"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        price_quality_mode=True,
        min_loss_count=3,
        min_resolved_count=8,
        min_pre_event_detectability=0.90,
        max_coverage_cost=0.05,
        max_non_target_damage=0.02,
        max_win_removal_rate=0.35,
        discovery_only=True,
        shadow_only=True,
    )
    assert report["status_label"] == "feature_gap_blocked"
    assert report["candidate_interventions"] == []


def test_stale_price_discovery_ignores_diagnostic_only_rows(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            _audit_row(
                idx,
                result="loss",
                price_source_hint="close_only_snapshot",
            )
            for idx in range(10)
        ]
    )
    backfilled, _ = backfill_break_even_fields(rows)
    annotated = annotate_stale_price_dependency_rows(backfilled)
    report = run_improvement_discovery(
        selected_rows=annotated.copy(),
        candidate_pool_rows=annotated.copy(),
        outputs_dir=tmp_path / "diagnostic_only",
        ledger_path=tmp_path / "ledger.jsonl",
        input_manifest={"price_quality_rows": "synthetic_diagnostic.csv"},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["MARKET_PRICE_MISPLACEMENT"],
        target_subregions=["MARKET_PRICE_MISPLACEMENT__stale_price_dependency"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        price_quality_mode=True,
        min_loss_count=3,
        min_resolved_count=8,
        min_pre_event_detectability=0.90,
        max_coverage_cost=0.05,
        max_non_target_damage=0.02,
        max_win_removal_rate=0.35,
        discovery_only=True,
        shadow_only=True,
    )
    assert annotated["would_change_decision"].sum() == 0
    assert report["status_label"] == "feature_gap_blocked"
    assert report["candidate_interventions"] == []


def test_stale_price_discovery_does_not_create_production_sidecars(tmp_path: Path) -> None:
    rows = _focused_price_quality_rows(losses=7, wins=0)
    outputs_dir = tmp_path / "stale_discovery"
    report = run_improvement_discovery(
        selected_rows=rows.copy(),
        candidate_pool_rows=rows.copy(),
        outputs_dir=outputs_dir,
        ledger_path=tmp_path / "ledger.jsonl",
        input_manifest={"price_quality_rows": "synthetic.csv"},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["MARKET_PRICE_MISPLACEMENT"],
        target_subregions=["MARKET_PRICE_MISPLACEMENT__stale_price_dependency"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        price_quality_mode=True,
        min_loss_count=3,
        min_resolved_count=8,
        min_pre_event_detectability=0.90,
        max_coverage_cost=0.05,
        max_non_target_damage=0.02,
        max_win_removal_rate=0.35,
        discovery_only=True,
        shadow_only=True,
    )
    assert report["mode"] == "price_quality_discovery_only"
    assert not (outputs_dir / "failure_mode_adjustments.csv").exists()


def test_intervention_candidates_are_empty_unless_actionability_passes(tmp_path: Path) -> None:
    rows = _focused_price_quality_rows(losses=7, wins=0)
    report = run_improvement_discovery(
        selected_rows=rows.copy(),
        candidate_pool_rows=rows.copy(),
        outputs_dir=tmp_path / "needs_sample",
        ledger_path=tmp_path / "ledger.jsonl",
        input_manifest={"price_quality_rows": "synthetic.csv"},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["MARKET_PRICE_MISPLACEMENT"],
        target_subregions=["MARKET_PRICE_MISPLACEMENT__stale_price_dependency"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        price_quality_mode=True,
        min_loss_count=3,
        min_resolved_count=8,
        min_pre_event_detectability=0.90,
        max_coverage_cost=0.05,
        max_non_target_damage=0.02,
        max_win_removal_rate=0.35,
        discovery_only=True,
        shadow_only=True,
    )
    assert report["status_label"] == "needs_more_sample"
    assert report["candidate_interventions"] == []


def test_required_outputs_are_written_for_price_quality_run(tmp_path: Path) -> None:
    rows = _focused_price_quality_rows(losses=7, wins=0)
    outputs_dir = tmp_path / "outputs"
    run_improvement_discovery(
        selected_rows=rows.copy(),
        candidate_pool_rows=rows.copy(),
        outputs_dir=outputs_dir,
        ledger_path=tmp_path / "ledger.jsonl",
        input_manifest={"price_quality_rows": "synthetic.csv"},
        data_proc_root=tmp_path / "missing_proc",
        target_failure_modes=["MARKET_PRICE_MISPLACEMENT"],
        target_subregions=["MARKET_PRICE_MISPLACEMENT__stale_price_dependency"],
        excluded_failure_modes=set(),
        excluded_market_families=set(),
        discover_subregions=True,
        price_quality_mode=True,
        min_loss_count=3,
        min_resolved_count=8,
        min_pre_event_detectability=0.90,
        max_coverage_cost=0.05,
        max_non_target_damage=0.02,
        max_win_removal_rate=0.35,
        discovery_only=True,
        shadow_only=True,
    )
    for name in [
        "failure_mode_scoreboard.csv",
        "failure_subregion_scoreboard.csv",
        "unknown_failure_clusters.csv",
        "intervention_candidates.csv",
        "improvement_discovery_report.md",
        "improvement_discovery_report.json",
        "stale_price_dependency_rows.csv",
        "stale_price_dependency_summary.json",
    ]:
        assert (outputs_dir / name).exists(), name
