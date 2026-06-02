from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.build_needs_more_sample_queue import build_needs_more_sample_queue
from research.safe_state.evaluate_safe_state_shadow_results import evaluate_safe_state_shadow_results
from research.safe_state.expand_comparable_state_sampling import expand_comparable_state_sampling
from research.safe_state.lock_true_unstable_shadow_rejections import lock_true_unstable_shadow_rejections
from research.safe_state.recheck_needs_more_sample_candidates import recheck_needs_more_sample_candidates
from research.safe_state.run_safe_state_evidence_lifecycle import run_safe_state_evidence_lifecycle
from research.safe_state.safe_state_evidence_gap_report import build_safe_state_evidence_gap_report
from research.safe_state.safe_state_evidence_ledger import append_safe_state_evidence_ledger


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::fox-pts",
        "game_id": "game_1",
        "game_date": "2026-05-20",
        "market_date": "2026-05-20",
        "player": "Test_Player",
        "player_name": "Test_Player",
        "target": "PTS",
        "market_type": "PTS_OVER",
        "side": "OVER",
        "direction": "OVER",
        "line": 14.5,
        "market_line": 14.5,
        "edge_defendability_tier": "EDGE_DEFENDABLE",
        "price_validity_status": "PRICE_VALID",
        "market_side_price": -110,
        "market_side_break_even": 0.5238,
        "stress_probability": 0.60,
        "lcb_probability": 0.56,
        "stress_edge": 0.07,
        "lcb_edge": 0.03,
        "forecastability_tier": "HIGH_FORECASTABILITY",
        "forecastability_gap_primary": "FORECASTABILITY_GAP_USAGE_STATE",
        "forecastability_gap_fixability": "NEEDS_MORE_SAMPLE",
        "forecastability_gap_severity": "MEDIUM",
        "similar_state_count": 2,
        "similar_state_reliability_tier": "INSUFFICIENT_SAMPLE",
        "similar_state_tightness_score": 0.0,
        "structural_mispricing_tier": "PRICE_ONLY_EDGE",
        "safe_state_tier": "SAFE_STATE_INSUFFICIENT_EVIDENCE",
        "line_zone": "NEAR_MEDIAN",
        "expected_minutes_band_low": 30,
        "expected_minutes_band_high": 35,
        "minutes_recent_median": 33,
    }
    row.update(overrides)
    return row


def _resolution(**overrides: object) -> dict[str, object]:
    row = {
        "candidate_id": "candidate::fox-pts",
        "player": "Test_Player",
        "game_id": "game_1",
        "market_date": "2026-05-20",
        "target": "PTS",
        "market_type": "PTS_OVER",
        "side": "OVER",
        "line": 14.5,
        "edge_defendability_tier": "EDGE_DEFENDABLE",
        "forecastability_gap_primary": "FORECASTABILITY_GAP_USAGE_STATE",
        "gap_family": "FORECASTABILITY_GAP_USAGE_STATE",
        "gap_subtype": "USAGE_SAMPLE_INSUFFICIENT",
        "gap_fixability": "NEEDS_MORE_SAMPLE",
        "gap_severity": "MEDIUM",
        "recommended_next_action": "NEEDS_MORE_SAMPLE",
    }
    row.update(overrides)
    return row


def _root(**overrides: object) -> dict[str, object]:
    row = {
        "candidate_id": "candidate::fox-pts",
        "root_cause_primary": "INSUFFICIENT_COMPARABLE_STATES",
        "recommended_repair": "NEEDS_MORE_SAMPLE",
    }
    row.update(overrides)
    return row


def _write_common(tmp_path: Path, *, true_unstable: bool = False) -> tuple[Path, Path, Path, Path]:
    annotated = pd.DataFrame([_candidate()])
    resolution = pd.DataFrame([_resolution()])
    root = pd.DataFrame([_root()])
    blockers = pd.DataFrame(
        [
            {
                "candidate_id": "candidate::fox-pts",
                "primary_blocker": "FORECASTABILITY_GAP_USAGE_STATE",
                "secondary_blockers": "",
                "missing_features": "",
                "evidence_gap_type": "FORECASTABILITY_GAP_USAGE_STATE",
            }
        ]
    )
    if true_unstable:
        annotated = pd.DataFrame(
            [
                _candidate(
                    candidate_id="candidate::unstable",
                    forecastability_gap_primary="FORECASTABILITY_GAP_MINUTES_STATE",
                    forecastability_gap_fixability="TRUE_UNSTABLE_STATE",
                    forecastability_gap_severity="CRITICAL",
                    safe_state_tier="SAFE_STATE_UNSTABLE",
                )
            ]
        )
        resolution = pd.DataFrame(
            [
                _resolution(
                    candidate_id="candidate::unstable",
                    forecastability_gap_primary="FORECASTABILITY_GAP_MINUTES_STATE",
                    gap_family="FORECASTABILITY_GAP_MINUTES_STATE",
                    gap_subtype="MINUTES_ROLE_UNSTABLE",
                    gap_fixability="TRUE_UNSTABLE_STATE",
                    gap_severity="CRITICAL",
                    recommended_next_action="KEEP_UNSAFE_TRUE_VOLATILITY",
                )
            ]
        )
        root = pd.DataFrame(
            [
                _root(
                    candidate_id="candidate::unstable",
                    root_cause_primary="REAL_MINUTES_VOLATILITY",
                    recommended_repair="KEEP_UNSAFE_TRUE_VOLATILITY",
                )
            ]
        )
    annotated_path = tmp_path / "annotated.csv"
    resolution_path = tmp_path / "resolution.csv"
    root_path = tmp_path / "root.csv"
    blockers_path = tmp_path / "blockers.csv"
    annotated.to_csv(annotated_path, index=False)
    resolution.to_csv(resolution_path, index=False)
    root.to_csv(root_path, index=False)
    blockers.to_csv(blockers_path, index=False)
    return annotated_path, resolution_path, root_path, blockers_path


def _write_logs(tmp_path: Path, *, scattered: bool = False) -> Path:
    data_proc = tmp_path / "Data-Proc"
    player_dir = data_proc / "test_player"
    player_dir.mkdir(parents=True)
    points = [15, 16, 15.5, 16.5, 15.2] if not scattered else [2, 35, 4, 34, 6]
    rows = [
        {"Date": f"2026-05-{day:02d}", "Player": "Test_Player", "MP": 32, "PTS": pts, "TRB": 4, "AST": 5, "FGA": 12}
        for day, pts in zip(range(10, 15), points)
    ]
    rows.append({"Date": "2026-05-20", "Player": "Test_Player", "MP": 32, "PTS": 100, "TRB": 20, "AST": 20, "FGA": 40})
    pd.DataFrame(rows).to_csv(player_dir / "2026_processed_processed.csv", index=False)
    return data_proc


def test_true_unstable_rows_enter_shadow_rejections(tmp_path: Path) -> None:
    annotated, resolution, root, blockers = _write_common(tmp_path, true_unstable=True)
    report = lock_true_unstable_shadow_rejections(
        output_dir=tmp_path,
        annotated_candidates_csv=annotated,
        blocker_resolution_rows_csv=resolution,
        root_cause_rows_csv=root,
        candidate_blockers_csv=blockers,
    )
    rows = pd.read_csv(tmp_path / "true_unstable_shadow_rejections.csv")

    assert report["locked_true_unstable_count"] == 1
    assert rows.iloc[0]["recommended_action"] == "KEEP_UNSAFE_TRUE_VOLATILITY"


def test_true_unstable_rows_are_not_actionable_pipeline_fixes(tmp_path: Path) -> None:
    annotated, _, _, _ = _write_common(tmp_path, true_unstable=True)
    pd.read_csv(annotated).to_csv(tmp_path / "safe_state_annotated_candidates.csv", index=False)
    pd.read_csv(annotated).to_csv(tmp_path / "price_defense_only_board.csv", index=False)
    report = build_safe_state_evidence_gap_report(
        output_dir=tmp_path,
        candidate_pool_csv=annotated,
        production_board_csv=annotated,
        safe_state_dir=tmp_path,
    )

    assert report["non_actionable_true_instability"]
    assert all(row["priority"] == "REJECT_UNSAFE" for row in report["non_actionable_true_instability"])


def test_needs_more_sample_rows_enter_queue(tmp_path: Path) -> None:
    annotated, resolution, root, _ = _write_common(tmp_path)
    report = build_needs_more_sample_queue(
        output_dir=tmp_path,
        blocker_resolution_rows_csv=resolution,
        root_cause_rows_csv=root,
        annotated_candidates_csv=annotated,
    )
    rows = pd.read_csv(tmp_path / "needs_more_sample_queue.csv")

    assert report["needs_more_sample_count"] == 1
    assert rows.iloc[0]["queue_status"] == "NEEDS_MORE_SAMPLE"


def test_comparable_state_expansion_respects_pre_event_cutoff_and_excludes_current_game(tmp_path: Path) -> None:
    annotated, resolution, root, _ = _write_common(tmp_path)
    build_needs_more_sample_queue(output_dir=tmp_path, blocker_resolution_rows_csv=resolution, root_cause_rows_csv=root, annotated_candidates_csv=annotated)
    data_proc = _write_logs(tmp_path)
    expand_comparable_state_sampling(
        output_dir=tmp_path,
        needs_more_sample_queue_csv=tmp_path / "needs_more_sample_queue.csv",
        annotated_candidates_csv=annotated,
        data_proc_dir=data_proc,
    )
    rows = pd.read_csv(tmp_path / "comparable_state_expansion_rows.csv")
    level3 = rows.loc[rows["fallback_level"].eq(3)].iloc[0]

    assert level3["match_count"] == 5
    assert level3["p90_abs_error"] < 90


def test_fallback_level_widening_increases_uncertainty_warning(tmp_path: Path) -> None:
    annotated, resolution, root, _ = _write_common(tmp_path)
    build_needs_more_sample_queue(output_dir=tmp_path, blocker_resolution_rows_csv=resolution, root_cause_rows_csv=root, annotated_candidates_csv=annotated)
    data_proc = _write_logs(tmp_path)
    expand_comparable_state_sampling(
        output_dir=tmp_path,
        needs_more_sample_queue_csv=tmp_path / "needs_more_sample_queue.csv",
        annotated_candidates_csv=annotated,
        data_proc_dir=data_proc,
    )
    rows = pd.read_csv(tmp_path / "comparable_state_expansion_rows.csv")

    assert rows.loc[rows["fallback_level"].eq(1), "uncertainty_penalty"].iloc[0] < rows.loc[rows["fallback_level"].eq(3), "uncertainty_penalty"].iloc[0]


def test_sufficient_tight_samples_move_queue_row_to_ready_for_recheck(tmp_path: Path) -> None:
    annotated, resolution, root, _ = _write_common(tmp_path)
    build_needs_more_sample_queue(output_dir=tmp_path, blocker_resolution_rows_csv=resolution, root_cause_rows_csv=root, annotated_candidates_csv=annotated)
    expansion = pd.DataFrame(
        [
            {
                "candidate_id": "candidate::fox-pts",
                "fallback_level": 2,
                "fallback_label": "same_player_target_line_zone",
                "match_count": 8,
                "tightness_score": 0.80,
                "comparable_state_reliability_tier": "TIGHT",
                "expansion_status": "SUFFICIENT_TIGHT",
            }
        ]
    )
    expansion.to_csv(tmp_path / "comparable_state_expansion_rows.csv", index=False)
    report = recheck_needs_more_sample_candidates(
        output_dir=tmp_path,
        needs_more_sample_queue_csv=tmp_path / "needs_more_sample_queue.csv",
        comparable_state_expansion_rows_csv=tmp_path / "comparable_state_expansion_rows.csv",
        annotated_candidates_csv=annotated,
    )

    assert report["recheck_status_counts"]["PROMOTED_TO_SAFE_STATE_NEAR_CORE"] == 1


def test_scattered_samples_move_row_to_rejected_scattered(tmp_path: Path) -> None:
    annotated, resolution, root, _ = _write_common(tmp_path)
    build_needs_more_sample_queue(output_dir=tmp_path, blocker_resolution_rows_csv=resolution, root_cause_rows_csv=root, annotated_candidates_csv=annotated)
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate::fox-pts",
                "fallback_level": 2,
                "fallback_label": "same_player_target_line_zone",
                "match_count": 8,
                "tightness_score": 0.20,
                "comparable_state_reliability_tier": "SCATTERED",
                "expansion_status": "SUFFICIENT_SCATTERED",
            }
        ]
    ).to_csv(tmp_path / "comparable_state_expansion_rows.csv", index=False)
    report = recheck_needs_more_sample_candidates(
        output_dir=tmp_path,
        needs_more_sample_queue_csv=tmp_path / "needs_more_sample_queue.csv",
        comparable_state_expansion_rows_csv=tmp_path / "comparable_state_expansion_rows.csv",
        annotated_candidates_csv=annotated,
    )

    assert report["recheck_status_counts"]["REJECTED_SIMILAR_STATE_SCATTER"] == 1


def test_recheck_promotion_is_shadow_only(tmp_path: Path) -> None:
    annotated, resolution, root, _ = _write_common(tmp_path)
    build_needs_more_sample_queue(output_dir=tmp_path, blocker_resolution_rows_csv=resolution, root_cause_rows_csv=root, annotated_candidates_csv=annotated)
    pd.DataFrame(
        [{"candidate_id": "candidate::fox-pts", "fallback_level": 2, "expansion_status": "SUFFICIENT_TIGHT", "comparable_state_reliability_tier": "TIGHT"}]
    ).to_csv(tmp_path / "comparable_state_expansion_rows.csv", index=False)
    recheck_needs_more_sample_candidates(
        output_dir=tmp_path,
        needs_more_sample_queue_csv=tmp_path / "needs_more_sample_queue.csv",
        comparable_state_expansion_rows_csv=tmp_path / "comparable_state_expansion_rows.csv",
        annotated_candidates_csv=annotated,
    )
    rows = pd.read_csv(tmp_path / "needs_more_sample_recheck.csv")

    assert rows.iloc[0]["production_eligible"] is False or str(rows.iloc[0]["production_eligible"]).lower() == "false"


def test_evidence_ledger_is_append_only(tmp_path: Path) -> None:
    pd.DataFrame([{"candidate_id": "candidate::unstable", "recommended_action": "KEEP_UNSAFE_TRUE_VOLATILITY", "gap_fixability": "TRUE_UNSTABLE_STATE"}]).to_csv(
        tmp_path / "true_unstable_shadow_rejections.csv", index=False
    )
    ledger = tmp_path / "safe_state_evidence_ledger.jsonl"
    append_safe_state_evidence_ledger(ledger_path=ledger, run_id="run1", true_unstable_csv=tmp_path / "true_unstable_shadow_rejections.csv")
    append_safe_state_evidence_ledger(ledger_path=ledger, run_id="run2", true_unstable_csv=tmp_path / "true_unstable_shadow_rejections.csv")

    assert len(ledger.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_settlement_evaluator_does_not_promote_from_one_slate(tmp_path: Path) -> None:
    frame = pd.DataFrame([_candidate(actual_result="WIN")])
    for name in ["production_board_as_is", "price_defense_only_board", "safe_state_core_board", "true_unstable_shadow_rejections", "needs_more_sample_queue"]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    report = evaluate_safe_state_shadow_results(board_dir=tmp_path)

    assert report["promotion_ready"] is False
    assert report["promotion_claim"] is False


def test_lifecycle_keeps_production_unchanged(tmp_path: Path) -> None:
    annotated, resolution, root, blockers = _write_common(tmp_path)
    data_proc = _write_logs(tmp_path)
    report = run_safe_state_evidence_lifecycle(
        output_dir=tmp_path / "lifecycle",
        annotated_candidates_csv=annotated,
        blocker_resolution_rows_csv=resolution,
        root_cause_rows_csv=root,
        candidate_blockers_csv=blockers,
        data_proc_dir=data_proc,
        run_id="test_run",
    )

    assert report["production_behavior_changed"] is False
    assert report["promotion_claim"] is False
