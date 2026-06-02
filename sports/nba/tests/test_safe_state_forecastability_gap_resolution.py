from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.analyze_minutes_forecastability_gap import annotate_minutes_gap_decomposition
from research.safe_state.analyze_usage_forecastability_gap import annotate_usage_gap_decomposition
from research.safe_state.forecastability_blocker_resolution_report import build_forecastability_blocker_resolution_report
from research.safe_state.safe_state_classifier import annotate_safe_state
from research.safe_state.safe_state_evidence_gap_report import build_safe_state_evidence_gap_report


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::test",
        "game_id": "game_1",
        "game_date": "2026-05-26",
        "market_date": "2026-05-26",
        "player": "Test_Player",
        "player_name": "Test_Player",
        "target": "PTS",
        "market_type": "PTS_OVER",
        "side": "OVER",
        "direction": "OVER",
        "line": 20.5,
        "market_line": 20.5,
        "price_validity_status": "PRICE_VALID",
        "edge_defendability_tier": "EDGE_DEFENDABLE",
        "stress_edge": 0.06,
        "lcb_edge": 0.03,
        "forecastability_tier": "HIGH_FORECASTABILITY",
        "similar_state_reliability_tier": "TIGHT",
        "structural_mispricing_tier": "STRUCTURAL_MISPRICE_STRONG",
        "scenario_agreement": 0.80,
        "chaos_score": 0.20,
    }
    row.update(overrides)
    return row


def _minutes_gap(**overrides: object) -> dict[str, object]:
    row = _candidate(
        forecastability_gap_primary="FORECASTABILITY_GAP_MINUTES_STATE",
        forecastability_gap_fixability="TRUE_UNSTABLE_STATE",
        forecastability_gap_severity="HIGH",
        minutes_state_gap_type="FORECASTABILITY_GAP_MINUTES_STATE",
        minutes_state_sample_count=8,
        minutes_floor_recent=30,
        expected_minutes_band_low=30,
        expected_minutes_band_high=35,
        expected_minutes_band_width=5,
        minutes_recent_cv=0.10,
        minutes_recent_std=3,
    )
    row.update(overrides)
    return row


def _usage_gap(**overrides: object) -> dict[str, object]:
    row = _candidate(
        forecastability_gap_primary="FORECASTABILITY_GAP_USAGE_STATE",
        forecastability_gap_fixability="TRUE_UNSTABLE_STATE",
        forecastability_gap_severity="HIGH",
        usage_state_sample_count=8,
    )
    row.update(overrides)
    return row


def _write_logs(tmp_path: Path) -> Path:
    data_proc = tmp_path / "Data-Proc"
    player_dir = data_proc / "test_player"
    player_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Date": f"2026-05-{day:02d}", "Player": "Test_Player", "MP": 30 + (day % 2), "FGA": 12, "FTA": 4, "AST": 5, "TRB": 6}
            for day in range(10, 18)
        ]
    ).to_csv(player_dir / "2026_processed_processed.csv", index=False)
    return data_proc


def test_low_minutes_floor_creates_minutes_low_floor() -> None:
    out = annotate_minutes_gap_decomposition(pd.DataFrame([_minutes_gap(minutes_floor_recent=14.0)]))

    assert out.iloc[0]["minutes_gap_subtype"] == "MINUTES_LOW_FLOOR"
    assert out.iloc[0]["minutes_gap_fixability"] == "TRUE_UNSTABLE_STATE"


def test_wide_minutes_band_creates_minutes_wide_band() -> None:
    out = annotate_minutes_gap_decomposition(pd.DataFrame([_minutes_gap(expected_minutes_band_width=13.0)]))

    assert out.iloc[0]["minutes_gap_subtype"] == "MINUTES_WIDE_BAND"


def test_high_minutes_cv_creates_minutes_high_volatility() -> None:
    out = annotate_minutes_gap_decomposition(pd.DataFrame([_minutes_gap(minutes_recent_cv=0.42)]))

    assert out.iloc[0]["minutes_gap_subtype"] == "MINUTES_HIGH_VOLATILITY"


def test_starter_bench_change_creates_minutes_role_unstable() -> None:
    out = annotate_minutes_gap_decomposition(
        pd.DataFrame([_minutes_gap(starter_status_change_count=2, starter_status_recent="bench_uncertain")])
    )

    assert out.iloc[0]["minutes_gap_subtype"] == "MINUTES_ROLE_UNSTABLE"


def test_missing_minutes_fields_with_logs_available_is_fixable_existing_logs(tmp_path: Path) -> None:
    data_proc = _write_logs(tmp_path)
    row = _candidate(forecastability_gap_primary="FORECASTABILITY_GAP_MINUTES_STATE")
    out = annotate_minutes_gap_decomposition(pd.DataFrame([row]), data_proc_dir=data_proc)

    assert out.iloc[0]["minutes_gap_subtype"] == "MINUTES_PIPELINE_MISSING"
    assert out.iloc[0]["minutes_gap_fixability"] == "FIXABLE_WITH_EXISTING_LOGS"


def test_fga_volatility_creates_usage_fga_volatile() -> None:
    out = annotate_usage_gap_decomposition(pd.DataFrame([_usage_gap(recent_fga_cv=0.45)]))

    assert out.iloc[0]["usage_gap_subtype"] == "USAGE_FGA_VOLATILE"
    assert out.iloc[0]["usage_gap_fixability"] == "TRUE_UNSTABLE_STATE"


def test_teammate_dependent_usage_creates_usage_teammate_dependent() -> None:
    out = annotate_usage_gap_decomposition(pd.DataFrame([_usage_gap(teammate_return_risk=0.70)]))

    assert out.iloc[0]["usage_gap_subtype"] == "USAGE_TEAMMATE_DEPENDENT"
    assert out.iloc[0]["usage_gap_fixability"] == "FIXABLE_WITH_NEW_PIPELINE_DATA"


def test_missing_usage_fields_creates_usage_pipeline_missing() -> None:
    out = annotate_usage_gap_decomposition(pd.DataFrame([_usage_gap()]))

    assert out.iloc[0]["usage_gap_subtype"] == "USAGE_PIPELINE_MISSING"
    assert out.iloc[0]["usage_gap_fixability"] == "FEATURE_MISSING"


def test_true_unstable_state_cannot_become_safe_state_near_core() -> None:
    row = _candidate(
        forecastability_gap_primary="FORECASTABILITY_GAP_MINUTES_STATE",
        forecastability_gap_fixability="TRUE_UNSTABLE_STATE",
        forecastability_gap_severity="CRITICAL",
    )
    out = annotate_safe_state(pd.DataFrame([row]))

    assert out.iloc[0]["safe_state_tier"] == "SAFE_STATE_UNSTABLE"


def test_one_fixable_non_critical_blocker_can_become_watch_near_core(tmp_path: Path) -> None:
    output_dir = tmp_path / "resolution"
    annotated = pd.DataFrame(
        [
            _candidate(
                forecastability_gap_primary="FORECASTABILITY_GAP_USAGE_STATE",
                forecastability_gap_fixability="FIXABLE_WITH_NEW_PIPELINE_DATA",
                forecastability_gap_severity="MEDIUM",
                usage_gap_subtype="USAGE_TEAMMATE_DEPENDENT",
                usage_gap_fixability="FIXABLE_WITH_NEW_PIPELINE_DATA",
                usage_gap_severity="MEDIUM",
                usage_gap_reason="teammate_dependency_score=0.700",
            )
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                "candidate_id": "candidate::test",
                "primary_blocker": "FORECASTABILITY_GAP_USAGE_STATE",
                "secondary_blockers": "",
                "missing_features": "teammate_availability",
                "evidence_gap_type": "FORECASTABILITY_GAP_USAGE_STATE",
            }
        ]
    )
    annotated_csv = output_dir / "annotated.csv"
    blockers_csv = output_dir / "blockers.csv"
    output_dir.mkdir()
    annotated.to_csv(annotated_csv, index=False)
    blockers.to_csv(blockers_csv, index=False)

    report = build_forecastability_blocker_resolution_report(
        output_dir=output_dir,
        annotated_candidates_csv=annotated_csv,
        candidate_blockers_csv=blockers_csv,
    )
    rows = pd.read_csv(output_dir / "forecastability_blocker_resolution_rows.csv")

    assert report["near_core_candidates_after_decomposition"] == 1
    assert rows.iloc[0]["recommended_next_action"] == "WATCH_NEAR_CORE"


def test_production_selection_remains_unchanged_in_evidence_report(tmp_path: Path) -> None:
    safe_state_dir = tmp_path / "safe_state"
    safe_state_dir.mkdir()
    annotated = pd.DataFrame([_minutes_gap(candidate_id="candidate::prod")])
    production = annotated.copy()
    annotated.to_csv(safe_state_dir / "safe_state_annotated_candidates.csv", index=False)
    production.to_csv(safe_state_dir / "production.csv", index=False)
    annotated.to_csv(safe_state_dir / "candidates.csv", index=False)
    annotated.to_csv(safe_state_dir / "price_defense_only_board.csv", index=False)

    report = build_safe_state_evidence_gap_report(
        output_dir=safe_state_dir,
        candidate_pool_csv=safe_state_dir / "candidates.csv",
        production_board_csv=safe_state_dir / "production.csv",
        safe_state_dir=safe_state_dir,
    )

    assert report["production_behavior_changed"] is False
    assert report["promotion_claim"] is False
    assert json.loads((safe_state_dir / "safe_state_evidence_gap_report.json").read_text(encoding="utf-8"))["shadow_only"] is True
