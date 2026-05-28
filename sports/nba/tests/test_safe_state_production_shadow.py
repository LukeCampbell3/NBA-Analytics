from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import compute_price_quality_frame
from research.safe_state.aggregate_safe_state_shadow_evidence import aggregate_safe_state_shadow_evidence
from research.safe_state.report_safe_state_production_status import build_safe_state_production_status
from research.safe_state.run_safe_state_production_shadow import run_safe_state_production_shadow
from research.safe_state.safe_state_promotion_gate import evaluate_safe_state_promotion_gate
from research.safe_state.update_safe_state_shadow_settlement import update_safe_state_shadow_settlement
from scripts.fetch_nba_market_props import _load_api_key_from_local_files


def _health(success: bool = True) -> dict[str, object]:
    return {
        "fetched_at_utc": "2026-05-27T16:00:00+00:00",
        "api_key_visible": success,
        "api_key": "supersecret-never-write",
        "request_success": success,
        "events_returned": 1 if success else 0,
        "odds_rows_returned": 4 if success else 0,
        "startsAt_available_count": 1 if success else 0,
        "side_specific_price_count": 4 if success else 0,
        "books_observed": ["draftkings"],
        "failure_reason": "" if success else "blocked",
    }


def _candidate(candidate_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "game_id": "game_1",
        "game_date": "2026-05-27",
        "market_date": "2026-05-27",
        "player": "Test Player",
        "player_name": "Test Player",
        "market_player_raw": "Test Player",
        "team": "AAA",
        "opponent": "BBB",
        "target": "PTS",
        "direction": "OVER",
        "side": "OVER",
        "market_type": "PTS_OVER",
        "line": 10.5,
        "market_line": 10.5,
        "market_side_price": -110,
        "market_side_break_even": 0.5238095,
        "market_side_decimal_odds": 1.9091,
        "over_price": -110,
        "under_price": -110,
        "price_source": "sportsgameodds_live_entry",
        "price_source_type": "LIVE_ENTRY",
        "price_validity_status": "PRICE_VALID",
        "diagnostic_only_flag": False,
        "timestamp_safe_flag": True,
        "timestamp_safety_basis": "EVENT_START_VERIFIED",
        "odds_snapshot_time": "2026-05-27T16:00:00Z",
        "prediction_snapshot_time": "2026-05-27T16:01:00Z",
        "selector_run_time": "2026-05-27T16:02:00Z",
        "market_commence_time_utc": "2026-05-28T00:30:00Z",
        "book": "draftkings",
        "provider": "sportsgameodds",
        "model_probability": 0.61,
        "stress_probability": 0.60,
        "lcb_probability": 0.56,
        "stress_edge": 0.076,
        "lcb_edge": 0.036,
        "edge_defendability_tier": "EDGE_DEFENDABLE",
        "forecastability_tier": "LOW_FORECASTABILITY",
        "forecastability_gap_primary": "FORECASTABILITY_GAP_MINUTES_STATE",
        "forecastability_gap_fixability": "TRUE_UNSTABLE_STATE",
        "forecastability_gap_severity": "CRITICAL",
        "similar_state_count": 4,
        "similar_state_reliability_tier": "INSUFFICIENT_SAMPLE",
        "similar_state_tightness_score": 0.25,
        "structural_mispricing_tier": "PRICE_ONLY_EDGE",
        "safe_state_tier": "SAFE_STATE_UNSTABLE",
        "safe_state_score": 0.44,
        "gap_subtype": "MINUTES_WIDE_BAND",
        "expected_minutes_band_low": 16,
        "expected_minutes_band_high": 34,
        "minutes_recent_median": 25,
        "gap_percentile": 0.95,
        "final_confidence": 0.70,
        "market_books": 1,
        "history_rows": 20,
        "thompson_ev": 0.01,
        "ev_adjusted": 0.01,
        "expected_win_rate": 0.60,
        "abs_edge": 0.05,
        "edge": 0.05,
        "belief_uncertainty": 0.1,
        "recommendation": "lean",
    }
    row.update(overrides)
    return row


def _write_shadow_inputs(tmp_path: Path) -> dict[str, Path]:
    health = tmp_path / "provider_healthcheck.json"
    health.write_text(json.dumps(_health(True)), encoding="utf-8")
    unstable = _candidate("candidate::unstable")
    sample = _candidate(
        "candidate::sample",
        player="Sample Player",
        player_name="Sample Player",
        market_player_raw="Sample Player",
        forecastability_gap_primary="FORECASTABILITY_GAP_USAGE_STATE",
        forecastability_gap_fixability="NEEDS_MORE_SAMPLE",
        forecastability_gap_severity="MEDIUM",
        safe_state_tier="SAFE_STATE_INSUFFICIENT_EVIDENCE",
        gap_subtype="USAGE_SAMPLE_INSUFFICIENT",
    )
    production = tmp_path / "production.csv"
    candidates = tmp_path / "candidates.csv"
    annotated = tmp_path / "annotated.csv"
    pd.DataFrame([unstable, sample]).to_csv(production, index=False)
    pd.DataFrame([unstable, sample]).to_csv(candidates, index=False)
    pd.DataFrame([unstable, sample]).to_csv(annotated, index=False)
    resolution = tmp_path / "resolution.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate::unstable",
                "player": "Test Player",
                "game_id": "game_1",
                "market_date": "2026-05-27",
                "target": "PTS",
                "direction": "OVER",
                "side": "OVER",
                "market_line": 10.5,
                "line": 10.5,
                "edge_defendability_tier": "EDGE_DEFENDABLE",
                "forecastability_gap_primary": "FORECASTABILITY_GAP_MINUTES_STATE",
                "gap_family": "FORECASTABILITY_GAP_MINUTES_STATE",
                "gap_subtype": "MINUTES_WIDE_BAND",
                "gap_fixability": "TRUE_UNSTABLE_STATE",
                "gap_severity": "CRITICAL",
                "recommended_next_action": "KEEP_UNSAFE_TRUE_VOLATILITY",
            },
            {
                "candidate_id": "candidate::sample",
                "player": "Sample Player",
                "game_id": "game_1",
                "market_date": "2026-05-27",
                "target": "PTS",
                "direction": "OVER",
                "side": "OVER",
                "market_line": 10.5,
                "line": 10.5,
                "edge_defendability_tier": "EDGE_DEFENDABLE",
                "forecastability_gap_primary": "FORECASTABILITY_GAP_USAGE_STATE",
                "gap_family": "FORECASTABILITY_GAP_USAGE_STATE",
                "gap_subtype": "USAGE_SAMPLE_INSUFFICIENT",
                "gap_fixability": "NEEDS_MORE_SAMPLE",
                "gap_severity": "MEDIUM",
                "recommended_next_action": "NEEDS_MORE_SAMPLE",
            },
        ]
    ).to_csv(resolution, index=False)
    root = tmp_path / "root.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate::unstable",
                "root_cause_primary": "REAL_MINUTES_VOLATILITY",
                "recommended_repair": "KEEP_UNSAFE_TRUE_VOLATILITY",
            },
            {
                "candidate_id": "candidate::sample",
                "root_cause_primary": "INSUFFICIENT_COMPARABLE_STATES",
                "recommended_repair": "NEEDS_MORE_SAMPLE",
            },
        ]
    ).to_csv(root, index=False)
    return {
        "health": health,
        "production": production,
        "candidates": candidates,
        "annotated": annotated,
        "resolution": resolution,
        "root": root,
    }


def test_provider_failure_stops_before_board_generation_and_hides_key(tmp_path: Path) -> None:
    health = tmp_path / "provider_healthcheck.json"
    health.write_text(json.dumps(_health(False)), encoding="utf-8")
    report = run_safe_state_production_shadow(
        season=2026,
        run_date="2026-05-27",
        output_dir=tmp_path / "run",
        provider_healthcheck_json=health,
        skip_production_pipeline=True,
    )

    assert report["status"] == "PROVIDER_BLOCKED"
    assert not (tmp_path / "run" / "production_board_as_is.csv").exists()
    assert "supersecret-never-write" not in (tmp_path / "run" / "provider_healthcheck.json").read_text(encoding="utf-8")


def test_sportsgameodds_key_can_resolve_from_parent_dotenv(tmp_path: Path) -> None:
    nested = tmp_path / "sports" / "nba" / "predictions" / "Player-Predictor" / "scripts"
    nested.mkdir(parents=True)
    (tmp_path / ".env").write_text("SPORTSGAMEODDS_API_KEY=structured_dotenv_key\n", encoding="utf-8")

    assert _load_api_key_from_local_files(nested) == "structured_dotenv_key"


def test_production_shadow_runner_keeps_production_board_unchanged_and_writes_shadow_outputs(tmp_path: Path) -> None:
    paths = _write_shadow_inputs(tmp_path)
    report = run_safe_state_production_shadow(
        season=2026,
        run_date="2026-05-27",
        output_dir=tmp_path / "run",
        provider_healthcheck_json=paths["health"],
        production_board_csv=paths["production"],
        candidate_pool_csv=paths["candidates"],
        annotated_candidates_csv=paths["annotated"],
        blocker_resolution_rows_csv=paths["resolution"],
        root_cause_rows_csv=paths["root"],
        skip_production_pipeline=True,
    )
    manifest = json.loads((tmp_path / "run" / "safe_state_production_shadow_manifest.json").read_text(encoding="utf-8"))

    assert report["production_behavior_changed"] is False
    assert manifest["production_board_hash"]
    assert (tmp_path / "run" / "safe_state_shadow_board_membership.csv").exists()
    assert (tmp_path / "run" / "true_unstable_shadow_rejections.csv").exists()
    assert (tmp_path / "run" / "needs_more_sample_queue.csv").exists()
    assert (tmp_path / "run" / "safe_state_near_core_board.csv").exists()
    assert "supersecret-never-write" not in (tmp_path / "run" / "safe_state_production_shadow_manifest.json").read_text(encoding="utf-8")


def test_event_start_verified_requires_snapshot_before_commence_time() -> None:
    before = pd.DataFrame(
        [
            _candidate(
                "candidate::before",
                odds_snapshot_time="2026-05-27T16:00:00Z",
                market_commence_time_utc="2026-05-28T00:30:00Z",
            )
        ]
    )
    after = pd.DataFrame(
        [
            _candidate(
                "candidate::after",
                odds_snapshot_time="2026-05-28T01:00:00Z",
                market_commence_time_utc="2026-05-28T00:30:00Z",
            )
        ]
    )

    before_quality = compute_price_quality_frame(before, record_scope="candidate")
    after_quality = compute_price_quality_frame(after, record_scope="candidate")

    assert before_quality.iloc[0]["timestamp_safety_basis"] == "EVENT_START_VERIFIED"
    assert bool(before_quality.iloc[0]["timestamp_safe_flag"]) is True
    assert after_quality.iloc[0]["timestamp_safety_basis"] != "EVENT_START_VERIFIED"
    assert bool(after_quality.iloc[0]["timestamp_safe_flag"]) is False


def test_settlement_updater_keeps_unresolved_pending_and_resolves_actual_rows(tmp_path: Path) -> None:
    paths = _write_shadow_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_safe_state_production_shadow(
        season=2026,
        run_date="2026-05-27",
        output_dir=run_dir,
        provider_healthcheck_json=paths["health"],
        production_board_csv=paths["production"],
        candidate_pool_csv=paths["candidates"],
        annotated_candidates_csv=paths["annotated"],
        blocker_resolution_rows_csv=paths["resolution"],
        root_cause_rows_csv=paths["root"],
        skip_production_pipeline=True,
    )
    actuals = tmp_path / "actuals.csv"
    pd.DataFrame([{"candidate_id": "candidate::unstable", "actual_stat": 11.0}]).to_csv(actuals, index=False)
    update_safe_state_shadow_settlement(run_dir=run_dir, actuals_source=actuals, output_dir=run_dir)
    metrics = pd.read_csv(run_dir / "safe_state_shadow_settlement_metrics.csv")
    production = metrics.loc[metrics["variant"].eq("production_board_as_is")].iloc[0]

    assert production["resolved_rows"] == 1
    assert production["pending_rows"] == 1
    assert production["pushes"] == 0


def test_aggregator_separates_pending_and_resolved_and_promotion_gate_stays_false(tmp_path: Path) -> None:
    paths = _write_shadow_inputs(tmp_path)
    base = tmp_path / "safe_state"
    run_dir = base / "20260527"
    run_safe_state_production_shadow(
        season=2026,
        run_date="2026-05-27",
        output_dir=run_dir,
        provider_healthcheck_json=paths["health"],
        production_board_csv=paths["production"],
        candidate_pool_csv=paths["candidates"],
        annotated_candidates_csv=paths["annotated"],
        blocker_resolution_rows_csv=paths["resolution"],
        root_cause_rows_csv=paths["root"],
        skip_production_pipeline=True,
    )
    aggregate = aggregate_safe_state_shadow_evidence(base_dir=base, output_dir=base / "aggregate")
    gate = evaluate_safe_state_promotion_gate(
        aggregate_metrics_csv=base / "aggregate" / "safe_state_shadow_aggregate_metrics.csv",
        output_dir=base / "aggregate",
    )
    status = build_safe_state_production_status(base_dir=base, output_dir=base / "aggregate")

    assert aggregate["promotion_gate"]["promotion_ready"] is False
    assert gate["promotion_ready"] is False
    assert status["promotion_ready"] is False
    rows = pd.read_csv(base / "aggregate" / "safe_state_shadow_aggregate_metrics.csv")
    production = rows.loc[rows["variant"].eq("production_board_as_is")].iloc[0]
    assert production["pending_rows"] >= 1
