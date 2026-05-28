from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.event_time_resolver import resolve_event_times
from research.market_quality.priced_event_ledger import build_priced_event_ledger_frame
from research.market_quality.report_edge_defense import build_edge_defense_report
from research.validation.diagnose_recency_gate import diagnose_recency_gate


def _base_price_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::event-time",
        "player": "Event Time",
        "market_player_raw": "Event Time",
        "player_name": "Event Time",
        "team": "SAS",
        "opponent": "OKC",
        "game_id": "game_1",
        "market_event_id": "game_1",
        "game_date": "2026-05-26",
        "market_date": "2026-05-26",
        "target": "PTS",
        "direction": "OVER",
        "side": "OVER",
        "market_type": "PTS_OVER",
        "market_line": 20.5,
        "line": 20.5,
        "market_side_price": -110.0,
        "over_price": -110.0,
        "under_price": -110.0,
        "price_source": "current_market_snapshot_pre_event",
        "price_source_type": "ARCHIVED_ENTRY",
        "odds_snapshot_time": "2026-05-26T18:00:00+00:00",
        "prediction_snapshot_time": "2026-05-26T18:05:00+00:00",
        "selector_run_time": "2026-05-26T18:06:00+00:00",
        "expected_win_rate": 0.58,
        "model_probability": 0.59,
        "stress_probability": 0.58,
        "lcb_probability": 0.56,
        "forecastability_score": 0.80,
        "scenario_agreement": 0.75,
        "chaos_score": 0.20,
    }
    row.update(overrides)
    return row


def test_event_time_resolver_uses_provider_time() -> None:
    rows = pd.DataFrame(
        [
            {
                "Market_Event_ID": "game_1",
                "Market_Date": "2026-05-26",
                "Market_Home_Team": "SAS",
                "Market_Away_Team": "OKC",
                "Market_Commence_Time_UTC": "2026-05-27T00:30:00+00:00",
            }
        ]
    )

    resolved = resolve_event_times(rows)

    assert str(resolved.iloc[0]["event_time_source"]) == "PROVIDER"
    assert str(resolved.iloc[0]["event_time_confidence"]) == "exact"
    assert pd.notna(resolved.iloc[0]["market_commence_time_utc"])


def test_event_time_resolver_falls_back_to_schedule_exact_match() -> None:
    rows = pd.DataFrame(
        [
            {
                "Market_Event_ID": "game_1",
                "Market_Date": "2026-05-26",
                "Market_Home_Team": "SAS",
                "Market_Away_Team": "OKC",
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "game_id": "game_1",
                "game_date": "2026-05-26",
                "home_team": "SAS",
                "away_team": "OKC",
                "commence_time_utc": "2026-05-27T00:30:00+00:00",
            }
        ]
    )

    resolved = resolve_event_times(rows, schedule_rows=schedule)

    assert str(resolved.iloc[0]["event_time_source"]) == "NBA_SCHEDULE"
    assert str(resolved.iloc[0]["event_time_confidence"]) == "exact"
    assert pd.notna(resolved.iloc[0]["market_commence_time_utc"])


def test_unresolved_event_time_blocks_event_start_verified_safety() -> None:
    ledger = build_priced_event_ledger_frame(pd.DataFrame([_base_price_row()]), record_scope="candidate")
    row = ledger.iloc[0]

    assert bool(row["timestamp_safe_flag"]) is False
    assert row["timestamp_safety_basis"] == "NOT_VERIFIED"
    assert row["timestamp_safety_blocked_reason"] == "missing_event_time_and_no_explicit_prelock_run"
    assert row["price_validity_status"] == "STALE_PRICE"


def test_timestamp_safe_flag_false_when_price_missing() -> None:
    ledger = build_priced_event_ledger_frame(
        pd.DataFrame(
            [
                _base_price_row(
                    market_side_price=pd.NA,
                    over_price=pd.NA,
                    market_commence_time_utc="2026-05-27T00:30:00+00:00",
                )
            ]
        ),
        record_scope="candidate",
    )
    row = ledger.iloc[0]

    assert bool(row["timestamp_safe_flag"]) is False
    assert row["price_validity_status"] == "MISSING_PRICE"


def test_timestamp_safe_flag_false_for_diagnostic_only_price() -> None:
    ledger = build_priced_event_ledger_frame(
        pd.DataFrame(
            [
                _base_price_row(
                    price_source_type="CLOSE_ONLY_DIAGNOSTIC",
                    market_commence_time_utc="2026-05-27T00:30:00+00:00",
                )
            ]
        ),
        record_scope="candidate",
    )
    row = ledger.iloc[0]

    assert bool(row["timestamp_safe_flag"]) is False
    assert row["timestamp_safety_basis"] == "NOT_VERIFIED"
    assert row["price_validity_status"] == "DIAGNOSTIC_ONLY"
    assert row["edge_defendability_tier"] == "EDGE_DIAGNOSTIC_ONLY"


def test_prelock_run_verified_does_not_equal_event_start_verified() -> None:
    prelock = build_priced_event_ledger_frame(
        pd.DataFrame([_base_price_row(explicit_prelock_run_flag=True)]),
        record_scope="candidate",
    )
    event_start = build_priced_event_ledger_frame(
        pd.DataFrame([_base_price_row(market_commence_time_utc="2026-05-27T00:30:00+00:00")]),
        record_scope="candidate",
    )

    assert bool(prelock.iloc[0]["timestamp_safe_flag"]) is True
    assert prelock.iloc[0]["timestamp_safety_basis"] == "PRELOCK_RUN_VERIFIED"
    assert bool(event_start.iloc[0]["timestamp_safe_flag"]) is True
    assert event_start.iloc[0]["timestamp_safety_basis"] == "EVENT_START_VERIFIED"


def test_event_start_verified_requires_odds_snapshot_before_commence() -> None:
    ledger = build_priced_event_ledger_frame(
        pd.DataFrame(
            [
                _base_price_row(
                    market_commence_time_utc="2026-05-27T00:30:00+00:00",
                    odds_snapshot_time="2026-05-27T00:30:00+00:00",
                    price_source_type="LIVE_ENTRY",
                )
            ]
        ),
        record_scope="candidate",
    )
    assert bool(ledger.iloc[0]["timestamp_safe_flag"]) is False
    assert ledger.iloc[0]["timestamp_safety_basis"] == "NOT_VERIFIED"
    assert ledger.iloc[0]["timestamp_safety_blocked_reason"] == "odds_snapshot_not_before_event_start"


def test_recency_diagnosis_detects_stale_history_root_cause(tmp_path: Path) -> None:
    selector_csv = tmp_path / "selector.csv"
    final_json = tmp_path / "final.json"
    slate_csv = tmp_path / "slate.csv"
    pd.DataFrame(
        [
            {
                "player": "A",
                "target": "PTS",
                "recency_factor": 0.40,
                "market_date": "2026-05-26",
                "last_history_date": "2026-04-25",
            },
            {
                "player": "B",
                "target": "AST",
                "recency_factor": 0.42,
                "market_date": "2026-05-26",
                "last_history_date": "2026-04-26",
            },
        ]
    ).to_csv(selector_csv, index=False)
    pd.DataFrame([{"player": "A"}]).to_csv(slate_csv, index=False)
    final_json.write_text(
        json.dumps(
            {
                "policy": {"min_recency_factor": 0.88, "max_history_staleness_days": 14},
                "pipeline_stage_counts": {"after_initial_pool_gate": 2, "after_recency": 0, "final_board_rows": 0},
            }
        ),
        encoding="utf-8",
    )

    diagnosis = diagnose_recency_gate(
        selector_csv=selector_csv,
        final_json=final_json,
        slate_csv=slate_csv,
        output_dir=tmp_path,
    )

    assert diagnosis["freshness_root_cause"] == "player_game_logs_stale"
    assert diagnosis["data_pipeline_history_freshness_is_stale"] is True
    assert (tmp_path / "recency_removed_rows.csv").exists()


def test_edge_defense_report_handles_empty_final_board(tmp_path: Path) -> None:
    audit_csv = tmp_path / "price_provenance_audit.csv"
    recency_json = tmp_path / "recency_gate_diagnosis.json"
    build_priced_event_ledger_frame(
        pd.DataFrame([_base_price_row(explicit_prelock_run_flag=True)]),
        record_scope="candidate",
    ).to_csv(audit_csv, index=False)
    recency_json.write_text(
        json.dumps({"rows_before_recency": 3, "rows_after_recency": 0, "freshness_root_cause": "player_game_logs_stale"}),
        encoding="utf-8",
    )

    report = build_edge_defense_report(
        output_dir=tmp_path,
        price_audit_csv=audit_csv,
        recency_diagnosis_json=recency_json,
    )

    assert report["total_candidate_rows"] == 1
    assert report["total_selected_rows"] == 0
    assert report["recency_blocks_production_selection"] is True
    selected_detail = pd.read_csv(tmp_path / "edge_defense_selected_rows.csv")
    assert selected_detail.empty
