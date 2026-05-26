from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from fetch_nba_market_props import normalize_wide_snapshot
from collect_covers_historical_props import build_wide_from_covers_long
from research.market_quality.common import augment_with_snapshot_prices
from research.market_quality.priced_event_ledger import build_priced_event_ledger_frame
from run_daily_market_pipeline import annotate_market_snapshot_provenance, persist_market_snapshot_manifest


def test_normalize_wide_snapshot_adds_price_provenance_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "Player": "Test_Player",
                "Market_Date": "2026-05-01",
                "Market_Player_Raw": "Test Player",
                "Market_Event_ID": "evt_1",
                "Market_PTS": 22.5,
                "Market_PTS_over_price": -115.0,
                "Market_PTS_under_price": -105.0,
                "Market_Fetched_At_UTC": "2026-05-01T15:00:00+00:00",
            }
        ]
    )

    _, wide = normalize_wide_snapshot(raw, "2026-05-01T15:05:00+00:00")
    row = wide.iloc[0]

    assert row["Market_Provider"] == "snapshot"
    assert row["Market_Book"] == "aggregate_market_snapshot"
    assert row["Market_Price_Source"] == "snapshot_input"
    assert row["Market_Price_Source_Type"] == "ARCHIVED_ENTRY"
    assert row["Market_Snapshot_ID"] == "snapshot:2026-05-01T15:00:00+00:00"


def test_annotate_market_snapshot_provenance_fills_missing_run_snapshot_fields() -> None:
    frame = pd.DataFrame(
        [
            {
                "Player": "Run Snapshot",
                "Market_Date": "2026-05-02",
                "Market_Event_ID": "evt_run",
                "Market_PTS": 18.5,
                "Market_PTS_over_price": -110.0,
                "Market_PTS_under_price": -110.0,
                "Market_Fetched_At_UTC": "2026-05-02T14:30:00+00:00",
            }
        ]
    )

    annotated = annotate_market_snapshot_provenance(
        frame,
        source_manifest={"provider": "odds_api", "input_path": "raw/source/latest_player_props_wide.parquet"},
    )
    row = annotated.iloc[0]

    assert row["Market_Provider"] == "odds_api"
    assert row["Market_Book"] == "aggregate_market_snapshot"
    assert row["Market_Price_Source"] == "raw/source/latest_player_props_wide.parquet"
    assert row["Market_Price_Source_Type"] == "ARCHIVED_ENTRY"
    assert row["Market_Snapshot_ID"] == "odds_api:2026-05-02T14:30:00+00:00"


def test_persist_market_snapshot_manifest_writes_run_local_copy(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    source_snapshot = source_dir / "latest_player_props_wide.parquet"
    source_snapshot.touch()
    source_manifest = {
        "provider": "odds_api",
        "fetched_at_utc": "2026-05-02T14:30:00+00:00",
        "input_path": "raw/source/latest_player_props_wide.parquet",
    }
    (source_dir / "latest_manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_snapshot = run_dir / "current_market_snapshot_20260502.parquet"
    output_snapshot.touch()

    meta = persist_market_snapshot_manifest(
        source_snapshot_path=source_snapshot,
        output_snapshot_path=output_snapshot,
        run_stamp="20260502",
        snapshot_meta={"mode": "requested_window", "selected_row_count": 12},
    )

    latest_manifest_path = Path(meta["latest_manifest_path"])
    stamped_manifest_path = Path(meta["stamped_manifest_path"])
    assert latest_manifest_path.exists()
    assert stamped_manifest_path.exists()

    payload = json.loads(latest_manifest_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "odds_api"
    assert payload["snapshot_selection_meta"]["selected_row_count"] == 12
    assert payload["output_snapshot_path"] == str(output_snapshot.resolve())


def test_timestamp_safety_uses_commence_time_not_utc_date_boundary() -> None:
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate::late-night",
                "player": "Late Night",
                "market_player_raw": "Late Night",
                "player_name": "Late Night",
                "team": "AAA",
                "opponent": "BBB",
                "market_event_id": "evt_late",
                "game_id": "evt_late",
                "game_date": "2026-04-29",
                "market_date": "2026-04-29",
                "market_commence_time_utc": "2026-04-30T04:00:00+00:00",
                "target": "PTS",
                "direction": "OVER",
                "side": "OVER",
                "market_type": "PTS_OVER",
                "market_line": 22.5,
                "line": 22.5,
                "market_side_price": -110.0,
                "over_price": -110.0,
                "under_price": -110.0,
                "price_source": "current_market_snapshot_pre_event",
                "odds_snapshot_time": "2026-04-30T03:17:00+00:00",
                "prediction_snapshot_time": "2026-04-30T03:20:00+00:00",
                "selector_run_time": "2026-04-30T03:21:00+00:00",
                "expected_win_rate": 0.56,
                "model_probability": 0.57,
                "stress_probability": 0.56,
                "lcb_probability": 0.55,
                "p_push": 0.0,
                "forecastability_score": 0.80,
                "scenario_agreement": 0.75,
                "chaos_score": 0.20,
                "recommendation": "strong",
                "history_rows": 100,
            }
        ]
    )

    ledger = build_priced_event_ledger_frame(rows, record_scope="selected")
    row = ledger.iloc[0]

    assert bool(row["timestamp_safe_flag"]) is True
    assert row["price_validity_status"] == "PRICE_VALID"


def test_augment_with_snapshot_prices_handles_missing_snapshot_side_columns() -> None:
    selector_rows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate::no-snapshot-columns",
                "player": "No Snapshot",
                "market_player_raw": "No Snapshot",
                "player_key": "No_Snapshot",
                "market_date": "2026-05-03",
                "target": "PTS",
                "direction": "OVER",
                "source_candidate_pool_csv": "tmp/nonexistent/upcoming_market_play_selector.csv",
            }
        ]
    )

    augmented = augment_with_snapshot_prices(selector_rows)

    assert "snapshot_no_vig_probability" in augmented.columns
    assert pd.isna(augmented.iloc[0]["snapshot_no_vig_probability"])


def test_augment_with_snapshot_prices_preserves_existing_odds_snapshot_time(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    snapshot = pd.DataFrame(
        [
            {
                "Market_Date": "2026-05-03",
                "Player": "Snapshot_Player",
                "Market_Player_Raw": "Snapshot Player",
                "Market_Event_ID": "evt_snapshot",
                "Market_Provider": "covers_historical",
                "Market_Book": "covers_consensus",
                "Market_Price_Source": "covers_matchup_props",
                "Market_Price_Source_Type": "ARCHIVED_ENTRY",
                "Market_Snapshot_ID": "covers_historical:2026-05-03T14:00:00+00:00",
                "Market_PTS": 12.5,
                "Market_PTS_over_price": -110.0,
                "Market_PTS_under_price": -110.0,
                "Market_Fetched_At_UTC": "2026-05-03T14:00:00+00:00",
            }
        ]
    )
    snapshot.to_parquet(run_dir / "current_market_snapshot_20260503.parquet", index=False)
    selector_rows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate::snapshot-player",
                "player": "Snapshot Player",
                "market_player_raw": "Snapshot Player",
                "player_key": "Snapshot_Player",
                "market_date": "2026-05-03",
                "target": "PTS",
                "direction": "OVER",
                "odds_snapshot_time": "2026-05-03T14:00:00+00:00",
                "source_candidate_pool_csv": str(run_dir / "upcoming_market_play_selector_20260503.csv"),
            }
        ]
    )

    augmented = augment_with_snapshot_prices(selector_rows)

    assert "odds_snapshot_time" in augmented.columns
    assert str(augmented.iloc[0]["odds_snapshot_time"]) == "2026-05-03T14:00:00+00:00"


def test_covers_wide_snapshot_populates_price_provenance_fields() -> None:
    long_df = pd.DataFrame(
        [
            {
                "fetched_at_utc": "2026-05-26T13:39:09+00:00",
                "event_id": "380231",
                "commence_time_utc": pd.NaT,
                "event_date_et": "2026-05-26",
                "home_team": "OKC",
                "away_team": "SAS",
                "bookmaker_key": "covers_consensus",
                "bookmaker_title": "Covers",
                "market_key": "player_points",
                "player_name_raw": "Test Player",
                "player_name_norm": "Test_Player",
                "line": 12.5,
                "over_price": -110.0,
                "under_price": pd.NA,
                "book_count": 3,
                "market_line_std": 0.0,
            }
        ]
    )

    wide = build_wide_from_covers_long(long_df)
    row = wide.iloc[0]

    assert row["Market_Provider"] == "covers_historical"
    assert row["Market_Book"] == "covers_consensus"
    assert row["Market_Price_Source"] == "covers_matchup_props"
    assert row["Market_Price_Source_Type"] == "ARCHIVED_ENTRY"
    assert row["Market_Snapshot_ID"] == "covers_historical:2026-05-26T13:39:09+00:00"
