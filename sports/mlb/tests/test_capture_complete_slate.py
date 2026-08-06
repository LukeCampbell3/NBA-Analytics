from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sports.mlb.governance.capture_complete_slate import capture_snapshot


GOVERNANCE_ROOT = Path(__file__).resolve().parents[1] / "governance"


def _provider_rows() -> pd.DataFrame:
    base = {
        "source": "test",
        "sportsbook": "draftkings",
        "event_id": "event-1",
        "game_start_utc": "2026-08-06T23:00:00Z",
        "player_name": "Test Player",
        "market_type": "batter_hits",
        "line": 0.5,
        "observed_at_utc": "2026-08-06T15:00:00Z",
        "parser_version": "test-parser-v1",
        "validation_status": "VALID",
    }
    return pd.DataFrame(
        [
            {**base, "side": "over", "price_american": -150, "price_decimal": 1.666667, "raw_record_hash": "a" * 64},
            {**base, "side": "under", "price_american": 120, "price_decimal": 2.2, "raw_record_hash": "b" * 64},
        ]
    )


def _pool_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Player": "Test Player",
                "Player_ID": "test_player",
                "Target": "H",
                "Prediction": 1.1,
                "Last_History_Date": "2026-08-05",
                "Model_Selected": "et",
                "Matchup_Network_Version": "network-v1",
                "Game_Status_Detail": "Scheduled",
            }
        ]
    )


def test_capture_writes_immutable_full_universe(tmp_path: Path) -> None:
    provider_path = tmp_path / "provider.csv"
    pool_path = tmp_path / "pool.csv"
    run_dir = tmp_path / "run"
    _provider_rows().to_csv(provider_path, index=False)
    _pool_rows().to_csv(pool_path, index=False)
    args = SimpleNamespace(
        provider_csv=provider_path,
        pool_csv=pool_path,
        run_dir=run_dir,
        run_date=date(2026, 8, 6),
        policy_registry=GOVERNANCE_ROOT / "policies" / "mlb_policy_family_v1.json",
        evidence_inventory=GOVERNANCE_ROOT / "evidence_inventory.json",
    )

    first = capture_snapshot(args)
    second = capture_snapshot(args)

    assert first["manifest"]["snapshot_id"] == second["manifest"]["snapshot_id"]
    assert first["manifest"]["candidate_universe_rows"] == 2
    assert first["manifest"]["eligible_input_rows"] == 2
    assert first["manifest"]["capture_label"] == "FULL_SLATE_SNAPSHOT"
    assert first["governance_status"]["candidate_authorization_enabled"] is False

    manifest_path = Path(first["governance_status"]["snapshot_manifest"])
    universe_path = manifest_path.parent / "candidate_universe.csv.gz"
    with gzip.open(universe_path, "rt", encoding="utf-8") as handle:
        universe = pd.read_csv(handle)
    assert universe["side"].tolist() == ["OVER", "UNDER"]
    assert universe["model_score"].round(1).tolist() == [0.6, -0.6]
    assert set(universe["settlement"]) == {"PENDING"}

    status = json.loads((run_dir / "governance" / "governance_status.json").read_text(encoding="utf-8"))
    assert status["publication_mode"] == "SHADOW_RESEARCH_ONLY"
    assert status["certificate_status"] == "NO_ACTIVE_PROSPECTIVE_CERTIFICATE"
