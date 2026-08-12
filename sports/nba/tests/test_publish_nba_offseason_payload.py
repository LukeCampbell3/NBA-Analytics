from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "sports/nba/predictions/Player-Predictor/scripts/publish_nba_offseason_payload.py"
)
SPEC = importlib.util.spec_from_file_location("publish_nba_offseason_payload", SCRIPT)
assert SPEC and SPEC.loader
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


def test_offseason_payload_is_explicitly_withheld_and_calibrated() -> None:
    calibration = {
        "status": "passed",
        "method": "segment_monotonic_safety",
        "evidence_scope": "FULL_CANDIDATE_POOL_REPLAY",
        "locked_metrics": {"rows": 1067},
        "historical_support": {},
    }

    payload = PUBLISHER.build_payload("2026-08-11", calibration)

    assert payload["run_date"] == "2026-08-11"
    assert payload["publication_status"] == "suppressed"
    assert payload["publication_gate"]["blockers"] == ["offseason_no_slate"]
    assert payload["confidence_calibration"] == calibration
    assert payload["plays"] == []
