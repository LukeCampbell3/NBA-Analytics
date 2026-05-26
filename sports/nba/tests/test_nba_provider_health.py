from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
SCRIPT_ROOT = PLAYER_PREDICTOR_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from check_nba_provider_health import _count_starts_at, build_provider_healthcheck_report  # noqa: E402


def test_count_starts_at_detects_event_start_times() -> None:
    events = [
        {"status": {"startsAt": "2026-05-27T00:30:00.000Z"}},
        {"status": {"startsAt": None}},
        {"status": {"startsAt": ""}},
    ]

    assert _count_starts_at(events) == 1


def test_build_provider_healthcheck_report_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "provider_healthcheck.json"
    report = build_provider_healthcheck_report(
        api_key_visible=True,
        request_success=False,
        events_returned=0,
        odds_rows_returned=0,
        starts_at_available_count=0,
        side_specific_price_count=0,
        books_observed=["draftkings", "fanduel"],
        failure_reason="missing key",
        output_path=output_path,
        fetched_at_utc="2026-05-26T15:00:00+00:00",
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["api_key_visible"] is True
    assert payload["request_success"] is False
    assert payload["events_returned"] == 0
    assert payload["books_observed"] == ["draftkings", "fanduel"]
    assert report == payload
