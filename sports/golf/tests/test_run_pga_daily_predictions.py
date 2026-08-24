from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for rel in ("scripts", "predictions", "parlay_v2"):
    sys.path.insert(0, str(REPO_ROOT / "sports" / "golf" / rel))

import fetch_pga_event as fetcher  # noqa: E402
import run_pga_daily_predictions as runner  # noqa: E402


def test_build_daily_payload_reports_no_event_honestly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fetcher, "fetch_season_calendar", lambda: [])
    payload = runner.build_daily_payload(raw_root=tmp_path / "raw", calibration_ledger=tmp_path / "ledger.jsonl")
    assert payload["status"] == "no_event_in_calendar"
    assert payload["top_10"] == []
    assert payload["candidates"] == []


def test_build_daily_payload_reports_field_not_posted_honestly(monkeypatch, tmp_path) -> None:
    fake_event = fetcher.ScheduledEvent("999", "Fake Event", "2026-09-01T00:00Z", "2026-09-04T00:00Z")
    monkeypatch.setattr(fetcher, "fetch_season_calendar", lambda: [fake_event])
    monkeypatch.setattr(fetcher, "resolve_current_or_next_event", lambda calendar, as_of=None: fake_event)
    monkeypatch.setattr(fetcher, "fetch_event_leaderboard", lambda event_id, timeout_seconds=20.0: {"event_id": event_id, "status": "STATUS_SCHEDULED", "completed": False, "competitors": []})
    payload = runner.build_daily_payload(raw_root=tmp_path / "raw", calibration_ledger=tmp_path / "ledger.jsonl")
    assert payload["status"] == "field_not_posted"
    assert payload["field_size"] == 0


def test_has_real_cut_detects_full_field_vs_small_playoff_field() -> None:
    assert runner.has_real_cut(field_size=156) is True  # a real, standard full PGA Tour field
    assert runner.has_real_cut(field_size=30) is False  # TOUR Championship-sized, real no-cut event


def test_write_web_payload_writes_valid_json(tmp_path) -> None:
    out_path = runner.write_web_payload({"status": "ok", "top_10": []}, web_data_root=tmp_path)
    assert out_path.exists()
    import json
    assert json.loads(out_path.read_text())["status"] == "ok"
