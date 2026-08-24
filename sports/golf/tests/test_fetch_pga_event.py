from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLF_SCRIPTS_ROOT = REPO_ROOT / "sports" / "golf" / "scripts"
sys.path.insert(0, str(GOLF_SCRIPTS_ROOT))

import fetch_pga_event as fetcher  # noqa: E402


def _event(event_id: str, start: str, end: str, name: str = "Test Event") -> fetcher.ScheduledEvent:
    return fetcher.ScheduledEvent(event_id=event_id, name=name, start_date=start, end_date=end)


def test_resolve_current_or_next_event_prefers_event_underway() -> None:
    calendar = [
        _event("1", "2026-08-13T07:00Z", "2026-08-16T07:00Z"),
        _event("2", "2026-08-20T07:00Z", "2026-08-23T07:00Z"),
        _event("3", "2026-08-27T07:00Z", "2026-08-30T07:00Z"),
    ]
    as_of = datetime(2026, 8, 22, tzinfo=timezone.utc)
    resolved = fetcher.resolve_current_or_next_event(calendar, as_of=as_of)
    assert resolved is not None
    assert resolved.event_id == "2"


def test_resolve_current_or_next_event_falls_back_to_next_future_event() -> None:
    """No real event underway right now (a genuine, expected state most of
    the week) -- must resolve the real next scheduled one, never guess."""
    calendar = [
        _event("1", "2026-08-13T07:00Z", "2026-08-16T07:00Z"),
        _event("2", "2026-08-27T07:00Z", "2026-08-30T07:00Z"),
    ]
    as_of = datetime(2026, 8, 24, tzinfo=timezone.utc)
    resolved = fetcher.resolve_current_or_next_event(calendar, as_of=as_of)
    assert resolved is not None
    assert resolved.event_id == "2"


def test_resolve_current_or_next_event_returns_none_past_the_real_calendar() -> None:
    calendar = [_event("1", "2026-08-13T07:00Z", "2026-08-16T07:00Z")]
    as_of = datetime(2026, 12, 31, tzinfo=timezone.utc)
    assert fetcher.resolve_current_or_next_event(calendar, as_of=as_of) is None


def test_recent_completed_events_returns_most_recent_first_within_limit() -> None:
    calendar = [
        _event("1", "2026-07-02T07:00Z", "2026-07-05T07:00Z"),
        _event("2", "2026-07-09T07:00Z", "2026-07-12T07:00Z"),
        _event("3", "2026-07-16T07:00Z", "2026-07-19T07:00Z"),
        _event("4", "2026-08-27T07:00Z", "2026-08-30T07:00Z"),  # not yet completed
    ]
    as_of = datetime(2026, 8, 24, tzinfo=timezone.utc)
    recent = fetcher.recent_completed_events(calendar, as_of=as_of, limit=2)
    assert [event.event_id for event in recent] == ["3", "2"]


def _competitor(strokes_by_round: dict[int, tuple[float, str]], *, player_id="1", name="Test Player") -> dict:
    return {
        "athlete": {"id": player_id, "displayName": name, "headshot": {"href": "http://example.com/h.png"}, "flag": {"alt": "USA"}},
        "status": {
            "position": {"displayName": "T1", "isTie": True},
            "detail": "-4(F)",
            "type": {"state": "post", "completed": True, "description": "Finish"},
        },
        "score": {"value": 276.0, "displayValue": "-4"},
        "linescores": [
            {"period": round_num, "value": value, "displayValue": display}
            for round_num, (value, display) in strokes_by_round.items()
        ],
    }


def test_extract_player_rounds_skips_espn_unplayed_round_placeholder() -> None:
    """Real bug found this session: ESPN encodes an unplayed round (after
    a withdrawal, or not yet reached) as value=0.0/displayValue="-", not
    null. A literal 0.0 must never be treated as a real score of 71 under
    par."""
    competitor = _competitor({1: (82.0, "+12"), 2: (0.0, "-"), 3: (0.0, "-")}, name="Withdrawn Player")
    result = fetcher.extract_player_rounds(competitor)
    assert [r["round"] for r in result["rounds"]] == [1]
    assert result["rounds"][0]["strokes"] == 82.0


def test_extract_player_rounds_keeps_real_played_rounds() -> None:
    competitor = _competitor({1: (67.0, "-3"), 2: (69.0, "-1"), 3: (71.0, "+1"), 4: (71.0, "+1")})
    result = fetcher.extract_player_rounds(competitor)
    assert len(result["rounds"]) == 4
    assert result["rounds"][0]["strokes"] == 67.0
    assert result["player_name"] == "Test Player"
    assert result["completed"] is True


def test_extract_player_rounds_rejects_implausible_low_strokes_defensively() -> None:
    """Defense in depth beyond the "-" marker check: any strokes value
    below the lowest 18-hole score ever recorded on the PGA Tour (58)
    cannot be a real played round, whatever displayValue says."""
    competitor = _competitor({1: (12.0, "-59")})
    result = fetcher.extract_player_rounds(competitor)
    assert result["rounds"] == []


def test_persist_event_snapshot_writes_timestamped_and_latest_copies(tmp_path: Path) -> None:
    payload = {"event_id": "999", "players": []}
    written = fetcher.persist_event_snapshot("999", payload, raw_root=tmp_path)
    assert written.exists()
    latest = tmp_path / "999" / "latest_leaderboard.json"
    assert latest.exists()
    assert latest.read_text(encoding="utf-8") == written.read_text(encoding="utf-8")
