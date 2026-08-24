from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sports.nfl.scripts import settle_parlay_v2_calibration as settle


def _play(player_id: str, event_id: str, game_start_utc: str = "2026-09-13T17:00:00Z", **overrides) -> dict:
    play = {
        "player": f"Player {player_id}", "player_id": player_id, "position": "QB", "team": "AAA",
        "opponent": "BBB", "event_id": event_id, "game_start_utc": game_start_utc,
        "market": "passing", "target": "passing", "direction": "OVER", "line": 249.5,
        "projection": 260.0, "raw_model_probability": 0.6, "calibrated_hit_probability": 0.6,
        "model_hit_probability": 0.6, "no_vig_probability": 0.55, "probability_advantage": 0.05,
        "meta_policy_score": 0.9, "confidence_in_support": True,
        "selected_side_price": -120.0, "selected_sportsbook_key": "draftkings",
        "market_books": 3, "market_common_books": 2, "available_sportsbooks": ["draftkings", "fanduel", "betmgm"],
        "offers": {"draftkings": {"price": -120.0, "snapshot_time_utc": "2026-09-13T12:00:00Z"}},
        "market_source": "live", "price_confirmed": True, "snapshot_time_utc": "2026-09-13T12:00:00Z",
        "price_age_seconds": 100, "policy_version": "nfl_passing_loss_aware_meta_policy_v2",
        "candidate_authorized": False, "action_status": "review", "risk_flags": [],
    }
    play.update(overrides)
    return play


def _fixture_schedule() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2026, "week": 2, "commence_time_utc": pd.Timestamp("2026-09-13T17:00:00Z")},
        {"season": 2026, "week": 2, "commence_time_utc": pd.Timestamp("2026-09-13T20:00:00Z")},
        {"season": 2026, "week": 3, "commence_time_utc": pd.Timestamp("2026-09-20T17:00:00Z")},
    ])


# ---------------------------------------------------------------------
# resolve_season_week -- real, never a calendar-date guess
# ---------------------------------------------------------------------

def test_resolve_season_week_matches_real_kickoff_time() -> None:
    plays = [_play("p1", "evt1"), _play("p2", "evt2", game_start_utc="2026-09-13T20:00:00Z")]
    assert settle.resolve_season_week(plays, _fixture_schedule()) == (2026, 2)


def test_resolve_season_week_returns_none_when_no_match_within_tolerance() -> None:
    plays = [_play("p1", "evt1", game_start_utc="2026-11-01T17:00:00Z")]  # far from any fixture kickoff
    assert settle.resolve_season_week(plays, _fixture_schedule()) is None


def test_resolve_season_week_returns_none_when_plays_disagree() -> None:
    """A real data inconsistency (plays spanning two different real
    weeks in one snapshot) must never be silently resolved to one guess."""
    plays = [_play("p1", "evt1", game_start_utc="2026-09-13T17:00:00Z"), _play("p2", "evt2", game_start_utc="2026-09-20T17:00:00Z")]
    assert settle.resolve_season_week(plays, _fixture_schedule()) is None


def test_resolve_season_week_returns_none_with_no_timestamps() -> None:
    plays = [_play("p1", "evt1", game_start_utc=None)]
    assert settle.resolve_season_week(plays, _fixture_schedule()) is None


def test_resolve_season_week_returns_none_with_empty_schedule() -> None:
    plays = [_play("p1", "evt1")]
    assert settle.resolve_season_week(plays, pd.DataFrame(columns=["season", "week", "commence_time_utc"])) is None


# ---------------------------------------------------------------------
# settle_snapshot
# ---------------------------------------------------------------------

def test_settle_snapshot_admits_real_graded_plays(tmp_path, monkeypatch) -> None:
    plays = [_play("p1", "evt1"), _play("p2", "evt2", target="receiving", direction="OVER", line=59.5)]
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"plays": plays}))

    actuals = pd.DataFrame([
        {"player_id": "p1", "season": 2026, "week": 2, "passing_yards": 275.0, "rushing_yards": 0.0, "receiving_yards": 0.0},
        {"player_id": "p2", "season": 2026, "week": 2, "passing_yards": 0.0, "rushing_yards": 0.0, "receiving_yards": 40.0},
    ])
    monkeypatch.setattr(settle.ingest, "load_season_actuals", lambda season, cache_path=None: actuals)

    ledger_path = tmp_path / "ledger.jsonl"
    result = settle.settle_snapshot(snapshot_path, _fixture_schedule(), ledger_path=ledger_path)
    assert result["status"] == "settled"
    assert result["season"] == 2026
    assert result["week"] == 2
    assert result["admitted"] == 2  # p1 wins (275 > 249.5), p2 loses (40 < 59.5) -- both graded


def test_settle_snapshot_missing_file_is_no_plays() -> None:
    result = settle.settle_snapshot(Path("/nonexistent/snapshot.json"), _fixture_schedule())
    assert result["status"] == "unreadable"


def test_settle_snapshot_empty_plays_is_safe(tmp_path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"plays": []}))
    result = settle.settle_snapshot(snapshot_path, _fixture_schedule())
    assert result["status"] == "no_plays"


def test_settle_snapshot_unresolvable_season_week_is_safe(tmp_path) -> None:
    plays = [_play("p1", "evt1", game_start_utc="2026-11-01T17:00:00Z")]
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"plays": plays}))
    result = settle.settle_snapshot(snapshot_path, _fixture_schedule())
    assert result["status"] == "season_week_unresolved"


# ---------------------------------------------------------------------
# settle_previous_day
# ---------------------------------------------------------------------

def test_settle_previous_day_no_snapshot_dir_is_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settle, "PRODUCTION_SNAPSHOTS_ROOT", tmp_path / "snapshots")
    result = settle.settle_previous_day("2026-09-14")
    assert result["status"] == "no_snapshot_dir"
    assert result["target_date"] == "2026-09-13"


def test_settle_previous_day_settles_every_file_in_target_dir(tmp_path, monkeypatch) -> None:
    snapshots_root = tmp_path / "snapshots"
    target_dir = snapshots_root / "2026-09-13"
    target_dir.mkdir(parents=True)
    plays = [_play("p1", "evt1")]
    (target_dir / "run1.json").write_text(json.dumps({"plays": plays}))
    (target_dir / "run2.json").write_text(json.dumps({"plays": []}))

    monkeypatch.setattr(settle, "PRODUCTION_SNAPSHOTS_ROOT", snapshots_root)
    monkeypatch.setattr(settle, "load_schedule", lambda: _fixture_schedule())
    actuals = pd.DataFrame([{"player_id": "p1", "season": 2026, "week": 2, "passing_yards": 0.0, "rushing_yards": 0.0, "receiving_yards": 0.0}])
    monkeypatch.setattr(settle.ingest, "load_season_actuals", lambda season, cache_path=None: actuals)

    result = settle.settle_previous_day("2026-09-14", ledger_path=tmp_path / "ledger.jsonl")
    assert result["status"] == "checked"
    assert len(result["results"]) == 2
    statuses = {r["status"] for r in result["results"]}
    assert statuses == {"settled", "no_plays"}


# ---------------------------------------------------------------------
# main() -- must never break the daily workflow
# ---------------------------------------------------------------------

def test_main_never_raises_on_schedule_fetch_failure(monkeypatch, capsys) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("real network hiccup fetching the schedule")

    monkeypatch.setattr(settle, "settle_previous_day", _boom)
    monkeypatch.setattr("sys.argv", ["settle_parlay_v2_calibration.py", "--run-date", "2026-09-14"])
    assert settle.main() == 0
    assert "error" in capsys.readouterr().out
