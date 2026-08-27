from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import settle_published_predictions as settle  # noqa: E402


NOW = "2026-08-26T03:00:00+00:00"


def _final_feed(player_name: str, stat_group: str, stat_key: str, value: float, side: str = "away", final: bool = True) -> dict:
    return {
        "gameData": {"status": {"abstractGameState": "Final" if final else "In Progress", "detailedState": "Final" if final else "In Progress"}},
        "liveData": {
            "boxscore": {
                "teams": {
                    side: {"players": {"ID1": {"person": {"fullName": player_name}, "stats": {stat_group: {stat_key: value}}}}},
                    ("home" if side == "away" else "away"): {"players": {}},
                }
            }
        },
    }


def _play(**overrides) -> dict:
    row = {
        "player_display_name": "Carlos Cortes",
        "player": "Carlos Cortes",
        "target": "TB",
        "market_line": 1.5,
        "direction": "UNDER",
        "game_id": "824970",
        "player_type": "hitter",
    }
    row.update(overrides)
    return row


def test_grade_outcome_matches_backtest_prediction_method_semantics():
    assert settle.grade_outcome(2.0, 1.5, "OVER") == "won"
    assert settle.grade_outcome(1.0, 1.5, "OVER") == "lost"
    assert settle.grade_outcome(1.5, 1.5, "OVER") == "push"
    assert settle.grade_outcome(1.0, 1.5, "UNDER") == "won"
    assert settle.grade_outcome(2.0, 1.5, "UNDER") == "lost"
    assert settle.grade_outcome(1.5, 1.5, "UNDER") == "push"


def test_resolve_stat_spec_pitcher_strikeouts_default_role():
    assert settle.resolve_stat_spec("K", None) == ("pitching", "strikeOuts")
    assert settle.resolve_stat_spec("K", "hitter") == ("batting", "strikeOuts")


def test_resolve_stat_spec_unsupported_target_returns_none():
    assert settle.resolve_stat_spec("WALKS", "hitter") is None


def test_settle_row_wins_a_real_final_game():
    row = _play(target="TB", market_line=1.5, direction="UNDER")
    feed = _final_feed("Carlos Cortes", "batting", "totalBases", 1.0)

    touched = settle.settle_row(row, lambda game_id: feed, NOW)

    assert touched is True
    assert row["settlement_status"] == "won"
    assert row["settlement_actual_value"] == 1.0
    assert row["settlement_source"] == "mlb_statsapi_live_feed"
    assert row["settlement_checked_at"] == NOW
    assert "settlement_reason" not in row


def test_settle_row_loses_a_real_final_game():
    row = _play(target="TB", market_line=1.5, direction="UNDER")
    feed = _final_feed("Carlos Cortes", "batting", "totalBases", 3.0)

    settle.settle_row(row, lambda game_id: feed, NOW)

    assert row["settlement_status"] == "lost"


def test_settle_row_pushes_on_an_exact_line():
    row = _play(target="TB", market_line=2.0, direction="OVER")
    feed = _final_feed("Carlos Cortes", "batting", "totalBases", 2.0)

    settle.settle_row(row, lambda game_id: feed, NOW)

    assert row["settlement_status"] == "push"


def test_settle_row_stays_pending_while_game_is_not_final():
    row = _play()
    feed = _final_feed("Carlos Cortes", "batting", "totalBases", 1.0, final=False)

    touched = settle.settle_row(row, lambda game_id: feed, NOW)

    assert touched is True
    assert row["settlement_status"] == "pending"
    assert row["settlement_reason"] == "game_not_final"
    assert "settlement_actual_value" not in row


def test_settle_row_pending_when_player_not_in_either_boxscore():
    row = _play()
    feed = _final_feed("Someone Else", "batting", "totalBases", 1.0)

    settle.settle_row(row, lambda game_id: feed, NOW)

    assert row["settlement_status"] == "pending"
    assert row["settlement_reason"] == "player_not_found"


def test_settle_row_pending_on_a_real_fetch_error_never_raises():
    row = _play()

    def raising_fetch(game_id):
        raise ConnectionError("boom")

    touched = settle.settle_row(row, raising_fetch, NOW)

    assert touched is True
    assert row["settlement_status"] == "pending"
    assert row["settlement_reason"] == "fetch_error:ConnectionError"


def test_settle_row_never_refetches_an_already_resolved_row():
    row = _play(settlement_status="won", settlement_actual_value=1.0)
    calls = []

    def tracking_fetch(game_id):
        calls.append(game_id)
        raise AssertionError("must not fetch an already-resolved row")

    touched = settle.settle_row(row, tracking_fetch, NOW)

    assert touched is False
    assert calls == []
    assert row["settlement_status"] == "won"  # untouched -- append-only, never re-graded


def test_settle_row_leaves_a_row_missing_required_fields_untouched():
    row = {"player": "No Game Id Player", "target": "TB", "market_line": 1.5, "direction": "OVER"}

    touched = settle.settle_row(row, lambda game_id: {}, NOW)

    assert touched is False
    assert "settlement_status" not in row


def test_settle_row_never_modifies_prediction_time_fields():
    """append outcome fields only, never modify prediction-time fields --
    same principle settle_mlb_production_shadow.py already states."""
    row = _play(target="TB", market_line=1.5, direction="UNDER", prediction=0.83, market_source="real")
    before = {k: v for k, v in row.items() if not k.startswith("settlement_")}
    feed = _final_feed("Carlos Cortes", "batting", "totalBases", 1.0)

    settle.settle_row(row, lambda game_id: feed, NOW)

    after = {k: v for k, v in row.items() if not k.startswith("settlement_")}
    assert before == after


def test_iter_settleable_rows_covers_plays_ticket_legs_and_parlay_legs():
    payload = {
        "plays": [_play()],
        "daily_parlay": {"selected_ticket": {"legs": [_play(player="Leg Player")]}},
        "parlays": {
            "selected_parlay": {
                "leg_1": {"player": "Pete Alonso", "target": "R", "line": 0.5, "side": "OVER", "game_id": "823016"},
                "leg_2": {"player": "Pete Crow-Armstrong", "target": "TB", "line": 1.5, "side": "OVER", "game_id": "825042"},
            }
        },
    }
    rows = settle.iter_settleable_rows(payload)
    assert len(rows) == 4


def test_iter_settleable_rows_skips_unrendered_ticket_ladder():
    payload = {"daily_parlay": {"ticket_ladder": [{"legs": [_play()]}]}}
    assert settle.iter_settleable_rows(payload) == []


def test_settle_payload_counts_outcomes_across_all_row_shapes():
    payload = {
        "plays": [_play(target="TB", market_line=1.5, direction="UNDER")],
        "parlays": {
            "selected_parlay": {
                "leg_1": {"player": "Pete Alonso", "target": "R", "line": 0.5, "side": "OVER", "game_id": "823016"},
            }
        },
    }
    feeds = {
        "824970": _final_feed("Carlos Cortes", "batting", "totalBases", 1.0),
        "823016": _final_feed("Pete Alonso", "batting", "runs", 2.0),
    }

    counts = settle.settle_payload(payload, lambda game_id: feeds[game_id], NOW)

    assert counts == {"won": 2, "lost": 0, "push": 0, "pending": 0, "touched": 2}


def test_settle_file_round_trips_through_disk_and_is_a_no_op_once_resolved(tmp_path):
    path = tmp_path / "2026-08-11.json"
    path.write_text(json.dumps({"run_date": "2026-08-11", "plays": [_play(target="TB", market_line=1.5, direction="UNDER")]}), encoding="utf-8")
    feed = _final_feed("Carlos Cortes", "batting", "totalBases", 1.0)

    first = settle.settle_file(path, lambda game_id: feed, NOW)
    assert first == {"won": 1, "lost": 0, "push": 0, "pending": 0, "touched": 1}

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["plays"][0]["settlement_status"] == "won"

    def must_not_fetch(game_id):
        raise AssertionError("resolved row must not be refetched on the next run")

    second = settle.settle_file(path, must_not_fetch, NOW)
    assert second == {"won": 0, "lost": 0, "push": 0, "pending": 0, "touched": 0}


def test_run_settles_daily_predictions_and_every_history_file(tmp_path):
    data_dir = tmp_path / "data"
    history_dir = data_dir / "history"
    history_dir.mkdir(parents=True)

    (data_dir / "daily_predictions.json").write_text(
        json.dumps({"run_date": "2026-08-26", "plays": [_play(target="TB", market_line=1.5, direction="UNDER", game_id="1")]}),
        encoding="utf-8",
    )
    (history_dir / "2026-08-10.json").write_text(
        json.dumps({"run_date": "2026-08-10", "plays": [_play(target="TB", market_line=1.5, direction="UNDER", game_id="2")]}),
        encoding="utf-8",
    )
    (history_dir / "index.json").write_text(json.dumps({"dates": ["2026-08-10"]}), encoding="utf-8")

    feeds = {
        "1": _final_feed("Carlos Cortes", "batting", "totalBases", 1.0),
        "2": _final_feed("Carlos Cortes", "batting", "totalBases", 1.0),
    }

    def fake_fetcher(request_timeout=20.0, sleep_between_requests=0.0):
        return lambda game_id: feeds[game_id]

    original = settle.make_live_feed_fetcher
    settle.make_live_feed_fetcher = fake_fetcher
    try:
        report = settle.run(data_dir=data_dir, report_path=tmp_path / "report.json")
    finally:
        settle.make_live_feed_fetcher = original

    assert report["total"] == {"won": 2, "lost": 0, "push": 0, "pending": 0, "touched": 2}
    assert set(report["files_touched"]) == {"daily_predictions.json", "history/2026-08-10.json"}
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["total"]["won"] == 2


def test_run_with_only_date_settles_a_single_history_file_and_skips_daily(tmp_path):
    data_dir = tmp_path / "data"
    history_dir = data_dir / "history"
    history_dir.mkdir(parents=True)

    def must_not_fetch(game_id):
        raise AssertionError("daily_predictions.json must be skipped when --only-date is set")

    (data_dir / "daily_predictions.json").write_text(
        json.dumps({"run_date": "2026-08-26", "plays": [_play(game_id="999")]}), encoding="utf-8"
    )
    (history_dir / "2026-08-10.json").write_text(
        json.dumps({"run_date": "2026-08-10", "plays": [_play(target="TB", market_line=1.5, direction="UNDER", game_id="2")]}),
        encoding="utf-8",
    )

    feed = _final_feed("Carlos Cortes", "batting", "totalBases", 1.0)

    def fake_fetcher(request_timeout=20.0, sleep_between_requests=0.0):
        return lambda game_id: feed if game_id == "2" else (_ for _ in ()).throw(must_not_fetch(game_id))

    original = settle.make_live_feed_fetcher
    settle.make_live_feed_fetcher = fake_fetcher
    try:
        report = settle.run(data_dir=data_dir, only_date="2026-08-10", report_path=None)
    finally:
        settle.make_live_feed_fetcher = original

    assert report["total"]["won"] == 1
    assert list(report["files_touched"]) == ["history/2026-08-10.json"]
