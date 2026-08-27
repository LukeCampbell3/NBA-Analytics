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


def _score_feed(home_runs: int, away_runs: int, innings: list | None = None, final: bool = True) -> dict:
    status = "Final" if final else "In Progress"
    return {
        "gameData": {"status": {"abstractGameState": status, "detailedState": status}},
        "liveData": {
            "linescore": {
                "teams": {"home": {"runs": home_runs}, "away": {"runs": away_runs}},
                "innings": innings if innings is not None else [],
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


# ---------------------------------------------------------------------------
# Pitchers-Only (pitcher_parlay_predictions.json) -- implied K/pitcher, no
# explicit target field on the row itself.
# ---------------------------------------------------------------------------

def _pitcher_k_leg(**overrides) -> dict:
    row = {"pitcher_name": "Joey Cantillo", "line": 6.5, "side": "under", "game_id": "823988", "team": "CLE", "opponent": "LAA"}
    row.update(overrides)
    return row


def test_settle_pitcher_k_row_settles_the_implied_strikeouts_market():
    row = _pitcher_k_leg(line=6.5, side="under")
    feed = _final_feed("Joey Cantillo", "pitching", "strikeOuts", 5.0)

    touched = settle.settle_pitcher_k_row(row, lambda game_id: feed, NOW)

    assert touched is True
    assert row["settlement_status"] == "won"
    assert row["settlement_actual_value"] == 5.0


def test_settle_pitcher_k_row_never_refetches_an_already_resolved_leg():
    row = _pitcher_k_leg(settlement_status="lost")

    def must_not_fetch(game_id):
        raise AssertionError("resolved leg must not be refetched")

    assert settle.settle_pitcher_k_row(row, must_not_fetch, NOW) is False


def test_iter_pitcher_k_rows_reads_only_the_selected_pair():
    payload = {
        "legs": [_pitcher_k_leg(), _pitcher_k_leg(pitcher_name="Someone Else")],  # not rendered, deliberately excluded
        "parlay": {"leg_a": _pitcher_k_leg(), "leg_b": _pitcher_k_leg(pitcher_name="Nick Lodolo", game_id="823180")},
    }
    rows = settle.iter_pitcher_k_rows(payload)
    assert len(rows) == 2
    assert {r["pitcher_name"] for r in rows} == {"Joey Cantillo", "Nick Lodolo"}


def test_iter_pitcher_k_rows_empty_without_a_selected_parlay():
    assert settle.iter_pitcher_k_rows({"legs": [_pitcher_k_leg()]}) == []


# ---------------------------------------------------------------------------
# Same-Game (same_game_predictions.json) -- team moneyline / totals, not a
# player stat at all.
# ---------------------------------------------------------------------------

def _team_leg(**overrides) -> dict:
    # Deliberately no game_id here -- in the real payload it lives one
    # level up, on the parent combo_candidates[] entry (see
    # settle_team_market_row's own docstring for why).
    row = {"market": "moneyline", "side": "away", "price_american": 106}
    row.update(overrides)
    return row


def test_resolve_team_market_outcome_moneyline_winner_and_loser():
    feed = _score_feed(home_runs=4, away_runs=7)
    won_status, won_value, _ = settle.resolve_team_market_outcome(_team_leg(side="away"), feed)
    lost_status, lost_value, _ = settle.resolve_team_market_outcome(_team_leg(side="home"), feed)
    assert (won_status, won_value) == ("won", None)
    assert (lost_status, lost_value) == ("lost", None)


def test_resolve_team_market_outcome_moneyline_tie_is_unsettleable():
    feed = _score_feed(home_runs=3, away_runs=3)
    status, _, reason = settle.resolve_team_market_outcome(_team_leg(side="home"), feed)
    assert status is None
    assert reason == "tie_unsettleable"


def test_resolve_team_market_outcome_game_total_over_and_under():
    feed = _score_feed(home_runs=4, away_runs=5)  # total = 9
    over_status, over_value, _ = settle.resolve_team_market_outcome(_team_leg(market="game_total", side="over", line=7.5), feed)
    under_status, _, _ = settle.resolve_team_market_outcome(_team_leg(market="game_total", side="under", line=7.5), feed)
    assert over_status == "won"
    assert over_value == 9.0
    assert under_status == "lost"


def test_resolve_team_market_outcome_first_5_innings_total_sums_only_first_5():
    innings = [{"num": i, "home": {"runs": 1}, "away": {"runs": 0}} for i in range(1, 10)]  # 9 innings, 1 run each half for home
    feed = _score_feed(home_runs=9, away_runs=0, innings=innings)
    status, value, _ = settle.resolve_team_market_outcome(_team_leg(market="first_5_innings_total", side="over", line=4.5), feed)
    assert status == "won"
    assert value == 5.0  # only innings 1-5, not the full 9-inning total


def test_resolve_team_market_outcome_first_5_innings_pending_when_game_ended_early():
    feed = _score_feed(home_runs=3, away_runs=2, innings=[{"num": i, "home": {"runs": 1}, "away": {"runs": 0}} for i in range(1, 4)])
    status, _, reason = settle.resolve_team_market_outcome(_team_leg(market="first_5_innings_total", side="over", line=2.5), feed)
    assert status is None
    assert reason == "incomplete_f5_data"


def test_settle_team_market_row_writes_won_for_a_real_final_game():
    row = _team_leg(market="moneyline", side="away")
    feed = _score_feed(home_runs=1, away_runs=6)

    touched = settle.settle_team_market_row(row, "824234", lambda game_id: feed, NOW)

    assert touched is True
    assert row["settlement_status"] == "won"
    assert row["settlement_source"] == "mlb_statsapi_live_feed"
    assert "settlement_actual_value" not in row  # moneyline has no single stat value
    assert "game_id" not in row  # never written onto the leg -- it's the combo's field, not the leg's


def test_settle_team_market_row_stays_pending_while_not_final():
    row = _team_leg()
    feed = _score_feed(home_runs=1, away_runs=6, final=False)

    settle.settle_team_market_row(row, "824234", lambda game_id: feed, NOW)

    assert row["settlement_status"] == "pending"
    assert row["settlement_reason"] == "game_not_final"


def test_settle_team_market_row_returns_false_without_a_game_id():
    row = _team_leg()

    def must_not_fetch(game_id):
        raise AssertionError("must not fetch without a real game_id")

    assert settle.settle_team_market_row(row, None, must_not_fetch, NOW) is False
    assert "settlement_status" not in row


def test_settle_team_market_row_never_refetches_an_already_resolved_leg():
    row = _team_leg(settlement_status="won")

    def must_not_fetch(game_id):
        raise AssertionError("resolved leg must not be refetched")

    assert settle.settle_team_market_row(row, "824234", must_not_fetch, NOW) is False


def test_settle_same_game_payload_reads_game_id_from_the_parent_combo():
    """Real production schema: game_id lives on the combo_candidates[]
    entry, not on leg_a/leg_b themselves -- this is the bug the first
    version of this settlement path had (every leg silently untouched)."""
    payload = {
        "games": [
            {
                "combo_candidates": [
                    {"game_id": "1", "leg_a": _team_leg(side="away"), "leg_b": _team_leg(market="game_total", side="over", line=8.5)},
                    {"game_id": "1", "leg_a": _team_leg(side="home")},
                ]
            },
            {"combo_candidates": [{"game_id": "2", "leg_a": _team_leg(side="away")}]},
        ]
    }
    feeds = {"1": _score_feed(home_runs=2, away_runs=9), "2": _score_feed(home_runs=5, away_runs=1)}

    counts = settle.settle_same_game_payload(payload, lambda game_id: feeds[game_id], NOW)

    assert counts["touched"] == 4
    assert payload["games"][0]["combo_candidates"][0]["leg_a"]["settlement_status"] == "won"  # away, 9 > 2
    assert payload["games"][1]["combo_candidates"][0]["leg_a"]["settlement_status"] == "lost"  # away, 1 < 5


def test_settle_same_game_file_round_trips_through_disk(tmp_path):
    path = tmp_path / "same_game_predictions.json"
    path.write_text(
        json.dumps({"games": [{"combo_candidates": [{"game_id": "824234", "leg_a": _team_leg(side="away")}]}]}), encoding="utf-8"
    )
    feed = _score_feed(home_runs=2, away_runs=9)

    counts = settle.settle_same_game_file(path, lambda game_id: feed, NOW)

    assert counts == {"won": 1, "lost": 0, "push": 0, "pending": 0, "touched": 1}
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["games"][0]["combo_candidates"][0]["leg_a"]["settlement_status"] == "won"


def test_run_settles_same_game_and_pitcher_parlay_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    (data_dir / "same_game_predictions.json").write_text(
        json.dumps({"games": [{"combo_candidates": [{"game_id": "1", "leg_a": _team_leg(side="away")}]}]}), encoding="utf-8"
    )
    (data_dir / "pitcher_parlay_predictions.json").write_text(
        json.dumps({"parlay": {"leg_a": _pitcher_k_leg(game_id="2", line=6.5, side="under")}}), encoding="utf-8"
    )

    feeds = {
        "1": _score_feed(home_runs=1, away_runs=5),
        "2": _final_feed("Joey Cantillo", "pitching", "strikeOuts", 5.0),
    }

    def fake_fetcher(request_timeout=20.0, sleep_between_requests=0.0):
        return lambda game_id: feeds[game_id]

    original = settle.make_live_feed_fetcher
    settle.make_live_feed_fetcher = fake_fetcher
    try:
        report = settle.run(data_dir=data_dir, report_path=None)
    finally:
        settle.make_live_feed_fetcher = original

    assert report["total"] == {"won": 2, "lost": 0, "push": 0, "pending": 0, "touched": 2}
    assert set(report["files_touched"]) == {"same_game_predictions.json", "pitcher_parlay_predictions.json"}
