from pathlib import Path

from sports.nfl.scripts.update_pick_publication_history import empty_ledger, observe, published_picks


def play(player: str, event: str, kickoff: str, *, line: float = 250.5) -> dict:
    return {
        "player": player,
        "player_id": player.lower().replace(" ", "-"),
        "event_id": event,
        "market": "player_pass_yds",
        "direction": "OVER",
        "line": line,
        "selected_sportsbook_key": "fanduel",
        "selected_side_price": -110,
        "game_start_utc": kickoff,
    }


def payload(at: str, plays: list[dict]) -> dict:
    return {"run_date": "2026-09-13", "generated_at_utc": at, "plays": plays}


def test_removed_pregame_pick_is_retained_with_reason():
    kickoff = "2026-09-13T17:00:00Z"
    ledger = observe(empty_ledger(), payload("2026-09-13T14:00:00Z", [play("A QB", "g1", kickoff)]), {}, source="one")
    ledger = observe(ledger, payload("2026-09-13T15:00:00Z", []), {}, source="two")

    assert ledger["summary"]["REMOVED_BEFORE_KICKOFF"] == 1
    assert ledger["picks"][0]["removal_reason"] == "NO_LONGER_SELECTED_ON_REFRESH"
    assert [event["event"] for event in ledger["picks"][0]["events"]] == ["PUBLISHED", "REMOVED_BEFORE_KICKOFF"]


def test_pick_present_until_after_kickoff_is_locked_not_erased():
    kickoff = "2026-09-13T17:00:00Z"
    ledger = observe(empty_ledger(), payload("2026-09-13T16:45:00Z", [play("A QB", "g1", kickoff)]), {}, source="one")
    ledger = observe(ledger, payload("2026-09-13T18:00:00Z", []), {}, source="two")

    assert ledger["summary"]["LOCKED_AFTER_KICKOFF"] == 1
    assert ledger["picks"][0]["removal_reason"] == "GAME_STARTED"


def test_republished_pick_preserves_full_lifecycle():
    kickoff = "2026-09-13T17:00:00Z"
    row = play("A QB", "g1", kickoff)
    ledger = observe(empty_ledger(), payload("2026-09-13T14:00:00Z", [row]), {}, source="one")
    ledger = observe(ledger, payload("2026-09-13T15:00:00Z", []), {}, source="two")
    ledger = observe(ledger, payload("2026-09-13T16:00:00Z", [row]), {}, source="three")

    record = ledger["picks"][0]
    assert record["status"] == "ACTIVE"
    assert record["appearances"] == 2
    assert [event["event"] for event in record["events"]][-1] == "REPUBLISHED"


def test_week_market_singles_and_parlay_legs_are_tracked():
    kickoff = "2026-09-13T17:00:00Z"
    market = {
        "best_available_singles": [{**play("A QB", "g1", kickoff), "side": "OVER"}],
        "pools": {"passer_parlay": [{**play("B QB", "g2", kickoff), "side": "OVER"}]},
    }
    rows = published_picks({"plays": []}, market)
    assert {row["product"] for row in rows} == {"qualified_single", "passer_parlay"}


def test_workflow_only_publishes_kickoff_day_slates():
    workflow = (Path(__file__).resolve().parents[3] / ".github/workflows/nfl-predictions.yml").read_text()
    assert "  push:" not in workflow
    assert workflow.count("--window-days 1") == 2
    assert "update_pick_publication_history.py" in workflow
