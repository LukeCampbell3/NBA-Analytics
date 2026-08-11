from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sports.nfl.predictions.daily_policy import (
    build_shadow_parlay,
    score_market_offers,
    select_live_board,
)
from sports.nfl.predictions.live_market import (
    flatten_event_odds,
    load_fixture_slate,
    write_complete_slate,
)
from sports.nfl.predictions.live_scoring import add_market_placeholders


def event_payload() -> dict:
    return {
        "id": "game-1",
        "commence_time": "2026-09-10T00:20:00Z",
        "home_team": "Philadelphia Eagles",
        "away_team": "Dallas Cowboys",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "last_update": "2026-09-09T14:00:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Quarterback A", "point": 249.5, "price": -105},
                            {"name": "Under", "description": "Quarterback A", "point": 249.5, "price": -115},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "last_update": "2026-09-09T14:01:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Quarterback A", "point": 249.5, "price": 100},
                            {"name": "Under", "description": "Quarterback A", "point": 249.5, "price": -120},
                        ],
                    }
                ],
            },
        ],
    }


def test_live_market_retains_complete_two_sided_rows() -> None:
    rows = flatten_event_odds(event_payload(), fetched_at_utc="2026-09-09T14:02:00Z")

    assert len(rows) == 2
    assert {row["bookmaker"] for row in rows} == {"fanduel", "draftkings"}
    assert all(row["target"] == "passing" for row in rows)
    assert all(row["over_price"] is not None and row["under_price"] is not None for row in rows)


def test_live_board_requires_fresh_multibook_executable_market() -> None:
    rows = pd.DataFrame(flatten_event_odds(event_payload(), fetched_at_utc="2026-09-09T14:02:00Z"))
    rows["player_key"] = "quarterback a"
    rows["player_id"] = "qb-a"
    rows["position"] = "QB"
    rows["recent_team"] = "PHI"
    rows["opponent_team"] = "DAL"
    rows["prediction"] = 275.0
    scored = score_market_offers(
        rows,
        np.array([0.64, 0.64]),
        now_utc=datetime(2026, 9, 9, 15, 0, tzinfo=timezone.utc),
    )

    plays, audit = select_live_board(scored)

    assert audit["selected_candidates"] == 1
    assert plays[0]["direction"] == "OVER"
    assert plays[0]["selected_sportsbook_key"] == "draftkings"
    assert plays[0]["selected_side_price"] == 100.0
    assert plays[0]["market_books"] == 2
    assert plays[0]["candidate_authorized"] is False


def test_parlay_constructor_is_always_withheld_after_failed_holdout() -> None:
    first = {
        "event_id": "a",
        "player_id": "p1",
        "model_hit_probability": 0.64,
        "offers": {"draftkings": {"price": -110}},
    }
    second = {
        "event_id": "b",
        "player_id": "p2",
        "model_hit_probability": 0.63,
        "offers": {"draftkings": {"price": -105}},
    }

    parlay = build_shadow_parlay([first, second])

    assert parlay["available"] is True
    assert parlay["status"] == "withheld"
    assert parlay["validation_status"] == "failed_locked_holdout"
    assert parlay["selected_ticket"]["candidate_authorized"] is False


def test_current_roster_allows_traded_veteran_identity() -> None:
    stats = pd.DataFrame(
        [
            {
                "player_id": "veteran-1",
                "player_display_name": "Veteran QB",
                "position": "QB",
                "recent_team": "OLD",
                "opponent_team": "X",
                "season": 2025,
                "week": 18,
                "season_type": "REG",
                "attempts": 30.0,
            }
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "player": "Veteran QB",
                "season": 2026,
                "week": 1,
                "event_id": "new-game",
                "home_abbr": "NEW",
                "away_abbr": "OPP",
            }
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "gsis_id": "veteran-1",
                "full_name": "Veteran QB",
                "team": "NEW",
                "season": 2026,
                "week": 1,
            }
        ]
    )

    history, accepted, audit = add_market_placeholders(
        stats, markets, current_roster=roster
    )

    assert len(accepted) == 1
    assert audit["team_mismatch_players"] == 0
    placeholder = history.loc[history["season"].eq(2026)].iloc[0]
    assert placeholder["recent_team"] == "NEW"
    assert placeholder["opponent_team"] == "OPP"


def test_complete_snapshot_can_be_replayed_without_refetch(tmp_path) -> None:
    rows = flatten_event_odds(
        event_payload(), fetched_at_utc="2026-09-09T14:02:00Z"
    )
    snapshot = tmp_path / "snapshot.json"
    write_complete_slate(snapshot, rows, {"provider": "the_odds_api"})

    replayed, audit = load_fixture_slate(snapshot)

    assert replayed == rows
    assert audit["provider"] == "the_odds_api"
    assert audit["replayed_from_snapshot"] is True
