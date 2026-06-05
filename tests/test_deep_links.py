from __future__ import annotations

from sports.shared.odds.deep_links import (
    build_parlay_sportsbook_options,
    choose_bet_link,
    enrich_parlay_payload_with_sportsbooks,
    match_play_to_catalog,
)


def test_choose_bet_link_prefers_deepest_available_level() -> None:
    outcome = {"link": "https://example.com/outcome"}
    market = {"link": "https://example.com/market"}
    bookmaker = {"key": "fanduel", "link": "https://example.com/event"}

    link, quality = choose_bet_link(outcome, market, bookmaker)

    assert link == "https://example.com/outcome"
    assert quality == "outcome"


def test_match_play_to_catalog_prefers_event_and_outcome_match() -> None:
    play = {
        "player_display_name": "Tyrese Maxey",
        "target": "PTS",
        "direction": "UNDER",
        "market_line": 27.5,
        "market_date": "2026-05-06",
        "market_home_team": "PHI",
        "market_away_team": "NYK",
    }
    catalog = [
        {
            "market_key": "player_points",
            "outcome_side": "UNDER",
            "player": "Tyrese Maxey",
            "line": 27.5,
            "home_team_code": "PHI",
            "away_team_code": "NYK",
            "event_date": "2026-05-06",
            "bookmaker": "fanduel",
            "bookmaker_title": "FanDuel",
            "odds_price": -112,
            "link_quality": "outcome",
            "betslip_link": "https://example.com/fd/maxey",
        },
        {
            "market_key": "player_points",
            "outcome_side": "UNDER",
            "player": "Tyrese Maxey",
            "line": 27.5,
            "home_team_code": "MIN",
            "away_team_code": "SAS",
            "event_date": "2026-05-06",
            "bookmaker": "draftkings",
            "bookmaker_title": "DraftKings",
            "odds_price": -105,
            "link_quality": "outcome",
            "betslip_link": "https://example.com/dk/maxey",
        },
    ]

    match = match_play_to_catalog(play, catalog, sport="nba")

    assert match is not None
    assert match["bookmaker"] == "fanduel"
    assert match["betslip_link"] == "https://example.com/fd/maxey"


def test_build_parlay_sportsbook_options_marks_complete_books() -> None:
    legs = [
        {"play_key": "leg-1", "bookmaker": "fanduel", "bookmaker_title": "FanDuel", "betslip_link": "https://example.com/fd/1"},
        {"play_key": "leg-2", "bookmaker": "fanduel", "bookmaker_title": "FanDuel", "betslip_link": "https://example.com/fd/2"},
        {"play_key": "leg-2", "bookmaker": "draftkings", "bookmaker_title": "DraftKings", "betslip_link": "https://example.com/dk/2"},
    ]

    options = build_parlay_sportsbook_options(legs)

    assert options[0]["bookmaker"] == "fanduel"
    assert options[0]["complete"] is True
    assert options[0]["covered_leg_count"] == 2


def test_enrich_parlay_payload_with_sportsbooks_builds_frontend_board() -> None:
    parlay_payload = {
        "plays": [
            {
                "play_key": "leg-1",
                "player_display_name": "Tyrese Maxey",
                "target": "PTS",
                "direction": "UNDER",
                "market_line": 27.5,
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "betslip_link": "https://example.com/fd/maxey",
                "odds_american": -112,
                "expected_win_rate": 0.61,
            },
            {
                "play_key": "leg-2",
                "player_display_name": "Victor Wembanyama",
                "target": "PTS",
                "direction": "UNDER",
                "market_line": 26.5,
                "bookmaker": "fanduel",
                "bookmaker_title": "FanDuel",
                "betslip_link": "https://example.com/fd/wemby",
                "odds_american": -108,
                "expected_win_rate": 0.59,
            },
        ],
        "pairs": [
            {
                "ticket_rank": 1,
                "projected_probability": 0.36,
                "leg_count": 2,
                "legs": [
                    {"play_key": "leg-1", "player": "Tyrese Maxey", "target": "PTS", "direction": "UNDER"},
                    {"play_key": "leg-2", "player": "Victor Wembanyama", "target": "PTS", "direction": "UNDER"},
                ],
            }
        ],
        "summary": {},
    }

    enriched = enrich_parlay_payload_with_sportsbooks(parlay_payload, sport="nba")

    assert enriched["parlay_board"]["parlays"]
    parlay = enriched["parlay_board"]["parlays"][0]
    assert parlay["recommended_sportsbook"]["bookmaker"] == "fanduel"
    assert parlay["recommended_sportsbook"]["complete"] is True
    assert len(parlay["recommended_sportsbook"]["links"]) == 2
