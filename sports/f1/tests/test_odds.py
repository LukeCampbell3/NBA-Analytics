from __future__ import annotations

from sports.f1.predictions.odds import attach_consensus_market, flatten_kalshi_event, flatten_polymarket_event


def test_polymarket_binary_winner_market_is_normalized() -> None:
    event = {
        "id": "evt", "title": "Formula 1 Italian Grand Prix Winner", "active": True,
        "markets": [{
            "question": "Will Lando Norris win the Italian Grand Prix?", "groupItemTitle": "Lando Norris to win",
            "active": True, "closed": False, "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.25", "0.75"]', "bestAsk": 0.27, "liquidityNum": 5000,
        }],
    }
    rows = flatten_polymarket_event(event, "2026-08-13T00:00:00+00:00")
    assert rows[0]["driver"] == "Lando Norris"
    assert rows[0]["market_probability"] == 0.27
    assert rows[0]["bookmaker"] == "polymarket"


def test_kalshi_winner_market_is_normalized() -> None:
    event = {
        "event_ticker": "KXF1", "title": "Who will win the Formula 1 Italian Grand Prix?",
        "markets": [{
            "ticker": "KXF1-LN", "title": "Will Lando Norris finish in first in the main race?", "yes_sub_title": "Lando Norris",
            "yes_ask_dollars": "0.2800", "updated_time": "2026-08-13T00:00:00Z",
        }],
    }
    rows = flatten_kalshi_event(event, "2026-08-13T00:00:00+00:00")
    assert rows[0]["driver"] == "Lando Norris"
    assert rows[0]["market_probability"] == 0.28


def test_consensus_matches_names_and_selects_lowest_ask() -> None:
    projections = [{"driver": "Lando Norris", "win_probability": 0.31}]
    observations = [
        {"driver": "Lando Norris", "bookmaker": "polymarket", "bookmaker_title": "Polymarket", "market_probability": 0.27, "decimal_price": 1 / 0.27, "american_price": 270},
        {"driver": "L. Norris", "bookmaker": "kalshi", "bookmaker_title": "Kalshi", "market_probability": 0.29, "decimal_price": 1 / 0.29, "american_price": 245},
    ]
    attach_consensus_market(projections, observations)
    assert projections[0]["market_probability"] == 0.28
    assert projections[0]["best_book"] == "Polymarket"
    assert abs(projections[0]["edge"] - 0.03) < 1e-12
