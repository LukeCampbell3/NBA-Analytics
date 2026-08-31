from sports.mlb.unified.adapters import adapt_legacy_play, adapt_team_leg
from sports.mlb.unified.decision import DecisionPolicy, decide


def test_legacy_final_probability_is_not_replaced_by_intermediate_ev_probability():
    candidate = adapt_legacy_play({
        "game_id": "g1", "player": "A Player", "player_id": "1", "team": "SEA", "opponent": "TOR",
        "target": "TB", "direction": "OVER", "market_line": 1.5, "selected_side_price": 105,
        "final_hit_probability": .61, "estimated_hit_probability": .74, "model_hit_probability": .78,
        "historical_bucket_support": 100, "lineup_status": "confirmed", "price_confirmed": True,
    })
    result = decide(candidate, DecisionPolicy())
    assert result.usable_probability == .61
    assert result.conservative_expected_value == .61 * 2.05 - 1


def test_team_leg_without_uncertainty_enters_pool_but_fails_closed():
    candidate = adapt_team_leg(
        {"market": "game_total", "side": "over", "line": 8.5, "price_american": -110,
         "model_probability": .62, "leg_authorized": False, "support_blocking_dimensions": ["market_support"]},
        {"game_id": "g1", "home_team": "SEA", "away_team": "TOR"},
    )
    result = decide(candidate, DecisionPolicy())
    assert "UNCERTAINTY_UNAVAILABLE" in result.rejection_reasons
    assert "SUPPORT_INVALID" in result.rejection_reasons
