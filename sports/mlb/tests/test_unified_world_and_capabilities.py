import numpy as np
import pytest

from sports.mlb.unified.market_conditioning import condition_mask
from sports.mlb.unified.market_queries import event_market_query, query_mask
from sports.mlb.unified.player_share import PlayerShare, allocate_team_events, conditional_probability, smoothed_shares
from sports.mlb.unified.settlement import Settlement, settle
from sports.mlb.unified.trajectory import simulate_team_runs


def test_player_allocations_preserve_team_identity():
    shares = smoothed_shares([PlayerShare("a", 20, 100), PlayerShare("b", 10, 80)])
    team = np.array([0, 1, 5, 2])
    allocation = allocate_team_events(team, shares, seed=4)
    assert np.array_equal(sum(allocation.values()), team)


def test_team_and_f5_queries_use_same_worlds():
    batch = simulate_team_runs(4, 4, trials=100, seed=2)
    full = query_mask(batch, market_type="game_total", side="over", line=8.5)
    f5 = query_mask(batch, market_type="first_5_innings_total", side="over", line=8.5)
    assert np.all(~f5 | full)


def test_event_markets_fail_closed_without_model_and_identity():
    with pytest.raises(ValueError, match="EVENT_MODEL_REQUIRED"):
        event_market_query(market_type="pa_pitch_count", event_identity="g:b:pa1", event_model_available=False)
    with pytest.raises(ValueError, match="EVENT_IDENTITY_UNAVAILABLE"):
        event_market_query(market_type="pa_pitch_count", event_identity=None, event_model_available=True)


def test_market_conditioning_does_not_invent_pmf_authority():
    event = np.array([1, 0, 1, 0], bool)
    result = condition_mask(event, np.ones(4), None, identification_level=3)
    assert result.probability == .5 and result.authority is False


def test_conditional_lift_and_deterministic_settlement():
    metric = conditional_probability(np.array([1,1,0,0]), np.array([1,0,1,0]))
    assert metric["conditional_probability"] == .5
    assert settle(market_type="batter_hits", side="over", line=.5, observed=1) == Settlement.WIN
    assert settle(market_type="pa_pitch_count", side="over", line=4.5, observed=5) == Settlement.BLOCKED
