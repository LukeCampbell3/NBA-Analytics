from sports.mlb.unified.risk import RiskPolicy, stake_units


def test_staking_is_zero_until_explicitly_authorized():
    assert stake_units(mode="flat", bankroll_units=100, probability=.7, decimal_price=2, policy=RiskPolicy()) == 0


def test_fractional_kelly_is_capped_when_authorized():
    policy = RiskPolicy(production_staking_authorized=True, maximum_wager_units=.5, maximum_fractional_kelly=.1)
    assert 0 < stake_units(mode="fractional_kelly", bankroll_units=100, probability=.7, decimal_price=2, policy=policy) <= .5
