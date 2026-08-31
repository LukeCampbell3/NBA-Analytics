from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskPolicy:
    production_staking_authorized: bool = False
    maximum_wager_units: float = 1.0
    maximum_slate_units: float = 5.0
    maximum_correlated_units: float = 1.0
    maximum_same_game_units: float = 1.0
    maximum_daily_drawdown_units: float = 5.0
    maximum_fractional_kelly: float = 0.10


def kelly_fraction(probability: float, decimal_price: float) -> float:
    if not 0 <= probability <= 1 or decimal_price <= 1:
        raise ValueError("invalid probability or price")
    net = decimal_price - 1.0
    return max(0.0, (probability * decimal_price - 1.0) / net)


def stake_units(*, mode: str, bankroll_units: float, probability: float,
                decimal_price: float, policy: RiskPolicy) -> float:
    if not policy.production_staking_authorized:
        return 0.0
    if bankroll_units <= 0:
        return 0.0
    if mode == "flat":
        proposed = 1.0
    elif mode == "fractional_kelly":
        proposed = bankroll_units * policy.maximum_fractional_kelly * kelly_fraction(probability, decimal_price)
    else:
        raise ValueError("unsupported stake mode")
    return min(proposed, policy.maximum_wager_units, policy.maximum_slate_units, bankroll_units)
