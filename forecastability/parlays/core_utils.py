"""
Core utilities for parlay subsystem.
Handles odds conversions, probability calculations, and fundamental price/probability mappings.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import math


class PriceStatus(str, Enum):
    """Status of a priced binary event."""
    PRICE_VALID = "PRICE_VALID"
    PRICE_DEPENDENT = "PRICE_DEPENDENT"
    PASS_AT_PRICE = "PASS_AT_PRICE"
    MISSING_PRICE = "MISSING_PRICE"
    STALE_PRICE = "STALE_PRICE"
    INVALID_ODDS = "INVALID_ODDS"


class LegStatus(str, Enum):
    """Membership tier for a single leg."""
    SEED_PLAYABLE = "SEED_PLAYABLE"
    BALANCED_PLAYABLE = "BALANCED_PLAYABLE"
    PRICE_DEPENDENT = "PRICE_DEPENDENT"
    NEWS_DEPENDENT = "NEWS_DEPENDENT"
    BOUNDARY_SHADOW = "BOUNDARY_SHADOW"
    PASS = "PASS"


class ParlayDecision(str, Enum):
    """Final decision label for a parlay."""
    PARLAY_SEED_SHADOW = "PARLAY_SEED_SHADOW"
    PARLAY_BALANCED_SHADOW = "PARLAY_BALANCED_SHADOW"
    PARLAY_PRICE_DEPENDENT = "PARLAY_PRICE_DEPENDENT"
    PARLAY_NEWS_DEPENDENT = "PARLAY_NEWS_DEPENDENT"
    PARLAY_BOUNDARY_SHADOW = "PARLAY_BOUNDARY_SHADOW"
    PASS_LEG_NOT_IN_BETTABLE_SET = "PASS_LEG_NOT_IN_BETTABLE_SET"
    PASS_PAIRWISE_SUBSET_FAIL = "PASS_PAIRWISE_SUBSET_FAIL"
    PASS_JOINT_EV_NEGATIVE = "PASS_JOINT_EV_NEGATIVE"
    PASS_STRESS_FAIL = "PASS_STRESS_FAIL"
    PASS_SHARED_FAILURE_RISK = "PASS_SHARED_FAILURE_RISK"
    PASS_SHARED_EVENT_SUPPLY = "PASS_SHARED_EVENT_SUPPLY"
    PASS_SAME_GAME_INCOMPATIBLE = "PASS_SAME_GAME_INCOMPATIBLE"
    PASS_PRICE_INVALID = "PASS_PRICE_INVALID"
    PASS_SGP_PAYOUT_TOO_LOW = "PASS_SGP_PAYOUT_TOO_LOW"
    PASS_DEPENDENCY_UNKNOWN = "PASS_DEPENDENCY_UNKNOWN"
    PASS_TOO_MANY_CORRELATED_LEGS = "PASS_TOO_MANY_CORRELATED_LEGS"
    PASS_LOW_LCB_EDGE = "PASS_LOW_LCB_EDGE"


class CorrelationClass(str, Enum):
    """Dependency class between legs."""
    CROSS_GAME_WEAK_DEPENDENCE = "CROSS_GAME_WEAK_DEPENDENCE"
    SAME_GAME_PACE_POSITIVE = "SAME_GAME_PACE_POSITIVE"
    SAME_GAME_BLOWOUT_NEGATIVE = "SAME_GAME_BLOWOUT_NEGATIVE"
    SAME_TEAM_USAGE_COMPETING = "SAME_TEAM_USAGE_COMPETING"
    SAME_TEAM_ASSIST_SCORER_POSITIVE = "SAME_TEAM_ASSIST_SCORER_POSITIVE"
    SAME_PLAYER_OVERLAP = "SAME_PLAYER_OVERLAP"
    SAME_EVENT_SUPPLY_CONFLICT = "SAME_EVENT_SUPPLY_CONFLICT"
    MARKET_CORRELATED = "MARKET_CORRELATED"
    UNKNOWN_DEPENDENCE = "UNKNOWN_DEPENDENCE"


class NewsStatus(str, Enum):
    """News/status condition of a player/event."""
    CLEAR = "CLEAR"
    NEWS_DEPENDENT = "NEWS_DEPENDENT"
    OUT = "OUT"
    QUESTIONABLE = "QUESTIONABLE"
    MINUTES_LIMIT_RISK = "MINUTES_LIMIT_RISK"
    LINEUP_UNCONFIRMED = "LINEUP_UNCONFIRMED"


@dataclass
class OddsConversion:
    """Result of converting between American, Decimal, and Implied Probability."""
    american_odds: float
    decimal_odds: float
    implied_prob: float
    break_even_prob: float


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to Decimal odds."""
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(american_odds))


def decimal_to_american(decimal_odds: float) -> float:
    """Convert Decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    else:
        return -100.0 / (decimal_odds - 1.0)


def american_to_implied_prob(american_odds: float) -> float:
    """
    Convert American odds to implied probability (with vigorish).
    """
    decimal = american_to_decimal(american_odds)
    return 1.0 / decimal


def implied_prob_to_american(implied_prob: float) -> float:
    """Convert implied probability back to American odds."""
    decimal = 1.0 / implied_prob
    return decimal_to_american(decimal)


def american_to_break_even_prob(american_odds: float) -> float:
    """
    Break-even probability is the implied probability.
    If you bet at these odds, you need to win at least this % of the time to break even.
    """
    return american_to_implied_prob(american_odds)


def calculate_odds_conversion(american_odds: float) -> OddsConversion:
    """
    Full odds conversion pipeline.
    """
    decimal = american_to_decimal(american_odds)
    implied = american_to_implied_prob(american_odds)
    break_even = american_to_break_even_prob(american_odds)
    
    return OddsConversion(
        american_odds=american_odds,
        decimal_odds=decimal,
        implied_prob=implied,
        break_even_prob=break_even
    )


def min_acceptable_odds_for_edge(
    stress_probability: float,
    required_edge_margin: float = 0.015
) -> float:
    """
    Given a stress-tested probability and required edge margin,
    compute the minimum acceptable American odds.
    
    Example:
    - stress_probability = 0.535
    - required_edge_margin = 0.015
    - max acceptable break-even = 0.535 - 0.015 = 0.520
    - convert 0.520 break-even back to American odds
    
    Returns American odds (positive or negative).
    """
    max_break_even = stress_probability - required_edge_margin
    if max_break_even <= 0.0:
        return None  # Impossible to achieve required edge
    
    return implied_prob_to_american(max_break_even)


def edge_from_probability_and_odds(
    true_probability: float,
    american_odds: float
) -> float:
    """
    Calculate edge (EV per dollar wagered) given true probability and American odds.
    
    edge = (true_prob * decimal_odds - 1)
    """
    decimal = american_to_decimal(american_odds)
    edge = (true_probability * decimal) - 1.0
    return edge


def ev_from_edge(edge: float, wager: float = 1.0) -> float:
    """Expected value = edge * wager."""
    return edge * wager


def parlay_synthetic_odds(american_odds_list: list) -> float:
    """
    Calculate synthetic parlay American odds from list of individual American odds.
    
    Synthetic = product of decimal odds, converted back to American.
    """
    decimal_products = 1.0
    for american in american_odds_list:
        decimal = american_to_decimal(american)
        decimal_products *= decimal
    
    parlay_american = decimal_to_american(decimal_products)
    return parlay_american


def parlay_break_even_prob(american_odds_list: list) -> float:
    """
    Calculate break-even probability for a parlay at synthetic odds.
    """
    synthetic_american = parlay_synthetic_odds(american_odds_list)
    return american_to_break_even_prob(synthetic_american)


def lcb_edge(
    raw_edge: float,
    uncertainty_penalty: float = 0.01,
    edge_fragility: float = 0.0
) -> float:
    """
    Lower Confidence Bound edge: apply penalties for uncertainty and fragility.
    
    lcb_edge = raw_edge - uncertainty_penalty - edge_fragility
    """
    return max(0.0, raw_edge - uncertainty_penalty - edge_fragility)


def robust_edge(
    raw_edge: float,
    stress_downshift: float = 0.01
) -> float:
    """
    Robust edge: apply stress downshift.
    """
    return max(0.0, raw_edge - stress_downshift)


if __name__ == "__main__":
    # Test basic conversions
    print("Test American Odds Conversions:")
    print(f"-110 → {american_to_decimal(-110):.4f} decimal")
    print(f"-110 → {american_to_implied_prob(-110):.4f} implied prob")
    print(f"Bet $100 at -110: break-even {american_to_break_even_prob(-110):.2%}")
    
    print("\nTest Min Acceptable Odds:")
    min_odds = min_acceptable_odds_for_edge(0.535, required_edge_margin=0.015)
    print(f"For p=53.5% with 1.5% margin: min odds = {min_odds:.0f}")
    
    print("\nTest Edge Calculation:")
    edge = edge_from_probability_and_odds(0.525, -110)
    print(f"True prob 52.5% at -110: edge = {edge:.4f} ({edge:.2%})")
    
    print("\nTest Parlay Odds:")
    parlay_american = parlay_synthetic_odds([-110, -110])
    print(f"2x -110 parlay: {parlay_american:.0f} American ({american_to_decimal(parlay_american):.4f} decimal)")
    print(f"Break-even: {parlay_break_even_prob([-110, -110]):.2%}")
