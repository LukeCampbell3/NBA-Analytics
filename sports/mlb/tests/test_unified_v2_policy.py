from __future__ import annotations

import pytest

from sports.mlb.unified.v2_policy import (
    UnifiedPolicyV2, discrete_over_probability, discrete_settlement,
    evaluate_v2_candidate, implied_probability, poisson_binomial_cdf,
    remove_two_way_vig,
)


def test_odds_conversion_and_vig_removal():
    assert implied_probability(-150) == pytest.approx(.6)
    assert implied_probability(120) == pytest.approx(1 / 2.2)
    over, under = remove_two_way_vig(-110, -110)
    assert over == pytest.approx(.5)
    assert under == pytest.approx(.5)


def test_hits_and_total_bases_use_exact_discrete_mass_and_push_rules():
    hits = {0: .35, 1: .40, 2: .20, 3: .05}
    total_bases = {0: .40, 1: .25, 2: .15, 3: .05, 4: .15}
    assert discrete_over_probability(hits, .5) == pytest.approx(.65)
    assert discrete_over_probability(hits, 1.5) == pytest.approx(.25)
    assert discrete_over_probability(total_bases, 1.5) == pytest.approx(.35)
    assert discrete_settlement(1, 1.0, "OVER") == "push"
    assert discrete_settlement(2, 1.5, "OVER") == "won"


def test_poisson_binomial_cdf_uses_individual_probabilities():
    assert poisson_binomial_cdf([.2, .8], 0) == pytest.approx(.16)
    assert poisson_binomial_cdf([.2, .8], 1) == pytest.approx(.84)


def _candidate(**overrides):
    value = {
        "capability": "batter_hits", "identity_status": "CONFIRMED",
        "lineup_status": "CONFIRMED", "player_status": "ACTIVE",
        "support_size": 100, "support_status": "IN_SUPPORT",
        "quote_timestamp": "2026-08-30T15:00:00Z",
        "decision_timestamp": "2026-08-30T15:05:00Z",
        "market_id": "1", "selection_id": "2", "usable_probability": .68,
        "uncertainty": .03, "quoted_odds": -150,
    }
    value.update(overrides)
    return value


def test_v2_admissibility_fails_closed_and_zero_pick_slates_are_valid():
    assert evaluate_v2_candidate(_candidate())["admissible"] is True
    assert evaluate_v2_candidate(_candidate(quote_timestamp=None))["admissible"] is False
    assert evaluate_v2_candidate(_candidate(support_status="OUT_OF_SUPPORT"))["admissible"] is False
    assert evaluate_v2_candidate(_candidate(support_size=4))["admissible"] is False
    assert evaluate_v2_candidate(_candidate(usable_probability=.95, quoted_odds=-2000))["admissible"] is False


def test_v2_policy_hash_is_deterministic_and_production_is_not_authorized():
    first, second = UnifiedPolicyV2(), UnifiedPolicyV2()
    assert first.policy_hash == second.policy_hash
    assert len(first.policy_hash) == 64
    assert first.production_authorized is False
