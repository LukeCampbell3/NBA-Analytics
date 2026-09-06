from __future__ import annotations

from dataclasses import replace

from sports.mlb.advanced.schema import AdvancedCandidateContext, BatterProcessProfile, PitcherProcessProfile
from sports.mlb.advanced.sequential_pa_model import (
    contact_outcome_probabilities,
    expected_pa_distribution,
    pa_event_probabilities,
    simulate_hitter_market,
)


def batter(**overrides):
    base = BatterProcessProfile(
        player_id=1,
        player_name="Test Batter",
        as_of_date="2026-09-05",
        sample_pa=300,
        sample_bbe=200,
        k_rate=0.20,
        bb_rate=0.09,
        hbp_rate=0.01,
        hr_rate=0.04,
        contact_rate=0.80,
        whiff_rate=0.20,
        xba=0.34,
        xslg=0.56,
        xwoba=0.38,
        hard_hit_rate=0.45,
        barrel_rate=0.10,
        support=0.95,
    )
    return replace(base, **overrides)


def pitcher(**overrides):
    base = PitcherProcessProfile(
        player_id=2,
        player_name="Test Pitcher",
        as_of_date="2026-09-05",
        sample_pa=400,
        sample_bbe=250,
        k_rate=0.24,
        bb_rate=0.08,
        hbp_rate=0.01,
        hr_rate=0.03,
        k_minus_bb_rate=0.16,
        whiff_rate=0.24,
        xba_allowed=0.31,
        xslg_allowed=0.49,
        xwoba_allowed=0.32,
        xfip=3.70,
        siera=3.65,
        projected_ip=5.8,
        projected_pitches=92.0,
        support=0.95,
    )
    return replace(base, **overrides)


def context(b=None, p=None, **overrides):
    base = AdvancedCandidateContext(
        game_id="123",
        run_date="2026-09-05",
        batter=b or batter(),
        pitcher=p or pitcher(),
        direct_matchup=None,
        batting_order=2,
        is_home=False,
        team_expected_runs=4.8,
        park_factor=1.0,
        defense_residual=0.0,
        defense_status="SPECIFIC_DEFENSE_AVAILABLE",
        data_freshness_status="FRESH",
        missing_components=(),
    )
    return replace(base, **overrides)


def test_pa_event_tree_sums_to_one_and_hr_is_separate_from_contact():
    probs = pa_event_probabilities(batter(), pitcher(), times_through_order=1)
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert set(probs) == {"K", "BB", "HBP", "HR", "NON_HR_CONTACT", "OTHER"}
    assert probs["HR"] > 0
    assert probs["NON_HR_CONTACT"] > 0


def test_contact_tree_sums_to_one():
    probs = contact_outcome_probabilities(
        batter(), pitcher(), direct_matchup=None, defense_residual=0.0, park_factor=1.0
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert set(probs) == {"OUT", "1B", "2B", "3B", "ROE_OTHER"}


def test_higher_strikeout_pitcher_reduces_non_hr_contact_probability():
    low = pa_event_probabilities(batter(), pitcher(k_rate=0.17), times_through_order=1)
    high = pa_event_probabilities(batter(), pitcher(k_rate=0.36), times_through_order=1)
    assert high["K"] > low["K"]
    assert high["NON_HR_CONTACT"] < low["NON_HR_CONTACT"]


def test_better_contact_quality_increases_extra_base_share_and_expected_tb():
    weak = contact_outcome_probabilities(
        batter(xba=0.27, xslg=0.38), pitcher(xba_allowed=0.28, xslg_allowed=0.40),
        direct_matchup=None, defense_residual=0.0, park_factor=1.0,
    )
    strong = contact_outcome_probabilities(
        batter(xba=0.39, xslg=0.72), pitcher(xba_allowed=0.36, xslg_allowed=0.65),
        direct_matchup=None, defense_residual=0.0, park_factor=1.0,
    )
    weak_tb = weak["1B"] + 2 * weak["2B"] + 3 * weak["3B"]
    strong_tb = strong["1B"] + 2 * strong["2B"] + 3 * strong["3B"]
    assert strong_tb > weak_tb


def test_defense_residual_is_zero_centered_and_directional():
    average = contact_outcome_probabilities(batter(), pitcher(), direct_matchup=None, defense_residual=0.0, park_factor=1.0)
    elite = contact_outcome_probabilities(batter(), pitcher(), direct_matchup=None, defense_residual=-0.03, park_factor=1.0)
    poor = contact_outcome_probabilities(batter(), pitcher(), direct_matchup=None, defense_residual=0.03, park_factor=1.0)
    average_hit = average["1B"] + average["2B"] + average["3B"]
    elite_hit = elite["1B"] + elite["2B"] + elite["3B"]
    poor_hit = poor["1B"] + poor["2B"] + poor["3B"]
    assert elite_hit < average_hit < poor_hit


def test_top_order_hitter_has_more_expected_pa_than_bottom_order():
    top = expected_pa_distribution(batting_order=1, is_home=False, team_expected_runs=4.5)
    bottom = expected_pa_distribution(batting_order=9, is_home=False, team_expected_runs=4.5)
    top_mean = sum(pa * prob for pa, prob in top.items())
    bottom_mean = sum(pa * prob for pa, prob in bottom.items())
    assert top_mean > bottom_mean


def test_simulation_tracks_pa_and_ab_separately_and_probability_identities():
    result = simulate_hitter_market(context(), target="H", market_line=0.5, trials=8000)
    assert result.expected_ab < result.expected_pa
    assert abs(result.hit_over_0_5_probability - (1.0 - result.p_h_0)) < 1e-12
    assert abs(result.p_h_0 + result.p_h_1 + result.p_h_ge_2 - 1.0) < 1e-12
    assert abs(result.p_tb_0 + result.p_tb_1 + result.p_tb_ge_2 - 1.0) < 1e-12
    assert result.market_clear_probabilities["H|OVER|0.5"] == result.hit_over_0_5_probability


def test_tb_over_1_5_is_direct_simulated_tail():
    result = simulate_hitter_market(context(), target="TB", market_line=1.5, trials=8000)
    assert result.market_clear_probabilities["TB|OVER|1.5"] == result.tb_over_1_5_probability
    assert result.tb_over_1_5_probability == result.p_tb_ge_2


def test_simulation_is_deterministic_for_same_candidate_and_seed_contract():
    first = simulate_hitter_market(context(), target="H", market_line=0.5, trials=6000)
    second = simulate_hitter_market(context(), target="H", market_line=0.5, trials=6000)
    assert first.p_h_0 == second.p_h_0
    assert first.expected_hits == second.expected_hits


def test_missing_specific_defense_only_increases_uncertainty_not_fake_adjustment():
    ctx = context(defense_status="AVERAGE_CONTEXT_RESIDUAL_ONLY_UNTIL_SPECIFIC_OAA_IS_AVAILABLE", defense_residual=0.0)
    result = simulate_hitter_market(ctx, target="H", market_line=0.5, trials=6000)
    assert result.diagnostics["defense_residual"] == 0.0
    assert result.uncertainty_components["defense_specificity_missing"] == 1.0
