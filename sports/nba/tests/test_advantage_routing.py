"""Unit + reconciliation tests for the advantage-routing analysis package
(spec sections 36-38). No live network access is required anywhere in
this file -- fixtures are small, frozen, hand-built objects, and the
reconciliation/JSON-hygiene tests read the already-generated real player
artifacts off disk rather than re-fetching anything.
"""

from __future__ import annotations

import math
from pathlib import Path

import json

import pytest

from sports.nba.analytics.advantage_routing.build.build_player import OUTPUT_ROOT
from sports.nba.analytics.advantage_routing.build.validate import reconcile_player_artifact, validate_all
from sports.nba.analytics.advantage_routing.gravity.gravity_model import build_gravity_profile
from sports.nba.analytics.advantage_routing.models.schemas import EvidenceStatus, Metric
from sports.nba.analytics.advantage_routing.routing.recipients import build_recipient_network
from sports.nba.analytics.advantage_routing.routing.states import (
    ROUTING_STATE_UNAVAILABLE_REASON,
    classify_routing_state,
    classify_shot_zone_from_text,
)
from sports.nba.analytics.advantage_routing.simulation.monte_carlo import (
    DEFAULT_SEED,
    MonteCarloInputs,
    RateObservation,
    run_monte_carlo,
)
from sports.nba.analytics.advantage_routing.simulation.saturation import (
    saturation_retention,
    turnover_growth_from_saturation,
)
from sports.nba.analytics.advantage_routing.simulation.usage import (
    ScenarioParameters,
    SimulationBaseline,
    simulate_scenario,
    standard_scenarios,
    touch_multiplier,
)
from sports.nba.analytics.advantage_routing.sources.bball_ref import (
    RealAssistEvent,
    SeasonShootingTable,
    ShotTypeRow,
    ZoneShootingRow,
)
from sports.nba.analytics.advantage_routing.stats.pass_value import build_pass_value_model
from sports.nba.analytics.advantage_routing.stats.shrinkage import beta_binomial_shrink, dirichlet_shrink
from sports.nba.analytics.advantage_routing.sources.bball_ref import LeagueShootingBaseline
from sports.nba.analytics.advantage_routing.sources import bball_ref as bball_ref_module


# ---------------------------------------------------------------------
# Metric / schema factories
# ---------------------------------------------------------------------

def test_metric_unavailable_has_no_value_and_a_reason() -> None:
    m = Metric.unavailable("x", reason="no source")
    assert m.value is None
    assert m.status == EvidenceStatus.UNAVAILABLE.value
    assert m.reason == "no source"


def test_metric_observed_derived_reconstructed_simulated_statuses() -> None:
    assert Metric.observed("a", 1.0, source="s").status == EvidenceStatus.OBSERVED.value
    assert Metric.derived("b", 1.0, method="m").status == EvidenceStatus.DERIVED.value
    assert Metric.reconstructed("c", 1.0, method="m", confidence=0.5).status == EvidenceStatus.RECONSTRUCTED.value
    assert Metric.simulated("d", 1.0, method="m").status == EvidenceStatus.SIMULATED.value


# ---------------------------------------------------------------------
# Shrinkage (spec section 14)
# ---------------------------------------------------------------------

def test_beta_binomial_shrink_pulls_small_sample_toward_prior() -> None:
    # 1 success in 2 trials (raw rate 0.5) with a prior mean of 0.2 and a
    # strong prior -- the shrunk rate must move toward 0.2, away from 0.5.
    result = beta_binomial_shrink(successes=1, trials=2, prior_mean=0.2, prior_strength=20.0)
    assert result.raw_rate == pytest.approx(0.5)
    assert result.shrunk_rate < 0.5
    assert result.shrunk_rate > 0.2
    assert result.credible_interval_low < result.shrunk_rate < result.credible_interval_high


def test_beta_binomial_shrink_converges_to_raw_rate_with_large_sample() -> None:
    # A large real sample should dominate a weak prior.
    result = beta_binomial_shrink(successes=800, trials=1000, prior_mean=0.2, prior_strength=8.0)
    assert result.shrunk_rate == pytest.approx(0.8, abs=0.02)


def test_beta_binomial_shrink_rejects_invalid_prior_mean() -> None:
    with pytest.raises(ValueError):
        beta_binomial_shrink(successes=1, trials=2, prior_mean=1.5)


def test_dirichlet_shrink_shares_sum_to_one() -> None:
    counts = {"a": 5, "b": 3, "c": 0}
    results = dirichlet_shrink(counts, prior_strength=6.0)
    assert sum(r.shrunk_share for r in results) == pytest.approx(1.0, abs=1e-9)
    assert sum(r.raw_share for r in results) == pytest.approx(1.0, abs=1e-9)


def test_dirichlet_shrink_pulls_zero_count_category_above_zero() -> None:
    counts = {"a": 5, "b": 3, "c": 0}
    results = dirichlet_shrink(counts, prior_strength=6.0)
    zero_cat = next(r for r in results if r.category == "c")
    assert zero_cat.raw_share == 0.0
    assert zero_cat.shrunk_share > 0.0  # prior mass keeps it off exactly zero


def test_dirichlet_shrink_empty_input() -> None:
    assert dirichlet_shrink({}) == []


# ---------------------------------------------------------------------
# Saturation curve (spec section 20)
# ---------------------------------------------------------------------

def test_saturation_retention_is_one_at_or_below_h_equals_one() -> None:
    assert saturation_retention(1.0, 0.55) == pytest.approx(1.0)
    assert saturation_retention(0.5, 0.55) == pytest.approx(1.0)  # shrinking role never "loses" efficiency


def test_saturation_retention_decreases_as_h_grows() -> None:
    r1 = saturation_retention(1.2, 0.55)
    r2 = saturation_retention(2.0, 0.55)
    assert 0.0 < r2 < r1 < 1.0


def test_turnover_growth_from_saturation_is_zero_at_full_retention() -> None:
    assert turnover_growth_from_saturation(1.0) == pytest.approx(0.0)


def test_turnover_growth_from_saturation_increases_as_retention_falls() -> None:
    g1 = turnover_growth_from_saturation(0.9)
    g2 = turnover_growth_from_saturation(0.5)
    assert g2 > g1 > 0.0


# ---------------------------------------------------------------------
# Usage/role simulator + simulation-validation invariants (spec section 38)
# ---------------------------------------------------------------------

def _baseline() -> SimulationBaseline:
    return SimulationBaseline(
        baseline_decision_touches_per_game=14.0,
        baseline_ast_per_game=2.5,
        baseline_tov_per_game=1.8,
        baseline_ast_per_touch=2.5 / 14.0,
        baseline_tov_per_touch=1.8 / 14.0,
        baseline_makes_per_touch=0.20,
        current_usage_pct=15.0,
    )


def test_touch_multiplier_is_one_at_current_usage() -> None:
    assert touch_multiplier(15.0, 15.0, elasticity=0.6) == pytest.approx(1.0)


def test_touch_multiplier_zero_current_usage_is_safe() -> None:
    # Guards the real edge case of a player with no recorded usage.
    assert touch_multiplier(20.0, 0.0, elasticity=0.6) == 1.0


def test_simulate_scenario_at_current_usage_reduces_to_baseline() -> None:
    """Invariant: pass_tendency_change=0 and target_usage==current_usage
    must give H=1.0, full retention, zero turnover growth, and simulated
    touches/assists/turnovers matching the baseline rates exactly."""
    baseline = _baseline()
    params = ScenarioParameters(target_usage_pct=baseline.current_usage_pct, pass_tendency_change=0.0)
    result = simulate_scenario(baseline, params, scenario_name="identity")

    assert result.touch_multiplier_h.value == pytest.approx(1.0)
    assert result.efficiency_retention_used.value == pytest.approx(1.0)
    assert result.turnover_growth_used.value == pytest.approx(0.0)
    assert result.simulated_decision_touches.value == pytest.approx(baseline.baseline_decision_touches_per_game)
    assert result.simulated_assists.value == pytest.approx(baseline.baseline_ast_per_game)
    assert result.simulated_turnovers.value == pytest.approx(baseline.baseline_tov_per_game)


def test_more_touches_increase_expected_counts_not_probabilities() -> None:
    """Invariant: a higher target usage (more decision touches) must
    increase expected assist/make COUNTS, while the underlying per-touch
    RATE (ast_per_touch) used is unchanged -- the simulator never
    inflates the rate itself, only the volume it is applied to."""
    baseline = _baseline()
    low = simulate_scenario(baseline, ScenarioParameters(target_usage_pct=15.0), scenario_name="low")
    high = simulate_scenario(baseline, ScenarioParameters(target_usage_pct=30.0), scenario_name="high")

    assert high.simulated_decision_touches.value > low.simulated_decision_touches.value
    assert high.simulated_assists.value > low.simulated_assists.value
    # The rate itself (assists / decision touches) must not exceed the
    # baseline rate once retention is < 1 -- higher volume never implies
    # a higher per-touch rate than what was actually observed.
    implied_rate = high.simulated_assists.value / high.simulated_decision_touches.value
    assert implied_rate <= baseline.baseline_ast_per_touch + 1e-9


def test_reduced_retention_must_not_increase_assists() -> None:
    """Invariant: lowering efficiency_retention, holding everything else
    fixed, must not increase simulated assists or receiver makes."""
    baseline = _baseline()
    high_retention = simulate_scenario(
        baseline, ScenarioParameters(target_usage_pct=25.0, efficiency_retention=0.95, turnover_growth=0.0), scenario_name="hi"
    )
    low_retention = simulate_scenario(
        baseline, ScenarioParameters(target_usage_pct=25.0, efficiency_retention=0.60, turnover_growth=0.0), scenario_name="lo"
    )
    assert low_retention.simulated_assists.value < high_retention.simulated_assists.value
    assert low_retention.simulated_receiver_makes.value < high_retention.simulated_receiver_makes.value


def test_higher_turnover_growth_never_reduces_turnovers() -> None:
    """Invariant: increasing turnover_growth, holding everything else
    fixed, must not decrease simulated turnovers."""
    baseline = _baseline()
    low_growth = simulate_scenario(baseline, ScenarioParameters(target_usage_pct=25.0, turnover_growth=0.0), scenario_name="lo")
    high_growth = simulate_scenario(baseline, ScenarioParameters(target_usage_pct=25.0, turnover_growth=0.5), scenario_name="hi")
    assert high_growth.simulated_turnovers.value > low_growth.simulated_turnovers.value


def test_standard_scenarios_optimistic_beats_conservative_on_fixed_retention() -> None:
    """OPTIMISTIC and CONSERVATIVE use fixed (not saturation-derived)
    retention/turnover-growth values (0.97/0.05 vs 0.80/0.25), and both
    share the same H at a given target usage -- so OPTIMISTIC assists
    must exceed CONSERVATIVE assists, and CONSERVATIVE turnovers must
    exceed OPTIMISTIC turnovers, at every target usage. NEUTRAL uses its
    own dynamic saturation curve (a different k) and is deliberately NOT
    asserted to fall strictly between the two -- that would assume a
    monotonic relationship the model does not actually guarantee."""
    baseline = _baseline()
    scenarios = standard_scenarios(baseline, target_usage_pct=28.0)
    opt, con = scenarios["OPTIMISTIC"], scenarios["CONSERVATIVE"]
    assert opt.simulated_assists.value > con.simulated_assists.value
    assert opt.simulated_receiver_makes.value > con.simulated_receiver_makes.value
    assert opt.simulated_turnovers.value < con.simulated_turnovers.value
    assert opt.efficiency_retention_used.value == pytest.approx(0.97)
    assert con.efficiency_retention_used.value == pytest.approx(0.80)


def test_simulate_scenario_never_produces_nan_or_infinite_output() -> None:
    baseline = _baseline()
    result = simulate_scenario(baseline, ScenarioParameters(target_usage_pct=40.0, pass_tendency_change=0.5), scenario_name="extreme")
    for m in (
        result.touch_multiplier_h, result.simulated_decision_touches, result.simulated_passes,
        result.simulated_assists, result.simulated_receiver_makes, result.simulated_turnovers,
    ):
        assert m.value is not None
        assert math.isfinite(m.value)


# ---------------------------------------------------------------------
# Monte Carlo reproducibility (spec section 21)
# ---------------------------------------------------------------------

def _mc_inputs() -> MonteCarloInputs:
    return MonteCarloInputs(
        decision_touches=RateObservation(successes=200, trials=1000, prior_mean=0.2),
        ast_per_touch=RateObservation(successes=40, trials=200, prior_mean=0.18),
        makes_per_touch=RateObservation(successes=30, trials=200, prior_mean=0.15),
        tov_per_touch=RateObservation(successes=25, trials=200, prior_mean=0.12),
        baseline_decision_touches_per_game=14.0,
        current_usage_pct=15.0,
    )


def test_monte_carlo_is_reproducible_with_same_seed() -> None:
    inputs = _mc_inputs()
    params = ScenarioParameters(target_usage_pct=25.0)
    r1 = run_monte_carlo(inputs, params, scenario_name="s", seed=DEFAULT_SEED, n_draws=500)
    r2 = run_monte_carlo(inputs, params, scenario_name="s", seed=DEFAULT_SEED, n_draws=500)
    assert r1.assists.as_dict() == r2.assists.as_dict()
    assert r1.turnovers.as_dict() == r2.turnovers.as_dict()
    assert r1.receiver_makes.as_dict() == r2.receiver_makes.as_dict()


def test_monte_carlo_different_seeds_are_not_required_to_match() -> None:
    inputs = _mc_inputs()
    params = ScenarioParameters(target_usage_pct=25.0)
    r1 = run_monte_carlo(inputs, params, scenario_name="s", seed=1, n_draws=500)
    r2 = run_monte_carlo(inputs, params, scenario_name="s", seed=2, n_draws=500)
    assert r1.assists.median != r2.assists.median  # different seeds draw different distributions


def test_monte_carlo_percentiles_are_ordered() -> None:
    inputs = _mc_inputs()
    result = run_monte_carlo(inputs, ScenarioParameters(target_usage_pct=25.0), n_draws=2000, seed=DEFAULT_SEED)
    for summary in (result.assists, result.turnovers, result.receiver_makes):
        assert summary.p10 <= summary.p25 <= summary.median <= summary.p75 <= summary.p90


# ---------------------------------------------------------------------
# Shot-zone / routing-state classification honesty (spec sections 9, 37)
# ---------------------------------------------------------------------

def test_classify_shot_zone_three_pointer_is_above_break_with_caveat() -> None:
    result = classify_shot_zone_from_text("3-pt jump shot from 28 ft", 28.0, is_three=True)
    assert result.zone == "ABOVE_BREAK_3"
    assert result.status == EvidenceStatus.DERIVED.value
    assert result.caveat is not None and "corner" in result.caveat.lower()


def test_classify_shot_zone_dunk_is_rim() -> None:
    result = classify_shot_zone_from_text("Dunk", 0.0, is_three=False)
    assert result.zone == "RIM"


@pytest.mark.parametrize(
    ("distance", "expected_zone"),
    [(2.0, "RIM"), (3.0, "RIM"), (8.0, "SHORT_PAINT"), (10.0, "SHORT_PAINT"), (18.0, "MIDRANGE")],
)
def test_classify_shot_zone_by_distance_buckets(distance: float, expected_zone: str) -> None:
    result = classify_shot_zone_from_text("jump shot", distance, is_three=False)
    assert result.zone == expected_zone


def test_classify_shot_zone_missing_distance_defaults_midrange_honestly() -> None:
    result = classify_shot_zone_from_text("shot", None, is_three=False)
    assert result.zone == "MIDRANGE"
    assert "no_distance_reported" in result.method


def test_classify_routing_state_is_always_honestly_unavailable() -> None:
    """This is not a placeholder bug -- see routing/states.py module
    docstring. The test locks in that honesty boundary."""
    result = classify_routing_state()
    assert result.status == EvidenceStatus.UNAVAILABLE.value
    assert result.caveat == ROUTING_STATE_UNAVAILABLE_REASON


# ---------------------------------------------------------------------
# Recipient network (spec section 7)
# ---------------------------------------------------------------------

def _fixture_assists() -> list[RealAssistEvent]:
    return [
        RealAssistEvent("g1", "passerslug01", "Passer One", "recva01", "Recipient A", "Dunk", 0.0, False),
        RealAssistEvent("g1", "passerslug01", "Passer One", "recva01", "Recipient A", "Layup", 2.0, False),
        RealAssistEvent("g2", "passerslug01", "Passer One", "recva01", "Recipient A", "3-pt jump shot from 25 ft", 25.0, True),
        RealAssistEvent("g2", "passerslug01", "Passer One", "recvb01", "Recipient B", "Jump shot from 17 ft", 17.0, False),
        RealAssistEvent("g3", "passerslug01", "Passer One", "recvb01", "Recipient B", "Jump shot from 9 ft", 9.0, False),
    ]


def test_recipient_network_assist_counts_and_share() -> None:
    network = build_recipient_network(
        "Passer One", _fixture_assists(), games_sampled=3, games_available_total=10, season="2025-26",
    )
    assert network.sample_size == 5
    recipient_a = next(r for r in network.recipients if r.recipient_slug == "recva01")
    recipient_b = next(r for r in network.recipients if r.recipient_slug == "recvb01")
    assert recipient_a.assists.value == 3
    assert recipient_b.assists.value == 2
    assert recipient_a.assist_share.value == pytest.approx(3 / 5)
    assert recipient_b.assist_share.value == pytest.approx(2 / 5)
    total_share = sum(r.assist_share.value for r in network.recipients)
    assert total_share == pytest.approx(1.0, abs=1e-9)


def test_recipient_network_zone_breakdown_sums_to_assists() -> None:
    network = build_recipient_network(
        "Passer One", _fixture_assists(), games_sampled=3, games_available_total=10, season="2025-26",
    )
    for recipient in network.recipients:
        assert sum(recipient.zone_breakdown.values()) == recipient.assists.value


def test_recipient_network_pass_dependent_fields_are_honestly_unavailable() -> None:
    network = build_recipient_network(
        "Passer One", _fixture_assists(), games_sampled=3, games_available_total=10, season="2025-26",
    )
    for recipient in network.recipients:
        assert recipient.passes.status == EvidenceStatus.UNAVAILABLE.value
        assert recipient.pass_share.status == EvidenceStatus.UNAVAILABLE.value
        assert recipient.ast_per_pass.status == EvidenceStatus.UNAVAILABLE.value
        assert recipient.recipient_leverage.status == EvidenceStatus.UNAVAILABLE.value


def test_recipient_network_empty_input_is_safe() -> None:
    network = build_recipient_network("Nobody", [], games_sampled=0, games_available_total=0, season="2025-26")
    assert network.sample_size == 0
    assert network.recipients == []


# ---------------------------------------------------------------------
# Gravity model (spec section 10)
# ---------------------------------------------------------------------

def _fixture_shooting_table() -> SeasonShootingTable:
    return SeasonShootingTable(
        player_slug="testplayer01",
        season_end_year="2026",
        zones=[
            ZoneShootingRow("At Rim", fg=150, fga=220, fg_pct=0.682, fg_assisted=90, fg_assisted_pct=0.6),
            ZoneShootingRow("3 to <10 ft", fg=40, fga=100, fg_pct=0.40, fg_assisted=30, fg_assisted_pct=0.75),
            ZoneShootingRow("10 to <16 ft", fg=15, fga=40, fg_pct=0.375, fg_assisted=25, fg_assisted_pct=0.9),
            ZoneShootingRow("16 ft to <3-pt", fg=10, fga=30, fg_pct=0.333, fg_assisted=20, fg_assisted_pct=0.9),
            ZoneShootingRow("3-pt", fg=20, fga=60, fg_pct=0.333, fg_assisted=55, fg_assisted_pct=0.9),
        ],
        shot_types=[
            ShotTypeRow("DUNK", fg=50, fga=60, fg_pct=0.833, fg_assisted=30, fg_assisted_pct=0.5),
            ShotTypeRow("HOOK_SHOT", fg=20, fga=35, fg_pct=0.571, fg_assisted=5, fg_assisted_pct=0.15),
        ],
        season_fga=450,
        season_fg_assisted_pct=0.62,
        url="https://example.invalid/fixture",
    )


def test_gravity_profile_populates_five_mechanisms_and_leaves_short_roll_unavailable() -> None:
    profile = build_gravity_profile(
        "Test Player", "2025-26", shooting_table=_fixture_shooting_table(),
        mean_fga_per_game=12.0, mean_fta_per_game=3.0, games_played=50,
    )
    assert "SHORT_ROLL_GRAVITY" not in profile.mechanisms_present
    assert profile.components["SHORT_ROLL_GRAVITY"]["summary"].status == EvidenceStatus.UNAVAILABLE.value
    for mech in ("PAINT_FACEUP_GRAVITY", "VERTICAL_GRAVITY", "POP_GRAVITY", "PERIMETER_GRAVITY"):
        assert mech in profile.mechanisms_present


def test_gravity_profile_post_scoring_is_reconstructed_with_confidence() -> None:
    profile = build_gravity_profile(
        "Test Player", "2025-26", shooting_table=_fixture_shooting_table(),
        mean_fga_per_game=12.0, mean_fta_per_game=3.0, games_played=50,
    )
    index_metric = profile.components["POST_SCORING_GRAVITY"]["post_scoring_gravity_index"]
    assert index_metric.status == EvidenceStatus.RECONSTRUCTED.value
    assert index_metric.confidence == pytest.approx(0.45)


def test_gravity_profile_with_no_shooting_table_is_all_unavailable() -> None:
    profile = build_gravity_profile(
        "Test Player", "2025-26", shooting_table=None, mean_fga_per_game=None, mean_fta_per_game=None, games_played=0,
    )
    assert profile.mechanisms_present == []
    for metrics in profile.components.values():
        assert all(m.status == EvidenceStatus.UNAVAILABLE.value for m in metrics.values())


def _fixture_shooting_table_no_threes_no_hooks() -> SeasonShootingTable:
    """A real player profile that never shoots 3s and never takes a hook
    (e.g. a traditional rim-only big) -- confirmed real case: Ivica
    Zubac's real 2025-26 shooting table has no "3-pt" row at all, and
    summed real zone FGA matches season_fga exactly."""
    return SeasonShootingTable(
        player_slug="rimonly01",
        season_end_year="2026",
        zones=[
            ZoneShootingRow("At Rim", fg=150, fga=222, fg_pct=0.676, fg_assisted=90, fg_assisted_pct=0.6),
            ZoneShootingRow("3 to <10 ft", fg=90, fga=221, fg_pct=0.407, fg_assisted=140, fg_assisted_pct=0.75),
            ZoneShootingRow("10 to <16 ft", fg=15, fga=39, fg_pct=0.385, fg_assisted=30, fg_assisted_pct=0.9),
            # No "16 ft to <3-pt" and no "3-pt" row -- real 0 attempts.
        ],
        shot_types=[
            ShotTypeRow("DUNK", fg=80, fga=94, fg_pct=0.851, fg_assisted=50, fg_assisted_pct=0.5),
            # No HOOK_SHOT row -- real 0 attempts.
        ],
        season_fga=482,  # 222 + 221 + 39, real: zone totals sum exactly to season_fga
        season_fg_assisted_pct=0.68,
        url="https://example.invalid/fixture-rim-only",
    )


def test_gravity_profile_missing_zone_is_real_zero_not_null() -> None:
    """A zone/shot-type entirely absent from the real table (e.g. no
    real 3-point attempts this season) must report a real 0, not a
    missing/null value -- the absence itself is the real observation."""
    profile = build_gravity_profile(
        "Rim Only Player", "2025-26", shooting_table=_fixture_shooting_table_no_threes_no_hooks(),
        mean_fga_per_game=10.0, mean_fta_per_game=3.0, games_played=50,
    )
    pop = profile.components["POP_GRAVITY"]
    assert pop["three_pa_season"].value == 0
    assert pop["three_pa_season"].status == EvidenceStatus.OBSERVED.value
    assert pop["three_pa_share_of_fga"].value == pytest.approx(0.0)
    # Real 0 attempts must still count POP_GRAVITY as present -- the
    # mechanism has a real, computed value, it just happens to be zero.
    assert "POP_GRAVITY" in profile.mechanisms_present


def test_gravity_profile_zero_hook_attempts_still_yields_computable_post_index() -> None:
    """The most common real case in the league (a player who never
    takes a hook shot) must not null out the entire post-scoring-gravity
    index -- the hook term should simply contribute zero."""
    profile = build_gravity_profile(
        "Rim Only Player", "2025-26", shooting_table=_fixture_shooting_table_no_threes_no_hooks(),
        mean_fga_per_game=10.0, mean_fta_per_game=3.0, games_played=50,
    )
    post = profile.components["POST_SCORING_GRAVITY"]
    assert post["hook_shot_attempts_season"].value == 0
    assert post["post_scoring_gravity_index"].value is not None
    assert post["post_scoring_gravity_index"].status == EvidenceStatus.RECONSTRUCTED.value


# ---------------------------------------------------------------------
# Expected pass value (spec section 13)
# ---------------------------------------------------------------------

def _fixture_league_baseline() -> LeagueShootingBaseline:
    return LeagueShootingBaseline(
        season_end_year="2026",
        fg_pct_rim=0.65, fg_pct_short_paint=0.40, fg_pct_midrange=0.42,
        fg_pct_long_midrange=0.42, fg_pct_three=0.36,
        freq_rim=0.35, freq_short_paint=0.10, freq_midrange=0.15,
        freq_long_midrange=0.0, freq_three=0.40,
        url="https://example.invalid/fixture-league",
    )


def test_pass_value_model_rim_is_above_baseline_and_midrange_below() -> None:
    model = build_pass_value_model(_fixture_league_baseline(), season="2025-26")
    rim_added = model.added_pass_value_by_zone["RIM"].value
    midrange_added = model.added_pass_value_by_zone["MIDRANGE"].value
    assert rim_added > 0
    assert midrange_added < 0


def test_pass_value_model_none_baseline_is_all_unavailable() -> None:
    model = build_pass_value_model(None, season="2025-26")
    assert model.baseline_expected_points.status == EvidenceStatus.UNAVAILABLE.value
    assert all(m.status == EvidenceStatus.UNAVAILABLE.value for m in model.expected_points_by_zone.values())


# ---------------------------------------------------------------------
# Reconciliation + JSON hygiene over the real generated player artifacts
# (spec section 36; reads already-built JSON off disk, no network)
# ---------------------------------------------------------------------

def _find_no_nan_or_inf(node, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            bad += _find_no_nan_or_inf(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            bad += _find_no_nan_or_inf(value, f"{path}[{i}]")
    elif isinstance(node, float):
        if math.isnan(node) or math.isinf(node):
            bad.append(path)
    return bad


def _generated_player_artifact_paths() -> list[Path]:
    if not OUTPUT_ROOT.is_dir():
        return []
    return [p for p in sorted(OUTPUT_ROOT.glob("*.json")) if p.name != "players.json"]


@pytest.mark.skipif(not _generated_player_artifact_paths(), reason="no generated advantage-routing player artifacts on disk")
@pytest.mark.parametrize("artifact_path", _generated_player_artifact_paths(), ids=lambda p: p.stem)
def test_generated_player_artifact_reconciles_cleanly(artifact_path: Path) -> None:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    report = reconcile_player_artifact(artifact)
    failures = [c for c in report.checks if c.status == "FAIL"]
    assert not failures, f"{artifact_path.name}: {failures}"


@pytest.mark.skipif(not _generated_player_artifact_paths(), reason="no generated advantage-routing player artifacts on disk")
@pytest.mark.parametrize("artifact_path", _generated_player_artifact_paths(), ids=lambda p: p.stem)
def test_generated_player_artifact_has_no_nan_or_infinite_values(artifact_path: Path) -> None:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert _find_no_nan_or_inf(artifact) == []


@pytest.mark.skipif(not _generated_player_artifact_paths(), reason="no generated advantage-routing player artifacts on disk")
def test_validate_all_reports_all_pass() -> None:
    reports = validate_all(OUTPUT_ROOT)
    assert reports, "expected at least one generated player artifact to validate"
    for report in reports:
        assert report.all_passed, f"{report.player_name}: {[c for c in report.checks if c.status == 'FAIL']}"


# ---------------------------------------------------------------------
# Season game-list Inactive/Did-Not-Play filtering (real bug: a naive
# "every /boxscores/ link on the page" extraction silently included
# games a player never appeared in, corrupting the "most recent N real
# games" sample for anyone with a real recent injury absence -- e.g.
# real 2025-26 Domantas Sabonis: 83 raw gamelog rows, but only 22 were
# real appearances, the rest Inactive/Did Not Play placeholder rows
# that still link to that game's real boxscore.)
# ---------------------------------------------------------------------

_FIXTURE_GAMELOG_HTML = """
<div id="div_player_game_log_reg">
<table id="player_game_log_reg">
<tbody>
<tr class="partial_table"><td data-stat="date"><a href="/boxscores/202510220PHO.html">2025-10-22</a></td>
<td data-stat="is_starter" colspan="26">Inactive</td></tr>
<tr id="player_game_log_reg.1"><td data-stat="date"><a href="/boxscores/202510240SAC.html">2025-10-24</a></td>
<td data-stat="pts">18</td><td data-stat="ast">7</td></tr>
<tr class="partial_table"><td data-stat="date"><a href="/boxscores/202510260SAC.html">2025-10-26</a></td>
<td data-stat="is_starter" colspan="26">Did Not Play</td></tr>
<tr id="player_game_log_reg.2"><td data-stat="date"><a href="/boxscores/202510280OKC.html">2025-10-28</a></td>
<td data-stat="pts">21</td><td data-stat="ast">9</td></tr>
</tbody>
</table>
</div>
"""


def test_fetch_season_game_ids_excludes_inactive_and_dnp_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bball_ref_module, "_get", lambda url, *, cache_key: _FIXTURE_GAMELOG_HTML)
    game_ids = bball_ref_module.fetch_season_game_ids("testpl01", "2026")
    assert game_ids == ["202510240SAC", "202510280OKC"]
    assert "202510220PHO" not in game_ids  # Inactive row
    assert "202510260SAC" not in game_ids  # Did Not Play row


# ---------------------------------------------------------------------
# Player-search name resolution across diacritics (real bug: bball-ref's
# search-result label often carries the player's real native-spelling
# diacritics -- e.g. real "Alperen Şengün" -- even when the caller's own
# player list uses the plain-ASCII form matching this repo's
# Player-Predictor box-score naming, so a literal substring match against
# the raw label silently failed and resolve_player_slug returned None.)
# ---------------------------------------------------------------------

_FIXTURE_SEARCH_HTML = """
<div id="players">
<div class="search-item"><div class="search-item-name">
<a href="/players/s/sengual01.html">Alperen Şengün (2022-2026)</a>
</div></div>
<div class="search-item"><div class="search-item-name">
<a href="/players/d/doncilu01.html">Luka Dončić (2018-2026)</a>
</div></div>
</div>
"""


def test_ascii_fold_strips_diacritics() -> None:
    assert bball_ref_module._ascii_fold("Şengün") == "Sengun"
    assert bball_ref_module._ascii_fold("Dončić") == "Doncic"
    assert bball_ref_module._ascii_fold("Plain Name") == "Plain Name"


def test_resolve_player_slug_matches_across_diacritics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bball_ref_module, "_get", lambda url, *, cache_key: _FIXTURE_SEARCH_HTML)
    assert bball_ref_module.resolve_player_slug("Alperen Sengun") == "sengual01"
    assert bball_ref_module.resolve_player_slug("Luka Doncic") == "doncilu01"
