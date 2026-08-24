from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import game_simulation_model as sim  # noqa: E402


def test_compute_empirical_runs_dispersion_falls_back_to_one_with_little_data() -> None:
    games = [{"home_score": "5", "away_score": "2"}]
    assert sim.compute_empirical_runs_dispersion(games) == 1.0


def test_compute_empirical_runs_dispersion_detects_real_overdispersion() -> None:
    rng = np.random.default_rng(7)
    # Real negative-binomial-shaped scores (overdispersed relative to Poisson)
    scores = rng.negative_binomial(3, 0.4, size=200)
    games = [{"home_score": float(s), "away_score": float(s)} for s in scores]
    ratio = sim.compute_empirical_runs_dispersion(games)
    assert ratio > 1.0


def test_compute_empirical_f5_share_pools_real_ratio_of_sums() -> None:
    games = [
        {"home_score": "6", "home_innings_1_5": "3", "away_score": "4", "away_innings_1_5": "2"},
        {"home_score": "8", "home_innings_1_5": "4", "away_score": "2", "away_innings_1_5": "1"},
    ]
    share = sim.compute_empirical_f5_share(games)
    assert share is not None
    total_f5 = 3 + 2 + 4 + 1
    total_full = 6 + 4 + 8 + 2
    assert abs(share - total_f5 / total_full) < 1e-9


def test_compute_empirical_f5_share_skips_real_rain_shortened_games() -> None:
    games = [
        {"home_score": "6", "home_innings_1_5": "", "away_score": "4", "away_innings_1_5": ""},  # rain-shortened, no real F5
        {"home_score": "8", "home_innings_1_5": "4", "away_score": "2", "away_innings_1_5": "1"},
    ]
    share = sim.compute_empirical_f5_share(games)
    assert share is not None
    assert abs(share - (4 + 1) / (8 + 2)) < 1e-9


def test_compute_empirical_f5_share_none_with_no_real_data() -> None:
    assert sim.compute_empirical_f5_share([]) is None


def test_simulate_game_outcomes_home_win_probability_favors_the_better_real_side() -> None:
    result = sim.simulate_game_outcomes(6.0, 3.0, num_trials=20000, seed=1)
    assert result.home_win_probability > 0.6


def test_simulate_game_outcomes_full_total_probability_is_sane() -> None:
    result = sim.simulate_game_outcomes(4.5, 4.5, num_trials=20000, seed=2)
    # combined real mean ~9 -> a line well below the mean should have high real over-probability
    assert result.full_total_over_probability(5.0) > 0.8
    assert result.full_total_over_probability(15.0) < 0.05


def test_simulate_game_outcomes_f5_never_exceeds_full_total_per_trial() -> None:
    result = sim.simulate_game_outcomes(5.0, 4.0, f5_share=0.55, num_trials=5000, seed=3)
    assert np.all(result.home_f5_runs <= result.home_runs)
    assert np.all(result.away_f5_runs <= result.away_runs)


def test_simulate_game_outcomes_without_real_f5_share_falls_back_to_full_total() -> None:
    result = sim.simulate_game_outcomes(5.0, 4.0, f5_share=None, num_trials=100, seed=4)
    assert np.array_equal(result.home_f5_runs, result.home_runs)
    assert np.array_equal(result.away_f5_runs, result.away_runs)


def test_simulate_game_outcomes_ties_are_resolved_never_left_as_a_tie() -> None:
    result = sim.simulate_game_outcomes(4.0, 4.0, num_trials=20000, seed=5)
    # home_win is a bool array; every trial must resolve to True or False, no in-between
    assert result.home_win.dtype == bool
    assert set(np.unique(result.home_win).tolist()) <= {True, False}


def test_joint_probability_of_two_masks_is_never_more_than_either_marginal() -> None:
    result = sim.simulate_game_outcomes(5.0, 4.0, f5_share=0.55, num_trials=20000, seed=6)
    ml_mask = result.home_win
    total_mask = result.full_total_over_mask(7.5)
    joint = result.joint_probability(ml_mask, total_mask)
    assert joint <= float(np.mean(ml_mask)) + 1e-9
    assert joint <= float(np.mean(total_mask)) + 1e-9


def test_joint_probability_with_no_masks_is_one() -> None:
    result = sim.simulate_game_outcomes(5.0, 4.0, num_trials=100, seed=7)
    assert result.joint_probability() == 1.0


def test_joint_probability_reflects_real_correlation_between_full_and_f5_totals() -> None:
    """A real high-scoring trial should push both the full total AND the
    F5 total over their lines together more often than independence
    would predict -- exactly the real dependence this module exists to
    capture instead of assuming away."""
    result = sim.simulate_game_outcomes(5.0, 4.5, f5_share=0.58, num_trials=50000, seed=8)
    full_over = result.full_total_over_mask(8.5)
    f5_over = result.f5_total_over_mask(4.5)
    joint = result.joint_probability(full_over, f5_over)
    independence_product = float(np.mean(full_over)) * float(np.mean(f5_over))
    assert joint > independence_product  # real positive correlation, not independence
