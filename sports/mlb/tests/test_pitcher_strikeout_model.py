from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import pitcher_strikeout_model as model  # noqa: E402


def test_season_stats_computes_real_rate_properties():
    stats = model.PitcherStrikeoutSeasonStats(pitcher_id=1, name="Test Pitcher", games_started=15, games_pitched=15, outs=259, strikeouts=79)
    assert abs(stats.innings_pitched - 86.333333) < 1e-4
    assert abs(stats.strikeouts_per_9 - 8.235521) < 1e-3
    assert abs(stats.innings_per_start - 5.755556) < 1e-3
    assert stats.has_real_sample is True


def test_projected_mean_strikeouts_is_ip_per_start_times_own_k_rate():
    stats = model.PitcherStrikeoutSeasonStats(pitcher_id=1, name="Test Pitcher", games_started=15, games_pitched=15, outs=259, strikeouts=79)
    projected = stats.projected_mean_strikeouts
    expected = stats.innings_per_start * (stats.strikeouts_per_9 / 9.0)
    assert abs(projected - expected) < 1e-9


def test_projected_mean_strikeouts_none_below_real_minimum_sample():
    stats = model.PitcherStrikeoutSeasonStats(pitcher_id=1, name="Rookie", games_started=2, games_pitched=2, outs=30, strikeouts=8)
    assert stats.has_real_sample is False
    assert stats.projected_mean_strikeouts is None


def test_season_stats_handles_zero_outs_without_crashing():
    stats = model.PitcherStrikeoutSeasonStats(pitcher_id=1, name="No Innings", games_started=0, games_pitched=0, outs=0, strikeouts=0)
    assert stats.strikeouts_per_9 is None
    assert stats.innings_per_start is None
    assert stats.projected_mean_strikeouts is None


def _fake_season_payload(games_started: int, outs: int, strikeouts: int, games_pitched: int | None = None) -> dict:
    return {
        "stats": [
            {
                "splits": [
                    {
                        "season": "2026",
                        "stat": {
                            "gamesStarted": games_started,
                            "gamesPitched": games_pitched if games_pitched is not None else games_started,
                            "outs": outs, "strikeOuts": strikeouts,
                        },
                    }
                ]
            }
        ]
    }


def test_fetch_pitcher_season_stats_parses_real_response_shape():
    stats = model.fetch_pitcher_season_stats(
        608331, 2026, name="Test Pitcher",
        fetch_json=lambda url: _fake_season_payload(15, 259, 79),
    )
    assert stats is not None
    assert stats.pitcher_id == 608331
    assert stats.games_started == 15
    assert stats.strikeouts == 79


def test_has_real_sample_false_for_a_pitcher_who_also_pitched_in_relief_this_season():
    """A real swingman/demoted starter's season aggregate mixes relief
    innings into the same total -- gamesPitched > gamesStarted must
    block the per-start projection, never silently overstate it."""
    stats = model.fetch_pitcher_season_stats(
        685299, 2026, name="Real Mixed-Role Pitcher",
        fetch_json=lambda url: _fake_season_payload(6, 230, 64, games_pitched=18),
    )
    assert stats is not None
    assert stats.is_pure_starter_this_season is False
    assert stats.has_real_sample is False
    assert stats.projected_mean_strikeouts is None


def test_fetch_pitcher_season_stats_returns_none_on_fetch_failure():
    def failing_fetch(url: str):
        raise OSError("network down")

    assert model.fetch_pitcher_season_stats(1, 2026, fetch_json=failing_fetch) is None


def test_fetch_pitcher_season_stats_returns_none_when_no_splits():
    assert model.fetch_pitcher_season_stats(1, 2026, fetch_json=lambda url: {"stats": [{"splits": []}]}) is None
    assert model.fetch_pitcher_season_stats(1, 2026, fetch_json=lambda url: {"stats": []}) is None


def test_poisson_over_probability_matches_hand_computed_values():
    # mean=5, line=5.5 -> P(X >= 6) for a Poisson(5)
    prob = model.poisson_over_probability(5.5, 5.0)
    assert abs(prob - 0.384039) < 1e-4


def test_poisson_over_probability_higher_mean_gives_higher_probability():
    low = model.poisson_over_probability(5.5, 4.0)
    high = model.poisson_over_probability(5.5, 7.0)
    assert high > low


def test_poisson_over_probability_bounds_between_zero_and_one():
    assert 0.0 <= model.poisson_over_probability(0.5, 0.001) <= 1.0
    assert 0.0 <= model.poisson_over_probability(20.5, 5.0) <= 1.0


def test_poisson_over_probability_none_for_negative_mean():
    assert model.poisson_over_probability(5.5, -1.0) is None
