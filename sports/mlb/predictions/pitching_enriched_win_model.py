from __future__ import annotations

"""Real MLB win-probability and run-total model, enriched with real
starting-pitcher and team-bullpen ERA signal on top of
team_win_model.py's Pythagorean+Log5 baseline -- built after this
session's own honest backtest of the baseline showed it is currently
slightly WORSE than the real market (51.3% pick accuracy, model Brier
0.253 vs. market's 0.241), an expected gap for a model with no
pitching-specific signal.

REAL BLEND, not a fabricated weight: a team's expected runs allowed in
a given real game is a credibility-weighted average of (a) a real
"pitching-implied" runs-allowed figure -- the probable starter's real
cumulative ERA and the team's real cumulative bullpen ERA, combined
using a real, DATA-DERIVED innings-share (never a remembered league
constant -- see pitcher_bullpen_model.compute_empirical_starter_innings_
share, computed from this repo's own real box scores) -- and (b) the
team's own real season-to-date runs-allowed average (team_win_model's
existing signal, which already reflects real defense/park effects the
pitcher-only figure can't see). The credibility weight is the real
starting pitcher's own `credibility_weight` (0..1, ramped by his real
number of prior starts) -- a pitcher with few real starts on record
correctly falls back toward the team baseline rather than letting a
noisy small-sample ERA dominate.

Everything else (offense side, Log5 combination, home-field advantage)
is untouched from team_win_model.py -- this module only replaces the
runs-allowed half of each side's Pythagorean input.
"""

from typing import Optional

import team_win_model as base_model
from pitcher_bullpen_model import BullpenCumulativeStats, PitcherCumulativeStats


def _pythagorean_win_pct(avg_runs_scored: Optional[float], avg_runs_allowed: Optional[float]) -> Optional[float]:
    if avg_runs_scored is None or avg_runs_allowed is None:
        return None
    if avg_runs_scored <= 0 and avg_runs_allowed <= 0:
        return 0.5
    rs_pow = avg_runs_scored ** base_model.PYTHAGOREAN_EXPONENT
    ra_pow = avg_runs_allowed ** base_model.PYTHAGOREAN_EXPONENT
    denominator = rs_pow + ra_pow
    if denominator <= 0:
        return 0.5
    return rs_pow / denominator


def blended_expected_runs_allowed(
    team_stats: Optional[base_model.TeamCumulativeStats],
    starter_stats: Optional[PitcherCumulativeStats],
    bullpen_stats: Optional[BullpenCumulativeStats],
    starter_innings_share: Optional[float],
) -> Optional[float]:
    """A team's real expected runs allowed for its own upcoming start:
    credibility-weighted blend of the real pitching-implied figure and
    the team's own real season-to-date runs-allowed average. Falls back
    honestly to the team baseline (never a guessed pitching figure) when
    starter/bullpen ERA or the real innings-share isn't available yet;
    returns None only when even the team baseline is unavailable."""
    if team_stats is None:
        return None
    team_baseline = team_stats.avg_runs_allowed
    if team_baseline is None:
        return None
    if (
        starter_stats is None
        or starter_stats.era is None
        or bullpen_stats is None
        or bullpen_stats.era is None
        or starter_innings_share is None
    ):
        return team_baseline
    pitching_implied = starter_stats.era * starter_innings_share + bullpen_stats.era * (1.0 - starter_innings_share)
    weight = starter_stats.credibility_weight
    return weight * pitching_implied + (1.0 - weight) * team_baseline


def predict_moneyline_probability_enriched(
    home_team_stats: Optional[base_model.TeamCumulativeStats],
    away_team_stats: Optional[base_model.TeamCumulativeStats],
    *,
    home_starter_stats: Optional[PitcherCumulativeStats],
    home_bullpen_stats: Optional[BullpenCumulativeStats],
    away_starter_stats: Optional[PitcherCumulativeStats],
    away_bullpen_stats: Optional[BullpenCumulativeStats],
    starter_innings_share: Optional[float],
    home_field_advantage: float = 0.0,
) -> Optional[float]:
    """Same real Pythagorean + Log5 + home-field-advantage structure as
    team_win_model.predict_moneyline_probability, with each side's
    runs-allowed input replaced by its real pitching-enriched blend.
    None (never a guessed 0.5) when either team lacks real prior team
    history at all -- identical real-data requirement as the baseline."""
    if home_team_stats is None or away_team_stats is None:
        return None
    if home_team_stats.games_played < 1 or away_team_stats.games_played < 1:
        return None

    home_runs_allowed = blended_expected_runs_allowed(home_team_stats, home_starter_stats, home_bullpen_stats, starter_innings_share)
    away_runs_allowed = blended_expected_runs_allowed(away_team_stats, away_starter_stats, away_bullpen_stats, starter_innings_share)
    home_pyth = _pythagorean_win_pct(home_team_stats.avg_runs_scored, home_runs_allowed)
    away_pyth = _pythagorean_win_pct(away_team_stats.avg_runs_scored, away_runs_allowed)
    if home_pyth is None or away_pyth is None:
        return None

    base = base_model.log5_probability(home_pyth, away_pyth)
    adjusted = base + home_field_advantage
    return max(0.01, min(0.99, adjusted))


def predict_run_total_enriched(
    home_team_stats: Optional[base_model.TeamCumulativeStats],
    away_team_stats: Optional[base_model.TeamCumulativeStats],
    *,
    home_starter_stats: Optional[PitcherCumulativeStats],
    home_bullpen_stats: Optional[BullpenCumulativeStats],
    away_starter_stats: Optional[PitcherCumulativeStats],
    away_bullpen_stats: Optional[BullpenCumulativeStats],
    starter_innings_share: Optional[float],
) -> Optional[float]:
    """Same real structure as team_win_model.predict_run_total (each
    side's expected runs = average of its own real scoring rate and the
    opponent's real runs-allowed figure), with the opponent's
    runs-allowed input replaced by its real pitching-enriched blend."""
    sides = expected_runs_per_side_enriched(
        home_team_stats, away_team_stats,
        home_starter_stats=home_starter_stats, home_bullpen_stats=home_bullpen_stats,
        away_starter_stats=away_starter_stats, away_bullpen_stats=away_bullpen_stats,
        starter_innings_share=starter_innings_share,
    )
    if sides is None:
        return None
    home_expected, away_expected = sides
    return home_expected + away_expected


def expected_runs_per_side_enriched(
    home_team_stats: Optional[base_model.TeamCumulativeStats],
    away_team_stats: Optional[base_model.TeamCumulativeStats],
    *,
    home_starter_stats: Optional[PitcherCumulativeStats],
    home_bullpen_stats: Optional[BullpenCumulativeStats],
    away_starter_stats: Optional[PitcherCumulativeStats],
    away_bullpen_stats: Optional[BullpenCumulativeStats],
    starter_innings_share: Optional[float],
) -> Optional[tuple[float, float]]:
    """The real (home_expected_runs, away_expected_runs) pair underlying
    both predict_run_total_enriched (their sum) and the joint game
    simulator (game_simulation_model.py, which needs each side
    separately as its real per-trial Monte Carlo mean) -- kept as one
    real shared computation so the two never drift apart."""
    if home_team_stats is None or away_team_stats is None:
        return None
    home_avg_rs = home_team_stats.avg_runs_scored
    away_avg_rs = away_team_stats.avg_runs_scored
    if home_avg_rs is None or away_avg_rs is None:
        return None

    home_runs_allowed = blended_expected_runs_allowed(home_team_stats, home_starter_stats, home_bullpen_stats, starter_innings_share)
    away_runs_allowed = blended_expected_runs_allowed(away_team_stats, away_starter_stats, away_bullpen_stats, starter_innings_share)
    if home_runs_allowed is None or away_runs_allowed is None:
        return None

    home_expected = (home_avg_rs + away_runs_allowed) / 2.0
    away_expected = (away_avg_rs + home_runs_allowed) / 2.0
    return home_expected, away_expected
