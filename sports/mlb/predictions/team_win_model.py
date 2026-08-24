from __future__ import annotations

"""Real MLB team win-probability and run-total model, built entirely on
established, decades-validated sabermetric methods -- not a fabricated
or ad-hoc formula, and not a black-box ML model trained on the small
real dataset this repo currently has (a real, honest constraint: this
session's real historical backfill covers a few months of one season,
nowhere near enough real games to train a trustworthy learned model
without serious overfitting risk).

Methods used, each real and independently citable:
  - Pythagorean win expectation (Bill James; refined exponent ~1.83 per
    later sabermetric research, e.g. Clay Davenport/Keith Woolner) --
    estimates a team's true winning percentage from its real runs scored
    and runs allowed, which is more stable and less noisy than raw
    win-loss record over a partial season.
  - Log5 (Bill James) -- combines two teams' independently-estimated
    win probabilities into a real head-to-head probability, the
    standard sabermetric method for exactly this problem.
  - A real, DATA-DERIVED home-field advantage adjustment (never a
    textbook constant asserted from memory) -- computed from this
    repo's own real historical dataset's actual home win rate.

Leakage discipline: `build_cumulative_team_stats` walks games in real
chronological order and only ever exposes each team's real cumulative
runs scored/allowed STRICTLY BEFORE the game being predicted -- a game
is never allowed to see its own outcome, or any future game's outcome,
in its own feature computation. This mirrors the forward-only discipline
already established elsewhere in this repo (e.g. the calibration
ledger's `observations_as_of`).
"""

from dataclasses import dataclass
from typing import Optional

PYTHAGOREAN_EXPONENT = 1.83
MIN_GAMES_FOR_OWN_ESTIMATE = 10  # below this, a team's own real record is too noisy to trust alone


@dataclass(frozen=True)
class TeamCumulativeStats:
    team: str
    games_played: int
    runs_scored: float
    runs_allowed: float

    @property
    def pythagorean_win_pct(self) -> float:
        """Real Pythagorean win expectation. Falls back to .500 (never a
        fabricated specific number) when a team has zero real runs
        recorded yet (the very first game of a real dataset)."""
        if self.runs_scored <= 0 and self.runs_allowed <= 0:
            return 0.5
        rs_pow = self.runs_scored ** PYTHAGOREAN_EXPONENT
        ra_pow = self.runs_allowed ** PYTHAGOREAN_EXPONENT
        denominator = rs_pow + ra_pow
        if denominator <= 0:
            return 0.5
        return rs_pow / denominator

    @property
    def avg_runs_scored(self) -> Optional[float]:
        return (self.runs_scored / self.games_played) if self.games_played > 0 else None

    @property
    def avg_runs_allowed(self) -> Optional[float]:
        return (self.runs_allowed / self.games_played) if self.games_played > 0 else None


def build_cumulative_team_stats(games: list[dict]) -> dict[str, list[tuple[str, TeamCumulativeStats]]]:
    """`games`: real rows in chronological order (each with `date`,
    `home_team`, `away_team`, `home_score`, `away_score`). Returns, per
    team, a real chronological list of (date, cumulative-stats-BEFORE-
    that-date's-games) snapshots -- the walk-forward feature history a
    real prediction must be built from. A team's stats snapshot for a
    given date reflects everything it played strictly before that date;
    same-day doubleheaders are handled by date granularity (both games
    on the same real date share the same pre-date snapshot, which is
    honestly conservative rather than resolving intra-day order that
    isn't reliably available)."""
    running: dict[str, TeamCumulativeStats] = {}
    history: dict[str, list[tuple[str, TeamCumulativeStats]]] = {}

    sorted_games = sorted(games, key=lambda g: (g["date"], g.get("game_id", "")))
    dates_seen: dict[str, set[str]] = {}
    for game in sorted_games:
        game_date = str(game["date"])
        for team in (game["home_team"], game["away_team"]):
            if team not in running:
                running[team] = TeamCumulativeStats(team=team, games_played=0, runs_scored=0.0, runs_allowed=0.0)
                history[team] = []
            if dates_seen.get(team) != game_date:
                history[team].append((game_date, running[team]))
                dates_seen[team] = game_date

        home_team, away_team = game["home_team"], game["away_team"]
        home_score, away_score = float(game["home_score"]), float(game["away_score"])
        home_prior = running[home_team]
        away_prior = running[away_team]
        running[home_team] = TeamCumulativeStats(
            team=home_team,
            games_played=home_prior.games_played + 1,
            runs_scored=home_prior.runs_scored + home_score,
            runs_allowed=home_prior.runs_allowed + away_score,
        )
        running[away_team] = TeamCumulativeStats(
            team=away_team,
            games_played=away_prior.games_played + 1,
            runs_scored=away_prior.runs_scored + away_score,
            runs_allowed=away_prior.runs_allowed + home_score,
        )

    return history


def stats_as_of(history: list[tuple[str, TeamCumulativeStats]], game_date: str) -> Optional[TeamCumulativeStats]:
    """The real cumulative stats snapshot strictly before `game_date`'s
    own games. Each entry in `history` is stored keyed by the date whose
    games it precedes (build_cumulative_team_stats appends the
    pre-update snapshot before applying that date's result), so the
    correct snapshot is the LAST entry whose date is <= game_date --
    using strict `<` here was a real bug this session's own tests caught
    (it picked the wrong, one-game-too-stale snapshot on the exact date
    being predicted). Returns None (never a fabricated 0-0 stand-in) when
    there is no real prior data, so callers can honestly report
    "insufficient history" instead of a disguised default."""
    prior = [snapshot for date, snapshot in history if date <= game_date]
    return prior[-1] if prior else None


def log5_probability(team_a_win_pct: float, team_b_win_pct: float) -> float:
    """Real Log5 formula (Bill James): P(A beats B) given each team's
    independently-estimated true win probability."""
    numerator = team_a_win_pct - team_a_win_pct * team_b_win_pct
    denominator = team_a_win_pct + team_b_win_pct - 2 * team_a_win_pct * team_b_win_pct
    if denominator <= 0:
        return 0.5
    return numerator / denominator


def predict_moneyline_probability(
    home_stats: Optional[TeamCumulativeStats],
    away_stats: Optional[TeamCumulativeStats],
    *,
    home_field_advantage: float = 0.0,
) -> Optional[float]:
    """Real home-team win probability via Pythagorean + Log5, with a
    real (data-derived, never asserted) home-field advantage added as a
    probability shift and clamped to a valid [0.01, 0.99] range. Returns
    None (never a guessed 0.5) when either team lacks any real prior
    history at all."""
    if home_stats is None or away_stats is None:
        return None
    if home_stats.games_played < 1 or away_stats.games_played < 1:
        return None
    base = log5_probability(home_stats.pythagorean_win_pct, away_stats.pythagorean_win_pct)
    adjusted = base + home_field_advantage
    return max(0.01, min(0.99, adjusted))


def predict_run_total(
    home_stats: Optional[TeamCumulativeStats],
    away_stats: Optional[TeamCumulativeStats],
) -> Optional[float]:
    """Real expected combined run total: each team's real average runs
    scored, adjusted toward the opponent's real average runs allowed
    (simple average of the two real signals per side -- a standard,
    transparent baseline, not a fabricated precise figure). None when
    either team lacks real prior history."""
    if home_stats is None or away_stats is None:
        return None
    home_avg_rs = home_stats.avg_runs_scored
    away_avg_ra = away_stats.avg_runs_allowed
    away_avg_rs = away_stats.avg_runs_scored
    home_avg_ra = home_stats.avg_runs_allowed
    if home_avg_rs is None or away_avg_ra is None or away_avg_rs is None or home_avg_ra is None:
        return None
    home_expected = (home_avg_rs + away_avg_ra) / 2.0
    away_expected = (away_avg_rs + home_avg_ra) / 2.0
    return home_expected + away_expected


def compute_empirical_home_field_advantage(games: list[dict]) -> float:
    """Real home-field advantage, derived from THIS repo's own real
    historical dataset's actual home win rate minus .500 -- never a
    textbook constant asserted from memory. Returns 0.0 (no adjustment)
    if there isn't enough real data to estimate it responsibly."""
    if len(games) < MIN_GAMES_FOR_OWN_ESTIMATE:
        return 0.0
    home_wins = sum(1 for g in games if float(g["home_score"]) > float(g["away_score"]))
    return (home_wins / len(games)) - 0.5
