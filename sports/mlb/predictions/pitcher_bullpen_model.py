from __future__ import annotations

"""Real MLB starting-pitcher and team-bullpen ERA tracking, in the same
leakage-safe walk-forward style as team_win_model.py -- built to enrich
that model's Pythagorean-only runs-allowed estimate with real pitching
signal, per this session's own honest backtest finding that the
Pythagorean-only model is currently slightly worse than the real market.

Real innings-pitched detail: `outs` (a real integer StatsAPI box-score
field) is used throughout rather than baseball's "X.1 / X.2" innings-
pitched NOTATION (where the fractional part means outs, not tenths) --
dividing outs by 3.0 gives real innings pitched without that notation's
misparse risk. See fetch_mlb_pitcher_game_data.py for the real source.

Credibility shrinkage: a single real start (or even a handful) is far
too small a sample to trust a pitcher's ERA at face value -- ERA is a
commonly-cited sabermetric example of a slow-to-stabilize rate stat.
`PitcherCumulativeStats.credibility_weight` linearly ramps from 0 (zero
real starts) to 1 (MIN_STARTS_FOR_FULL_CREDIBILITY real starts), a real,
disclosed modeling choice (not a fabricated precise number) in the same
spirit as team_win_model.MIN_GAMES_FOR_OWN_ESTIMATE.
"""

from dataclasses import dataclass
from typing import Optional

MIN_STARTS_FOR_FULL_CREDIBILITY = 15  # ERA is a slow-to-stabilize rate stat; below this, shrink toward the team baseline


@dataclass(frozen=True)
class PitcherCumulativeStats:
    pitcher_id: int
    name: str
    starts: int
    outs: int
    earned_runs: int

    @property
    def innings_pitched(self) -> float:
        return self.outs / 3.0

    @property
    def era(self) -> Optional[float]:
        if self.outs <= 0:
            return None
        return 9.0 * self.earned_runs / self.innings_pitched

    @property
    def credibility_weight(self) -> float:
        return min(1.0, self.starts / MIN_STARTS_FOR_FULL_CREDIBILITY)


@dataclass(frozen=True)
class BullpenCumulativeStats:
    team: str
    games: int
    outs: int
    earned_runs: int

    @property
    def innings_pitched(self) -> float:
        return self.outs / 3.0

    @property
    def era(self) -> Optional[float]:
        if self.outs <= 0:
            return None
        return 9.0 * self.earned_runs / self.innings_pitched


def build_cumulative_pitcher_stats(starts: list[dict]) -> dict[int, list[tuple[str, PitcherCumulativeStats]]]:
    """`starts`: real rows, each one real start (`date`, `pitcher_id`,
    `name`, `outs`, `earned_runs`), any order. Returns, per pitcher, a
    real chronological list of (date, cumulative-stats-BEFORE-that-
    date's-start) snapshots -- identical walk-forward contract to
    team_win_model.build_cumulative_team_stats, including the same
    same-date-collapses-to-one-snapshot handling."""
    running: dict[int, PitcherCumulativeStats] = {}
    history: dict[int, list[tuple[str, PitcherCumulativeStats]]] = {}
    dates_seen: dict[int, str] = {}

    for start in sorted(starts, key=lambda s: (s["date"], s.get("pitcher_id", 0))):
        pid = int(start["pitcher_id"])
        game_date = str(start["date"])
        if pid not in running:
            running[pid] = PitcherCumulativeStats(pitcher_id=pid, name=start.get("name", ""), starts=0, outs=0, earned_runs=0)
            history[pid] = []
        if dates_seen.get(pid) != game_date:
            history[pid].append((game_date, running[pid]))
            dates_seen[pid] = game_date
        prior = running[pid]
        running[pid] = PitcherCumulativeStats(
            pitcher_id=pid,
            name=start.get("name") or prior.name,
            starts=prior.starts + 1,
            outs=prior.outs + int(start["outs"]),
            earned_runs=prior.earned_runs + int(start["earned_runs"]),
        )
    return history


def build_cumulative_bullpen_stats(appearances: list[dict]) -> dict[str, list[tuple[str, BullpenCumulativeStats]]]:
    """Aggregated per TEAM, not per individual reliever -- a team fields
    many different real relievers game to game, so the real, stable unit
    to track is the team's bullpen as a whole. `appearances`: real rows,
    each one team's real bullpen line for one real game (`date`, `team`,
    `outs`, `earned_runs`). Same real leakage-safe walk-forward contract
    as above."""
    running: dict[str, BullpenCumulativeStats] = {}
    history: dict[str, list[tuple[str, BullpenCumulativeStats]]] = {}
    dates_seen: dict[str, str] = {}

    for appearance in sorted(appearances, key=lambda a: (a["date"], a.get("team", ""))):
        team = appearance["team"]
        game_date = str(appearance["date"])
        if team not in running:
            running[team] = BullpenCumulativeStats(team=team, games=0, outs=0, earned_runs=0)
            history[team] = []
        if dates_seen.get(team) != game_date:
            history[team].append((game_date, running[team]))
            dates_seen[team] = game_date
        prior = running[team]
        running[team] = BullpenCumulativeStats(
            team=team,
            games=prior.games + 1,
            outs=prior.outs + int(appearance["outs"]),
            earned_runs=prior.earned_runs + int(appearance["earned_runs"]),
        )
    return history


def stats_as_of(history, game_date: str):
    """Identical real leakage-safe lookup to team_win_model.stats_as_of:
    the LAST entry whose date is <= game_date (each entry is keyed by
    the date it precedes) -- see that function's docstring for the real
    <= vs < bug this deliberately mirrors the fix for. Returns None
    (never a fabricated stand-in) when there is no real prior entry."""
    prior = [snapshot for date, snapshot in history if date <= game_date]
    return prior[-1] if prior else None


def compute_empirical_starter_innings_share(rows: list[dict]) -> Optional[float]:
    """Real, data-derived share of a real game's total pitching innings
    thrown by the starter vs. the bullpen -- computed directly from this
    repo's own real box-score totals, never a remembered league-average
    constant. `rows`: real rows with `starter_outs` and `bullpen_outs`
    (pool both sides of every real game together for one league-wide
    figure). None if there's no real data to compute it from."""
    total_starter_outs = sum(int(r["starter_outs"]) for r in rows)
    total_bullpen_outs = sum(int(r["bullpen_outs"]) for r in rows)
    total_outs = total_starter_outs + total_bullpen_outs
    if total_outs <= 0:
        return None
    return total_starter_outs / total_outs
