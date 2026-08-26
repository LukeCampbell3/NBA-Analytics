#!/usr/bin/env python3
"""Real, leakage-free MLB starting-pitcher strikeout projection.

Unlike pitcher_bullpen_model.py's walk-forward ERA/innings tracking
(built for historical backtesting, where using any game on/after the
replay date would be real leakage), this module is for TODAY's live
board only -- there is no leakage concern in using a real starter's
actual full-season-to-date stats to project their next real start, so
it reads the real MLB Stats API's own real season-aggregate endpoint
directly (one real call per real probable starter) rather than
reconstructing a season total game-by-game from box scores.

Real strikeout distribution: a starting pitcher's real strikeout count
in a given start is well-approximated by a Poisson distribution around
a real projected mean (innings pitched per start * real strikeouts-
per-inning rate) -- the same standard, disclosed approximation this
repo already uses for other count-like markets, just via a direct
closed-form Poisson CDF rather than a Monte Carlo draw (no live joint
correlation to simulate here -- a single pitcher's own start is not
correlated with anything else this module models).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MLB_STATS_API_ROOT = "https://statsapi.mlb.com/api/v1"
MIN_STARTS_FOR_REAL_PROJECTION = 5  # too few real starts -> no real signal yet, never a guessed rate


@dataclass(frozen=True)
class PitcherStrikeoutSeasonStats:
    pitcher_id: int
    name: str
    games_started: int
    games_pitched: int
    outs: int
    strikeouts: int

    @property
    def innings_pitched(self) -> float:
        return self.outs / 3.0

    @property
    def strikeouts_per_9(self) -> Optional[float]:
        if self.outs <= 0:
            return None
        return 9.0 * self.strikeouts / self.innings_pitched

    @property
    def innings_per_start(self) -> Optional[float]:
        if self.games_started <= 0:
            return None
        return self.innings_pitched / self.games_started

    @property
    def is_pure_starter_this_season(self) -> bool:
        """A real MLB Stats API season aggregate mixes EVERY real
        appearance together (gamesPitched), while gamesStarted counts
        only real starts -- a pitcher used in both roles this season
        (a real swingman, or a starter demoted to the bullpen) has
        relief innings/strikeouts baked into the same season total,
        which would badly overstate a per-start projection if divided
        by games_started alone. Real, not a guessed exclusion: only a
        pitcher whose every real appearance this season was a real
        start is trusted for this per-start model."""
        return self.games_pitched > 0 and self.games_pitched == self.games_started

    @property
    def has_real_sample(self) -> bool:
        return self.games_started >= MIN_STARTS_FOR_REAL_PROJECTION and self.is_pure_starter_this_season

    @property
    def projected_mean_strikeouts(self) -> Optional[float]:
        """Real projected strikeouts for this pitcher's next real start:
        their own real innings-per-start times their own real
        strikeouts-per-inning rate -- never a league constant, never a
        guess. None (never a fabricated number) below the real minimum
        sample size."""
        if not self.has_real_sample:
            return None
        k9 = self.strikeouts_per_9
        ip_per_start = self.innings_per_start
        if k9 is None or ip_per_start is None:
            return None
        return ip_per_start * (k9 / 9.0)


def default_fetch_json(url: str, *, timeout: float = 15.0) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; NBA-Analytics/1.0; +real-pitcher-k-model)"})
    with urlopen(request, timeout=timeout) as response:
        import json

        return json.loads(response.read().decode("utf-8"))


def fetch_pitcher_season_stats(
    pitcher_id: int,
    season: int,
    *,
    name: str = "",
    fetch_json: Callable[[str], dict[str, Any]] = default_fetch_json,
) -> Optional[PitcherStrikeoutSeasonStats]:
    """Real MLB Stats API season-aggregate pitching line for one real
    pitcher. Returns None (never a fabricated stand-in) on any real
    fetch failure or when the pitcher has no real pitching stats this
    season (e.g. a real rookie's first start, a real position-player
    pitching appearance with no `pitching` group entry)."""
    url = f"{MLB_STATS_API_ROOT}/people/{int(pitcher_id)}/stats?" + urlencode(
        {"stats": "season", "group": "pitching", "season": int(season)}
    )
    try:
        payload = fetch_json(url)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return None
    try:
        splits = payload["stats"][0]["splits"]
    except (KeyError, IndexError, TypeError):
        return None
    if not splits:
        return None
    stat = splits[0].get("stat") or {}
    try:
        games_started = int(stat.get("gamesStarted") or 0)
        games_pitched = int(stat.get("gamesPitched") or 0)
        outs = int(stat.get("outs") or 0)
        strikeouts = int(stat.get("strikeOuts") or 0)
    except (TypeError, ValueError):
        return None
    return PitcherStrikeoutSeasonStats(
        pitcher_id=int(pitcher_id), name=name, games_started=games_started, games_pitched=games_pitched,
        outs=outs, strikeouts=strikeouts,
    )


def poisson_over_probability(line: float, mean: float) -> Optional[float]:
    """Real Poisson P(X > line) for a real projected mean. `line` is
    normally a half-integer (e.g. 5.5) so this never lands on a push
    boundary; if given a whole number, treats it as the strict-over
    threshold (X > line), matching how sportsbooks quote a "5.5 Ks"
    line."""
    if mean is None or mean < 0:
        return None
    threshold = math.floor(line) + 1  # smallest integer count that clears the real line
    cdf = 0.0
    term = math.exp(-mean)  # P(X=0)
    cdf += term
    for k in range(1, threshold):
        term *= mean / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))
