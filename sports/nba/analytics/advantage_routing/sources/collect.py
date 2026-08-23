"""Orchestrates a real, bounded data collection pass for one player:
resolves their Basketball-Reference slug, fetches their real season
shooting-zone table, and fetches real play-by-play for a documented
SAMPLE of their most recent real games (not the full season -- see
``GAMES_SAMPLED_PER_PLAYER`` below) to reconstruct a real, if partial,
assist-based recipient network.

WHY A SAMPLE, NOT THE FULL SEASON: fetching every play-by-play page for
a 70+ game season, for multiple players, against a source that asks for
<= ~20 requests/minute, would take a very long time and carries real
rate-limiting risk. A bounded, clearly-documented sample keeps every
number this module produces genuinely real (never fabricated) while
keeping collection time and request volume reasonable. The exact sample
size and which games were used is recorded in the output and surfaces
in every player artifact's provenance -- this is a disclosed
approximation, not a hidden one. Re-running with a larger
GAMES_SAMPLED_PER_PLAYER (or None for the full season) is a one-line
change once a run has more time budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import bball_ref

GAMES_SAMPLED_PER_PLAYER = 25


@dataclass
class PlayerRealDataBundle:
    player_name: str
    season: str
    player_slug: Optional[str]
    shooting_table: Optional[bball_ref.SeasonShootingTable]
    games_sampled: list[str]
    games_available_total: int
    assists_as_passer: list[bball_ref.RealAssistEvent]
    turnovers: list[bball_ref.RealTurnoverEvent]

    @property
    def data_available(self) -> bool:
        return self.player_slug is not None


def collect_player_real_data(player_name: str, season: str = "2025-26", *, games_sampled: int = GAMES_SAMPLED_PER_PLAYER) -> PlayerRealDataBundle:
    season_end_year = str(int(season.split("-")[0]) + 1) if "-" in season else season
    slug = bball_ref.resolve_player_slug(player_name)
    if slug is None:
        return PlayerRealDataBundle(
            player_name=player_name, season=season, player_slug=None, shooting_table=None,
            games_sampled=[], games_available_total=0, assists_as_passer=[], turnovers=[],
        )

    shooting_table = bball_ref.fetch_season_shooting_table(slug, season_end_year)
    all_game_ids = bball_ref.fetch_season_game_ids(slug, season_end_year)
    # Most recent N real games, chronological order preserved for
    # reproducibility -- never a random/arbitrary subset.
    sampled_game_ids = all_game_ids[-games_sampled:] if games_sampled else all_game_ids

    assists_as_passer: list[bball_ref.RealAssistEvent] = []
    turnovers: list[bball_ref.RealTurnoverEvent] = []
    for game_id in sampled_game_ids:
        game_assists, game_turnovers = bball_ref.fetch_game_events(slug, game_id)
        assists_as_passer.extend(a for a in game_assists if a.passer_slug == slug)
        turnovers.extend(t for t in game_turnovers if t.player_slug == slug)

    return PlayerRealDataBundle(
        player_name=player_name, season=season, player_slug=slug, shooting_table=shooting_table,
        games_sampled=sampled_game_ids, games_available_total=len(all_game_ids),
        assists_as_passer=assists_as_passer, turnovers=turnovers,
    )
