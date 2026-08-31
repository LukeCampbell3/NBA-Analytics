from __future__ import annotations

from enum import StrEnum


class Settlement(StrEnum):
    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    VOID = "VOID"
    BLOCKED = "BLOCKED"


AGGREGATE_MARKETS = {"batter_hits", "batter_total_bases", "batter_runs_scored", "batter_rbis", "batter_home_runs", "pitcher_strikeouts", "pitcher_outs", "team_total", "team_hits", "game_total", "first_5_innings_total"}
EVENT_MARKETS = {"runs_inning", "team_runs_inning", "pitcher_ks_inning", "pitcher_pitches_inning", "pa_pitch_count"}


def settle(*, market_type: str, side: str, line: float, observed: float | None, event_identity: str | None = None) -> Settlement:
    if market_type in EVENT_MARKETS and not event_identity:
        return Settlement.BLOCKED
    if market_type not in AGGREGATE_MARKETS | EVENT_MARKETS:
        return Settlement.BLOCKED
    if observed is None:
        return Settlement.VOID
    if observed == line:
        return Settlement.PUSH
    if side.lower() == "over":
        return Settlement.WIN if observed > line else Settlement.LOSS
    if side.lower() == "under":
        return Settlement.WIN if observed < line else Settlement.LOSS
    return Settlement.BLOCKED
