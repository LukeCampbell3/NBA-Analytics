from __future__ import annotations

import numpy as np

from .trajectory import TrajectoryBatch


SUPPORTED_WORLD_QUERIES = {"moneyline", "game_total", "first_5_innings_total", "team_total"}


def query_mask(batch: TrajectoryBatch, *, market_type: str, side: str, line: float | None = None, team: str | None = None) -> np.ndarray:
    side = side.lower()
    if market_type == "moneyline":
        if side == "home": return batch.home_runs > batch.away_runs
        if side == "away": return batch.away_runs > batch.home_runs
        raise ValueError("moneyline side must be home or away")
    if line is None:
        raise ValueError("line is required")
    if market_type == "game_total":
        values = batch.home_runs + batch.away_runs
    elif market_type == "first_5_innings_total":
        values = batch.home_runs_by_inning[:, :5].sum(axis=1) + batch.away_runs_by_inning[:, :5].sum(axis=1)
    elif market_type == "team_total":
        values = batch.home_runs if team == "home" else batch.away_runs if team == "away" else None
        if values is None: raise ValueError("team_total requires home/away team")
    else:
        raise ValueError(f"UNSUPPORTED_WORLD_QUERY:{market_type}")
    if side == "over": return values > line
    if side == "under": return values < line
    raise ValueError("total side must be over or under")


def event_market_query(*, market_type: str, event_identity: str | None, event_model_available: bool) -> None:
    if not event_model_available:
        raise ValueError("EVENT_MODEL_REQUIRED")
    if not event_identity:
        raise ValueError("EVENT_IDENTITY_UNAVAILABLE")
