from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlayerShare:
    player_id: str
    event_count: float
    opportunity_count: float


def smoothed_shares(rows: list[PlayerShare], *, prior_strength: float = 20.0) -> dict[str, float]:
    if not rows:
        raise ValueError("PLAYER_SHARE_DATA_REQUIRED")
    raw = np.array([max(row.event_count, 0.0) for row in rows], dtype=float)
    opportunities = np.array([max(row.opportunity_count, 0.0) for row in rows], dtype=float)
    baseline = 1.0 / len(rows)
    estimates = (raw + prior_strength * baseline) / np.maximum(opportunities + prior_strength, 1e-12)
    total = estimates.sum()
    if total <= 0:
        raise ValueError("PLAYER_SHARE_MODEL_REQUIRED")
    normalized = estimates / total
    return {row.player_id: float(value) for row, value in zip(rows, normalized)}


def allocate_team_events(team_events: np.ndarray, shares: dict[str, float], *, seed: int) -> dict[str, np.ndarray]:
    if not shares:
        raise ValueError("PLAYER_SHARE_MODEL_REQUIRED")
    players = tuple(shares)
    probabilities = np.array([shares[player] for player in players], dtype=float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    result = {player: np.zeros(len(team_events), dtype=np.int16) for player in players}
    for index, events in enumerate(team_events.astype(int)):
        allocation = rng.multinomial(max(events, 0), probabilities)
        for player, count in zip(players, allocation):
            result[player][index] = count
    if not np.array_equal(sum(result.values(), np.zeros(len(team_events), dtype=np.int16)), team_events):
        raise AssertionError("team/player accounting invariant failed")
    return result


def conditional_probability(event_mask: np.ndarray, state_mask: np.ndarray) -> dict[str, float | None]:
    event = event_mask.astype(bool)
    state = state_mask.astype(bool)
    marginal = float(event.mean())
    support = int(state.sum())
    conditional = float(event[state].mean()) if support else None
    lift = None if conditional is None or marginal == 0 else conditional / marginal - 1.0
    return {"marginal_probability": marginal, "conditional_probability": conditional, "conditional_lift": lift, "support": support}
