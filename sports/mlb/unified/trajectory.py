from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryBatch:
    """Compact shared-world results; no per-pitch Python objects."""

    home_runs_by_inning: np.ndarray
    away_runs_by_inning: np.ndarray
    player_hits: dict[str, np.ndarray] = field(default_factory=dict)
    player_runs: dict[str, np.ndarray] = field(default_factory=dict)
    home_player_ids: tuple[str, ...] = ()
    away_player_ids: tuple[str, ...] = ()
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def trials(self) -> int:
        return int(self.home_runs_by_inning.shape[0])

    @property
    def home_runs(self) -> np.ndarray:
        return self.home_runs_by_inning.sum(axis=1)

    @property
    def away_runs(self) -> np.ndarray:
        return self.away_runs_by_inning.sum(axis=1)

    def validate(self) -> None:
        if self.home_runs_by_inning.shape != self.away_runs_by_inning.shape:
            raise ValueError("home/away inning arrays must share a shape")
        if self.home_runs_by_inning.ndim != 2 or self.home_runs_by_inning.shape[1] < 5:
            raise ValueError("inning arrays must be trials x innings")
        if np.any(self.home_runs_by_inning < 0) or np.any(self.away_runs_by_inning < 0):
            raise ValueError("runs cannot be negative")
        for values in (*self.player_hits.values(), *self.player_runs.values(), *self.masks.values()):
            if len(values) != self.trials:
                raise ValueError("trajectory vectors must share trial count")
        if self.home_player_ids:
            summed = sum((self.player_runs[player] for player in self.home_player_ids), np.zeros(self.trials, dtype=int))
            if not np.array_equal(summed, self.home_runs):
                raise ValueError("home team runs must equal summed player runs")
        if self.away_player_ids:
            summed = sum((self.player_runs[player] for player in self.away_player_ids), np.zeros(self.trials, dtype=int))
            if not np.array_equal(summed, self.away_runs):
                raise ValueError("away team runs must equal summed player runs")

    def probability(self, mask_reference: str) -> float:
        if mask_reference not in self.masks:
            raise KeyError(mask_reference)
        return float(np.mean(self.masks[mask_reference]))

    def joint_probability(self, references: list[str]) -> float:
        if not references:
            raise ValueError("at least one trajectory mask is required")
        mask = np.ones(self.trials, dtype=bool)
        for reference in references:
            mask &= self.masks[reference].astype(bool)
        return float(mask.mean())


def simulate_team_runs(
    home_expected_runs: float,
    away_expected_runs: float,
    *,
    trials: int = 10_000,
    innings: int = 9,
    seed: int = 0,
) -> TrajectoryBatch:
    """Initial aggregate backbone, explicitly not an event-level simulator."""
    if trials <= 0 or innings < 5:
        raise ValueError("invalid simulation dimensions")
    rng = np.random.default_rng(seed)
    home = rng.poisson(max(home_expected_runs, 0.0) / innings, size=(trials, innings)).astype(np.int16)
    away = rng.poisson(max(away_expected_runs, 0.0) / innings, size=(trials, innings)).astype(np.int16)
    batch = TrajectoryBatch(home_runs_by_inning=home, away_runs_by_inning=away)
    batch.validate()
    return batch
