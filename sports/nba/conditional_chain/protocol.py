from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrozenSelectorProtocol:
    version: str = "ROBUST_STATE_INTERSECTION_Q25_V1"
    lookback_games: int = 20
    jeffreys_alpha: float = 0.5
    jeffreys_beta: float = 0.5
    credible_lower_quantile: float = 0.10
    break_even_probability: float = 110.0 / 210.0
    over_bonus: float = 0.005
    parlay_legs: int = 4
    reservoir_size: int = 10
    publication_floor: float = 0.6858420628569935

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AllocationPathProtocol:
    version: str = "NBA_ALLOCATION_PATH_V1_1_FROZEN"
    market: str = "player_points"
    checkpoints_minutes: tuple[int, ...] = (-240, -120, -60, -30, -5)
    max_checkpoint_age_minutes: int = 20
    minimum_independent_engines: int = 2
    minimum_stable_players: int = 4
    minimum_stable_coverage: float = 0.70
    practical_mae_improvement: float = 0.005
    one_sided_alpha: float = 0.05
    checkpoint_alpha: float = 0.05 / 3.0
    minimum_train_events: int = 20
    minimum_confirmation_events: int = 20
    bootstrap_samples: int = 10_000
    sign_flip_samples: int = 50_000
    random_seed: int = 20260820
    endpoint_model: str = "standard_scaler_ridge_alpha_1"
    representation_unit: str = "event_team_market"
    statistical_unit: str = "game_event"
    path_mode: str = "shadow_only_until_incremental_value_supported"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_SELECTOR_PROTOCOL = FrozenSelectorProtocol()
ALLOCATION_PATH_PROTOCOL = AllocationPathProtocol()
