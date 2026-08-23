from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AllocationPathProtocol:
    """Frozen checkpoint schedule for an NFL pregame path-evidence layer.

    Ported from sports/mlb/conditional_chain/protocol.py, itself adapted
    from NBA's five-phase ``T-240/-120/-60/-30/-5`` schedule. Not currently
    wired into any NFL pipeline (sports/nfl/conditional_chain only ports
    outcome_worlds.py's sport-agnostic joint-outcome-world machinery for
    PARLAY_V2's candidate adapter); kept here only so this module mirrors
    its MLB counterpart's shape if a future NFL path-evidence shadow layer
    is ever built the same way MLB's was.
    """

    version: str = "NFL_ALLOCATION_PATH_V1_SHADOW"
    checkpoints_minutes: tuple[int, ...] = (-1440, -360, -90, -15, -2)
    max_checkpoint_age_minutes: int = 45
    minimum_train_events: int = 20
    minimum_confirmation_events: int = 20
    bootstrap_samples: int = 10_000
    sign_flip_samples: int = 50_000
    random_seed: int = 20260820
    endpoint_model: str = "standard_scaler_logistic_c1"
    representation_unit: str = "event_player_market"
    statistical_unit: str = "game_event"
    path_mode: str = "shadow_only_until_incremental_value_supported"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BinaryOutcomeSetProtocol:
    version: str = "NFL_BINARY_OUTCOME_SET_V1_SHADOW"
    maximum_candidates: int = 10
    target_miscoverage: float = 0.10
    minimum_calibration_slates: int = 20
    requested_leg_counts: tuple[int, ...] = (2, 3, 4)
    calibration_method: str = "label_powerset_aps_deterministic"
    proof_contract: str = "all_retained_worlds_assign_every_selected_leg_a_win"
    binary_success_definition: str = "settled_win_vs_not_full_payout_win"
    require_path_certificate: bool = True
    score_epsilon: float = 1e-12
    publication_mode: str = "shadow_only_until_selective_risk_certificate"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ALLOCATION_PATH_PROTOCOL = AllocationPathProtocol()
BINARY_OUTCOME_SET_PROTOCOL = BinaryOutcomeSetProtocol()
