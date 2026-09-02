from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceState(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    WALK_FORWARD_VALIDATION = "WALK_FORWARD_VALIDATION"
    LOCKED_VALIDATION = "LOCKED_VALIDATION"
    PROSPECTIVE_SHADOW = "PROSPECTIVE_SHADOW"
    CERTIFIED = "CERTIFIED"


class CapabilityState(StrEnum):
    SUPPORTED = "SUPPORTED"
    SHADOW_ONLY = "SHADOW_ONLY"
    DISCOVERY = "DISCOVERY"
    DATA_REQUIRED = "DATA_REQUIRED"
    MODEL_REQUIRED = "MODEL_REQUIRED"
    EVENT_MODEL_REQUIRED = "EVENT_MODEL_REQUIRED"
    EVENT_IDENTITY_UNAVAILABLE = "EVENT_IDENTITY_UNAVAILABLE"
    PRICE_UNAVAILABLE = "PRICE_UNAVAILABLE"
    HISTORICAL_REPLAY_UNAVAILABLE = "HISTORICAL_REPLAY_UNAVAILABLE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class SourceField:
    value: Any
    source: str
    as_of_utc: str
    available: bool = True
    fallback: str | None = None


@dataclass(frozen=True)
class GameState:
    game_id: str
    inning: int = 1
    half: str = "top"
    outs: int = 0
    home_score: int = 0
    away_score: int = 0
    runners_on_base: int = 0
    batting_team: str = ""
    fielding_team: str = ""
    current_batter: str | None = None
    current_pitcher: str | None = None
    batting_order_position: int | None = None
    times_through_order: int = 0
    pitcher_pitch_count: int = 0
    pitcher_inning_pitch_count: int = 0
    batter_pa_pitch_count: int = 0
    bullpen_state: str = "unknown"
    lineup_state: str = "unknown"
    role_state: str = "unknown"

    def validate(self) -> None:
        if not self.game_id:
            raise ValueError("game_id is required")
        if self.inning < 1 or self.outs not in {0, 1, 2}:
            raise ValueError("invalid inning/outs state")
        if self.half not in {"top", "bottom"}:
            raise ValueError("half must be top or bottom")
        if min(self.home_score, self.away_score, self.pitcher_pitch_count, self.pitcher_inning_pitch_count, self.batter_pa_pitch_count) < 0:
            raise ValueError("counting state cannot be negative")


@dataclass(frozen=True)
class MarketCapability:
    market_type: str
    status: CapabilityState
    model: str | None = None
    settlement: str | None = None
    blocker: str | None = None
    requires_event_identity: bool = False


@dataclass
class BetCandidate:
    candidate_id: str
    game_id: str
    subject_type: str
    subject_id: str
    team: str
    opponent: str
    market_type: str
    period: str
    event_identity: str | None
    side: str
    line: float | None
    sportsbook: str
    sportsbook_market_id: str | None
    sportsbook_selection_id: str | None
    american_price: float | None
    decimal_price: float | None
    structural_probability: float | None
    market_conditioned_probability: float | None
    raw_probability: float | None
    calibrated_probability: float | None
    uncertainty: float | None
    usable_probability: float | None
    market_break_even_probability: float | None = None
    no_vig_probability: float | None = None
    probability_edge: float | None = None
    expected_value: float | None = None
    conservative_expected_value: float | None = None
    support_status: str = "UNKNOWN"
    lineup_status: str = "UNKNOWN"
    role_status: str = "UNKNOWN"
    identity_status: str = "UNKNOWN"
    trajectory_mask_reference: str | None = None
    dependencies: list[str] = field(default_factory=list)
    evidence_state: EvidenceState = EvidenceState.DEVELOPMENT
    publication_authority: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    source_payload: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_state"] = self.evidence_state.value
        return data


@dataclass
class Ticket:
    ticket_id: str
    ticket_type: str
    leg_count: int
    legs: list[BetCandidate]
    combined_decimal_price: float | None
    independent_product_probability: float
    joint_probability: float
    dependency_delta: float
    break_even_probability: float | None
    probability_edge: float | None
    conservative_expected_value: float | None
    evidence_state: EvidenceState
    publication_authority: bool = False
    sportsbook: str | None = None
    betslip_url: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_state"] = self.evidence_state.value
        data["legs"] = [leg.to_dict() for leg in self.legs]
        return data
