"""
models.py - Data models and schemas

Defines core data structures used across the system.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum


class UsageBand(str, Enum):
    """Player usage level"""
    HIGH = "high"
    MEDIUM = "med"
    LOW = "low"


class Archetype(str, Enum):
    """Player archetype"""
    INITIATOR_CREATOR = "initiator_creator"
    SHOOTING_SPECIALIST = "shooting_specialist"
    RIM_PROTECTOR = "rim_protector"
    VERSATILE_WING = "versatile_wing"
    CONNECTOR = "connector"
    ATHLETIC_FINISHER = "athletic_finisher"
    COMBO_GUARD = "combo_guard"
    DEFAULT = "default"


class DefensiveBurden(str, Enum):
    """Defensive burden level"""
    HIGH = "high"
    MEDIUM = "med"
    LOW = "low"


class TrustLevel(str, Enum):
    """Data trust level"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgingPhase(str, Enum):
    """Player aging phase"""
    PRE_GROWTH = "pre_growth"
    GROWTH = "growth"
    PLATEAU = "plateau"
    DECLINE = "decline"


@dataclass
class PlayerInfo:
    """Basic player information"""
    id: str
    name: str
    team: str
    season: int
    position: str
    age: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerIdentity:
    """Player identity and role"""
    usage_band: str
    primary_archetype: str
    position: str
    secondary_archetypes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShotProfile:
    """Shot distribution profile"""
    three_rate: float
    volume: float
    rim_rate: Optional[float] = None
    mid_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OffensiveMetrics:
    """Offensive performance metrics"""
    shot_profile: Dict[str, Any]
    creation: Dict[str, Any]
    efficiency: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefensiveMetrics:
    """Defensive performance metrics"""
    burden: Dict[str, Any]
    performance: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactMetrics:
    """Impact metrics"""
    net: float
    offensive: float
    defensive: float
    source: str = "estimated"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrustAssessment:
    """Data trust assessment"""
    score: float
    level: str
    data_quality: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyAssessment:
    """Uncertainty assessment"""
    overall: float
    sample_size: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerMetadata:
    """Player metadata"""
    games_played: str
    minutes: float
    data_quality: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlayerCard:
    """Complete player card"""
    player: Dict[str, Any]
    identity: Dict[str, Any]
    offense: Dict[str, Any]
    defense: Dict[str, Any]
    impact: Dict[str, Any]
    metadata: Dict[str, Any]
    trust: Optional[Dict[str, Any]] = None
    uncertainty: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerCard':
        return cls(**data)


@dataclass
class AgingCurve:
    """Aging curve parameters"""
    archetype: str
    peak_age: float
    growth_window: tuple
    plateau_window: tuple
    decline_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype": self.archetype,
            "peak_age": self.peak_age,
            "growth_window": list(self.growth_window),
            "plateau_window": list(self.plateau_window),
            "decline_rate": self.decline_rate
        }


@dataclass
class ContractInfo:
    """Contract information"""
    player_id: str
    player_name: str
    salary_by_year: Dict[int, float]
    contract_length: int
    current_year: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValuationResult:
    """Valuation result"""
    player_id: str
    player_name: str
    season: int
    current_wins_added: float
    market_value_by_year: Dict[int, float]
    salary_by_year: Dict[int, float]
    surplus_by_year: Dict[int, float]
    npv_surplus: float
    trade_value_low: float
    trade_value_base: float
    trade_value_high: float
    aging_multipliers: Dict[int, float]
    peak_age: float
    current_phase: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BreakoutAnalysis:
    """Breakout potential analysis"""
    player_name: str
    can_breakout: bool
    opportunity_score: float
    signal_strength: float
    confidence: float
    factors: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefensePortability:
    """Defense portability analysis"""
    player_name: str
    defensive_role: str
    portability_score: float
    portability_level: str
    switch_ability: float
    failure_modes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactSanityCheck:
    """Impact sanity check result"""
    player_name: str
    sanity_level: str
    flags: List[str]
    impact_summary: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoutingReport:
    """Scouting report"""
    player_name: str
    role_summary: str
    strengths: List[str]
    weaknesses: List[str]
    archetype: str
    usage_band: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    scouting_report: Dict[str, Any]
    breakout_potential: Dict[str, Any]
    defense_portability: Dict[str, Any]
    impact_sanity: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Schema validation helpers

def validate_player_card(card: Dict[str, Any]) -> bool:
    """Validate player card has required fields"""
    required_fields = ['player', 'identity', 'offense', 'defense', 'impact', 'metadata']
    return all(field in card for field in required_fields)


def validate_player_info(player: Dict[str, Any]) -> bool:
    """Validate player info has required fields"""
    required_fields = ['name', 'team', 'season']
    return all(field in player for field in required_fields)


def get_schema_version() -> str:
    """Get current schema version"""
    return "1.0.0"


def get_supported_archetypes() -> List[str]:
    """Get list of supported archetypes"""
    return [a.value for a in Archetype]


def get_supported_usage_bands() -> List[str]:
    """Get list of supported usage bands"""
    return [u.value for u in UsageBand]


def get_supported_aging_phases() -> List[str]:
    """Get list of supported aging phases"""
    return [p.value for p in AgingPhase]
