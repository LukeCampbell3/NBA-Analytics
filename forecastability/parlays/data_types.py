"""
Data models and types for the parlay subsystem.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from core_utils import LegStatus, NewsStatus, ParlayDecision


class MarketFamily(str, Enum):
    """Market families in NBA player props."""
    PTS = "PTS"
    REB = "REB"
    AST = "AST"
    PRA = "PRA"
    PR = "PR"
    PA = "PA"
    RA = "RA"
    THREES = "3PM"
    STL = "STL"
    BLK = "BLK"
    TO = "TO"
    FTA = "FTA"
    FGM = "FGM"
    FTM = "FTM"


class Side(str, Enum):
    """Bet side."""
    OVER = "OVER"
    UNDER = "UNDER"


class LineType(str, Enum):
    """Type of line."""
    MAIN_LINE = "MAIN_LINE"
    ALT_LINE = "ALT_LINE"
    COMBO_MARKET = "COMBO_MARKET"


@dataclass
class PlayerState:
    """Current player state snapshot."""
    player_id: str
    player_name: str
    team: str
    position: str
    expected_minutes: float
    expected_minutes_lower: float
    expected_minutes_upper: float
    expected_usage: float
    expected_role_rating: float
    season_avg_stat_value: float
    recent_avg_stat_value: float
    injury_status: str
    latest_news: Optional[str] = None
    minutes_limit: Optional[int] = None


@dataclass
class OpponentState:
    """Opponent state relevant to player performance."""
    opponent_id: str
    opponent_name: str
    opponent_defensive_rank: float
    opponent_pace_factor: float
    opponent_stat_allowed_rate: float
    opponent_blowout_risk: float
    opponent_lineup_status: str


@dataclass
class TeamState:
    """Team state affecting player performance."""
    team_id: str
    team_name: str
    team_offensive_efficiency: float
    team_rebound_rate: float
    team_assist_rate: float
    team_three_point_rate: float
    team_turnover_rate: float
    team_pace: float
    backup_availability: str
    starter_health_status: str


@dataclass
class RoleState:
    """Player's role within the team."""
    primary_usage_rank: int
    shot_attempts_share: float
    assist_attempts_share: float
    rebound_share: float
    is_starter: bool
    is_backup: bool
    minutes_volatility: float


@dataclass
class MarketState:
    """Market conditions for the specific prop."""
    market_family: str
    line_movement_since_open_pct: float
    books_with_line_count: int
    min_available_line: float
    max_available_line: float
    mode_line: float
    volume_of_trading: str  # LOW, MEDIUM, HIGH
    weather_impact: Optional[str] = None


@dataclass
class ScenarioState:
    """Scenario breakdown for player performance."""
    positive_state_mass: float  # % probability of plan-holds
    negative_state_mass: float
    mild_minutes_loss_weight: float
    foul_trouble_weight: float
    blowout_pull_weight: float
    role_shift_weight: float
    usage_spike_weight: float
    team_offense_collapse_weight: float
    opponent_scheme_disruption_weight: float


@dataclass
class PriceState:
    """Price information for the binary event."""
    american_odds: float
    decimal_odds: float
    implied_prob: float
    break_even_prob: float
    best_book: str
    is_best_price: bool
    price_age_minutes: int
    vig_estimate: float


@dataclass
class PricedBinaryEvent:
    """
    A single priced binary event in the universe.
    Represents one player + game + market + side + line + book + time snapshot.
    """
    # Identity
    event_id: str
    game_id: str
    game_date: str  # YYYY-MM-DD
    snapshot_time: datetime
    player_id: str
    player_name: str
    team: str
    opponent: str
    book: str
    market_type: str  # e.g., "PLAYER_POINTS"
    side: Side
    line: float
    is_main_line: bool
    is_alt_line: bool
    is_combo_market: bool
    market_family: MarketFamily
    
    # Price
    odds_american: float
    odds_decimal: float
    implied_prob_raw: float
    no_vig_prob: Optional[float] = None
    best_book_for_line: str = ""
    best_price_for_side: bool = False
    min_acceptable_odds: Optional[float] = None
    price_status: str = "PRICE_VALID"
    
    # Distribution
    model_mean: float = 0.0
    model_std: float = 0.0
    q10: float = 0.0
    q25: float = 0.0
    q50: float = 0.0
    q75: float = 0.0
    q90: float = 0.0
    line_percentile: float = 0.0
    p_over_raw: float = 0.0
    p_under_raw: float = 0.0
    p_side_raw: float = 0.0
    p_side_stress: float = 0.0
    p_side_lcb: float = 0.0
    p_push: float = 0.0
    
    # Edge
    raw_edge: float = 0.0
    robust_edge: float = 0.0
    lcb_edge: float = 0.0
    raw_ev: float = 0.0
    robust_ev: float = 0.0
    lcb_ev: float = 0.0
    edge_fragility: float = 0.0
    
    # Reliability
    forecastability_score: float = 0.0
    plan_reliability: float = 0.0
    scenario_agreement: float = 0.0
    management_volatility_score: float = 0.0
    market_instability_score: float = 0.0
    chaos_score: float = 0.0
    similar_state_count: int = 0
    similar_state_p80_abs_error: float = 0.0
    interval_width: float = 0.0
    team_rebound_rate: float = 0.0
    rebound_share: float = 0.0
    team_shooting_efficiency_risk: float = 0.0
    opponent_shooting_efficiency_risk: float = 0.0
    wing_rebound_leakage_score: float = 0.0
    upper_band_line_penalty: float = 0.0
    minutes_band_failure_risk: float = 0.0
    directional_failure_risk: float = 0.0
    
    # Scenario
    positive_state_mass: float = 0.0
    negative_state_mass: float = 0.0
    plan_holds_weight: float = 0.0
    mild_minutes_loss_weight: float = 0.0
    foul_trouble_weight: float = 0.0
    blowout_pull_weight: float = 0.0
    role_shift_weight: float = 0.0
    usage_spike_weight: float = 0.0
    team_offense_collapse_weight: float = 0.0
    opponent_scheme_disruption_weight: float = 0.0
    top_positive_scenarios: List[str] = field(default_factory=list)
    top_negative_scenarios: List[str] = field(default_factory=list)
    top_failure_modes: List[str] = field(default_factory=list)
    
    # News
    news_status: str = "CLEAR"
    injury_dependency_score: float = 0.0
    lineup_dependency_score: float = 0.0
    rerun_required_after_news_flag: bool = False
    
    # Metadata
    confidence_level: str = "MEDIUM"
    analyst_notes: str = ""
    data_sources: List[str] = field(default_factory=list)


@dataclass
class SingleLegEvaluation:
    """Evaluation of a single leg for parlay acceptance."""
    event_id: str
    player_name: str
    player_market: str
    side: Side
    line: float
    odds_american: float

    # Tier classification
    leg_status: str
    tier: str

    # Acceptance decision
    accepted_into_single_leg_pool: bool
    game_id: str = ""
    team: str = ""
    break_even_prob: float = 0.0
    p_side_stress: float = 0.0
    p_side_lcb: float = 0.0
    
    # Edges
    lcb_edge: float = 0.0
    robust_edge: float = 0.0
    min_acceptable_odds: Optional[float] = None
    
    # Reasons
    rejection_reasons: List[str] = field(default_factory=list)
    promotion_requirements: List[str] = field(default_factory=list)
    
    # Quality metrics
    forecastability_score: float = 0.0
    plan_reliability: float = 0.0
    scenario_agreement: float = 0.0


@dataclass
class ParlayLeg:
    """A single leg in a parlay candidate."""
    event_id: str
    player_name: str
    market_family: str
    side: Side
    line: float
    odds_american: float
    p_stress: float
    p_lcb: float
    lcb_edge: float
    game_id: str


@dataclass
class JointState:
    """Joint state between parlay legs."""
    shared_success_scenarios: List[str] = field(default_factory=list)
    shared_failure_modes: List[str] = field(default_factory=list)
    shared_event_supply_pools: List[str] = field(default_factory=list)
    dependency_classes: List[str] = field(default_factory=list)
    empirical_correlation: Optional[float] = None
    correlation_confidence: float = 0.0


@dataclass
class ParlayCandidate:
    """A candidate parlay for acceptance/rejection."""
    parlay_id: str
    checkpoint: str
    snapshot_time: datetime
    legs: List[ParlayLeg]
    
    # Price info
    combined_decimal_odds: float
    combined_american_odds: float
    parlay_break_even_prob: float
    price_source: str  # SYNTHETIC or BOOK_QUOTED
    price_validity: str
    price_gap_vs_synthetic: Optional[float] = None
    same_game_price_penalty: float = 0.0
    
    # Probability
    p_joint_naive: float = 0.0
    p_joint_adjusted: float = 0.0
    p_joint_stress: float = 0.0
    p_joint_lcb: float = 0.0
    
    # EV
    raw_joint_edge: float = 0.0
    robust_joint_edge: float = 0.0
    lcb_joint_edge: float = 0.0
    raw_joint_ev: float = 0.0
    robust_joint_ev: float = 0.0
    lcb_joint_ev: float = 0.0
    joint_probability_confidence: float = 0.0
    
    # Risk metrics
    shared_failure_risk: float = 0.0
    shared_event_supply_penalty: float = 0.0
    dependency_penalty: float = 0.0
    edge_fragility: float = 0.0
    
    # Quality
    anchor_leg_idx: int = 0
    companion_leg_indices: List[int] = field(default_factory=list)
    compatible_state_score: float = 0.0
    min_leg_forecastability: float = 0.0
    min_leg_plan_reliability: float = 0.0
    min_leg_scenario_agreement: float = 0.0
    price_quality_score: float = 0.0
    
    # Final decision
    decision: str = "PENDING"
    final_parlay_score: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    
    # Joint state metadata
    joint_state: Optional[JointState] = None
    
    # Metadata
    tier: str = "UNKNOWN"
    same_game: bool = False
    news_dependent: bool = False
    price_dependent: bool = False
