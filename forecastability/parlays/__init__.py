"""
NBA Parlay Subsystem - Shadow Mode

Robust parlay creation with joint-state filtering, event-supply detection,
shared failure mode analysis, and stress testing.

All operations in SHADOW MODE - no production impact.
"""

from core_utils import (
    PriceStatus,
    LegStatus,
    CorrelationClass,
    NewsStatus,
    american_to_decimal,
    decimal_to_american,
    american_to_implied_prob,
    american_to_break_even_prob,
    edge_from_probability_and_odds,
    min_acceptable_odds_for_edge,
    parlay_synthetic_odds,
    parlay_break_even_prob,
    lcb_edge,
    robust_edge,
)

from data_types import (
    MarketFamily,
    Side,
    PricedBinaryEvent,
    SingleLegEvaluation,
    ParlayLeg,
    JointState,
    ParlayCandidate,
)

from build_priced_event_universe import PricedEventUniverseBuilder
from line_zone_scanner import LineZoneScanner, LineZoneClassification
from single_leg_set_membership import SingleLegSetMembership
from anchor_companion_generator import AnchorCompanionGenerator
from shared_event_supply_engine import SharedEventSupplyEngine
from parlay_price_engine import ParlayPriceEngine
from parlay_probability_engine import ParlayProbabilityEngine
from parlay_stress_engine import ParlayStressEngine
from parlay_selector import ParlaySelector
from orchestrator import ParlaySubsystemOrchestrator

__version__ = "0.1.0"
__all__ = [
    "PriceStatus",
    "LegStatus",
    "CorrelationClass",
    "NewsStatus",
    "MarketFamily",
    "Side",
    "PricedBinaryEvent",
    "SingleLegEvaluation",
    "ParlayLeg",
    "JointState",
    "ParlayCandidate",
    "PricedEventUniverseBuilder",
    "LineZoneScanner",
    "LineZoneClassification",
    "SingleLegSetMembership",
    "AnchorCompanionGenerator",
    "SharedEventSupplyEngine",
    "ParlayPriceEngine",
    "ParlayProbabilityEngine",
    "ParlayStressEngine",
    "ParlaySelector",
    "ParlaySubsystemOrchestrator",
]
