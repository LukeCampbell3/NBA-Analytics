"""Canonical PAR-PVG v0.5 and PAR-F v0.6 model configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PAR_MODEL_VERSION = "par_pvg_v0_5"
PARF_MODEL_VERSION = "parf_v0_6"
POINTS_PER_WIN = 30.4
ACCOUNTING_TOLERANCE = 1e-6

CATEGORIES: dict[str, dict[str, str]] = {
    "SCORING": {"field": "scoring_par", "label": "Scoring"},
    "CREATION": {"field": "creation_par", "label": "Creation"},
    "BALL_SECURITY": {"field": "ball_security_par", "label": "Ball Security"},
    "PLAYTYPE_PNR": {"field": "playtype_pnr_par", "label": "Pick & Roll"},
    "SPACING": {"field": "spacing_par", "label": "Spacing"},
    "REBOUNDING": {"field": "rebounding_par", "label": "Rebounding"},
    "PERIMETER_DISRUPTION": {"field": "perimeter_disruption_par", "label": "Perimeter Disruption"},
    "RIM_DEFENSE": {"field": "rim_defense_par", "label": "Rim Defense"},
    "CONTEST_DEFENSE": {"field": "contest_defense_par", "label": "Contest Defense"},
    "HUSTLE": {"field": "hustle_par", "label": "Hustle"},
    "RESIDUAL": {"field": "residual_par", "label": "Residual"},
}

SOURCE_TIERS: dict[str, dict[str, Any]] = {
    "TIER_A_DIRECT": {"label": "Direct observed event", "reliability_weight": 1.00, "confidence_tier": "high"},
    "TIER_B_TRACKING_BACKED": {"label": "Validated tracking-derived evidence", "reliability_weight": 0.90, "confidence_tier": "high"},
    "TIER_C_CONFIRMED_HIDDEN_ROLE": {"label": "Confirmed hidden-role inference", "reliability_weight": 0.75, "confidence_tier": "medium"},
    "TIER_D_SHRUNK_PROXY": {"label": "Shrunk proxy", "reliability_weight": 0.55, "confidence_tier": "low"},
    "TIER_E_UNSUPPORTED": {"label": "Unsupported", "reliability_weight": 0.00, "confidence_tier": "unsupported"},
}

ROLE_PROFILES = [
    "primary_creator",
    "secondary_creator",
    "scoring_guard",
    "movement_shooter",
    "three_and_d_wing",
    "connector",
    "combo_forward",
    "roll_big",
    "stretch_big",
    "rim_protector",
]

ATOM_REGISTRY: dict[str, dict[str, Any]] = {
    "scoring_volume_above_replacement": {
        "category": "SCORING",
        "label": "Scoring volume above replacement",
        "persistence_key": "scoring_volume",
        "value_family": "box_visible",
    },
    "scoring_efficiency_above_replacement": {
        "category": "SCORING",
        "label": "Scoring efficiency above replacement",
        "persistence_key": "scoring_efficiency",
        "value_family": "box_visible",
    },
    "stable_shot_skill": {"category": "SCORING", "label": "Stable shot skill", "persistence_key": "stable_shot_skill"},
    "contested_shotmaking": {"category": "SCORING", "label": "Contested shotmaking", "persistence_key": "contested_shotmaking"},
    "passing_creation": {"category": "CREATION", "label": "Passing creation", "persistence_key": "creation_passing", "value_family": "box_visible"},
    "advantage_creation": {"category": "CREATION", "label": "Advantage creation", "persistence_key": "creation_passing"},
    "connective_passing": {"category": "CREATION", "label": "Connective passing", "persistence_key": "creation_passing"},
    "creation_reads": {"category": "CREATION", "label": "Creation reads", "persistence_key": "creation_passing"},
    "turnover_control": {"category": "BALL_SECURITY", "label": "Turnover control", "persistence_key": "turnover_control"},
    "possession_preservation": {"category": "BALL_SECURITY", "label": "Possession preservation", "persistence_key": "turnover_control"},
    "negative_turnover_value": {"category": "BALL_SECURITY", "label": "Negative turnover value", "persistence_key": "turnover_control", "value_family": "box_visible"},
    "pnr_ball_handler": {"category": "PLAYTYPE_PNR", "label": "PnR ball-handler value", "persistence_key": "pnr_ball_handler"},
    "pnr_roll_man": {"category": "PLAYTYPE_PNR", "label": "PnR roll-man value", "persistence_key": "pnr_roll_man"},
    "screen_derived_value": {"category": "PLAYTYPE_PNR", "label": "Screen-derived value", "persistence_key": "pnr_roll_man"},
    "spacing_volume": {"category": "SPACING", "label": "Spacing volume", "persistence_key": "spacing_volume"},
    "spacing_efficiency": {"category": "SPACING", "label": "Spacing efficiency", "persistence_key": "spacing_efficiency"},
    "gravity_displacement": {"category": "SPACING", "label": "Gravity/displacement", "persistence_key": "spacing_volume"},
    "offensive_rebounding": {"category": "REBOUNDING", "label": "Offensive rebounding", "persistence_key": "oreb"},
    "defensive_rebounding": {"category": "REBOUNDING", "label": "Defensive rebounding", "persistence_key": "defensive_rebounding"},
    "possession_extension": {"category": "REBOUNDING", "label": "Possession extension", "persistence_key": "oreb"},
    "steals": {"category": "PERIMETER_DISRUPTION", "label": "Steals", "persistence_key": "steals", "value_family": "box_visible"},
    "steal_pressure": {"category": "PERIMETER_DISRUPTION", "label": "Steal pressure", "persistence_key": "steals"},
    "deflections": {"category": "PERIMETER_DISRUPTION", "label": "Deflections", "persistence_key": "deflections_hustle"},
    "blocks": {"category": "RIM_DEFENSE", "label": "Blocks", "persistence_key": "blocks"},
    "rim_deterrence_tracking": {"category": "RIM_DEFENSE", "label": "Rim deterrence", "persistence_key": "rim_deterrence_tracking"},
    "rim_deterrence_proxy": {"category": "RIM_DEFENSE", "label": "Rim deterrence proxy", "persistence_key": "rim_deterrence_proxy"},
    "rim_protection": {"category": "RIM_DEFENSE", "label": "Rim protection", "persistence_key": "blocks"},
    "contest_suppression_tracking": {"category": "CONTEST_DEFENSE", "label": "Tracking-backed contest suppression", "persistence_key": "contest_defense_tracking"},
    "contest_suppression_proxy": {"category": "CONTEST_DEFENSE", "label": "Proxy-backed contest suppression", "persistence_key": "contest_defense_proxy"},
    "shot_quality_suppression": {"category": "CONTEST_DEFENSE", "label": "Shot-quality suppression", "persistence_key": "contest_defense_tracking"},
    "loose_ball_possession_creation": {"category": "HUSTLE", "label": "Loose-ball possession creation", "persistence_key": "deflections_hustle"},
    "recoveries": {"category": "HUSTLE", "label": "Recoveries", "persistence_key": "deflections_hustle"},
    "plus_minus_residual": {"category": "RESIDUAL", "label": "Plus-minus residual", "persistence_key": "plus_minus_residual"},
}

PERSISTENCE_VALUES = {
    "scoring_volume": 0.70,
    "scoring_efficiency": 0.55,
    "stable_shot_skill": 0.65,
    "contested_shotmaking": 0.45,
    "creation_passing": 0.75,
    "turnover_control": 0.75,
    "spacing_volume": 0.80,
    "spacing_efficiency": 0.55,
    "pnr_ball_handler": 0.65,
    "pnr_roll_man": 0.70,
    "oreb": 0.80,
    "defensive_rebounding": 0.70,
    "steals": 0.60,
    "blocks": 0.65,
    "contest_defense_tracking": 0.70,
    "contest_defense_proxy": 0.45,
    "rim_deterrence_tracking": 0.65,
    "rim_deterrence_proxy": 0.40,
    "deflections_hustle": 0.50,
    "plus_minus_residual": 0.35,
}

# Replacement baselines are keyed by role and atom type. Values are point-value
# expectations per 36 minutes for the frozen direct-event adapter documented in
# docs/par_frozen_model.md.
DEFAULT_REPLACEMENT_BASELINES = {
    role: {
        "scoring_volume_above_replacement": 12.0,
        "scoring_efficiency_above_replacement": 0.0,
        "passing_creation": 3.2,
        "negative_turnover_value": -2.2,
        "steals": 1.1,
    }
    for role in ROLE_PROFILES
}
DEFAULT_REPLACEMENT_BASELINES["primary_creator"].update({"scoring_volume_above_replacement": 18.0, "passing_creation": 7.0, "negative_turnover_value": -3.7})
DEFAULT_REPLACEMENT_BASELINES["secondary_creator"].update({"scoring_volume_above_replacement": 15.5, "passing_creation": 5.0, "negative_turnover_value": -3.0})
DEFAULT_REPLACEMENT_BASELINES["scoring_guard"].update({"scoring_volume_above_replacement": 17.0, "passing_creation": 3.8})
DEFAULT_REPLACEMENT_BASELINES["movement_shooter"].update({"scoring_volume_above_replacement": 11.5, "passing_creation": 2.4})
DEFAULT_REPLACEMENT_BASELINES["rim_protector"].update({"scoring_volume_above_replacement": 9.0, "passing_creation": 2.0})
DEFAULT_REPLACEMENT_BASELINES["roll_big"].update({"scoring_volume_above_replacement": 10.0, "passing_creation": 2.0})


@dataclass(frozen=True)
class ModelConfig:
    par_model_version: str = PAR_MODEL_VERSION
    parf_model_version: str = PARF_MODEL_VERSION
    points_per_win: float = POINTS_PER_WIN
    accounting_tolerance: float = ACCOUNTING_TOLERANCE
    categories: dict[str, dict[str, str]] = None  # type: ignore[assignment]
    atom_registry: dict[str, dict[str, Any]] = None  # type: ignore[assignment]
    source_tiers: dict[str, dict[str, Any]] = None  # type: ignore[assignment]
    persistence_values: dict[str, float] = None  # type: ignore[assignment]
    replacement_baselines: dict[str, dict[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", CATEGORIES)
        object.__setattr__(self, "atom_registry", ATOM_REGISTRY)
        object.__setattr__(self, "source_tiers", SOURCE_TIERS)
        object.__setattr__(self, "persistence_values", PERSISTENCE_VALUES)
        object.__setattr__(self, "replacement_baselines", DEFAULT_REPLACEMENT_BASELINES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "par_model_version": self.par_model_version,
            "parf_model_version": self.parf_model_version,
            "points_per_win": self.points_per_win,
            "accounting_tolerance": self.accounting_tolerance,
            "categories": self.categories,
            "atom_registry": self.atom_registry,
            "source_tiers": self.source_tiers,
            "persistence_values": self.persistence_values,
            "role_profiles": ROLE_PROFILES,
            "replacement_baselines": self.replacement_baselines,
            "identity": {
                "par": "BoxVisiblePAR + ConfirmedHiddenRolePAR + ShrunkProxyPAR - OverlapLeakage",
                "war_equivalent": "PAR / 30.4",
                "par_1000": "(PAR / Minutes) * 1000",
                "pvg_score": "50 + 45 * tanh(PAR_1000 / 210)",
            },
        }


MODEL_CONFIG = ModelConfig()
