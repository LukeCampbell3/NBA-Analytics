"""
PHASE 6: Team Environment Failure Modes

Add named team-level failure modes and their exposures:

TEAM_OFFENSE_COLLAPSE
LOW_TEAM_ASSIST_ENVIRONMENT
POOR_SHOOTING_ENVIRONMENT
PACE_COLLAPSE
BLOWOUT_PULL
FOUL_ENVIRONMENT_SPIKE
ROTATION_SHIFT
OPPONENT_PRESSURE_DISRUPTION
REBOUND_SUPPLY_COLLAPSE
USAGE_CONCENTRATION_SHIFT
GARBAGE_TIME_DISTORTION

For every leg, estimate exposure to these failure modes.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging

from data_types import ParlayLeg, MarketFamily, Side

logger = logging.getLogger(__name__)


@dataclass
class FailureMode:
    """Named team-level failure mode."""
    mode_name: str
    description: str
    impact_probability: float  # P(mode occurs)
    leg_susceptibility: float  # P(leg fails | mode)
    

TEAM_FAILURE_MODES = {
    "TEAM_OFFENSE_COLLAPSE": {
        "description": "Team efficiency drops significantly",
        "affected_markets": ["PTS", "PRA", "PR", "PA", "AST"],
        "severity": 0.08,
    },
    "LOW_TEAM_ASSIST_ENVIRONMENT": {
        "description": "Team makes fewer assists (ball movement slows)",
        "affected_markets": ["AST", "PA", "PRA"],
        "severity": 0.05,
    },
    "POOR_SHOOTING_ENVIRONMENT": {
        "description": "Team shooting % drops below normal",
        "affected_markets": ["PTS", "3PM", "FTA"],
        "severity": 0.06,
    },
    "PACE_COLLAPSE": {
        "description": "Game pace slows significantly",
        "affected_markets": ["PTS", "REB", "AST"],
        "severity": 0.04,
    },
    "BLOWOUT_PULL": {
        "description": "Game becomes a blowout, player pulled early",
        "affected_markets": ["PTS", "REB", "AST"],
        "severity": 0.03,
    },
    "FOUL_ENVIRONMENT_SPIKE": {
        "description": "Referees call many fouls, affecting team play",
        "affected_markets": ["PTS", "REB", "AST"],
        "severity": 0.03,
    },
    "ROTATION_SHIFT": {
        "description": "Coach changes rotations, player gets different minutes",
        "affected_markets": ["PTS", "REB", "AST"],
        "severity": 0.02,
    },
    "OPPONENT_PRESSURE_DISRUPTION": {
        "description": "Opponent pressure disrupts team offense",
        "affected_markets": ["PTS", "AST", "TO"],
        "severity": 0.04,
    },
    "REBOUND_SUPPLY_COLLAPSE": {
        "description": "Rebound supply shrinks (game pace or opponent dominance)",
        "affected_markets": ["REB"],
        "severity": 0.05,
    },
    "USAGE_CONCENTRATION_SHIFT": {
        "description": "Ball usage concentrates with other players",
        "affected_markets": ["PTS", "AST"],
        "severity": 0.03,
    },
    "GARBAGE_TIME_DISTORTION": {
        "description": "Game becomes garbage time, stats distorted",
        "affected_markets": ["PTS", "REB", "AST"],
        "severity": 0.02,
    },
}


class TeamEnvironmentFailureEngine:
    """
    Analyzes team-level failure modes and their impact on individual leg probabilities.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.failure_modes = TEAM_FAILURE_MODES
    
    def analyze_leg_exposure(
        self,
        leg: ParlayLeg,
        team_stats: Optional[Dict] = None,
        opponent_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze a single leg's exposure to team failure modes.
        
        Returns:
            {
                "exposed_failure_modes": [...],
                "total_exposure_probability": 0.0-1.0,
                "failure_mode_impacts": {...},
                "stress_probability_adjustment": 0.0-1.0,
            }
        """
        
        exposed_modes = []
        total_impact = 0.0
        failure_impacts = {}
        
        # Market family exposure to failure modes
        market = leg.market_family if isinstance(leg.market_family, str) else leg.market_family.value
        
        for mode_name, mode_info in self.failure_modes.items():
            if market in mode_info.get("affected_markets", []):
                # Leg is exposed to this failure mode
                severity = mode_info.get("severity", 0.0)
                
                # Adjust severity based on team stats if available
                if team_stats:
                    severity = self._adjust_severity_by_team_stats(
                        mode_name, severity, team_stats
                    )
                
                if severity > 0.01:  # Only track meaningful exposure
                    exposed_modes.append(mode_name)
                    total_impact += severity
                    failure_impacts[mode_name] = {
                        "severity": severity,
                        "affected": True,
                    }
        
        # Stress adjustment: reduce leg probability based on exposure
        # Maximum stress is proportional to total exposure
        stress_adjustment = min(0.15, total_impact * 0.5)
        
        return {
            "exposed_failure_modes": exposed_modes,
            "total_exposure_probability": min(0.5, total_impact),
            "failure_mode_impacts": failure_impacts,
            "stress_probability_adjustment": stress_adjustment,
            "exposure_count": len(exposed_modes),
        }
    
    def analyze_parlay_shared_exposure(
        self,
        legs: List[ParlayLeg],
        team_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze shared exposure to team failure modes across parlay legs.
        
        Returns:
            {
                "shared_failure_modes": [...],
                "shared_exposure_count": int,
                "shared_exposure_penalty": 0.0-0.5,
                "kills_all_legs_modes": [...],
                "kills_multiple_legs_modes": {...},
            }
        """
        
        # Collect all exposures per leg
        leg_exposures = []
        for leg in legs:
            exposure = self.analyze_leg_exposure(leg, team_stats)
            leg_exposures.append({
                "leg": leg,
                "modes": set(exposure.get("exposed_failure_modes", [])),
            })
        
        # Find shared failure modes
        if leg_exposures:
            shared_modes = leg_exposures[0]["modes"].copy()
            for exposure in leg_exposures[1:]:
                shared_modes = shared_modes.intersection(exposure["modes"])
        else:
            shared_modes = set()
        
        # Count how many legs each failure mode kills
        kills_by_mode = {}
        for mode_name in self.failure_modes.keys():
            leg_count = sum(
                1 for exp in leg_exposures
                if mode_name in exp["modes"]
            )
            if leg_count > 0:
                kills_by_mode[mode_name] = leg_count
        
        # Find modes that kill all legs
        kills_all = {
            mode: count for mode, count in kills_by_mode.items()
            if count == len(legs)
        }
        
        # Find modes that kill multiple legs
        kills_multiple = {
            mode: count for mode, count in kills_by_mode.items()
            if 1 < count < len(legs)
        }
        
        # Compute shared exposure penalty
        # If mode kills all legs, it's very dangerous
        # If mode kills multiple legs, it's moderately dangerous
        penalty = 0.0
        
        for mode, count in kills_all.items():
            penalty += self.failure_modes[mode].get("severity", 0.0) * 2.0
        
        for mode, count in kills_multiple.items():
            ratio = count / len(legs)
            penalty += self.failure_modes[mode].get("severity", 0.0) * ratio
        
        penalty = min(0.5, penalty)
        
        return {
            "shared_failure_modes": list(shared_modes),
            "shared_exposure_count": len(shared_modes),
            "shared_exposure_penalty": penalty,
            "kills_all_legs_modes": list(kills_all.keys()),
            "kills_multiple_legs_modes": kills_multiple,
        }
    
    def _adjust_severity_by_team_stats(
        self,
        mode_name: str,
        base_severity: float,
        team_stats: Dict
    ) -> float:
        """Adjust failure mode severity based on team statistics."""
        
        adjusted = base_severity
        
        # Example: if team has poor offensive efficiency, increase offense collapse exposure
        if mode_name == "TEAM_OFFENSE_COLLAPSE":
            if team_stats.get("offensive_efficiency", 1.0) < 0.95:
                adjusted *= 1.3
        
        # If team has poor assist rate, increase low-assist environment exposure
        if mode_name == "LOW_TEAM_ASSIST_ENVIRONMENT":
            if team_stats.get("assist_rate", 0.5) < 0.45:
                adjusted *= 1.2
        
        # If game is likely to be close, reduce blowout pull
        if mode_name == "BLOWOUT_PULL":
            if team_stats.get("expected_close_game", False):
                adjusted *= 0.5
        
        return adjusted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("TeamEnvironmentFailureEngine module loaded.")
