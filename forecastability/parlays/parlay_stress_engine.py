"""
PHASE 10: Parlay Stress Engine

Stress dimensions:
- lower plan-holds probability
- increase blowout risk
- increase foul trouble
- increase minutes loss
- increase role shift
- increase team offense collapse
- increase poor shooting environment
- increase rebound supply collapse
- increase market instability
- increase dependency penalty
- widen uncertainty intervals
- apply calibration penalty
- apply shared event-supply penalty
- apply shared failure penalty

Stress modes:
- mild
- severe
- worst_allowed_before_rejection
"""

from typing import List, Dict
import logging

from core_utils import edge_from_probability_and_odds
from data_types import ParlayLeg

logger = logging.getLogger(__name__)


class ParlayStressEngine:
    """
    Applies stress testing to parlay probability and EV.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            # Individual stress factors
            "plan_holds_downshift": 0.10,
            "blowout_upshift": 0.05,
            "foul_trouble_upshift": 0.03,
            "minutes_loss_upshift": 0.05,
            "role_shift_upshift": 0.04,
            "team_offense_collapse_upshift": 0.05,
            "poor_shooting_upshift": 0.04,
            "rebound_supply_collapse_upshift": 0.04,
            
            # Parlay stress factors
            "calibration_penalty": 0.01,
            "dependency_uncertainty_penalty": 0.02,
            "shared_event_supply_penalty": 0.02,
            
            # Acceptance thresholds
            "min_joint_stress_ev": 0.0,
            "min_joint_lcb_edge": 0.010,
        }
    
    def stress_test_parlay(
        self,
        p_joint_adjusted: float,
        parlay_break_even_prob: float,
        lcb_joint_edge: float,
        robust_joint_edge: float,
        shared_failure_risk: float = 0.0,
        shared_event_supply_penalty: float = 0.0,
        edge_fragility: float = 0.0,
        stress_mode: str = "severe"
    ) -> Dict:
        """
        Apply stress testing to parlay probability and EV.
        
        Returns:
            {
                "p_joint_stress": stressed probability,
                "stress_ev": stressed EV,
                "passes_stress": bool,
                "failure_mode": "...",
                "margin_over_breakeven": 0.XX,
            }
        """
        
        # Apply stress downshift
        stress_factor = self._get_stress_factor(stress_mode)
        p_stress = max(0.0, p_joint_adjusted - stress_factor)
        
        # Apply shared failure risk
        p_stress_with_failure = p_stress * (1.0 - shared_failure_risk)
        
        # Apply shared event supply penalty
        p_stress_final = p_stress_with_failure * (1.0 - shared_event_supply_penalty)
        
        # Calculate stressed EV
        decimal_odds = 1.0 / parlay_break_even_prob if parlay_break_even_prob > 0 else 0.0
        stress_ev = (p_stress_final * decimal_odds) - 1.0 if decimal_odds > 0 else 0.0
        # Margin over break-even
        margin = p_stress_final - parlay_break_even_prob
        
        # Determine acceptance
        passes = (
            p_stress_final > parlay_break_even_prob + 0.01  # 1% cushion
            and margin > 0.0
            and lcb_joint_edge >= self.config.get("min_joint_lcb_edge", 0.010)
        )
        
        failure_mode = "NONE" if passes else self._determine_failure_mode(
            p_stress_final, parlay_break_even_prob, lcb_joint_edge, edge_fragility
        )
        
        return {
            "p_joint_stress": p_stress_final,
            "stress_ev": None,  # TODO: compute properly
            "passes_stress": passes,
            "failure_mode": failure_mode,
            "margin_over_breakeven": margin,
            "shared_failure_risk_applied": shared_failure_risk,
            "shared_event_supply_penalty_applied": shared_event_supply_penalty,
        }
    
    def _get_stress_factor(self, stress_mode: str) -> float:
        """Get total stress downshift based on mode."""
        
        factors = {
            "plan_holds_downshift": self.config.get("plan_holds_downshift", 0.10),
            "blowout_upshift": self.config.get("blowout_upshift", 0.05),
            "foul_trouble_upshift": self.config.get("foul_trouble_upshift", 0.03),
            "minutes_loss_upshift": self.config.get("minutes_loss_upshift", 0.05),
        }
        
        if stress_mode == "mild":
            # Apply only main factors
            total = factors["plan_holds_downshift"] * 0.5
        elif stress_mode == "severe":
            # Apply all factors
            total = (
                factors["plan_holds_downshift"]
                + factors["blowout_upshift"]
                + factors["foul_trouble_upshift"]
                + factors["minutes_loss_upshift"]
            )
        elif stress_mode == "worst_allowed":
            # Max stress but still needs to maintain some edge
            total = sum(factors.values()) * 1.2
        else:
            total = factors["plan_holds_downshift"]
        
        return min(0.5, total)  # Cap at 50% downshift
    
    def _determine_failure_mode(
        self,
        p_stress: float,
        break_even: float,
        lcb_edge: float,
        edge_fragility: float
    ) -> str:
        """Determine which stress factor caused failure."""
        
        if p_stress <= break_even:
            return "PROBABILITY_BELOW_BREAKEVEN"
        
        if lcb_edge < self.config.get("min_joint_lcb_edge", 0.010):
            return "LCB_EDGE_INSUFFICIENT"
        
        if edge_fragility > 0.08:
            return "EDGE_TOO_FRAGILE"
        
        return "UNKNOWN_FAILURE"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ParlayStressEngine module loaded.")
