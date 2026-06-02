"""
PHASE 9: Parlay Probability Engine

Do not blindly multiply raw probabilities.

For cross-game:
p_joint_naive = product(p_leg_stress)
Then adjust for:
- global calibration penalty
- dependency penalty
- market correlation
- news/systematic uncertainty
- parlay model uncertainty

For same-game:
Use scenario simulation when possible.
P(parlay hits) = sum_s P(s) * P(all legs hit | s)

If scenario simulation is unavailable:
- use pairwise correlation adjustment
- apply shared event-supply penalty
- apply shared failure penalty
- use lower-bound probability
"""

from typing import List, Dict, Optional
import logging
import math

from data_types import ParlayLeg, JointState

logger = logging.getLogger(__name__)


class ParlayProbabilityEngine:
    """
    Computes joint probability for parlay legs.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            "calibration_penalty": 0.01,
            "dependency_uncertainty_penalty": 0.02,
            "shared_event_supply_penalty": 0.02,
            "min_correlation": -0.3,
            "max_correlation": 0.3,
        }
    
    def compute_joint_probability(
        self,
        legs: List[ParlayLeg],
        joint_state: Optional[JointState] = None,
        empirical_correlation: Optional[float] = None,
        shared_event_supply_penalty: float = 0.0,
        same_game: bool = False,
    ) -> Dict:
        """
        Compute joint probability from individual leg probabilities.
        
        Returns:
            {
                "p_joint_naive": product of individual probabilities,
                "p_joint_adjusted": with correlation/dependency adjustments,
                "p_joint_stress": conservative estimate,
                "p_joint_lcb": lower confidence bound,
                "joint_probability_confidence": 0.0-1.0,
            }
        """
        
        if not legs:
            return None
        
        # Extract individual probabilities
        p_stress = [leg.p_stress for leg in legs]
        p_lcb = [leg.p_lcb for leg in legs]
        
        # Naive product (independence assumption)
        p_naive = 1.0
        for p in p_stress:
            p_naive *= p
        
        # Prefer empirical correlation from joint state when available
        if empirical_correlation is None and joint_state is not None:
            empirical_correlation = joint_state.empirical_correlation

        # Compute adjusted probability with dependency penalty
        if same_game:
            p_adjusted = self._adjust_same_game(p_stress, empirical_correlation)
        else:
            p_adjusted = self._adjust_cross_game(p_stress, empirical_correlation)
        
        # Apply shared event supply penalty
        p_adjusted_with_supply = p_adjusted * (1.0 - shared_event_supply_penalty)
        
        # Apply calibration penalty
        calibration_penalty = self.config.get("calibration_penalty", 0.01)
        p_stress_with_cal = max(0.0, p_adjusted_with_supply - calibration_penalty)
        
        # LCB: apply additional conservatism
        p_lcb_adjusted = 1.0
        for p in p_lcb:
            p_lcb_adjusted *= p
        
        p_lcb_adjusted = max(0.0, p_lcb_adjusted - calibration_penalty)
        
        # Confidence: based on leg count and correlation uncertainty
        confidence = self._compute_confidence(len(legs), same_game, empirical_correlation)
        
        return {
            "p_joint_naive": p_naive,
            "p_joint_adjusted": p_adjusted_with_supply,
            "p_joint_stress": p_stress_with_cal,
            "p_joint_lcb": p_lcb_adjusted,
            "joint_probability_confidence": confidence,
        }
    
    def _adjust_same_game(
        self,
        p_stress: List[float],
        empirical_correlation: Optional[float] = None
    ) -> float:
        """
        Adjust for same-game correlation.
        
        Same-game correlation can be positive (both hit in blowout scenario)
        or negative (one hits at expense of other).
        """
        
        if len(p_stress) < 2:
            return p_stress[0] if p_stress else 0.0
        
        # Start with naive
        p_product = 1.0
        for p in p_stress:
            p_product *= p
        
        # Apply correlation adjustment
        if empirical_correlation is not None:
            # Empirical correlation available: use it
            correlation = max(
                self.config.get("min_correlation", -0.3),
                min(empirical_correlation, self.config.get("max_correlation", 0.3))
            )
            
            # Positive correlation increases joint probability
            # Negative correlation decreases it
            adjustment = 1.0 + (correlation * 0.1)  # 10% swing per +/- 1.0 correlation
            p_adjusted = p_product * adjustment
        else:
            # No empirical correlation: apply conservative adjustment
            dependency_penalty = self.config.get("dependency_uncertainty_penalty", 0.02)
            p_adjusted = p_product - dependency_penalty
        
        return max(0.0, p_adjusted)
    
    def _adjust_cross_game(
        self,
        p_stress: List[float],
        empirical_correlation: Optional[float] = None
    ) -> float:
        """
        Adjust for cross-game correlation.
        
        Cross-game should be weakly correlated, but apply light adjustment
        for market-wide shocks, systematic biases, etc.
        """
        
        if len(p_stress) < 2:
            return p_stress[0] if p_stress else 0.0
        
        # Start with naive
        p_product = 1.0
        for p in p_stress:
            p_product *= p
        
        # Apply light cross-game adjustment
        if empirical_correlation is not None:
            # Small adjustment for empirical correlation
            adjustment = 1.0 + (empirical_correlation * 0.05)  # 5% swing per +/- 1.0
            p_adjusted = p_product * adjustment
        else:
            # Very light penalty for cross-game
            p_adjusted = p_product - 0.002 * len(p_stress)
        
        return max(0.0, p_adjusted)
    
    def _compute_confidence(
        self,
        leg_count: int,
        same_game: bool,
        empirical_correlation: Optional[float]
    ) -> float:
        """
        Compute confidence in joint probability estimate.
        
        Lower confidence with:
        - More legs (combinatorial explosion of uncertainty)
        - Same-game (more complex dependencies)
        - No empirical correlation data
        """
        
        # Base confidence
        confidence = 0.8
        
        # Reduce for additional legs
        confidence -= (leg_count - 2) * 0.05  # -5% per leg beyond 2
        
        # Reduce for same-game
        if same_game:
            confidence -= 0.1
        
        # Reduce if no empirical correlation
        if empirical_correlation is None:
            confidence -= 0.05
        
        return max(0.5, confidence)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ParlayProbabilityEngine module loaded.")
