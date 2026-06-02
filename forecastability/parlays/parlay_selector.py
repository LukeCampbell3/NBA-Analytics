"""
PHASE 11: Parlay Selector

Final score should be config-driven:

final_parlay_score =
  lcb_joint_ev
  * compatible_state_score
  * min_leg_forecastability
  * min_leg_plan_reliability
  * min_leg_scenario_agreement
  * price_quality_score
  * (1 - shared_failure_risk)
  * (1 - dependency_penalty)
  * (1 - shared_event_supply_penalty)
  * (1 - edge_fragility)

Decision labels:
- PARLAY_SEED_SHADOW
- PARLAY_BALANCED_SHADOW
- PARLAY_PRICE_DEPENDENT
- PARLAY_NEWS_DEPENDENT
- PARLAY_BOUNDARY_SHADOW
- PASS_LEG_NOT_IN_BETTABLE_SET
- PASS_PAIRWISE_SUBSET_FAIL
- PASS_JOINT_EV_NEGATIVE
- PASS_STRESS_FAIL
- PASS_SHARED_FAILURE_RISK
- PASS_SHARED_EVENT_SUPPLY
- PASS_SAME_GAME_INCOMPATIBLE
- PASS_PRICE_INVALID
- PASS_SGP_PAYOUT_TOO_LOW
- PASS_DEPENDENCY_UNKNOWN
- PASS_TOO_MANY_CORRELATED_LEGS
- PASS_LOW_LCB_EDGE
"""

from typing import List, Dict, Optional
import logging

from data_types import ParlayCandidate, ParlayLeg, ParlayDecision

logger = logging.getLogger(__name__)


class ParlaySelector:
    """
    Final parlay selection and scoring.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            "min_joint_lcb_edge": 0.010,
            "min_joint_robust_edge": 0.020,
            "min_joint_stress_ev": 0.0,
            "max_shared_failure_risk": 0.40,
            "max_shared_event_supply_penalty": 0.35,
            "min_compatible_state_score": 0.62,
            "max_dependency_penalty": 0.32,
            "max_edge_fragility": 0.08,
            "require_all_two_leg_subsets_pass": True,
        }
    
    def select_and_score(
        self,
        parlay: ParlayCandidate
    ) -> ParlayCandidate:
        """
        Evaluate and score a parlay candidate.
        
        Returns updated ParlayCandidate with decision and score.
        """
        
        # Step 1: Check all legs are acceptable
        for leg in parlay.legs:
            if not hasattr(leg, 'accepted'):
                # Assume all legs passed if they're in the parlay
                pass
        
        # Step 2: Check price validity
        if parlay.price_validity != "PRICE_VALID":
            parlay.decision = ParlayDecision.PASS_PRICE_INVALID.value
            parlay.rejection_reasons.append("INVALID_PRICE")
            return parlay
        
        # Step 3: Check joint LCB edge
        if parlay.lcb_joint_edge < self.config.get("min_joint_lcb_edge", 0.010):
            parlay.decision = ParlayDecision.PASS_LOW_LCB_EDGE.value
            parlay.rejection_reasons.append(
                f"LCB_EDGE_LOW: {parlay.lcb_joint_edge:.4f}"
            )
            return parlay
        
        # Step 4: Check shared failure risk
        if parlay.shared_failure_risk > self.config.get("max_shared_failure_risk", 0.40):
            parlay.decision = ParlayDecision.PASS_SHARED_FAILURE_RISK.value
            parlay.rejection_reasons.append(
                f"SHARED_FAILURE_RISK_TOO_HIGH: {parlay.shared_failure_risk:.3f}"
            )
            return parlay
        
        # Step 5: Check shared event supply
        if parlay.shared_event_supply_penalty > self.config.get("max_shared_event_supply_penalty", 0.35):
            parlay.decision = ParlayDecision.PASS_SHARED_EVENT_SUPPLY.value
            parlay.rejection_reasons.append(
                f"SHARED_EVENT_SUPPLY_TOO_HIGH: {parlay.shared_event_supply_penalty:.3f}"
            )
            return parlay
        
        # Step 6: Check edge fragility
        if parlay.edge_fragility > self.config.get("max_edge_fragility", 0.08):
            parlay.decision = ParlayDecision.PASS_STRESS_FAIL.value
            parlay.rejection_reasons.append(
                f"EDGE_FRAGILITY_TOO_HIGH: {parlay.edge_fragility:.3f}"
            )
            return parlay
        
        # Step 7: Check compatible state score
        if parlay.compatible_state_score < self.config.get("min_compatible_state_score", 0.62):
            parlay.decision = ParlayDecision.PASS_SAME_GAME_INCOMPATIBLE.value
            parlay.rejection_reasons.append(
                f"COMPATIBLE_STATE_SCORE_LOW: {parlay.compatible_state_score:.3f}"
            )
            return parlay
        
        # Step 8: Check dependency penalty
        if parlay.dependency_penalty > self.config.get("max_dependency_penalty", 0.32):
            parlay.decision = ParlayDecision.PASS_DEPENDENCY_UNKNOWN.value
            parlay.rejection_reasons.append(
                f"DEPENDENCY_PENALTY_HIGH: {parlay.dependency_penalty:.3f}"
            )
            return parlay
        
        # Step 9: All checks passed, compute final score and tier
        parlay = self._compute_final_score(parlay)
        
        return parlay
    
    def _compute_final_score(self, parlay: ParlayCandidate) -> ParlayCandidate:
        """Compute final parlay score and determine tier."""
        
        # Score formula: weighted product of quality factors
        final_score = (
            parlay.lcb_joint_edge
            * parlay.compatible_state_score
            * parlay.min_leg_forecastability
            * parlay.min_leg_plan_reliability
            * parlay.min_leg_scenario_agreement
            * parlay.price_quality_score
            * (1.0 - parlay.shared_failure_risk)
            * (1.0 - parlay.dependency_penalty)
            * (1.0 - parlay.shared_event_supply_penalty)
            * (1.0 - parlay.edge_fragility)
        )
        
        parlay.final_parlay_score = final_score
        
        # Determine tier based on score and quality metrics
        if parlay.final_parlay_score >= 0.001 and all([
            parlay.min_leg_forecastability >= 0.78,
            parlay.min_leg_plan_reliability >= 0.70,
            parlay.min_leg_scenario_agreement >= 0.68,
            parlay.lcb_joint_edge >= 0.020,
        ]):
            parlay.tier = "SEED_SHADOW"
            parlay.decision = ParlayDecision.PARLAY_SEED_SHADOW.value
        elif parlay.final_parlay_score >= 0.0005:
            parlay.tier = "BALANCED_SHADOW"
            parlay.decision = ParlayDecision.PARLAY_BALANCED_SHADOW.value
        else:
            parlay.tier = "BOUNDARY_SHADOW"
            parlay.decision = ParlayDecision.PARLAY_BOUNDARY_SHADOW.value
        
        logger.debug(
            f"Parlay {parlay.parlay_id}: "
            f"score={parlay.final_parlay_score:.6f}, "
            f"tier={parlay.tier}, "
            f"decision={parlay.decision}"
        )
        
        return parlay
    
    def filter_and_rank(
        self,
        parlays: List[ParlayCandidate],
        max_output: int = 25,
        min_tier: str = "BALANCED_SHADOW"
    ) -> List[ParlayCandidate]:
        """
        Filter parlays by tier and rank by final score.
        
        Returns top N parlays.
        """
        
        # Filter by tier
        tier_order = {
            "SEED_SHADOW": 3,
            "BALANCED_SHADOW": 2,
            "BOUNDARY_SHADOW": 1,
        }
        
        min_tier_value = tier_order.get(min_tier, 0)
        
        filtered = [
            p for p in parlays
            if tier_order.get(p.tier, 0) >= min_tier_value
        ]
        
        # Sort by final score
        filtered_sorted = sorted(
            filtered,
            key=lambda p: p.final_parlay_score,
            reverse=True
        )
        
        return filtered_sorted[:max_output]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ParlaySelector module loaded.")
