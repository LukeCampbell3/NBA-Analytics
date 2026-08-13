"""
PHASE 4: Anchor + Companion Parlay Construction

Do not generate parlays from all accepted legs.

Instead:
1. Select anchor legs from SEED_PLAYABLE and strong BALANCED_PLAYABLE.
2. Generate companion candidates with compatible but not identical failure modes.
3. Only allow PRICE_DEPENDENT companions if current price satisfies min_acceptable_odds.
4. Exclude NEWS_DEPENDENT legs until rerun after news clarity.
5. Allow 3-leg parlays only if every 2-leg subset passes.

Do not create Cartesian product of all legs.
"""

from typing import List, Dict, Optional
import logging

from .core_utils import LegStatus
from .data_types import SingleLegEvaluation, ParlayLeg, MarketFamily, Side

logger = logging.getLogger(__name__)


class AnchorCompanionGenerator:
    """
    Generates parlays via anchor + companion pattern,
    not Cartesian product explosion.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            "top_n_anchor_legs": 20,
            "top_n_companions_per_anchor": 8,
            "max_legs_same_game": 2,
            "min_legs_per_parlay": 2,
            "require_all_two_leg_subsets_pass": True,
            "max_legs_default": 2,
            "allow_three_leg_if_all_pairs_pass": False,
        }
    
    def generate_parlay_candidates(
        self,
        leg_evaluations: List[SingleLegEvaluation],
    ) -> List[Dict]:
        """
        Generate parlay candidates using anchor + companion pattern.
        
        Returns list of parlay candidate specifications (not full parlays).
        """
        
        # Filter for acceptable legs
        accepted = [
            leg for leg in leg_evaluations
            if leg.accepted_into_single_leg_pool
        ]
        
        if len(accepted) < 2:
            logger.info(f"Not enough accepted legs for parlays: {len(accepted)}")
            return []
        
        # Sort by LCB edge to find anchor candidates
        accepted_sorted = sorted(
            accepted,
            key=lambda x: x.lcb_edge,
            reverse=True
        )
        
        # Select top anchor candidates
        anchors = accepted_sorted[:self.config.get("top_n_anchor_legs", 20)]
        logger.info(f"Selected {len(anchors)} anchor candidates")
        
        candidates = []
        
        # For each anchor, find companion legs
        for anchor_idx, anchor in enumerate(anchors):
            companions = self._find_companions(anchor, accepted)
            
            logger.debug(
                f"Anchor {anchor_idx}: {anchor.player_name} {anchor.player_market} "
                f"{anchor.side.value} {anchor.line} - found {len(companions)} companions"
            )
            
            # Create 2-leg parlay candidates
            for comp_idx, companion in enumerate(companions):
                if comp_idx >= self.config.get("top_n_companions_per_anchor", 8):
                    break
                
                parlay_spec = {
                    "anchor_idx": len([e for e in accepted if e.event_id == anchor.event_id]),
                    "anchor_event_id": anchor.event_id,
                    "anchor_leg": anchor,
                    "companion_event_id": companion.event_id,
                    "companion_leg": companion,
                    "leg_count": 2,
                    "compatible_reasons": self._get_compatibility_reasons(anchor, companion),
                }
                
                candidates.append(parlay_spec)
        
        logger.info(f"Generated {len(candidates)} 2-leg parlay candidates")
        
        if self.config.get("allow_three_leg_if_all_pairs_pass", False):
            three_leg_candidates = self._generate_three_leg_candidates(anchors, accepted)
            candidates.extend(three_leg_candidates)
            logger.info(f"Generated {len(three_leg_candidates)} 3-leg parlay candidates")
        
        return candidates

    def _generate_three_leg_candidates(
        self,
        anchors: List[SingleLegEvaluation],
        accepted_legs: List[SingleLegEvaluation]
    ) -> List[Dict]:
        """Generate 3-leg parlay candidates when permitted."""
        three_leg_specs = []
        max_three_companions = 4
        
        for anchor_idx, anchor in enumerate(anchors):
            companions = self._find_companions(anchor, accepted_legs)
            companions = companions[:self.config.get("top_n_companions_per_anchor", 8)]
            
            if len(companions) < 2:
                continue
            
            # Build small companion combinations and verify pairwise compatibility
            for i in range(min(len(companions), max_three_companions)):
                for j in range(i + 1, min(len(companions), max_three_companions)):
                    companion_a = companions[i]
                    companion_b = companions[j]
                    
                    # Ensure all 2-leg subsets are compatible
                    if not self._is_compatible(anchor, companion_a):
                        continue
                    if not self._is_compatible(anchor, companion_b):
                        continue
                    if not self._is_compatible(companion_a, companion_b):
                        continue
                    
                    spec = {
                        "anchor_idx": anchor_idx,
                        "anchor_event_id": anchor.event_id,
                        "anchor_leg": anchor,
                        "companion_leg": companion_a,
                        "third_leg": companion_b,
                        "leg_count": 3,
                        "compatible_reasons": self._get_compatibility_reasons(anchor, companion_a)
                                            + self._get_compatibility_reasons(anchor, companion_b)
                                            + self._get_compatibility_reasons(companion_a, companion_b),
                    }
                    three_leg_specs.append(spec)
        
        return three_leg_specs
    
    def _find_companions(
        self,
        anchor: SingleLegEvaluation,
        available_legs: List[SingleLegEvaluation]
    ) -> List[SingleLegEvaluation]:
        """
        Find compatible companion legs for an anchor.
        
        Companion quality criteria:
        - compatible success scenario
        - diversified failure mode
        - positive robust EV
        - not purely dependent on same fragile state
        """
        
        companions = []
        
        # Exclude anchor itself
        candidates = [leg for leg in available_legs if leg.event_id != anchor.event_id]
        
        # Filter by basic acceptance
        candidates = [leg for leg in candidates if leg.accepted_into_single_leg_pool]
        
        # Filter by price: reject PRICE_DEPENDENT unless it meets min odds
        candidates = [
            leg for leg in candidates
            if leg.leg_status != LegStatus.PRICE_DEPENDENT.value
            or (leg.min_acceptable_odds and leg.odds_american >= leg.min_acceptable_odds)
        ]
        
        # Filter out NEWS_DEPENDENT
        candidates = [
            leg for leg in candidates
            if leg.leg_status != LegStatus.NEWS_DEPENDENT.value
        ]
        
        # Sort by LCB edge
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x.lcb_edge,
            reverse=True
        )
        
        # Additional compatibility check
        for companion in candidates_sorted:
            if self._is_compatible(anchor, companion):
                companions.append(companion)
        
        return companions
    
    def _is_compatible(
        self,
        leg_a: SingleLegEvaluation,
        leg_b: SingleLegEvaluation
    ) -> bool:
        """
        Check if two legs are compatible for parlay.
        
        Incompatibility:
        - Same player, same market, opposite sides (perfect script)
        - Same player, same market, same side (perfect redundancy)
        - Extreme correlation in failure modes
        """
        
        # Same player check
        if leg_a.player_name == leg_b.player_name:
            # If same player and same market
            if leg_a.player_market == leg_b.player_market:
                # Opposite sides (perfect script) is bad
                if leg_a.side != leg_b.side:
                    logger.debug(
                        f"Rejecting: {leg_a.player_name} {leg_a.player_market} "
                        f"opposite sides (perfect script)"
                    )
                    return False
                
                # Same side is also problematic
                logger.debug(
                    f"Rejecting: {leg_a.player_name} {leg_a.player_market} "
                    f"same side (redundant)"
                )
                return False
        
        # Different players generally OK (unless extreme correlation)
        return True
    
    def _get_compatibility_reasons(
        self,
        anchor: SingleLegEvaluation,
        companion: SingleLegEvaluation
    ) -> List[str]:
        """Get list of compatibility reasons."""
        reasons = []
        
        if anchor.player_name != companion.player_name:
            reasons.append("DIFFERENT_PLAYERS")
        
        if anchor.player_market != companion.player_market:
            reasons.append("DIFFERENT_MARKETS")
        
        if anchor.side != companion.side:
            reasons.append("OPPOSITE_SIDES")
        
        # Could add more sophisticated reasoning here
        # e.g., inverse failure mode analysis
        
        return reasons


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("AnchorCompanionGenerator module loaded.")
