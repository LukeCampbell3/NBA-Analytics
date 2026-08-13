"""
PHASE 5: Shared Event-Supply Engine

Detect when legs draw from the same limited event pool.

Event-supply pools:
- team rebounds
- opponent missed shots
- team assists
- made shots
- three-point attempts
- steals/turnovers
- blocks/rim attempts
- free throws/foul environment
- usage/shot attempts
- pace/possessions
- garbage time
"""

from typing import List, Tuple, Dict, Set
import logging

from .data_types import ParlayLeg, MarketFamily

logger = logging.getLogger(__name__)


class SharedEventSupplyEngine:
    """
    Detects shared event-supply risk between parlay legs.
    """
    
    # Define event supply pools
    REBOUNDS_POOL = {"REB"}
    ASSISTS_POOL = {"AST", "PA", "PRA"}
    POINTS_POOL = {"PTS", "PR", "PRA"}
    THREE_POOL = {"THREES"}
    STEALS_POOL = {"STL"}
    BLOCKS_POOL = {"BLK"}
    FREE_THROWS_POOL = {"FTA", "FTM"}
    TURNOVERS_POOL = {"TO"}
    
    # Event supply groups
    EVENT_SUPPLY_GROUPS = [
        ("REBOUNDS", REBOUNDS_POOL),
        ("ASSISTS", ASSISTS_POOL),
        ("POINTS", POINTS_POOL),
        ("THREE_POINTERS", THREE_POOL),
        ("STEALS", STEALS_POOL),
        ("BLOCKS", BLOCKS_POOL),
        ("FREE_THROWS", FREE_THROWS_POOL),
        ("TURNOVERS", TURNOVERS_POOL),
    ]
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def analyze_legs(self, legs: List[ParlayLeg]) -> Dict:
        """
        Analyze shared event supply risk between legs.
        
        Returns:
            {
                "shared_event_supply_pools": ["POOL1", "POOL2"],
                "shared_event_supply_score": 0.0-1.0,
                "shared_event_supply_penalty": 0.0-1.0,
                "conflicts": [...],
                "same_event_supply_rejection_flag": bool,
            }
        """
        
        shared_pools = set()
        conflict_pairs = []
        
        # Find all markets represented
        market_families = [leg.market_family for leg in legs]
        
        # Check for shared event pools
        for pool_name, pool_markets in self.EVENT_SUPPLY_GROUPS:
            matching_legs = [i for i, leg in enumerate(legs) if leg.market_family in pool_markets]
            
            if len(matching_legs) > 1:
                shared_pools.add(pool_name)
                
                # Log conflicts
                for i in range(len(matching_legs)):
                    for j in range(i + 1, len(matching_legs)):
                        idx_i = matching_legs[i]
                        idx_j = matching_legs[j]
                        
                        # Check if same player
                        same_player = legs[idx_i].player_name == legs[idx_j].player_name
                        
                        conflict_pairs.append({
                            "leg_i": idx_i,
                            "leg_j": idx_j,
                            "pool": pool_name,
                            "same_player": same_player,
                            "conflict_type": "SAME_PLAYER_SAME_POOL" if same_player else "SAME_POOL_DIFFERENT_PLAYERS",
                        })
        
        # Compute shared supply score (0 = no conflicts, 1 = all legs from same pool)
        if not legs:
            supply_score = 0.0
        else:
            supply_score = len(shared_pools) / max(len(self.EVENT_SUPPLY_GROUPS), 1)
        
        # Compute penalty based on conflicts
        penalty = self._compute_penalty(conflict_pairs, len(legs))
        
        # Determine rejection flag
        rejection_flag = self._should_reject(shared_pools, conflict_pairs, legs)
        
        return {
            "shared_event_supply_pools": list(shared_pools),
            "shared_event_supply_score": supply_score,
            "shared_event_supply_penalty": penalty,
            "conflicts": conflict_pairs,
            "same_event_supply_rejection_flag": rejection_flag,
        }
    
    def _compute_penalty(self, conflicts: List[Dict], leg_count: int) -> float:
        """
        Compute penalty based on number and severity of conflicts.
        
        Penalty increases with:
        - Same player in same pool
        - Multiple conflicts
        - More legs involved
        """
        
        if not conflicts:
            return 0.0
        
        # Base penalty per conflict
        base_penalty = 0.05
        
        # Severity multiplier for same-player conflicts
        same_player_conflicts = sum(1 for c in conflicts if c["same_player"])
        severity = 1.0 + (same_player_conflicts * 0.15)
        
        # Scale by leg count
        leg_scale = leg_count / 2.0  # Base scale for 2 legs
        
        penalty = min(0.5, base_penalty * len(conflicts) * severity * leg_scale)
        return penalty
    
    def _should_reject(
        self,
        shared_pools: Set[str],
        conflicts: List[Dict],
        legs: List[ParlayLeg]
    ) -> bool:
        """
        Determine if parlay should be rejected based on event supply conflict.
        """
        
        # Multiple pools from same team suggests usage collision
        if len(shared_pools) > 2:
            logger.debug(f"Rejecting: Too many shared pools ({len(shared_pools)})")
            return True
        
        # Same player in multiple supply pools is risky
        same_player_conflicts = [c for c in conflicts if c["same_player"]]
        if len(same_player_conflicts) > 1:
            logger.debug(f"Rejecting: Same player in {len(same_player_conflicts)} pools")
            return True
        
        # Multiple different-player conflicts in same pool suggest limited supply
        conflicts_by_pool = {}
        for c in conflicts:
            pool = c["pool"]
            if pool not in conflicts_by_pool:
                conflicts_by_pool[pool] = []
            conflicts_by_pool[pool].append(c)
        
        for pool, pool_conflicts in conflicts_by_pool.items():
            different_player_conflicts = [c for c in pool_conflicts if not c["same_player"]]
            if len(different_player_conflicts) > 1:
                logger.debug(f"Rejecting: Multiple different-player conflicts in {pool}")
                return True
        
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SharedEventSupplyEngine module loaded.")
