"""
PHASE 7: Joint State and Correlation Engine

Detect correlations between legs using empirical data:
- Same-player market covariance
- Teammate stat residual covariance
- Same-game stat residual covariance
- Pace-adjusted correlation
- Team assist/points/rebound residual correlation
- Market-family correlation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from data_types import ParlayLeg, MarketFamily, JointState
from core_utils import CorrelationClass

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Empirical correlation matrix for player/market combinations."""
    player_id: str
    market_family: str
    correlation_data: pd.DataFrame
    sample_size: int
    confidence: float


class CorrelationEngine:
    """
    Build and apply empirical correlation matrices from historical data.
    """
    
    def __init__(self, historical_outcomes_path: Optional[str] = None):
        """
        Initialize correlation engine.
        
        historical_outcomes_path: Path to CSV with settled outcomes
            Format: game_id, player_id, market_family, side, line, actual_value
        """
        self.historical_outcomes_path = historical_outcomes_path
        self.outcomes_df = None
        self.correlation_matrices = {}
        
        if historical_outcomes_path:
            self.load_historical_outcomes(historical_outcomes_path)
    
    def load_historical_outcomes(self, path: str) -> bool:
        """Load historical outcomes from CSV."""
        try:
            self.outcomes_df = pd.read_csv(path)
            logger.info(f"Loaded {len(self.outcomes_df)} historical outcomes")
            return True
        except Exception as e:
            logger.error(f"Failed to load historical outcomes: {e}")
            return False
    
    def compute_leg_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Dict:
        """
        Compute empirical correlation between two legs.
        
        Returns:
            {
                "correlation_coefficient": -1.0 to 1.0,
                "correlation_class": "...",
                "sample_size": int,
                "confidence": 0.0-1.0,
                "interpretation": "...",
            }
        """
        
        if not self.outcomes_df or self.outcomes_df.empty:
            return self._default_correlation(leg1, leg2)
        
        same_player = leg1.player_name == leg2.player_name
        same_game = leg1.game_id == leg2.game_id
        
        if same_player and not same_game:
            corr = self._compute_same_player_correlation(leg1, leg2)
        elif same_game:
            corr = self._compute_same_game_correlation(leg1, leg2)
        else:
            corr = self._compute_cross_game_correlation(leg1, leg2)
        
        return corr

    def compute_joint_state(self, legs: List[ParlayLeg]) -> JointState:
        """Compute a joint-state summary for a set of parlay legs."""
        state = JointState()
        if not legs or len(legs) < 2:
            return state
        
        correlations = []
        confidences = []
        dependency_classes = set()

        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                corr_result = self.compute_leg_correlation(legs[i], legs[j])
                correlations.append(corr_result.get("correlation_coefficient", 0.0))
                confidences.append(corr_result.get("confidence", 0.0))
                dependency_classes.add(corr_result.get("correlation_class", CorrelationClass.UNKNOWN_DEPENDENCE.value))
        
        state.empirical_correlation = float(np.mean(correlations)) if correlations else 0.0
        state.correlation_confidence = float(np.mean(confidences)) if confidences else 0.0
        state.dependency_classes = sorted(list(dependency_classes))
        
        return state
    
    def _compute_same_player_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Dict:
        """Compute correlation for same player across different games."""
        
        # Find outcomes for this player's markets
        player_outcomes = self.outcomes_df[
            self.outcomes_df["player_id"] == leg1.player_id
        ]
        
        if player_outcomes.empty or len(player_outcomes) < 10:
            return self._default_correlation(leg1, leg2)
        
        # Group by game and compute stats
        market1 = leg1.market_family if isinstance(leg1.market_family, str) else leg1.market_family.value
        market2 = leg2.market_family if isinstance(leg2.market_family, str) else leg2.market_family.value
        
        outcomes_m1 = player_outcomes[player_outcomes["market_family"] == market1]
        outcomes_m2 = player_outcomes[player_outcomes["market_family"] == market2]
        
        if outcomes_m1.empty or outcomes_m2.empty:
            return self._default_correlation(leg1, leg2)
        
        # Merge on game_id to find common games
        merged = pd.merge(
            outcomes_m1[["game_id", "actual_value"]],
            outcomes_m2[["game_id", "actual_value"]],
            on="game_id",
            suffixes=("_m1", "_m2")
        )
        
        if len(merged) < 5:
            return self._default_correlation(leg1, leg2)
        
        # Compute correlation
        correlation = merged["actual_value_m1"].corr(merged["actual_value_m2"])
        
        # Confidence based on sample size
        confidence = min(1.0, len(merged) / 50.0)
        
        # Classify
        if correlation > 0.1:
            corr_class = CorrelationClass.MARKET_CORRELATED.value
        else:
            corr_class = CorrelationClass.CROSS_GAME_WEAK_DEPENDENCE.value
        
        return {
            "correlation_coefficient": correlation if not np.isnan(correlation) else 0.0,
            "correlation_class": corr_class,
            "sample_size": len(merged),
            "confidence": confidence,
            "interpretation": f"Same player, {market1} vs {market2}, n={len(merged)} games",
        }
    
    def _compute_same_game_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Dict:
        """Compute correlation for legs in the same game."""
        
        game_outcomes = self.outcomes_df[
            self.outcomes_df["game_id"] == leg1.game_id
        ]
        
        if game_outcomes.empty:
            # Use typical same-game correlations
            market1 = leg1.market_family if isinstance(leg1.market_family, str) else leg1.market_family.value
            market2 = leg2.market_family if isinstance(leg2.market_family, str) else leg2.market_family.value
            
            return self._typical_same_game_correlation(market1, market2)
        
        # Find outcomes for both legs
        leg1_outcomes = game_outcomes[
            (game_outcomes["player_id"] == leg1.player_name) &
            (game_outcomes["market_family"] == leg1.market_family)
        ]
        
        leg2_outcomes = game_outcomes[
            (game_outcomes["player_id"] == leg2.player_name) &
            (game_outcomes["market_family"] == leg2.market_family)
        ]
        
        if len(leg1_outcomes) == 0 or len(leg2_outcomes) == 0:
            return self._typical_same_game_correlation(
                leg1.market_family if isinstance(leg1.market_family, str) else leg1.market_family.value,
                leg2.market_family if isinstance(leg2.market_family, str) else leg2.market_family.value
            )
        
        # Same player?
        if leg1.player_name == leg2.player_name:
            # Could be positive (usage effects) or negative (trade-off)
            # Example: high PTS usually means high AST (positive)
            # But high PTS can mean fewer REB attempts (negative)
            return self._same_player_same_game_correlation(
                leg1.market_family if isinstance(leg1.market_family, str) else leg1.market_family.value,
                leg2.market_family if isinstance(leg2.market_family, str) else leg2.market_family.value
            )
        else:
            # Different players, same game: consider pace effects
            return self._different_player_same_game_correlation(
                leg1.market_family if isinstance(leg1.market_family, str) else leg1.market_family.value,
                leg2.market_family if isinstance(leg2.market_family, str) else leg2.market_family.value
            )
    
    def _compute_cross_game_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Dict:
        """Compute correlation for different games, different players."""
        
        # Cross-game correlation should be very weak
        # Could apply market-family correlation if available
        
        market1 = leg1.market_family if isinstance(leg1.market_family, str) else leg1.market_family.value
        market2 = leg2.market_family if isinstance(leg2.market_family, str) else leg2.market_family.value
        
        # Market-family correlation: PTS/AST tend to be uncorrelated
        market_corr = self._market_family_correlation(market1, market2)
        
        return {
            "correlation_coefficient": market_corr,
            "correlation_class": CorrelationClass.CROSS_GAME_WEAK_DEPENDENCE.value,
            "sample_size": 0,
            "confidence": 0.3,  # Very low confidence
            "interpretation": f"Different games, {market1} vs {market2}, using typical correlation",
        }
    
    def _default_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Dict:
        """Default correlation when no historical data."""
        
        market1 = leg1.market_family if isinstance(leg1.market_family, str) else leg1.market_family.value
        market2 = leg2.market_family if isinstance(leg2.market_family, str) else leg2.market_family.value
        
        if leg1.player_name == leg2.player_name:
            if leg1.game_id == leg2.game_id:
                return self._same_player_same_game_correlation(market1, market2)
            else:
                return self._same_player_different_game_correlation(market1, market2)
        else:
            if leg1.game_id == leg2.game_id:
                return self._different_player_same_game_correlation(market1, market2)
            else:
                return self._cross_game_correlation(market1, market2)
    
    def _same_player_same_game_correlation(self, market1: str, market2: str) -> Dict:
        """Typical correlation for same player, same game."""
        
        # Market combinations
        if (market1 in ["PTS", "PR", "PRA"] and market2 in ["PTS", "PR", "PRA"]):
            # PTS and PRA are highly correlated
            corr = 0.65
            corr_class = CorrelationClass.SAME_PLAYER_OVERLAP.value
        elif (market1 in ["AST", "PA", "PRA"] and market2 in ["AST", "PA", "PRA"]):
            # AST markets moderately correlated
            corr = 0.40
            corr_class = CorrelationClass.SAME_TEAM_ASSIST_SCORER_POSITIVE.value
        elif (market1 in ["REB", "RA", "PRA"] and market2 in ["REB", "RA", "PRA"]):
            # REB markets moderately correlated
            corr = 0.35
            corr_class = CorrelationClass.SAME_PLAYER_OVERLAP.value
        elif market1 == "PTS" and market2 == "AST":
            # PTS and AST weakly positive
            corr = 0.15
            corr_class = CorrelationClass.SAME_TEAM_ASSIST_SCORER_POSITIVE.value
        elif market1 == "REB" and market2 == "AST":
            # REB and AST uncorrelated or slightly negative
            corr = -0.05
            corr_class = CorrelationClass.SAME_PLAYER_OVERLAP.value
        else:
            corr = 0.0
            corr_class = CorrelationClass.UNKNOWN_DEPENDENCE.value
        
        return {
            "correlation_coefficient": corr,
            "correlation_class": corr_class,
            "sample_size": 0,
            "confidence": 0.6,
            "interpretation": f"Same player, same game, {market1} vs {market2}",
        }
    
    def _same_player_different_game_correlation(self, market1: str, market2: str) -> Dict:
        """Typical correlation for same player, different games."""
        
        # Very weak across games (independent performances)
        corr = 0.05
        
        return {
            "correlation_coefficient": corr,
            "correlation_class": CorrelationClass.MARKET_CORRELATED.value,
            "sample_size": 0,
            "confidence": 0.4,
            "interpretation": f"Same player, different games, {market1} vs {market2}",
        }
    
    def _different_player_same_game_correlation(self, market1: str, market2: str) -> Dict:
        """Typical correlation for different players, same game."""
        
        # Same game means shared pace/blowout effects
        if market1 in ["PTS", "AST", "REB"] and market2 in ["PTS", "AST", "REB"]:
            # Pace effects: both points legs in same game positively correlated
            corr = 0.12
            corr_class = CorrelationClass.SAME_GAME_PACE_POSITIVE.value
        else:
            corr = 0.05
            corr_class = CorrelationClass.UNKNOWN_DEPENDENCE.value
        
        return {
            "correlation_coefficient": corr,
            "correlation_class": corr_class,
            "sample_size": 0,
            "confidence": 0.5,
            "interpretation": f"Different players, same game, {market1} vs {market2}",
        }
    
    def _typical_same_game_correlation(self, market1: str, market2: str) -> Dict:
        """Typical same-game correlation."""
        
        return self._different_player_same_game_correlation(market1, market2)
    
    def _cross_game_correlation(self, market1: str, market2: str) -> Dict:
        """Cross-game correlation (very weak)."""
        
        return {
            "correlation_coefficient": 0.0,
            "correlation_class": CorrelationClass.CROSS_GAME_WEAK_DEPENDENCE.value,
            "sample_size": 0,
            "confidence": 0.2,
            "interpretation": f"Cross-game, {market1} vs {market2}",
        }
    
    def _market_family_correlation(self, market1: str, market2: str) -> float:
        """Market family correlation matrix."""
        
        correlations = {
            ("PTS", "PTS"): 1.0,
            ("PTS", "AST"): 0.15,
            ("PTS", "REB"): 0.10,
            ("PTS", "3PM"): 0.50,
            ("AST", "AST"): 1.0,
            ("AST", "REB"): -0.05,
            ("REB", "REB"): 1.0,
            ("PRA", "PRA"): 1.0,
            ("PR", "RA"): 0.20,
        }
        
        # Check both directions
        corr = correlations.get((market1, market2))
        if corr is None:
            corr = correlations.get((market2, market1))
        
        if corr is None:
            corr = 0.0
        
        return corr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("CorrelationEngine module loaded.")
