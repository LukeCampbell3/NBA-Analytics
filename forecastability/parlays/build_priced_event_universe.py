"""
PHASE 1: Build the Priced Binary-Event Universe

Inputs:
- scenario_probability_matrix_latest.csv
- forecastability_board_latest.csv
- trusted_player_state_registry_latest.csv
- current odds/line snapshots
- book-level market data
- alternate lines if available
- settled outcomes for validation mode
- injury/news snapshot if available
- starting lineup snapshot if available

Output:
- priced_event_universe_latest.csv

Each row represents one priced binary event with identity, price, distribution,
edge, reliability, scenario, and news columns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import json

from core_utils import (
    american_to_decimal,
    american_to_implied_prob,
    american_to_break_even_prob,
    edge_from_probability_and_odds,
    lcb_edge,
    robust_edge,
)
from data_types import PricedBinaryEvent, MarketFamily, Side, NewsStatus

logger = logging.getLogger(__name__)


class PricedEventUniverseBuilder:
    """
    Builds the complete priced binary-event universe from available market data
    and probability distributions.
    """
    
    def __init__(
        self,
        scenario_prob_path: str,
        forecastability_path: str,
        player_state_path: str,
        odds_snapshot_path: str,
        config: Dict[str, Any] = None
    ):
        """Initialize builder with data paths and configuration."""
        self.scenario_prob_path = scenario_prob_path
        self.forecastability_path = forecastability_path
        self.player_state_path = player_state_path
        self.odds_snapshot_path = odds_snapshot_path
        self.config = config or {}
        
        # Data storage
        self.scenario_probs = None
        self.forecastability = None
        self.player_states = None
        self.odds_snapshots = None
        
        # Results
        self.priced_events: List[PricedBinaryEvent] = []
        
    def load_data(self) -> bool:
        """Load all required input data."""
        try:
            logger.info(f"Loading scenario probabilities from {self.scenario_prob_path}")
            self.scenario_probs = pd.read_csv(self.scenario_prob_path)
            
            logger.info(f"Loading forecastability from {self.forecastability_path}")
            self.forecastability = pd.read_csv(self.forecastability_path)
            
            logger.info(f"Loading player states from {self.player_state_path}")
            self.player_states = pd.read_csv(self.player_state_path)
            
            logger.info(f"Loading odds snapshot from {self.odds_snapshot_path}")
            self.odds_snapshots = pd.read_csv(self.odds_snapshot_path)
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def build_universe(self) -> List[PricedBinaryEvent]:
        """
        Main pipeline to build the priced event universe.
        
        Returns list of PricedBinaryEvent objects.
        """
        if not self.load_data():
            logger.error("Failed to load data")
            return []
        
        logger.info("Building priced event universe...")
        
        # Iterate through odds snapshots
        for idx, row in self.odds_snapshots.iterrows():
            event = self._build_single_event(row)
            if event:
                self.priced_events.append(event)
        
        logger.info(f"Built universe with {len(self.priced_events)} priced events")
        return self.priced_events
    
    def _build_single_event(self, odds_row: pd.Series) -> Optional[PricedBinaryEvent]:
        """
        Build a single priced binary event from an odds row and supporting data.
        """
        try:
            # Extract basic info
            player_id = odds_row.get("player_id")
            game_id = odds_row.get("game_id")
            game_date = odds_row.get("game_date")
            snapshot_time_str = odds_row.get("snapshot_time")
            side_str = odds_row.get("side", "OVER")
            line = float(odds_row.get("line", 0.0))
            american_odds = float(odds_row.get("odds_american", 0.0))
            book = odds_row.get("book", "UNKNOWN")
            market_family_str = odds_row.get("market_family", "PTS")
            
            # Validate required fields
            if not all([player_id, game_id, american_odds != 0.0]):
                logger.debug(f"Skipping event with missing required fields: {odds_row.to_dict()}")
                return None
            
            # Convert types
            side = Side.OVER if side_str.upper() == "OVER" else Side.UNDER
            market_family = MarketFamily[market_family_str] if market_family_str in MarketFamily.__members__ else MarketFamily.PTS
            
            snapshot_time = datetime.fromisoformat(snapshot_time_str) if isinstance(snapshot_time_str, str) else datetime.now()
            
            # Get probability distribution from scenario matrix
            prob_dist = self._get_probability_distribution(
                player_id, game_id, market_family_str, side_str, line
            )
            
            if not prob_dist:
                logger.debug(f"No probability distribution for {player_id} {market_family_str} {side_str} {line}")
                return None
            
            # Get forecastability metrics
            forecast_metrics = self._get_forecastability_metrics(player_id, market_family_str)
            
            # Get player state
            player_state = self._get_player_state(player_id)
            
            # Calculate prices
            decimal_odds = american_to_decimal(american_odds)
            implied_prob = american_to_implied_prob(american_odds)
            break_even = american_to_break_even_prob(american_odds)
            
            # Determine side probability based on side
            if side == Side.OVER:
                p_side_raw = prob_dist.get("p_over_raw", 0.5)
                p_side_stress = prob_dist.get("p_over_stress", 0.5)
            else:
                p_side_raw = prob_dist.get("p_under_raw", 0.5)
                p_side_stress = prob_dist.get("p_under_stress", 0.5)
            
            # Calculate edges
            raw_edge_val = edge_from_probability_and_odds(p_side_raw, american_odds)
            robust_edge_val = robust_edge(raw_edge_val, stress_downshift=0.01)
            
            # LCB edge with uncertainty penalty
            lcb_edge_val = lcb_edge(
                raw_edge_val,
                uncertainty_penalty=0.005 + (forecast_metrics.get("uncertainty", 0.01) * 0.01),
                edge_fragility=prob_dist.get("edge_fragility", 0.0)
            )
            
            # Validate price status
            price_status = self._determine_price_status(american_odds, break_even, p_side_stress)
            
            # Create event
            event = PricedBinaryEvent(
                event_id=f"{game_id}_{player_id}_{market_family_str}_{side_str}_{line}_{book}",
                game_id=game_id,
                game_date=game_date,
                snapshot_time=snapshot_time,
                player_id=player_id,
                player_name=player_state.get("player_name", "UNKNOWN"),
                team=player_state.get("team", "UNKNOWN"),
                opponent=odds_row.get("opponent", "UNKNOWN"),
                book=book,
                market_type=f"PLAYER_{market_family_str}",
                side=side,
                line=line,
                is_main_line=odds_row.get("is_main_line", True),
                is_alt_line=odds_row.get("is_alt_line", False),
                is_combo_market=odds_row.get("is_combo_market", False),
                market_family=market_family,
                
                # Price
                odds_american=american_odds,
                odds_decimal=decimal_odds,
                implied_prob_raw=implied_prob,
                best_book_for_line=book,
                price_status=price_status,
                
                # Distribution
                model_mean=prob_dist.get("mean", line),
                model_std=prob_dist.get("std", 2.0),
                q10=prob_dist.get("q10", line - 2.0),
                q25=prob_dist.get("q25", line - 1.0),
                q50=prob_dist.get("q50", line),
                q75=prob_dist.get("q75", line + 1.0),
                q90=prob_dist.get("q90", line + 2.0),
                line_percentile=prob_dist.get("line_percentile", 0.5),
                p_over_raw=prob_dist.get("p_over_raw", 0.5),
                p_under_raw=prob_dist.get("p_under_raw", 0.5),
                p_side_raw=p_side_raw,
                p_side_stress=p_side_stress,
                p_side_lcb=max(0.0, p_side_stress - 0.02),
                p_push=prob_dist.get("p_push", 0.01),
                
                # Edge
                raw_edge=raw_edge_val,
                robust_edge=robust_edge_val,
                lcb_edge=lcb_edge_val,
                raw_ev=raw_edge_val,  # EV per unit wagered
                robust_ev=robust_edge_val,
                lcb_ev=lcb_edge_val,
                edge_fragility=prob_dist.get("edge_fragility", 0.0),
                
                # Reliability
                forecastability_score=forecast_metrics.get("forecastability_score", 0.70),
                plan_reliability=forecast_metrics.get("plan_reliability", 0.70),
                scenario_agreement=forecast_metrics.get("scenario_agreement", 0.68),
                management_volatility_score=forecast_metrics.get("management_volatility", 0.25),
                market_instability_score=forecast_metrics.get("market_instability", 0.25),
                chaos_score=forecast_metrics.get("chaos_score", 0.25),
                similar_state_count=forecast_metrics.get("similar_state_count", 5),
                similar_state_p80_abs_error=forecast_metrics.get("p80_error", 0.05),
                interval_width=prob_dist.get("interval_width", 4.0),
                team_rebound_rate=player_state.get("team_rebound_rate", 0.50),
                rebound_share=player_state.get("rebound_share", 0.12),
                team_shooting_efficiency_risk=player_state.get("team_shooting_efficiency_risk", 0.0),
                opponent_shooting_efficiency_risk=player_state.get("opponent_shooting_efficiency_risk", 0.0),
                wing_rebound_leakage_score=player_state.get("wing_rebound_leakage_score", 0.0),
                upper_band_line_penalty=0.0,
                
                # Scenario
                positive_state_mass=prob_dist.get("positive_mass", 0.55),
                negative_state_mass=prob_dist.get("negative_mass", 0.40),
                plan_holds_weight=prob_dist.get("plan_holds", 0.55),
                
                # News
                news_status=player_state.get("news_status", "CLEAR"),
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error building single event: {e}")
            return None
    
    def _get_probability_distribution(
        self,
        player_id: str,
        game_id: str,
        market_family: str,
        side: str,
        line: float
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve probability distribution for a player/market/line from scenario matrix.
        """
        if self.scenario_probs is None:
            return None
        
        # Filter scenario probs by player, game, market
        mask = (
            (self.scenario_probs.get("player_id") == player_id) &
            (self.scenario_probs.get("game_id") == game_id) &
            (self.scenario_probs.get("market_family") == market_family)
        )
        
        matches = self.scenario_probs[mask]
        if matches.empty:
            return None
        
        # Get row with closest line match
        closest = matches.iloc[(matches["line"] - line).abs().argsort()[0]]
        
        # Extract distribution
        return {
            "mean": float(closest.get("mean", line)),
            "std": float(closest.get("std", 2.0)),
            "q10": float(closest.get("q10", line - 2.0)),
            "q25": float(closest.get("q25", line - 1.0)),
            "q50": float(closest.get("q50", line)),
            "q75": float(closest.get("q75", line + 1.0)),
            "q90": float(closest.get("q90", line + 2.0)),
            "line_percentile": float(closest.get("line_percentile", 0.5)),
            "p_over_raw": float(closest.get("p_over_raw", 0.5)),
            "p_under_raw": float(closest.get("p_under_raw", 0.5)),
            "p_over_stress": float(closest.get("p_over_stress", 0.48)),
            "p_under_stress": float(closest.get("p_under_stress", 0.48)),
            "p_push": float(closest.get("p_push", 0.01)),
            "positive_mass": float(closest.get("positive_mass", 0.55)),
            "negative_mass": float(closest.get("negative_mass", 0.40)),
            "plan_holds": float(closest.get("plan_holds", 0.55)),
            "edge_fragility": float(closest.get("edge_fragility", 0.0)),
            "interval_width": float(closest.get("interval_width", 4.0)),
        }
    
    def _get_forecastability_metrics(
        self,
        player_id: str,
        market_family: str
    ) -> Dict[str, float]:
        """Get forecastability metrics for a player/market combination."""
        if self.forecastability is None:
            return {}
        
        mask = (
            (self.forecastability.get("player_id") == player_id) &
            (self.forecastability.get("market_family") == market_family)
        )
        
        matches = self.forecastability[mask]
        if matches.empty:
            return {}
        
        row = matches.iloc[0]
        return {
            "forecastability_score": float(row.get("forecastability_score", 0.70)),
            "plan_reliability": float(row.get("plan_reliability", 0.70)),
            "scenario_agreement": float(row.get("scenario_agreement", 0.68)),
            "management_volatility": float(row.get("management_volatility", 0.25)),
            "market_instability": float(row.get("market_instability", 0.25)),
            "chaos_score": float(row.get("chaos_score", 0.25)),
            "similar_state_count": int(row.get("similar_state_count", 5)),
            "p80_error": float(row.get("p80_error", 0.05)),
            "uncertainty": float(row.get("uncertainty", 0.01)),
        }
    
    def _get_player_state(self, player_id: str) -> Dict[str, Any]:
        """Get current player state."""
        if self.player_states is None:
            return {}
        
        mask = self.player_states.get("player_id") == player_id
        matches = self.player_states[mask]
        if matches.empty:
            return {}
        
        row = matches.iloc[0]
        return {
            "player_name": row.get("player_name", "UNKNOWN"),
            "team": row.get("team", "UNKNOWN"),
            "expected_minutes": float(row.get("expected_minutes", 30.0)),
            "news_status": row.get("news_status", "CLEAR"),
            "injury_status": row.get("injury_status", "HEALTHY"),
        }
    
    def _determine_price_status(
        self,
        american_odds: float,
        break_even_prob: float,
        stress_prob: float
    ) -> str:
        """Determine if price is valid, dependent, or should be passed."""
        from core_utils import PriceStatus
        
        if american_odds == 0.0 or not american_odds:
            return PriceStatus.MISSING_PRICE.value
        
        if stress_prob <= break_even_prob + 0.01:
            return PriceStatus.PASS_AT_PRICE.value
        
        return PriceStatus.PRICE_VALID.value
    
    def export_to_csv(self, output_path: str) -> bool:
        """Export priced events to CSV."""
        try:
            if not self.priced_events:
                logger.warning("No priced events to export")
                return False
            
            # Convert to DataFrame
            data = []
            for event in self.priced_events:
                data.append({
                    "event_id": event.event_id,
                    "game_id": event.game_id,
                    "game_date": event.game_date,
                    "snapshot_time": event.snapshot_time.isoformat(),
                    "player_id": event.player_id,
                    "player_name": event.player_name,
                    "team": event.team,
                    "opponent": event.opponent,
                    "book": event.book,
                    "market_type": event.market_type,
                    "side": event.side.value,
                    "line": event.line,
                    "is_main_line": event.is_main_line,
                    "is_alt_line": event.is_alt_line,
                    "is_combo_market": event.is_combo_market,
                    "market_family": event.market_family.value,
                    "odds_american": event.odds_american,
                    "odds_decimal": event.odds_decimal,
                    "implied_prob_raw": event.implied_prob_raw,
                    "break_even_prob": event.odds_american,  # Will calculate properly
                    "price_status": event.price_status,
                    "model_mean": event.model_mean,
                    "model_std": event.model_std,
                    "q10": event.q10,
                    "q25": event.q25,
                    "q50": event.q50,
                    "q75": event.q75,
                    "q90": event.q90,
                    "line_percentile": event.line_percentile,
                    "p_over_raw": event.p_over_raw,
                    "p_under_raw": event.p_under_raw,
                    "p_side_raw": event.p_side_raw,
                    "p_side_stress": event.p_side_stress,
                    "p_side_lcb": event.p_side_lcb,
                    "p_push": event.p_push,
                    "raw_edge": event.raw_edge,
                    "robust_edge": event.robust_edge,
                    "lcb_edge": event.lcb_edge,
                    "raw_ev": event.raw_ev,
                    "robust_ev": event.robust_ev,
                    "lcb_ev": event.lcb_ev,
                    "edge_fragility": event.edge_fragility,
                    "forecastability_score": event.forecastability_score,
                    "plan_reliability": event.plan_reliability,
                    "scenario_agreement": event.scenario_agreement,
                    "management_volatility_score": event.management_volatility_score,
                    "market_instability_score": event.market_instability_score,
                    "chaos_score": event.chaos_score,
                    "similar_state_count": event.similar_state_count,
                    "similar_state_p80_abs_error": event.similar_state_p80_abs_error,
                    "positive_state_mass": event.positive_state_mass,
                    "negative_state_mass": event.negative_state_mass,
                    "plan_holds_weight": event.plan_holds_weight,
                    "news_status": event.news_status,
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} priced events to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    builder = PricedEventUniverseBuilder(
        scenario_prob_path="outputs/scenario_probability_matrix_latest.csv",
        forecastability_path="outputs/forecastability_board_latest.csv",
        player_state_path="outputs/trusted_player_state_registry_latest.csv",
        odds_snapshot_path="outputs/current_odds_snapshot.csv",
    )
    
    events = builder.build_universe()
    builder.export_to_csv("outputs/priced_event_universe_latest.csv")
