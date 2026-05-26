"""
PHASE 2: Line-Zone and Alternate-Line Scanner

Purpose:
The system must search for the best binary framing of a player-state,
not simply accept the main over/under.

For every player and market family, scan:
- main line over
- main line under
- alternate overs
- alternate unders
- combo markets
- opposite side
- ladder lines if available
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from core_utils import (
    american_to_break_even_prob,
    edge_from_probability_and_odds,
    lcb_edge,
    robust_edge,
)
from data_types import PricedBinaryEvent, Side

logger = logging.getLogger(__name__)


class LineZoneClassification:
    """Classification of a line within the distribution."""
    NEAR_MEDIAN = "LINE_NEAR_MEDIAN"
    OUTSIDE_MEDIAN_BAND = "LINE_OUTSIDE_MEDIAN_BAND"
    TAIL_PRICE_REQUIRED = "LINE_TAIL_PRICE_REQUIRED"
    PRICE_MISPLACED = "LINE_PRICE_MISPLACED"
    TOO_EXPENSIVE = "LINE_TOO_EXPENSIVE"
    UNAVAILABLE = "LINE_UNAVAILABLE"


class LineZoneScanner:
    """
    Scans all available lines for each player/market,
    finds best binary framing by robust EV or LCB edge.
    """
    
    def __init__(self, priced_events: List[PricedBinaryEvent], config: Dict = None):
        self.priced_events = priced_events
        self.config = config or {}
        
    def scan_universe(self) -> Dict[str, List[Dict]]:
        """
        Return best binary framings for each player/market combination.
        
        Returns:
            {
                "PLAYER_ID_MARKET_FAMILY": [
                    {
                        "side": OVER|UNDER,
                        "line": X,
                        "odds_american": Y,
                        "classification": "LINE_ZONE_CLASS",
                        "best_by_robust_ev": bool,
                        "best_by_lcb_edge": bool,
                        "reason": "explanation",
                    }
                ]
            }
        """
        
        results = {}
        
        # Group events by player/market
        grouped = self._group_by_player_market()
        
        for player_market, events in grouped.items():
            logger.debug(f"Scanning {player_market}: {len(events)} events")
            best_framings = self._find_best_framings(events)
            
            if best_framings:
                results[player_market] = best_framings
        
        return results
    
    def _group_by_player_market(self) -> Dict[str, List[PricedBinaryEvent]]:
        """Group events by player_id and market_family."""
        grouped = {}
        for event in self.priced_events:
            key = f"{event.player_id}_{event.market_family.value}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(event)
        return grouped
    
    def _find_best_framings(self, events: List[PricedBinaryEvent]) -> List[Dict]:
        """
        For a player/market, find the best framings by EV and edge.
        """
        results = []
        
        if not events:
            return results
        
        # Separate by side
        overs = [e for e in events if e.side == Side.OVER]
        unders = [e for e in events if e.side == Side.UNDER]
        
        # Find best over by robust EV
        best_over_ev = max(overs, key=lambda e: e.robust_ev) if overs else None
        
        # Find best over by LCB edge
        best_over_lcb = max(overs, key=lambda e: e.lcb_edge) if overs else None
        
        # Find best under by robust EV
        best_under_ev = max(unders, key=lambda e: e.robust_ev) if unders else None
        
        # Find best under by LCB edge
        best_under_lcb = max(unders, key=lambda e: e.lcb_edge) if unders else None
        
        # Collect results
        seen = set()
        
        if best_over_ev and best_over_ev.lcb_edge > 0:
            key = (best_over_ev.side, best_over_ev.line, best_over_ev.book)
            if key not in seen:
                results.append({
                    "side": best_over_ev.side.value,
                    "line": best_over_ev.line,
                    "odds_american": best_over_ev.odds_american,
                    "book": best_over_ev.book,
                    "classification": self._classify_line(best_over_ev),
                    "best_by_robust_ev": True,
                    "robust_ev": best_over_ev.robust_ev,
                    "lcb_edge": best_over_ev.lcb_edge,
                    "forecastability": best_over_ev.forecastability_score,
                    "reason": f"Best OVER by robust EV ({best_over_ev.robust_ev:.4f})",
                })
                seen.add(key)
        
        if best_over_lcb and best_over_lcb.lcb_edge > 0:
            key = (best_over_lcb.side, best_over_lcb.line, best_over_lcb.book)
            if key not in seen:
                results.append({
                    "side": best_over_lcb.side.value,
                    "line": best_over_lcb.line,
                    "odds_american": best_over_lcb.odds_american,
                    "book": best_over_lcb.book,
                    "classification": self._classify_line(best_over_lcb),
                    "best_by_lcb_edge": True,
                    "robust_ev": best_over_lcb.robust_ev,
                    "lcb_edge": best_over_lcb.lcb_edge,
                    "forecastability": best_over_lcb.forecastability_score,
                    "reason": f"Best OVER by LCB edge ({best_over_lcb.lcb_edge:.4f})",
                })
                seen.add(key)
        
        if best_under_ev and best_under_ev.lcb_edge > 0:
            key = (best_under_ev.side, best_under_ev.line, best_under_ev.book)
            if key not in seen:
                results.append({
                    "side": best_under_ev.side.value,
                    "line": best_under_ev.line,
                    "odds_american": best_under_ev.odds_american,
                    "book": best_under_ev.book,
                    "classification": self._classify_line(best_under_ev),
                    "best_by_robust_ev": True,
                    "robust_ev": best_under_ev.robust_ev,
                    "lcb_edge": best_under_ev.lcb_edge,
                    "forecastability": best_under_ev.forecastability_score,
                    "reason": f"Best UNDER by robust EV ({best_under_ev.robust_ev:.4f})",
                })
                seen.add(key)
        
        if best_under_lcb and best_under_lcb.lcb_edge > 0:
            key = (best_under_lcb.side, best_under_lcb.line, best_under_lcb.book)
            if key not in seen:
                results.append({
                    "side": best_under_lcb.side.value,
                    "line": best_under_lcb.line,
                    "odds_american": best_under_lcb.odds_american,
                    "book": best_under_lcb.book,
                    "classification": self._classify_line(best_under_lcb),
                    "best_by_lcb_edge": True,
                    "robust_ev": best_under_lcb.robust_ev,
                    "lcb_edge": best_under_lcb.lcb_edge,
                    "forecastability": best_under_lcb.forecastability_score,
                    "reason": f"Best UNDER by LCB edge ({best_under_lcb.lcb_edge:.4f})",
                })
                seen.add(key)
        
        return results
    
    def _classify_line(self, event: PricedBinaryEvent) -> str:
        """Classify a line within its zone."""
        
        # If line is within 0.5 of median
        if abs(event.line - event.q50) < 0.5:
            return LineZoneClassification.NEAR_MEDIAN
        
        # If outside median band but not in tails
        if abs(event.line - event.q50) < 1.5:
            return LineZoneClassification.OUTSIDE_MEDIAN_BAND
        
        # If in tail
        if event.line <= event.q10 or event.line >= event.q90:
            return LineZoneClassification.TAIL_PRICE_REQUIRED
        
        # If price doesn't match probability
        if event.p_side_raw < 0.45 and event.odds_american < 0:
            return LineZoneClassification.PRICE_MISPLACED
        
        if event.p_side_raw > 0.55 and event.odds_american > 0:
            return LineZoneClassification.TOO_EXPENSIVE
        
        return LineZoneClassification.OUTSIDE_MEDIAN_BAND


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("LineZoneScanner module loaded.")
