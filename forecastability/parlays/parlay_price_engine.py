"""
PHASE 8: Parlay Price Engine

For normal cross-game parlays:
- combined decimal odds = product(decimal_odds)

For same-game parlays:
- books may correlation-adjust payout (SGP)
- always support both modes: SYNTHETIC_PARLAY_PRICE and BOOK_QUOTED_SGP_PRICE

Compute:
- combined_decimal_odds
- combined_american_odds
- parlay_break_even_prob
- price_source: SYNTHETIC or BOOK_QUOTED
- price_validity
- price_gap_vs_synthetic
- same_game_price_penalty
"""

from typing import List, Optional, Dict
import logging
import math

from core_utils import (
    american_to_decimal,
    decimal_to_american,
    american_to_implied_prob,
)
from data_types import ParlayLeg, ParlayCandidate

logger = logging.getLogger(__name__)


class ParlayPriceEngine:
    """
    Computes parlay pricing from individual leg odds.
    Handles both synthetic (product) and book-quoted SGP pricing.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_sgp_payout_ratio = self.config.get("min_sgp_payout_ratio", 0.75)
    
    def compute_synthetic_parlay_price(
        self,
        american_odds_list: List[float]
    ) -> Dict:
        """
        Compute synthetic parlay price by multiplying decimal odds.
        """
        if not american_odds_list:
            return None
        
        decimal_product = 1.0
        for american_odds in american_odds_list:
            if american_odds == 0.0:
                return None
            decimal = american_to_decimal(american_odds)
            decimal_product *= decimal
        
        combined_american = decimal_to_american(decimal_product)
        break_even = american_to_implied_prob(combined_american)
        
        return {
            "combined_decimal_odds": decimal_product,
            "combined_american_odds": combined_american,
            "parlay_break_even_prob": break_even,
            "price_source": "SYNTHETIC",
        }
    
    def compute_parlay_from_legs(
        self,
        legs: List[ParlayLeg],
        book_quoted_odds: Optional[float] = None,
        same_game: bool = False
    ) -> Dict:
        """
        Compute complete parlay pricing.
        
        Inputs:
            legs: List of parlay legs with odds
            book_quoted_odds: Actual odds from book (overrides synthetic)
            same_game: Whether this is a same-game parlay
        
        Returns:
            Dict with pricing info
        """
        american_odds_list = [leg.odds_american for leg in legs]
        
        # Get synthetic price
        synthetic = self.compute_synthetic_parlay_price(american_odds_list)
        if not synthetic:
            return {
                "price_validity": "MISSING_PRICE",
                "price_source": "UNKNOWN",
            }
        
        # If book-quoted price is provided, use it
        if book_quoted_odds:
            decimal_book = american_to_decimal(book_quoted_odds)
            break_even_book = american_to_implied_prob(book_quoted_odds)
            
            # Check SGP payout reduction
            synthetic_decimal = synthetic["combined_decimal_odds"]
            payout_ratio = decimal_book / synthetic_decimal if synthetic_decimal > 0 else 0.0
            
            price_penalty = 0.0
            if payout_ratio < self.min_sgp_payout_ratio:
                price_penalty = (self.min_sgp_payout_ratio - payout_ratio)
                logger.warning(
                    f"SGP payout heavily reduced: {payout_ratio:.2%} "
                    f"(synthetic: {synthetic_decimal:.4f}, book: {decimal_book:.4f})"
                )
            
            return {
                "combined_decimal_odds": decimal_book,
                "combined_american_odds": book_quoted_odds,
                "parlay_break_even_prob": break_even_book,
                "price_source": "BOOK_QUOTED",
                "price_gap_vs_synthetic": decimal_book - synthetic_decimal,
                "same_game_price_penalty": price_penalty,
                "price_validity": "PRICE_VALID" if payout_ratio >= self.min_sgp_payout_ratio else "PRICE_INVALID",
                "payout_ratio": payout_ratio,
            }
        
        # Use synthetic
        return {
            "combined_decimal_odds": synthetic["combined_decimal_odds"],
            "combined_american_odds": synthetic["combined_american_odds"],
            "parlay_break_even_prob": synthetic["parlay_break_even_prob"],
            "price_source": "SYNTHETIC",
            "price_gap_vs_synthetic": 0.0,
            "same_game_price_penalty": 0.0,
            "price_validity": "PRICE_VALID",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    engine = ParlayPriceEngine()
    
    price = engine.compute_synthetic_parlay_price([-110, -110])
    print(f"2x -110 parlay: {price}")
    
    price = engine.compute_synthetic_parlay_price([-110, +100])
    print(f"-110 and +100 parlay: {price}")
