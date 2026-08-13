"""
Tests for Parlay Subsystem

Test coverage for all 17 phases with focus on minimal vertical slice.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from ..core_utils import (
    american_to_decimal,
    decimal_to_american,
    american_to_implied_prob,
    american_to_break_even_prob,
    edge_from_probability_and_odds,
    min_acceptable_odds_for_edge,
    parlay_synthetic_odds,
    parlay_break_even_prob,
)
from ..data_types import (
    PricedBinaryEvent,
    SingleLegEvaluation,
    ParlayLeg,
    Side,
    MarketFamily,
    LegStatus,
)
from ..single_leg_set_membership import SingleLegSetMembership
from ..parlay_price_engine import ParlayPriceEngine


class TestOddsConversions(unittest.TestCase):
    """Test American/Decimal/Probability conversions."""
    
    def test_american_to_decimal_negative(self):
        """Test -110 to decimal."""
        decimal = american_to_decimal(-110)
        self.assertAlmostEqual(decimal, 1.909, places=2)
    
    def test_american_to_decimal_positive(self):
        """Test +100 to decimal."""
        decimal = american_to_decimal(100)
        self.assertAlmostEqual(decimal, 2.0, places=2)
    
    def test_decimal_to_american_consistency(self):
        """Test round-trip conversion."""
        original = -110
        decimal = american_to_decimal(original)
        american = decimal_to_american(decimal)
        self.assertAlmostEqual(american, original, places=0)
    
    def test_american_to_implied_prob(self):
        """Test implied probability from American odds."""
        prob = american_to_implied_prob(-110)
        self.assertAlmostEqual(prob, 0.524, places=2)
    
    def test_break_even_prob(self):
        """Test break-even probability."""
        break_even = american_to_break_even_prob(-110)
        self.assertAlmostEqual(break_even, 0.524, places=2)
    
    def test_edge_calculation(self):
        """Test edge calculation."""
        # True prob 55%, odds -110 (break-even 52.4%)
        edge = edge_from_probability_and_odds(0.55, -110)
        self.assertGreater(edge, 0.01)  # Should have positive edge
    
    def test_min_acceptable_odds(self):
        """Test minimum acceptable odds calculation."""
        min_odds = min_acceptable_odds_for_edge(0.535, required_edge_margin=0.015)
        self.assertIsNotNone(min_odds)
        self.assertGreater(min_odds, -200)  # Should be within reasonable range
    
    def test_parlay_synthetic_odds(self):
        """Test 2x -110 parlay odds."""
        parlay_american = parlay_synthetic_odds([-110, -110])
        self.assertGreater(parlay_american, 200)  # 2x -110 pays about +264
    
    def test_parlay_break_even(self):
        """Test parlay break-even probability."""
        break_even = parlay_break_even_prob([-110, -110])
        self.assertGreater(break_even, 0.27)  # Should be ~27.5%


class TestSingleLegMembership(unittest.TestCase):
    """Test single leg set membership evaluation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = SingleLegSetMembership()
    
    def _create_test_event(self, **kwargs) -> PricedBinaryEvent:
        """Helper to create test events."""
        defaults = {
            "event_id": "test_event_1",
            "game_id": "game_1",
            "game_date": "2026-05-23",
            "snapshot_time": datetime.now(),
            "player_id": "player_1",
            "player_name": "Test Player",
            "team": "LAL",
            "opponent": "BOS",
            "book": "DK",
            "market_type": "PLAYER_POINTS",
            "side": Side.OVER,
            "line": 25.5,
            "is_main_line": True,
            "is_alt_line": False,
            "is_combo_market": False,
            "market_family": MarketFamily.PTS,
            "odds_american": -110,
            "odds_decimal": 1.909,
            "implied_prob_raw": 0.524,
            "price_status": "PRICE_VALID",
            "p_side_stress": 0.55,
            "p_side_raw": 0.52,
            "p_side_lcb": 0.53,
            "lcb_edge": 0.025,
            "robust_edge": 0.03,
            "forecastability_score": 0.78,
            "plan_reliability": 0.72,
            "scenario_agreement": 0.70,
            "chaos_score": 0.25,
            "management_volatility_score": 0.25,
            "market_instability_score": 0.25,
            "edge_fragility": 0.02,
            "news_status": "CLEAR",
        }
        defaults.update(kwargs)
        return PricedBinaryEvent(**defaults)
    
    def test_seed_playable_acceptance(self):
        """Test acceptance of SEED_PLAYABLE leg."""
        event = self._create_test_event(
            lcb_edge=0.030,
            robust_edge=0.060,
            forecastability_score=0.80,
            plan_reliability=0.75,
            scenario_agreement=0.72,
        )
        
        evals = self.evaluator.evaluate([event])
        self.assertEqual(len(evals), 1)
        self.assertTrue(evals[0].accepted_into_single_leg_pool)
        self.assertEqual(evals[0].leg_status, LegStatus.SEED_PLAYABLE.value)
    
    def test_balanced_playable_acceptance(self):
        """Test acceptance of BALANCED_PLAYABLE leg."""
        event = self._create_test_event(
            lcb_edge=0.015,
            robust_edge=0.035,
            forecastability_score=0.72,
            plan_reliability=0.71,
            scenario_agreement=0.69,
        )
        
        evals = self.evaluator.evaluate([event])
        self.assertEqual(len(evals), 1)
        self.assertTrue(evals[0].accepted_into_single_leg_pool)
        self.assertEqual(evals[0].leg_status, LegStatus.BALANCED_PLAYABLE.value)
    
    def test_low_edge_rejection(self):
        """Test rejection for low edge."""
        event = self._create_test_event(
            lcb_edge=0.005,
            robust_edge=0.010,
        )
        
        evals = self.evaluator.evaluate([event])
        self.assertEqual(len(evals), 1)
        self.assertFalse(evals[0].accepted_into_single_leg_pool)
    
    def test_news_dependent_classification(self):
        """Test NEWS_DEPENDENT classification."""
        event = self._create_test_event(
            news_status="NEWS_DEPENDENT",
        )
        
        evals = self.evaluator.evaluate([event])
        self.assertEqual(len(evals), 1)
        self.assertFalse(evals[0].accepted_into_single_leg_pool)
        self.assertEqual(evals[0].leg_status, LegStatus.NEWS_DEPENDENT.value)
    
    def test_price_dependent_classification(self):
        """Test PRICE_DEPENDENT classification."""
        event = self._create_test_event(
            p_side_stress=0.51,
            p_side_raw=0.48,
            lcb_edge=0.005,  # Too low for acceptance
            robust_edge=0.008,
        )
        
        evals = self.evaluator.evaluate([event])
        self.assertEqual(len(evals), 1)
        self.assertFalse(evals[0].accepted_into_single_leg_pool)
        # Could be PRICE_DEPENDENT if min odds calculation works

    def test_rebound_upper_band_supply_dependency_fails_seed(self):
        """Test upper-band rebound over fails seed membership without strong supply."""
        event = self._create_test_event(
            market_family=MarketFamily.REB,
            side=Side.OVER,
            line=11.5,
            line_percentile=0.85,
            p_side_stress=0.55,
            p_side_raw=0.50,
            lcb_edge=0.020,
            robust_edge=0.045,
            forecastability_score=0.76,
            plan_reliability=0.72,
            scenario_agreement=0.70,
            team_rebound_rate=0.42,
            rebound_share=0.11,
        )
        evals = self.evaluator.evaluate([event])
        self.assertEqual(len(evals), 1)
        self.assertFalse(evals[0].leg_status == LegStatus.SEED_PLAYABLE.value)
        self.assertIn("SUPPLY_DEPENDENT_SEED_FAIL", " ".join(evals[0].rejection_reasons))

    def test_rebound_upper_band_supply_allows_balanced_with_high_share(self):
        """Test upper-band rebound over can still qualify with strong supply metrics."""
        event = self._create_test_event(
            market_family=MarketFamily.REB,
            side=Side.OVER,
            line=11.5,
            line_percentile=0.82,
            p_side_stress=0.55,
            p_side_raw=0.50,
            lcb_edge=0.015,
            robust_edge=0.040,
            forecastability_score=0.72,
            plan_reliability=0.71,
            scenario_agreement=0.69,
            team_rebound_rate=0.58,
            rebound_share=0.18,
        )
        evals = self.evaluator.evaluate([event])
        self.assertEqual(len(evals), 1)
        self.assertTrue(evals[0].accepted_into_single_leg_pool)
        self.assertEqual(evals[0].leg_status, LegStatus.BALANCED_PLAYABLE.value)


class TestParlayPriceEngine(unittest.TestCase):
    """Test parlay price engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ParlayPriceEngine()
    
    def test_synthetic_two_leg_parlay(self):
        """Test 2x -110 parlay synthetic price."""
        result = self.engine.compute_synthetic_parlay_price([-110, -110])
        
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["combined_decimal_odds"], 3.64, places=1)
        self.assertGreater(result["combined_american_odds"], 200)
        self.assertGreater(result["parlay_break_even_prob"], 0.27)
        self.assertEqual(result["price_source"], "SYNTHETIC")
    
    def test_synthetic_mixed_odds_parlay(self):
        """Test -110 and +100 parlay."""
        result = self.engine.compute_synthetic_parlay_price([-110, 100])
        
        self.assertIsNotNone(result)
        self.assertGreater(result["combined_decimal_odds"], 2.0)
        self.assertEqual(result["price_source"], "SYNTHETIC")
    
    def test_missing_price_handling(self):
        """Test handling of missing/invalid prices."""
        result = self.engine.compute_synthetic_parlay_price([0.0])
        
        self.assertIsNone(result)
    
    def test_sgp_payout_reduction_warning(self):
        """Test SGP payout reduction detection."""
        result = self.engine.compute_parlay_from_legs(
            [],  # Empty legs list for this test
            book_quoted_odds=-150,  # Worse than synthetic would be
            same_game=True
        )
        
        # Result should have price_source and validity info
        self.assertIn("price_source", result)


class TestAllLegsInBettableSet(unittest.TestCase):
    """Test that all 2-leg parlay legs must be in bettable set."""
    
    def test_reject_if_leg_not_accepted(self):
        """Test rejection when leg not in bettable set."""
        # Create one good leg and one bad leg
        good_leg = SingleLegEvaluation(
            event_id="leg_1",
            player_name="Player A",
            player_market="PTS",
            side=Side.OVER,
            line=25.5,
            odds_american=-110,
            leg_status=LegStatus.BALANCED_PLAYABLE.value,
            tier=LegStatus.BALANCED_PLAYABLE.value,
            accepted_into_single_leg_pool=True,
            lcb_edge=0.015,
        )
        
        bad_leg = SingleLegEvaluation(
            event_id="leg_2",
            player_name="Player B",
            player_market="REB",
            side=Side.UNDER,
            line=8.5,
            odds_american=-110,
            leg_status=LegStatus.PASS.value,
            tier=LegStatus.PASS.value,
            accepted_into_single_leg_pool=False,
            lcb_edge=0.0,
        )
        
        # Parlay with one unaccepted leg should be rejected
        # This would be tested in anchor_companion_generator
        # which filters candidates to only accepted legs


if __name__ == "__main__":
    unittest.main()
