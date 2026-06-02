"""
PHASE 3: Single-Leg Set Membership with Tiers

Implement tiered membership:

SEED_PLAYABLE:
- high LCB edge
- high robust EV
- high forecastability
- high plan reliability
- high scenario agreement
- low chaos
- low management volatility
- low market instability
- strong price validation
- no unresolved news risk

BALANCED_PLAYABLE:
- positive robust EV
- positive LCB edge
- acceptable forecastability
- acceptable scenario agreement
- stress probability beats break-even
- no major failure concentration

PRICE_DEPENDENT:
- model probability may be valid
- current price does not clear threshold
- candidate becomes playable only at min_acceptable_odds or better

NEWS_DEPENDENT:
- model state may be valid
- injury/lineup/minutes role not resolved
- rerun required after news checkpoint

BOUNDARY_SHADOW:
- close to accepted region
- useful for validation
- not allowed in parlays unless explicitly enabled

PASS:
- fails price, reliability, stress, news, or scenario checks
"""

from typing import List, Dict, Optional
import logging

from core_utils import (
    LegStatus,
    min_acceptable_odds_for_edge,
    lcb_edge as calc_lcb_edge,
)
from data_types import PricedBinaryEvent, SingleLegEvaluation, NewsStatus, MarketFamily, Side

logger = logging.getLogger(__name__)


class SingleLegSetMembership:
    """
    Evaluates each priced event for membership in the single-leg bettable set B_t,
    with tiered classifications.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize with policy configuration."""
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration thresholds."""
        return {
            "min_lcb_edge_seed": 0.025,
            "min_lcb_edge_balanced": 0.010,
            "min_robust_edge_seed": 0.050,
            "min_robust_edge_balanced": 0.025,
            "min_stress_edge": 0.015,
            "min_forecastability_seed": 0.78,
            "min_forecastability_balanced": 0.70,
            "min_plan_reliability": 0.70,
            "min_scenario_agreement": 0.68,
            "max_chaos_score": 0.35,
            "max_management_volatility": 0.35,
            "max_market_instability": 0.35,
            "max_edge_fragility": 0.06,
            "require_valid_current_price": True,
            "max_stale_minutes": 20,
            "supply_line_percentile_threshold": 0.75,
            "min_supply_score_seed": 0.55,
            "min_supply_score_balanced": 0.40,
        }
    
    def evaluate(self, events: List[PricedBinaryEvent]) -> List[SingleLegEvaluation]:
        """
        Evaluate all events for single-leg pool membership.
        
        Returns list of SingleLegEvaluation objects.
        """
        results = []
        
        for event in events:
            evaluation = self._evaluate_single_event(event)
            results.append(evaluation)
        
        return results
    
    def _evaluate_single_event(self, event: PricedBinaryEvent) -> SingleLegEvaluation:
        """
        Evaluate a single priced event for membership and tier classification.
        """
        
        # Collect rejection reasons
        rejection_reasons = []
        promotion_requirements = []
        
        # Step 1: Check price status
        if event.price_status != "PRICE_VALID":
            if event.price_status == "MISSING_PRICE":
                rejection_reasons.append("MISSING_PRICE")
                return self._reject_event(event, rejection_reasons, "PASS")
            elif event.price_status == "STALE_PRICE":
                rejection_reasons.append("STALE_PRICE")
                return self._reject_event(event, rejection_reasons, "PASS")
            elif event.price_status == "PASS_AT_PRICE":
                rejection_reasons.append("PASS_AT_PRICE")
                return self._reject_event(event, rejection_reasons, "PASS")
        
        # Step 2: Check news status
        if event.news_status == NewsStatus.OUT.value:
            rejection_reasons.append("OUT_PLAYER")
            return self._reject_event(event, rejection_reasons, "PASS")
        
        if event.news_status in [NewsStatus.NEWS_DEPENDENT.value, NewsStatus.QUESTIONABLE.value]:
            # Mark as NEWS_DEPENDENT, but don't reject yet
            return self._classify_news_dependent(event, rejection_reasons)
        
        if event.news_status == NewsStatus.MINUTES_LIMIT_RISK.value:
            rejection_reasons.append("MINUTES_LIMIT_RISK")
            return self._reject_event(event, rejection_reasons, "NEWS_DEPENDENT")
        
        # Step 3: Check break-even beat
        break_even_beat = event.p_side_stress - event.p_side_raw
        if break_even_beat < self.config.get("min_stress_edge", 0.015):
            rejection_reasons.append(f"STRESS_EDGE_TOO_LOW ({break_even_beat:.3f})")
            
            # Check if it's PRICE_DEPENDENT (could work at better price)
            if event.p_side_stress > event.p_side_raw:
                # Try to find min acceptable odds
                min_odds = min_acceptable_odds_for_edge(
                    event.p_side_stress,
                    required_edge_margin=self.config.get("min_stress_edge", 0.015)
                )
                if min_odds:
                    return self._classify_price_dependent(
                        event, min_odds, rejection_reasons
                    )
            
            return self._reject_event(event, rejection_reasons, "PASS")
        
        # Step 4: Try SEED tier first
        seed_check = self._check_seed_tier(event)
        if seed_check["passed"]:
            return SingleLegEvaluation(
                event_id=event.event_id,
                player_name=event.player_name,
                player_market=f"{event.market_family.value}",
                side=event.side,
                line=event.line,
                odds_american=event.odds_american,
                game_id=event.game_id,
                team=event.team,
                leg_status=LegStatus.SEED_PLAYABLE.value,
                tier=LegStatus.SEED_PLAYABLE.value,
                accepted_into_single_leg_pool=True,
                break_even_prob=event.p_side_raw,
                p_side_stress=event.p_side_stress,
                p_side_lcb=event.p_side_lcb,
                lcb_edge=event.lcb_edge,
                robust_edge=event.robust_edge,
                forecastability_score=event.forecastability_score,
                plan_reliability=event.plan_reliability,
                scenario_agreement=event.scenario_agreement,
            )
        else:
            rejection_reasons.extend(seed_check["reasons"])
        
        # Step 5: Try BALANCED tier
        balanced_check = self._check_balanced_tier(event)
        if balanced_check["passed"]:
            return SingleLegEvaluation(
                event_id=event.event_id,
                player_name=event.player_name,
                player_market=f"{event.market_family.value}",
                side=event.side,
                line=event.line,
                odds_american=event.odds_american,
                game_id=event.game_id,
                team=event.team,
                leg_status=LegStatus.BALANCED_PLAYABLE.value,
                tier=LegStatus.BALANCED_PLAYABLE.value,
                accepted_into_single_leg_pool=True,
                break_even_prob=event.p_side_raw,
                p_side_stress=event.p_side_stress,
                p_side_lcb=event.p_side_lcb,
                lcb_edge=event.lcb_edge,
                robust_edge=event.robust_edge,
                forecastability_score=event.forecastability_score,
                plan_reliability=event.plan_reliability,
                scenario_agreement=event.scenario_agreement,
            )
        else:
            rejection_reasons.extend(balanced_check["reasons"])
        
        # Step 6: Check if PRICE_DEPENDENT
        if event.robust_ev > 0 and event.lcb_edge > 0:
            min_odds = min_acceptable_odds_for_edge(
                event.p_side_stress,
                required_edge_margin=self.config.get("min_stress_edge", 0.015)
            )
            if min_odds:
                return self._classify_price_dependent(
                    event, min_odds, rejection_reasons
                )
        
        # All checks failed
        return self._reject_event(event, rejection_reasons, "PASS")
    
    def _check_seed_tier(self, event: PricedBinaryEvent) -> Dict:
        """Check if event qualifies for SEED_PLAYABLE tier."""
        reasons = []
        
        if event.lcb_edge < self.config.get("min_lcb_edge_seed", 0.025):
            reasons.append(f"LCB_EDGE_LOW_SEED ({event.lcb_edge:.4f})")
        
        if event.robust_edge < self.config.get("min_robust_edge_seed", 0.050):
            reasons.append(f"ROBUST_EDGE_LOW_SEED ({event.robust_edge:.4f})")
        
        if event.forecastability_score < self.config.get("min_forecastability_seed", 0.78):
            reasons.append(f"FORECASTABILITY_LOW_SEED ({event.forecastability_score:.3f})")
        
        if event.plan_reliability < self.config.get("min_plan_reliability", 0.70):
            reasons.append(f"PLAN_RELIABILITY_LOW ({event.plan_reliability:.3f})")
        
        if event.scenario_agreement < self.config.get("min_scenario_agreement", 0.68):
            reasons.append(f"SCENARIO_AGREEMENT_LOW ({event.scenario_agreement:.3f})")
        
        if event.chaos_score > self.config.get("max_chaos_score", 0.35):
            reasons.append(f"CHAOS_TOO_HIGH ({event.chaos_score:.3f})")
        
        if event.management_volatility_score > self.config.get("max_management_volatility", 0.35):
            reasons.append(f"MANAGEMENT_VOLATILITY_HIGH ({event.management_volatility_score:.3f})")
        
        if event.market_instability_score > self.config.get("max_market_instability", 0.35):
            reasons.append(f"MARKET_INSTABILITY_HIGH ({event.market_instability_score:.3f})")

        if self._is_supply_dependent_leg(event):
            supply_check = self._check_supply_dependent_leg(event, tier="seed")
            if not supply_check["passed"]:
                reasons.append(supply_check["reason"])
                reasons.extend(supply_check.get("details", []))

        return {
            "passed": len(reasons) == 0,
            "reasons": reasons,
        }
    
    def _check_balanced_tier(self, event: PricedBinaryEvent) -> Dict:
        """Check if event qualifies for BALANCED_PLAYABLE tier."""
        reasons = []
        
        if event.lcb_edge < self.config.get("min_lcb_edge_balanced", 0.010):
            reasons.append(f"LCB_EDGE_LOW_BALANCED ({event.lcb_edge:.4f})")
        
        if event.robust_edge < self.config.get("min_robust_edge_balanced", 0.025):
            reasons.append(f"ROBUST_EDGE_LOW_BALANCED ({event.robust_edge:.4f})")
        
        if event.forecastability_score < self.config.get("min_forecastability_balanced", 0.70):
            reasons.append(f"FORECASTABILITY_LOW_BALANCED ({event.forecastability_score:.3f})")
        
        if event.plan_reliability < self.config.get("min_plan_reliability", 0.70):
            reasons.append(f"PLAN_RELIABILITY_LOW ({event.plan_reliability:.3f})")
        
        if event.scenario_agreement < self.config.get("min_scenario_agreement", 0.68):
            reasons.append(f"SCENARIO_AGREEMENT_LOW ({event.scenario_agreement:.3f})")
        
        if event.edge_fragility > self.config.get("max_edge_fragility", 0.06):
            reasons.append(f"EDGE_FRAGILITY_HIGH ({event.edge_fragility:.3f})")

        if self._is_supply_dependent_leg(event):
            supply_check = self._check_supply_dependent_leg(event, tier="balanced")
            if not supply_check["passed"]:
                reasons.append(supply_check["reason"])
                reasons.extend(supply_check.get("details", []))

        return {
            "passed": len(reasons) == 0,
            "reasons": reasons,
        }

    def _is_supply_dependent_leg(self, event: PricedBinaryEvent) -> bool:
        """Identify legs that require supply validation."""
        if event.side != Side.OVER:
            return False

        if event.market_family == MarketFamily.REB:
            return True

        if event.market_family in {MarketFamily.STL, MarketFamily.BLK, MarketFamily.AST}:
            return True

        return False

    def _check_supply_dependent_leg(self, event: PricedBinaryEvent, tier: str) -> Dict:
        """Validate supply-dependent legs for upper-band lines."""
        if event.line_percentile <= self.config.get("supply_line_percentile_threshold", 0.75):
            return {"passed": True, "reason": ""}

        supply_score = self._compute_supply_dependency_score(event)
        min_threshold = self.config.get(
            "min_supply_score_seed" if tier == "seed" else "min_supply_score_balanced",
            0.55 if tier == "seed" else 0.40
        )

        if supply_score >= min_threshold:
            return {"passed": True, "reason": ""}

        detail = [
            f"SUPPLY_SCORE={supply_score:.3f}",
            f"REQUIRED_THRESHOLD={min_threshold:.3f}",
            f"LINE_PERCENTILE={event.line_percentile:.3f}",
        ]

        return {
            "passed": False,
            "reason": f"SUPPLY_DEPENDENT_{tier.upper()}_FAIL ({supply_score:.3f})",
            "details": detail,
        }

    def _compute_supply_dependency_score(self, event: PricedBinaryEvent) -> float:
        """Compute a normalized supply score for supply-dependent legs."""
        score = 0.0

        if event.market_family == MarketFamily.REB and event.side == Side.OVER:
            score += 0.20
            score += min(0.35, event.team_rebound_rate * 0.35)
            score += min(0.30, event.rebound_share * 0.30)
            score -= min(0.20, event.team_shooting_efficiency_risk * 0.20)
            score -= min(0.20, event.opponent_shooting_efficiency_risk * 0.20)
            score -= min(0.15, event.wing_rebound_leakage_score * 0.15)
            if event.line_percentile > 0.90:
                score -= 0.10
            elif event.line_percentile > 0.80:
                score -= 0.05
        else:
            # Generic supply-dependent markets: use forecastability and line position
            score += 0.30
            score += min(0.30, event.forecastability_score * 0.30)
            score += max(0.0, 0.20 - (event.line_percentile - 0.75) * 0.40)
            score -= min(0.15, event.chaos_score * 0.15)
            score -= min(0.10, event.edge_fragility * 0.10)

        return max(0.0, min(1.0, score))

    def _classify_news_dependent(
        self,
        event: PricedBinaryEvent,
        rejection_reasons: List[str]
    ) -> SingleLegEvaluation:
        """Classify as NEWS_DEPENDENT."""
        rejection_reasons.append("NEWS_DEPENDENT")
        return SingleLegEvaluation(
            event_id=event.event_id,
            player_name=event.player_name,
            player_market=f"{event.market_family.value}",
            side=event.side,
            line=event.line,
            odds_american=event.odds_american,
            game_id=event.game_id,
            team=event.team,
            leg_status=LegStatus.NEWS_DEPENDENT.value,
            tier=LegStatus.NEWS_DEPENDENT.value,
            accepted_into_single_leg_pool=False,
            rejection_reasons=rejection_reasons,
        )
    
    def _classify_price_dependent(
        self,
        event: PricedBinaryEvent,
        min_acceptable_odds: float,
        rejection_reasons: List[str]
    ) -> SingleLegEvaluation:
        """Classify as PRICE_DEPENDENT and compute min acceptable odds."""
        rejection_reasons.append(f"PRICE_DEPENDENT_MIN_ODDS_{min_acceptable_odds:.0f}")
        return SingleLegEvaluation(
            event_id=event.event_id,
            player_name=event.player_name,
            player_market=f"{event.market_family.value}",
            side=event.side,
            line=event.line,
            odds_american=event.odds_american,
            game_id=event.game_id,
            team=event.team,
            leg_status=LegStatus.PRICE_DEPENDENT.value,
            tier=LegStatus.PRICE_DEPENDENT.value,
            accepted_into_single_leg_pool=False,
            min_acceptable_odds=min_acceptable_odds,
            rejection_reasons=rejection_reasons,
            promotion_requirements=[
                f"Obtain odds of {min_acceptable_odds:.0f} or better"
            ],
        )
    
    def _reject_event(
        self,
        event: PricedBinaryEvent,
        reasons: List[str],
        tier: str
    ) -> SingleLegEvaluation:
        """Create a rejection evaluation."""
        return SingleLegEvaluation(
            event_id=event.event_id,
            player_name=event.player_name,
            player_market=f"{event.market_family.value}",
            side=event.side,
            line=event.line,
            odds_american=event.odds_american,
            game_id=event.game_id,
            team=event.team,
            leg_status=LegStatus.PASS.value,
            tier=tier,
            accepted_into_single_leg_pool=False,
            rejection_reasons=reasons,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("SingleLegSetMembership module loaded.")
