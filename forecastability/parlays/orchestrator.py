"""
Parlay Subsystem Orchestrator

Minimal vertical slice: Phase 1-4 + supporting infrastructure.

Pipeline:
1. Build priced event universe (Phase 1)
2. Scan line zones (Phase 2)
3. Evaluate single leg membership (Phase 3)
4. Generate anchor + companion candidates (Phase 4)
5. Compute parlay prices (Phase 8)
6. Detect shared event supply (Phase 5)
7. Compute joint probability (Phase 9)
8. Apply stress (Phase 10)
9. Select and score (Phase 11)
10. Report results

All operations in SHADOW MODE by default.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional
import json

from build_priced_event_universe import PricedEventUniverseBuilder
from line_zone_scanner import LineZoneScanner
from single_leg_set_membership import SingleLegSetMembership
from anchor_companion_generator import AnchorCompanionGenerator
from parlay_price_engine import ParlayPriceEngine
from shared_event_supply_engine import SharedEventSupplyEngine
from parlay_probability_engine import ParlayProbabilityEngine
from parlay_stress_engine import ParlayStressEngine
from parlay_selector import ParlaySelector
from team_environment_failure_engine import TeamEnvironmentFailureEngine
from correlation_engine import CorrelationEngine
from checkpoint_manager import CheckpointManager, CheckpointTracker, PriceTracker
from report_generator import ReportBuilder
from validator import HistoricalValidator
from data_types import PricedBinaryEvent, ParlayCandidate, ParlayLeg, JointState

logger = logging.getLogger(__name__)


class ParlaySubsystemOrchestrator:
    """
    Main orchestrator for parlay subsystem.
    Runs the full pipeline in SHADOW MODE.
    """
    
    def __init__(self, config_path: str = "config/parlay_policy.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Pipeline components
        self.price_engine = ParlayPriceEngine(self.config)
        self.single_leg_eval = SingleLegSetMembership(self.config)
        self.generator = AnchorCompanionGenerator(self.config)
        self.supply_engine = SharedEventSupplyEngine(self.config)
        self.prob_engine = ParlayProbabilityEngine(self.config)
        self.stress_engine = ParlayStressEngine(self.config)
        self.failure_engine = TeamEnvironmentFailureEngine(self.config)
        self.correlation_engine = CorrelationEngine(
            self.config.get("historical_outcomes_path")
        )
        self.checkpoint_manager = CheckpointManager()
        self.checkpoint_tracker = CheckpointTracker()
        self.price_tracker = PriceTracker()
        self.validator = HistoricalValidator(
            self.config.get("settled_outcomes_path")
        )
        self.selector = ParlaySelector(self.config)
        
        # Results
        self.priced_events: List[PricedBinaryEvent] = []
        self.leg_evaluations = []
        self.parlay_candidates = []
        self.final_parlays = []
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML."""
        import yaml
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            logger.warning(f"Could not load config from {self.config_path}, using defaults")
            return {}
    
    def run_pipeline(
        self,
        scenario_prob_path: str,
        forecastability_path: str,
        player_state_path: str,
        odds_snapshot_path: str,
        output_dir: str = "outputs"
    ) -> Dict:
        """
        Run the full parlay subsystem pipeline.
        
        Returns summary of results.
        """
        
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Parlay Subsystem Pipeline (SHADOW MODE)")
        logger.info("=" * 60)
        
        # Phase 1: Build priced event universe
        logger.info("\n[Phase 1] Building priced event universe...")
        builder = PricedEventUniverseBuilder(
            scenario_prob_path,
            forecastability_path,
            player_state_path,
            odds_snapshot_path,
            self.config
        )
        
        self.priced_events = builder.build_universe()
        logger.info(f"✓ Built {len(self.priced_events)} priced events")
        
        if not self.priced_events:
            logger.error("No priced events built. Aborting pipeline.")
            return self._summary(start_time)
        
        # Phase 2: Scan line zones
        logger.info("\n[Phase 2] Scanning line zones...")
        scanner = LineZoneScanner(self.priced_events, self.config)
        line_zones = scanner.scan_universe()
        logger.info(f"✓ Scanned {len(line_zones)} player/market combinations")
        
        # Phase 3: Evaluate single leg membership
        logger.info("\n[Phase 3] Evaluating single leg membership...")
        self.leg_evaluations = self.single_leg_eval.evaluate(self.priced_events)
        
        accepted_count = sum(1 for leg in self.leg_evaluations if leg.accepted_into_single_leg_pool)
        logger.info(f"✓ Evaluated {len(self.leg_evaluations)} legs")
        logger.info(f"  → {accepted_count} ACCEPTED ({100*accepted_count/len(self.leg_evaluations):.1f}%)")
        
        # Capture initial checkpoint snapshot and record prices
        checkpoint_name = self.checkpoint_manager.get_checkpoint_for_time(start_time) or "OPEN"
        self.checkpoint_tracker.add_snapshot(checkpoint_name, start_time, self.leg_evaluations)
        for eval in self.leg_evaluations:
            self.price_tracker.record_price(eval.event_id, eval.odds_american, start_time)
        
        if accepted_count < 2:
            logger.warning("Not enough accepted legs for parlay construction")
            return self._summary(start_time)
        
        # Phase 4: Generate parlay candidates
        logger.info("\n[Phase 4] Generating parlay candidates (anchor + companion)...")
        parlay_specs = self.generator.generate_parlay_candidates(self.leg_evaluations)
        logger.info(f"✓ Generated {len(parlay_specs)} candidate specs")
        
        # Convert specs to full parlay candidates
        logger.info("\n[Phases 5-11] Evaluating parlay candidates...")
        for spec_idx, spec in enumerate(parlay_specs):
            parlay = self._build_parlay_candidate(spec)
            if parlay:
                self.parlay_candidates.append(parlay)
            
            if (spec_idx + 1) % 10 == 0:
                logger.debug(f"  Processed {spec_idx + 1}/{len(parlay_specs)} specs")
        
        logger.info(f"✓ Built {len(self.parlay_candidates)} full parlay candidates")
        
        # Phase 11: Select and rank
        logger.info("\n[Phase 11] Selecting and ranking parlays...")
        for parlay in self.parlay_candidates:
            parlay = self.selector.select_and_score(parlay)
        
        # Filter and rank
        max_output = self.config.get("generation", {}).get("max_parlays_output", 25)
        self.final_parlays = self.selector.filter_and_rank(
            self.parlay_candidates,
            max_output=max_output,
            min_tier="BALANCED_SHADOW"
        )
        
        logger.info(f"✓ Final selection: {len(self.final_parlays)} parlays")
        
        # Export results
        logger.info(f"\n[Export] Writing results to {output_dir}...")
        self._export_results(output_dir, builder)
        
        # Summary
        summary = self._summary(start_time)
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete (SHADOW MODE)")
        logger.info("=" * 60)
        
        return summary
    
    def _build_parlay_candidate(self, spec: Dict) -> Optional[ParlayCandidate]:
        """Build a full parlay candidate from a spec."""
        
        try:
            anchor_leg_data = spec.get("anchor_leg")
            companion_leg_data = spec.get("companion_leg")
            third_leg_data = spec.get("third_leg")
            
            legs = [
                ParlayLeg(
                    event_id=anchor_leg_data.event_id,
                    player_name=anchor_leg_data.player_name,
                    market_family=anchor_leg_data.player_market,
                    side=anchor_leg_data.side,
                    line=anchor_leg_data.line,
                    odds_american=anchor_leg_data.odds_american,
                    p_stress=anchor_leg_data.p_side_stress,
                    p_lcb=anchor_leg_data.p_side_lcb,
                    lcb_edge=anchor_leg_data.lcb_edge,
                    game_id=anchor_leg_data.game_id,
                ),
                ParlayLeg(
                    event_id=companion_leg_data.event_id,
                    player_name=companion_leg_data.player_name,
                    market_family=companion_leg_data.player_market,
                    side=companion_leg_data.side,
                    line=companion_leg_data.line,
                    odds_american=companion_leg_data.odds_american,
                    p_stress=companion_leg_data.p_side_stress,
                    p_lcb=companion_leg_data.p_side_lcb,
                    lcb_edge=companion_leg_data.lcb_edge,
                    game_id=companion_leg_data.game_id,
                ),
            ]
            
            if third_leg_data:
                legs.append(
                    ParlayLeg(
                        event_id=third_leg_data.event_id,
                        player_name=third_leg_data.player_name,
                        market_family=third_leg_data.player_market,
                        side=third_leg_data.side,
                        line=third_leg_data.line,
                        odds_american=third_leg_data.odds_american,
                        p_stress=third_leg_data.p_side_stress,
                        p_lcb=third_leg_data.p_side_lcb,
                        lcb_edge=third_leg_data.lcb_edge,
                        game_id=third_leg_data.game_id,
                    )
                )
            
            same_game = len({leg.game_id for leg in legs if leg.game_id}) < len(legs)
            joint_state = self.correlation_engine.compute_joint_state(legs)
            failure_exposure = self.failure_engine.analyze_parlay_shared_exposure(legs)
            
            price_result = self.price_engine.compute_parlay_from_legs(
                legs,
                book_quoted_odds=None,
                same_game=same_game
            )
            
            supply_result = self.supply_engine.analyze_legs(legs)
            
            prob_result = self.prob_engine.compute_joint_probability(
                legs,
                joint_state=joint_state,
                shared_event_supply_penalty=supply_result.get("shared_event_supply_penalty", 0.0),
                same_game=same_game,
            )
            
            stress_result = self.stress_engine.stress_test_parlay(
                prob_result.get("p_joint_adjusted", 0.0),
                price_result.get("parlay_break_even_prob", 0.5),
                lcb_joint_edge=max(0.0, prob_result.get("p_joint_lcb", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                robust_joint_edge=max(0.0, prob_result.get("p_joint_adjusted", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                shared_failure_risk=failure_exposure.get("shared_exposure_penalty", 0.0),
                shared_event_supply_penalty=supply_result.get("shared_event_supply_penalty", 0.0),
                edge_fragility=max(
                    anchor_leg_data.edge_fragility if hasattr(anchor_leg_data, 'edge_fragility') else 0.0,
                    companion_leg_data.edge_fragility if hasattr(companion_leg_data, 'edge_fragility') else 0.0,
                    third_leg_data.edge_fragility if third_leg_data and hasattr(third_leg_data, 'edge_fragility') else 0.0,
                ),
            )
            
            dependency_penalty = 0.05
            if joint_state.empirical_correlation is not None:
                dependency_penalty = min(0.25, abs(joint_state.empirical_correlation) * 0.10)
            elif any("UNKNOWN" in cls for cls in joint_state.dependency_classes):
                dependency_penalty = 0.08
            
            compatible_state_score = max(
                0.45,
                1.0 - max(
                    failure_exposure.get("shared_exposure_penalty", 0.0),
                    abs(joint_state.empirical_correlation or 0.0) * 0.05,
                )
            )
            
            parlay = ParlayCandidate(
                parlay_id=f"NBA_PARLAY_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{spec.get('anchor_idx', 0)}_{len(self.parlay_candidates)}",
                checkpoint=self.checkpoint_manager.get_checkpoint_for_time(datetime.now()) or "FINAL_ODDS",
                snapshot_time=datetime.now(),
                legs=legs,
                combined_decimal_odds=price_result.get("combined_decimal_odds", 1.0),
                combined_american_odds=price_result.get("combined_american_odds", 0.0),
                parlay_break_even_prob=price_result.get("parlay_break_even_prob", 0.5),
                price_source=price_result.get("price_source", "UNKNOWN"),
                price_validity=price_result.get("price_validity", "UNKNOWN"),
                p_joint_naive=prob_result.get("p_joint_naive", 0.0),
                p_joint_adjusted=prob_result.get("p_joint_adjusted", 0.0),
                p_joint_stress=stress_result.get("p_joint_stress", 0.0),
                p_joint_lcb=prob_result.get("p_joint_lcb", 0.0),
                raw_joint_edge=max(0.0, prob_result.get("p_joint_naive", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                robust_joint_edge=max(0.0, prob_result.get("p_joint_adjusted", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                lcb_joint_edge=max(0.0, prob_result.get("p_joint_lcb", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                raw_joint_ev=max(0.0, prob_result.get("p_joint_naive", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                robust_joint_ev=max(0.0, prob_result.get("p_joint_adjusted", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                lcb_joint_ev=max(0.0, prob_result.get("p_joint_lcb", 0.0) - price_result.get("parlay_break_even_prob", 0.5)),
                joint_probability_confidence=prob_result.get("joint_probability_confidence", 0.0),
                shared_failure_risk=failure_exposure.get("shared_exposure_penalty", 0.0),
                shared_event_supply_penalty=supply_result.get("shared_event_supply_penalty", 0.0),
                dependency_penalty=dependency_penalty,
                edge_fragility=max(
                    anchor_leg_data.edge_fragility if hasattr(anchor_leg_data, 'edge_fragility') else 0.0,
                    companion_leg_data.edge_fragility if hasattr(companion_leg_data, 'edge_fragility') else 0.0,
                    third_leg_data.edge_fragility if third_leg_data and hasattr(third_leg_data, 'edge_fragility') else 0.0,
                ),
                anchor_leg_idx=spec.get("anchor_idx", 0),
                companion_leg_indices=[0, 1] if not third_leg_data else [0, 1, 2],
                compatible_state_score=compatible_state_score,
                min_leg_forecastability=min(
                    anchor_leg_data.forecastability_score,
                    companion_leg_data.forecastability_score,
                    third_leg_data.forecastability_score if third_leg_data else 1.0,
                ),
                min_leg_plan_reliability=min(
                    anchor_leg_data.plan_reliability,
                    companion_leg_data.plan_reliability,
                    third_leg_data.plan_reliability if third_leg_data else 1.0,
                ),
                min_leg_scenario_agreement=min(
                    anchor_leg_data.scenario_agreement,
                    companion_leg_data.scenario_agreement,
                    third_leg_data.scenario_agreement if third_leg_data else 1.0,
                ),
                price_quality_score=0.75,
                joint_state=joint_state,
            )
            
            return parlay
            
        except Exception as e:
            logger.error(f"Error building parlay candidate: {e}")
            return None
    
    def _summary(self, start_time: datetime) -> Dict:
        """Create pipeline summary."""
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "mode": "SHADOW_MODE",
            "elapsed_seconds": elapsed,
            "priced_events_count": len(self.priced_events),
            "leg_evaluations_count": len(self.leg_evaluations),
            "accepted_legs_count": sum(1 for leg in self.leg_evaluations if leg.accepted_into_single_leg_pool),
            "parlay_candidates_count": len(self.parlay_candidates),
            "final_parlays_count": len(self.final_parlays),
            "final_parlays": [
                {
                    "parlay_id": p.parlay_id,
                    "tier": p.tier,
                    "decision": p.decision,
                    "final_score": p.final_parlay_score,
                    "legs_count": len(p.legs),
                    "lcb_edge": p.lcb_joint_edge,
                    "joint_probability": p.p_joint_stress,
                }
                for p in self.final_parlays[:5]  # Top 5
            ]
        }
    
    def _export_results(self, output_dir: str, builder: PricedEventUniverseBuilder):
        """Export results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export priced universe
        builder.export_to_csv(f"{output_dir}/priced_event_universe_latest.csv")
        
        # Export parlays JSON
        if self.final_parlays:
            parlay_data = [
                {
                    "parlay_id": p.parlay_id,
                    "tier": p.tier,
                    "decision": p.decision,
                    "final_score": p.final_parlay_score,
                    "legs": [
                        {
                            "player": leg.player_name,
                            "market": leg.market_family,
                            "side": leg.side,
                            "line": leg.line,
                            "odds": leg.odds_american,
                            "probability": leg.p_stress,
                        }
                        for leg in p.legs
                    ],
                    "break_even_prob": p.parlay_break_even_prob,
                    "joint_probability": p.p_joint_stress,
                    "lcb_edge": p.lcb_joint_edge,
                }
                for p in self.final_parlays
            ]
            
            with open(f"{output_dir}/parlays_latest.json", "w") as f:
                json.dump(parlay_data, f, indent=2)
        
        logger.info(f"✓ Exported results to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Parlay Subsystem Orchestrator loaded.")
    logger.info("To run: orchestrator = ParlaySubsystemOrchestrator()")
    logger.info("        summary = orchestrator.run_pipeline(...)")
