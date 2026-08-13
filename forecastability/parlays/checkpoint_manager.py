"""
PHASE 12: Timing Checkpoints

Support for rerunning at specific checkpoints:

CHECKPOINT_OPEN: 07:00 AM
CHECKPOINT_MIDDAY: 12:00 PM
CHECKPOINT_INJURY_REPORT: 1:00 PM (after injury reports)
CHECKPOINT_LINEUP_CONFIRMED: 5:00 PM (lineups confirmed)
CHECKPOINT_FINAL_ODDS: 7:00 PM (final odds before games)

A leg can change tier as information resolves:
- NEWS_DEPENDENT → BALANCED_PLAYABLE
- PRICE_DEPENDENT → BALANCED_PLAYABLE
- BALANCED_PLAYABLE → PASS_AT_PRICE
- SEED_PLAYABLE → PASS_NEWS_REGRESSION

Every output includes:
- checkpoint
- snapshot_time
- stale_price_flag
- news_rerun_required
- line_moved_since_previous_snapshot
- tier_change_since_previous_snapshot
"""

from typing import List, Dict, Optional
from datetime import datetime, time
import logging

from .data_types import SingleLegEvaluation, LegStatus

logger = logging.getLogger(__name__)


class Checkpoint:
    """Represents a timing checkpoint during the day."""
    
    def __init__(
        self,
        name: str,
        trigger_time: time,
        description: str,
        priority: int = 0
    ):
        self.name = name
        self.trigger_time = trigger_time
        self.description = description
        self.priority = priority
    
    def is_ready(self, current_time: datetime) -> bool:
        """Check if checkpoint trigger time has passed."""
        current = current_time.time()
        return current >= self.trigger_time


# Standard checkpoints
DEFAULT_CHECKPOINTS = [
    Checkpoint("CHECKPOINT_OPEN", time(7, 0), "Market opens", 1),
    Checkpoint("CHECKPOINT_MIDDAY", time(12, 0), "Midday update", 2),
    Checkpoint("CHECKPOINT_INJURY_REPORT", time(13, 0), "Injury reports released", 3),
    Checkpoint("CHECKPOINT_LINEUP_CONFIRMED", time(17, 0), "Lineups confirmed", 4),
    Checkpoint("CHECKPOINT_FINAL_ODDS", time(19, 0), "Final odds before games", 5),
]


class CheckpointManager:
    """
    Manages timing checkpoints and reruns throughout the day.
    """
    
    def __init__(self, custom_checkpoints: Optional[List[Checkpoint]] = None):
        self.checkpoints = custom_checkpoints or DEFAULT_CHECKPOINTS
        self.checkpoint_history = []
    
    def get_next_checkpoint(self, current_time: datetime) -> Optional[Checkpoint]:
        """Get the next checkpoint that needs to run."""
        
        current = current_time.time()
        
        for checkpoint in sorted(self.checkpoints, key=lambda c: c.priority):
            if checkpoint.trigger_time > current:
                return checkpoint
        
        return None
    
    def should_rerun(self, current_time: datetime, last_run_time: datetime) -> bool:
        """Check if we should rerun the pipeline at current time."""
        
        # Has any checkpoint passed since last run?
        for checkpoint in self.checkpoints:
            checkpoint_dt = datetime.combine(
                current_time.date(),
                checkpoint.trigger_time
            )
            
            if last_run_time < checkpoint_dt <= current_time:
                return True
        
        return False
    
    def get_checkpoint_for_time(self, current_time: datetime) -> Optional[str]:
        """Get the checkpoint name for the current time."""
        
        current = current_time.time()
        
        # Find the most recent checkpoint that has passed
        most_recent = None
        for checkpoint in self.checkpoints:
            if checkpoint.trigger_time <= current:
                if most_recent is None or checkpoint.priority > most_recent.priority:
                    most_recent = checkpoint
        
        return most_recent.name if most_recent else None


class CheckpointSnapshot:
    """Snapshot of evaluations at a specific checkpoint."""
    
    def __init__(
        self,
        checkpoint_name: str,
        snapshot_time: datetime,
        evaluations: List[SingleLegEvaluation]
    ):
        self.checkpoint_name = checkpoint_name
        self.snapshot_time = snapshot_time
        self.evaluations = evaluations
    
    def get_tier_distribution(self) -> Dict[str, int]:
        """Get count of legs by tier."""
        distribution = {}
        for eval in self.evaluations:
            tier = eval.leg_status
            distribution[tier] = distribution.get(tier, 0) + 1
        return distribution


class CheckpointTracker:
    """
    Tracks evaluations across multiple checkpoints throughout the day.
    """
    
    def __init__(self):
        self.snapshots: Dict[str, CheckpointSnapshot] = {}
        self.tier_transitions: Dict[str, List[str]] = {}  # event_id → [old_tier, new_tier]
    
    def add_snapshot(
        self,
        checkpoint_name: str,
        snapshot_time: datetime,
        evaluations: List[SingleLegEvaluation]
    ):
        """Add a snapshot at a checkpoint."""
        
        snapshot = CheckpointSnapshot(checkpoint_name, snapshot_time, evaluations)
        self.snapshots[checkpoint_name] = snapshot
        
        # Track tier transitions
        if len(self.snapshots) > 1:
            self._track_transitions(evaluations)
    
    def _track_transitions(self, evaluations: List[SingleLegEvaluation]):
        """Track tier changes between last snapshot and current."""
        
        # Find previous checkpoint
        previous_checkpoint = None
        checkpoints = list(self.snapshots.keys())
        if len(checkpoints) >= 2:
            previous_checkpoint = checkpoints[-2]
        
        if not previous_checkpoint:
            return
        
        previous_evals = {
            e.event_id: e for e in self.snapshots[previous_checkpoint].evaluations
        }
        
        for current_eval in evaluations:
            previous_eval = previous_evals.get(current_eval.event_id)
            if previous_eval:
                if current_eval.leg_status != previous_eval.leg_status:
                    self.tier_transitions[current_eval.event_id] = [
                        previous_eval.leg_status,
                        current_eval.leg_status
                    ]
    
    def get_summary(self) -> Dict:
        """Get summary of checkpoint tracking."""
        
        summary = {
            "checkpoints_run": len(self.snapshots),
            "tier_distributions": {},
            "tier_transitions": self.tier_transitions,
            "news_resolved": 0,
            "price_improved": 0,
        }
        
        # Count NEWS_DEPENDENT to BALANCED transitions
        for event_id, (old_tier, new_tier) in self.tier_transitions.items():
            if old_tier == LegStatus.NEWS_DEPENDENT.value and new_tier in [
                LegStatus.BALANCED_PLAYABLE.value,
                LegStatus.SEED_PLAYABLE.value
            ]:
                summary["news_resolved"] += 1
            
            if old_tier == LegStatus.PRICE_DEPENDENT.value and new_tier in [
                LegStatus.BALANCED_PLAYABLE.value,
                LegStatus.SEED_PLAYABLE.value
            ]:
                summary["price_improved"] += 1
        
        # Tier distributions per checkpoint
        for checkpoint_name, snapshot in self.snapshots.items():
            summary["tier_distributions"][checkpoint_name] = snapshot.get_tier_distribution()
        
        return summary


class PriceTracker:
    """Tracks price movements between checkpoints."""
    
    def __init__(self):
        self.price_history: Dict[str, List[Dict]] = {}  # event_id → list of (time, price)
    
    def record_price(
        self,
        event_id: str,
        american_odds: float,
        checkpoint_time: datetime
    ):
        """Record a price for an event."""
        
        if event_id not in self.price_history:
            self.price_history[event_id] = []
        
        self.price_history[event_id].append({
            "time": checkpoint_time,
            "odds": american_odds,
        })
    
    def get_price_movement(self, event_id: str) -> Dict:
        """Calculate price movement for an event."""
        
        if event_id not in self.price_history or len(self.price_history[event_id]) < 2:
            return {
                "movement_pct": 0.0,
                "price_stable": True,
                "samples": len(self.price_history.get(event_id, [])),
            }
        
        prices = self.price_history[event_id]
        first_price = prices[0]["odds"]
        latest_price = prices[-1]["odds"]
        
        if first_price == 0:
            movement_pct = 0.0
        else:
            # Movement as percentage change
            movement_pct = abs(latest_price - first_price) / abs(first_price)
        
        # Threshold: 8% movement is significant
        price_stable = movement_pct < 0.08
        
        return {
            "movement_pct": movement_pct,
            "price_stable": price_stable,
            "first_price": first_price,
            "latest_price": latest_price,
            "samples": len(prices),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test checkpoint manager
    manager = CheckpointManager()
    now = datetime.now()
    
    print(f"Current time: {now}")
    print(f"Next checkpoint: {manager.get_next_checkpoint(now)}")
    print(f"Current checkpoint: {manager.get_checkpoint_for_time(now)}")
