"""
PHASE 13: Historical Backtesting & Validation

Compare strategies:
1. Naive top-probability stacking (highest P all legs hit)
2. Naive top-EV stacking (highest EV regardless of state)
3. Single-leg robust set only (edge filter, no parlays)
4. Anchor+companion (this system)
5. Joint-state filtering (this system with team environment)
6. Stress EV (this system with stress testing)

Compute metrics:
- Hit rate (% of plays that won)
- ROI (units won / units wagered)
- Profit units (absolute units won)
- Brier score (probability calibration)
- Expected Calibration Error (ECE)
- Max drawdown
- Sharpe ratio

Validation hypothesis:
- Joint-state filtering outperforms naive stacking by >5% ROI
- Stress EV > non-stress EV
- Alt-line expansion maintains calibration (ECE <0.05)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from data_types import ParlayCandidate, JointState

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a single parlay/leg bet in historical data."""
    
    event_id: str
    prediction_time: datetime
    predicted_prob: float
    actual_outcome: bool  # Did it hit?
    american_odds: float
    units_wagered: float = 1.0
    units_won: float = 0.0
    

@dataclass
class StrategyMetrics:
    """Metrics for a backtest strategy."""
    
    strategy_name: str
    total_plays: int
    winning_plays: int
    hit_rate: float
    total_units_wagered: float
    total_units_won: float
    roi: float  # (total_units_won - total_units_wagered) / total_units_wagered
    brier_score: float  # Mean squared error of probabilities
    ece: float  # Expected Calibration Error
    max_drawdown: float
    sharpe_ratio: float
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.strategy_name}:\n"
            f"  Hit Rate: {self.hit_rate:.1%}\n"
            f"  ROI: {self.roi:.1%}\n"
            f"  Plays: {self.total_plays}\n"
            f"  Brier Score: {self.brier_score:.4f}\n"
            f"  ECE: {self.ece:.4f}\n"
        )


class HistoricalValidator:
    """
    Backtest system against historical outcomes.
    """
    
    def __init__(self, settled_outcomes_path: Optional[str] = None):
        """
        settled_outcomes_path: CSV with columns:
            - event_id
            - prediction_time
            - predicted_prob
            - actual_outcome (0/1)
            - american_odds
        """
        self.settled_outcomes_path = settled_outcomes_path
        self.settled_df = None
        
        if settled_outcomes_path:
            self.load_settled_outcomes(settled_outcomes_path)
    
    def load_settled_outcomes(self, path: str) -> bool:
        """Load settled outcomes."""
        try:
            self.settled_df = pd.read_csv(path)
            self.settled_df["prediction_time"] = pd.to_datetime(
                self.settled_df["prediction_time"]
            )
            logger.info(f"Loaded {len(self.settled_df)} settled outcomes")
            return True
        except Exception as e:
            logger.error(f"Failed to load settled outcomes: {e}")
            return False
    
    def backtest_strategy(
        self,
        strategy_name: str,
        predictions: List[Dict],
        leg_weights: Optional[Dict[str, float]] = None
    ) -> Optional[StrategyMetrics]:
        """
        Backtest a strategy against settled outcomes.
        
        predictions: List of {
            "event_id": str,
            "predicted_prob": float,
            "american_odds": float,
            "units": float (optional, default 1.0)
        }
        
        leg_weights: For 2-leg parlays, weight of each leg if provided
        """
        
        if not self.settled_df or self.settled_df.empty:
            logger.warning(f"No settled outcomes available for {strategy_name}")
            return None
        
        results = []
        
        for pred in predictions:
            event_id = pred["event_id"]
            pred_prob = pred["predicted_prob"]
            american_odds = pred["american_odds"]
            units = pred.get("units", 1.0)
            
            # Find settled outcome
            settled = self.settled_df[self.settled_df["event_id"] == event_id]
            
            if settled.empty:
                continue
            
            actual_outcome = bool(settled.iloc[0]["actual_outcome"])
            
            # Calculate units won
            if actual_outcome:
                # Convert American odds to decimal
                if american_odds > 0:
                    decimal_odds = 1 + (american_odds / 100)
                else:
                    decimal_odds = 1 + (100 / abs(american_odds))
                
                units_won = units * (decimal_odds - 1)
            else:
                units_won = 0.0
            
            results.append(BacktestResult(
                event_id=event_id,
                prediction_time=datetime.now(),
                predicted_prob=pred_prob,
                actual_outcome=actual_outcome,
                american_odds=american_odds,
                units_wagered=units,
                units_won=units_won,
            ))
        
        if not results:
            logger.warning(f"No settled outcomes matched for {strategy_name}")
            return None
        
        # Compute metrics
        return self._compute_metrics(strategy_name, results)
    
    def _compute_metrics(
        self,
        strategy_name: str,
        results: List[BacktestResult]
    ) -> StrategyMetrics:
        """Compute metrics from backtest results."""
        
        total_plays = len(results)
        winning_plays = sum(1 for r in results if r.actual_outcome)
        hit_rate = winning_plays / total_plays if total_plays > 0 else 0.0
        
        total_units_wagered = sum(r.units_wagered for r in results)
        total_units_won = sum(r.units_won for r in results)
        
        roi = (total_units_won - total_units_wagered) / total_units_wagered if total_units_wagered > 0 else 0.0
        
        # Brier score: mean squared error of probability
        brier_score = np.mean([
            (r.predicted_prob - (1.0 if r.actual_outcome else 0.0)) ** 2
            for r in results
        ])
        
        # Expected Calibration Error
        ece = self._compute_ece(results)
        
        # Max drawdown
        max_drawdown = self._compute_max_drawdown(results)
        
        # Sharpe ratio (units won / std dev of units won)
        units_sequence = [r.units_won - r.units_wagered for r in results]
        sharpe_ratio = np.mean(units_sequence) / (np.std(units_sequence) + 1e-6)
        
        return StrategyMetrics(
            strategy_name=strategy_name,
            total_plays=total_plays,
            winning_plays=winning_plays,
            hit_rate=hit_rate,
            total_units_wagered=total_units_wagered,
            total_units_won=total_units_won,
            roi=roi,
            brier_score=brier_score,
            ece=ece,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
        )
    
    def _compute_ece(self, results: List[BacktestResult]) -> float:
        """Compute Expected Calibration Error."""
        
        if not results:
            return 0.0
        
        # Bin predictions into 10 buckets
        bins = np.arange(0, 1.1, 0.1)
        
        ece = 0.0
        for i in range(len(bins) - 1):
            bin_min, bin_max = bins[i], bins[i + 1]
            
            # Get predictions in this bin
            in_bin = [r for r in results if bin_min <= r.predicted_prob < bin_max]
            
            if not in_bin:
                continue
            
            # Expected probability (mean of predictions in bin)
            expected_prob = np.mean([r.predicted_prob for r in in_bin])
            
            # Actual outcome rate
            actual_rate = np.mean([1.0 if r.actual_outcome else 0.0 for r in in_bin])
            
            # Calibration error for this bin (weighted by bin size)
            bin_weight = len(in_bin) / len(results)
            ece += bin_weight * abs(expected_prob - actual_rate)
        
        return ece
    
    def _compute_max_drawdown(self, results: List[BacktestResult]) -> float:
        """Compute maximum drawdown."""
        
        cumulative_units = 0.0
        peak = 0.0
        max_drawdown = 0.0
        
        for r in results:
            cumulative_units += r.units_won - r.units_wagered
            peak = max(peak, cumulative_units)
            drawdown = peak - cumulative_units
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown


class StrategyComparison:
    """Compare multiple strategies against each other."""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyMetrics] = {}
    
    def add_strategy(self, metrics: StrategyMetrics):
        """Add a strategy's metrics."""
        self.strategies[metrics.strategy_name] = metrics
    
    def get_best_strategy(self, metric_name: str = "roi") -> Optional[StrategyMetrics]:
        """Get best strategy by a specific metric."""
        
        if not self.strategies:
            return None
        
        if metric_name == "roi":
            return max(self.strategies.values(), key=lambda m: m.roi)
        elif metric_name == "hit_rate":
            return max(self.strategies.values(), key=lambda m: m.hit_rate)
        elif metric_name == "sharpe_ratio":
            return max(self.strategies.values(), key=lambda m: m.sharpe_ratio)
        elif metric_name == "ece":
            return min(self.strategies.values(), key=lambda m: m.ece)
        
        return None
    
    def comparison_table(self) -> pd.DataFrame:
        """Get comparison table."""
        
        data = []
        for strategy_name, metrics in self.strategies.items():
            data.append({
                "Strategy": strategy_name,
                "Plays": metrics.total_plays,
                "Hit Rate": f"{metrics.hit_rate:.1%}",
                "ROI": f"{metrics.roi:.1%}",
                "Brier": f"{metrics.brier_score:.4f}",
                "ECE": f"{metrics.ece:.4f}",
                "Sharpe": f"{metrics.sharpe_ratio:.2f}",
                "Max DD": f"{metrics.max_drawdown:.1f}",
            })
        
        return pd.DataFrame(data)
    
    def summary_report(self) -> str:
        """Get summary report."""
        
        report = "STRATEGY COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for strategy_name, metrics in self.strategies.items():
            report += metrics.summary()
            report += "\n"
        
        # Recommend best strategy
        best_roi = self.get_best_strategy("roi")
        if best_roi:
            report += f"\n\nBEST ROI: {best_roi.strategy_name} ({best_roi.roi:.1%})\n"
        
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("HistoricalValidator module loaded.")
