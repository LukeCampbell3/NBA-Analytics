"""
value_players.py - Player valuation with contract analysis and aging curves

Consolidates functionality from:
- core/contract_valuation.py
- core/aging_curve.py
- core/integrated_valuation.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ValuationResult:
    """Player valuation result"""
    player_id: str
    player_name: str
    season: int
    
    # Impact
    current_wins_added: float
    
    # Market value
    market_value_by_year: Dict[int, float]
    
    # Contract
    salary_by_year: Dict[int, float]
    surplus_by_year: Dict[int, float]
    npv_surplus: float
    
    # Trade value
    trade_value_low: float
    trade_value_base: float
    trade_value_high: float
    
    # Aging
    aging_multipliers: Dict[int, float]
    peak_age: float
    current_phase: str


class PlayerValuator:
    """Main player valuation engine"""
    
    def __init__(self):
        # Market parameters
        self.dollars_per_win = 3.5  # $3.5M per win
        self.salary_cap = 141.0  # $141M
        self.discount_rate = 0.08  # 8% annual
        
        # Aging curve parameters by archetype
        self.aging_curves = {
            "initiator_creator": {"peak": 28.5, "growth": (19, 26), "plateau": (26, 31), "decline": -0.04},
            "shooting_specialist": {"peak": 30.0, "growth": (21, 28), "plateau": (28, 34), "decline": -0.03},
            "rim_protector": {"peak": 27.0, "growth": (20, 25), "plateau": (25, 30), "decline": -0.06},
            "versatile_wing": {"peak": 29.0, "growth": (21, 27), "plateau": (27, 32), "decline": -0.045},
            "connector": {"peak": 31.0, "growth": (23, 29), "plateau": (29, 35), "decline": -0.025},
            "default": {"peak": 28.0, "growth": (20, 26), "plateau": (26, 31), "decline": -0.05}
        }
    
    def load_player_card(self, card_path: Path) -> Dict[str, Any]:
        """Load player card from JSON"""
        with open(card_path, 'r') as f:
            return json.load(f)
    
    def convert_impact_to_wins(self, card: Dict[str, Any]) -> float:
        """Convert player impact to wins added"""
        impact = card.get('impact', {})
        metadata = card.get('metadata', {})
        
        net_impact = impact.get('net', 0.0)
        minutes = float(metadata.get('minutes', 0.0))
        games = int(str(metadata.get('games_played', '0')).replace(',', ''))
        
        if minutes > 0 and games > 0:
            # WAR ≈ (Net Rating × Minutes / 48 × Games) / 30
            wins = (net_impact * (minutes / 48) * games) / 30.0
        else:
            # Fallback estimate
            wins = net_impact * 0.5
        
        # Apply trust adjustment
        trust = card.get('trust', {}).get('score', 0.5)
        wins *= trust
        
        return wins
    
    def get_aging_curve(self, archetype: str) -> Dict[str, Any]:
        """Get aging curve parameters for archetype"""
        return self.aging_curves.get(archetype, self.aging_curves["default"])
    
    def calculate_aging_multiplier(self, age: float, archetype: str, years_ahead: int = 0) -> float:
        """Calculate aging multiplier for future performance"""
        curve = self.get_aging_curve(archetype)
        future_age = age + years_ahead
        
        growth_start, growth_end = curve["growth"]
        plateau_start, plateau_end = curve["plateau"]
        peak_age = curve["peak"]
        decline_rate = curve["decline"]
        
        if future_age <= growth_end:
            # Growth phase
            progress = (future_age - growth_start) / (growth_end - growth_start)
            multiplier = 0.9 + progress * 0.2  # 0.9 to 1.1
        elif future_age <= plateau_end:
            # Plateau phase
            multiplier = 1.1
        else:
            # Decline phase
            years_in_decline = future_age - plateau_end
            multiplier = 1.1 * ((1.0 + decline_rate) ** years_in_decline)
        
        return max(0.5, min(1.3, multiplier))
    
    def calculate_market_value(self, wins: float, age: float, archetype: str, season: int) -> Dict[int, float]:
        """Calculate market value for current and future years"""
        base_value = wins * self.dollars_per_win
        
        # Cap at reasonable bounds
        base_value = min(base_value, self.salary_cap * 0.35)
        base_value = max(base_value, 2.0)
        
        # Project 5 years
        market_values = {}
        for year_offset in range(5):
            year = season + year_offset
            age_mult = self.calculate_aging_multiplier(age, archetype, year_offset)
            market_values[year] = base_value * age_mult
        
        return market_values
    
    def estimate_salary(self, market_value: float) -> float:
        """Estimate salary based on market value (simplified)"""
        # Assume players get ~80% of market value on average
        return market_value * 0.8
    
    def calculate_surplus(self, market_values: Dict[int, float], salaries: Dict[int, float]) -> Dict[int, float]:
        """Calculate surplus value by year"""
        surplus = {}
        for year, market_val in market_values.items():
            salary = salaries.get(year, self.estimate_salary(market_val))
            surplus[year] = market_val - salary
        
        return surplus
    
    def calculate_npv(self, cashflows: Dict[int, float]) -> float:
        """Calculate Net Present Value"""
        npv = 0.0
        current_year = list(cashflows.keys())[0] if cashflows else 2025
        
        for year, cf in cashflows.items():
            years_diff = year - current_year
            if years_diff >= 0:
                discount_factor = 1.0 / ((1.0 + self.discount_rate) ** years_diff)
                npv += cf * discount_factor
        
        return npv
    
    def calculate_trade_value_bands(self, npv_surplus: float, uncertainty: float) -> Tuple[float, float, float]:
        """Calculate low/base/high trade value bands"""
        base = npv_surplus
        band_width = 0.3 + uncertainty * 0.4
        
        low = base * (1.0 - band_width)
        high = base * (1.0 + band_width)
        
        return max(0.0, low), base, high
    
    def identify_aging_phase(self, age: float, archetype: str) -> str:
        """Identify current aging phase"""
        curve = self.get_aging_curve(archetype)
        growth_start, growth_end = curve["growth"]
        plateau_start, plateau_end = curve["plateau"]
        
        if age < growth_start:
            return "pre_growth"
        elif age <= growth_end:
            return "growth"
        elif age <= plateau_end:
            return "plateau"
        else:
            return "decline"
    
    def valuate_player(self, card: Dict[str, Any]) -> ValuationResult:
        """Main valuation function"""
        player = card.get('player', {})
        identity = card.get('identity', {})
        
        player_id = player.get('id', '')
        player_name = player.get('name', 'Unknown')
        season = player.get('season', 2025)
        age = player.get('age', 25.0)
        archetype = identity.get('primary_archetype', 'default')
        
        # Calculate wins
        wins = self.convert_impact_to_wins(card)
        
        # Market value
        market_values = self.calculate_market_value(wins, age, archetype, season)
        
        # Estimate salaries (in real scenario, would load actual contract data)
        salaries = {year: self.estimate_salary(val) for year, val in market_values.items()}
        
        # Surplus
        surplus = self.calculate_surplus(market_values, salaries)
        npv_surplus = self.calculate_npv(surplus)
        
        # Trade value bands
        uncertainty = card.get('uncertainty', {}).get('overall', 0.5)
        trade_low, trade_base, trade_high = self.calculate_trade_value_bands(npv_surplus, uncertainty)
        
        # Aging
        aging_multipliers = {
            year_offset: self.calculate_aging_multiplier(age, archetype, year_offset)
            for year_offset in range(5)
        }
        
        curve = self.get_aging_curve(archetype)
        current_phase = self.identify_aging_phase(age, archetype)
        
        return ValuationResult(
            player_id=player_id,
            player_name=player_name,
            season=season,
            current_wins_added=wins,
            market_value_by_year=market_values,
            salary_by_year=salaries,
            surplus_by_year=surplus,
            npv_surplus=npv_surplus,
            trade_value_low=trade_low,
            trade_value_base=trade_base,
            trade_value_high=trade_high,
            aging_multipliers=aging_multipliers,
            peak_age=curve["peak"],
            current_phase=current_phase
        )
    
    def generate_report(self, result: ValuationResult) -> Dict[str, Any]:
        """Generate valuation report"""
        return {
            "player": {
                "id": result.player_id,
                "name": result.player_name,
                "season": result.season
            },
            "impact": {
                "wins_added": round(result.current_wins_added, 2)
            },
            "market_value": {
                "by_year": {str(k): round(v, 2) for k, v in result.market_value_by_year.items()}
            },
            "contract": {
                "salary_by_year": {str(k): round(v, 2) for k, v in result.salary_by_year.items()},
                "surplus_by_year": {str(k): round(v, 2) for k, v in result.surplus_by_year.items()},
                "npv_surplus": round(result.npv_surplus, 2)
            },
            "trade_value": {
                "low": round(result.trade_value_low, 2),
                "base": round(result.trade_value_base, 2),
                "high": round(result.trade_value_high, 2)
            },
            "aging": {
                "current_phase": result.current_phase,
                "peak_age": result.peak_age,
                "multipliers": {f"year_{k}": round(v, 3) for k, v in result.aging_multipliers.items()}
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Value players with contract and aging analysis")
    parser.add_argument('--cards', type=Path, required=True, help="Player cards directory or file")
    parser.add_argument('--output', type=Path, default=Path('valuations'), help="Output directory")
    
    args = parser.parse_args()
    
    valuator = PlayerValuator()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process cards
    if args.cards.is_file():
        card_paths = [args.cards]
    else:
        card_paths = list(args.cards.glob('*.json'))
        card_paths = [p for p in card_paths if not p.name.startswith('cards_summary')]
    
    print(f"Valuating {len(card_paths)} players...")
    
    results = []
    for card_path in card_paths:
        try:
            card = valuator.load_player_card(card_path)
            result = valuator.valuate_player(card)
            report = valuator.generate_report(result)
            
            # Save individual report
            player_name = result.player_name.replace(' ', '_')
            output_path = args.output / f"{player_name}_valuation.json"
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            results.append(report)
            
        except Exception as e:
            print(f"  Error valuating {card_path.name}: {e}")
            continue
    
    # Save summary
    summary = {
        "total_players": len(results),
        "top_surplus_players": sorted(
            results,
            key=lambda x: x['contract']['npv_surplus'],
            reverse=True
        )[:20]
    }
    
    with open(args.output / "valuation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUCCESS] Valuated {len(results)} players")
    print(f"  Reports saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
