"""
PHASE 14: Detailed HTML Reporting

For each parlay (accepted or rejected):
- Legs (player, market, line, odds, side)
- Prices (individual, break-even, parlay, SGP vs synthetic)
- Probabilities (individual, joint, stress, LCB)
- EV (individual legs, joint, stress EV)
- Line-zone classification (NEAR_MEDIAN, etc.)
- Shared event supply conflicts
- Team failure mode exposure
- Decision label (SEED_SHADOW, BALANCED_SHADOW, etc.)
- Rejection reasons with explanation

Output: HTML with interactive tables, visualizations, failure mode breakdown
"""

import html
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .data_types import ParlayCandidate, SingleLegEvaluation

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generate detailed HTML reports for parlay system."""
    
    def __init__(self):
        self.report_parts = []
    
    def generate_report(
        self,
        title: str,
        summary_stats: Dict,
        accepted_parlays: List[Dict],
        rejected_parlays: List[Dict],
        single_legs: List[SingleLegEvaluation],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Generate complete HTML report.
        
        accepted_parlays: List of {
            "parlay": ParlayCandidate,
            "tier": str,
            "score": float,
            "joint_prob": float,
            "lcb_joint_ev": float,
            "failure_modes": List[str],
            "exposure": Dict,
        }
        
        rejected_parlays: List of {
            "parlay": ParlayCandidate,
            "rejection_reason": str,
            "details": Dict,
        }
        """
        
        timestamp = timestamp or datetime.now()
        
        html_parts = []
        
        # Header
        html_parts.append(self._generate_header(title, timestamp))
        
        # Summary section
        html_parts.append(self._generate_summary_section(summary_stats))
        
        # Accepted parlays
        if accepted_parlays:
            html_parts.append(self._generate_accepted_section(accepted_parlays))
        
        # Rejected parlays
        if rejected_parlays:
            html_parts.append(self._generate_rejected_section(rejected_parlays))
        
        # Single legs (seeds + balanced)
        if single_legs:
            html_parts.append(self._generate_single_legs_section(single_legs))
        
        # Footer
        html_parts.append(self._generate_footer())
        
        return "\n".join(html_parts)
    
    def _generate_header(self, title: str, timestamp: datetime) -> str:
        """Generate HTML header."""
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #2ca02c; margin-top: 30px; }}
        h3 {{ color: #d62728; }}
        
        .header {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .timestamp {{ color: #666; font-size: 0.9em; }}
        
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .summary-card {{
            background-color: #f9f9f9;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            border-radius: 3px;
        }}
        
        .summary-card h4 {{
            margin: 0 0 8px 0;
            color: #1f77b4;
        }}
        
        .summary-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        th {{
            background-color: #1f77b4;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{ background-color: #f0f0f0; }}
        
        .tier-SEED {{ background-color: #d4edda; }}
        .tier-BALANCED {{ background-color: #fff3cd; }}
        .tier-BOUNDARY {{ background-color: #f8d7da; }}
        
        .leg-box {{
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
        }}
        
        .leg-player {{
            font-weight: bold;
            color: #1f77b4;
        }}
        
        .leg-market {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .leg-odds {{
            float: right;
            font-weight: bold;
            color: #d62728;
        }}
        
        .metrics-table {{
            font-size: 0.85em;
            margin-top: 10px;
        }}
        
        .metrics-table td {{
            padding: 6px 12px;
        }}
        
        .rejection-reason {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 3px;
        }}
        
        .failure-mode {{
            background-color: #e7d4f5;
            border: 1px solid #d1b3e8;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{html.escape(title)}</h1>
        <div class="timestamp">Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}</div>
    </div>
"""
    
    def _generate_summary_section(self, stats: Dict) -> str:
        """Generate summary statistics section."""
        
        html = '<div class="summary"><h2>Summary</h2><div class="summary-grid">'
        
        for key, value in stats.items():
            display_key = key.replace("_", " ").title()
            html += f"""
            <div class="summary-card">
                <h4>{display_key}</h4>
                <div class="value">{value}</div>
            </div>
            """
        
        html += '</div></div>'
        return html
    
    def _generate_accepted_section(self, accepted_parlays: List[Dict]) -> str:
        """Generate accepted parlays section."""
        
        html = '<div class="section"><h2>Accepted Parlays</h2>'
        
        for i, parlay_info in enumerate(accepted_parlays, 1):
            parlay = parlay_info.get("parlay")
            tier = parlay_info.get("tier", "UNKNOWN")
            score = parlay_info.get("score", 0.0)
            joint_prob = parlay_info.get("joint_prob", 0.0)
            lcb_ev = parlay_info.get("lcb_joint_ev", 0.0)
            
            tier_class = f"tier-{tier.split('_')[0]}" if tier else ""
            
            html += f'<div style="margin-bottom: 20px; padding: 15px; background-color: #f0f8f0; border-radius: 5px; {tier_class}">'
            html += f'<h3>Parlay #{i} - {tier}</h3>'
            html += f'<p><strong>Score:</strong> {score:.4f} | <strong>Joint Prob:</strong> {joint_prob:.1%} | <strong>LCB EV:</strong> {lcb_ev:.2%}</p>'
            
            # Legs
            html += '<div><strong>Legs:</strong>'
            if hasattr(parlay, 'legs'):
                for leg in parlay.legs:
                    player = getattr(leg, 'player_name', 'Unknown')
                    market = getattr(leg, 'market_family', 'Unknown')
                    odds = getattr(leg, 'american_odds', 'N/A')
                    side = getattr(leg, 'side', 'Unknown')
                    html += f'<div class="leg-box">'
                    html += f'<div class="leg-player">{html.escape(player)}</div>'
                    html += f'<div class="leg-market">{market} {side}</div>'
                    html += f'<div class="leg-odds">{odds}</div>'
                    html += f'</div>'
            html += '</div>'
            
            # Failure modes
            failure_modes = parlay_info.get("failure_modes", [])
            if failure_modes:
                html += '<div><strong>Failure Mode Exposure:</strong>'
                for mode in failure_modes:
                    html += f'<div class="failure-mode">{html.escape(mode)}</div>'
                html += '</div>'
            
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _generate_rejected_section(self, rejected_parlays: List[Dict]) -> str:
        """Generate rejected parlays section."""
        
        html = '<div class="section"><h2>Rejected Parlays</h2>'
        html += f'<p>{len(rejected_parlays)} parlays rejected.</p>'
        
        # Group by rejection reason
        by_reason = {}
        for reject_info in rejected_parlays:
            reason = reject_info.get("rejection_reason", "Unknown")
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(reject_info)
        
        for reason, rejects in by_reason.items():
            html += f'<h3>{html.escape(reason)} ({len(rejects)} parlays)</h3>'
            html += '<ul>'
            for i, reject_info in enumerate(rejects[:5], 1):  # Show first 5
                details = reject_info.get("details", {})
                detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
                html += f'<li>{detail_str}</li>'
            if len(rejects) > 5:
                html += f'<li><em>... and {len(rejects) - 5} more</em></li>'
            html += '</ul>'
        
        html += '</div>'
        return html
    
    def _generate_single_legs_section(self, legs: List[SingleLegEvaluation]) -> str:
        """Generate single legs section."""
        
        html = '<div class="section"><h2>Single Legs Available</h2>'
        
        # Filter to playable legs
        playable = [l for l in legs if l.is_accepted()]
        
        if playable:
            html += f'<p>{len(playable)} playable legs available.</p>'
            
            # Create table
            html += '<table><thead><tr>'
            html += '<th>Player</th><th>Market</th><th>LCB Edge</th><th>Tier</th><th>Line Zone</th>'
            html += '</tr></thead><tbody>'
            
            for leg in playable[:50]:  # Show first 50
                player = html.escape(getattr(leg, 'player_name', 'Unknown'))
                market = getattr(leg, 'market_family', 'Unknown')
                edge = getattr(leg, 'lcb_edge_pct', 0.0)
                tier = leg.leg_status
                zone = getattr(leg, 'line_zone_classification', 'Unknown')
                
                tier_class = f"tier-{tier.split('_')[0]}" if tier else ""
                
                html += f'<tr class="{tier_class}">'
                html += f'<td>{player}</td>'
                html += f'<td>{market}</td>'
                html += f'<td class="positive">{edge:.2%}</td>'
                html += f'<td>{tier}</td>'
                html += f'<td>{zone}</td>'
                html += '</tr>'
            
            html += '</tbody></table>'
        else:
            html += '<p><em>No playable legs available.</em></p>'
        
        html += '</div>'
        return html
    
    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        
        return """
    <div class="footer">
        <p>NBA Parlay System - Phase 14 HTML Report Generator</p>
        <p>SHADOW MODE - No real money has been wagered.</p>
    </div>
</body>
</html>
"""


class ReportBuilder:
    """Builder for creating reports with incremental updates."""
    
    def __init__(self, title: str):
        self.title = title
        self.summary_stats = {}
        self.accepted_parlays = []
        self.rejected_parlays = []
        self.single_legs = []
    
    def add_summary_stat(self, key: str, value):
        """Add a summary statistic."""
        self.summary_stats[key] = value
        return self
    
    def add_accepted_parlay(self, parlay_info: Dict):
        """Add an accepted parlay."""
        self.accepted_parlays.append(parlay_info)
        return self
    
    def add_rejected_parlay(self, parlay_info: Dict):
        """Add a rejected parlay."""
        self.rejected_parlays.append(parlay_info)
        return self
    
    def add_single_leg(self, leg: SingleLegEvaluation):
        """Add a single leg evaluation."""
        self.single_legs.append(leg)
        return self
    
    def build(self) -> str:
        """Build the final HTML report."""
        generator = HTMLReportGenerator()
        return generator.generate_report(
            self.title,
            self.summary_stats,
            self.accepted_parlays,
            self.rejected_parlays,
            self.single_legs
        )
    
    def save(self, filepath: str):
        """Save report to file."""
        html = self.build()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Report saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("HTMLReportGenerator module loaded.")
