"""
analyze_players.py - Comprehensive player analysis

Consolidates functionality from:
- core/breakout_detector.py
- core/def_portability.py
- core/impact_sanity.py
- core/scouting_report.py
- core/scenario_screening.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import math


def load_player_card(card_path: Path) -> Dict[str, Any]:
    """Load player card from JSON"""
    with open(card_path, 'r') as f:
        return json.load(f)


def safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert to float"""
    try:
        return float(x) if x is not None else default
    except:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between bounds"""
    return max(lo, min(hi, x))


# ============================================================================
# BREAKOUT DETECTION
# ============================================================================

def detect_breakout_potential(card: Dict[str, Any]) -> Dict[str, Any]:
    """Detect breakout opportunity for a player"""
    player = card.get('player', {})
    identity = card.get('identity', {})
    offense = card.get('offense', {})
    metadata = card.get('metadata', {})
    performance = card.get('performance', {})
    traditional = performance.get('traditional', {})
    advanced = performance.get('advanced', {})
    
    age = player.get('age', 25.0)
    usage_band = identity.get('usage_band', 'low')
    usage_rate = safe_float(advanced.get('usage_rate', None), float('nan'))
    archetype = identity.get('primary_archetype', 'default')
    
    # Opportunity score (based on age and current usage)
    if age < 24:
        age_factor = 0.9
    elif age <= 27:
        age_factor = 1.0
    elif age <= 30:
        age_factor = 0.7
    else:
        age_factor = 0.4
    
    # Prefer precise usage rate when available; keep band fallback for legacy cards.
    if not math.isnan(usage_rate):
        if usage_rate <= 0.22:
            usage_factor = 0.5
        elif usage_rate <= 0.26:
            # Peak opportunity in medium-usage range.
            usage_factor = 0.5 + ((usage_rate - 0.22) / 0.04) * 0.2
        elif usage_rate <= 0.32:
            usage_factor = 0.7 - ((usage_rate - 0.26) / 0.06) * 0.4
        else:
            usage_factor = 0.3
    else:
        usage_factor = 0.5 if usage_band == 'low' else (0.7 if usage_band == 'med' else 0.3)
    
    opportunity_score = (age_factor * 0.6 + usage_factor * 0.4) * 100
    
    # Signal strength (based on efficiency and creation)
    creation = offense.get('creation', {})
    efficiency = offense.get('efficiency', {})
    
    scoring = safe_float(creation.get('scoring', traditional.get('points_per_game', 0.0)))
    ast_tov = safe_float(efficiency.get('ast_tov_ratio', 0.0))
    if ast_tov <= 0:
        assists = safe_float(traditional.get('assists_per_game', 0.0))
        turnovers = safe_float(traditional.get('turnovers_per_game', 0.0))
        ast_tov = assists / max(turnovers, 0.8) if assists > 0 else 1.0
    
    signal_strength = min(100, (scoring / 20.0 * 50) + (min(ast_tov / 2.0, 1.0) * 50))
    
    # Confidence (based on sample size)
    games_source = metadata.get('games_played', traditional.get('games_played', 0))
    games = int(str(games_source).replace(',', ''))
    confidence = min(1.0, games / 60.0)
    
    # Can breakout happen?
    can_breakout = (opportunity_score >= 40 and signal_strength >= 35 and confidence >= 0.5)
    
    return {
        "player": {
            "name": player.get('name'),
            "team": player.get('team'),
            "age": age
        },
        "can_breakout": can_breakout,
        "opportunity_score": round(opportunity_score, 1),
        "signal_strength": round(signal_strength, 1),
        "confidence": round(confidence, 3),
        "factors": {
            "age_factor": round(age_factor, 2),
            "usage_factor": round(usage_factor, 2),
            "current_usage_band": usage_band,
            "current_usage_rate": None if math.isnan(usage_rate) else round(usage_rate, 3)
        }
    }


# ============================================================================
# DEFENSE PORTABILITY
# ============================================================================

def analyze_defense_portability(card: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze defensive portability and matchup flexibility"""
    player = card.get('player', {})
    defense = card.get('defense', {})
    defense_assessment = card.get('defense_assessment', {})
    identity = card.get('identity', {})
    performance = card.get('performance', {})
    traditional = performance.get('traditional', {})
    
    position = (
        identity.get('position')
        or player.get('position')
        or ''
    )
    performance = defense.get('performance', {})
    matchup_profile = defense_assessment.get('matchup_profile', {})
    
    # Defensive role
    stocks = safe_float(
        performance.get(
            'stocks_per_game',
            safe_float(traditional.get('steals_per_game', 0.0)) + safe_float(traditional.get('blocks_per_game', 0.0))
        )
    )
    
    # Position-based role weights
    pos_upper = str(position).upper()
    if pos_upper in ['PG', 'SG', 'G', 'GUARD']:
        role = "perimeter_guard"
        role_base = 0.55
    elif pos_upper in ['SF', 'F', 'WING', 'FORWARD']:
        role = "wing_defender"
        role_base = 0.75
    elif pos_upper in ['PF', 'C', 'BIG', 'CENTER']:
        role = "versatile_forward"
        role_base = 0.45
    else:
        role = "unknown"
        role_base = 0.5
    
    # Versatility from matchup profile distribution.
    vs_guards = safe_float(matchup_profile.get('vs_guards', 0.0))
    vs_wings = safe_float(matchup_profile.get('vs_wings', 0.0))
    vs_bigs = safe_float(matchup_profile.get('vs_bigs', 0.0))
    matchup_total = vs_guards + vs_wings + vs_bigs
    if matchup_total > 0:
        probs = [vs_guards / matchup_total, vs_wings / matchup_total, vs_bigs / matchup_total]
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy += -p * math.log(p)
        versatility = entropy / math.log(3.0)
    else:
        versatility = 0.5

    switch_ability = clamp(0.5 * role_base + 0.5 * versatility, 0.0, 1.0)

    # Portability score based on player-only inputs (no team defensive metrics).
    mpg = safe_float(traditional.get('minutes_per_game', 0.0))
    games = safe_float(traditional.get('games_played', 0.0))
    stocks_per_36 = stocks * 36.0 / max(mpg, 8.0)
    event_creation = min(1.0, stocks_per_36 / 3.2)
    burden_factor = clamp((mpg / 36.0) * 0.8 + min(1.0, games / 82.0) * 0.2, 0.0, 1.0)
    
    portability_score = (switch_ability * 0.45 + event_creation * 0.35 + burden_factor * 0.20)
    
    if portability_score >= 0.75:
        portability_level = "high"
    elif portability_score >= 0.55:
        portability_level = "medium"
    else:
        portability_level = "low"
    
    # Failure modes
    failure_modes = []
    if stocks_per_36 < 1.2:
        failure_modes.append("Low defensive event creation")
    if switch_ability < 0.5:
        failure_modes.append("Limited switchability")
    if burden_factor < 0.4:
        failure_modes.append("Low defensive burden - role player")
    
    return {
        "player": {
            "name": player.get('name'),
            "team": player.get('team'),
            "position": position
        },
        "defensive_role": role,
        "portability": {
            "score": round(portability_score, 3),
            "level": portability_level,
            "switch_ability": round(switch_ability, 2)
        },
        "failure_modes": failure_modes
    }


# ============================================================================
# IMPACT SANITY CHECK
# ============================================================================

def check_impact_sanity(card: Dict[str, Any]) -> Dict[str, Any]:
    """Sanity check impact metrics"""
    player = card.get('player', {})
    impact = card.get('impact', {})
    offense = card.get('offense', {})
    defense = card.get('defense', {})
    
    net_impact = safe_float(impact.get('net', 0.0))
    off_impact = safe_float(impact.get('offensive', 0.0))
    def_impact = safe_float(impact.get('defensive', 0.0))
    
    # Sanity checks
    flags = []
    
    # Check 1: Net impact should roughly equal off + def
    expected_net = off_impact + def_impact
    if abs(net_impact - expected_net) > 2.0:
        flags.append(f"Net impact mismatch: {net_impact:.1f} vs expected {expected_net:.1f}")
    
    # Check 2: Offensive impact should align with scoring
    scoring = safe_float(offense.get('creation', {}).get('scoring', 0.0))
    if off_impact > 3.0 and scoring < 10.0:
        flags.append(f"High offensive impact ({off_impact:.1f}) but low scoring ({scoring:.1f})")
    
    # Check 3: Defensive impact should align with stocks
    stocks = safe_float(defense.get('performance', {}).get('stocks_per_game', 0.0))
    if def_impact > 2.0 and stocks < 1.0:
        flags.append(f"High defensive impact ({def_impact:.1f}) but low stocks ({stocks:.1f})")
    
    # Overall sanity
    if len(flags) == 0:
        sanity_level = "pass"
    elif len(flags) <= 1:
        sanity_level = "warning"
    else:
        sanity_level = "fail"
    
    return {
        "player": {
            "name": player.get('name'),
            "team": player.get('team')
        },
        "sanity_level": sanity_level,
        "flags": flags,
        "impact_summary": {
            "net": round(net_impact, 2),
            "offensive": round(off_impact, 2),
            "defensive": round(def_impact, 2)
        }
    }


# ============================================================================
# SCOUTING REPORT
# ============================================================================

def generate_scouting_report(card: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive scouting report"""
    player = card.get('player', {})
    identity = card.get('identity', {})
    offense = card.get('offense', {})
    defense = card.get('defense', {})
    impact = card.get('impact', {})
    
    # Strengths and weaknesses
    strengths = []
    weaknesses = []
    
    # Offense
    shot_profile = offense.get('shot_profile', {})
    three_rate = safe_float(shot_profile.get('three_rate', 0.0))
    
    if three_rate >= 0.4:
        strengths.append("High three-point volume")
    elif three_rate < 0.2:
        weaknesses.append("Limited three-point shooting")
    
    creation = offense.get('creation', {})
    scoring = safe_float(creation.get('scoring', 0.0))
    
    if scoring >= 18.0:
        strengths.append("High-volume scorer")
    elif scoring < 8.0:
        weaknesses.append("Limited scoring production")
    
    # Defense
    burden = defense.get('burden', {})
    if burden.get('level') == 'high':
        strengths.append("High defensive burden")
    elif burden.get('level') == 'low':
        weaknesses.append("Low defensive responsibility")
    
    # Impact
    net = safe_float(impact.get('net', 0.0))
    if net >= 3.0:
        strengths.append("Strong positive impact")
    elif net < -1.0:
        weaknesses.append("Negative impact")
    
    # Role summary
    usage_band = identity.get('usage_band', 'low')
    archetype = identity.get('primary_archetype', 'unknown')
    
    if usage_band == 'high':
        role_summary = f"High-usage {archetype}"
    elif usage_band == 'med':
        role_summary = f"Medium-usage {archetype}"
    else:
        role_summary = f"Low-usage {archetype}"
    
    return {
        "player": {
            "name": player.get('name'),
            "team": player.get('team'),
            "position": identity.get('position'),
            "age": player.get('age')
        },
        "role_summary": role_summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "archetype": archetype,
        "usage_band": usage_band
    }


# ============================================================================
# MAIN ANALYSIS RUNNER
# ============================================================================

def analyze_player(card: Dict[str, Any]) -> Dict[str, Any]:
    """Run all analyses on a player"""
    return {
        "scouting_report": generate_scouting_report(card),
        "breakout_potential": detect_breakout_potential(card),
        "defense_portability": analyze_defense_portability(card),
        "impact_sanity": check_impact_sanity(card)
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive player analysis")
    parser.add_argument('--cards', type=Path, required=True, help="Player cards directory or file")
    parser.add_argument('--output', type=Path, default=Path('analysis'), help="Output directory")
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process cards
    if args.cards.is_file():
        card_paths = [args.cards]
    else:
        card_paths = list(args.cards.glob('*.json'))
        card_paths = [p for p in card_paths if not p.name.startswith('cards_summary')]
    
    print(f"Analyzing {len(card_paths)} players...")
    
    results = []
    breakout_candidates = []
    high_portability = []
    sanity_failures = []
    
    for card_path in card_paths:
        try:
            card = load_player_card(card_path)
            analysis = analyze_player(card)
            
            # Save individual analysis
            player_name = card['player']['name'].replace(' ', '_')
            output_path = args.output / f"{player_name}_analysis.json"
            
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            results.append(analysis)
            
            # Collect interesting cases
            if analysis['breakout_potential']['can_breakout']:
                breakout_candidates.append(analysis['breakout_potential'])
            
            if analysis['defense_portability']['portability']['level'] == 'high':
                high_portability.append(analysis['defense_portability'])
            
            if analysis['impact_sanity']['sanity_level'] == 'fail':
                sanity_failures.append(analysis['impact_sanity'])
            
        except Exception as e:
            print(f"  Error analyzing {card_path.name}: {e}")
            continue
    
    # Generate summary
    summary = {
        "total_players": len(results),
        "breakout_count": len(breakout_candidates),
        "breakout_rate": round((len(breakout_candidates) / len(results)) if results else 0.0, 4),
        "high_portability_count": len(high_portability),
        "high_portability_rate": round((len(high_portability) / len(results)) if results else 0.0, 4),
        "impact_sanity_failure_count": len(sanity_failures),
        "breakout_candidates": sorted(
            breakout_candidates,
            key=lambda x: x['opportunity_score'],
            reverse=True
        )[:20],
        "high_portability_defenders": sorted(
            high_portability,
            key=lambda x: x['portability']['score'],
            reverse=True
        )[:20],
        "impact_sanity_failures": sanity_failures
    }
    
    with open(args.output / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUCCESS] Analyzed {len(results)} players")
    print(f"  Breakout candidates: {len(breakout_candidates)}")
    print(f"  High portability defenders: {len(high_portability)}")
    print(f"  Impact sanity failures: {len(sanity_failures)}")
    print(f"  Reports saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
