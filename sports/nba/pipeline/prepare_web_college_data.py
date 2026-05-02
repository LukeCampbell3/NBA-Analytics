#!/usr/bin/env python3
"""
prepare_web_college_data.py - Build college web payloads.

Creates:
- web/data/college_cards.json
- web/data/college_valuations.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
NBA_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = NBA_ROOT.parents[1]

# Add shared src to path for imports.
sys.path.insert(0, str(REPO_ROOT / "src"))

from value_college_players import build_card_from_row, resolve_column_map  # noqa: E402
from value_players import PlayerValuator  # noqa: E402


def normalize_home_links(web_dir: Path) -> None:
    """
    Normalize Home nav anchors so they always point to site root (/).
    """
    if not web_dir.exists():
        return

    html_files = sorted(web_dir.glob("*.html"))
    pattern = re.compile(r'(<a\b[^>]*\bhref=["\'])index\.html(["\'][^>]*>\s*Home\s*</a>)', re.IGNORECASE)

    for html_path in html_files:
        try:
            original = html_path.read_text(encoding="utf-8")
        except OSError:
            continue
        updated = pattern.sub(r'\1/\2', original)
        if updated != original:
            html_path.write_text(updated, encoding="utf-8")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_usage_band(value: str) -> str:
    usage = str(value or "").lower()
    if usage == "med":
        return "medium"
    return usage if usage in {"low", "medium", "high"} else "low"


def derive_shot_profile(card: Dict[str, Any]) -> Dict[str, float]:
    trad = card.get("performance", {}).get("traditional", {})
    fga = max(0.0, safe_float(trad.get("field_goal_attempts_per_game"), 0.0))
    three_pa = max(0.0, safe_float(trad.get("three_point_attempts_per_game"), 0.0))
    three_freq = clamp((three_pa / fga) if fga > 0 else 0.33, 0.05, 0.75)
    rim_freq = clamp(0.45 - (three_freq * 0.35), 0.18, 0.62)
    mid_freq = max(0.05, 1.0 - rim_freq - three_freq)
    total = rim_freq + mid_freq + three_freq
    rim_freq /= total
    mid_freq /= total
    three_freq /= total
    return {
        "rim_frequency": round(rim_freq, 3),
        "mid_range_frequency": round(mid_freq, 3),
        "three_point_frequency": round(three_freq, 3),
    }


def derive_creation_profile(card: Dict[str, Any]) -> Dict[str, float]:
    trad = card.get("performance", {}).get("traditional", {})
    adv = card.get("performance", {}).get("advanced", {})
    position = str(card.get("player", {}).get("position", "")).upper()
    usage = safe_float(adv.get("usage_rate"), 0.20)
    assists = safe_float(trad.get("assists_per_game"), 0.0)
    points = safe_float(trad.get("points_per_game"), 0.0)
    base_assisted = 0.56
    if "G" in position:
        base_assisted = 0.48
    elif "C" in position or "B" in position:
        base_assisted = 0.70
    assisted = clamp(base_assisted - ((usage - 0.20) * 0.35), 0.20, 0.90)
    return {
        "drives_per_game": round(max(0.0, (points * 0.55) + (assists * 0.45)), 2),
        "paint_touches_per_game": round(max(0.0, points * 0.42), 2),
        "assisted_rate": round(assisted, 3),
        "isolation_frequency": round(clamp(usage * (0.30 if "G" in position else 0.18), 0.03, 0.22), 3),
        "pick_and_roll_frequency": round(clamp(usage * (0.42 if "G" in position else 0.16), 0.05, 0.35), 3),
    }


def derive_defense_assessment(card: Dict[str, Any]) -> Dict[str, Any]:
    position = str(card.get("player", {}).get("position", "")).upper()
    trust_score = safe_float(
        card.get("v1_1_enhancements", {}).get("trust_assessment", {}).get("score"),
        60.0,
    )
    if "G" in position:
        matchup = {"vs_guards": 0.64, "vs_wings": 0.28, "vs_bigs": 0.08}
    elif "C" in position or "B" in position:
        matchup = {"vs_guards": 0.10, "vs_wings": 0.26, "vs_bigs": 0.64}
    else:
        matchup = {"vs_guards": 0.30, "vs_wings": 0.50, "vs_bigs": 0.20}
    return {
        "matchup_profile": matchup,
        "estimated_metrics": {"foul_rate": 3.1},
        "visibility": {"observability_score": round(clamp(trust_score, 20.0, 95.0), 1)},
    }


def load_college_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"College players file not found: {path}")
    if path.stat().st_size <= 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_web_college_payloads(input_path: Path, web_data_dir: Path) -> int:
    web_data_dir.mkdir(parents=True, exist_ok=True)
    cards_path = web_data_dir / "college_cards.json"
    valuations_path = web_data_dir / "college_valuations.json"

    df = load_college_dataframe(input_path)
    if df.empty:
        cards_path.write_text("[]\n", encoding="utf-8")
        valuations_path.write_text("[]\n", encoding="utf-8")
        print(f"No college rows available in {input_path}. Wrote empty web payloads.")
        return 0

    col_map = resolve_column_map(list(df.columns))
    valuator = PlayerValuator()
    cards: List[Dict[str, Any]] = []
    valuations: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        card = build_card_from_row(row=row, col_map=col_map)
        card.setdefault("identity", {})
        card["identity"]["usage_band"] = normalize_usage_band(card["identity"].get("usage_band", "low"))
        card["shot_profile"] = derive_shot_profile(card)
        card["creation_profile"] = derive_creation_profile(card)
        card["defense_assessment"] = derive_defense_assessment(card)
        card["comparables"] = {"similar_players": []}
        card["quality_flags"] = []

        trust_score = safe_float(card.get("v1_1_enhancements", {}).get("trust_assessment", {}).get("score"), 60.0)
        uncertainty = safe_float(card.get("uncertainty", {}).get("overall"), 0.4)
        games_played = safe_float(card.get("performance", {}).get("traditional", {}).get("games_played"), 0.0)
        plus_minus = safe_float(card.get("performance", {}).get("advanced", {}).get("plus_minus"), 0.0)
        card["value_metrics"] = {
            "epm": round(plus_minus, 3),
            "lebron": round((plus_minus * 0.9), 3),
            "player_value_score": 50.0,
            "source": "college_proxy",
            "trust_score": round(trust_score, 1),
            "uncertainty": round(uncertainty, 3),
            "games_played": int(games_played),
        }
        card["possession_decomposition"] = {
            "intrinsic_offense_est": round(clamp((plus_minus + 3.0) / 8.0, 0.0, 1.0), 3),
            "context_adjustment_est": round(clamp(uncertainty * 0.9, 0.0, 1.0), 3),
            "adjusted_value_est": round(clamp((plus_minus + 2.0) / 7.0, 0.0, 1.0), 3),
            "translation_confidence": round(clamp((trust_score / 100.0) * (1.0 - uncertainty), 0.0, 1.0), 3),
        }

        result = valuator.valuate_player(card)
        valuation_report = valuator.generate_report(result)
        valuation_report.setdefault("player", {})
        valuation_report["player"]["team"] = card.get("player", {}).get("team", "")

        cards.append(card)
        valuations.append(valuation_report)

    # Normalize player_value_score from valuation wins-added distribution for ranking stability.
    wins_vals = [safe_float(v.get("impact", {}).get("wins_added"), 0.0) for v in valuations]
    mean = sum(wins_vals) / max(1, len(wins_vals))
    variance = sum((x - mean) ** 2 for x in wins_vals) / max(1, len(wins_vals))
    std = math.sqrt(variance) if variance > 1e-9 else 1.0
    for card, wins in zip(cards, wins_vals):
        z = (wins - mean) / std
        score = clamp(50 + (15 * z), 0, 100)
        card.setdefault("value_metrics", {})
        card["value_metrics"]["player_value_score"] = round(score, 1)

    cards.sort(key=lambda x: (str(x.get("player", {}).get("name", "")).lower(), int(x.get("player", {}).get("season", 0))))
    valuations.sort(key=lambda x: (str(x.get("player", {}).get("name", "")).lower(), int(x.get("player", {}).get("season", 0))))

    with open(cards_path, "w", encoding="utf-8") as handle:
        json.dump(cards, handle, indent=2)
    with open(valuations_path, "w", encoding="utf-8") as handle:
        json.dump(valuations, handle, indent=2)

    print(f"[SUCCESS] Wrote {len(cards)} college cards -> {cards_path}")
    print(f"[SUCCESS] Wrote {len(valuations)} college valuations -> {valuations_path}")
    normalize_home_links(NBA_ROOT / "web")
    return 0


def main() -> int:
    input_path = REPO_ROOT / "data" / "processed" / "college" / "players_season.csv"
    web_data_dir = NBA_ROOT / "web" / "data"
    return build_web_college_payloads(input_path=input_path, web_data_dir=web_data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
