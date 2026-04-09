#!/usr/bin/env python3
"""
prepare_web_data.py - Prepare data for web frontend

This script:
1. Copies player card JSON files from data/processed/player_cards to web/data
2. Generates valuation JSON files for each player
3. Creates combined cards.json and valuations.json for the web app
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import sys
import re
import unicodedata
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from value_players import PlayerValuator


def normalize_home_links(web_dir: Path) -> None:
    """
    Normalize Home nav anchors so they always point to site root (/).
    """
    if not web_dir.exists():
        return

    html_files = sorted(web_dir.glob("*.html"))
    # href="index.html" ...>Home</a>  -> href="/" ...>Home</a>
    pattern = re.compile(r'(<a\b[^>]*\bhref=["\'])index\.html(["\'][^>]*>\s*Home\s*</a>)', re.IGNORECASE)

    for html_path in html_files:
        try:
            original = html_path.read_text(encoding="utf-8")
        except OSError:
            continue
        updated = pattern.sub(r'\1/\2', original)
        if updated != original:
            html_path.write_text(updated, encoding="utf-8")


def normalize_name(value: str) -> str:
    """Normalize names for fuzzy cross-source matching."""
    text = unicodedata.normalize('NFKD', str(value or ''))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_lebron_lookup(raw_dir: Path, season: int) -> Dict[str, Dict[str, float]]:
    """Load LEBRON values by player id and name/team keys."""
    path = raw_dir / f"lebron-data-{season}.csv"
    if not path.exists():
        return {"by_id": {}, "by_name_team": {}, "by_name": {}}

    df = pd.read_csv(path)
    if df.empty:
        return {"by_id": {}, "by_name_team": {}, "by_name": {}}

    name_col = "Player" if "Player" in df.columns else "PLAYER_NAME"
    team_col = "Team" if "Team" in df.columns else "TEAM_ABBREVIATION"
    id_col = "nba_id" if "nba_id" in df.columns else ("PLAYER_ID" if "PLAYER_ID" in df.columns else None)
    lebron_col = "LEBRON" if "LEBRON" in df.columns else None
    if lebron_col is None:
        return {"by_id": {}, "by_name_team": {}, "by_name": {}}

    by_id: Dict[str, float] = {}
    by_name_team: Dict[str, float] = {}
    by_name: Dict[str, float] = {}

    for _, row in df.iterrows():
        value = row.get(lebron_col, None)
        if pd.isna(value):
            continue
        lebron = float(value)

        if id_col:
            pid = str(row.get(id_col, "")).strip()
            if pid and pid.lower() != "nan":
                by_id[pid] = lebron

        name = normalize_name(row.get(name_col, ""))
        team = str(row.get(team_col, "")).strip().upper()
        if name:
            by_name[name] = lebron
            if team:
                by_name_team[f"{name}|{team}"] = lebron

    return {"by_id": by_id, "by_name_team": by_name_team, "by_name": by_name}


def load_epm_lookup(raw_dir: Path, season: int) -> Dict[str, Dict[str, float]]:
    """
    Load EPM values when an EPM file exists.
    Expected filename pattern: *epm*{season}*.csv
    """
    candidates = list(raw_dir.glob(f"*epm*{season}*.csv"))
    if not candidates:
        return {"by_id": {}, "by_name_team": {}, "by_name": {}}

    path = candidates[0]
    df = pd.read_csv(path)
    if df.empty:
        return {"by_id": {}, "by_name_team": {}, "by_name": {}}

    id_col = next((c for c in df.columns if c.lower() in {"player_id", "nba_id", "id"}), None)
    name_col = next((c for c in df.columns if "player" in c.lower() and "name" in c.lower()), None)
    team_col = next((c for c in df.columns if "team" in c.lower() and "id" not in c.lower()), None)
    epm_col = next((c for c in df.columns if c.lower() == "epm" or c.lower().endswith("_epm")), None)
    if epm_col is None:
        return {"by_id": {}, "by_name_team": {}, "by_name": {}}

    by_id: Dict[str, float] = {}
    by_name_team: Dict[str, float] = {}
    by_name: Dict[str, float] = {}

    for _, row in df.iterrows():
        value = row.get(epm_col, None)
        if pd.isna(value):
            continue
        epm = float(value)

        if id_col:
            pid = str(row.get(id_col, "")).strip()
            if pid and pid.lower() != "nan":
                by_id[pid] = epm

        if name_col:
            name = normalize_name(row.get(name_col, ""))
            team = str(row.get(team_col, "")).strip().upper() if team_col else ""
            if name:
                by_name[name] = epm
                if team:
                    by_name_team[f"{name}|{team}"] = epm

    return {"by_id": by_id, "by_name_team": by_name_team, "by_name": by_name}


def lookup_metric(card: Dict[str, Any], lookup: Dict[str, Dict[str, float]]) -> Any:
    """Lookup a metric value by id, then name+team, then name."""
    player = card.get("player", {})
    pid = str(player.get("id", "")).strip()
    team = str(player.get("team", "")).strip().upper()
    name = normalize_name(player.get("name", ""))

    if pid and pid in lookup["by_id"]:
        return lookup["by_id"][pid]
    key = f"{name}|{team}" if name and team else ""
    if key and key in lookup["by_name_team"]:
        return lookup["by_name_team"][key]
    if name and name in lookup["by_name"]:
        return lookup["by_name"][name]
    return None


def prepare_web_data():
    """Prepare all data for web frontend"""
    
    # Paths - use full data directory
    source_dir = Path('data/processed/player_cards')
    web_data_dir = Path('web/data')
    raw_dir = Path('data/raw')
    
    # Create web data directory
    web_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all player card files
    card_files = list(source_dir.glob('*_final.json'))
    
    if not card_files:
        print(f"Error: No player card files found in {source_dir}")
        return 1
    
    print(f"Found {len(card_files)} player cards")
    
    # Initialize valuator
    valuator = PlayerValuator()
    lebron_cache: Dict[int, Dict[str, Dict[str, float]]] = {}
    epm_cache: Dict[int, Dict[str, Dict[str, float]]] = {}
    
    # Process each player
    all_cards = []
    all_valuations = []
    
    for i, card_file in enumerate(card_files, 1):
        try:
            # Load card
            with open(card_file, 'r', encoding='utf-8') as f:
                card = json.load(f)

            season = int(card.get("player", {}).get("season", 2025))
            if season not in lebron_cache:
                lebron_cache[season] = load_lebron_lookup(raw_dir, season)
            if season not in epm_cache:
                epm_cache[season] = load_epm_lookup(raw_dir, season)

            lebron_val = lookup_metric(card, lebron_cache[season])
            epm_val = lookup_metric(card, epm_cache[season])
            value_metrics = card.setdefault("value_metrics", {})
            if lebron_val is not None:
                value_metrics["lebron"] = round(float(lebron_val), 3)
            if epm_val is not None:
                value_metrics["epm"] = round(float(epm_val), 3)
            
            all_cards.append(card)
            
            # Generate valuation
            result = valuator.valuate_player(card)
            valuation = valuator.generate_report(result)
            all_valuations.append(valuation)
            
            if i % 50 == 0:
                print(f"  Processed {i}/{len(card_files)} players...")
            
        except Exception as e:
            print(f"  Error processing {card_file.name}: {e}")
            continue
    
    # Save combined files
    cards_path = web_data_dir / 'cards.json'
    with open(cards_path, 'w', encoding='utf-8') as f:
        json.dump(all_cards, f, indent=2)
    
    valuations_path = web_data_dir / 'valuations.json'
    with open(valuations_path, 'w', encoding='utf-8') as f:
        json.dump(all_valuations, f, indent=2)
    
    print(f"\n[SUCCESS] Prepared web data:")
    print(f"  - {len(all_cards)} player cards -> {cards_path}")
    print(f"  - {len(all_valuations)} valuations -> {valuations_path}")
    normalize_home_links(Path("web"))
    print(f"\nTo build the static site bundle:")
    print(f"  python build_static_site.py")
    
    return 0


if __name__ == "__main__":
    exit(prepare_web_data())
