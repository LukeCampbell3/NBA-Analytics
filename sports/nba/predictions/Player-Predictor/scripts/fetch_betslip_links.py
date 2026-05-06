#!/usr/bin/env python3
"""Fetch FanDuel bet slip deep links from The Odds API.

Uses the includeLinks=true parameter to get direct "add to bet slip" URLs
for each player prop outcome. These links pre-populate the user's FanDuel
bet slip with the exact selection.

Usage:
    python fetch_betslip_links.py --sport basketball_nba --markets player_points,player_rebounds,player_assists
    python fetch_betslip_links.py --sport baseball_mlb --markets batter_hits,batter_total_bases,batter_runs_scored,pitcher_strikeouts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]

API_BASE = "https://api.the-odds-api.com/v4/sports"

# Map our internal target names to Odds API market keys
NBA_MARKET_MAP = {
    "PTS": "player_points",
    "TRB": "player_rebounds",
    "AST": "player_assists",
}

MLB_MARKET_MAP = {
    "H": "batter_hits",
    "TB": "batter_total_bases",
    "R": "batter_runs_scored",
    "HR": "batter_home_runs",
    "RBI": "batter_rbis",
    "K": "pitcher_strikeouts",
}


def resolve_api_key() -> str:
    """Resolve the Odds API key from env or config."""
    for key in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
        val = os.getenv(key)
        if val:
            return val

    # Try config.local.yaml
    config_path = REPO_ROOT / "config.local.yaml"
    if config_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            odds = cfg.get("odds_api", {})
            if isinstance(odds, dict) and odds.get("api_key"):
                return odds["api_key"]
        except Exception:
            pass

    raise RuntimeError("No Odds API key found. Set ODDS_API_KEY env var or add to config.local.yaml")


def fetch_player_prop_links(
    sport: str,
    markets: list[str],
    api_key: str,
    regions: str = "us",
) -> list[dict]:
    """Fetch player prop odds with bet slip links from The Odds API.

    Returns a list of dicts with: player, market, line, direction, odds, link, bookmaker
    """
    url = f"{API_BASE}/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "includeLinks": "true",
        "includeSids": "true",
        "bookmakers": "fanduel",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    events = response.json()

    results = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        event_id = event.get("id", "")

        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", outcome.get("name", ""))
                    direction = outcome.get("name", "")  # "Over" or "Under"
                    price = outcome.get("price")
                    point = outcome.get("point")
                    link = outcome.get("link")

                    if player and point is not None and link:
                        results.append({
                            "player": player,
                            "market_key": market_key,
                            "direction": direction.upper() if direction else "",
                            "line": float(point),
                            "odds_american": int(price) if price else -110,
                            "link": link,
                            "bookmaker": book_key,
                            "home_team": home,
                            "away_team": away,
                            "event_id": event_id,
                        })

    return results


def match_picks_to_links(
    picks: list[dict],
    links: list[dict],
    sport: str = "nba",
) -> list[dict]:
    """Match our prediction picks to the fetched bet slip links.

    Returns picks with added 'betslip_link' and 'betslip_odds' fields.
    """
    # Build lookup: (player_lower, market_key, direction, line) -> link
    link_lookup = {}
    for link_item in links:
        player_key = link_item["player"].lower().strip()
        market = link_item["market_key"]
        direction = link_item["direction"]
        line = link_item["line"]
        key = (player_key, market, direction, line)
        link_lookup[key] = link_item

    market_map = NBA_MARKET_MAP if sport == "nba" else MLB_MARKET_MAP

    for pick in picks:
        player = str(pick.get("player_display_name", pick.get("player", ""))).lower().strip()
        target = str(pick.get("target", "")).upper()
        direction = str(pick.get("direction", "")).upper()
        line = float(pick.get("market_line", 0) or 0)

        market_key = market_map.get(target, "")
        if not market_key:
            continue

        # Try exact match
        key = (player, market_key, direction, line)
        match = link_lookup.get(key)

        # Try fuzzy player match
        if not match:
            for lk, lv in link_lookup.items():
                if (lk[1] == market_key and lk[2] == direction and lk[3] == line
                        and (player in lk[0] or lk[0] in player)):
                    match = lv
                    break

        if match:
            pick["betslip_link"] = match["link"]
            pick["betslip_odds"] = match["odds_american"]
            pick["betslip_bookmaker"] = match["bookmaker"]

    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="basketball_nba", choices=["basketball_nba", "baseball_mlb"])
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    api_key = resolve_api_key()

    if args.sport == "basketball_nba":
        markets = list(NBA_MARKET_MAP.values())
    else:
        markets = list(MLB_MARKET_MAP.values())

    print(f"Fetching {args.sport} player prop links...")
    links = fetch_player_prop_links(args.sport, markets, api_key)
    print(f"  Found {len(links)} bet slip links")

    out_path = args.out_json or Path(f"betslip_links_{args.sport}.json")
    out_path.write_text(json.dumps(links, indent=2), encoding="utf-8")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
