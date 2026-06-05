#!/usr/bin/env python3
"""
Scrape NBA player action/game images from NBA.com player pages.

NBA.com player pages embed action photos in their hero sections and
photo galleries. This script builds a mapping of player_id -> image URLs
for headshot, action, and alternate images.

Output: sports/nba/web/data/player_images.json
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

WORKSPACE = Path(__file__).resolve().parents[3]
CARDS_PATH = WORKSPACE / "sports" / "nba" / "web" / "data" / "cards.json"
OUTPUT_PATH = WORKSPACE / "sports" / "nba" / "web" / "data" / "player_images.json"

# NBA.com CDN patterns
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
# NBA.com stats player page sometimes has action images
NBA_PLAYER_PAGE = "https://www.nba.com/player/{player_id}"
# NBA CDN serves player profile images (action shots) at this pattern
NBA_PROFILE_ACTION = "https://cdn.nba.com/photos/actionshots/{player_id}_hi.png"


def get_unique_players() -> List[Dict[str, str]]:
    """Get unique players from cards.json."""
    if not CARDS_PATH.exists():
        return []
    data = json.loads(CARDS_PATH.read_text())
    seen = set()
    players = []
    for card in data:
        pid = card.get("player", {}).get("id", "")
        name = card.get("player", {}).get("name", "")
        if pid and pid not in seen:
            seen.add(pid)
            players.append({"id": pid, "name": name})
    return players


def build_image_set(player_id: str, player_name: str) -> Dict[str, Any]:
    """Build image URL set for a player.
    
    Uses multiple NBA CDN patterns that are known to work:
    - Standard headshot (always works)
    - Large headshot with different crop
    - Silhouette/alternate crop
    """
    # Standard headshot - guaranteed to work
    headshot = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    # Smaller crop - different framing
    headshot_sm = f"https://cdn.nba.com/headshots/nba/latest/520x380/{player_id}.png"
    
    return {
        "player_id": player_id,
        "player_name": player_name,
        "headshot": headshot,
        "headshot_alt": headshot_sm,
        "action": None,  # Will be populated if found
        "source": "nba_cdn",
    }


def try_fetch_action_image(player_id: str) -> Optional[str]:
    """Try to find an action image URL for a player.
    
    Checks NBA.com's known CDN paths for action/profile images.
    Returns URL if found (200 response), None otherwise.
    """
    # Try various known NBA CDN patterns
    candidates = [
        f"https://cdn.nba.com/photos/actionshots/{player_id}_hi.png",
        f"https://cdn.nba.com/photos/actionshots/{player_id}.png",
    ]
    
    for url in candidates:
        try:
            r = requests.head(url, timeout=3, allow_redirects=True)
            if r.status_code == 200:
                return url
        except Exception:
            pass
    
    return None


def scrape_nba_player_page_images(player_id: str) -> List[str]:
    """Scrape image URLs from a player's NBA.com page."""
    url = f"https://www.nba.com/player/{player_id}"
    try:
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if r.status_code != 200:
            return []
        
        # Find image URLs in the page source
        # NBA.com embeds player images in various formats
        img_pattern = re.compile(
            r'https://cdn\.nba\.com/[^"\'>\s]+(?:action|player|photo)[^"\'>\s]*\.(?:png|jpg|jpeg)',
            re.IGNORECASE
        )
        matches = img_pattern.findall(r.text)
        # Filter out tiny thumbnails and duplicates
        unique = list(set(m for m in matches if "1040" in m or "actionshot" in m or "photo" in m))
        return unique[:3]  # Max 3 images per player
    except Exception:
        return []


def build_all_player_images(scrape_pages: bool = False, max_scrape: int = 50) -> Dict[str, Any]:
    """Build complete player image mapping.
    
    Args:
        scrape_pages: Whether to scrape NBA.com player pages (slow, rate-limited)
        max_scrape: Max players to scrape pages for
    """
    players = get_unique_players()
    print(f"Building image set for {len(players)} players...")
    
    images = {}
    scraped = 0
    
    for i, player in enumerate(players):
        pid = player["id"]
        name = player["name"]
        
        img_set = build_image_set(pid, name)
        
        # Try action image CDN paths (fast, just HEAD requests)
        action_url = try_fetch_action_image(pid)
        if action_url:
            img_set["action"] = action_url
        
        # Optionally scrape player pages for more images
        if scrape_pages and scraped < max_scrape and not action_url:
            page_images = scrape_nba_player_page_images(pid)
            if page_images:
                img_set["action"] = page_images[0]
                if len(page_images) > 1:
                    img_set["gallery"] = page_images[1:]
            scraped += 1
            if scraped % 10 == 0:
                print(f"  Scraped {scraped}/{max_scrape} player pages...")
            time.sleep(0.5)  # Rate limit
        
        images[pid] = img_set
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(players)} players...")
    
    return images


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape-pages", action="store_true", 
                        help="Scrape NBA.com player pages for action images (slow)")
    parser.add_argument("--max-scrape", type=int, default=50,
                        help="Max player pages to scrape")
    args = parser.parse_args()
    
    images = build_all_player_images(
        scrape_pages=args.scrape_pages, 
        max_scrape=args.max_scrape
    )
    
    # Stats
    total = len(images)
    with_action = sum(1 for v in images.values() if v.get("action"))
    print(f"\nResults: {total} players, {with_action} with action images")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(images, indent=2), encoding="utf-8")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
