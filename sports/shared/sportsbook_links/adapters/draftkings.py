"""DraftKings sportsbook adapter.

Scrapes DraftKings' public sportscontent API for player prop markets
and selection IDs. These IDs are used to construct betslip deep links.

DK betslip URL format:
    https://sportsbook.draftkings.com/?outcomes={selectionId}

The API is public (no auth required) but unofficial — DraftKings does
not guarantee stability. The adapter handles failures gracefully.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..models import SportsbookSelection
from ..normalize import normalize_player_name, normalize_side

# DraftKings sportscontent API base
DK_API_BASE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1"

# League IDs
DK_LEAGUES = {
    "nba": "42648",
    "mlb": "84240",
}

# Category/subcategory IDs for player props
# Format: {sport: [(category_id, subcategory_id, market_key), ...]}
DK_PROP_ENDPOINTS = {
    "nba": [
        ("1215", "12488", "player_points"),       # Points O/U
        ("1216", "12492", "player_rebounds"),      # Rebounds O/U
        ("1217", "12495", "player_assists"),       # Assists O/U
        ("1218", "12497", "player_threes"),        # Threes O/U
    ],
    "mlb": [
        ("743", "6607", "batter_total_bases"),    # Total Bases O/U
        ("743", "6606", "batter_hits"),           # Hits O/U
        ("743", "6604", "batter_runs_scored"),    # Runs O/U
        ("743", "6605", "batter_rbis"),           # RBIs O/U
        ("743", "6608", "batter_home_runs"),      # Home Runs O/U
        ("740", "6574", "pitcher_strikeouts"),    # Strikeouts O/U
    ],
}

DK_TIMEOUT = 15.0
DK_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _request_dk(url: str) -> dict[str, Any]:
    """Make a request to the DraftKings API."""
    req = urllib.request.Request(url, headers={
        "User-Agent": DK_USER_AGENT,
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=DK_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_betslip_url(selection_id: str) -> str:
    """Construct a DraftKings betslip deep link from a selection ID."""
    encoded = urllib.parse.quote(selection_id, safe="")
    return f"https://sportsbook.draftkings.com/?outcomes={encoded}"


def scrape_player_props(sport: str) -> list[SportsbookSelection]:
    """Scrape all available player prop selections from DraftKings.

    Returns a list of SportsbookSelection objects with resolved deep links.
    """
    canonical = sport.lower().strip()
    league_id = DK_LEAGUES.get(canonical)
    endpoints = DK_PROP_ENDPOINTS.get(canonical, [])

    if not league_id or not endpoints:
        return []

    all_selections: list[SportsbookSelection] = []
    now_utc = datetime.now(timezone.utc).isoformat()

    for category_id, subcategory_id, market_key in endpoints:
        url = f"{DK_API_BASE}/leagues/{league_id}/categories/{category_id}/subcategories/{subcategory_id}"
        try:
            data = _request_dk(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            continue

        events_by_id = {str(e["id"]): e for e in data.get("events", [])}
        markets_by_id = {str(m["id"]): m for m in data.get("markets", [])}

        for sel in data.get("selections", []):
            selection_id = str(sel.get("id", ""))
            market_id = str(sel.get("marketId", ""))
            market = markets_by_id.get(market_id, {})
            event_id = str(market.get("eventId", ""))
            event = events_by_id.get(event_id, {})

            # Extract player name from market name or participants
            market_name = str(market.get("name", ""))
            participants = sel.get("participants", [])
            player_name = ""
            if participants:
                player_name = str(participants[0].get("name", ""))
            if not player_name:
                # Parse from market name: "Player Name Total Bases O/U"
                player_name = market_name.rsplit(" O/U", 1)[0].rsplit(" Over/Under", 1)[0].strip()

            # Extract side (Over/Under)
            label = str(sel.get("label", "")).strip()
            side = normalize_side(label)

            # Extract line
            line = sel.get("points")
            if line is not None:
                line = float(line)

            # Extract odds
            odds_data = sel.get("displayOdds", {})
            american_odds = odds_data.get("american", "")
            price = None
            if american_odds:
                try:
                    price = int(american_odds.replace("+", ""))
                except (ValueError, TypeError):
                    pass

            # Extract event info
            event_name = str(event.get("name", ""))
            # Parse home/away from event name "Away @ Home"
            home_team = ""
            away_team = ""
            if " @ " in event_name:
                parts = event_name.split(" @ ", 1)
                away_team = parts[0].strip()
                home_team = parts[1].strip()

            if not selection_id or not player_name or side not in ("OVER", "UNDER"):
                continue

            deeplink_url = build_betslip_url(selection_id)

            all_selections.append(SportsbookSelection(
                sport=canonical,
                book="draftkings",
                game_date=str(event.get("startEventDate", ""))[:10],
                event_name=event_name,
                home_team=home_team,
                away_team=away_team,
                player_name=player_name,
                normalized_player_name=normalize_player_name(player_name),
                market_key=market_key,
                market_name=market_name,
                side=side,
                line=line,
                price=price,
                book_event_id=event_id,
                book_market_id=market_id,
                book_selection_id=selection_id,
                deeplink_url=deeplink_url,
                deeplink_quality="betslip",
                scraped_at=now_utc,
            ))

    return all_selections
