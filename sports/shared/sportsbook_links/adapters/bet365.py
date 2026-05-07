"""Bet365 sportsbook adapter.

Bet365 does NOT expose a public API for player props or stable betslip
deep links. Unlike DraftKings (public sportscontent API with selection IDs)
or FanDuel (sbapi with marketId/selectionId), bet365 uses a single-page
application with hash-based routing that doesn't support external betslip
prefilling.

This adapter provides event-level links that navigate users to the correct
sport/game page on bet365, where they can find and add the prop to their
slip manually.

Bet365 US availability: NJ, CO, VA, OH, IA, KY, LA, IN, AZ, NC only.

Link format:
    https://www.bet365.com/#/AC/B18/C20604387/D48/{sport_path}/{event_path}/

Since bet365 doesn't expose stable event IDs publicly, we construct
search-style navigation links that land users on the correct sport page.
"""
from __future__ import annotations

import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any

from ..models import SportsbookSelection
from ..normalize import normalize_player_name, normalize_side


# Bet365 sport navigation paths
BET365_BASE = "https://www.bet365.com"

BET365_SPORT_PATHS = {
    "nba": "#/AC/B18/C20604387/D48/E1/F2/",  # Basketball > NBA
    "mlb": "#/AC/B17/C20604387/D48/E2/F2/",  # Baseball > MLB
}

# Bet365 hub pages (public, no login required)
BET365_HUB_URLS = {
    "nba": "https://www.bet365.com/hub/en-us/basketball/nba",
    "mlb": "https://www.bet365.com/hub/en-us/baseball/mlb",
}


def build_event_link(sport: str) -> str:
    """Build a bet365 link to the sport's main page.

    Since bet365 doesn't support stable external deep links to specific
    selections, we link to the sport hub page where users can navigate
    to their desired game and prop.
    """
    canonical = sport.lower().strip()
    return BET365_HUB_URLS.get(canonical, f"{BET365_BASE}/#/AC/B18/C20604387/")


def build_player_search_link(player_name: str, sport: str = "nba") -> str:
    """Build a bet365 link that's as close to the player's props as possible.

    Bet365 doesn't have a public search URL that accepts query parameters
    for player props. The best we can do is link to the sport page.
    """
    # Bet365 doesn't support search URLs like FanDuel does.
    # Return the sport hub page as the best available option.
    return build_event_link(sport)


def scrape_player_props(sport: str) -> list[SportsbookSelection]:
    """Bet365 does not expose a public player props API.

    This function returns an empty list. Bet365 selections cannot be
    scraped without browser automation (against ToS) or a paid partner
    integration.

    The adapter exists so that:
    1. The system can generate bet365 event-level links as fallbacks
    2. If bet365 ever exposes a public API, this adapter is ready
    3. The resolver can offer bet365 as a user-selectable book option
    """
    # No public API available for bet365 player props.
    # Unlike DraftKings (sportscontent API) or FanDuel (sbapi),
    # bet365 does not expose market/selection IDs publicly.
    return []
