"""Data models for sportsbook deep-link system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SportsbookSelection:
    """A single sportsbook outcome that can be linked to a betslip."""

    sport: str
    book: str
    game_date: str
    event_name: str

    home_team: str
    away_team: str

    player_name: str
    normalized_player_name: str
    market_key: str  # canonical: player_points, batter_total_bases, etc.
    market_name: str  # raw from sportsbook
    side: str  # OVER / UNDER
    line: Optional[float]
    price: Optional[int]  # American odds

    # Sportsbook-native IDs
    book_event_id: Optional[str] = None
    book_market_id: Optional[str] = None
    book_selection_id: Optional[str] = None
    book_option_id: Optional[str] = None
    book_price_id: Optional[str] = None
    book_event_url: Optional[str] = None

    # Resolved link
    deeplink_url: Optional[str] = None
    deeplink_quality: str = "none"  # betslip | event | search | book_home | none

    scraped_at: Optional[str] = None


@dataclass
class DeeplinkResult:
    """Result of matching a pick to a sportsbook selection."""

    betslip_link: Optional[str] = None
    deeplink_quality: str = "none"
    bookmaker: str = ""
    bookmaker_title: str = ""
    odds_american: Optional[int] = None
    book_event_id: Optional[str] = None
    book_market_id: Optional[str] = None
    book_selection_id: Optional[str] = None


BOOK_TITLES = {
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "bet365": "bet365",
}

BOOK_HOMEPAGES = {
    "fanduel": "https://sportsbook.fanduel.com/",
    "draftkings": "https://sportsbook.draftkings.com/",
    "betmgm": "https://sports.betmgm.com/",
    "caesars": "https://www.caesars.com/sportsbook-and-casino",
    "bet365": "https://www.bet365.com/",
}
