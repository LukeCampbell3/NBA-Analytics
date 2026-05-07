"""Match final prediction picks to scraped sportsbook selections."""
from __future__ import annotations

from typing import Any

from .models import DeeplinkResult, SportsbookSelection, BOOK_TITLES, BOOK_HOMEPAGES
from .normalize import normalize_player_name, normalize_side, target_to_market_key


def _name_match_score(pick_name: str, selection_name: str) -> int:
    """Score how well two normalized player names match. 0 = no match."""
    if not pick_name or not selection_name:
        return 0
    if pick_name == selection_name:
        return 100
    # One contains the other (handles "CJ McCollum" vs "C.J. McCollum")
    if pick_name in selection_name or selection_name in pick_name:
        return 85
    # Last name match
    pick_parts = pick_name.split()
    sel_parts = selection_name.split()
    if pick_parts and sel_parts and pick_parts[-1] == sel_parts[-1]:
        # Same last name, check first initial
        if pick_parts[0][0] == sel_parts[0][0]:
            return 75
    return 0


def match_pick_to_selections(
    pick: dict[str, Any],
    selections: list[SportsbookSelection],
    sport: str,
    *,
    preferred_books: tuple[str, ...] = ("fanduel", "draftkings"),
) -> DeeplinkResult | None:
    """Find the best sportsbook selection match for a prediction pick.

    Args:
        pick: A prediction pick dict with player, target, direction, market_line
        selections: List of scraped sportsbook selections
        sport: "nba" or "mlb"
        preferred_books: Book priority order

    Returns:
        DeeplinkResult if matched, None otherwise
    """
    pick_player = normalize_player_name(
        pick.get("player_display_name") or pick.get("player") or ""
    )
    pick_target = str(pick.get("target", "")).upper()
    pick_side = normalize_side(pick.get("direction", ""))
    pick_line = pick.get("market_line")
    if pick_line is not None:
        pick_line = float(pick_line)

    market_key = target_to_market_key(pick_target, sport)

    if not pick_player or not market_key or not pick_side:
        return None

    # Score all candidates
    candidates: list[tuple[int, int, SportsbookSelection]] = []

    for sel in selections:
        # Must match market key
        if sel.market_key != market_key:
            continue

        # Must match side
        if sel.side != pick_side:
            continue

        # Name match
        name_score = _name_match_score(pick_player, sel.normalized_player_name)
        if name_score < 75:
            continue

        # Line match (exact or close)
        line_score = 0
        if pick_line is not None and sel.line is not None:
            if abs(pick_line - sel.line) < 0.01:
                line_score = 100  # Exact match
            elif abs(pick_line - sel.line) <= 0.5:
                line_score = 50  # Close (e.g., 1.5 vs 2.0)
            else:
                continue  # Line too far off
        elif pick_line is None or sel.line is None:
            line_score = 30  # Can't verify line, partial credit

        # Book priority
        book_priority = 0
        try:
            book_priority = len(preferred_books) - preferred_books.index(sel.book)
        except ValueError:
            pass

        total_score = name_score + line_score + book_priority * 10
        candidates.append((total_score, book_priority, sel))

    if not candidates:
        return None

    # Pick the best match
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, _, best = candidates[0]

    return DeeplinkResult(
        betslip_link=best.deeplink_url,
        deeplink_quality=best.deeplink_quality,
        bookmaker=best.book,
        bookmaker_title=BOOK_TITLES.get(best.book, best.book.title()),
        odds_american=best.price,
        book_event_id=best.book_event_id,
        book_market_id=best.book_market_id,
        book_selection_id=best.book_selection_id,
    )


def build_search_fallback(player_name: str, sport: str = "mlb") -> DeeplinkResult:
    """Build a FanDuel search URL fallback when no deep link is available."""
    clean = str(player_name or "").replace("_", " ").strip()
    if not clean:
        nav = "nba" if sport.lower() == "nba" else "mlb"
        url = f"https://sportsbook.fanduel.com/navigation/{nav}"
    else:
        encoded = urllib.parse.quote_plus(clean)
        url = f"https://sportsbook.fanduel.com/search?q={encoded}&tab=player-props"

    return DeeplinkResult(
        betslip_link=url,
        deeplink_quality="search",
        bookmaker="fanduel",
        bookmaker_title="FanDuel",
    )


def build_bet365_fallback(player_name: str, sport: str = "mlb") -> DeeplinkResult:
    """Build a bet365 event-level link as an alternative sportsbook option.

    Bet365 doesn't support betslip deep links or search URLs, so this
    links to the sport hub page. Quality is 'event' (not 'betslip').
    """
    from .adapters.bet365 import build_event_link

    url = build_event_link(sport)
    return DeeplinkResult(
        betslip_link=url,
        deeplink_quality="event",
        bookmaker="bet365",
        bookmaker_title="bet365",
    )


# Need urllib.parse for the fallback
import urllib.parse
