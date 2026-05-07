"""Deep-link resolver: scrapes sportsbooks and attaches links to picks.

This is the main entry point for the deep-link enrichment layer.
Call `enrich_picks_with_deeplinks()` after your final board is selected.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import draftkings, fanduel
from .adapters import bet365 as bet365_adapter
from .matcher import match_pick_to_selections, build_search_fallback
from .models import SportsbookSelection, BOOK_TITLES


# Storage paths
DEEPLINK_DATA_ROOT = Path(__file__).resolve().parents[2] / "shared" / "data" / "sportsbook_links"


def scrape_all_selections(
    sport: str,
    *,
    books: tuple[str, ...] = ("draftkings", "fanduel"),
) -> list[SportsbookSelection]:
    """Scrape player prop selections from all configured sportsbooks.

    Args:
        sport: "nba" or "mlb"
        books: Which sportsbooks to scrape

    Returns:
        Combined list of selections from all books
    """
    all_selections: list[SportsbookSelection] = []

    for book in books:
        try:
            if book == "draftkings":
                sels = draftkings.scrape_player_props(sport)
                all_selections.extend(sels)
            elif book == "fanduel":
                sels = fanduel.scrape_player_props(sport)
                all_selections.extend(sels)
            elif book == "bet365":
                sels = bet365_adapter.scrape_player_props(sport)
                all_selections.extend(sels)
        except Exception as e:
            print(f"  [warning] {book} scrape failed for {sport}: {e}")

    return all_selections


def save_selection_index(
    selections: list[SportsbookSelection],
    sport: str,
    run_date: str | None = None,
) -> Path | None:
    """Save scraped selections to a local JSON index file."""
    if not selections:
        return None

    date_str = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = DEEPLINK_DATA_ROOT / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sel in selections:
        rows.append({
            "sport": sel.sport,
            "book": sel.book,
            "game_date": sel.game_date,
            "event_name": sel.event_name,
            "home_team": sel.home_team,
            "away_team": sel.away_team,
            "player_name": sel.player_name,
            "normalized_player_name": sel.normalized_player_name,
            "market_key": sel.market_key,
            "market_name": sel.market_name,
            "side": sel.side,
            "line": sel.line,
            "price": sel.price,
            "book_event_id": sel.book_event_id,
            "book_market_id": sel.book_market_id,
            "book_selection_id": sel.book_selection_id,
            "deeplink_url": sel.deeplink_url,
            "deeplink_quality": sel.deeplink_quality,
            "scraped_at": sel.scraped_at,
        })

    out_path = out_dir / f"deeplink_index_{sport}_{date_str}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Also write a "latest" symlink-style file
    latest_path = out_dir / f"deeplink_index_{sport}_latest.json"
    latest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return out_path


def enrich_picks_with_deeplinks(
    picks: list[dict[str, Any]],
    sport: str,
    *,
    books: tuple[str, ...] = ("draftkings", "fanduel"),
    preferred_books: tuple[str, ...] = ("draftkings", "fanduel"),
    save_index: bool = True,
    run_date: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enrich final prediction picks with sportsbook deep links.

    This is the main function to call after your final board is selected.
    It scrapes sportsbook APIs, matches picks to selections, and attaches
    deep links to each pick.

    Args:
        picks: List of prediction pick dicts
        sport: "nba" or "mlb"
        books: Which sportsbooks to scrape
        preferred_books: Priority order for book selection
        save_index: Whether to save the scraped index locally
        run_date: Optional run date for file naming

    Returns:
        (enriched_picks, summary) tuple
    """
    summary: dict[str, Any] = {
        "sport": sport,
        "books_scraped": list(books),
        "total_picks": len(picks),
        "betslip_links": 0,
        "event_links": 0,
        "search_fallbacks": 0,
        "no_link": 0,
        "selections_scraped": 0,
        "by_book": {},
    }

    # Step 1: Scrape sportsbook selections
    print(f"  Scraping sportsbook props ({', '.join(books)})...")
    selections = scrape_all_selections(sport, books=books)
    summary["selections_scraped"] = len(selections)

    by_book_count: dict[str, int] = {}
    for sel in selections:
        by_book_count[sel.book] = by_book_count.get(sel.book, 0) + 1
    for book, count in by_book_count.items():
        print(f"    {BOOK_TITLES.get(book, book)}: {count} selections")

    # Step 2: Save index
    if save_index and selections:
        idx_path = save_selection_index(selections, sport, run_date)
        if idx_path:
            summary["index_path"] = str(idx_path)

    # Step 3: Match picks to selections
    enriched = []
    for pick in picks:
        pick_copy = dict(pick)

        result = match_pick_to_selections(
            pick_copy, selections, sport, preferred_books=preferred_books,
        )

        if result and result.betslip_link:
            pick_copy["betslip_link"] = result.betslip_link
            pick_copy["deeplink_quality"] = result.deeplink_quality
            pick_copy["bookmaker"] = result.bookmaker
            pick_copy["bookmaker_title"] = result.bookmaker_title
            if result.odds_american is not None:
                pick_copy["odds_american"] = result.odds_american
            pick_copy["book_event_id"] = result.book_event_id
            pick_copy["book_market_id"] = result.book_market_id
            pick_copy["book_selection_id"] = result.book_selection_id

            if result.deeplink_quality == "betslip":
                summary["betslip_links"] += 1
            elif result.deeplink_quality == "event":
                summary["event_links"] += 1

            summary["by_book"][result.bookmaker] = summary["by_book"].get(result.bookmaker, 0) + 1
        else:
            # Fallback to FanDuel search URL
            player_name = pick_copy.get("player_display_name") or pick_copy.get("player", "")
            fallback = build_search_fallback(player_name, sport)
            pick_copy["betslip_link"] = fallback.betslip_link
            pick_copy["deeplink_quality"] = fallback.deeplink_quality
            pick_copy["bookmaker"] = fallback.bookmaker
            pick_copy["bookmaker_title"] = fallback.bookmaker_title
            summary["search_fallbacks"] += 1

        # Always attach alternative sportsbook links so users can choose
        player_name = pick_copy.get("player_display_name") or pick_copy.get("player", "")
        pick_copy["sportsbook_options"] = _build_sportsbook_options(
            pick_copy, player_name, sport, selections, preferred_books,
        )

        enriched.append(pick_copy)

    total_deep = summary["betslip_links"] + summary["event_links"]
    print(f"  Deep links: {total_deep}/{len(picks)} picks ({summary['search_fallbacks']} search fallbacks)")

    return enriched, summary


def _build_sportsbook_options(
    pick: dict[str, Any],
    player_name: str,
    sport: str,
    selections: list[SportsbookSelection],
    preferred_books: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build a list of sportsbook link options for a pick.

    Each option represents a different sportsbook the user can bet on.
    Options are ordered by link quality (betslip > search > event).
    """
    from .matcher import build_bet365_fallback

    options: list[dict[str, Any]] = []
    seen_books: set[str] = set()

    # Primary link (already attached to pick)
    primary_book = pick.get("bookmaker", "")
    if primary_book:
        seen_books.add(primary_book)
        options.append({
            "bookmaker": primary_book,
            "bookmaker_title": BOOK_TITLES.get(primary_book, primary_book.title()),
            "betslip_link": pick.get("betslip_link", ""),
            "deeplink_quality": pick.get("deeplink_quality", "none"),
        })

    # Try other books from selections
    for book in preferred_books:
        if book in seen_books:
            continue
        result = match_pick_to_selections(
            pick, selections, sport, preferred_books=(book,),
        )
        if result and result.betslip_link and result.bookmaker not in seen_books:
            seen_books.add(result.bookmaker)
            options.append({
                "bookmaker": result.bookmaker,
                "bookmaker_title": result.bookmaker_title,
                "betslip_link": result.betslip_link,
                "deeplink_quality": result.deeplink_quality,
            })

    # Add FanDuel search if not already present
    if "fanduel" not in seen_books:
        fallback = build_search_fallback(player_name, sport)
        options.append({
            "bookmaker": "fanduel",
            "bookmaker_title": "FanDuel",
            "betslip_link": fallback.betslip_link,
            "deeplink_quality": "search",
        })

    # Add bet365 event link
    if "bet365" not in seen_books:
        b365 = build_bet365_fallback(player_name, sport)
        options.append({
            "bookmaker": "bet365",
            "bookmaker_title": "bet365",
            "betslip_link": b365.betslip_link,
            "deeplink_quality": b365.deeplink_quality,
        })

    return options
