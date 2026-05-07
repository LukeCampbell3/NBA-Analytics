"""FanDuel sportsbook adapter.

Scrapes FanDuel's public sbapi for player prop markets with
marketId and selectionId. These are used to construct betslip links.

FanDuel betslip URL format:
    https://sportsbook.fanduel.com/addToBetslip?marketId={marketId}&selectionId={selectionId}

The API is public (no auth) but unofficial. FanDuel's prop structure
uses threshold markets ("To Record 2+ Total Bases") rather than
Over/Under lines, so matching requires mapping thresholds to lines.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..models import SportsbookSelection
from ..normalize import normalize_player_name, normalize_side

# FanDuel sbapi base (NJ endpoint, works for public data)
FD_API_BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
FD_API_KEY = "FhMFpcPWXMeyZxOx"

# Event type IDs
FD_EVENT_TYPES = {
    "nba": "6",
    "mlb": "7511",
}

# Tab names for player props
FD_PROP_TABS = {
    "nba": ["player-props"],
    "mlb": ["batter-props", "pitcher-props"],
}

FD_TIMEOUT = 15.0
FD_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Market name patterns to canonical market keys
MLB_MARKET_PATTERNS = {
    r"Total Bases": "batter_total_bases",
    r"Hits": "batter_hits",
    r"Runs Scored|(?<!Home )Runs(?! Line)": "batter_runs_scored",
    r"RBIs": "batter_rbis",
    r"Home Run": "batter_home_runs",
    r"Strikeouts": "pitcher_strikeouts",
}

NBA_MARKET_PATTERNS = {
    r"Points": "player_points",
    r"Rebounds": "player_rebounds",
    r"Assists": "player_assists",
    r"Three": "player_threes",
}


def _request_fd(url: str) -> dict[str, Any]:
    """Make a request to the FanDuel API."""
    req = urllib.request.Request(url, headers={
        "User-Agent": FD_USER_AGENT,
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=FD_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_betslip_url(market_id: str, selection_id: str) -> str:
    """Construct a FanDuel betslip deep link."""
    params = urllib.parse.urlencode({
        "marketId": market_id,
        "selectionId": selection_id,
    })
    return f"https://sportsbook.fanduel.com/addToBetslip?{params}"


def _classify_market(market_name: str, sport: str) -> tuple[str, str, float | None]:
    """Classify a FanDuel market name into (market_key, side, line).

    FanDuel uses threshold markets like "To Record 2+ Total Bases" which
    maps to OVER 1.5 (the player needs 2+ to clear, so the line is 1.5).

    Returns (market_key, side, line) or ("", "", None) if unrecognized.
    """
    patterns = MLB_MARKET_PATTERNS if sport == "mlb" else NBA_MARKET_PATTERNS
    market_key = ""
    for pattern, key in patterns.items():
        if re.search(pattern, market_name, re.IGNORECASE):
            market_key = key
            break

    if not market_key:
        return "", "", None

    # Parse threshold: "To Record 2+ Total Bases" -> OVER 1.5
    # "Player Over/Under X.5 Points" -> depends on runner label
    threshold_match = re.search(r"(\d+)\+", market_name)
    if threshold_match:
        threshold = int(threshold_match.group(1))
        line = threshold - 0.5  # "2+" means line is 1.5
        return market_key, "OVER", line

    # Over/Under format: "Player Points Over/Under"
    if "Over" in market_name or "Under" in market_name or "O/U" in market_name:
        # Side determined by runner label, line from handicap
        return market_key, "", None

    return market_key, "", None


def _get_events(sport: str) -> dict[str, dict[str, Any]]:
    """Get current events for a sport."""
    event_type_id = FD_EVENT_TYPES.get(sport)
    if not event_type_id:
        return {}

    url = f"{FD_API_BASE}/content-managed-page?page=SPORT&eventTypeId={event_type_id}&_ak={FD_API_KEY}"
    try:
        data = _request_fd(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return {}

    return data.get("attachments", {}).get("events", {})


def scrape_player_props(sport: str) -> list[SportsbookSelection]:
    """Scrape player prop selections from FanDuel.

    Returns a list of SportsbookSelection objects with resolved deep links.
    """
    canonical = sport.lower().strip()
    tabs = FD_PROP_TABS.get(canonical, [])
    if not tabs:
        return []

    events = _get_events(canonical)
    if not events:
        return []

    all_selections: list[SportsbookSelection] = []
    now_utc = datetime.now(timezone.utc).isoformat()

    for event_id, event in events.items():
        event_name = str(event.get("name", ""))
        open_date = str(event.get("openDate", ""))[:10]

        # Skip events without a real date
        if not open_date or open_date.startswith("2099"):
            continue

        for tab in tabs:
            url = f"{FD_API_BASE}/event-page?eventId={event_id}&tab={tab}&_ak={FD_API_KEY}"
            try:
                data = _request_fd(url)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
                continue

            markets = data.get("attachments", {}).get("markets", {})

            for market_id, market in markets.items():
                market_name = str(market.get("marketName", ""))
                market_key, default_side, default_line = _classify_market(market_name, canonical)

                if not market_key:
                    continue

                runners = market.get("runners", [])
                for runner in runners:
                    selection_id = str(runner.get("selectionId", ""))
                    player_name = str(runner.get("runnerName", ""))
                    handicap = runner.get("handicap")

                    if not selection_id or not player_name:
                        continue

                    # Determine side and line
                    side = default_side
                    line = default_line

                    runner_name_lower = player_name.lower()
                    if "over" in runner_name_lower:
                        side = "OVER"
                        player_name = re.sub(r"\s*(over|under)\s*$", "", player_name, flags=re.IGNORECASE).strip()
                    elif "under" in runner_name_lower:
                        side = "UNDER"
                        player_name = re.sub(r"\s*(over|under)\s*$", "", player_name, flags=re.IGNORECASE).strip()

                    if handicap is not None:
                        line = float(handicap)

                    # For threshold markets without explicit side, it's always OVER
                    if not side and default_line is not None:
                        side = "OVER"

                    # Extract odds
                    odds_data = runner.get("winRunnerOdds", {})
                    american_data = odds_data.get("americanDisplayOdds", {})
                    price = american_data.get("americanOdds")
                    if price is not None:
                        price = int(price)

                    # Parse home/away from event name
                    home_team = ""
                    away_team = ""
                    # FanDuel format: "Away Team (Pitcher) @ Home Team (Pitcher)"
                    if " @ " in event_name:
                        parts = event_name.split(" @ ", 1)
                        away_team = re.sub(r"\s*\(.*?\)\s*$", "", parts[0]).strip()
                        home_team = re.sub(r"\s*\(.*?\)\s*$", "", parts[1]).strip()

                    deeplink_url = build_betslip_url(str(market_id), selection_id)

                    all_selections.append(SportsbookSelection(
                        sport=canonical,
                        book="fanduel",
                        game_date=open_date,
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
                        book_event_id=str(event_id),
                        book_market_id=str(market_id),
                        book_selection_id=selection_id,
                        deeplink_url=deeplink_url,
                        deeplink_quality="betslip",
                        scraped_at=now_utc,
                    ))

    return all_selections
