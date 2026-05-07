from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SPORT_ALIASES = {
    "nba": "nba",
    "basketball_nba": "nba",
    "mlb": "mlb",
    "baseball_mlb": "mlb",
}

SPORT_API_KEYS = {
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
}

DEFAULT_BOOKMAKERS = ("fanduel", "draftkings", "betmgm", "caesars")

BOOKMAKER_TITLES = {
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
}

BOOKMAKER_HOMEPAGES = {
    "betmgm": "https://sports.betmgm.com/",
    "caesars": "https://sportsbook.caesars.com/us/",
    "draftkings": "https://sportsbook.draftkings.com/",
    "fanduel": "https://sportsbook.fanduel.com/",
}

LINK_QUALITY_ORDER = {
    "outcome": 4,
    "market": 3,
    "event": 2,
    "book_home": 1,
    "none": 0,
}

NBA_TARGET_MARKET_KEYS = {
    "PTS": ("player_points",),
    "TRB": ("player_rebounds",),
    "REB": ("player_rebounds",),
    "AST": ("player_assists",),
    "3PM": ("player_threes",),
    "3PT": ("player_threes",),
    "PRA": ("player_points_rebounds_assists",),
    "PR": ("player_points_rebounds",),
    "PA": ("player_points_assists",),
    "RA": ("player_rebounds_assists",),
}

MLB_TARGET_MARKET_KEYS = {
    "H": ("batter_hits",),
    "TB": ("batter_total_bases",),
    "R": ("batter_runs_scored",),
    "RBI": ("batter_rbis",),
    "HR": ("batter_home_runs",),
    "HRR": ("batter_hits_runs_rbis",),
    "K": ("pitcher_strikeouts",),
    "OUTS": ("pitcher_outs",),
    "ER": ("pitcher_earned_runs",),
    "BB": ("pitcher_walks",),
    "HA": ("pitcher_hits_allowed",),
}

NBA_TEAM_CODES = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WSH",
}

MLB_TEAM_CODES = {
    "Arizona Diamondbacks": "ARI",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _canonical_sport(value: str) -> str:
    return SPORT_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())


def _target_market_keys(sport: str) -> dict[str, tuple[str, ...]]:
    canonical = _canonical_sport(sport)
    if canonical == "nba":
        return NBA_TARGET_MARKET_KEYS
    if canonical == "mlb":
        return MLB_TARGET_MARKET_KEYS
    return {}


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def american_to_decimal(odds: Any, default: float | None = None) -> float | None:
    value = safe_float(odds)
    if value is None or abs(value) < 1e-9:
        return default
    if value > 0:
        return round(1.0 + (value / 100.0), 3)
    return round(1.0 + (100.0 / abs(value)), 3)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_person(value: Any) -> str:
    text = normalize_text(value)
    replacements = {
        " jr": "",
        " sr": "",
        " ii": "",
        " iii": "",
        " iv": "",
    }
    for old, new in replacements.items():
        if text.endswith(old):
            text = text[: -len(old)].strip() + new
    return text.strip()


def normalize_side(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "over":
        return "OVER"
    if text == "under":
        return "UNDER"
    return str(value or "").strip().upper()


def _team_code_map(sport: str) -> dict[str, str]:
    canonical = _canonical_sport(sport)
    if canonical == "nba":
        return NBA_TEAM_CODES
    if canonical == "mlb":
        return MLB_TEAM_CODES
    return {}


def team_code_for(value: Any, sport: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if len(upper) <= 4 and upper.replace(".", "").isalpha():
        if upper == "BKN":
            return "BKN"
        if upper == "WSH":
            return "WSH"
        if upper == "WAS" and _canonical_sport(sport) == "mlb":
            return "WSH"
        return upper
    code_map = _team_code_map(sport)
    if text in code_map:
        return code_map[text]
    normalized = normalize_text(text)
    for team_name, team_code in code_map.items():
        if normalize_text(team_name) == normalized:
            return team_code
    return upper


def _bookmaker_priority(bookmaker_key: str, bookmakers: tuple[str, ...]) -> int:
    try:
        return len(bookmakers) - bookmakers.index(bookmaker_key)
    except ValueError:
        return 0


def _bookmaker_homepage(bookmaker_key: Any) -> str | None:
    return BOOKMAKER_HOMEPAGES.get(str(bookmaker_key or "").strip().lower())


def choose_bet_link(outcome: dict[str, Any], market: dict[str, Any], bookmaker: dict[str, Any]) -> tuple[str | None, str]:
    outcome_link = str(outcome.get("link") or "").strip()
    if outcome_link:
        return outcome_link, "outcome"
    market_link = str(market.get("link") or "").strip()
    if market_link:
        return market_link, "market"
    bookmaker_link = str(bookmaker.get("link") or "").strip()
    if bookmaker_link:
        return bookmaker_link, "event"
    return _bookmaker_homepage(bookmaker.get("key")), "book_home"


def _request_json(base_url: str, params: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{base_url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "NBA-Analytics/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        return payload, headers


def resolve_api_key(repo_root: Path | None = None) -> str | None:
    for env_key in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
        value = os.getenv(env_key)
        if value:
            return value

    base = repo_root.resolve() if repo_root else Path(__file__).resolve()
    candidate_names = ("config.local.yaml", ".env.local", ".env", "config.yaml")
    for search_root in [base, *base.parents]:
        for name in candidate_names:
            candidate = search_root / name
            if not candidate.exists():
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError:
                continue
            env_match = re.search(r"(?im)^(?:THE_ODDS_API_KEY|ODDS_API_KEY)\s*=\s*['\"]?([^'\"\r\n]+)", text)
            if env_match:
                return env_match.group(1).strip()
            yaml_match = re.search(
                r"(?ims)odds_api\s*:\s*(?:\n[ \t]+[A-Za-z0-9_]+\s*:\s*.*?)*?\n[ \t]+api_key\s*:\s*['\"]?([^'\"\r\n]+)",
                text,
            )
            if yaml_match:
                return yaml_match.group(1).strip()
            loose_match = re.search(r"(?im)\bapi_key\s*:\s*['\"]?([^'\"\r\n]+)", text)
            if loose_match:
                return loose_match.group(1).strip()
    return None


def _parse_iso_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(text)
        return parsed.date().isoformat()
    except ValueError:
        pass
    match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    return None


def _play_market_keys(play: dict[str, Any], sport: str) -> tuple[str, ...]:
    explicit = str(play.get("market_key") or "").strip()
    if explicit:
        return (explicit,)
    target = str(play.get("target") or "").strip().upper()
    return _target_market_keys(sport).get(target, ())


def _play_name_aliases(play: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for field in ("player_display_name", "player", "market_player_raw"):
        value = str(play.get(field) or "").replace("_", " ").strip()
        normalized = normalize_person(value)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return aliases


def _player_match_score(play_aliases: list[str], candidate_name: str) -> int:
    candidate = normalize_person(candidate_name)
    if not candidate:
        return 0
    for alias in play_aliases:
        if alias == candidate:
            return 100
    for alias in play_aliases:
        if alias and candidate and (alias in candidate or candidate in alias):
            return 85
    return 0


def _extract_catalog_entries(payload: dict[str, Any], sport: str) -> list[dict[str, Any]]:
    canonical = _canonical_sport(sport)
    event_date = _parse_iso_date(payload.get("commence_time"))
    home_code = team_code_for(payload.get("home_team"), canonical)
    away_code = team_code_for(payload.get("away_team"), canonical)

    rows: list[dict[str, Any]] = []
    for bookmaker in payload.get("bookmakers", []):
        book_key = str(bookmaker.get("key") or "").strip().lower()
        book_title = str(bookmaker.get("title") or BOOKMAKER_TITLES.get(book_key) or book_key.title()).strip()
        for market in bookmaker.get("markets", []):
            market_key = str(market.get("key") or "").strip()
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description") or outcome.get("participant") or outcome.get("name")
                link, quality = choose_bet_link(outcome, market, bookmaker)
                rows.append(
                    {
                        "sport": canonical,
                        "odds_event_id": str(payload.get("id") or ""),
                        "event_date": event_date,
                        "commence_time_utc": payload.get("commence_time"),
                        "home_team": payload.get("home_team"),
                        "away_team": payload.get("away_team"),
                        "home_team_code": home_code,
                        "away_team_code": away_code,
                        "bookmaker": book_key,
                        "bookmaker_title": book_title,
                        "bookmaker_sid": bookmaker.get("sid"),
                        "event_link": str(bookmaker.get("link") or "").strip() or None,
                        "market_key": market_key,
                        "market_sid": market.get("sid"),
                        "market_link": str(market.get("link") or "").strip() or None,
                        "player": str(player_name or "").strip(),
                        "player_norm": normalize_person(player_name),
                        "outcome_name": str(outcome.get("name") or "").strip(),
                        "outcome_side": normalize_side(outcome.get("name")),
                        "outcome_sid": outcome.get("sid"),
                        "line": safe_float(outcome.get("point")),
                        "odds_price": safe_float(outcome.get("price")),
                        "betslip_link": link,
                        "link_quality": quality,
                    }
                )
    return rows


def fetch_deep_link_catalog(
    plays: list[dict[str, Any]],
    *,
    sport: str,
    repo_root: Path | None = None,
    api_key: str | None = None,
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical = _canonical_sport(sport)
    summary = {
        "available": False,
        "sport": canonical,
        "bookmakers": list(bookmakers),
        "catalog_count": 0,
        "matched_event_count": 0,
        "requested_market_count": 0,
    }
    resolved_api_key = api_key or resolve_api_key(repo_root)
    if not resolved_api_key:
        summary["reason"] = "missing_odds_api_key"
        return [], summary

    market_keys = sorted({market_key for play in plays for market_key in _play_market_keys(play, canonical)})
    if not market_keys:
        summary["reason"] = "no_supported_market_keys"
        return [], summary
    summary["requested_market_count"] = len(market_keys)

    event_filters = {
        (
            _parse_iso_date(play.get("market_date") or play.get("run_date")),
            team_code_for(play.get("market_home_team"), canonical),
            team_code_for(play.get("market_away_team"), canonical),
        )
        for play in plays
    }
    event_filters.discard((None, "", ""))

    events_url = f"https://api.the-odds-api.com/v4/sports/{SPORT_API_KEYS[canonical]}/events"
    try:
        events_payload, _ = _request_json(events_url, {"apiKey": resolved_api_key})
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        summary["reason"] = f"events_request_failed: {exc}"
        return [], summary

    if not isinstance(events_payload, list):
        summary["reason"] = "events_payload_not_list"
        return [], summary

    event_rows: list[tuple[dict[str, Any], tuple[str | None, str, str]]] = []
    for event in events_payload:
        event_rows.append(
            (
                event,
                (
                    _parse_iso_date(event.get("commence_time")),
                    team_code_for(event.get("home_team"), canonical),
                    team_code_for(event.get("away_team"), canonical),
                ),
            )
        )

    matched_events: list[dict[str, Any]] = []
    for event, event_key in event_rows:
        if not event_filters or event_key in event_filters:
            matched_events.append(event)

    if not matched_events and event_filters:
        team_only_filters = {
            (home_code, away_code)
            for _event_date, home_code, away_code in event_filters
            if home_code or away_code
        }
        for event, (_event_date, home_code, away_code) in event_rows:
            if (home_code, away_code) in team_only_filters:
                matched_events.append(event)

    if not matched_events:
        summary["reason"] = "no_matching_events"
        return [], summary
    summary["matched_event_count"] = len(matched_events)

    catalog: list[dict[str, Any]] = []
    odds_url_template = f"https://api.the-odds-api.com/v4/sports/{SPORT_API_KEYS[canonical]}/events/{{event_id}}/odds"
    for event in matched_events:
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            continue
        try:
            odds_payload, _ = _request_json(
                odds_url_template.format(event_id=event_id),
                {
                    "apiKey": resolved_api_key,
                    "regions": "us",
                    "bookmakers": ",".join(bookmakers),
                    "markets": ",".join(market_keys),
                    "oddsFormat": "american",
                    "includeLinks": "true",
                    "includeSids": "true",
                },
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            continue
        if isinstance(odds_payload, dict):
            catalog.extend(_extract_catalog_entries(odds_payload, canonical))

    summary["available"] = True
    summary["catalog_count"] = len(catalog)
    if not catalog:
        summary["reason"] = "no_catalog_entries"
    return catalog, summary


def match_play_to_catalog(
    play: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    sport: str,
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> dict[str, Any] | None:
    canonical = _canonical_sport(sport)
    market_keys = set(_play_market_keys(play, canonical))
    if not market_keys:
        return None

    direction = normalize_side(play.get("direction"))
    line = safe_float(play.get("market_line") if play.get("market_line") is not None else play.get("line"))
    home_code = team_code_for(play.get("market_home_team"), canonical)
    away_code = team_code_for(play.get("market_away_team"), canonical)
    market_date = _parse_iso_date(play.get("market_date") or play.get("run_date"))
    aliases = _play_name_aliases(play)

    best_match: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None

    for candidate in catalog:
        if candidate.get("market_key") not in market_keys:
            continue
        if direction and candidate.get("outcome_side") != direction:
            continue
        candidate_line = safe_float(candidate.get("line"))
        if line is not None and candidate_line is not None and abs(candidate_line - line) > 1e-9:
            continue

        player_score = _player_match_score(aliases, str(candidate.get("player") or ""))
        if player_score <= 0:
            continue

        event_score = 0
        if home_code and home_code == str(candidate.get("home_team_code") or ""):
            event_score += 20
        if away_code and away_code == str(candidate.get("away_team_code") or ""):
            event_score += 20
        if market_date and market_date == candidate.get("event_date"):
            event_score += 10

        quality_score = LINK_QUALITY_ORDER.get(str(candidate.get("link_quality") or "none"), 0)
        bookmaker_score = _bookmaker_priority(str(candidate.get("bookmaker") or ""), bookmakers)
        price_score = safe_float(candidate.get("odds_price")) or -1000.0

        score = (
            float(player_score),
            float(event_score),
            float(quality_score),
            float(bookmaker_score),
            float(price_score),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_match = candidate

    return dict(best_match) if best_match else None


# ─────────────────────────────────────────────────────────
# FanDuel Search URL Construction (fallback when API unavailable)
# ─────────────────────────────────────────────────────────

FANDUEL_SEARCH_BASE = "https://sportsbook.fanduel.com/search"

FANDUEL_SPORT_NAV = {
    "nba": "https://sportsbook.fanduel.com/navigation/nba",
    "mlb": "https://sportsbook.fanduel.com/navigation/mlb",
}


def build_fanduel_search_url(player_name: str, sport: str = "mlb") -> str:
    """Construct a FanDuel search URL that lands on the player's prop page.

    This is the most reliable way to link users to a specific player's props
    without requiring API deep-link access.
    """
    clean_name = str(player_name or "").replace("_", " ").strip()
    if not clean_name:
        return FANDUEL_SPORT_NAV.get(_canonical_sport(sport), FANDUEL_SEARCH_BASE)
    encoded = urllib.parse.quote_plus(clean_name)
    return f"{FANDUEL_SEARCH_BASE}?q={encoded}&tab=player-props"


def _apply_fanduel_search_fallback(
    plays: list[dict[str, Any]],
    sport: str,
) -> None:
    """Populate betslip_link and bookmaker fields using FanDuel search URLs.

    Applied in-place to any play that doesn't already have a betslip_link.
    """
    for play in plays:
        if play.get("betslip_link"):
            continue
        player_name = str(
            play.get("player_display_name")
            or play.get("player")
            or ""
        ).replace("_", " ").strip()
        play["betslip_link"] = build_fanduel_search_url(player_name, sport)
        play["bookmaker"] = play.get("bookmaker") or "fanduel"
        play["bookmaker_title"] = play.get("bookmaker_title") or "FanDuel"
        play["link_quality"] = play.get("link_quality") or "search"
        play["sportsbook_homepage"] = play.get("sportsbook_homepage") or BOOKMAKER_HOMEPAGES.get("fanduel")


def attach_sportsbook_links_to_plays(
    plays: list[dict[str, Any]],
    *,
    sport: str,
    repo_root: Path | None = None,
    api_key: str | None = None,
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical = _canonical_sport(sport)
    enriched = [dict(play) for play in plays]
    catalog, summary = fetch_deep_link_catalog(
        enriched,
        sport=canonical,
        repo_root=repo_root,
        api_key=api_key,
        bookmakers=bookmakers,
    )

    matched_count = 0
    outcome_link_count = 0
    for play in enriched:
        play.setdefault("market_key", next(iter(_play_market_keys(play, canonical)), ""))
        play.setdefault("outcome_name", normalize_side(play.get("direction")))
        play.setdefault("line", safe_float(play.get("market_line") if play.get("market_line") is not None else play.get("line")))
        play.setdefault("bookmaker", None)
        play.setdefault("bookmaker_title", None)
        play.setdefault("event_link", None)
        play.setdefault("market_link", None)
        play.setdefault("betslip_link", None)
        play.setdefault("link_quality", None)
        play.setdefault("bookmaker_sid", None)
        play.setdefault("market_sid", None)
        play.setdefault("outcome_sid", None)
        play.setdefault("sportsbook_homepage", None)

        match = match_play_to_catalog(play, catalog, sport=canonical, bookmakers=bookmakers)
        if not match:
            odds_value = play.get("odds_american")
            if safe_float(odds_value) is not None:
                play["odds_decimal"] = american_to_decimal(odds_value, default=safe_float(play.get("odds_decimal")))
            continue

        matched_count += 1
        if match.get("link_quality") == "outcome":
            outcome_link_count += 1

        odds_price = safe_float(match.get("odds_price"))
        play.update(
            {
                "market_key": match.get("market_key") or play.get("market_key"),
                "bookmaker": match.get("bookmaker"),
                "bookmaker_title": match.get("bookmaker_title"),
                "event_link": match.get("event_link"),
                "market_link": match.get("market_link"),
                "betslip_link": match.get("betslip_link"),
                "link_quality": match.get("link_quality"),
                "bookmaker_sid": match.get("bookmaker_sid"),
                "market_sid": match.get("market_sid"),
                "outcome_sid": match.get("outcome_sid"),
                "odds_event_id": match.get("odds_event_id"),
                "sportsbook_homepage": _bookmaker_homepage(match.get("bookmaker")),
            }
        )
        if odds_price is not None:
            odds_int = int(round(odds_price))
            play["odds_price"] = odds_int
            play["odds_american"] = odds_int
            play["odds_decimal"] = american_to_decimal(odds_int)
        elif safe_float(play.get("odds_american")) is not None:
            play["odds_decimal"] = american_to_decimal(play.get("odds_american"), default=safe_float(play.get("odds_decimal")))

    summary["matched_play_count"] = matched_count
    summary["outcome_link_count"] = outcome_link_count
    summary["play_count"] = len(enriched)
    summary["coverage_rate"] = (matched_count / len(enriched)) if enriched else 0.0

    # Fallback: for any plays that still lack a betslip_link, construct
    # FanDuel search URLs so users always have a clickable path to bet.
    _apply_fanduel_search_fallback(enriched, canonical)
    fallback_count = sum(1 for p in enriched if p.get("link_quality") == "search")
    summary["search_fallback_count"] = fallback_count

    return enriched, summary


def _parlay_option_sort_key(option: dict[str, Any], bookmakers: tuple[str, ...]) -> tuple[int, int, int]:
    return (
        int(bool(option.get("complete"))),
        int(option.get("covered_leg_count", 0)),
        _bookmaker_priority(str(option.get("bookmaker") or ""), bookmakers),
    )


def build_parlay_sportsbook_options(
    legs: list[dict[str, Any]],
    *,
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> list[dict[str, Any]]:
    total_leg_keys = {
        str(leg.get("play_key") or f"index:{index}")
        for index, leg in enumerate(legs)
    }
    grouped: dict[str, dict[str, Any]] = {}
    for index, leg in enumerate(legs):
        bookmaker = str(leg.get("bookmaker") or "").strip().lower()
        link = str(leg.get("betslip_link") or "").strip()
        if not bookmaker or not link:
            continue
        leg_key = str(leg.get("play_key") or f"index:{index}")
        option = grouped.setdefault(
            bookmaker,
            {
                "bookmaker": bookmaker,
                "bookmaker_title": leg.get("bookmaker_title") or BOOKMAKER_TITLES.get(bookmaker) or bookmaker.title(),
                "links": [],
                "leg_indices": [],
                "leg_keys": [],
                "link_quality": leg.get("link_quality"),
                "primary_link": link,
            },
        )
        if link not in option["links"]:
            option["links"].append(link)
        option["leg_indices"].append(index)
        if leg_key not in option["leg_keys"]:
            option["leg_keys"].append(leg_key)
        if LINK_QUALITY_ORDER.get(str(leg.get("link_quality") or "none"), 0) > LINK_QUALITY_ORDER.get(str(option.get("link_quality") or "none"), 0):
            option["link_quality"] = leg.get("link_quality")
            option["primary_link"] = link

    options: list[dict[str, Any]] = []
    for option in grouped.values():
        covered_leg_count = len(set(option["leg_keys"]))
        options.append(
            {
                **option,
                "covered_leg_count": covered_leg_count,
                "complete": covered_leg_count == len(total_leg_keys),
            }
        )

    options.sort(key=lambda item: _parlay_option_sort_key(item, bookmakers), reverse=True)
    return options


def enrich_parlay_payload_with_sportsbooks(
    parlay_payload: dict[str, Any],
    *,
    sport: str,
    default_leg_odds: int = -110,
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> dict[str, Any]:
    payload = {
        "plays": [dict(play) for play in parlay_payload.get("plays", [])],
        "pairs": [dict(pair) for pair in parlay_payload.get("pairs", [])],
        "summary": dict(parlay_payload.get("summary", {})),
    }
    plays_by_key = {
        str(play.get("play_key") or ""): play
        for play in payload["plays"]
        if str(play.get("play_key") or "").strip()
    }

    parlays_out: list[dict[str, Any]] = []
    for pair in payload["pairs"]:
        legs: list[dict[str, Any]] = []
        for leg in pair.get("legs", []):
            play_key = str(leg.get("play_key") or "")
            source = plays_by_key.get(play_key, {})
            merged = dict(source)
            merged.update(leg)
            if safe_float(merged.get("market_line")) is not None:
                merged["market_line"] = float(merged["market_line"])
            if safe_float(merged.get("odds_american")) is None:
                merged["odds_american"] = default_leg_odds
            merged["odds_decimal"] = american_to_decimal(merged.get("odds_american"), default=american_to_decimal(default_leg_odds))
            legs.append(merged)

        sportsbook_options = build_parlay_sportsbook_options(legs, bookmakers=bookmakers)
        decimal_prices = [safe_float(leg.get("odds_decimal")) or american_to_decimal(default_leg_odds) or 1.909 for leg in legs]
        parlay_decimal = round(math.prod(decimal_prices), 2) if decimal_prices else 1.0
        if parlay_decimal <= 1.0:
            parlay_american = default_leg_odds
        elif parlay_decimal >= 2.0:
            parlay_american = int(round((parlay_decimal - 1.0) * 100.0))
        else:
            parlay_american = int(round(-100.0 / max(parlay_decimal - 1.0, 1e-9)))

        probability_fields = ("parlay_leg_probability", "expected_win_rate", "estimated_graded_hit_rate")
        leg_probabilities: list[float] = []
        for leg in legs:
            for field in probability_fields:
                value = safe_float(leg.get(field))
                if value is not None:
                    leg_probabilities.append(value)
                    break

        parlays_out.append(
            {
                **pair,
                "legs": legs,
                "sportsbook_options": sportsbook_options,
                "recommended_sportsbook": sportsbook_options[0] if sportsbook_options else None,
                "joint_probability": safe_float(pair.get("projected_probability")),
                "adjusted_probability": safe_float(pair.get("projected_probability")),
                "odds_decimal": parlay_decimal,
                "odds_american": f"+{parlay_american}" if parlay_american > 0 else str(parlay_american),
                "payout_per_dollar": round(max(parlay_decimal - 1.0, 0.0), 2),
                "avg_win_rate": (sum(leg_probabilities) / len(leg_probabilities)) if leg_probabilities else None,
                "n_games": len(
                    {
                        str(leg.get("game_key") or leg.get("game_id") or leg.get("odds_event_id") or "")
                        for leg in legs
                        if str(leg.get("game_key") or leg.get("game_id") or leg.get("odds_event_id") or "").strip()
                    }
                ),
            }
        )

    payload["pairs"] = parlays_out
    payload["parlay_board"] = {
        "parlays": parlays_out,
        "diagnostics": payload.get("summary", {}),
    }
    payload["summary"]["sportsbook_complete_parlay_count"] = int(
        sum(1 for parlay in parlays_out if parlay.get("recommended_sportsbook", {}).get("complete"))
    )
    return payload
