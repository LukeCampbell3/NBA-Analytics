from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

from .io_utils import ensure_dir, normalize_player_name, write_csv, write_json

ET = ZoneInfo("America/New_York")
ODDS_API_IO_BASE = "https://api.odds-api.io/v3"
SPORTSGAMEODDS_BASE = "https://api.sportsgameodds.com/v2"
COVERS_MLB_PROPS_URL = "https://www.covers.com/sport/baseball/mlb/player-props"
DEFAULT_PROVIDER = "covers"
SUPPORTED_PROVIDERS = ("covers", "odds_api_io", "sportsgameodds")
DEFAULT_BOOKMAKERS = ("DraftKings", "FanDuel", "BetMGM", "Caesars")
DEFAULT_MARKETS = ("H", "HR", "RBI", "K", "ER")

TARGET_CONFIG = {
    "H": {
        "player_type": "hitter",
        "odds_api_aliases": ("player props - hits", "batter hits", "hits"),
        "sportsgameodds_stat_ids": ("batting_hits",),
    },
    "HR": {
        "player_type": "hitter",
        "odds_api_aliases": ("player props - home runs", "batter home runs", "home runs", "home run"),
        "sportsgameodds_stat_ids": ("batting_home_runs",),
    },
    "RBI": {
        "player_type": "hitter",
        "odds_api_aliases": ("player props - rbis", "batter rbis", "rbis", "rbi"),
        "sportsgameodds_stat_ids": ("batting_rbis",),
    },
    "K": {
        "player_type": "pitcher",
        "odds_api_aliases": ("player props - strikeouts", "pitcher strikeouts", "strikeouts", "strikeout"),
        "sportsgameodds_stat_ids": ("pitching_strikeouts",),
    },
    "ER": {
        "player_type": "pitcher",
        "odds_api_aliases": ("player props - earned runs", "pitcher earned runs", "earned runs", "earned run"),
        "sportsgameodds_stat_ids": ("pitching_earned_runs",),
    },
}

COVERS_MARKET_MAP = {
    "H": "MLB_GAME_PLAYER_HITS",
    "HR": "MLB_GAME_PLAYER_HOME_RUNS",
    "RBI": "MLB_GAME_PLAYER_RBIS",
    "K": "MLB_GAME_PLAYER_PITCHER_STRIKEOUTS",
    "ER": "MLB_GAME_PLAYER_PITCHER_ALLOWED_EARNED_RUNS",
}

COVERS_EVENT_LABEL_MAP = {
    "MLB_GAME_PLAYER_HITS": "Total Hits",
    "MLB_GAME_PLAYER_HOME_RUNS": "Total Home Runs",
    "MLB_GAME_PLAYER_RBIS": "Total RBIs",
    "MLB_GAME_PLAYER_PITCHER_STRIKEOUTS": "Strikeouts Thrown",
    "MLB_GAME_PLAYER_PITCHER_ALLOWED_EARNED_RUNS": "Earned Runs Allowed",
}

COVERS_BOOKMAKER_MAP = {
    "bet365": "Bet365",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "draft-kings": "DraftKings",
    "draftkings": "DraftKings",
    "fan-d": "FanDuel",
    "fanduel": "FanDuel",
    "fanatics": "Fanatics",
    "hard-rock": "HardRock",
    "hardrock": "HardRock",
    "espn-bet": "ESPNBet",
    "espnbet": "ESPNBet",
    "rivers": "BetRivers",
    "sugarhouse": "BetRivers",
}


@dataclass(frozen=True)
class MarketCollectorConfig:
    api_key: str
    market_root: Path
    provider: str = DEFAULT_PROVIDER
    mode: str = "current"
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS
    markets: tuple[str, ...] = DEFAULT_MARKETS
    event_date: str | None = None
    request_timeout_sec: int = 30
    limit: int = 30
    promote_to_latest: bool = True


def provider_env_var(provider: str) -> str:
    provider = str(provider).strip().lower()
    if provider == "covers":
        return ""
    if provider == "odds_api_io":
        return "ODDS_API_IO_KEY"
    if provider == "sportsgameodds":
        return "SPORTSGAMEODDS_API_KEY"
    raise ValueError(f"Unsupported provider: {provider}")


def provider_env_candidates(provider: str) -> tuple[str, ...]:
    provider = str(provider).strip().lower()
    if provider == "covers":
        return ()
    if provider == "odds_api_io":
        return ("ODDS_API_IO_KEY", "ODDS_API_KEY")
    if provider == "sportsgameodds":
        return ("SPORTSGAMEODDS_API_KEY", "SPORTS_GAME_ODDS_API_KEY", "SPORTS_ODDS_API_KEY_HEADER")
    raise ValueError(f"Unsupported provider: {provider}")


def provider_requires_api_key(provider: str) -> bool:
    return len(provider_env_candidates(provider)) > 0


def load_provider_api_key(provider: str, explicit: str | None = None) -> str | None:
    if explicit:
        value = str(explicit).strip()
        return value or None

    for name in provider_env_candidates(provider):
        value = os.getenv(name)
        if value:
            value = str(value).strip()
            if value:
                return value
    dotenv_values = _load_dotenv_values()
    for name in provider_env_candidates(provider):
        value = dotenv_values.get(name)
        if value:
            value = str(value).strip()
            if value:
                return value
    return None


def default_event_date() -> str:
    return datetime.now(ET).date().isoformat()


def collect_market_lines(config: MarketCollectorConfig) -> dict:
    provider = str(config.provider).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}. Expected one of {SUPPORTED_PROVIDERS}")
    if str(config.mode).strip().lower() != "current":
        raise NotImplementedError(
            f"{provider} collection is current-only in this repo flow. Historical line archiving is not implemented here."
        )

    event_date = str(config.event_date or default_event_date())
    provider_root = ensure_dir(Path(config.market_root).resolve() / provider)
    snapshot_root = ensure_dir(provider_root / "raw" / _utc_stamp())

    if provider == "covers":
        raw_payload = _collect_from_covers(config, event_date=event_date, snapshot_root=snapshot_root)
        long_df = _normalize_covers_payload(raw_payload, config, event_date=event_date)
    elif provider == "odds_api_io":
        raw_payload = _collect_from_odds_api_io(config, event_date=event_date, snapshot_root=snapshot_root)
        long_df = _normalize_odds_api_io_payload(raw_payload["odds"], config, event_date=event_date)
    else:
        raw_payload = _collect_from_sportsgameodds(config, event_date=event_date, snapshot_root=snapshot_root)
        long_df = _normalize_sportsgameodds_payload(raw_payload["events"], config, event_date=event_date)

    wide_df = _build_wide_market_file(long_df, provider=provider)
    output_paths = _write_provider_outputs(provider_root=provider_root, long_df=long_df, wide_df=wide_df)
    manifest = _build_manifest(
        config=config,
        provider=provider,
        event_date=event_date,
        raw_payload=raw_payload,
        long_df=long_df,
        wide_df=wide_df,
        output_paths=output_paths,
    )
    write_json(provider_root / "latest_manifest.json", manifest)
    if config.promote_to_latest:
        publish_market_outputs(manifest=manifest, market_root=Path(config.market_root).resolve())
    return manifest


def publish_market_outputs(manifest: dict, market_root: Path) -> dict:
    root = ensure_dir(Path(market_root).resolve())
    outputs = manifest.get("output_files", {})
    copied: dict[str, str] = {}
    for source_key, target_name in (
        ("long", "latest_player_props_long.csv"),
        ("wide", "latest_player_props_wide.csv"),
        ("manifest", "latest_manifest.json"),
    ):
        source = outputs.get(source_key)
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            continue
        target = root / target_name
        ensure_dir(target.parent)
        shutil.copy2(source_path, target)
        copied[source_key] = str(target)
    return copied


def compare_provider_manifests(manifests: list[dict]) -> dict:
    ranked = sorted(
        [m for m in manifests if m.get("status") == "ok"],
        key=lambda item: (
            float(item.get("quality_score", 0.0)),
            int(item.get("normalized_wide_rows", 0)),
            int(item.get("unique_players", 0)),
        ),
        reverse=True,
    )
    winner = ranked[0] if ranked else None
    return {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "winner_provider": winner.get("provider") if winner else None,
        "winner_quality_score": winner.get("quality_score") if winner else 0.0,
        "ranked": ranked,
    }


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@lru_cache(maxsize=1)
def _load_dotenv_values() -> dict[str, str]:
    candidates = []
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".env")
        candidates.append(parent / ".env.local")
    values: dict[str, str] = {}
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in values:
                values[key] = value
    return values


def _http_get_json(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_sec: int = 30,
) -> Any:
    query = urlencode({k: v for k, v in params.items() if v is not None and v != ""}, doseq=False)
    request = Request(f"{url}?{query}" if query else url, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching {url}: {exc}") from exc


def _http_get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_sec: int = 30,
) -> str:
    query = urlencode({k: v for k, v in (params or {}).items() if v is not None and v != ""}, doseq=False)
    request = Request(
        f"{url}?{query}" if query else url,
        headers=headers
        or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching {url}: {exc}") from exc


def _collect_from_covers(config: MarketCollectorConfig, *, event_date: str, snapshot_root: Path) -> dict[str, Any]:
    league_html = _http_get_text(COVERS_MLB_PROPS_URL, timeout_sec=config.request_timeout_sec)
    (snapshot_root / "league_page.html").write_text(league_html, encoding="utf-8")

    matchup_links = _extract_covers_matchup_links(league_html)
    market_payloads: list[dict[str, Any]] = []
    for game_id, matchup_url in matchup_links:
        matchup_html = _http_get_text(matchup_url, timeout_sec=config.request_timeout_sec)
        available_markets = _extract_covers_available_markets(matchup_html)
        game_dir = ensure_dir(snapshot_root / str(game_id))
        (game_dir / "matchup_page.html").write_text(matchup_html, encoding="utf-8")
        for target in config.markets:
            target_name = str(target).upper()
            covers_market = COVERS_MARKET_MAP.get(target_name)
            if not covers_market or covers_market not in available_markets:
                continue
            html = _http_get_text(
                f"https://www.covers.com/sport/player-props/matchup/mlb/{game_id}",
                params={
                    "propEvent": covers_market,
                    "countryCode": "US",
                    "stateProv": "TN",
                    "isLeagueVersion": "False",
                    "isMatchupExperiment": "true",
                },
                timeout_sec=config.request_timeout_sec,
            )
            safe_market = normalize_player_name(covers_market)
            (game_dir / f"{safe_market}.html").write_text(html, encoding="utf-8")
            market_payloads.append(
                {
                    "game_id": str(game_id),
                    "matchup_url": matchup_url,
                    "market_code": covers_market,
                    "target": target_name,
                    "html": html,
                }
            )
    return {
        "league_url": COVERS_MLB_PROPS_URL,
        "matchups": [{"game_id": game_id, "url": url} for game_id, url in matchup_links],
        "markets": market_payloads,
    }


def _collect_from_odds_api_io(config: MarketCollectorConfig, *, event_date: str, snapshot_root: Path) -> dict[str, Any]:
    bookmaker_names = [book.strip() for book in config.bookmakers if str(book).strip()]
    bookmakers = ",".join(bookmaker_names)
    start_et = datetime.fromisoformat(event_date).replace(tzinfo=ET)
    end_et = start_et + timedelta(days=1)
    league_slug = _discover_odds_api_io_mlb_league_slug(config)
    events, attempts = _fetch_odds_api_events_with_fallbacks(
        config=config,
        event_date=event_date,
        start_et=start_et,
        end_et=end_et,
        league_slug=league_slug,
        preferred_bookmaker=bookmaker_names[0] if bookmaker_names else None,
    )
    write_json(
        snapshot_root / "events.json",
        {"provider": "odds_api_io", "event_date": event_date, "attempts": attempts, "data": events},
    )
    filtered_events = [
        event
        for event in events
        if _event_matches_date(event.get("date"), event_date) and _odds_api_event_is_mlb(event)
    ]
    odds_batches: list[dict[str, Any]] = []
    for chunk in _chunk([event.get("id") for event in filtered_events if event.get("id") is not None], size=10):
        payload = _http_get_json(
            f"{ODDS_API_IO_BASE}/odds/multi",
            params={
                "apiKey": config.api_key,
                "eventIds": ",".join(str(item) for item in chunk),
                "bookmakers": bookmakers,
            },
            timeout_sec=config.request_timeout_sec,
        )
        odds_batches.extend(payload if isinstance(payload, list) else [payload])
    write_json(snapshot_root / "odds.json", {"provider": "odds_api_io", "event_date": event_date, "data": odds_batches})
    return {"events": filtered_events, "odds": odds_batches}


def _fetch_odds_api_events_with_fallbacks(
    *,
    config: MarketCollectorConfig,
    event_date: str,
    start_et: datetime,
    end_et: datetime,
    league_slug: str | None,
    preferred_bookmaker: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_limit = max(int(config.limit), 100)
    utc_start = start_et.astimezone(UTC)
    utc_end = end_et.astimezone(UTC)
    attempts: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []

    def add_variant(
        *,
        use_league: bool,
        bookmaker: str | None,
        status: str | None,
        from_dt: datetime | None,
        to_dt: datetime | None,
        note: str,
    ) -> None:
        params: dict[str, Any] = {
            "apiKey": config.api_key,
            "sport": "baseball",
            "limit": base_limit,
        }
        if use_league and league_slug:
            params["league"] = league_slug
        if bookmaker:
            params["bookmaker"] = bookmaker
        if status:
            params["status"] = status
        if from_dt is not None:
            params["from"] = from_dt.isoformat().replace("+00:00", "Z")
        if to_dt is not None:
            params["to"] = to_dt.isoformat().replace("+00:00", "Z")
        params["__note"] = note
        variants.append(params)

    add_variant(
        use_league=True,
        bookmaker=preferred_bookmaker,
        status="pending,live",
        from_dt=utc_start,
        to_dt=utc_end,
        note="league+bookmaker+day_window+pending_live",
    )
    add_variant(
        use_league=False,
        bookmaker=preferred_bookmaker,
        status="pending,live",
        from_dt=utc_start,
        to_dt=utc_end,
        note="bookmaker+day_window+pending_live",
    )
    add_variant(
        use_league=False,
        bookmaker=preferred_bookmaker,
        status="pending",
        from_dt=utc_start,
        to_dt=utc_end,
        note="bookmaker+day_window+pending_only",
    )
    add_variant(
        use_league=False,
        bookmaker=None,
        status="pending,live",
        from_dt=utc_start,
        to_dt=utc_end,
        note="day_window+pending_live",
    )
    add_variant(
        use_league=False,
        bookmaker=preferred_bookmaker,
        status=None,
        from_dt=utc_start - timedelta(hours=12),
        to_dt=utc_end + timedelta(hours=12),
        note="bookmaker+expanded_window+no_status",
    )
    add_variant(
        use_league=False,
        bookmaker=None,
        status=None,
        from_dt=utc_start - timedelta(hours=12),
        to_dt=utc_end + timedelta(hours=12),
        note="expanded_window+no_status",
    )

    last_error: RuntimeError | None = None
    for variant in variants:
        note = str(variant.pop("__note"))
        try:
            events = _fetch_odds_api_events_pages(config=config, params=variant)
            if event_date == default_event_date():
                live_events = _fetch_odds_api_live_events(config=config)
                if live_events:
                    events = _merge_event_lists(events, live_events)
            filtered = [event for event in events if _event_matches_date(event.get("date"), event_date)]
            mlb_filtered = [event for event in filtered if _odds_api_event_is_mlb(event)]
            attempts.append(
                {
                    "note": note,
                    "params": {k: v for k, v in variant.items() if k != "apiKey"},
                    "raw_count": int(len(events)),
                    "date_filtered_count": int(len(filtered)),
                    "mlb_filtered_count": int(len(mlb_filtered)),
                }
            )
            if mlb_filtered:
                return events, attempts
            if filtered:
                return events, attempts
            if events:
                return events, attempts
        except RuntimeError as exc:
            last_error = exc
            attempts.append(
                {
                    "note": note,
                    "params": {k: v for k, v in variant.items() if k != "apiKey"},
                    "error": str(exc),
                }
            )
            continue
    if last_error is not None:
        raise last_error
    return [], attempts


def _fetch_odds_api_live_events(*, config: MarketCollectorConfig) -> list[dict[str, Any]]:
    page = _http_get_json(
        f"{ODDS_API_IO_BASE}/events/live",
        params={
            "apiKey": config.api_key,
            "sport": "baseball",
        },
        timeout_sec=config.request_timeout_sec,
    )
    return page if isinstance(page, list) else []


def _fetch_odds_api_events_pages(*, config: MarketCollectorConfig, params: dict[str, Any]) -> list[dict[str, Any]]:
    all_events: list[dict[str, Any]] = []
    limit = int(params.get("limit", 100) or 100)
    for page_idx in range(0, 5):
        page_params = dict(params)
        page_params["skip"] = page_idx * limit
        page = _http_get_json(
            f"{ODDS_API_IO_BASE}/events",
            params=page_params,
            timeout_sec=config.request_timeout_sec,
        )
        batch = page if isinstance(page, list) else []
        all_events.extend(batch)
        if len(batch) < limit:
            break
    return all_events


def _merge_event_lists(*event_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for event_list in event_lists:
        for event in event_list:
            event_id = str((event or {}).get("id", "")).strip()
            dedupe_key = event_id or json.dumps(event, sort_keys=True, ensure_ascii=True)
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            merged.append(event)
    return merged


def _collect_from_sportsgameodds(config: MarketCollectorConfig, *, event_date: str, snapshot_root: Path) -> dict[str, Any]:
    start_et = datetime.fromisoformat(event_date).replace(tzinfo=ET)
    end_et = start_et + timedelta(days=1)
    params = {
        "apiKey": config.api_key,
        "leagueID": "MLB",
        "oddsAvailable": "true",
        "includeOpposingOdds": "true",
        "startsAfter": start_et.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "startsBefore": end_et.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "limit": config.limit,
    }
    bookmaker_ids = ",".join(_normalize_sgo_bookmaker(book) for book in config.bookmakers if str(book).strip())
    if bookmaker_ids:
        params["bookmakerID"] = bookmaker_ids

    events: list[dict[str, Any]] = []
    cursor: str | None = None
    page_index = 0
    while True:
        page_index += 1
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        page = _http_get_json(
            f"{SPORTSGAMEODDS_BASE}/events",
            params=page_params,
            timeout_sec=config.request_timeout_sec,
        )
        write_json(snapshot_root / f"events_page_{page_index:02d}.json", page)
        batch = page.get("data", []) if isinstance(page, dict) else []
        events.extend(batch)
        cursor = page.get("nextCursor") if isinstance(page, dict) else None
        if not cursor:
            break
    return {"events": events}


def _normalize_odds_api_io_payload(
    payload: list[dict[str, Any]],
    config: MarketCollectorConfig,
    *,
    event_date: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = {str(item).upper() for item in config.markets}
    for event in payload:
        home = str(event.get("home", "")).strip()
        away = str(event.get("away", "")).strip()
        commence = event.get("date")
        bookmakers = event.get("bookmakers") or {}
        for bookmaker, markets in bookmakers.items():
            for market in markets or []:
                target = _map_odds_api_market_name(market.get("name"), targets=targets)
                if target is None:
                    continue
                for odd in market.get("odds") or []:
                    player = str(odd.get("label", "")).strip()
                    if not player:
                        continue
                    line = _to_float(odd.get("hdp"))
                    rows.append(
                        {
                            "Date": event_date,
                            "Player": player,
                            "Player_Type": TARGET_CONFIG[target]["player_type"],
                            "Target": target,
                            "Line": line,
                            "Over_Price": _decimal_to_american(odd.get("over")),
                            "Under_Price": _decimal_to_american(odd.get("under")),
                            "Bookmaker": str(bookmaker).strip(),
                            "Provider": "odds_api_io",
                            "Market_Player_Norm": normalize_player_name(player),
                            "Market_Fetched_At_UTC": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "Market_Event_ID": str(event.get("id", "")),
                            "Market_Commence_Time_UTC": _format_utc(commence),
                            "Market_Home_Team": home,
                            "Market_Away_Team": away,
                            "Market_Name": str(market.get("name", "")).strip(),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _normalize_covers_payload(payload: dict[str, Any], config: MarketCollectorConfig, *, event_date: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    allowed_books = {_normalize_book_name(book) for book in config.bookmakers if str(book).strip()}
    fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for market_payload in payload.get("markets", []):
        game_id = str(market_payload.get("game_id", "")).strip()
        target = str(market_payload.get("target", "")).upper()
        html = str(market_payload.get("html", ""))
        if not html or target not in TARGET_CONFIG:
            continue
        for article in _extract_covers_articles(html):
            market_label = article.get("market_label", "")
            if market_label != COVERS_EVENT_LABEL_MAP.get(COVERS_MARKET_MAP.get(target, ""), market_label):
                continue
            for book_row in article.get("books", []):
                bookmaker = str(book_row.get("bookmaker", "")).strip()
                if not bookmaker:
                    continue
                if allowed_books and _normalize_book_name(bookmaker) not in allowed_books:
                    continue
                line = book_row.get("line")
                rows.append(
                    {
                        "Date": event_date,
                        "Player": article.get("player", ""),
                        "Player_Type": TARGET_CONFIG[target]["player_type"],
                        "Target": target,
                        "Line": line,
                        "Over_Price": book_row.get("over_price"),
                        "Under_Price": book_row.get("under_price"),
                        "Bookmaker": bookmaker,
                        "Provider": "covers",
                        "Market_Player_Norm": normalize_player_name(article.get("player", "")),
                        "Market_Fetched_At_UTC": fetched_at,
                        "Market_Event_ID": game_id,
                        "Market_Commence_Time_UTC": "",
                        "Market_Home_Team": article.get("home_team", ""),
                        "Market_Away_Team": article.get("away_team", ""),
                        "Market_Name": market_label,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _normalize_sportsgameodds_payload(
    payload: list[dict[str, Any]],
    config: MarketCollectorConfig,
    *,
    event_date: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = {str(item).upper() for item in config.markets}
    stat_to_target: dict[str, str] = {}
    for target in targets:
        for stat_id in TARGET_CONFIG.get(target, {}).get("sportsgameodds_stat_ids", ()):
            stat_to_target[stat_id] = target

    for event in payload:
        starts_at = (((event.get("status") or {}).get("startsAt")) if isinstance(event, dict) else None)
        if not _event_matches_date(starts_at, event_date):
            continue
        players = event.get("players") or {}
        odds = event.get("odds") or {}
        pair_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for odd_id, odd in odds.items():
            stat_id = str(odd.get("statID", "")).strip()
            target = stat_to_target.get(stat_id)
            if target is None:
                continue
            player_id = str(odd.get("playerID", "")).strip()
            if not player_id:
                continue
            player = _resolve_sgo_player_name(players.get(player_id), player_id=player_id)
            pair_id = _pair_identifier(odd_id=str(odd_id), opposing_odd_id=odd.get("opposingOddID"))
            by_bookmaker = odd.get("byBookmaker") or {}
            for bookmaker_id, book_data in by_bookmaker.items():
                book_line = _to_float((book_data or {}).get("spread"))
                base_line = _to_float(odd.get("bookSpread"))
                line = book_line if book_line is not None else base_line
                side = _resolve_side(
                    side=(book_data or {}).get("overUnder"),
                    fallback=odd.get("sideID"),
                    odd_id=str(odd_id),
                )
                if side is None:
                    continue
                price = _to_float((book_data or {}).get("odds"))
                if price is None:
                    price = _to_float(odd.get("bookOdds"))
                group_key = (str(event.get("eventID", "")), player, target, str(bookmaker_id))
                group = pair_rows.setdefault(
                    group_key + (pair_id,),
                    {
                        "Date": event_date,
                        "Player": player,
                        "Player_Type": TARGET_CONFIG[target]["player_type"],
                        "Target": target,
                        "Line": line,
                        "Over_Price": math.nan,
                        "Under_Price": math.nan,
                        "Bookmaker": str(bookmaker_id).strip(),
                        "Provider": "sportsgameodds",
                        "Market_Player_Norm": normalize_player_name(player),
                        "Market_Fetched_At_UTC": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "Market_Event_ID": str(event.get("eventID", "")),
                        "Market_Commence_Time_UTC": _format_utc(starts_at),
                        "Market_Home_Team": _safe_team_name(((event.get("teams") or {}).get("home") or {}).get("names")),
                        "Market_Away_Team": _safe_team_name(((event.get("teams") or {}).get("away") or {}).get("names")),
                        "Market_Name": str(odd.get("marketName", stat_id)).strip(),
                    },
                )
                if group.get("Line") is None and line is not None:
                    group["Line"] = line
                if side == "over":
                    group["Over_Price"] = price
                elif side == "under":
                    group["Under_Price"] = price
        rows.extend(pair_rows.values())
    return pd.DataFrame.from_records(rows)


def _build_wide_market_file(long_df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    if long_df.empty:
        columns = [
            "Date",
            "Player",
            "Player_Type",
            "Market_Player_Norm",
            "Market_Fetched_At_UTC",
            "Market_Event_ID",
            "Market_Commence_Time_UTC",
            "Market_Home_Team",
            "Market_Away_Team",
            "Market_Provider",
        ]
        for target in DEFAULT_MARKETS:
            columns.extend(
                [
                    f"Market_{target}",
                    f"Synthetic_Market_{target}",
                    f"Market_{target}_books",
                    f"Market_{target}_over_price",
                    f"Market_{target}_under_price",
                    f"Market_{target}_line_std",
                    f"Market_Provider_{target}",
                ]
            )
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = long_df.groupby(["Date", "Player", "Player_Type"], sort=False)
    for (date_value, player, player_type), group in grouped:
        summary: dict[str, Any] = {
            "Date": date_value,
            "Player": player,
            "Player_Type": player_type,
            "Market_Player_Norm": normalize_player_name(player),
            "Market_Fetched_At_UTC": _latest_timestamp(group["Market_Fetched_At_UTC"]),
            "Market_Event_ID": _first_non_empty(group["Market_Event_ID"]),
            "Market_Commence_Time_UTC": _first_non_empty(group["Market_Commence_Time_UTC"]),
            "Market_Home_Team": _first_non_empty(group["Market_Home_Team"]),
            "Market_Away_Team": _first_non_empty(group["Market_Away_Team"]),
            "Market_Provider": provider,
        }
        for target in DEFAULT_MARKETS:
            target_rows = group.loc[group["Target"] == target].copy()
            if target_rows.empty:
                continue
            lines = pd.to_numeric(target_rows["Line"], errors="coerce").dropna()
            summary[f"Market_{target}"] = float(lines.median()) if not lines.empty else math.nan
            summary[f"Synthetic_Market_{target}"] = summary[f"Market_{target}"]
            summary[f"Market_{target}_books"] = int(target_rows["Bookmaker"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
            summary[f"Market_{target}_over_price"] = _series_mean(target_rows["Over_Price"])
            summary[f"Market_{target}_under_price"] = _series_mean(target_rows["Under_Price"])
            summary[f"Market_{target}_line_std"] = _series_std(lines)
            summary[f"Market_Provider_{target}"] = provider
        rows.append(summary)
    out = pd.DataFrame.from_records(rows)
    out = out.sort_values(["Date", "Player_Type", "Player"], ignore_index=True)
    return out


def _write_provider_outputs(*, provider_root: Path, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> dict[str, str]:
    ensure_dir(provider_root)
    long_path = provider_root / "latest_player_props_long.csv"
    wide_path = provider_root / "latest_player_props_wide.csv"
    write_csv(long_df, long_path)
    write_csv(wide_df, wide_path)
    return {
        "long": str(long_path),
        "wide": str(wide_path),
        "manifest": str(provider_root / "latest_manifest.json"),
    }


def _build_manifest(
    *,
    config: MarketCollectorConfig,
    provider: str,
    event_date: str,
    raw_payload: dict[str, Any],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    target_counts: dict[str, int] = {}
    books_by_target: dict[str, float] = {}
    for target in DEFAULT_MARKETS:
        if wide_df.empty or f"Market_{target}" not in wide_df.columns:
            target_counts[target] = 0
            books_by_target[target] = 0.0
            continue
        target_frame = wide_df.loc[pd.to_numeric(wide_df.get(f"Market_{target}"), errors="coerce").notna()]
        target_counts[target] = int(len(target_frame))
        books_by_target[target] = _series_mean(target_frame.get(f"Market_{target}_books", pd.Series(dtype=float)))
    unique_players = int(wide_df["Player"].nunique()) if not wide_df.empty and "Player" in wide_df.columns else 0
    covered_targets = sum(1 for count in target_counts.values() if count > 0)
    quality_score = float(
        (len(wide_df) * 1.0)
        + (unique_players * 1.5)
        + (covered_targets * 20.0)
        + sum(books_by_target.values())
    )
    return {
        "status": "ok",
        "provider": provider,
        "mode": config.mode,
        "event_date": event_date,
        "bookmakers": list(config.bookmakers),
        "markets": list(config.markets),
        "raw_event_count": int(len(raw_payload.get("events", []))),
        "normalized_long_rows": int(len(long_df)),
        "normalized_wide_rows": int(len(wide_df)),
        "unique_players": unique_players,
        "target_rows": target_counts,
        "avg_books_by_target": books_by_target,
        "quality_score": round(quality_score, 3),
        "output_files": output_paths,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }


def _event_matches_date(value: Any, event_date: str) -> bool:
    stamp = _parse_datetime(value)
    if stamp is None:
        return False
    return stamp.astimezone(ET).date().isoformat() == event_date


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _format_utc(value: Any) -> str:
    dt = _parse_datetime(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt is not None else ""


def _normalize_sgo_bookmaker(value: str) -> str:
    return normalize_player_name(value).replace("_", "").lower()


def _discover_odds_api_io_mlb_league_slug(config: MarketCollectorConfig) -> str | None:
    leagues = _http_get_json(
        f"{ODDS_API_IO_BASE}/leagues",
        params={
            "apiKey": config.api_key,
            "sport": "baseball",
            "all": "true",
        },
        timeout_sec=config.request_timeout_sec,
    )
    candidates: list[tuple[int, str]] = []
    for league in leagues if isinstance(leagues, list) else []:
        slug = str((league or {}).get("slug", "")).strip()
        name = str((league or {}).get("name", "")).strip()
        score = _mlb_league_score(slug=slug, name=name)
        if score > 0 and slug:
            candidates.append((score, slug))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_covers_matchup_links(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if "/sport/baseball/mlb/matchup/" not in href or "#props" not in href:
            continue
        game_id = _extract_covers_game_id(href)
        if not game_id or game_id in seen:
            continue
        seen.add(game_id)
        out.append((game_id, _absolute_covers_url(href)))
    return out


def _extract_covers_available_markets(html: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: set[str] = set()
    for item in soup.select("#prop-events-list li[data-event-name]"):
        value = str(item.get("data-event-name", "")).strip()
        if value:
            out.add(value)
    return out


def _extract_covers_articles(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    articles: list[dict[str, Any]] = []
    for article in soup.find_all("article", class_="player-prop-article"):
        heading = article.find("h2")
        if heading is None:
            continue
        title = heading.get_text(" ", strip=True)
        market_label = _extract_covers_market_label(title)
        if not market_label:
            continue
        player = _extract_covers_player_name(article=article, title=title, market_label=market_label)
        away_team, home_team = _extract_covers_matchup_teams(article)
        base_line = _extract_covers_prop_line(article)
        books = _extract_covers_book_rows(article, fallback_line=base_line)
        if not player or not books:
            continue
        articles.append(
            {
                "player": player,
                "market_label": market_label,
                "line": base_line,
                "away_team": away_team,
                "home_team": home_team,
                "books": books,
            }
        )
    return articles


def _extract_covers_book_rows(article: Any, *, fallback_line: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    collapse = article.find("div", class_="collapse")
    if collapse is None:
        return rows
    for row in collapse.find_all("div", class_="other-odds-row"):
        bookmaker = _extract_covers_bookmaker_name(row)
        over_line, over_price = _parse_line_and_price(_text_or_empty(row.find("div", class_="other-over-odds")))
        under_line, under_price = _parse_line_and_price(_text_or_empty(row.find("div", class_="other-under-odds")))
        row_line = over_line if over_line is not None else under_line if under_line is not None else fallback_line
        if bookmaker and (over_price is not None or under_price is not None):
            rows.append(
                {
                    "bookmaker": bookmaker,
                    "line": row_line,
                    "over_price": over_price,
                    "under_price": under_price,
                }
            )
    return rows


def _extract_covers_bookmaker_name(row: Any) -> str:
    figure = row.find("figure")
    if figure is not None:
        caption = figure.find("figcaption")
        if caption is not None:
            text = caption.get_text(" ", strip=True)
            if text:
                return _normalize_covers_bookmaker(text)
    image = row.find("img")
    if image is not None:
        src = str(image.get("src", "")).strip().lower()
        stem = Path(src.split("?")[0]).stem
        if stem:
            return _normalize_covers_bookmaker(stem)
    return ""


def _normalize_covers_bookmaker(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "-").replace("_", "-")
    return COVERS_BOOKMAKER_MAP.get(key, str(value).strip())


def _extract_covers_market_label(title: str) -> str:
    for label in COVERS_EVENT_LABEL_MAP.values():
        if f" {label} Props" in title:
            return label
    return ""


def _extract_covers_player_name(*, article: Any, title: str, market_label: str) -> str:
    modal_name = article.select_one("dialog h3")
    if modal_name is not None:
        text = modal_name.get_text(" ", strip=True)
        if text:
            return text
    picture = article.find("img", alt=True)
    if picture is not None:
        alt = str(picture.get("alt", "")).strip()
        if alt and alt.lower() not in {"draftkings", "fanduel", "caesars"}:
            return alt
    marker = f" {market_label} Props"
    if marker in title:
        return title.split(marker, 1)[0].strip()
    return ""


def _extract_covers_matchup_teams(article: Any) -> tuple[str, str]:
    away = ""
    home = ""
    away_node = article.find("span", class_="away-shortname")
    home_node = article.find("span", class_="home-shortname")
    if away_node is not None:
        away = away_node.get_text(" ", strip=True)
    if home_node is not None:
        home = home_node.get_text(" ", strip=True)
    return away, home


def _extract_covers_prop_line(article: Any) -> float | None:
    event_div = article.find("div", class_="player-event")
    if event_div is None:
        return None
    parent_text = event_div.parent.get_text(" ", strip=True)
    match = _first_number_match(parent_text)
    return float(match) if match is not None else None


def _extract_covers_game_id(href: str) -> str:
    parts = str(href).split("/matchup/", 1)
    if len(parts) != 2:
        return ""
    tail = parts[1]
    return tail.split("/", 1)[0].strip()


def _absolute_covers_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return f"https://www.covers.com{href}"


def _odds_api_event_is_mlb(event: dict[str, Any]) -> bool:
    league = event.get("league") or {}
    slug = str(league.get("slug", "")).strip()
    name = str(league.get("name", "")).strip()
    return _mlb_league_score(slug=slug, name=name) > 0


def _mlb_league_score(*, slug: str, name: str) -> int:
    slug_text = str(slug).strip().lower()
    name_text = str(name).strip().lower()
    score = 0
    if slug_text == "mlb":
        score += 100
    if "major league baseball" in name_text:
        score += 100
    if "major-league-baseball" in slug_text:
        score += 90
    if slug_text.endswith("-mlb") or slug_text.startswith("mlb-") or "mlb" in slug_text:
        score += 40
    if "major league" in name_text and "baseball" in name_text:
        score += 40
    if "baseball" in slug_text:
        score += 10
    return score


def _map_odds_api_market_name(name: Any, *, targets: set[str]) -> str | None:
    market_name = str(name or "").strip().lower()
    for target, config in TARGET_CONFIG.items():
        if target not in targets:
            continue
        if any(alias in market_name for alias in config.get("odds_api_aliases", ())):
            return target
    return None


def _resolve_sgo_player_name(player_data: Any, *, player_id: str) -> str:
    if isinstance(player_data, dict):
        name = str(player_data.get("name", "")).strip()
        if name:
            return name
        first = str(player_data.get("firstName", "")).strip()
        last = str(player_data.get("lastName", "")).strip()
        combined = " ".join(part for part in (first, last) if part)
        if combined:
            return combined
    return player_id.replace("_", " ").replace("  ", " ").title().strip()


def _pair_identifier(*, odd_id: str, opposing_odd_id: Any) -> str:
    other = str(opposing_odd_id or "").strip()
    parts = sorted([part for part in (odd_id, other) if part])
    return "|".join(parts) if parts else odd_id


def _resolve_side(*, side: Any, fallback: Any, odd_id: str) -> str | None:
    for value in (side, fallback, odd_id):
        text = str(value or "").strip().lower()
        if not text:
            continue
        if "over" in text:
            return "over"
        if "under" in text:
            return "under"
    return None


def _safe_team_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("long", "medium", "short"):
            text = str(value.get(key, "")).strip()
            if text:
                return text
    return ""


def _normalize_book_name(value: str) -> str:
    return normalize_player_name(value).lower()


def _text_or_empty(node: Any) -> str:
    if node is None:
        return ""
    return node.get_text(" ", strip=True)


def _first_number_match(text: str) -> float | None:
    import re

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(text))
    if not match:
        return None
    return _to_float(match.group(1))


def _parse_line_and_price(text: str) -> tuple[float | None, float | None]:
    import re

    clean = str(text).strip()
    if not clean:
        return None, None
    line_match = re.search(r"[ou]([0-9]+(?:\.[0-9]+)?)", clean.lower())
    price_match = re.search(r"([+-]\d+)", clean)
    line = _to_float(line_match.group(1)) if line_match else None
    price = _to_float(price_match.group(1)) if price_match else None
    return line, price


def _decimal_to_american(value: Any) -> float:
    decimal = _to_float(value)
    if decimal is None or decimal <= 1.0:
        return math.nan
    if decimal >= 2.0:
        return float(round((decimal - 1.0) * 100.0))
    return float(round(-100.0 / (decimal - 1.0)))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _chunk(items: list[Any], *, size: int) -> list[list[Any]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _series_mean(series: pd.Series | Any) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def _series_std(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    if len(series) == 1:
        return 0.0
    return float(series.std(ddof=0))


def _latest_timestamp(series: pd.Series) -> str:
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if parsed.empty:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return parsed.max().strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_non_empty(series: pd.Series) -> str:
    for value in series.astype(str):
        text = value.strip()
        if text and text.lower() != "nan":
            return text
    return ""
