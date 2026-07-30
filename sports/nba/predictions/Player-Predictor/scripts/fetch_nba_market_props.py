#!/usr/bin/env python3
"""
Fetch and normalize NBA player prop markets from a provider into a stable contract.

This script is intentionally standalone and optional:
- it never changes model artifacts
- it caches raw payloads for debugging
- it writes normalized long + wide snapshots that other scripts can consume

Current providers:
- rotowire: scrape the public same-day RotoWire multi-book props board
- snapshot: ingest an already-fetched CSV/parquet snapshot and normalize it
- odds_api: optional legacy fallback, disabled unless explicitly allowed
- sportsgameodds: live SportsGameOdds v2 events snapshot

This keeps the rest of the pipeline provider-agnostic while you evaluate or
swap market sources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba"
SPORT_KEY = "basketball_nba"
DEFAULT_MARKETS = ["player_points", "player_rebounds", "player_assists"]
DEFAULT_BOOKMAKERS = ["draftkings", "fanduel"]
EASTERN_TZ = "America/New_York"
ROTOWIRE_URL = "https://www.rotowire.com/betting/nba/player-props.php"
ROTOWIRE_PROP_MAP = {
    "pts": "player_points",
    "reb": "player_rebounds",
    "ast": "player_assists",
}
ROTOWIRE_BOOK_TITLES = {
    "betrivers": "BetRivers",
    "caesars": "Caesars",
    "draftkings": "DraftKings",
    "fanatics": "Fanatics",
    "fanduel": "FanDuel",
    "hardrock": "Hard Rock",
    "mgm": "BetMGM",
    "thescore": "theScore",
}
MARKET_WIDE_COLUMNS = [
    "Market_Date",
    "Player",
    "Market_Player_Raw",
    "Market_Event_ID",
    "Market_Commence_Time_UTC",
    "Market_Home_Team",
    "Market_Away_Team",
    "Market_Provider",
    "Market_Book",
    "Market_Price_Source",
    "Market_Price_Source_Type",
    "Market_Snapshot_ID",
    "Market_PTS",
    "Market_TRB",
    "Market_AST",
    "Market_PTS_books",
    "Market_TRB_books",
    "Market_AST_books",
    "Market_PTS_over_price",
    "Market_TRB_over_price",
    "Market_AST_over_price",
    "Market_PTS_under_price",
    "Market_TRB_under_price",
    "Market_AST_under_price",
    "Market_PTS_line_std",
    "Market_TRB_line_std",
    "Market_AST_line_std",
    "Market_Fetched_At_UTC",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def derive_snapshot_id(*, provider: str, fetched_at_utc: str, fallback_label: str = "") -> str:
    provider_text = str(provider or "").strip() or "unknown_provider"
    fetched_text = str(fetched_at_utc or "").strip()
    if fetched_text:
        return f"{provider_text}:{fetched_text}"
    fallback_text = str(fallback_label or "").strip()
    if fallback_text:
        return f"{provider_text}:{fallback_text}"
    return provider_text


def normalize_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch normalized NBA player prop lines.")
    parser.add_argument(
        "--provider",
        type=str,
        default="rotowire",
        choices=["rotowire", "odds_api", "snapshot", "sportsgameodds"],
        help="Market data provider. RotoWire scrapes the public same-day multi-book board without an API key.",
    )
    parser.add_argument(
        "--allow-odds-api",
        action="store_true",
        help="Explicitly allow live Odds API access for NBA. Default behavior keeps NBA off the API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Odds API key. Defaults to SPORTSGAMEODDS_API_KEY / THE_ODDS_API_KEY / ODDS_API_KEY.",
    )
    parser.add_argument("--input-path", type=Path, default=None, help="Input CSV/parquet for --provider snapshot.")
    parser.add_argument("--regions", type=str, default="us", help="API regions parameter.")
    parser.add_argument("--markets", type=str, default=",".join(DEFAULT_MARKETS), help="Comma-separated market keys.")
    parser.add_argument("--bookmakers", type=str, default=",".join(DEFAULT_BOOKMAKERS), help="Comma-separated bookmakers.")
    parser.add_argument("--odds-format", type=str, default="american", choices=["american", "decimal"], help="Odds format.")
    parser.add_argument("--date-format", type=str, default="iso", choices=["iso", "unix"], help="Date format.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for raw + normalized snapshots.")
    parser.add_argument("--event-limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Cooldown between event calls.")
    parser.add_argument("--event-date", type=str, default=None, help="Required RotoWire board date in YYYY-MM-DD format.")
    parser.add_argument("--page-url", type=str, default=ROTOWIRE_URL, help="RotoWire NBA player props page URL.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="RotoWire page request timeout.")
    return parser.parse_args()


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        text = str(value or "").strip()
        lowered = text.lower()
        if text and "paste-your" not in lowered and "your_api_key" not in lowered:
            return text
    return None


def _load_api_key_from_yaml(path: Path) -> str | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    odds_api = payload.get("odds_api")
    secrets = payload.get("secrets")
    return _first_non_empty(
        payload.get("SPORTSGAMEODDS_API_KEY"),
        payload.get("ODDS_API_KEY"),
        payload.get("THE_ODDS_API_KEY"),
        odds_api.get("sportsgameodds_api_key") if isinstance(odds_api, dict) else None,
        odds_api.get("sportsgameodds_key") if isinstance(odds_api, dict) else None,
        odds_api.get("api_key") if isinstance(odds_api, dict) else None,
        odds_api.get("odds_api_key") if isinstance(odds_api, dict) else None,
        secrets.get("SPORTSGAMEODDS_API_KEY") if isinstance(secrets, dict) else None,
        secrets.get("ODDS_API_KEY") if isinstance(secrets, dict) else None,
        secrets.get("THE_ODDS_API_KEY") if isinstance(secrets, dict) else None,
        secrets.get("sportsgameodds_api_key") if isinstance(secrets, dict) else None,
        secrets.get("odds_api_key") if isinstance(secrets, dict) else None,
    )


def _load_api_key_from_dotenv(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in {"SPORTSGAMEODDS_API_KEY", "ODDS_API_KEY", "THE_ODDS_API_KEY"}:
            continue
        cleaned = value.strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return None


def _load_api_key_from_local_files(start_path: Path) -> str | None:
    candidate_names = ("config.local.yaml", ".env.local", ".env", "config.yaml")
    checked: set[Path] = set()
    for base in [start_path, *start_path.parents]:
        for name in candidate_names:
            candidate = (base / name).resolve()
            if candidate in checked or not candidate.exists():
                continue
            checked.add(candidate)
            if candidate.suffix.lower() in {".yaml", ".yml"}:
                value = _load_api_key_from_yaml(candidate)
            else:
                value = _load_api_key_from_dotenv(candidate)
            if value:
                return value
    return None


def resolve_api_key(explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key
    for key in ("SPORTSGAMEODDS_API_KEY", "THE_ODDS_API_KEY", "ODDS_API_KEY"):
        value = os.getenv(key)
        if value:
            return value
    local_value = _load_api_key_from_local_files(Path(__file__).resolve().parent)
    if local_value:
        return local_value
    raise RuntimeError("Missing Odds API key. Set SPORTSGAMEODDS_API_KEY, THE_ODDS_API_KEY, create config.local.yaml, or pass --api-key.")


def request_json(base_url: str, params: dict[str, object]) -> tuple[object, dict[str, str]]:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{base_url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Player-Predictor/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {key.lower(): value for key, value in response.headers.items()}
            return payload, headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Odds API request failed [{exc.code}] {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Odds API network failure for {url}: {exc}") from exc


def safe_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _event_date_et(commence_time_value: str) -> str | None:
    if not commence_time_value:
        return None
    parsed = pd.to_datetime(commence_time_value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return str(parsed.tz_convert(EASTERN_TZ).date())


def _collapse_market_outcomes(
    *,
    event: dict,
    bookmaker: dict,
    market: dict,
    fetched_at_utc: str,
) -> list[dict]:
    grouped: dict[tuple[str, float | None], dict] = {}
    for outcome in market.get("outcomes", []):
        player_name = outcome.get("description") or outcome.get("participant")
        if not player_name:
            continue
        point = outcome.get("point")
        point_value = float(point) if point is not None else None
        group_key = (str(player_name), point_value)
        row = grouped.setdefault(
            group_key,
            {
                "fetched_at_utc": fetched_at_utc,
                "event_id": event.get("id"),
                "commence_time_utc": event.get("commence_time"),
                "event_date_et": _event_date_et(event.get("commence_time")),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "bookmaker_key": bookmaker.get("key"),
                "bookmaker_title": bookmaker.get("title"),
                "market_key": market.get("key"),
                "player_name_raw": str(player_name),
                "player_name_norm": normalize_name(str(player_name)),
                "line": point_value,
                "over_price": np.nan,
                "under_price": np.nan,
            },
        )
        name = str(outcome.get("name", "")).strip().lower()
        price = outcome.get("price")
        if name == "over":
            row["over_price"] = float(price) if price is not None else np.nan
        elif name == "under":
            row["under_price"] = float(price) if price is not None else np.nan
    return list(grouped.values())


def _wide_from_long_props(
    long_df: pd.DataFrame,
    fetched_at_utc: str,
    *,
    provider: str,
    price_source: str,
    market_book: str = "aggregate_market_snapshot",
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    for col in ["line", "over_price", "under_price"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    consensus = (
        long_df.groupby(["event_date_et", "player_name_norm", "player_name_raw", "market_key"], dropna=False)
        .agg(
            market_line=("line", "median"),
            market_line_std=("line", "std"),
            over_price_avg=("over_price", "mean"),
            under_price_avg=("under_price", "mean"),
            book_count=("bookmaker_key", "nunique"),
            event_count=("event_id", "nunique"),
            first_event_id=("event_id", "min"),
            first_commence_time_utc=("commence_time_utc", "min"),
            first_home_team=("home_team", "min"),
            first_away_team=("away_team", "min"),
        )
        .reset_index()
    )
    consensus["market_line_std"] = consensus["market_line_std"].fillna(0.0)

    value_map = {
        "player_points": "Market_PTS",
        "player_rebounds": "Market_TRB",
        "player_assists": "Market_AST",
    }
    books_map = {
        "player_points": "Market_PTS_books",
        "player_rebounds": "Market_TRB_books",
        "player_assists": "Market_AST_books",
    }
    over_map = {
        "player_points": "Market_PTS_over_price",
        "player_rebounds": "Market_TRB_over_price",
        "player_assists": "Market_AST_over_price",
    }
    under_map = {
        "player_points": "Market_PTS_under_price",
        "player_rebounds": "Market_TRB_under_price",
        "player_assists": "Market_AST_under_price",
    }
    spread_map = {
        "player_points": "Market_PTS_line_std",
        "player_rebounds": "Market_TRB_line_std",
        "player_assists": "Market_AST_line_std",
    }

    def _pivot(metric_col: str, rename_map: dict[str, str]) -> pd.DataFrame:
        wide = (
            consensus.pivot_table(
                index=["event_date_et", "player_name_norm", "player_name_raw"],
                columns="market_key",
                values=metric_col,
                aggfunc="first",
            )
            .rename(columns=rename_map)
            .reset_index()
        )
        wide.columns.name = None
        return wide

    metadata = consensus[
        [
            "event_date_et",
            "player_name_norm",
            "player_name_raw",
            "first_event_id",
            "first_commence_time_utc",
            "first_home_team",
            "first_away_team",
        ]
    ].drop_duplicates(subset=["event_date_et", "player_name_norm", "player_name_raw"], keep="last")

    wide = metadata.merge(
        _pivot("market_line", value_map),
        how="left",
        on=["event_date_et", "player_name_norm", "player_name_raw"],
    )
    for metric_col, rename_map in [
        ("book_count", books_map),
        ("over_price_avg", over_map),
        ("under_price_avg", under_map),
        ("market_line_std", spread_map),
    ]:
        wide = wide.merge(
            _pivot(metric_col, rename_map),
            how="left",
            on=["event_date_et", "player_name_norm", "player_name_raw"],
        )

    wide = wide.rename(
        columns={
            "event_date_et": "Market_Date",
            "player_name_norm": "Player",
            "player_name_raw": "Market_Player_Raw",
            "first_event_id": "Market_Event_ID",
            "first_commence_time_utc": "Market_Commence_Time_UTC",
            "first_home_team": "Market_Home_Team",
            "first_away_team": "Market_Away_Team",
        }
    )
    wide["Market_Fetched_At_UTC"] = fetched_at_utc
    wide["Market_Provider"] = provider
    wide["Market_Book"] = market_book
    wide["Market_Price_Source"] = price_source
    wide["Market_Price_Source_Type"] = "LIVE_ENTRY"
    wide["Market_Snapshot_ID"] = derive_snapshot_id(
        provider=provider,
        fetched_at_utc=fetched_at_utc,
        fallback_label="latest_player_props_wide",
    )
    return wide


def _extract_data_array_literal(script_text: str) -> str:
    data_idx = script_text.find("data:")
    if data_idx < 0:
        raise ValueError("settings.data array not found in script block")
    start = script_text.find("[", data_idx)
    if start < 0:
        raise ValueError("settings.data opening '[' not found")

    depth = 0
    in_string = False
    string_char = ""
    escaped = False
    for pos in range(start, len(script_text)):
        char = script_text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_char:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            string_char = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return script_text[start : pos + 1]
    raise ValueError("Unterminated settings.data array in RotoWire script block")


def extract_rotowire_page_payload(html: str) -> tuple[str, dict[str, list[dict[str, object]]]]:
    date_match = re.search(r'const dayNBA\s*=\s*"([0-9]{4}-[0-9]{2}-[0-9]{2})"', html)
    if not date_match:
        raise RuntimeError("Unable to locate the RotoWire NBA board date. The page may not have an active slate.")
    page_date = str(date_match.group(1))

    bundles: dict[str, list[dict[str, object]]] = {}
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL)
    for script_text in scripts:
        if "const settings" not in script_text or "data:" not in script_text:
            continue
        prop_match = re.search(r'const prop\s*=\s*"([a-z]+)"', script_text)
        if not prop_match:
            continue
        prop = str(prop_match.group(1)).strip().lower()
        if prop not in ROTOWIRE_PROP_MAP:
            continue
        rows = json.loads(_extract_data_array_literal(script_text))
        if isinstance(rows, list) and rows:
            bundles[prop] = [row for row in rows if isinstance(row, dict)]

    if not bundles:
        raise RuntimeError("No supported NBA prop bundles were found on the RotoWire page.")
    return page_date, bundles


def _rotowire_book_keys(row: dict[str, object], prop: str) -> list[str]:
    suffixes = (f"_{prop}", f"_{prop}Over", f"_{prop}Under")
    books: set[str] = set()
    for key in row:
        for suffix in suffixes:
            if key.endswith(suffix):
                books.add(key[: -len(suffix)])
                break
    return sorted(books)


def _rotowire_matchup(row: dict[str, object]) -> tuple[str | None, str | None]:
    team = str(row.get("team") or "").strip().upper()
    opponent = str(row.get("opp") or "").strip().upper()
    if not team:
        return None, opponent.removeprefix("@") or None
    if opponent.startswith("@"):
        return opponent[1:] or None, team
    return team, opponent or None


def _numeric_value(value: object) -> float:
    text = str(value or "").strip().replace("+", "")
    if not text:
        return float("nan")
    parsed = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float("nan")


def build_rotowire_frames(
    *,
    market_date: str,
    bundles: dict[str, list[dict[str, object]]],
    fetched_at_utc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []
    for prop, market_key in ROTOWIRE_PROP_MAP.items():
        for row in bundles.get(prop, []):
            player_raw = str(row.get("name") or "").strip()
            player_norm = normalize_name(player_raw)
            if not player_raw or not player_norm:
                continue
            home_team, away_team = _rotowire_matchup(row)
            for bookmaker_key in _rotowire_book_keys(row, prop):
                line = _numeric_value(row.get(f"{bookmaker_key}_{prop}"))
                if not np.isfinite(line):
                    continue
                long_rows.append(
                    {
                        "fetched_at_utc": fetched_at_utc,
                        "event_id": str(row.get("gameID") or ""),
                        "commence_time_utc": pd.NaT,
                        "event_date_et": market_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker_key": bookmaker_key,
                        "bookmaker_title": ROTOWIRE_BOOK_TITLES.get(bookmaker_key, bookmaker_key.title()),
                        "market_key": market_key,
                        "player_name_raw": player_raw,
                        "player_name_norm": player_norm,
                        "line": line,
                        "over_price": _numeric_value(row.get(f"{bookmaker_key}_{prop}Over")),
                        "under_price": _numeric_value(row.get(f"{bookmaker_key}_{prop}Under")),
                    }
                )

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame(columns=MARKET_WIDE_COLUMNS)
    wide_df = _wide_from_long_props(
        long_df,
        fetched_at_utc,
        provider="rotowire",
        price_source="rotowire_embedded_multi_book",
        market_book="rotowire_consensus",
    )
    for column in MARKET_WIDE_COLUMNS:
        if column not in wide_df.columns:
            wide_df[column] = pd.NA
    return long_df, wide_df[MARKET_WIDE_COLUMNS].copy()


def _rotowire_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/134.0 Safari/537.36"
            ),
        }
    )
    return session


def fetch_from_rotowire(
    args: argparse.Namespace,
    fetched_at_utc: str,
) -> tuple[list[dict], dict[str, dict], pd.DataFrame, pd.DataFrame, dict]:
    requested_date = str(args.event_date or "").strip() or None
    if requested_date:
        datetime.fromisoformat(requested_date)

    with _rotowire_session() as session:
        response = session.get(args.page_url, timeout=float(args.timeout_seconds))
        response.raise_for_status()
        html = response.text

    page_date, bundles = extract_rotowire_page_payload(html)
    if requested_date and page_date != requested_date:
        raise RuntimeError(
            f"RotoWire NBA board date mismatch: requested {requested_date}, page contains {page_date}."
        )
    long_df, wide_df = build_rotowire_frames(
        market_date=page_date,
        bundles=bundles,
        fetched_at_utc=fetched_at_utc,
    )
    if long_df.empty or wide_df.empty:
        raise RuntimeError(f"RotoWire NBA board for {page_date} did not contain usable PTS/TRB/AST lines.")

    events = [
        {
            "event_id": str(event_id),
            "market_date": page_date,
        }
        for event_id in sorted(long_df["event_id"].dropna().astype(str).unique())
    ]
    manifest = {
        "provider": "rotowire",
        "fetched_at_utc": fetched_at_utc,
        "source_url": str(args.page_url),
        "page_date": page_date,
        "event_date_requested": requested_date,
        "bundle_kinds": sorted(bundles),
        "bundle_rows": {prop: int(len(rows)) for prop, rows in bundles.items()},
        "bookmakers": sorted(long_df["bookmaker_key"].dropna().astype(str).unique()),
        "event_count_requested": int(len(events)),
        "event_count_fetched": int(len(events)),
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "errors": [],
    }
    return events, bundles, long_df, wide_df, manifest


def normalize_event_odds(events: list[dict], event_odds: dict[str, dict], fetched_at_utc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    event_by_id = {str(event.get("id")): event for event in events}
    for event_id, payload in event_odds.items():
        event = event_by_id.get(str(event_id), payload)
        bookmakers = payload.get("bookmakers", [])
        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                rows.extend(
                    _collapse_market_outcomes(
                        event=event,
                        bookmaker=bookmaker,
                        market=market,
                        fetched_at_utc=fetched_at_utc,
                    )
                )

    long_df = pd.DataFrame(rows)
    wide = _wide_from_long_props(
        long_df,
        fetched_at_utc,
        provider="odds_api",
        price_source="odds_api_market_snapshot",
    )
    return long_df, wide


def normalize_wide_snapshot(df: pd.DataFrame, fetched_at_utc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    out = df.copy()
    rename_map = {}
    if "event_date_et" in out.columns and "Market_Date" not in out.columns:
        rename_map["event_date_et"] = "Market_Date"
    if "player_name_norm" in out.columns and "Player" not in out.columns:
        rename_map["player_name_norm"] = "Player"
    if "player_name_raw" in out.columns and "Market_Player_Raw" not in out.columns:
        rename_map["player_name_raw"] = "Market_Player_Raw"
    if rename_map:
        out = out.rename(columns=rename_map)

    if "Player" not in out.columns:
        raise ValueError("Snapshot provider requires a Player column or player_name_norm.")
    if "Market_Date" not in out.columns:
        raise ValueError("Snapshot provider requires a Market_Date column or event_date_et.")

    out["Player"] = out["Player"].astype(str).map(normalize_name)
    out["Market_Date"] = pd.to_datetime(out["Market_Date"], errors="coerce").dt.date.astype(str)
    if "Market_Player_Raw" not in out.columns:
        out["Market_Player_Raw"] = out["Player"]
    for market_col in ["Market_Event_ID", "Market_Commence_Time_UTC", "Market_Home_Team", "Market_Away_Team"]:
        if market_col not in out.columns:
            out[market_col] = pd.NA
    if "Market_Fetched_At_UTC" not in out.columns:
        out["Market_Fetched_At_UTC"] = fetched_at_utc
    if "Market_Provider" not in out.columns:
        out["Market_Provider"] = "snapshot"
    if "Market_Book" not in out.columns:
        out["Market_Book"] = "aggregate_market_snapshot"
    if "Market_Price_Source" not in out.columns:
        out["Market_Price_Source"] = "snapshot_input"
    if "Market_Price_Source_Type" not in out.columns:
        out["Market_Price_Source_Type"] = np.where(
            pd.Series(out["Market_Fetched_At_UTC"], index=out.index).fillna("").astype(str).str.strip().ne(""),
            "ARCHIVED_ENTRY",
            "UNKNOWN",
        )
    if "Market_Snapshot_ID" not in out.columns:
        provider_series = out["Market_Provider"].fillna("snapshot").astype(str)
        fetched_series = out["Market_Fetched_At_UTC"].fillna("").astype(str)
        event_series = out.get("Market_Event_ID", pd.Series("", index=out.index)).fillna("").astype(str)
        out["Market_Snapshot_ID"] = [
            derive_snapshot_id(provider=provider, fetched_at_utc=fetched_at, fallback_label=event_id)
            for provider, fetched_at, event_id in zip(provider_series, fetched_series, event_series)
        ]

    for col in MARKET_WIDE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    numeric_cols = [
        col
        for col in MARKET_WIDE_COLUMNS
        if col.startswith("Market_")
        and col
        not in {
            "Market_Date",
            "Market_Player_Raw",
            "Market_Event_ID",
            "Market_Commence_Time_UTC",
            "Market_Home_Team",
            "Market_Away_Team",
            "Market_Fetched_At_UTC",
            "Market_Provider",
            "Market_Book",
            "Market_Price_Source",
            "Market_Price_Source_Type",
            "Market_Snapshot_ID",
        }
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    wide_df = out[MARKET_WIDE_COLUMNS].drop_duplicates(subset=["Market_Date", "Player"], keep="last").copy()

    long_rows = []
    target_map = {
        "PTS": "player_points",
        "TRB": "player_rebounds",
        "AST": "player_assists",
    }
    for _, row in wide_df.iterrows():
        for short_target, market_key in target_map.items():
            line = row.get(f"Market_{short_target}")
            if pd.isna(line):
                continue
            long_rows.append(
                {
                    "fetched_at_utc": row.get("Market_Fetched_At_UTC", fetched_at_utc),
                    "event_id": row.get("Market_Event_ID", np.nan),
                    "commence_time_utc": row.get("Market_Commence_Time_UTC", np.nan),
                    "event_date_et": row["Market_Date"],
                    "home_team": row.get("Market_Home_Team", np.nan),
                    "away_team": row.get("Market_Away_Team", np.nan),
                    "bookmaker_key": np.nan,
                    "bookmaker_title": np.nan,
                    "market_key": market_key,
                    "player_name_raw": row.get("Market_Player_Raw", row["Player"]),
                    "player_name_norm": row["Player"],
                    "line": float(line),
                    "over_price": row.get(f"Market_{short_target}_over_price", np.nan),
                    "under_price": row.get(f"Market_{short_target}_under_price", np.nan),
                }
            )
    long_df = pd.DataFrame(long_rows)
    return long_df, wide_df


SGO_MARKET_MAP = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
}


def _sgo_team_abbr(event: dict, side: str) -> str | None:
    teams = event.get("teams") if isinstance(event.get("teams"), dict) else {}
    team = teams.get(side) if isinstance(teams, dict) else None
    if not isinstance(team, dict):
        return None
    names = team.get("names") if isinstance(team.get("names"), dict) else {}
    for key in ("short", "medium", "long"):
        value = names.get(key)
        if value:
            return str(value)
    return str(team.get("teamID")) if team.get("teamID") else None


def _sgo_event_start(event: dict) -> str | None:
    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    for value in [
        status.get("startsAt") if isinstance(status, dict) else None,
        event.get("startsAt"),
        event.get("startTime"),
        event.get("commence_time"),
    ]:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _sgo_event_date_et(event: dict) -> str | None:
    starts_at = _sgo_event_start(event)
    return _event_date_et(starts_at or "")


def _sgo_player_name(event: dict, odd: dict) -> str | None:
    player_id = str(odd.get("playerID") or odd.get("statEntityID") or "").strip()
    players = event.get("players") if isinstance(event.get("players"), dict) else {}
    player = players.get(player_id) if isinstance(players, dict) else None
    if isinstance(player, dict):
        for key in ("name", "fullName", "displayName"):
            if player.get(key):
                return str(player[key])
        first = str(player.get("firstName") or "").strip()
        last = str(player.get("lastName") or "").strip()
        if first or last:
            return f"{first} {last}".strip()
    market_name = str(odd.get("marketName") or "").strip()
    stat = str(odd.get("statID") or "").strip().lower()
    if market_name and stat:
        marker = f" {stat.replace('_', ' ').title()} "
        if marker in market_name:
            return market_name.split(marker, 1)[0].strip()
    return player_id or None


def _sgo_float(value: object) -> float:
    text = str(value or "").strip().replace("+", "")
    if not text:
        return np.nan
    return float(pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0])


def _sgo_book_rows(event: dict, odd: dict, fetched_at_utc: str) -> list[dict]:
    stat_id = str(odd.get("statID") or "").strip().lower()
    market_key = SGO_MARKET_MAP.get(stat_id)
    if market_key is None:
        return []
    if str(odd.get("periodID") or "").strip().lower() != "game":
        return []
    if str(odd.get("betTypeID") or "").strip().lower() != "ou":
        return []
    side = str(odd.get("sideID") or "").strip().lower()
    if side not in {"over", "under"}:
        return []
    player_name = _sgo_player_name(event, odd)
    if not player_name:
        return []
    base = {
        "fetched_at_utc": fetched_at_utc,
        "event_id": event.get("eventID") or event.get("id"),
        "commence_time_utc": _sgo_event_start(event),
        "event_date_et": _sgo_event_date_et(event),
        "home_team": _sgo_team_abbr(event, "home"),
        "away_team": _sgo_team_abbr(event, "away"),
        "market_key": market_key,
        "player_name_raw": player_name,
        "player_name_norm": normalize_name(player_name),
        "line": _sgo_float(odd.get("bookOverUnder") or odd.get("fairOverUnder")),
        "over_price": np.nan,
        "under_price": np.nan,
    }

    rows: list[dict] = []
    by_book = odd.get("byBookmaker") if isinstance(odd.get("byBookmaker"), dict) else {}
    for bookmaker_key, book_odd in by_book.items():
        if not isinstance(book_odd, dict):
            continue
        if book_odd.get("available") is False:
            continue
        row = dict(base)
        row["bookmaker_key"] = str(bookmaker_key)
        row["bookmaker_title"] = str(bookmaker_key)
        row["line"] = _sgo_float(book_odd.get("overUnder") or base["line"])
        price = _sgo_float(book_odd.get("odds"))
        if side == "over":
            row["over_price"] = price
        else:
            row["under_price"] = price
        rows.append(row)

    if not rows:
        row = dict(base)
        row["bookmaker_key"] = "sportsgameodds_consensus"
        row["bookmaker_title"] = "SportsGameOdds Consensus"
        price = _sgo_float(odd.get("bookOdds") or odd.get("fairOdds"))
        if side == "over":
            row["over_price"] = price
        else:
            row["under_price"] = price
        rows.append(row)
    return rows


def normalize_sportsgameodds_events(events: list[dict], fetched_at_utc: str, bookmaker_filter: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for event in events:
        odds = event.get("odds") if isinstance(event.get("odds"), dict) else {}
        for odd in odds.values():
            if not isinstance(odd, dict):
                continue
            for row in _sgo_book_rows(event, odd, fetched_at_utc):
                if bookmaker_filter and str(row.get("bookmaker_key", "")).lower() not in bookmaker_filter:
                    continue
                rows.append(row)
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, pd.DataFrame()
    wide_df = _wide_from_long_props(
        long_df,
        fetched_at_utc,
        provider="sportsgameodds",
        price_source="sportsgameodds_events_v2",
    )
    return long_df, wide_df


def fetch_from_sportsgameodds(args: argparse.Namespace, fetched_at_utc: str) -> tuple[list[dict], dict[str, dict], pd.DataFrame, pd.DataFrame, dict]:
    api_key = resolve_api_key(args.api_key)
    base_url = "https://api.sportsgameodds.com/v2/events"
    events: list[dict] = []
    cursor: str | None = None
    errors: list[dict[str, str]] = []
    page_count = 0
    while True:
        params: dict[str, object] = {
            "leagueID": "NBA",
            "oddsAvailable": "true",
            "limit": 50,
        }
        if cursor:
            params["cursor"] = cursor
        try:
            payload, _headers = request_json(base_url, params | {"apiKey": api_key})
        except Exception as exc:
            errors.append({"event_id": "events", "error": str(exc)})
            break
        page_count += 1
        if not isinstance(payload, dict) or payload.get("success") is False:
            errors.append({"event_id": "events", "error": str(payload)})
            break
        data = payload.get("data", [])
        if not isinstance(data, list):
            errors.append({"event_id": "events", "error": f"unexpected data payload: {type(data)!r}"})
            break
        events.extend([event for event in data if isinstance(event, dict)])
        if args.event_limit is not None and len(events) >= int(args.event_limit):
            events = events[: int(args.event_limit)]
            break
        cursor = payload.get("nextCursor")
        if not cursor:
            break
        if page_count >= 10:
            errors.append({"event_id": "events", "error": "pagination stopped at safety limit 10"})
            break
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    bookmaker_filter = {item.strip().lower() for item in args.bookmakers.split(",") if item.strip()}
    long_df, wide_df = normalize_sportsgameodds_events(events, fetched_at_utc, bookmaker_filter=bookmaker_filter or None)
    manifest = {
        "provider": "sportsgameodds",
        "fetched_at_utc": fetched_at_utc,
        "sport": SPORT_KEY,
        "league_id": "NBA",
        "markets": ["player_points", "player_rebounds", "player_assists"],
        "bookmakers": sorted(bookmaker_filter),
        "event_count_requested": int(len(events)),
        "event_count_fetched": int(len(events)),
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "pages_fetched": int(page_count),
        "errors": errors,
    }
    return events, {}, long_df, wide_df, manifest


def fetch_from_odds_api(args: argparse.Namespace, fetched_at_utc: str) -> tuple[list[dict], dict[str, dict], pd.DataFrame, pd.DataFrame, dict]:
    api_key = resolve_api_key(args.api_key)
    markets = [item.strip() for item in args.markets.split(",") if item.strip()]
    bookmakers = [item.strip() for item in args.bookmakers.split(",") if item.strip()]
    events_url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events"
    odds_url_template = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{{event_id}}/odds"

    events, events_headers = request_json(
        events_url,
        {
            "apiKey": api_key,
            "dateFormat": args.date_format,
        },
    )
    if not isinstance(events, list):
        raise RuntimeError(f"Unexpected events payload: {type(events)!r}")
    if args.event_limit is not None:
        events = events[: args.event_limit]

    event_payloads: dict[str, dict] = {}
    errors: list[dict[str, str]] = []
    for idx, event in enumerate(events, start=1):
        event_id = str(event.get("id"))
        try:
            payload, _headers = request_json(
                odds_url_template.format(event_id=event_id),
                {
                    "apiKey": api_key,
                    "regions": args.regions,
                    "markets": ",".join(markets),
                    "bookmakers": ",".join(bookmakers),
                    "oddsFormat": args.odds_format,
                    "dateFormat": args.date_format,
                },
            )
            event_payloads[event_id] = payload
        except Exception as exc:
            errors.append({"event_id": event_id, "error": str(exc)})
        if idx < len(events) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    long_df, wide_df = normalize_event_odds(events, event_payloads, fetched_at_utc)
    manifest = {
        "provider": "odds_api",
        "fetched_at_utc": fetched_at_utc,
        "sport": SPORT_KEY,
        "markets": markets,
        "bookmakers": bookmakers,
        "regions": args.regions,
        "odds_format": args.odds_format,
        "event_count_requested": int(len(events)),
        "event_count_fetched": int(len(event_payloads)),
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "events_remaining_header": events_headers.get("x-requests-remaining"),
        "events_used_header": events_headers.get("x-requests-used"),
        "errors": errors,
    }
    return events, event_payloads, long_df, wide_df, manifest


def fetch_from_snapshot(args: argparse.Namespace, fetched_at_utc: str) -> tuple[list[dict], dict[str, dict], pd.DataFrame, pd.DataFrame, dict]:
    if args.input_path is None or not args.input_path.exists():
        raise RuntimeError("Snapshot provider requires --input-path pointing to an existing CSV/parquet file.")
    snapshot_df = load_table(args.input_path)
    long_df, wide_df = normalize_wide_snapshot(snapshot_df, fetched_at_utc)
    manifest = {
        "provider": "snapshot",
        "fetched_at_utc": fetched_at_utc,
        "input_path": str(args.input_path),
        "event_count_requested": 0,
        "event_count_fetched": 0,
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "errors": [],
    }
    return [], {}, long_df, wide_df, manifest


def write_snapshot(outdir: Path, stamp: str, events: list[dict], event_payloads: dict[str, dict], long_df: pd.DataFrame, wide_df: pd.DataFrame, manifest: dict) -> None:
    raw_dir = outdir / "raw" / stamp
    norm_dir = outdir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    safe_write_json(raw_dir / "events.json", events)
    safe_write_json(raw_dir / "event_odds.json", event_payloads)
    safe_write_json(raw_dir / "manifest.json", manifest)

    if not long_df.empty:
        long_df.to_parquet(norm_dir / f"player_props_long_{stamp}.parquet", index=False)
        long_df.to_csv(norm_dir / f"player_props_long_{stamp}.csv", index=False)
        long_df.to_parquet(outdir / "latest_player_props_long.parquet", index=False)
        long_df.to_csv(outdir / "latest_player_props_long.csv", index=False)

    if not wide_df.empty:
        wide_df.to_parquet(norm_dir / f"player_props_wide_{stamp}.parquet", index=False)
        wide_df.to_csv(norm_dir / f"player_props_wide_{stamp}.csv", index=False)
        wide_df.to_parquet(outdir / "latest_player_props_wide.parquet", index=False)
        wide_df.to_csv(outdir / "latest_player_props_wide.csv", index=False)

    safe_write_json(outdir / "latest_manifest.json", manifest)


def append_history(outdir: Path, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> dict:
    summary = {
        "history_long_rows": 0,
        "history_wide_rows": 0,
    }
    if not long_df.empty:
        history_long_path = outdir / "history_player_props_long.parquet"
        if history_long_path.exists():
            history_long = pd.read_parquet(history_long_path)
            combined_long = pd.concat([history_long, long_df], ignore_index=True)
        else:
            combined_long = long_df.copy()
        long_dedupe_cols = [
            "event_date_et",
            "player_name_norm",
            "market_key",
            "bookmaker_key",
            "line",
            "over_price",
            "under_price",
            "fetched_at_utc",
        ]
        combined_long = combined_long.drop_duplicates(subset=[col for col in long_dedupe_cols if col in combined_long.columns], keep="last")
        write_table(combined_long, history_long_path)
        write_table(combined_long, outdir / "history_player_props_long.csv")
        summary["history_long_rows"] = int(len(combined_long))

    if not wide_df.empty:
        history_wide_path = outdir / "history_player_props_wide.parquet"
        if history_wide_path.exists():
            history_wide = pd.read_parquet(history_wide_path)
            combined_wide = pd.concat([history_wide, wide_df], ignore_index=True)
        else:
            combined_wide = wide_df.copy()
        wide_dedupe_cols = [
            "Market_Date",
            "Player",
            "Market_Fetched_At_UTC",
        ]
        combined_wide = combined_wide.drop_duplicates(subset=[col for col in wide_dedupe_cols if col in combined_wide.columns], keep="last")
        write_table(combined_wide, history_wide_path)
        write_table(combined_wide, outdir / "history_player_props_wide.csv")
        summary["history_wide_rows"] = int(len(combined_wide))
    return summary


def main() -> None:
    args = parse_args()
    stamp = utc_compact_timestamp()
    fetched_at_utc = utc_now_iso()

    if args.provider == "rotowire":
        events, event_payloads, long_df, wide_df, manifest = fetch_from_rotowire(args, fetched_at_utc)
    elif args.provider == "odds_api":
        if not args.allow_odds_api:
            raise RuntimeError(
                "NBA Odds API access is disabled by default. "
                "Use local snapshot inputs, or pass both --provider odds_api and --allow-odds-api to override."
            )
        events, event_payloads, long_df, wide_df, manifest = fetch_from_odds_api(args, fetched_at_utc)
    elif args.provider == "sportsgameodds":
        events, event_payloads, long_df, wide_df, manifest = fetch_from_sportsgameodds(args, fetched_at_utc)
    elif args.provider == "snapshot":
        if args.input_path is None:
            raise RuntimeError(
                "NBA snapshot mode requires --input-path. "
                "This keeps NBA on local market lines instead of hitting the Odds API."
            )
        events, event_payloads, long_df, wide_df, manifest = fetch_from_snapshot(args, fetched_at_utc)
    else:
        raise RuntimeError(f"Unsupported provider: {args.provider}")

    write_snapshot(args.outdir, stamp, events, event_payloads, long_df, wide_df, manifest)
    history_summary = append_history(args.outdir, long_df, wide_df)
    manifest.update(history_summary)
    safe_write_json(args.outdir / "latest_manifest.json", manifest)

    print("\n" + "=" * 80)
    print("NBA MARKET PROPS FETCH COMPLETE")
    print("=" * 80)
    print(f"Provider:         {manifest.get('provider')}")
    print(f"Events requested: {len(events)}")
    print(f"Events fetched:   {len(event_payloads)}")
    print(f"Long rows:        {len(long_df)}")
    print(f"Wide rows:        {len(wide_df)}")
    print(f"History wide:     {history_summary['history_wide_rows']}")
    print(f"Errors:           {len(manifest.get('errors', []))}")
    print(f"Output:           {args.outdir}")
    errors = manifest.get("errors", [])
    if errors:
        print("Sample error:")
        print(f"  {errors[0]['event_id']}: {errors[0]['error']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
