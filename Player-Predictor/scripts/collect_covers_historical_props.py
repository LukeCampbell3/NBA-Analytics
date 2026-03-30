#!/usr/bin/env python3
"""
Collect historical NBA player props from Covers matchup pages and normalize them
into the shared market history contract.

This is intended to remove the manual snapshot step:
- crawl Covers matchup pages by numeric game id
- extract historical player points / rebounds / assists lines
- append to history_player_props_long/wide.(parquet|csv)

The collector is designed to work against the seasons already present in
Data-Proc, so we only crawl as far back as the local dataset requires.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_nba_market_props import (  # noqa: E402
    DEFAULT_OUTDIR,
    MARKET_WIDE_COLUMNS,
    append_history,
    normalize_name,
    safe_write_json,
    write_snapshot,
)


DATA_DIR = REPO_ROOT / "Data-Proc"
COVERS_ODDS_URL = "https://www.covers.com/sport/basketball/nba/odds"
COVERS_MATCHUP_URL = "https://www.covers.com/sport/basketball/nba/matchup/{game_id}/picks#props"
TARGET_MAP = {
    "POINTS SCORED": ("PTS", "player_points"),
    "TOTAL REBOUNDS": ("TRB", "player_rebounds"),
    "TOTAL ASSISTS": ("AST", "player_assists"),
}
DATE_RE = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})")
PREDICTION_RE = re.compile(r"\b([ou])\s*([0-9]+(?:\.[0-9]+)?)\b", re.I)
PRICE_RE = re.compile(r"([ou])\s*([0-9]+(?:\.[0-9]+)?)\s*([+-]\d+)", re.I)
TEAM_NAME_ALIASES = {
    "atl": "ATL",
    "hawks": "ATL",
    "atlanta": "ATL",
    "bos": "BOS",
    "celtics": "BOS",
    "boston": "BOS",
    "bk": "BKN",
    "bkn": "BKN",
    "nets": "BKN",
    "brooklyn": "BKN",
    "cha": "CHA",
    "hornets": "CHA",
    "charlotte": "CHA",
    "chi": "CHI",
    "bulls": "CHI",
    "chicago": "CHI",
    "cle": "CLE",
    "cavaliers": "CLE",
    "cavs": "CLE",
    "cleveland": "CLE",
    "dal": "DAL",
    "mavericks": "DAL",
    "mavs": "DAL",
    "dallas": "DAL",
    "den": "DEN",
    "nuggets": "DEN",
    "denver": "DEN",
    "det": "DET",
    "pistons": "DET",
    "detroit": "DET",
    "gsw": "GSW",
    "warriors": "GSW",
    "golden_state": "GSW",
    "golden state": "GSW",
    "hou": "HOU",
    "rockets": "HOU",
    "houston": "HOU",
    "ind": "IND",
    "pacers": "IND",
    "indiana": "IND",
    "lac": "LAC",
    "clippers": "LAC",
    "la_clippers": "LAC",
    "la clippers": "LAC",
    "los_angeles_clippers": "LAC",
    "los angeles clippers": "LAC",
    "lal": "LAL",
    "lakers": "LAL",
    "la_lakers": "LAL",
    "la lakers": "LAL",
    "los_angeles_lakers": "LAL",
    "los angeles lakers": "LAL",
    "mem": "MEM",
    "grizzlies": "MEM",
    "memphis": "MEM",
    "mia": "MIA",
    "heat": "MIA",
    "miami": "MIA",
    "mil": "MIL",
    "bucks": "MIL",
    "milwaukee": "MIL",
    "min": "MIN",
    "timberwolves": "MIN",
    "wolves": "MIN",
    "minnesota": "MIN",
    "nop": "NOP",
    "pelicans": "NOP",
    "new_orleans": "NOP",
    "new orleans": "NOP",
    "ny": "NYK",
    "nyk": "NYK",
    "knicks": "NYK",
    "new_york": "NYK",
    "new york": "NYK",
    "okc": "OKC",
    "thunder": "OKC",
    "oklahoma_city": "OKC",
    "oklahoma city": "OKC",
    "orl": "ORL",
    "magic": "ORL",
    "orlando": "ORL",
    "phi": "PHI",
    "76ers": "PHI",
    "sixers": "PHI",
    "philadelphia": "PHI",
    "phx": "PHX",
    "suns": "PHX",
    "phoenix": "PHX",
    "por": "POR",
    "trail_blazers": "POR",
    "trail blazers": "POR",
    "blazers": "POR",
    "portland": "POR",
    "sac": "SAC",
    "kings": "SAC",
    "sacramento": "SAC",
    "sa": "SAS",
    "sas": "SAS",
    "spurs": "SAS",
    "san_antonio": "SAS",
    "san antonio": "SAS",
    "tor": "TOR",
    "raptors": "TOR",
    "toronto": "TOR",
    "uta": "UTA",
    "jazz": "UTA",
    "utah": "UTA",
    "was": "WAS",
    "wizards": "WAS",
    "washington": "WAS",
}


@dataclass
class ParsedGame:
    game_id: int
    market_date: str | None
    long_rows: list[dict]
    page_title: str | None
    matchup_teams: tuple[str, ...] = ()
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _player_aliases(player_dir_name: str) -> set[str]:
    normalized = normalize_name(player_dir_name)
    aliases = {normalized}
    parts = [part for part in normalized.split("_") if part]
    if len(parts) >= 2:
        aliases.add(f"{parts[0][0]}_{'_'.join(parts[1:])}")
        aliases.add(f"{parts[0][0]}_{parts[-1]}")
    return aliases


@lru_cache(maxsize=1)
def build_player_csv_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for player_dir in DATA_DIR.iterdir():
        if not player_dir.is_dir():
            continue
        csv_paths = sorted(player_dir.glob("*_processed_processed.csv"))
        if not csv_paths:
            continue
        for alias in _player_aliases(player_dir.name):
            index.setdefault(alias, []).extend(csv_paths)
    return index


@lru_cache(maxsize=4096)
def load_player_team_frame(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["Date", "MATCHUP"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.loc[df["Date"].notna()].sort_values("Date").reset_index(drop=True)
    return df


def team_abbr_from_matchup(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"\s*([A-Z]{2,3})\b", str(value))
    return match.group(1) if match else None


def resolve_player_team_abbr(player_name_norm: str, market_date: str | None) -> str | None:
    if not player_name_norm:
        return None
    candidates = build_player_csv_index().get(normalize_name(player_name_norm), [])
    if not candidates:
        return None
    target_date = pd.Timestamp(market_date).normalize() if market_date else None
    best_date = None
    best_team = None
    for csv_path in candidates:
        try:
            team_df = load_player_team_frame(str(csv_path))
        except Exception:
            continue
        if team_df.empty:
            continue
        if target_date is not None:
            eligible = team_df.loc[team_df["Date"] <= target_date]
            if not eligible.empty:
                row = eligible.iloc[-1]
                row_date = row["Date"]
            else:
                row = team_df.iloc[-1]
                row_date = row["Date"]
        else:
            row = team_df.iloc[-1]
            row_date = row["Date"]
        team_abbr = team_abbr_from_matchup(row.get("MATCHUP"))
        if team_abbr is None:
            continue
        if best_date is None or pd.Timestamp(row_date) > pd.Timestamp(best_date):
            best_date = row_date
            best_team = team_abbr
    return best_team


def normalize_team_token(value: str) -> str:
    return normalize_name(value).lower()


def map_team_token_to_abbr(value: str) -> str | None:
    token = normalize_team_token(value)
    return TEAM_NAME_ALIASES.get(token)


def parse_matchup_teams(page_title: str | None, heading_text: str | None = None) -> tuple[str, ...]:
    candidates = [page_title or "", heading_text or ""]
    patterns = [
        re.compile(r"(.+?)\s+vs\s+(.+?)\s+Prediction", re.I),
        re.compile(r"(.+?)\s+vs\s+(.+?)\s+Picks", re.I),
        re.compile(r"(.+?)\s+vs\s+(.+?)\s+Props", re.I),
    ]
    for text in candidates:
        cleaned = " ".join(str(text).split())
        for pattern in patterns:
            match = pattern.search(cleaned)
            if not match:
                continue
            left = map_team_token_to_abbr(match.group(1).strip())
            right = map_team_token_to_abbr(match.group(2).strip())
            teams = tuple(team for team in (left, right) if team)
            if len(teams) == 2:
                return teams
    return ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect historical Covers NBA player props into market history.")
    parser.add_argument("--date-from", type=str, default=None, help="Inclusive minimum date YYYY-MM-DD. Defaults to min processed date.")
    parser.add_argument("--date-to", type=str, default=None, help="Inclusive maximum date YYYY-MM-DD. Defaults to max processed date.")
    parser.add_argument("--start-game-id", type=int, default=None, help="Optional explicit Covers game id to start crawling from.")
    parser.add_argument("--scan-count", type=int, default=None, help="How many descending game ids to scan. Defaults from local date span.")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent page fetches.")
    parser.add_argument("--batch-size", type=int, default=64, help="Descending id batch size.")
    parser.add_argument("--pause-seconds", type=float, default=0.2, help="Pause between batches.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for normalized history.")
    parser.add_argument("--player-limit", type=int, default=None, help="Optional smoke-test limit when deriving local date range.")
    return parser.parse_args()


def derive_local_date_range(player_limit: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    csv_paths = sorted(DATA_DIR.glob("*/*_processed_processed.csv"))
    if player_limit is not None:
        csv_paths = csv_paths[:player_limit]
    min_date = None
    max_date = None
    for path in csv_paths:
        try:
            date_series = pd.to_datetime(pd.read_csv(path, usecols=["Date"])["Date"], errors="coerce")
        except Exception:
            continue
        local_min = date_series.min()
        local_max = date_series.max()
        if pd.isna(local_min) or pd.isna(local_max):
            continue
        min_date = local_min if min_date is None else min(min_date, local_min)
        max_date = local_max if max_date is None else max(max_date, local_max)
    if min_date is None or max_date is None:
        raise RuntimeError("Unable to derive processed date range from Data-Proc.")
    return pd.Timestamp(min_date).normalize(), pd.Timestamp(max_date).normalize()


def infer_start_game_id(session: requests.Session) -> int:
    response = session.get(COVERS_ODDS_URL, timeout=30)
    response.raise_for_status()
    ids = [int(match) for match in re.findall(r"/sport/basketball/nba/matchup/(\d+)", response.text)]
    if not ids:
        raise RuntimeError("Unable to infer latest Covers game id from current odds page.")
    return max(ids)


def estimate_scan_count(date_from: pd.Timestamp, date_to: pd.Timestamp) -> int:
    span_days = max(1, int((date_to - date_from).days) + 1)
    season_estimate = max(1, math.ceil(span_days / 365.0))
    return max(2500, season_estimate * 2200)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def parse_market_date(html: str) -> str | None:
    match = DATE_RE.search(html)
    if not match:
        return None
    parsed = pd.to_datetime(match.group(1), errors="coerce")
    if pd.isna(parsed):
        return None
    return str(parsed.date())


def parse_title(html: str) -> str | None:
    match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    soup = BeautifulSoup(html, "html.parser")
    if soup.title is None:
        return None
    return " ".join(soup.title.get_text(" ", strip=True).split())


def parse_heading_text(soup: BeautifulSoup) -> str | None:
    for selector in ["h1", "h2"]:
        for item in soup.select(selector):
            text = " ".join(item.get_text(" ", strip=True).split())
            if " vs " in text.lower():
                return text
    return None


def parse_book_offers(row) -> list[dict]:
    offers = []
    container = row.select_one("[data-odds='odds']")
    if container is None:
        return offers
    for anchor in container.select("a.book-odds"):
        text = " ".join(anchor.get_text(" ", strip=True).split())
        match = PRICE_RE.search(text)
        if not match:
            continue
        side, line, price = match.groups()
        offers.append(
            {
                "side": side.lower(),
                "line": float(line),
                "price": float(price),
            }
        )
    return offers


def parse_prop_rows_from_html(game_id: int, html: str) -> ParsedGame:
    market_date = parse_market_date(html)
    page_title = parse_title(html)
    soup = BeautifulSoup(html, "html.parser")
    heading_text = parse_heading_text(soup)
    matchup_teams = parse_matchup_teams(page_title, heading_text)
    home_team = matchup_teams[0] if len(matchup_teams) >= 1 else pd.NA
    away_team = matchup_teams[1] if len(matchup_teams) >= 2 else pd.NA
    if "PLAYER PROPS" not in html or market_date is None:
        return ParsedGame(game_id=game_id, market_date=market_date, long_rows=[], page_title=page_title, matchup_teams=matchup_teams)
    long_rows: list[dict] = []

    for row in soup.find_all("tr", class_="game-projections-container"):
        market_badge = row.select_one("span._badge")
        player_link = row.select_one("a.player-link")
        prediction_span = row.select_one("span.prediction")
        if market_badge is None or player_link is None or prediction_span is None:
            continue

        market_label = " ".join(market_badge.get_text(" ", strip=True).split()).upper()
        if market_label not in TARGET_MAP:
            continue

        player_raw = " ".join(player_link.get_text(" ", strip=True).split())
        player_norm = normalize_name(player_raw)
        prediction_text = " ".join(prediction_span.get_text(" ", strip=True).split())
        pred_match = PREDICTION_RE.search(prediction_text)
        if pred_match is None:
            continue

        side, line = pred_match.groups()
        side = side.lower()
        line_value = float(line)
        offers = parse_book_offers(row)
        if not offers:
            offers = [{"side": side, "line": line_value, "price": float("nan")}]

        aligned_offers = [offer for offer in offers if offer["side"] == side]
        if not aligned_offers:
            aligned_offers = offers

        line_values = [offer["line"] for offer in aligned_offers if offer.get("line") is not None]
        price_values = [offer["price"] for offer in aligned_offers if not pd.isna(offer.get("price"))]
        market_line = float(pd.Series(line_values).median()) if line_values else line_value
        market_line_std = float(pd.Series(line_values).std(ddof=0)) if len(line_values) > 1 else 0.0
        avg_price = float(pd.Series(price_values).mean()) if price_values else float("nan")

        long_rows.append(
            {
                "fetched_at_utc": utc_now_iso(),
                "event_id": str(game_id),
                "commence_time_utc": pd.NaT,
                "event_date_et": market_date,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker_key": "covers_consensus",
                "bookmaker_title": "Covers",
                "market_key": TARGET_MAP[market_label][1],
                "player_name_raw": player_raw,
                "player_name_norm": player_norm,
                "line": market_line,
                "over_price": avg_price if side == "o" else float("nan"),
                "under_price": avg_price if side == "u" else float("nan"),
                "book_count": len(aligned_offers),
                "market_line_std": market_line_std,
            }
        )

    return ParsedGame(game_id=game_id, market_date=market_date, long_rows=long_rows, page_title=page_title, matchup_teams=matchup_teams)


def fetch_and_parse_game(game_id: int) -> ParsedGame:
    session = make_session()
    url = COVERS_MATCHUP_URL.format(game_id=game_id)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return parse_prop_rows_from_html(game_id, response.text)
    except Exception as exc:
        return ParsedGame(game_id=game_id, market_date=None, long_rows=[], page_title=None, error=str(exc))


def build_wide_from_covers_long(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=MARKET_WIDE_COLUMNS)

    consensus = (
        long_df.groupby(["event_date_et", "player_name_norm", "player_name_raw", "market_key"], dropna=False)
        .agg(
            market_line=("line", "median"),
            market_line_std=("market_line_std", "mean"),
            over_price_avg=("over_price", "mean"),
            under_price_avg=("under_price", "mean"),
            book_count=("book_count", "max"),
            first_event_id=("event_id", "min"),
            first_commence_time_utc=("commence_time_utc", "min"),
            first_home_team=("home_team", "min"),
            first_away_team=("away_team", "min"),
            first_fetched_at=("fetched_at_utc", "max"),
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

    def pivot(metric_col: str, rename_map: dict[str, str]) -> pd.DataFrame:
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
            "first_fetched_at",
        ]
    ].drop_duplicates(subset=["event_date_et", "player_name_norm", "player_name_raw"], keep="last")

    wide = metadata.merge(
        pivot("market_line", value_map),
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
            pivot(metric_col, rename_map),
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
            "first_fetched_at": "Market_Fetched_At_UTC",
        }
    )
    if "Market_Fetched_At_UTC" not in wide.columns:
        wide["Market_Fetched_At_UTC"] = utc_now_iso()
    for col in MARKET_WIDE_COLUMNS:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide[MARKET_WIDE_COLUMNS].drop_duplicates(subset=["Market_Date", "Player"], keep="last")
    return wide


def main() -> None:
    args = parse_args()
    local_min, local_max = derive_local_date_range(args.player_limit)
    date_from = pd.Timestamp(args.date_from).normalize() if args.date_from else local_min
    date_to = pd.Timestamp(args.date_to).normalize() if args.date_to else local_max

    session = make_session()
    start_game_id = args.start_game_id or infer_start_game_id(session)
    scan_count = args.scan_count or estimate_scan_count(date_from, date_to)
    game_ids = list(range(start_game_id, start_game_id - scan_count, -1))

    results: list[ParsedGame] = []
    for offset in range(0, len(game_ids), args.batch_size):
        batch_ids = game_ids[offset : offset + args.batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            batch_results = list(executor.map(fetch_and_parse_game, batch_ids))
        results.extend(batch_results)
        if args.pause_seconds > 0 and offset + args.batch_size < len(game_ids):
            time.sleep(args.pause_seconds)

    kept = []
    errors = []
    for result in results:
        if result.error:
            errors.append({"game_id": result.game_id, "error": result.error})
            continue
        if result.market_date is None:
            continue
        result_date = pd.Timestamp(result.market_date)
        if result_date < date_from or result_date > date_to:
            continue
        if result.long_rows and result.matchup_teams:
            filtered_rows = []
            for row in result.long_rows:
                player_team = resolve_player_team_abbr(str(row.get("player_name_norm", "")), result.market_date)
                if player_team is None or player_team in result.matchup_teams:
                    filtered_rows.append(row)
            result.long_rows = filtered_rows
        if result.long_rows:
            kept.append(result)

    all_long_rows = [row for result in kept for row in result.long_rows]
    long_df = pd.DataFrame(all_long_rows)
    if not long_df.empty:
        long_df["player_name_norm"] = long_df["player_name_norm"].astype(str).map(normalize_name)
        long_df["book_count"] = pd.to_numeric(long_df["book_count"], errors="coerce")
        long_df["market_line_std"] = pd.to_numeric(long_df["market_line_std"], errors="coerce")
        wide_df = build_wide_from_covers_long(long_df)
    else:
        long_df = pd.DataFrame()
        wide_df = pd.DataFrame(columns=MARKET_WIDE_COLUMNS)

    stamp = utc_compact_timestamp()
    manifest = {
        "provider": "covers_historical",
        "fetched_at_utc": utc_now_iso(),
        "date_from": str(date_from.date()),
        "date_to": str(date_to.date()),
        "start_game_id": start_game_id,
        "scan_count": scan_count,
        "games_scanned": len(game_ids),
        "games_with_props": len(kept),
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "errors": errors[:100],
    }

    write_snapshot(args.outdir, stamp, [], {}, long_df, wide_df, manifest)
    history_summary = append_history(args.outdir, long_df, wide_df)
    manifest.update(history_summary)
    safe_write_json(args.outdir / "latest_manifest.json", manifest)

    print("\n" + "=" * 80)
    print("COVERS HISTORICAL PROP COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Date range:       {manifest['date_from']} -> {manifest['date_to']}")
    print(f"Start game id:    {start_game_id}")
    print(f"Games scanned:    {len(game_ids)}")
    print(f"Games with props: {len(kept)}")
    print(f"Long rows:        {len(long_df)}")
    print(f"Wide rows:        {len(wide_df)}")
    print(f"History wide:     {history_summary['history_wide_rows']}")
    print(f"Errors:           {len(errors)}")
    print(f"Output:           {args.outdir}")


if __name__ == "__main__":
    main()
