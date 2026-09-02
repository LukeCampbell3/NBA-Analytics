#!/usr/bin/env python3
"""Fetch real NFL player-prop lines from RotoWire's public multi-book board.

Direct NFL analogue of sports/nba/predictions/Player-Predictor/scripts/
fetch_nba_market_props.py's `rotowire` provider path. The NBA scraper has
been in real production use for months; this module scrapes the same
public site's NFL page (https://www.rotowire.com/betting/nfl/player-props.php)
under the same JavaScript-embedded `const settings.data = [...]` pattern,
and emits a JSON snapshot in the exact shape run_nfl_daily_predictions.py's
--market-input flag already consumes (live_market.load_fixture_slate,
one observation per (event, player, market, book), both over and under
prices required per row).

Scope of this first NFL rotowire commit:
- Real Week 1 lines for the three NFL market keys the daily-predictions
  pipeline currently understands (live_market.MARKET_KEYS):
    player_pass_yds  <- rotowire `passyds` block
    player_rush_yds  <- rotowire `rushyds` block
    player_reception_yds <- rotowire `recyds` block
  The other 25 rotowire prop blocks on the same page (`firsttd`, `anytd`,
  `passtd`, `recs`, `tackle`, `sack`, ...) are real, priced, and worth
  wiring, but each one requires a corresponding projection/settlement
  path in the downstream NFL predictor -- adding market keys here without
  the downstream support would silently discard those rows or produce
  unbacked predictions. Follow-on commits will add each additional
  market family alongside its predictor evidence, one at a time.

- FanDuel-priority; every other real book (betrivers, caesars,
  draftkings, fanatics, betr, hardrock, mgm, thescore, circasports)
  is captured too. A row is emitted only when BOTH over and under prices
  exist at the same real line for one book -- the same real
  two-sided-complete gate flatten_event_odds already applies to The
  Odds API path.

- Deliberately does NOT touch model artifacts, settlement, or the
  frontend. Its only job is to write a JSON snapshot the existing
  pipeline can already consume.

Not scraped here (deliberate scope guardrails, per this session's
"one product at a time" plan):
- First TD / any TD / passing TD player-prop singles (need new market
  keys + downstream predictor support).
- Team-market totals over/under (a different real market shape --
  h2h and totals live under NFL's TEAM_MARKET_KEYS path, not the
  player-prop path this scraper feeds).
- Passer / rusher / receiver same-family parlays and same-game
  parlays (each is its own separate module; see
  sports/mlb/parlay_v2/ and sports/mlb/scripts/select_mlb_same_game_bets.py
  for the shape they follow when they land).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROTOWIRE_URL = "https://www.rotowire.com/betting/nfl/player-props.php"

# The three real prop blocks whose downstream NFL predictor exists
# today (live_market.MARKET_KEYS). Any prop key not in this map is
# ignored on purpose to prevent silently emitting rows the pipeline
# would then discard or -- worse -- try to project without evidence.
ROTOWIRE_PROP_MAP: dict[str, tuple[str, str]] = {
    "passyds": ("player_pass_yds", "passing"),
    "rushyds": ("player_rush_yds", "rushing"),
    "recyds": ("player_reception_yds", "receiving"),
}
# Retained for a separate odds-only research surface. These rows never enter
# build_observations and therefore cannot be mistaken for modeled two-sided
# candidates.
ROTOWIRE_RESEARCH_INVENTORY_PROPS = {"firsttd"}

ROTOWIRE_BOOK_TITLES: dict[str, str] = {
    "betr": "betr",
    "betrivers": "BetRivers",
    "caesars": "Caesars",
    "circasports": "Circa Sports",
    "draftkings": "DraftKings",
    "fanatics": "Fanatics",
    "fanduel": "FanDuel",
    "hardrock": "Hard Rock",
    "mgm": "BetMGM",
    "thescore": "theScore",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rotowire_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            # Rotowire's page is rendered as a normal browser page; a real
            # UA header is required to receive the same HTML a browser gets.
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def _extract_data_array_literal(script_text: str) -> str:
    """Balance-parse the JavaScript `data:[...]` literal inside a
    <script>const settings = { ... data: [...] ...}</script> block.
    Same character-class-aware brace matcher the NBA scraper uses --
    JSON.parse-compatible output for the same real reason (rotowire
    embeds valid JSON inside the JavaScript literal)."""
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


def extract_rotowire_nfl_page_payload(html: str) -> tuple[int | None, dict[str, list[dict[str, Any]]]]:
    """Returns (nfl_week, per-prop-rows).

    The NBA scraper reads `const dayNBA = "YYYY-MM-DD"`; the NFL page
    uses `const dayNFL = "1"` -- an integer WEEK NUMBER, not a date.
    Retained here anyway because the week label is real signal for the
    downstream consumer (a snapshot claiming to be Week 1 but scraped
    from a Week 5 page would silently mis-partition rows).
    """
    week_match = re.search(r'const dayNFL\s*=\s*"([0-9]+)"', html)
    week: int | None
    try:
        week = int(week_match.group(1)) if week_match else None
    except (TypeError, ValueError):
        week = None

    bundles: dict[str, list[dict[str, Any]]] = {}
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL)
    for script_text in scripts:
        if "const settings" not in script_text or "data:" not in script_text:
            continue
        prop_match = re.search(r'const prop\s*=\s*"([a-zA-Z]+)"', script_text)
        if not prop_match:
            continue
        prop = str(prop_match.group(1)).strip().lower()
        if prop not in ROTOWIRE_PROP_MAP and prop not in ROTOWIRE_RESEARCH_INVENTORY_PROPS:
            continue
        try:
            rows = json.loads(_extract_data_array_literal(script_text))
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(rows, list) and rows:
            bundles[prop] = [row for row in rows if isinstance(row, dict)]
    return week, bundles


def _numeric_price(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("+", "")
    if not text or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _numeric_line(value: Any) -> float | None:
    return _numeric_price(value)  # same permissive parse


def _rotowire_book_keys_for_prop(row: dict[str, Any], prop: str) -> list[str]:
    suffixes = (f"_{prop}", f"_{prop}Over", f"_{prop}Under")
    books: set[str] = set()
    for key in row:
        for suffix in suffixes:
            if key.endswith(suffix):
                books.add(key[: -len(suffix)])
                break
    return sorted(books)


def _rotowire_matchup(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Rotowire encodes home/away as team + opp where opp starts with
    '@' when the player is on the road. Return (home_team, away_team)."""
    team = str(row.get("team") or "").strip().upper() or None
    opp = str(row.get("opp") or "").strip().upper() or None
    if not team:
        return None, (opp.removeprefix("@") if opp else None)
    if opp and opp.startswith("@"):
        return opp[1:] or None, team
    return team, opp


def build_observations(
    *,
    week: int | None,
    bundles: dict[str, list[dict[str, Any]]],
    fetched_at_utc: str,
) -> list[dict[str, Any]]:
    """Emit one observation per real (event, player, market, book) where
    both over and under prices exist at the same real line -- same
    two-sided-complete contract flatten_event_odds enforces on The Odds
    API path, so run_nfl_daily_predictions.py can consume this snapshot
    interchangeably with the odds-api path via --market-input."""
    observations: list[dict[str, Any]] = []
    for prop, (market_key, target) in ROTOWIRE_PROP_MAP.items():
        for row in bundles.get(prop, []):
            player = str(row.get("name") or "").strip()
            if not player:
                continue
            game_id = str(row.get("gameID") or "").strip() or None
            home_team, away_team = _rotowire_matchup(row)
            for book in _rotowire_book_keys_for_prop(row, prop):
                line = _numeric_line(row.get(f"{book}_{prop}"))
                over_price = _numeric_price(row.get(f"{book}_{prop}Over"))
                under_price = _numeric_price(row.get(f"{book}_{prop}Under"))
                # Two-sided-complete gate: real rotowire rows commonly
                # have a book with 1 or 2 of these three fields but not
                # all three (a book has posted a line but not both
                # prices yet). Skip those -- they are not usable
                # observations, and silently emitting them would defeat
                # the downstream flatten_event_odds-shaped contract.
                if line is None or over_price is None or under_price is None:
                    continue
                observations.append(
                    {
                        "event_id": game_id or "",
                        "commence_time_utc": None,
                        "home_team": home_team,
                        "away_team": away_team,
                        "player": player,
                        "market": market_key,
                        "target": target,
                        "line": line,
                        "bookmaker": book,
                        "bookmaker_title": ROTOWIRE_BOOK_TITLES.get(book, book.title()),
                        "over_price": over_price,
                        "under_price": under_price,
                        "snapshot_time_utc": fetched_at_utc,
                        "fetched_at_utc": fetched_at_utc,
                        "source": "rotowire_public_nfl_props",
                        "rotowire_week": week,
                    }
                )
    return observations


def fetch_page(url: str = ROTOWIRE_URL, *, timeout: float = 30.0) -> str:
    with _rotowire_session() as session:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text


def build_snapshot(html: str, *, fetched_at_utc: str | None = None) -> dict[str, Any]:
    """Turn raw rotowire HTML into a JSON snapshot in the exact schema
    live_market.load_fixture_slate consumes (top-level dict with
    `observations` list + `audit` metadata; each observation matches
    flatten_event_odds's row keys)."""
    ts = fetched_at_utc or utc_now_iso()
    week, bundles = extract_rotowire_nfl_page_payload(html)
    observations = build_observations(week=week, bundles=bundles, fetched_at_utc=ts)
    first_td_best_prices = []
    for row in bundles.get("firsttd", []):
        offers = []
        for book in _rotowire_book_keys_for_prop(row, "firsttd"):
            price = _numeric_price(row.get(f"{book}_firsttd"))
            if price is not None:
                offers.append((book, price))
        if not offers:
            continue
        book, price = max(offers, key=lambda item: item[1])
        home_team, away_team = _rotowire_matchup(row)
        first_td_best_prices.append({
            "event_id": str(row.get("gameID") or ""),
            "provider_player_id": str(row.get("playerID") or ""),
            "player": str(row.get("name") or "").strip(),
            "team": str(row.get("team") or "").strip(),
            "opponent": str(row.get("opp") or "").strip().lstrip("@"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmaker": book,
            "price": price,
            "snapshot_time_utc": ts,
            "source": "rotowire_public_nfl_props",
        })
    audit = {
        "provider": "rotowire_public_nfl_props",
        "sport_key": "americanfootball_nfl",
        "source_url": ROTOWIRE_URL,
        "fetched_at_utc": ts,
        "rotowire_week": week,
        "bundle_rows": {prop: len(rows) for prop, rows in bundles.items()},
        "observations_by_market": {
            market_key: sum(1 for obs in observations if obs["market"] == market_key)
            for market_key in {
                pair[0] for pair in ROTOWIRE_PROP_MAP.values()
            }
        },
        "observations_by_book": {
            book: sum(1 for obs in observations if obs["bookmaker"] == book)
            for book in sorted({obs["bookmaker"] for obs in observations})
        },
        "complete_two_sided_rows": len(observations),
        "first_td_best_prices": first_td_best_prices,
        "markets": sorted({pair[0] for pair in ROTOWIRE_PROP_MAP.values()}),
        "regions": "public_rotowire_scrape",
        "raw_source_sha256": None,
    }
    return {"schema_version": 1, "audit": audit, "observations": observations}


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON snapshot destination.")
    parser.add_argument(
        "--fixture-html",
        type=Path,
        default=None,
        help="Read HTML from this file instead of live-fetching -- used by tests only.",
    )
    parser.add_argument("--url", default=ROTOWIRE_URL, help="Override RotoWire NFL props URL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fixture_html:
        html = args.fixture_html.read_text(encoding="utf-8")
    else:
        html = fetch_page(args.url)
    snapshot = build_snapshot(html)
    write_snapshot(args.output, snapshot)
    audit = snapshot["audit"]
    print(
        json.dumps(
            {
                "rotowire_week": audit.get("rotowire_week"),
                "bundle_rows": audit.get("bundle_rows"),
                "observations_by_market": audit.get("observations_by_market"),
                "observations_by_book": audit.get("observations_by_book"),
                "complete_two_sided_rows": audit.get("complete_two_sided_rows"),
                "written": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
