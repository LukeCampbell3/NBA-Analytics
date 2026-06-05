#!/usr/bin/env python3
"""
Fetch real current NBA player prop odds into the v9.3 odds snapshot schema.

The public RotoWire player-props page exposes the current board in inline
JavaScript. This script extracts PTS/TRB/AST book prices and publishes:

- book-level long snapshots
- a canonical v9.3 attachment file using a preferred sportsbook per row

RotoWire current pages do not expose true closing prices. To keep the schema
loadable without pretending CLV is known, close_* fields are populated with the
current snapshot values and close_status marks them as provisional.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from market_odds_quality import add_american_odds_quality, odds_quality_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://www.rotowire.com/betting/nba/player-props.php?prop=pts"
DEFAULT_OUTDIR = ROOT / "data" / "market_odds" / "nba" / "rotowire"
SUPPORTED_MARKETS = {"pts": "PTS", "reb": "TRB", "ast": "AST"}
BOOK_TITLES = {
    "betrivers": "BetRivers",
    "caesars": "Caesars",
    "draftkings": "DraftKings",
    "fanatics": "Fanatics",
    "fanduel": "FanDuel",
    "hardrock": "Hard Rock",
    "mgm": "BetMGM",
    "thescore": "theScore",
}
PREFERRED_BOOKS = ["draftkings", "fanduel", "mgm", "caesars", "betrivers", "fanatics", "hardrock", "thescore"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_player_name(value: object) -> str:
    out = str(value or "").strip()
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


def to_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def no_vig_probs(over_odds: float, under_odds: float) -> tuple[float, float]:
    over = american_to_implied(over_odds)
    under = american_to_implied(under_odds)
    total = over + under
    if total <= 0 or not np.isfinite(total):
        return 0.5, 0.5
    return over / total, under / total


def _extract_array_literal(script_text: str, start_idx: int) -> str:
    data_idx = script_text.find("data:", start_idx)
    if data_idx < 0:
        raise ValueError("settings.data array not found")
    start = script_text.find("[", data_idx)
    if start < 0:
        raise ValueError("settings.data opening bracket not found")

    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for pos in range(start, len(script_text)):
        char = script_text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return script_text[start : pos + 1]
    raise ValueError("unterminated settings.data array")


def extract_rotowire_bundles(html: str) -> tuple[str, dict[str, list[dict[str, object]]]]:
    day_matches = re.findall(r'const dayNBA = "([0-9]{4}-[0-9]{2}-[0-9]{2})"', html)
    if not day_matches:
        raise RuntimeError("Could not find dayNBA in RotoWire NBA props page")
    market_date = day_matches[0]
    bundles: dict[str, list[dict[str, object]]] = {}
    for match in re.finditer(r'const prop = "([^"]+)"', html):
        prop = match.group(1)
        if prop not in SUPPORTED_MARKETS:
            continue
        settings_idx = html.find("const settings", match.start())
        if settings_idx < 0:
            continue
        array_literal = _extract_array_literal(html, settings_idx)
        rows = json.loads(array_literal)
        if isinstance(rows, list):
            bundles[prop] = rows
    if not bundles:
        raise RuntimeError("No supported PTS/TRB/AST prop bundles found in RotoWire page")
    return market_date, bundles


def _book_keys_for_row(row: dict[str, object], prop: str) -> list[str]:
    suffixes = (f"_{prop}", f"_{prop}Over", f"_{prop}Under")
    books: set[str] = set()
    for key in row:
        for suffix in suffixes:
            if key.endswith(suffix):
                books.add(key[: -len(suffix)])
                break
    return sorted(books)


def build_book_snapshots(market_date: str, bundles: dict[str, list[dict[str, object]]], fetched_at: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for prop, market in SUPPORTED_MARKETS.items():
        for row in bundles.get(prop, []):
            player = normalize_player_name(row.get("name"))
            if not player:
                continue
            for book in _book_keys_for_row(row, prop):
                line = to_float(row.get(f"{book}_{prop}"))
                over_odds = to_float(row.get(f"{book}_{prop}Over"))
                under_odds = to_float(row.get(f"{book}_{prop}Under"))
                if line is None or over_odds is None or under_odds is None:
                    continue
                if not (abs(float(over_odds)) >= 100 and abs(float(under_odds)) >= 100):
                    continue
                no_vig_over, no_vig_under = no_vig_probs(over_odds, under_odds)
                records.append(
                    {
                        "snapshot_time": fetched_at,
                        "snapshot_date": market_date,
                        "book": BOOK_TITLES.get(book, book),
                        "book_key": book,
                        "game_id": str(row.get("gameID") or ""),
                        "player_id": str(row.get("playerID") or ""),
                        "player": player,
                        "player_raw": str(row.get("name") or ""),
                        "market": market,
                        "line": line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "no_vig_over": no_vig_over,
                        "no_vig_under": no_vig_under,
                        "open_line": line,
                        "current_line": line,
                        "close_line": line,
                        "close_over_odds": over_odds,
                        "close_under_odds": under_odds,
                        "close_status": "provisional_current_snapshot_not_closing",
                        "team": row.get("team"),
                        "opponent": row.get("opp"),
                        "source": "rotowire_current_board",
                    }
                )
    return pd.DataFrame(records)


def build_canonical_snapshots(book_rows: pd.DataFrame) -> pd.DataFrame:
    if book_rows.empty:
        return book_rows.copy()
    ranked = book_rows.copy()
    preference = {book: idx for idx, book in enumerate(PREFERRED_BOOKS)}
    ranked["_book_rank"] = ranked["book_key"].map(preference).fillna(999).astype(int)
    ranked = ranked.sort_values(["snapshot_date", "game_id", "player", "market", "_book_rank", "book_key"])
    canonical = ranked.drop_duplicates(["snapshot_date", "game_id", "player_id", "player", "market"], keep="first").copy()
    canonical = canonical.drop(columns=["_book_rank"])
    canonical = canonical.rename(columns={"snapshot_date": "date"})
    return canonical


def write_outputs(outdir: Path, book_rows: pd.DataFrame, canonical: pd.DataFrame, manifest: dict) -> None:
    stamp = manifest["snapshot_stamp"]
    raw_dir = outdir / "raw" / stamp
    norm_dir = outdir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    book_rows.to_csv(norm_dir / f"nba_player_props_book_snapshots_{stamp}.csv", index=False)
    canonical.to_csv(norm_dir / f"nba_player_props_v9_3_snapshots_{stamp}.csv", index=False)
    book_rows.to_csv(outdir / "latest_nba_player_props_book_snapshots.csv", index=False)
    canonical.to_csv(outdir / "latest_nba_player_props_v9_3_snapshots.csv", index=False)
    (outdir / "latest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current NBA player prop market snapshots from RotoWire")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetched_at = utc_now_iso()
    response = requests.get(args.url, timeout=args.timeout_seconds, headers={"User-Agent": "NBA-Analytics/1.0"})
    response.raise_for_status()
    market_date, bundles = extract_rotowire_bundles(response.text)
    book_rows = add_american_odds_quality(build_book_snapshots(market_date, bundles, fetched_at))
    canonical = add_american_odds_quality(build_canonical_snapshots(book_rows))
    manifest = {
        "provider": "rotowire",
        "source_url": args.url,
        "snapshot_stamp": utc_stamp(),
        "fetched_at_utc": fetched_at,
        "market_date": market_date,
        "supported_markets": sorted(SUPPORTED_MARKETS.values()),
        "book_rows": int(len(book_rows)),
        "canonical_rows": int(len(canonical)),
        "books": sorted(book_rows["book"].dropna().unique().tolist()) if not book_rows.empty else [],
        "odds_quality": odds_quality_report(book_rows),
        "close_status": "provisional_current_snapshot_not_closing",
        "clv_ready": False,
        "notes": [
            "RotoWire current board gives current lines and prices, not true closing lines.",
            "close_* fields are current aliases so the schema can be attached; do not use this file for CLV promotion.",
        ],
    }
    write_outputs(args.outdir, book_rows, canonical, manifest)
    print(json.dumps({"manifest": manifest, "snapshot_file": str(args.outdir / "latest_nba_player_props_v9_3_snapshots.csv")}, indent=2))


if __name__ == "__main__":
    main()
