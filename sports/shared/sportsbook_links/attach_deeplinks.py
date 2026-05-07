#!/usr/bin/env python3
"""Attach sportsbook deep links to a final prediction board.

Usage:
    python -m sports.shared.sportsbook_links.attach_deeplinks --sport mlb --input plays.json --output plays_linked.json
    python -m sports.shared.sportsbook_links.attach_deeplinks --sport nba --input plays.json --output plays_linked.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.shared.sportsbook_links.resolver import enrich_picks_with_deeplinks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach sportsbook deep links to prediction picks.")
    parser.add_argument("--sport", required=True, choices=["nba", "mlb"], help="Sport to process.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file with picks (list of dicts).")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file (default: overwrite input).")
    parser.add_argument("--books", type=str, default="draftkings,fanduel", help="Comma-separated sportsbooks to scrape.")
    parser.add_argument("--run-date", type=str, default=None, help="Run date for index naming.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load picks
    input_data = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(input_data, dict):
        picks = input_data.get("plays", [])
        is_payload = True
    else:
        picks = input_data
        is_payload = False

    books = tuple(b.strip() for b in args.books.split(",") if b.strip())

    print(f"Attaching deep links: {len(picks)} {args.sport.upper()} picks, books={books}")

    enriched, summary = enrich_picks_with_deeplinks(
        picks,
        sport=args.sport,
        books=books,
        run_date=args.run_date,
    )

    # Write output
    if is_payload:
        input_data["plays"] = enriched
        input_data["sportsbook_link_summary"] = summary
        output_data = input_data
    else:
        output_data = enriched

    out_path = args.output or args.input
    out_path.write_text(json.dumps(output_data, indent=2, default=str), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Summary: {summary['betslip_links']} betslip, {summary['search_fallbacks']} search fallback")


if __name__ == "__main__":
    main()
