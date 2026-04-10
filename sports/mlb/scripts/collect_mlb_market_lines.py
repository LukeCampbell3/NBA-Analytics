#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.market_lines import (
    DEFAULT_BOOKMAKERS,
    DEFAULT_MARKETS,
    DEFAULT_PROVIDER,
    MarketCollectorConfig,
    SUPPORTED_PROVIDERS,
    collect_market_lines,
    load_provider_api_key,
    provider_requires_api_key,
    provider_env_var,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect current MLB player prop lines from a configured odds provider.")
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=DEFAULT_PROVIDER,
        help="Odds provider to use.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Provider API key. If omitted, uses the matching environment variable.",
    )
    parser.add_argument(
        "--market-root",
        type=Path,
        default=ROOT / "data" / "raw" / "market_odds" / "mlb",
        help="Directory for raw snapshots and normalized outputs.",
    )
    parser.add_argument("--event-date", type=str, default=None, help="Optional ET game date to fetch (YYYY-MM-DD).")
    parser.add_argument(
        "--bookmakers",
        type=str,
        default=",".join(DEFAULT_BOOKMAKERS),
        help="Comma-separated bookmaker names. SportsGameOdds IDs are derived automatically from the labels.",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=",".join(DEFAULT_MARKETS),
        help="Comma-separated normalized target markets to retain (H,HR,RBI,K,ER).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum upcoming events to request from the provider.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not copy this provider's outputs to the shared latest_player_props files at the market root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_provider_api_key(str(args.provider), explicit=args.api_key)
    if provider_requires_api_key(str(args.provider)) and not api_key:
        raise RuntimeError(
            f"Missing {args.provider} API key. Pass --api-key or set {provider_env_var(str(args.provider))}."
        )

    manifest = collect_market_lines(
        MarketCollectorConfig(
            api_key=str(api_key or ""),
            market_root=args.market_root,
            provider=str(args.provider),
            bookmakers=tuple(item.strip() for item in str(args.bookmakers).split(",") if item.strip()),
            markets=tuple(item.strip() for item in str(args.markets).split(",") if item.strip()),
            event_date=args.event_date,
            limit=int(args.limit),
            promote_to_latest=not bool(args.no_promote),
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
