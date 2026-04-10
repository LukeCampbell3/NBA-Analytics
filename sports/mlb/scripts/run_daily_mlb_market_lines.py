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
    compare_provider_manifests,
    default_event_date,
    load_provider_api_key,
    provider_env_candidates,
    provider_requires_api_key,
    provider_env_var,
    publish_market_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch today's MLB player-prop lines, compare providers, and publish the best feed."
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="covers,odds_api_io,sportsgameodds",
        help="Comma-separated providers to try in order.",
    )
    parser.add_argument("--event-date", type=str, default=None, help="ET game date to pull (YYYY-MM-DD).")
    parser.add_argument(
        "--market-root",
        type=Path,
        default=ROOT / "data" / "raw" / "market_odds" / "mlb",
        help="Directory for provider snapshots and published latest outputs.",
    )
    parser.add_argument(
        "--bookmakers",
        type=str,
        default=",".join(DEFAULT_BOOKMAKERS),
        help="Comma-separated bookmaker names to request from each provider.",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=",".join(DEFAULT_MARKETS),
        help="Comma-separated normalized target markets to retain (H,HR,RBI,K,ER).",
    )
    parser.add_argument("--limit", type=int, default=30, help="Max event count per provider request.")
    parser.add_argument("--odds-api-io-key", type=str, default=None, help="Explicit odds-api.io key override.")
    parser.add_argument("--sportsgameodds-api-key", type=str, default=None, help="Explicit SportsGameOdds key override.")
    parser.add_argument(
        "--primary-provider",
        type=str,
        default=DEFAULT_PROVIDER,
        help="Provider to fall back to if scores are tied.",
    )
    parser.add_argument(
        "--fail-on-missing-keys",
        action="store_true",
        help="Exit non-zero if any requested provider is missing its API key.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_date = str(args.event_date or default_event_date())
    providers = [item.strip() for item in str(args.providers).split(",") if item.strip()]
    bookmakers = tuple(item.strip() for item in str(args.bookmakers).split(",") if item.strip())
    markets = tuple(item.strip() for item in str(args.markets).split(",") if item.strip())

    manifests: list[dict] = []
    missing_keys: list[str] = []
    explicit_keys = {
        "odds_api_io": args.odds_api_io_key,
        "sportsgameodds": args.sportsgameodds_api_key,
    }
    for provider in providers:
        api_key = load_provider_api_key(provider, explicit=explicit_keys.get(provider))
        if provider_requires_api_key(provider) and not api_key:
            expected = "/".join(provider_env_candidates(provider))
            missing_keys.append(f"{provider}:{expected}")
            manifests.append(
                {
                    "status": "skipped",
                    "provider": provider,
                    "reason": f"Missing API key ({expected})",
                    "event_date": event_date,
                }
            )
            continue
        from pipeline.market_lines import collect_market_lines

        try:
            manifests.append(
                collect_market_lines(
                MarketCollectorConfig(
                    api_key=str(api_key or ""),
                    provider=provider,
                    market_root=args.market_root,
                        event_date=event_date,
                        bookmakers=bookmakers,
                        markets=markets,
                        limit=int(args.limit),
                        promote_to_latest=False,
                    )
                )
            )
        except Exception as exc:
            manifests.append(
                {
                    "status": "error",
                    "provider": provider,
                    "event_date": event_date,
                    "reason": str(exc),
                }
            )

    ok_manifests = [item for item in manifests if item.get("status") == "ok"]
    comparison = compare_provider_manifests(ok_manifests)

    winner = None
    if comparison.get("ranked"):
        ranked = comparison["ranked"]
        winner = ranked[0]
        if len(ranked) > 1:
            top = ranked[0]
            second = ranked[1]
            if float(top.get("quality_score", 0.0)) == float(second.get("quality_score", 0.0)):
                winner = next((item for item in ranked if item.get("provider") == args.primary_provider), top)
    comparison["winner_provider"] = winner.get("provider") if winner else None
    comparison["winner_quality_score"] = winner.get("quality_score") if winner else 0.0

    published = publish_market_outputs(manifest=winner, market_root=args.market_root) if winner else {}
    comparison["published_files"] = published
    comparison["providers"] = manifests
    comparison["requested_event_date"] = event_date
    comparison["missing_keys"] = missing_keys

    comparison_path = Path(args.market_root).resolve() / "provider_comparison_latest.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    if args.fail_on_missing_keys and missing_keys:
        raise RuntimeError(f"Missing provider keys: {', '.join(missing_keys)}")
    if not winner:
        provider_errors = [f"{item.get('provider')}: {item.get('reason')}" for item in manifests if item.get("status") == "error"]
        detail = f" Missing keys: {', '.join(missing_keys)}." if missing_keys else ""
        error_detail = f" Provider errors: {' | '.join(provider_errors)}." if provider_errors else ""
        raise RuntimeError(f"No provider returned a usable MLB market file.{detail}{error_detail}")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
