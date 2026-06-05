#!/usr/bin/env python3
"""
Quick diagnostic for SportsGameOdds MLB provider.

Usage:
  $env:SPORTSGAMEODDS_API_KEY = "your-key-here"
  python sports/mlb/predictions/odds/diagnose_provider.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "providers"))

from sportsgameodds_mlb_provider import SportsGameOddsMlbProvider


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from provider_credentials import get_sportsgameodds_api_key

    creds = get_sportsgameodds_api_key()
    if not creds["credentials_present"]:
        print("ERROR: SPORTSGAMEODDS_API_KEY not found")
        print("")
        print("Options:")
        print("  1. Create a .env file at the repo root with: SPORTSGAMEODDS_API_KEY=your-key")
        print('  2. PowerShell: $env:SPORTSGAMEODDS_API_KEY = "your-key-here"')
        print('  3. Bash:       export SPORTSGAMEODDS_API_KEY="your-key-here"')
        sys.exit(1)

    print(f"Credentials: present (length={creds['key_length']}, source={creds['key_source']})")
    print("")

    key = creds["api_key"]

    provider = SportsGameOddsMlbProvider(api_key=key)
    config = provider.validate_config()
    print(f"Config check: {config['status']}")
    print("")

    print("Calling SportsGameOdds v2 API...")
    print(f"  URL: https://api.sportsgameodds.com/v2/events")
    print(f"  Params: leagueID=MLB, oddsAvailable=true")
    print(f"  Auth: x-api-key header")
    print("")

    result = provider.collect_player_props()
    status = result.get("status")
    print(f"Result status: {status}")

    if status == "success":
        odds = result.get("odds", [])
        events_checked = result.get("events_checked", 0)
        print(f"Events checked: {events_checked}")
        print(f"Player props found: {len(odds)}")
        print("")

        if odds:
            # Show sample
            markets = {}
            for o in odds:
                m = o.get("market_canonical", "?")
                markets[m] = markets.get(m, 0) + 1

            print("Markets breakdown:")
            for m, count in sorted(markets.items(), key=lambda x: -x[1]):
                print(f"  {m}: {count} props")
            print("")

            print("Sample props (first 5):")
            for o in odds[:5]:
                print(f"  {o.get('player')} | {o.get('market_canonical')} {o.get('line')} | {o.get('home_team')} vs {o.get('away_team')}")

        # Normalize
        df = provider.normalize(odds)
        print(f"\nNormalized rows: {len(df)}")
        if not df.empty:
            print(f"Books: {sorted(df['book'].unique().tolist())}")
            print(f"Markets: {sorted(df['market_canonical'].unique().tolist())}")
            print(f"Players: {df['player'].nunique()}")

    elif status == "no_props":
        print(f"Message: {result.get('message', '')}")
        print("")
        print("This likely means:")
        print("  - No MLB games are scheduled today/upcoming")
        print("  - Or the API returned events but no player prop odds")
        print("")
        print("Debug info:")
        print(f"  Events checked: {result.get('events_checked', 0)}")
        if hasattr(provider, '_last_response_debug'):
            print(f"  Last response: {provider._last_response_debug}")

    elif status in ("missing_credentials", "api_error"):
        print(f"Message: {result.get('message', '')}")
        if result.get("code"):
            print(f"HTTP code: {result.get('code')}")
        if result.get("body"):
            print(f"Response body: {result.get('body')}")
        print("")
        print("Troubleshooting:")
        print("  1. Verify your API key is correct (check email from sportsgameodds.com)")
        print("  2. Ensure your plan includes MLB data")
        print("  3. Check if your account is active at sportsgameodds.com/pricing")

    else:
        print(f"Full result: {json.dumps(result, indent=2, default=str)}")

    sys.exit(0 if status == "success" else 1)


if __name__ == "__main__":
    main()
