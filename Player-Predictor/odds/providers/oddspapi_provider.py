#!/usr/bin/env python3
"""
OddsPapi Provider Adapter for v10.6

Connects to OddsPapi (https://oddspapi.io) for NBA player prop odds.
Free tier: 250 requests/month, historical odds always free.
117+ bookmakers including DraftKings, FanDuel, Pinnacle.

API base: https://v5.oddspapi.io/en
Auth: apiKey query parameter
Env var: ODDSPAPI_API_KEY
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


API_BASE = "https://v5.oddspapi.io/en"
SPORT_ID_NBA = 18  # Basketball NBA
ENV_KEY = "ODDSPAPI_API_KEY"

# Market mapping: OddsPapi market names -> our canonical codes
MARKET_MAP = {
    "player_points": "PTS",
    "player_rebounds": "TRB",
    "player_assists": "AST",
    "player_threes": "3PM",
    "player_steals": "STL",
    "player_blocks": "BLK",
}


class OddsPapiProvider:
    """OddsPapi odds provider for NBA player props."""

    name = "oddspapi"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv(ENV_KEY, "")

    def validate_config(self) -> dict:
        """Check if API key is configured."""
        if not self.api_key:
            return {
                "status": "missing_api_key",
                "human_action_required": True,
                "message": f"Set {ENV_KEY} environment variable or pass api_key. Sign up free at https://oddspapi.io",
                "signup_url": "https://oddspapi.io",
            }
        return {"status": "configured", "human_action_required": False}

    def quota_status(self) -> dict:
        """Check remaining quota (requires a lightweight API call)."""
        if not self.api_key:
            return {"status": "no_key", "remaining": 0}
        # OddsPapi doesn't have a dedicated quota endpoint on free tier
        # We'll know from response headers or errors
        return {"status": "unknown_until_call", "plan": "free_250_per_month"}

    def _request(self, endpoint: str, params: dict) -> tuple[Any, dict]:
        """Make authenticated API request."""
        params["apiKey"] = self.api_key
        url = f"{API_BASE}/{endpoint}"
        
        try:
            import requests as req_lib
            resp = req_lib.get(url, params=params, headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 NBA-Analytics/1.0",
            }, timeout=30)
            if resp.status_code == 200:
                return resp.json(), dict(resp.headers)
            else:
                return {"error": True, "code": resp.status_code, "body": resp.text[:500]}, {}
        except ImportError:
            # Fallback to urllib
            query = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{url}?{query}"
            req = urllib.request.Request(full_url, headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 NBA-Analytics/1.0",
            })
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    return data, headers
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                return {"error": True, "code": exc.code, "body": body[:500]}, {}
        except Exception as exc:
            return {"error": True, "code": 0, "body": str(exc)[:300]}, {}

    def collect_fixtures(self, sport_id: int = SPORT_ID_NBA) -> dict:
        """Get upcoming NBA fixtures/events."""
        config = self.validate_config()
        if config["status"] != "configured":
            return {"status": "not_configured", **config}

        data, headers = self._request("fixtures", {"sportId": sport_id})
        if isinstance(data, dict) and data.get("error"):
            return {"status": "api_error", "code": data.get("code"), "body": data.get("body", "")[:200]}

        if isinstance(data, list):
            return {"status": "success", "fixtures": data, "count": len(data)}
        return {"status": "unexpected_response", "type": str(type(data))}

    def collect_player_props(self, fixture_id: int | None = None, sport_id: int = SPORT_ID_NBA) -> dict:
        """Collect player prop odds for NBA."""
        config = self.validate_config()
        if config["status"] != "configured":
            return {"status": "not_configured", **config}

        params: dict = {"sportId": sport_id}
        if fixture_id:
            params["fixtureId"] = fixture_id

        data, headers = self._request("odds", params)
        if isinstance(data, dict) and data.get("error"):
            return {"status": "api_error", "code": data.get("code"), "body": data.get("body", "")[:200]}

        if isinstance(data, list):
            return {"status": "success", "odds": data, "count": len(data)}
        return {"status": "unexpected_response", "type": str(type(data))}

    def normalize(self, raw_odds: list[dict], snapshot_type: str = "prelock") -> pd.DataFrame:
        """Normalize OddsPapi response to v10.6 canonical schema."""
        records = []
        now = datetime.now(timezone.utc).isoformat()

        for item in raw_odds:
            # OddsPapi structure varies; adapt based on actual response
            fixture_id = item.get("fixtureId") or item.get("fixture_id")
            bookmaker = item.get("bookmakerName") or item.get("bookmaker", "unknown")
            market_key = item.get("marketKey") or item.get("market", "")
            player_name = item.get("playerName") or item.get("player", "")
            line = item.get("line") or item.get("handicap") or item.get("point")
            odds_value = item.get("odds") or item.get("price")
            side = item.get("side") or item.get("selection", "")
            commence_time = item.get("commenceTime") or item.get("startTime")

            # Map market
            canonical_market = MARKET_MAP.get(str(market_key).lower(), "")
            if not canonical_market:
                continue

            if line is None or odds_value is None:
                continue

            try:
                line_f = float(line)
                odds_f = float(odds_value)
            except (TypeError, ValueError):
                continue

            # Validate American odds
            if not (odds_f <= -100 or odds_f >= 100):
                continue

            records.append({
                "provider_name": "oddspapi",
                "source_event_id": str(fixture_id) if fixture_id else "",
                "game_id": str(fixture_id) if fixture_id else "",
                "snapshot_time_utc": now,
                "snapshot_type": snapshot_type,
                "commence_time_utc": str(commence_time) if commence_time else "",
                "player": str(player_name).replace(" ", "_"),
                "player_raw": str(player_name),
                "market": canonical_market,
                "line": line_f,
                "book": str(bookmaker),
                "side": str(side).upper(),
                "odds": odds_f,
                "over_odds": odds_f if str(side).lower() == "over" else np.nan,
                "under_odds": odds_f if str(side).lower() == "under" else np.nan,
                "source_market_key": str(market_key),
                "is_live": bool(item.get("isLive", False)),
                "is_valid_american_odds": True,
            })

        return pd.DataFrame(records) if records else pd.DataFrame()

    def write_snapshot(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """Write normalized snapshot to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = output_dir / f"oddspapi_nba_props_{stamp}.csv"
        df.to_csv(path, index=False)

        manifest = {
            "provider": "oddspapi",
            "snapshot_stamp": stamp,
            "rows": int(len(df)),
            "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
            "valid_odds_rate": 1.0,  # Already filtered
        }
        (output_dir / f"oddspapi_manifest_{stamp}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return path
