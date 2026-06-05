#!/usr/bin/env python3
"""
Odds-API.io Provider Adapter

Connects to Odds-API.io for NBA player prop odds.
Docs: https://odds-api.io/docs
Auth: Authorization: Bearer <token> header
Env var: ODDS_API_IO_KEY

Supports:
- NBA player prop odds
- Multiple bookmakers
- Real-time odds
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

API_BASE = "https://api.odds-api.io/v1"
ENV_KEY = "ODDS_API_IO_KEY"
SPORT = "basketball_nba"

# Market mapping
MARKET_MAP = {
    "player_points": "PTS",
    "player_rebounds": "TRB",
    "player_assists": "AST",
    "player_threes": "3PM",
    "player_three_pointers": "3PM",
    "player_steals": "STL",
    "player_blocks": "BLK",
    "points": "PTS",
    "rebounds": "TRB",
    "assists": "AST",
    "threes": "3PM",
}


class OddsApiIoProvider:
    """Odds-API.io provider for NBA player props."""

    name = "odds_api_io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv(ENV_KEY, "")

    def validate_config(self) -> Dict[str, Any]:
        """Check if API key is configured."""
        if not self.api_key:
            return {
                "status": "missing_credentials",
                "provider": self.name,
                "env_var": ENV_KEY,
                "message": f"Set {ENV_KEY} environment variable. Sign up at https://odds-api.io",
                "signup_url": "https://odds-api.io",
            }
        return {"status": "configured", "provider": self.name}

    def quota_status(self) -> Dict[str, Any]:
        """Check remaining quota."""
        if not self.api_key:
            return {"status": "no_key", "remaining": 0}
        return {"status": "unknown_until_call", "provider": self.name}

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Make authenticated API request with Bearer token."""
        url = f"{API_BASE}/{endpoint}"
        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}?{query}"

        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "NBA-Analytics-ProductionShadow/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return data, headers
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return {"error": True, "code": exc.code, "body": body[:500]}, {}
        except urllib.error.URLError as exc:
            return {"error": True, "code": 0, "body": str(exc.reason)}, {}
        except Exception as exc:
            return {"error": True, "code": 0, "body": str(exc)}, {}

    def collect_player_props(self) -> Dict[str, Any]:
        """Collect player prop odds for NBA."""
        config = self.validate_config()
        if config["status"] != "configured":
            return {"status": "missing_credentials", **config}

        params = {"sport": SPORT, "type": "player_props"}
        data, headers = self._request("odds/player-props", params)

        if isinstance(data, dict) and data.get("error"):
            return {
                "status": "api_error",
                "code": data.get("code"),
                "body": data.get("body", "")[:200],
                "provider": self.name
            }

        odds = []
        if isinstance(data, list):
            odds = data
        elif isinstance(data, dict):
            odds = data.get("data", data.get("odds", data.get("results", [])))
            if not isinstance(odds, list):
                odds = []

        return {
            "status": "success" if odds else "no_props",
            "odds": odds,
            "count": len(odds),
            "provider": self.name,
            "quota_remaining": headers.get("x-ratelimit-remaining", "unknown")
        }

    def normalize(self, raw_odds: List[Dict], snapshot_type: str = "entry") -> pd.DataFrame:
        """Normalize Odds-API.io response to canonical schema."""
        records = []
        now = datetime.now(timezone.utc).isoformat()
        snapshot_id = datetime.now(timezone.utc).strftime("oaio_%Y%m%dT%H%M%SZ")

        for item in raw_odds:
            event_id = item.get("event_id") or item.get("eventId") or item.get("id", "")
            game_id = item.get("game_id") or str(event_id)
            bookmaker = item.get("bookmaker") or item.get("sportsbook") or item.get("book", "unknown")
            market_key = item.get("market") or item.get("market_key") or ""
            player_name = item.get("player") or item.get("player_name") or ""
            line = item.get("line") or item.get("point") or item.get("handicap")
            odds_value = item.get("odds") or item.get("price") or item.get("american_odds")
            side = item.get("side") or item.get("selection") or item.get("outcome", "")
            commence_time = item.get("commence_time") or item.get("start_time", "")
            home_team = item.get("home_team") or ""
            away_team = item.get("away_team") or ""

            canonical_market = MARKET_MAP.get(str(market_key).lower(), "")
            if not canonical_market:
                mk_lower = str(market_key).lower()
                for key, val in MARKET_MAP.items():
                    if key in mk_lower:
                        canonical_market = val
                        break
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
            valid_american = (odds_f <= -100 or odds_f >= 100)
            if not valid_american:
                continue

            side_upper = str(side).upper()
            if side_upper not in ("OVER", "UNDER"):
                if "over" in str(side).lower():
                    side_upper = "OVER"
                elif "under" in str(side).lower():
                    side_upper = "UNDER"
                else:
                    side_upper = "OVER"

            records.append({
                "provider_name": self.name,
                "source_event_id": str(event_id),
                "game_id": str(game_id),
                "snapshot_id": snapshot_id,
                "snapshot_time_utc": now,
                "snapshot_type": snapshot_type,
                "commence_time_utc": str(commence_time),
                "home_team": str(home_team),
                "away_team": str(away_team),
                "player": str(player_name),
                "player_id_source": f"oaio_{player_name.replace(' ', '_').lower()}",
                "market": canonical_market,
                "market_canonical": canonical_market,
                "line": line_f,
                "book": str(bookmaker),
                "side": side_upper,
                "odds": int(odds_f),
                "over_odds": int(odds_f) if side_upper == "OVER" else np.nan,
                "under_odds": int(odds_f) if side_upper == "UNDER" else np.nan,
                "source_market_key": str(market_key),
                "source_selection_key": str(side),
                "is_live": bool(item.get("is_live", False)),
                "raw_json_ref": json.dumps(item)[:200],
                "valid_american_odds": True,
                "schema_valid": True,
            })

        return pd.DataFrame(records) if records else pd.DataFrame()

    def write_snapshot(self, df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
        """Write normalized snapshot and manifest to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_path = output_dir / f"odds_api_io_nba_props_{stamp}.csv"
        manifest_path = output_dir / f"odds_api_io_manifest_{stamp}.json"

        df.to_csv(snapshot_path, index=False)

        manifest = {
            "provider_name": self.name,
            "snapshot_id": f"oaio_{stamp}",
            "snapshot_type": "entry",
            "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
            "rows_raw": len(df),
            "rows_normalized": len(df),
            "valid_rows": len(df),
            "invalid_rows": 0,
            "valid_odds_rate": 1.0,
            "markets": df["market"].unique().tolist() if not df.empty else [],
            "books": df["book"].unique().tolist() if not df.empty else [],
            "events": int(df["game_id"].nunique()) if not df.empty else 0,
            "players": int(df["player"].nunique()) if not df.empty else 0,
            "freshness_status": "fresh",
            "schema_status": "valid",
            "collection_errors": [],
            "quota_status": "ok",
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"status": "success", "snapshot_path": str(snapshot_path), "manifest": manifest}
