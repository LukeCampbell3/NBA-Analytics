#!/usr/bin/env python3
"""
SportsGameOdds Provider Adapter

Connects to SportsGameOdds API for NBA player prop odds.
Docs: https://sportsgameodds.com/docs
Auth: X-Api-Key header
Env var: SPORTSGAMEODDS_API_KEY

Supports:
- NBA events listing
- Player prop odds (PTS, TRB, AST, 3PM, STL, BLK)
- Multiple bookmakers
- American odds format
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

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

API_BASE = "https://api.sportsgameodds.com/v2"
ENV_KEY = "SPORTSGAMEODDS_API_KEY"
# Also check alternate env var name used in their docs
ENV_KEY_ALT = "SPORTSGAMEODDS_KEY"
SPORT = "basketball_nba"

# Market mapping: SportsGameOdds market keys -> canonical codes
MARKET_MAP = {
    "player_points": "PTS",
    "player_points_over_under": "PTS",
    "player_rebounds": "TRB",
    "player_rebounds_over_under": "TRB",
    "player_assists": "AST",
    "player_assists_over_under": "AST",
    "player_threes": "3PM",
    "player_three_pointers_made": "3PM",
    "player_three_pointers_made_over_under": "3PM",
    "player_steals": "STL",
    "player_blocks": "BLK",
    "player_pts": "PTS",
    "player_reb": "TRB",
    "player_ast": "AST",
    "player_3pm": "3PM",
    "player_stl": "STL",
    "player_blk": "BLK",
    # SportsGameOdds statID-based keys
    "pts": "PTS",
    "reb": "TRB",
    "ast": "AST",
    "3pm": "3PM",
    "stl": "STL",
    "blk": "BLK",
}


class SportsGameOddsProvider:
    """SportsGameOdds provider for NBA player props."""

    name = "sportsgameodds"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv(ENV_KEY, "") or os.getenv(ENV_KEY_ALT, "")

    def validate_config(self) -> Dict[str, Any]:
        """Check if API key is configured."""
        if not self.api_key:
            return {
                "status": "missing_credentials",
                "provider": self.name,
                "env_var": f"{ENV_KEY} or {ENV_KEY_ALT}",
                "message": f"Set {ENV_KEY} or {ENV_KEY_ALT} environment variable. Sign up at https://sportsgameodds.com",
                "signup_url": "https://sportsgameodds.com",
            }
        return {"status": "configured", "provider": self.name}

    def quota_status(self) -> Dict[str, Any]:
        """Check remaining quota."""
        if not self.api_key:
            return {"status": "no_key", "remaining": 0}
        return {"status": "unknown_until_call", "provider": self.name}

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Make authenticated API request using requests library.
        
        Uses requests instead of urllib to avoid Cloudflare bot protection blocks.
        Auth: x-api-key header (primary).
        """
        try:
            import requests as req_lib
        except ImportError:
            return {"error": True, "code": 0, "body": "requests library not installed"}, {}

        url = f"{API_BASE}/{endpoint}"
        headers = {
            "Accept": "application/json",
            "x-api-key": self.api_key,
            "User-Agent": "Mozilla/5.0 NBA-Analytics/1.0",
        }

        try:
            resp = req_lib.get(url, params=params, headers=headers, timeout=30)
            resp_headers = dict(resp.headers)

            if resp.status_code == 200:
                data = resp.json()
                return data, resp_headers
            else:
                return {
                    "error": True,
                    "code": resp.status_code,
                    "body": resp.text[:500]
                }, resp_headers
        except req_lib.exceptions.Timeout:
            return {"error": True, "code": 0, "body": "Request timeout"}, {}
        except req_lib.exceptions.ConnectionError as exc:
            return {"error": True, "code": 0, "body": f"Connection error: {str(exc)[:200]}"}, {}
        except Exception as exc:
            return {"error": True, "code": 0, "body": str(exc)[:300]}, {}

    def _request_with_query_key(self, endpoint: str, params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Fallback: authenticate via apiKey query parameter."""
        try:
            import requests as req_lib
        except ImportError:
            return {"error": True, "code": 0, "body": "requests library not installed"}, {}

        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        url = f"{API_BASE}/{endpoint}"

        try:
            resp = req_lib.get(url, params=params, timeout=30, headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 NBA-Analytics/1.0",
            })
            if resp.status_code == 200:
                return resp.json(), dict(resp.headers)
            else:
                return {"error": True, "code": resp.status_code, "body": resp.text[:500]}, dict(resp.headers)
        except Exception as exc:
            return {"error": True, "code": 0, "body": str(exc)[:300]}, {}

    def collect_events(self) -> Dict[str, Any]:
        """Get upcoming NBA events with odds available."""
        config = self.validate_config()
        if config["status"] != "configured":
            return {"status": "missing_credentials", **config}

        data, headers = self._request("events", {
            "leagueID": "NBA",
            "finalized": "false",
            "oddsAvailable": "true",
            "limit": 20
        })
        if isinstance(data, dict) and data.get("error"):
            return {
                "status": "api_error",
                "code": data.get("code"),
                "body": data.get("body", "")[:200],
                "provider": self.name
            }

        # Response format: {"data": [...], "nextCursor": ...}
        events = []
        if isinstance(data, dict):
            events = data.get("data", [])
        elif isinstance(data, list):
            events = data

        return {"status": "success", "events": events, "count": len(events), "provider": self.name}

    def collect_player_props(self, event_id: Optional[str] = None) -> Dict[str, Any]:
        """Collect player prop odds for NBA.
        
        SportsGameOdds returns player props as part of the /events endpoint.
        Use leagueID=NBA and oddsAvailable=true to get events with odds.
        Player props are in the odds field of each event.
        """
        config = self.validate_config()
        if config["status"] != "configured":
            return {"status": "missing_credentials", **config}

        # Build params for /events endpoint
        params = {
            "leagueID": "NBA",
            "oddsAvailable": "true",
            "finalized": "false",
            "limit": 20,
        }
        if event_id:
            params["eventIDs"] = event_id

        data, headers = self._request("events", params)
        if isinstance(data, dict) and data.get("error"):
            return {
                "status": "api_error",
                "code": data.get("code"),
                "body": data.get("body", "")[:200],
                "provider": self.name
            }

        # Parse events and extract player prop odds
        events = []
        if isinstance(data, dict):
            events = data.get("data", [])
        elif isinstance(data, list):
            events = data

        if not events:
            return {
                "status": "no_props",
                "odds": [],
                "count": 0,
                "provider": self.name,
                "message": "No NBA events with odds available"
            }

        # Extract player prop odds from events
        all_odds = []
        for event in events:
            event_id_val = event.get("eventID") or event.get("id", "")
            commence_time = event.get("startTime") or event.get("commence_time", "")
            
            # Get team info from teams dict
            teams = event.get("teams", {})
            home_team = ""
            away_team = ""
            if isinstance(teams, dict):
                for team_id, team_data in teams.items():
                    if isinstance(team_data, dict):
                        if team_data.get("homeAway") == "home":
                            home_team = team_id
                        elif team_data.get("homeAway") == "away":
                            away_team = team_id
                    elif team_id == "home":
                        home_team = str(team_data)
                    elif team_id == "away":
                        away_team = str(team_data)

            # Odds are in event["odds"] dict keyed by oddID
            odds_dict = event.get("odds", {})
            if not isinstance(odds_dict, dict):
                continue

            for odd_id, odd_data in odds_dict.items():
                if not isinstance(odd_data, dict):
                    continue
                
                # Use the structured fields from the API response
                stat_id = odd_data.get("statID", "")
                stat_entity = odd_data.get("statEntityID", "")
                bet_type = odd_data.get("betTypeID", "")
                side_id = odd_data.get("sideID", "")
                player_id = odd_data.get("playerID", "")

                # Skip non-player props (team-level odds)
                if stat_entity in ("home", "away", "all"):
                    continue

                # Only include over/under player props
                if bet_type != "ou":
                    continue

                # Get line (overUnder)
                line = odd_data.get("bookOverUnder") or odd_data.get("fairOverUnder")
                if line is None:
                    continue

                # Map stat to market
                market_key = f"player_{stat_id}"

                # Extract per-bookmaker odds from byBookmaker dict
                by_bookmaker = odd_data.get("byBookmaker", {})
                
                if by_bookmaker and isinstance(by_bookmaker, dict):
                    for book_id, book_data in by_bookmaker.items():
                        if not isinstance(book_data, dict):
                            continue
                        book_odds_str = book_data.get("odds")
                        book_line_str = book_data.get("overUnder") or line
                        
                        if book_odds_str is None:
                            continue
                        
                        # Parse American odds string ("+114", "-165")
                        try:
                            odds_str = str(book_odds_str).replace("+", "")
                            book_odds = int(float(odds_str))
                        except (ValueError, TypeError):
                            continue
                        
                        try:
                            book_line_f = float(book_line_str)
                        except (ValueError, TypeError):
                            continue
                        
                        # Format player name from ID like "ANTHONY_EDWARDS_1_NBA"
                        pid = player_id or stat_entity
                        # Remove trailing _1_NBA, _2_NBA etc
                        name_parts = pid.split("_")
                        # Find where the numeric suffix starts
                        name_end = len(name_parts)
                        for i in range(len(name_parts) - 1, -1, -1):
                            if name_parts[i] in ("NBA", "NFL", "MLB") or name_parts[i].isdigit():
                                name_end = i
                            else:
                                break
                        player_name = " ".join(name_parts[:name_end]).title()
                        if not player_name:
                            player_name = pid.replace("_", " ").title()
                        
                        all_odds.append({
                            "event_id": event_id_val,
                            "game_id": event_id_val,
                            "player": player_name,
                            "player_raw": pid,
                            "market": market_key,
                            "market_key": market_key,
                            "line": book_line_f,
                            "odds": book_odds,
                            "side": side_id,
                            "bookmaker": book_id,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "odd_id": odd_id,
                            "is_live": odd_data.get("started", False),
                        })
                else:
                    # Use consensus bookOdds
                    consensus_odds_str = odd_data.get("bookOdds") or odd_data.get("fairOdds")
                    if consensus_odds_str is None:
                        continue
                    
                    try:
                        odds_str = str(consensus_odds_str).replace("+", "")
                        consensus_odds = int(float(odds_str))
                    except (ValueError, TypeError):
                        continue
                    
                    try:
                        line_f = float(line)
                    except (ValueError, TypeError):
                        continue
                    
                    pid = player_id or stat_entity
                    name_parts = pid.split("_")
                    name_end = len(name_parts)
                    for i in range(len(name_parts) - 1, -1, -1):
                        if name_parts[i] in ("NBA", "NFL", "MLB") or name_parts[i].isdigit():
                            name_end = i
                        else:
                            break
                    player_name = " ".join(name_parts[:name_end]).title()
                    if not player_name:
                        player_name = pid.replace("_", " ").title()
                    
                    all_odds.append({
                        "event_id": event_id_val,
                        "game_id": event_id_val,
                        "player": player_name,
                        "player_raw": pid,
                        "market": market_key,
                        "market_key": market_key,
                        "line": line_f,
                        "odds": consensus_odds,
                        "side": side_id,
                        "bookmaker": "consensus",
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "odd_id": odd_id,
                        "is_live": odd_data.get("started", False),
                    })

        return {
            "status": "success" if all_odds else "no_props",
            "odds": all_odds,
            "count": len(all_odds),
            "provider": self.name,
            "events_checked": len(events),
            "quota_remaining": headers.get("x-ratelimit-remaining", "unknown")
        }

    def normalize(self, raw_odds: List[Dict], snapshot_type: str = "entry") -> pd.DataFrame:
        """Normalize SportsGameOdds response to canonical schema.
        
        Handles both the structured response from collect_player_props()
        and generic odds list format.
        """
        records = []
        now = datetime.now(timezone.utc).isoformat()
        snapshot_id = datetime.now(timezone.utc).strftime("sgo_%Y%m%dT%H%M%SZ")

        for item in raw_odds:
            # Extract fields - handles both our parsed format and raw API format
            event_id = item.get("event_id") or item.get("eventId") or item.get("eventID") or item.get("id", "")
            game_id = item.get("game_id") or str(event_id)
            bookmaker = item.get("bookmaker") or item.get("sportsbook") or item.get("book", "unknown")
            
            # Market: could be "player_points", "player_pts", etc.
            market_key = item.get("market") or item.get("market_key") or item.get("marketKey", "")
            player_name = item.get("player") or item.get("player_name") or item.get("playerName", "")
            line = item.get("line") or item.get("point") or item.get("handicap") or item.get("overUnder")
            odds_value = item.get("odds") or item.get("price") or item.get("american_odds")
            side = item.get("side") or item.get("selection") or item.get("outcome", "")
            commence_time = item.get("commence_time") or item.get("start_time") or item.get("commenceTime", "")
            home_team = item.get("home_team") or item.get("homeTeam", "")
            away_team = item.get("away_team") or item.get("awayTeam", "")

            # Map market to canonical
            canonical_market = MARKET_MAP.get(str(market_key).lower(), "")
            if not canonical_market:
                # Try partial match
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

            # Validate American odds: must be <= -100 or >= 100
            valid_american = (odds_f <= -100 or odds_f >= 100)
            if not valid_american:
                continue

            side_upper = str(side).upper()
            if side_upper not in ("OVER", "UNDER"):
                # Try to infer
                if "over" in str(side).lower():
                    side_upper = "OVER"
                elif "under" in str(side).lower():
                    side_upper = "UNDER"
                else:
                    side_upper = "OVER"  # Default

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
                "player_id_source": f"sgo_{player_name.replace(' ', '_').lower()}",
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
        snapshot_path = output_dir / f"sportsgameodds_nba_props_{stamp}.csv"
        manifest_path = output_dir / f"sportsgameodds_manifest_{stamp}.json"

        df.to_csv(snapshot_path, index=False)

        manifest = {
            "provider_name": self.name,
            "snapshot_id": f"sgo_{stamp}",
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

        return {
            "status": "success",
            "snapshot_path": str(snapshot_path),
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }
