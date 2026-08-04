#!/usr/bin/env python3
"""
SportsGameOdds MLB Provider (v2 API)

Uses the SportsGameOdds v2 /events endpoint which returns events with
inline odds data. No separate odds endpoint needed.

Auth: x-api-key header OR apiKey query param (both supported).
Filter: leagueID=MLB, oddsAvailable=true
Odds structure: Event.odds.<oddID>.byBookmaker.<bookmakerID>

oddID pattern: {statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}
Player props use betTypeID=ou, sideID=over/under, statEntityID=PLAYER_ID

Market mapping uses mlb_market_mapper for role-aware disambiguation.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odds_contract import ensure_contract

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from mlb_market_mapper import (
    MarketMappingResult,
    SUPPORTED_CANONICAL_MARKETS,
    extract_event_pitchers,
    map_market,
)

BASE_URL = "https://api.sportsgameodds.com/v2"


class SportsGameOddsMlbProvider:
    """SportsGameOdds v2 provider for MLB player props."""

    def __init__(self, api_key: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key
        else:
            from provider_credentials import get_sportsgameodds_api_key
            creds = get_sportsgameodds_api_key()
            self.api_key = creds["api_key"] or ""
        self.base_url = BASE_URL
        self._last_response_debug: Dict[str, Any] = {}
        self._accounting: Dict[str, int] = {}

    def validate_config(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "missing_credentials", "message": "SPORTSGAMEODDS_API_KEY not set"}
        return {"status": "ok"}

    def get_accounting(self) -> Dict[str, Any]:
        """Return detailed row accounting from last collection."""
        return dict(self._accounting)

    def collect_player_props(self) -> Dict[str, Any]:
        """Collect MLB player props from SportsGameOdds v2 API."""
        if not REQUESTS_AVAILABLE:
            return {"status": "api_error", "message": "requests library not available"}

        config_check = self.validate_config()
        if config_check["status"] != "ok":
            return config_check
        if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("MLB_ENABLE_LIVE_API_TESTS") != "1":
            return {
                "status": "live_api_disabled_for_tests",
                "message": "Live SportsGameOdds API calls are disabled during pytest",
            }

        self._accounting = {
            "raw_events_returned": 0,
            "raw_odds_returned": 0,
            "raw_player_props_found": 0,
            "valid_over_under_pairs": 0,
            "normalized_book_rows": 0,
            "diagnostic_only_rows": 0,
            "unsupported_market_rows": 0,
            "ambiguous_market_rows": 0,
            "malformed_rows": 0,
            "unavailable_book_rows": 0,
        }

        try:
            events = self._get_events()
            if not events:
                return {"status": "no_props", "message": "No upcoming MLB events with odds found"}

            self._accounting["raw_events_returned"] = len(events)

            all_odds: List[Dict[str, Any]] = []
            diagnostic_odds: List[Dict[str, Any]] = []

            for event in events:
                odds, diag = self._extract_player_props(event)
                all_odds.extend(odds)
                diagnostic_odds.extend(diag)

            self._accounting["raw_player_props_found"] = len(all_odds) + len(diagnostic_odds)
            self._accounting["diagnostic_only_rows"] = len(diagnostic_odds)

            if not all_odds and not diagnostic_odds:
                return {
                    "status": "no_props",
                    "message": f"No player prop odds found in {len(events)} MLB events",
                    "events_checked": len(events),
                    "accounting": self._accounting,
                }

            if not all_odds:
                return {
                    "status": "no_props",
                    "message": f"All {len(diagnostic_odds)} props were diagnostic_only (ambiguous/unsupported)",
                    "events_checked": len(events),
                    "accounting": self._accounting,
                    "diagnostic_odds": diagnostic_odds,
                }

            return {
                "status": "success",
                "odds": all_odds,
                "diagnostic_odds": diagnostic_odds,
                "events_checked": len(events),
                "accounting": self._accounting,
            }

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            body = e.response.text[:300] if e.response is not None else ""
            if code == 401:
                return {"status": "missing_credentials", "message": f"API returned 401 Unauthorized: {body}"}
            if code == 403:
                return {"status": "missing_credentials", "message": f"API returned 403 Forbidden: {body}"}
            if code == 429:
                retry_after = e.response.headers.get("Retry-After") if e.response is not None else None
                return {
                    "status": "rate_limited",
                    "message": "Rate limited (429)",
                    "code": 429,
                    "retry_after": retry_after,
                }
            return {"status": "api_error", "code": code, "body": body, "message": f"HTTP {code}: {body}"}
        except requests.exceptions.ConnectionError as e:
            return {"status": "api_error", "message": f"Connection error: {str(e)[:200]}"}
        except requests.exceptions.Timeout:
            return {"status": "api_error", "message": "Request timed out"}
        except Exception as e:
            return {"status": "api_error", "message": str(e)[:300]}

    def _get_events(self) -> List[Dict[str, Any]]:
        """Get upcoming MLB events with odds from v2 /events endpoint."""
        url = f"{self.base_url}/events"
        headers = {"x-api-key": self.api_key}
        params = {
            "leagueID": "MLB",
            "oddsAvailable": "true",
        }

        all_events: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        for _ in range(3):
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            self._last_response_debug = {
                "status_code": resp.status_code,
                "success": data.get("success"),
                "data_count": len(data.get("data", [])),
                "has_next_cursor": bool(data.get("nextCursor")),
            }

            if not data.get("success", False):
                error_msg = data.get("error", "Unknown API error")
                raise requests.exceptions.HTTPError(
                    response=type("FakeResp", (), {"status_code": 400, "text": error_msg})()
                )

            events = data.get("data", [])
            all_events.extend(events)

            # Count raw odds
            for ev in events:
                self._accounting["raw_odds_returned"] += len(ev.get("odds", {}))

            cursor = data.get("nextCursor")
            if not cursor:
                break

        return all_events

    def _extract_player_props(self, event: Dict[str, Any]) -> tuple:
        """Extract player prop odds from a single event.

        Returns (production_odds, diagnostic_odds).
        """
        odds_dict = event.get("odds", {})
        if not odds_dict:
            return [], []

        # Event metadata
        teams = event.get("teams", {})
        home_team_data = teams.get("home", {})
        away_team_data = teams.get("away", {})
        home_names = home_team_data.get("names", {})
        away_names = away_team_data.get("names", {})
        home_team = home_names.get("short", home_names.get("medium", home_team_data.get("teamID", "")))
        away_team = away_names.get("short", away_names.get("medium", away_team_data.get("teamID", "")))

        event_id = event.get("eventID", "")
        status = event.get("status", {})
        starts_at = status.get("startsAt", "")
        players_dict = event.get("players", {})

        # Detect pitchers from event context
        event_pitchers = extract_event_pitchers(event)

        production: List[Dict[str, Any]] = []
        diagnostic: List[Dict[str, Any]] = []

        for odd_id, odd_data in odds_dict.items():
            if not isinstance(odd_data, dict):
                continue

            bet_type = odd_data.get("betTypeID", "")
            if bet_type != "ou":
                continue

            stat_id = odd_data.get("statID", "")
            side_id = odd_data.get("sideID", "")
            player_id = odd_data.get("playerID", odd_data.get("statEntityID", ""))

            if not player_id or player_id in ("home", "away", "all"):
                continue

            # Get line
            line_str = odd_data.get("bookOverUnder", odd_data.get("fairOverUnder", ""))
            if not line_str:
                self._accounting["malformed_rows"] += 1
                continue
            try:
                line = float(line_str)
            except (ValueError, TypeError):
                self._accounting["malformed_rows"] += 1
                continue

            # Get player info
            player_info = players_dict.get(player_id, {})
            player_name = player_info.get("name", "")
            if not player_name:
                fn = player_info.get("firstName", "")
                ln = player_info.get("lastName", "")
                player_name = f"{fn} {ln}".strip() if (fn or ln) else player_id.replace("_", " ").title()

            player_team_id = player_info.get("teamID", "")
            if player_team_id == home_team_data.get("teamID"):
                team = home_team
                opponent = away_team
                home_away = "home"
            elif player_team_id == away_team_data.get("teamID"):
                team = away_team
                opponent = home_team
                home_away = "away"
            else:
                team = ""
                opponent = ""
                home_away = ""

            # Map market with role context
            mapping = map_market(
                raw_stat_id=stat_id,
                player_id=player_id,
                player_name=player_name,
                line=line,
                event_pitchers=event_pitchers,
                player_position="",
                player_team_id=player_team_id,
            )

            by_bookmaker = odd_data.get("byBookmaker", {})
            consensus_odds_str = odd_data.get("bookOdds", "")

            row_base = {
                "sport": "MLB",
                "league": "MLB",
                "source_event_id": event_id,
                "game_id": event_id,
                "commence_time_utc": starts_at,
                "home_team": home_team,
                "away_team": away_team,
                "player": player_name,
                "player_id_source": player_id,
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "raw_stat_id": stat_id,
                "market": stat_id,
                "market_canonical": mapping.canonical_market_type,
                "market_mapping_confidence": mapping.confidence,
                "market_mapping_reason": mapping.reason,
                "line": line,
                "side": side_id,
                "consensus_odds": consensus_odds_str,
                "by_bookmaker": by_bookmaker,
                "is_live": bool(status.get("live", False)),
                "odd_id": odd_id,
                "raw_json_ref": json.dumps(odd_data)[:500],
            }

            if mapping.is_production:
                production.append(row_base)
            elif mapping.confidence == "ambiguous":
                self._accounting["ambiguous_market_rows"] += 1
                row_base["evidence_tier"] = "diagnostic_only"
                row_base["failure_reason"] = mapping.reason
                diagnostic.append(row_base)
            else:
                self._accounting["unsupported_market_rows"] += 1
                row_base["evidence_tier"] = "diagnostic_only"
                row_base["failure_reason"] = f"unsupported_market:{stat_id}"
                diagnostic.append(row_base)

        return production, diagnostic

    def normalize(self, raw_odds: List[Dict[str, Any]]) -> pd.DataFrame:
        """Normalize raw odds into canonical DataFrame with one row per book/side.

        Only produces rows with valid American odds from available bookmakers.
        """
        rows = []
        snapshot_time = datetime.now(timezone.utc).isoformat()

        for entry in raw_odds:
            by_bookmaker = entry.get("by_bookmaker", {})
            side = entry.get("side", "over")
            consensus_odds_str = entry.get("consensus_odds", "")

            entry_rows_added = 0

            if by_bookmaker:
                for book_name, book_data in by_bookmaker.items():
                    if not isinstance(book_data, dict):
                        continue
                    if book_data.get("available") is False:
                        self._accounting["unavailable_book_rows"] = self._accounting.get("unavailable_book_rows", 0) + 1
                        continue

                    odds_str = book_data.get("odds", "")
                    if not odds_str:
                        self._accounting["malformed_rows"] = self._accounting.get("malformed_rows", 0) + 1
                        continue

                    try:
                        odds_val = int(odds_str) if odds_str.lstrip("+-").isdigit() else float(odds_str)
                    except (ValueError, TypeError):
                        self._accounting["malformed_rows"] = self._accounting.get("malformed_rows", 0) + 1
                        continue

                    rows.append(self._build_row(entry, book_name, side, odds_val, snapshot_time))
                    entry_rows_added += 1

            elif consensus_odds_str:
                try:
                    odds_val = int(consensus_odds_str) if consensus_odds_str.lstrip("+-").isdigit() else float(consensus_odds_str)
                except (ValueError, TypeError):
                    self._accounting["malformed_rows"] = self._accounting.get("malformed_rows", 0) + 1
                    continue
                rows.append(self._build_row(entry, "consensus", side, odds_val, snapshot_time))
                entry_rows_added += 1

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")

        # Filter valid American odds
        pre_filter = len(df)
        df = df[((df["odds"] <= -100) | (df["odds"] >= 100)) & df["odds"].notna()].copy()
        self._accounting["malformed_rows"] = self._accounting.get("malformed_rows", 0) + (pre_filter - len(df))

        df["valid_american_odds"] = True
        df["schema_valid"] = True
        self._accounting["normalized_book_rows"] = len(df)
        return ensure_contract(
            df,
            source="sportsgameodds",
            acquisition_method="api",
            source_endpoint=f"{self.base_url}/events",
            parser_version="sportsgameodds-v2-parser-v1",
        )

    def _build_row(self, entry: Dict[str, Any], book: str, side: str,
                   odds_value: Any, snapshot_time: str) -> Dict[str, Any]:
        """Build a single canonical row."""
        return {
            "sport": "MLB",
            "league": "MLB",
            "provider_name": "sportsgameodds",
            "source_event_id": entry.get("source_event_id", ""),
            "game_id": entry.get("game_id", ""),
            "snapshot_id": "",
            "snapshot_time_utc": snapshot_time,
            "snapshot_type": "entry",
            "commence_time_utc": entry.get("commence_time_utc", ""),
            "home_team": entry.get("home_team", ""),
            "away_team": entry.get("away_team", ""),
            "player": entry.get("player", ""),
            "player_id_source": entry.get("player_id_source", ""),
            "team": entry.get("team", ""),
            "opponent": entry.get("opponent", ""),
            "home_away": entry.get("home_away", ""),
            "raw_stat_id": entry.get("raw_stat_id", entry.get("market", "")),
            "market": entry.get("market", ""),
            "market_canonical": entry.get("market_canonical", ""),
            "market_mapping_confidence": entry.get("market_mapping_confidence", ""),
            "market_mapping_reason": entry.get("market_mapping_reason", ""),
            "line": entry.get("line"),
            "book": book,
            "side": side,
            "odds": odds_value,
            "over_odds": odds_value if side == "over" else None,
            "under_odds": odds_value if side == "under" else None,
            "is_live": entry.get("is_live", False),
            "raw_json_ref": entry.get("raw_json_ref", ""),
            "valid_american_odds": False,
            "schema_valid": False,
        }
