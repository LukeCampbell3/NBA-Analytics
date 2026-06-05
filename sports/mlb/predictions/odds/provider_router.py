#!/usr/bin/env python3
"""
MLB Provider Router

Resilient provider flow:
  primary_provider -> fallback_provider -> stale_cache_readonly -> no_play

Requirements:
  - Primary provider tried first.
  - Fallback providers through common interface.
  - If no live provider works, read fresh-enough cache in readonly mode.
  - If no cache is valid, emit explicit no-play status.
  - Never silently return empty evidence.
  - Every provider result includes standardized fields.

Does NOT require The Odds API.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
MLB_SHADOW_DIR = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"
ODDS_STATUS_PATH = MLB_SHADOW_DIR / "odds_source_status.json"
SNAPSHOT_DIR = WORKSPACE / "sports" / "mlb" / "data" / "market_odds" / "production_shadow" / "snapshots"
PROVIDER_CONFIG_PATH = MLB_SHADOW_DIR / "mlb_provider_config.json"

sys.path.insert(0, str(Path(__file__).parent / "providers"))

PROVIDERS_AVAILABLE: Dict[str, Any] = {}

try:
    from sportsgameodds_mlb_provider import SportsGameOddsMlbProvider
    PROVIDERS_AVAILABLE["sportsgameodds"] = SportsGameOddsMlbProvider
except ImportError:
    pass


def validate_normalized_odds(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate a normalized MLB odds DataFrame."""
    if df.empty:
        return {
            "rows": 0, "valid_rows": 0, "invalid_rows": 0,
            "valid_odds_rate": 0.0, "missing_required_fields": [],
            "invalid_odds_count": 0, "markets": [], "books": [],
            "players": 0, "events": 0,
        }

    n = len(df)
    required_fields = ["provider_name", "player", "market_canonical", "line", "book", "side", "odds", "snapshot_time_utc"]
    missing = [f for f in required_fields if f not in df.columns]

    invalid_mask = pd.Series([False] * n, index=df.index)
    for field in required_fields:
        if field in df.columns:
            invalid_mask |= df[field].isna()

    if "odds" in df.columns:
        odds_invalid = ~((df["odds"] <= -100) | (df["odds"] >= 100))
        invalid_mask |= odds_invalid
        invalid_odds_count = int(odds_invalid.sum())
    else:
        invalid_odds_count = n

    if "line" in df.columns:
        invalid_mask |= pd.to_numeric(df["line"], errors="coerce").isna()

    valid_rows = int((~invalid_mask).sum())
    invalid_rows = int(invalid_mask.sum())
    valid_odds_rate = valid_rows / n if n > 0 else 0.0

    return {
        "rows": n,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "valid_odds_rate": float(valid_odds_rate),
        "missing_required_fields": missing,
        "invalid_odds_count": invalid_odds_count,
        "markets": df["market_canonical"].unique().tolist() if "market_canonical" in df.columns else [],
        "books": df["book"].unique().tolist() if "book" in df.columns else [],
        "players": int(df["player"].nunique()) if "player" in df.columns else 0,
        "events": int(df["game_id"].nunique()) if "game_id" in df.columns else 0,
    }


def build_provider_result(
    provider_name: str,
    status: str,
    *,
    is_live: bool = False,
    is_cache: bool = False,
    cache_age_minutes: float = 0.0,
    failure_reason: str = "",
    rows_collected: int = 0,
    valid_rows: int = 0,
    markets_covered: List[str] | None = None,
    retry_after: str | None = None,
) -> Dict[str, Any]:
    """Build a standardized provider result dict. Never silently empty."""
    result = {
        "provider_name": provider_name,
        "provider_status": status,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "is_live": is_live,
        "is_cache": is_cache,
        "cache_age_minutes": cache_age_minutes,
        "failure_reason": failure_reason,
        "rows_collected": rows_collected,
        "valid_rows": valid_rows,
        "markets_covered": markets_covered or [],
    }
    if retry_after is not None:
        result["retry_after"] = retry_after
    return result


class MlbProviderRouter:
    """Router for MLB odds providers with priority system.

    Flow: primary -> fallback -> stale_cache_readonly -> no_play
    """

    def __init__(self, max_cache_age_seconds: int = 3600):
        self.config = self._load_config()
        self.max_cache_age = self.config.get("freshness_limit_seconds", max_cache_age_seconds)
        self.min_valid_odds_rate = self.config.get("min_valid_odds_rate", 0.70)
        self.fallback_order = self.config.get("fallback_order", [
            "sportsgameodds", "propline", "odds_api_io", "oddspapi", "fresh_cache"
        ])
        self.attempts: List[Dict[str, Any]] = []
        self.provider_results: List[Dict[str, Any]] = []

    def _load_config(self) -> Dict[str, Any]:
        if PROVIDER_CONFIG_PATH.exists():
            try:
                return json.loads(PROVIDER_CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "fallback_order": ["sportsgameodds", "propline", "odds_api_io", "oddspapi", "fresh_cache"],
            "freshness_limit_seconds": 3600,
            "min_valid_odds_rate": 0.70,
        }

    def get_fresh_odds(self) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Get fresh MLB odds using priority system.

        Returns (dataframe_or_None, info_dict).
        info_dict always contains provider_results with explicit status for each attempt.
        Never silently returns empty — always emits no_play status on total failure.
        """
        self.attempts = []
        self.provider_results = []
        full_info: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sport": "MLB",
            "fallback_order": self.fallback_order,
            "providers_tried": [],
            "provider_results": [],
            "successful_provider": None,
            "rows_obtained": 0,
            "no_fresh_odds_available": False,
            "snapshot_age_seconds": 0,
            "valid_odds_rate": 0.0,
            "terminal_status": "",
        }
        rate_limited = False

        for provider_name in self.fallback_order:
            if rate_limited and provider_name != "fresh_cache":
                continue
            attempt = self._try_provider(provider_name)
            self.attempts.append(attempt)
            full_info["providers_tried"].append({k: v for k, v in attempt.items() if k != "_dataframe"})

            # Build standardized provider result
            pr = build_provider_result(
                provider_name=provider_name,
                status=attempt["status"],
                is_live=(attempt["status"] == "success" and provider_name != "fresh_cache" and not rate_limited),
                is_cache=(provider_name == "fresh_cache" and attempt["status"] == "success"),
                cache_age_minutes=attempt.get("snapshot_age_seconds", 0) / 60.0,
                failure_reason=attempt.get("error", ""),
                rows_collected=attempt.get("rows_raw", 0),
                valid_rows=attempt.get("rows_normalized", 0),
                markets_covered=attempt.get("_markets", []),
                retry_after=attempt.get("retry_after"),
            )
            self.provider_results.append(pr)
            full_info["provider_results"].append(pr)

            if attempt["status"] == "success":
                full_info["successful_provider"] = provider_name
                full_info["rows_obtained"] = attempt["rows_normalized"]
                full_info["snapshot_age_seconds"] = attempt.get("snapshot_age_seconds", 0)
                full_info["valid_odds_rate"] = attempt.get("valid_odds_rate", 1.0)
                full_info["terminal_status"] = "MLB_BLOCKED_PROVIDER_RATE_LIMIT" if rate_limited else "MLB_ENTRY_COLLECTION_ACTIVE"

                df = attempt.get("_dataframe")
                if df is not None and not df.empty:
                    df = df.copy()
                    df["provider_status"] = attempt["status"]
                    df["is_cache"] = provider_name == "fresh_cache"
                    df["is_live"] = provider_name != "fresh_cache" and not rate_limited
                    self._write_snapshot(df, provider_name)
                    self._save_status(full_info)
                    return df, full_info
            if attempt["status"] == "rate_limited":
                rate_limited = True
                full_info["no_fresh_odds_available"] = True
                full_info["terminal_status"] = "MLB_BLOCKED_PROVIDER_RATE_LIMIT"
                full_info["retry_after"] = attempt.get("retry_after")
                full_info["error"] = attempt.get("error", "Provider rate limited")
                continue

        # All providers failed — explicit no-play
        full_info["no_fresh_odds_available"] = True
        full_info["terminal_status"] = "MLB_BLOCKED_PROVIDER_RATE_LIMIT" if rate_limited else "MLB_BLOCKED_PROVIDER_FAILURE"
        full_info["error"] = "All MLB providers failed — no_play status"
        if rate_limited:
            full_info["error"] = "Provider rate limited"
        self._save_status(full_info)
        return None, full_info

    def _try_provider(self, provider_name: str) -> Dict[str, Any]:
        attempt: Dict[str, Any] = {
            "provider": provider_name,
            "status": "failed",
            "rows_raw": 0,
            "rows_normalized": 0,
            "valid_odds_rate": 0.0,
            "snapshot_age_seconds": 0,
            "error": "",
            "retry_after": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "_dataframe": None,
            "_markets": [],
        }

        if provider_name == "fresh_cache":
            return self._try_fresh_cache(attempt)

        if provider_name in ("propline", "odds_api_io", "oddspapi"):
            attempt["status"] = "not_installed"
            attempt["error"] = f"MLB adapter for {provider_name} not yet implemented"
            return attempt

        provider_class = PROVIDERS_AVAILABLE.get(provider_name)
        if provider_class is None:
            attempt["status"] = "not_installed"
            attempt["error"] = f"Provider module not available: {provider_name}"
            return attempt

        try:
            provider = provider_class()
            config_result = provider.validate_config()
            if config_result.get("status") in ("missing_credentials", "missing_api_key"):
                attempt["status"] = "missing_credentials"
                attempt["error"] = config_result.get("message", f"Missing API key for {provider_name}")
                return attempt

            result = provider.collect_player_props()

            if result.get("status") == "missing_credentials":
                attempt["status"] = "missing_credentials"
                attempt["error"] = result.get("message", "Missing credentials")
                return attempt
            if result.get("status") == "api_error":
                attempt["status"] = "api_error"
                attempt["error"] = f"API error: {result.get('message', '')[:100]}"
                return attempt
            if result.get("status") == "rate_limited":
                attempt["status"] = "rate_limited"
                attempt["error"] = "Provider rate limited (HTTP 429)"
                attempt["retry_after"] = result.get("retry_after")
                return attempt
            if result.get("status") == "live_api_disabled_for_tests":
                attempt["status"] = "live_api_disabled_for_tests"
                attempt["error"] = result.get("message", "Live API disabled during pytest")
                return attempt
            if result.get("status") == "no_props":
                attempt["status"] = "no_props"
                attempt["error"] = "No MLB player props available from this provider"
                return attempt
            if result.get("status") != "success":
                attempt["status"] = "unexpected"
                attempt["error"] = f"Unexpected: {result.get('status')}"
                return attempt

            raw_odds = result.get("odds", [])
            attempt["rows_raw"] = len(raw_odds)
            if not raw_odds:
                attempt["status"] = "no_props"
                attempt["error"] = "Empty odds response"
                return attempt

            df = provider.normalize(raw_odds)
            if df.empty:
                attempt["status"] = "normalization_failed"
                attempt["error"] = "Normalization produced 0 rows"
                return attempt

            validation = validate_normalized_odds(df)
            attempt["rows_normalized"] = validation["valid_rows"]
            attempt["valid_odds_rate"] = validation["valid_odds_rate"]
            attempt["_markets"] = validation.get("markets", [])

            if validation["valid_odds_rate"] < self.min_valid_odds_rate:
                attempt["status"] = "below_valid_odds_threshold"
                attempt["error"] = f"Valid odds rate {validation['valid_odds_rate']:.3f} < {self.min_valid_odds_rate}"
                return attempt

            if validation["invalid_rows"] > 0:
                valid_mask = (df["odds"] <= -100) | (df["odds"] >= 100)
                df = df[valid_mask].copy()

            attempt["status"] = "success"
            attempt["_dataframe"] = df
            attempt["snapshot_age_seconds"] = 0
            return attempt

        except Exception as e:
            attempt["status"] = "exception"
            attempt["error"] = str(e)[:200]
            return attempt

    def _try_fresh_cache(self, attempt: Dict[str, Any]) -> Dict[str, Any]:
        cache_files = sorted(SNAPSHOT_DIR.glob("*.csv")) if SNAPSHOT_DIR.exists() else []
        if not cache_files:
            attempt["status"] = "no_cache"
            attempt["error"] = "No MLB cache files found"
            return attempt

        latest = cache_files[-1]
        try:
            df = pd.read_csv(latest)
            if df.empty:
                attempt["status"] = "empty_cache"
                attempt["error"] = "Cache file is empty"
                return attempt

            if "snapshot_time_utc" in df.columns:
                ts = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
                latest_ts = ts.max()
                if pd.notna(latest_ts):
                    age = (datetime.now(timezone.utc) - latest_ts).total_seconds()
                    attempt["snapshot_age_seconds"] = int(age)
                    if age > self.max_cache_age:
                        attempt["status"] = "stale_cache"
                        attempt["error"] = f"Cache stale: age {age:.0f}s > limit {self.max_cache_age}s"
                        return attempt

            attempt["rows_raw"] = len(df)
            attempt["rows_normalized"] = len(df)
            attempt["valid_odds_rate"] = 1.0
            attempt["status"] = "success"
            attempt["_dataframe"] = df
            return attempt
        except Exception as e:
            attempt["status"] = "exception"
            attempt["error"] = str(e)[:200]
            return attempt

    def _write_snapshot(self, df: pd.DataFrame, provider_name: str):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = SNAPSHOT_DIR / f"{provider_name}_mlb_props_{stamp}.csv"
        df.to_csv(path, index=False)

    def _save_status(self, full_info: Dict[str, Any]):
        if os.environ.get("PYTEST_CURRENT_TEST") and ODDS_STATUS_PATH == MLB_SHADOW_DIR / "odds_source_status.json":
            return
        ODDS_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        status = {k: v for k, v in full_info.items() if k != "_dataframe"}
        ODDS_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
