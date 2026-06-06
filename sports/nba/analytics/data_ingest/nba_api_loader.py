"""
NBA API Data Loader

Rate-limit-safe ingestion from nba_api endpoints:
  - PlayerDashboardByGeneralSplits (traditional + advanced)
  - ShotChartDetail (shot locations)
  - PlayerDashPtShotLog / tracking endpoints (CatchShoot, PullUp, Drives, Passing)
  - LeagueDashPlayerStats (league-wide baselines)

Implements retries, backoff, source caching, and stale-data detection.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
CACHE_DIR = WORKSPACE / "sports" / "nba" / "analytics" / "data" / "cache"
RAW_DIR = WORKSPACE / "sports" / "nba" / "analytics" / "data" / "raw"

# Rate limiting
REQUEST_DELAY = 0.8  # seconds between requests
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

try:
    from nba_api.stats.endpoints import (
        leaguedashplayerstats,
        playerdashboardbygeneralsplits,
        shotchartdetail,
        commonplayerinfo,
    )
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False


def _ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(endpoint: str, params: str) -> Path:
    safe_name = f"{endpoint}_{params}".replace("/", "_").replace(" ", "_")[:100]
    return CACHE_DIR / f"{safe_name}.json"


def _is_cache_fresh(path: Path, max_age_hours: float = 24) -> bool:
    if not path.exists():
        return False
    age = (datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
    return age.total_seconds() < max_age_hours * 3600


def _request_with_retry(fetch_fn, retries: int = MAX_RETRIES) -> Optional[Any]:
    """Execute a fetch function with retries and backoff."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            result = fetch_fn()
            return result
        except Exception as e:
            if attempt < retries - 1:
                wait = REQUEST_DELAY * (RETRY_BACKOFF ** attempt)
                time.sleep(wait)
            else:
                print(f"  Failed after {retries} attempts: {str(e)[:100]}")
                return None
    return None


def fetch_league_stats(season: str = "2025-26", per_mode: str = "PerGame") -> Optional[pd.DataFrame]:
    """Fetch league-wide player stats for percentile baselines.

    Returns DataFrame with all players' traditional + advanced stats.
    """
    if not NBA_API_AVAILABLE:
        return _load_fallback_league_stats(season)

    _ensure_dirs()
    cache_key = f"league_stats_{season}_{per_mode}"
    cache = _cache_path("league_stats", cache_key)

    if _is_cache_fresh(cache):
        return pd.read_json(cache)

    def _fetch():
        data = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            timeout=30,
        )
        return data.get_data_frames()[0]

    df = _request_with_retry(_fetch)
    if df is not None and not df.empty:
        df.to_json(cache, orient="records")
    return df


def fetch_player_shooting(player_id: int, season: str = "2025-26") -> Optional[pd.DataFrame]:
    """Fetch shot chart detail for a player."""
    if not NBA_API_AVAILABLE:
        return None

    _ensure_dirs()
    cache = _cache_path("shots", f"{player_id}_{season}")
    if _is_cache_fresh(cache):
        return pd.read_json(cache)

    def _fetch():
        data = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            team_id=0,
            season_nullable=season,
            context_measure_simple="FGA",
            timeout=30,
        )
        return data.get_data_frames()[0]

    df = _request_with_retry(_fetch)
    if df is not None and not df.empty:
        df.to_json(cache, orient="records")
    return df


def fetch_tracking_catch_shoot(season: str = "2025-26") -> Optional[pd.DataFrame]:
    """Fetch catch-and-shoot tracking data for all players."""
    if not NBA_API_AVAILABLE:
        return None

    _ensure_dirs()
    cache = _cache_path("tracking", f"catch_shoot_{season}")
    if _is_cache_fresh(cache):
        return pd.read_json(cache)

    try:
        from nba_api.stats.endpoints import leaguedashptstats
        def _fetch():
            data = leaguedashptstats.LeagueDashPtStats(
                season=season,
                per_mode_simple="PerGame",
                pt_measure_type="CatchShoot",
                timeout=30,
            )
            return data.get_data_frames()[0]

        df = _request_with_retry(_fetch)
        if df is not None and not df.empty:
            df.to_json(cache, orient="records")
        return df
    except (ImportError, Exception) as e:
        print(f"  Tracking CatchShoot unavailable: {e}")
        return None


def fetch_tracking_drives(season: str = "2025-26") -> Optional[pd.DataFrame]:
    """Fetch drive tracking data."""
    if not NBA_API_AVAILABLE:
        return None

    _ensure_dirs()
    cache = _cache_path("tracking", f"drives_{season}")
    if _is_cache_fresh(cache):
        return pd.read_json(cache)

    try:
        from nba_api.stats.endpoints import leaguedashptstats
        def _fetch():
            data = leaguedashptstats.LeagueDashPtStats(
                season=season,
                per_mode_simple="PerGame",
                pt_measure_type="Drives",
                timeout=30,
            )
            return data.get_data_frames()[0]

        df = _request_with_retry(_fetch)
        if df is not None and not df.empty:
            df.to_json(cache, orient="records")
        return df
    except (ImportError, Exception) as e:
        print(f"  Tracking Drives unavailable: {e}")
        return None


def fetch_tracking_defense(season: str = "2025-26") -> Optional[pd.DataFrame]:
    """Fetch defensive tracking data."""
    if not NBA_API_AVAILABLE:
        return None

    _ensure_dirs()
    cache = _cache_path("tracking", f"defense_{season}")
    if _is_cache_fresh(cache):
        return pd.read_json(cache)

    try:
        from nba_api.stats.endpoints import leaguedashptstats
        def _fetch():
            data = leaguedashptstats.LeagueDashPtStats(
                season=season,
                per_mode_simple="PerGame",
                pt_measure_type="Defense",
                timeout=30,
            )
            return data.get_data_frames()[0]

        df = _request_with_retry(_fetch)
        if df is not None and not df.empty:
            df.to_json(cache, orient="records")
        return df
    except (ImportError, Exception) as e:
        print(f"  Tracking Defense unavailable: {e}")
        return None


def _load_fallback_league_stats(season: str) -> Optional[pd.DataFrame]:
    """Load from existing repo raw data as fallback when nba_api unavailable."""
    year = season.split("-")[0]
    fallback_paths = [
        WORKSPACE / "data" / "raw" / f"nba_base_{year}.csv",
        WORKSPACE / "data" / "raw" / f"nba_base_{int(year)+1}.csv",
    ]
    for p in fallback_paths:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None


def get_data_freshness_report() -> Dict[str, Any]:
    """Report cache freshness for all tracked endpoints."""
    _ensure_dirs()
    report = {}
    for f in CACHE_DIR.glob("*.json"):
        age_hours = (datetime.now(timezone.utc) - datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600
        report[f.stem] = {
            "path": str(f),
            "age_hours": round(age_hours, 1),
            "fresh": age_hours < 24,
        }
    return report
