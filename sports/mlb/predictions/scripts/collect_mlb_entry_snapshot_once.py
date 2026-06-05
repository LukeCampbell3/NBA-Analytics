#!/usr/bin/env python3
"""
MLB Entry Snapshot One-Shot Collection

Safe one-shot live collection command:
  - Calls provider_router
  - Collects current MLB prop board if credentials available
  - Normalizes rows into evidence schema
  - Writes only live_entry rows if provider data is real
  - Sets status = awaiting_close
  - Updates production_status.json
  - Never writes empty success output

Failure reasons:
  - missing_credentials
  - provider_auth_failed
  - provider_empty_response
  - no_supported_markets
  - no_mlb_games_today
  - provider_rate_limited
  - provider_unavailable
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "predictions" / "odds"))
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "predictions" / "odds" / "providers"))
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"))

# Load .env credentials before anything else
from provider_credentials import load_repo_env
load_repo_env()

MLB_SHADOW_DIR = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"
STATUS_PATH = MLB_SHADOW_DIR / "production_status.json"
CACHE_DIR = WORKSPACE / "sports" / "mlb" / "data" / "market_odds" / "production_shadow" / "cache"

SUPPORTED_MARKET_TYPES = {
    "batter_hits", "batter_total_bases", "batter_rbis", "batter_runs",
    "batter_strikeouts", "pitcher_strikeouts", "pitcher_hits_allowed",
    "pitcher_earned_runs",
}


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def collect_entry_snapshot_once() -> Dict[str, Any]:
    """One-shot entry snapshot collection.

    Returns structured result with explicit failure reasons.
    """
    from provider_router import MlbProviderRouter
    from evidence_lifecycle import collect_entry_snapshot, get_lifecycle_counts
    from evidence_schema import resolve_market_type, CANONICAL_TO_MARKET_TYPE

    now = datetime.now(timezone.utc)
    result: Dict[str, Any] = {
        "sport": "MLB",
        "collected_at": now.isoformat(),
        "success": False,
        "rows_collected": 0,
        "rows_written": 0,
        "diagnostic_only_rows": 0,
        "market_types_collected": [],
        "books_collected": [],
        "provider_used": None,
        "failure_reason": "",
        "terminal_state": "MLB_BLOCKED_PROVIDER_FAILURE",
    }

    # Get fresh odds from provider router
    router = MlbProviderRouter()
    df_odds, info = router.get_fresh_odds()

    # Determine failure reason from provider results
    if df_odds is None or df_odds.empty:
        result["failure_reason"] = _determine_failure_reason(info)
        result["terminal_state"] = info.get("terminal_status", "MLB_BLOCKED_PROVIDER_FAILURE")
        _update_status(result)
        return result

    provider_name = info.get("successful_provider", "unknown")
    result["provider_used"] = provider_name
    result["rows_collected"] = len(df_odds)

    # Separate supported vs unsupported markets
    if "market_canonical" in df_odds.columns:
        # Provider now outputs full canonical market types directly
        supported_mask = df_odds["market_canonical"].isin(SUPPORTED_MARKET_TYPES)
        supported_df = df_odds[supported_mask].copy()
        unsupported_df = df_odds[~supported_mask].copy()
    else:
        supported_df = df_odds.copy()
        unsupported_df = pd.DataFrame()

    # Log unsupported markets as diagnostic_only
    diagnostic_count = len(unsupported_df)
    result["diagnostic_only_rows"] = diagnostic_count
    if diagnostic_count > 0:
        _log_diagnostic_rows(unsupported_df, provider_name)

    if supported_df.empty:
        result["failure_reason"] = "no_supported_markets"
        result["terminal_state"] = "MLB_WAITING_FOR_FRESH_PROPS"
        _update_status(result)
        return result

    # Collect entry snapshot using lifecycle module
    entry_df, count = collect_entry_snapshot(supported_df, provider_name)

    if count == 0:
        result["failure_reason"] = "normalization_produced_zero_rows"
        result["terminal_state"] = "MLB_WAITING_FOR_FRESH_PROPS"
        _update_status(result)
        return result

    # Write provider cache
    _write_provider_cache(supported_df, provider_name)

    result["success"] = True
    result["rows_written"] = count
    result["market_types_collected"] = sorted(
        entry_df["market_type"].unique().tolist() if "market_type" in entry_df.columns else []
    )
    result["books_collected"] = sorted(
        entry_df["book"].unique().tolist() if "book" in entry_df.columns else []
    )
    result["failure_reason"] = ""
    result["terminal_state"] = "MLB_WAITING_FOR_CLOSE_LINES"

    _update_status(result)
    return result


def _determine_failure_reason(info: Dict[str, Any]) -> str:
    """Determine specific failure reason from provider info."""
    provider_results = info.get("provider_results", [])
    if not provider_results:
        return "provider_unavailable"

    # Check first provider result for specific failure
    for pr in provider_results:
        status = pr.get("provider_status", "")
        if status == "missing_credentials":
            return "missing_credentials"
        if status == "api_error":
            return "provider_auth_failed"
        if status == "no_props":
            return "no_mlb_games_today"

    return "provider_unavailable"


def _log_diagnostic_rows(df: pd.DataFrame, provider_name: str):
    """Log unsupported market rows as diagnostic_only."""
    from evidence_lifecycle import _append_dataset, DATASETS_DIR
    diag_path = DATASETS_DIR / "diagnostic_unsupported_markets.csv"
    diag_df = df.copy()
    diag_df["evidence_tier"] = "diagnostic_only"
    diag_df["row_source"] = "unsupported_market_log"
    diag_df["provider"] = provider_name
    diag_df["logged_at"] = datetime.now(timezone.utc).isoformat()
    _append_dataset(diag_path, diag_df)


def _write_provider_cache(df: pd.DataFrame, provider_name: str):
    """Write provider cache snapshot."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_path = CACHE_DIR / f"{provider_name}_mlb_cache_{stamp}.csv"
    df.to_csv(cache_path, index=False)

    meta_path = CACHE_DIR / "latest_cache_meta.json"
    meta = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider_name,
        "rows": len(df),
        "market_types": sorted(df["market_canonical"].unique().tolist()) if "market_canonical" in df.columns else [],
        "cache_version": "1.0",
        "cache_file": str(cache_path.name),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _update_status(result: Dict[str, Any]):
    """Update production_status.json with collection result."""
    try:
        from production_status_reporter import build_production_status
        build_production_status()
    except Exception:
        # Fallback: write minimal status
        status = {
            "sport": "MLB",
            "terminal_state": result["terminal_state"],
            "staking_enabled": False,
            "provider_status": result.get("provider_used") or "failed",
            "fresh_entry_rows_today": result.get("rows_written", 0),
            "failure_reason": result.get("failure_reason", ""),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2, default=_json_default), encoding="utf-8")


def main():
    result = collect_entry_snapshot_once()
    print(json.dumps(result, indent=2, default=_json_default))
    if not result["success"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
