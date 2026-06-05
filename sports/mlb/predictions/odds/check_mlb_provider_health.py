#!/usr/bin/env python3
"""
MLB Provider Healthcheck

Reports provider readiness for live collection:
  - credentials_present
  - auth_success
  - request_success
  - rows_returned
  - market_types_returned
  - books_returned
  - collected_at
  - failure_reason
  - terminal_state recommendation

If SPORTSGAMEODDS_API_KEY is missing:
  terminal_state = MLB_BLOCKED_PROVIDER_FAILURE
  failure_reason = missing_credentials
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent / "providers"))
sys.path.insert(0, str(Path(__file__).parent))

HEALTH_REPORT_PATH = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow" / "provider_health_report.json"


def _json_default(v: Any) -> Any:
    import numpy as np
    import pandas as pd
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def check_provider_health(provider_name: str = "sportsgameodds") -> Dict[str, Any]:
    """Check provider health and return structured report."""
    now = datetime.now(timezone.utc).isoformat()

    report: Dict[str, Any] = {
        "provider_name": provider_name,
        "credentials_present": False,
        "auth_success": False,
        "request_success": False,
        "rows_returned": 0,
        "raw_events_returned": 0,
        "raw_odds_returned": 0,
        "raw_player_props_found": 0,
        "normalized_rows_returned": 0,
        "diagnostic_only_rows": 0,
        "supported_market_counts": {},
        "unsupported_market_counts": {},
        "ambiguous_market_counts": 0,
        "market_types_returned": [],
        "books_returned": [],
        "collected_at": now,
        "failure_reason": "unknown",
        "terminal_state": "MLB_BLOCKED_PROVIDER_FAILURE",
    }

    # Check credentials
    if provider_name == "sportsgameodds":
        from provider_credentials import get_sportsgameodds_api_key
        creds = get_sportsgameodds_api_key()
        if not creds["credentials_present"]:
            report["failure_reason"] = "missing_credentials"
            report["terminal_state"] = "MLB_BLOCKED_PROVIDER_FAILURE"
            report["key_source"] = creds["key_source"]
            _write_report(report)
            return report
        report["credentials_present"] = True
        report["key_source"] = creds["key_source"]
        report["key_length"] = creds["key_length"]
        api_key = creds["api_key"]
    else:
        report["failure_reason"] = f"unsupported_provider: {provider_name}"
        _write_report(report)
        return report

    # Try to instantiate and validate
    try:
        from sportsgameodds_mlb_provider import SportsGameOddsMlbProvider
        provider = SportsGameOddsMlbProvider(api_key=api_key)
        config_result = provider.validate_config()
        if config_result.get("status") != "ok":
            report["failure_reason"] = f"provider_config_invalid: {config_result.get('message', '')}"
            _write_report(report)
            return report
        report["auth_success"] = True
    except ImportError:
        report["failure_reason"] = "provider_module_not_available"
        _write_report(report)
        return report
    except Exception as e:
        report["failure_reason"] = f"provider_init_error: {str(e)[:200]}"
        _write_report(report)
        return report

    # Try to collect props
    try:
        if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("MLB_ENABLE_LIVE_API_TESTS") != "1":
            report["failure_reason"] = "live_api_disabled_for_tests"
            report["terminal_state"] = "MLB_WAITING_FOR_FRESH_PROPS"
            _write_report(report)
            return report

        result = provider.collect_player_props()

        if result.get("status") == "missing_credentials":
            report["failure_reason"] = "provider_auth_failed"
            _write_report(report)
            return report
        elif result.get("status") == "api_error":
            report["failure_reason"] = f"provider_api_error: {result.get('message', result.get('body', ''))[:100]}"
            _write_report(report)
            return report
        elif result.get("status") == "rate_limited":
            report["failure_reason"] = "provider_rate_limited"
            report["terminal_state"] = "MLB_BLOCKED_PROVIDER_RATE_LIMIT"
            report["retry_after"] = result.get("retry_after")
            _write_report(report)
            return report
        elif result.get("status") == "no_props":
            report["request_success"] = True
            report["failure_reason"] = "no_mlb_games_today"
            report["terminal_state"] = "MLB_WAITING_FOR_FRESH_PROPS"
            _write_report(report)
            return report
        elif result.get("status") != "success":
            report["failure_reason"] = f"provider_unexpected: {result.get('status')}"
            _write_report(report)
            return report

        report["request_success"] = True
        raw_odds = result.get("odds", [])

        if not raw_odds:
            report["failure_reason"] = "provider_empty_response"
            report["terminal_state"] = "MLB_WAITING_FOR_FRESH_PROPS"
            _write_report(report)
            return report

        # Normalize to get market/book info
        df = provider.normalize(raw_odds)
        if df.empty:
            report["failure_reason"] = "normalization_produced_zero_rows"
            report["terminal_state"] = "MLB_WAITING_FOR_FRESH_PROPS"
            _write_report(report)
            return report

        report["rows_returned"] = len(df)
        report["normalized_rows_returned"] = len(df)
        report["market_types_returned"] = sorted(df["market_canonical"].unique().tolist()) if "market_canonical" in df.columns else []
        report["books_returned"] = sorted(df["book"].unique().tolist()) if "book" in df.columns else []
        report["failure_reason"] = None  # null on success, not empty string
        report["terminal_state"] = "MLB_ENTRY_COLLECTION_ACTIVE"

        # Accounting from provider
        accounting = provider.get_accounting() if hasattr(provider, 'get_accounting') else {}
        report["raw_events_returned"] = accounting.get("raw_events_returned", 0)
        report["raw_odds_returned"] = accounting.get("raw_odds_returned", 0)
        report["raw_player_props_found"] = accounting.get("raw_player_props_found", len(raw_odds))
        report["diagnostic_only_rows"] = accounting.get("diagnostic_only_rows", 0)
        report["ambiguous_market_counts"] = accounting.get("ambiguous_market_rows", 0)
        report["unsupported_market_counts"] = {"total": accounting.get("unsupported_market_rows", 0)}

        # Supported market counts
        if "market_canonical" in df.columns:
            report["supported_market_counts"] = df["market_canonical"].value_counts().to_dict()

    except Exception as e:
        report["failure_reason"] = f"provider_unavailable: {str(e)[:200]}"

    _write_report(report)
    return report


def _write_report(report: Dict[str, Any]):
    HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MLB Provider Healthcheck")
    parser.add_argument("--provider", default="sportsgameodds")
    args = parser.parse_args()

    report = check_provider_health(args.provider)
    print(json.dumps(report, indent=2, default=_json_default))

    if report["terminal_state"] == "MLB_BLOCKED_PROVIDER_FAILURE":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
