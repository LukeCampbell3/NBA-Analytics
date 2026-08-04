#!/usr/bin/env python3
"""Health report for the configured provider-neutral MLB odds chain."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "providers"))
HEALTH_REPORT_PATH = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow" / "provider_health_report.json"


def check_provider_health(provider_name: str = "provider_chain") -> dict[str, Any]:
    from provider_router import MlbProviderRouter

    priority = None if provider_name in {"provider_chain", "auto", ""} else [provider_name, "fresh_cache"]
    frame, info = MlbProviderRouter(provider_priority=priority).get_fresh_odds()
    results = info.get("provider_results", [])
    report: dict[str, Any] = {
        "provider_name": provider_name,
        "provider_priority": info.get("provider_priority", []),
        "successful_provider": info.get("successful_provider"),
        "successful_providers": info.get("successful_providers", []),
        "credentials_present": (
            any(row.get("provider_status") == "success" for row in results)
            or not any(row.get("provider_status") == "missing_credentials" for row in results)
        ),
        "auth_success": any(row.get("provider_status") == "success" for row in results),
        "request_success": frame is not None and not frame.empty,
        "rows_returned": int(len(frame)) if frame is not None else 0,
        "normalized_rows_returned": int(len(frame)) if frame is not None else 0,
        "market_types_returned": sorted(frame["market_type"].dropna().unique().tolist()) if frame is not None and not frame.empty else [],
        "books_returned": sorted(frame["sportsbook"].dropna().unique().tolist()) if frame is not None and not frame.empty else [],
        "provider_results": results,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "failure_reason": None if frame is not None and not frame.empty else info.get("error"),
        "terminal_state": info.get("terminal_status", "MLB_WAITING_FOR_FRESH_PROPS"),
        "source_state": info.get("source_state", ""),
    }
    HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB provider-neutral odds healthcheck")
    parser.add_argument("--provider", default="provider_chain")
    args = parser.parse_args()
    report = check_provider_health(args.provider)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["request_success"] else 1)


if __name__ == "__main__":
    main()
