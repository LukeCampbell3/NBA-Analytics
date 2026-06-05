#!/usr/bin/env python3
"""
MLB Close Snapshot One-Shot Collection

Collects the current prop board and matches close rows to prior entry rows.
Computes line movement and CLV only when both entry and close are real market rows.

If no entry rows exist, exits with:
  terminal_state = MLB_WAITING_FOR_FRESH_PROPS
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
LEDGER_PATH = MLB_SHADOW_DIR / "mlb_evidence_ledger.csv"


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


def collect_close_snapshot_once() -> Dict[str, Any]:
    """One-shot close snapshot collection."""
    from evidence_lifecycle import collect_close_snapshot, get_lifecycle_counts

    now = datetime.now(timezone.utc)
    result: Dict[str, Any] = {
        "sport": "MLB",
        "collected_at": now.isoformat(),
        "success": False,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "failure_reason": "",
        "terminal_state": "MLB_WAITING_FOR_FRESH_PROPS",
    }

    # Check if entry rows exist
    counts = get_lifecycle_counts()
    if counts["entry"] == 0 and counts["close"] == 0 and counts["settled_gold"] == 0:
        result["failure_reason"] = "no_entry_rows_exist"
        result["terminal_state"] = "MLB_WAITING_FOR_FRESH_PROPS"
        return result

    if counts["entry"] == 0:
        result["failure_reason"] = "no_rows_awaiting_close"
        result["terminal_state"] = "MLB_WAITING_FOR_SETTLEMENT" if counts["close"] > 0 else "MLB_WAITING_FOR_FRESH_PROPS"
        return result

    # Get fresh close odds
    from provider_router import MlbProviderRouter
    router = MlbProviderRouter()
    df_close, info = router.get_fresh_odds()

    if df_close is None or df_close.empty:
        result["failure_reason"] = "no_close_odds_available"
        result["terminal_state"] = "MLB_WAITING_FOR_CLOSE_LINES"
        return result

    provider_name = info.get("successful_provider", "unknown")

    # Match close rows to entry rows
    matched, unmatched = collect_close_snapshot(df_close, provider_name)

    result["matched_rows"] = matched
    result["unmatched_rows"] = unmatched

    if matched > 0:
        result["success"] = True
        result["failure_reason"] = ""
        result["terminal_state"] = "MLB_WAITING_FOR_SETTLEMENT"
    else:
        result["failure_reason"] = "no_matching_close_rows"
        result["terminal_state"] = "MLB_WAITING_FOR_CLOSE_LINES"

    # Update production status
    try:
        from production_status_reporter import build_production_status
        build_production_status()
    except Exception:
        pass

    return result


def main():
    result = collect_close_snapshot_once()
    print(json.dumps(result, indent=2, default=_json_default))
    if not result["success"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
