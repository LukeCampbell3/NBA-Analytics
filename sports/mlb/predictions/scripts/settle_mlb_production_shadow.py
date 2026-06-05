#!/usr/bin/env python3
"""
MLB Settlement Script

Settles MLB production-shadow rows using MLB StatsAPI or local Data-Proc-MLB.

Settlement rules:
  OVER hits if actual_value > line
  UNDER hits if actual_value < line
  push if actual_value == line

Appends outcome fields only. Never modifies prediction-time fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
MLB_SHADOW_DIR = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"
LEDGER_PATH = MLB_SHADOW_DIR / "mlb_live_ledger.csv"
DATA_PROC_MLB_DIR = WORKSPACE / "Player-Predictor" / "Data-Proc-MLB"

# Market → column in processed data
MARKET_STAT_MAP = {
    "K": "SO",
    "H": "H",
    "TB": "TB",
    "RBI": "RBI",
    "R": "R",
    "HR": "HR",
    "SO": "SO",
    "ER": "ER",
    "OUTS": "IP",
    "HA": "H",
    "BB": "BB",
}


def settle_pending_rows() -> Dict[str, Any]:
    """Settle all pending MLB rows."""
    if not LEDGER_PATH.exists():
        return {"settled": 0, "status": "no_ledger"}

    df = pd.read_csv(LEDGER_PATH)
    if df.empty:
        return {"settled": 0, "status": "empty_ledger"}

    # Pending settlement: has close but not settled
    pending_mask = (
        (df["close_snapshot_id"].notna()) & (df["close_snapshot_id"] != "") &
        ((df["settled_at"].isna()) | (df["settled_at"] == ""))
    )
    pending = df[pending_mask]

    if pending.empty:
        return {"settled": 0, "status": "no_pending"}

    settled_count = 0
    for idx in pending.index:
        row = df.loc[idx]
        actual = _resolve_actual_value(row)
        if actual is None:
            continue

        line = float(row["line"])
        side = str(row["side"]).lower()
        p_model = float(row.get("p_model_raw", 0.5))
        odds = float(row.get("odds", -110))

        # Settlement
        if actual > line:
            result = "HIT" if side == "over" else "LOSS"
        elif actual < line:
            result = "HIT" if side == "under" else "LOSS"
        else:
            result = "PUSH"

        # Unit profit
        if result == "HIT":
            profit = (odds / 100.0) if odds > 0 else (100.0 / abs(odds))
        elif result == "LOSS":
            profit = -1.0
        else:
            profit = 0.0

        # Brier
        outcome_binary = 1.0 if result == "HIT" else 0.0
        brier = (p_model - outcome_binary) ** 2
        market_prob = float(row.get("market_no_vig", 0.5))
        market_brier = (market_prob - outcome_binary) ** 2

        df.at[idx, "actual_value"] = actual
        df.at[idx, "hit_loss_push"] = result
        df.at[idx, "unit_profit"] = profit
        df.at[idx, "brier"] = brier
        df.at[idx, "market_brier"] = market_brier
        df.at[idx, "bss"] = 1.0 - (brier / market_brier) if market_brier > 0 else 0.0
        df.at[idx, "settled_at"] = datetime.now(timezone.utc).isoformat()
        settled_count += 1

    df.to_csv(LEDGER_PATH, index=False)

    report = {
        "settled": settled_count,
        "pending_remaining": int(pending_mask.sum()) - settled_count,
        "status": "success" if settled_count > 0 else "no_outcomes_available",
        "settled_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write settlement report
    report_path = MLB_SHADOW_DIR / f"settlement_report_{datetime.now().strftime('%Y%m%d')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _resolve_actual_value(row: pd.Series) -> Optional[float]:
    """Resolve actual stat value from local data."""
    player = str(row.get("player", "")).replace(" ", "_")
    market = str(row.get("market_canonical", ""))
    game_date = str(row.get("commence_time_utc", ""))[:10]

    stat_col = MARKET_STAT_MAP.get(market)
    if not stat_col:
        return None

    player_dir = DATA_PROC_MLB_DIR / player
    if not player_dir.exists():
        return None

    files = sorted(player_dir.glob("*processed*.csv"))
    if not files:
        return None

    for f in reversed(files):
        try:
            df = pd.read_csv(f)
            if "Date" not in df.columns or stat_col not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            match = df[df["Date"] == game_date]
            if not match.empty:
                val = pd.to_numeric(match.iloc[0][stat_col], errors="coerce")
                if pd.notna(val):
                    return float(val)
        except Exception:
            continue

    return None


def main():
    parser = argparse.ArgumentParser(description="MLB Settlement")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] Would settle pending MLB rows")
        return

    result = settle_pending_rows()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
