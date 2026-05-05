"""Closing Line Value (CLV) tracker.

CLV is the single most reliable indicator of long-term betting edge.
It measures whether you consistently get a better price than the market's
final assessment.  A bettor with positive CLV is profitable long-term
even through losing streaks.

This module:
  1. Records the line at time of pick (opening line)
  2. Compares to the closing line (if available)
  3. Computes CLV in points and implied probability
  4. Tracks rolling CLV to detect edge persistence or decay

For player props at -110/-110:
  - If you bet UNDER 24.5 and the line closes at 23.5, you have +1 point CLV
  - If you bet UNDER 24.5 and the line closes at 25.5, you have -1 point CLV
  - Consistent positive CLV = real edge over the market
  - Consistent negative CLV = the market is smarter than you

Usage:
  After each day's games, call `record_clv_outcome()` with the closing
  lines to update the rolling CLV tracker.  The `get_clv_summary()` function
  returns the current edge health metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CLVRecord:
    """A single CLV observation."""
    date: str
    player: str
    target: str
    direction: str
    opening_line: float
    closing_line: float | None = None
    clv_points: float = 0.0
    result: str = ""  # win/loss/push/pending


@dataclass
class CLVSummary:
    """Rolling CLV health metrics."""
    total_picks: int = 0
    picks_with_clv: int = 0
    avg_clv_points: float = 0.0
    positive_clv_rate: float = 0.0
    rolling_7d_clv: float = 0.0
    rolling_14d_clv: float = 0.0
    edge_status: str = "unknown"  # healthy/warning/decaying/no_data
    records: list[dict] = field(default_factory=list)


CLV_STORE_PATH = Path(__file__).resolve().parents[1] / "model" / "analysis" / "clv_tracker.json"


def compute_clv_points(
    direction: str,
    opening_line: float,
    closing_line: float,
) -> float:
    """Compute CLV in points (positive = you got a better line).

    For UNDER bets: CLV = opening_line - closing_line
      (if line dropped, you got a higher number = better for UNDER)
    For OVER bets: CLV = closing_line - opening_line
      (if line rose, you got a lower number = better for OVER)
    """
    dir_upper = str(direction).upper().strip()
    if dir_upper == "UNDER":
        return opening_line - closing_line
    elif dir_upper == "OVER":
        return closing_line - opening_line
    return 0.0


def record_pick(
    *,
    date_str: str,
    player: str,
    target: str,
    direction: str,
    opening_line: float,
    store_path: Path | None = None,
) -> None:
    """Record a pick's opening line for later CLV comparison."""
    path = store_path or CLV_STORE_PATH
    records = _load_records(path)

    records.append({
        "date": date_str,
        "player": player,
        "target": target,
        "direction": direction,
        "opening_line": opening_line,
        "closing_line": None,
        "clv_points": 0.0,
        "result": "pending",
        "recorded_at": datetime.utcnow().isoformat(),
    })

    _save_records(records, path)


def update_closing_lines(
    closing_data: list[dict],
    *,
    store_path: Path | None = None,
) -> int:
    """Update stored records with closing line data.

    closing_data: list of dicts with keys: date, player, target, closing_line, result
    Returns number of records updated.
    """
    path = store_path or CLV_STORE_PATH
    records = _load_records(path)

    # Build lookup
    closing_lookup = {}
    for item in closing_data:
        key = (str(item["date"]), str(item["player"]), str(item["target"]))
        closing_lookup[key] = item

    updated = 0
    for record in records:
        key = (record["date"], record["player"], record["target"])
        if key in closing_lookup:
            closing = closing_lookup[key]
            cl = closing.get("closing_line")
            if cl is not None:
                record["closing_line"] = float(cl)
                record["clv_points"] = compute_clv_points(
                    record["direction"],
                    record["opening_line"],
                    float(cl),
                )
            if "result" in closing:
                record["result"] = str(closing["result"])
            updated += 1

    _save_records(records, path)
    return updated


def get_clv_summary(
    *,
    store_path: Path | None = None,
    lookback_days: int = 30,
) -> CLVSummary:
    """Get rolling CLV health metrics."""
    path = store_path or CLV_STORE_PATH
    records = _load_records(path)

    if not records:
        return CLVSummary(edge_status="no_data")

    df = pd.DataFrame(records)
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter to records with CLV data
    has_clv = df[df["closing_line"].notna() & (df["closing_line"] != 0)]

    summary = CLVSummary(
        total_picks=len(df),
        picks_with_clv=len(has_clv),
    )

    if has_clv.empty:
        summary.edge_status = "no_data"
        return summary

    clv_values = pd.to_numeric(has_clv["clv_points"], errors="coerce").fillna(0.0)
    summary.avg_clv_points = float(clv_values.mean())
    summary.positive_clv_rate = float((clv_values > 0).mean())

    # Rolling windows
    now = pd.Timestamp.now()
    last_7d = has_clv[has_clv["_date"] >= (now - pd.Timedelta(days=7))]
    last_14d = has_clv[has_clv["_date"] >= (now - pd.Timedelta(days=14))]

    if not last_7d.empty:
        summary.rolling_7d_clv = float(pd.to_numeric(last_7d["clv_points"], errors="coerce").mean())
    if not last_14d.empty:
        summary.rolling_14d_clv = float(pd.to_numeric(last_14d["clv_points"], errors="coerce").mean())

    # Edge status assessment
    if summary.avg_clv_points > 0.3 and summary.positive_clv_rate > 0.55:
        summary.edge_status = "healthy"
    elif summary.avg_clv_points > 0 and summary.positive_clv_rate > 0.50:
        summary.edge_status = "marginal"
    elif summary.rolling_7d_clv < -0.5:
        summary.edge_status = "decaying"
    else:
        summary.edge_status = "warning"

    summary.records = records[-20:]  # last 20 for display
    return summary


def format_clv_summary(summary: CLVSummary) -> str:
    """Format CLV summary for display."""
    lines = [
        "CLV TRACKER",
        "=" * 40,
        f"  Total picks tracked: {summary.total_picks}",
        f"  Picks with CLV data: {summary.picks_with_clv}",
        f"  Avg CLV: {summary.avg_clv_points:+.2f} points",
        f"  Positive CLV rate: {summary.positive_clv_rate:.1%}",
        f"  7-day rolling CLV: {summary.rolling_7d_clv:+.2f}",
        f"  14-day rolling CLV: {summary.rolling_14d_clv:+.2f}",
        f"  Edge status: {summary.edge_status.upper()}",
    ]
    return "\n".join(lines)


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep last 500 records to prevent unbounded growth
    trimmed = records[-500:]
    path.write_text(json.dumps(trimmed, indent=2, default=str), encoding="utf-8")
