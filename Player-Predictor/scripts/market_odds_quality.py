"""Shared market odds quality checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def is_valid_american_odds(value: object) -> bool:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(odds) and odds != 0 and (odds <= -100.0 or odds >= 100.0))


def add_american_odds_quality(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_valid_over_odds"] = out["over_odds"].map(is_valid_american_odds) if "over_odds" in out.columns else False
    out["is_valid_under_odds"] = out["under_odds"].map(is_valid_american_odds) if "under_odds" in out.columns else False
    out["is_two_sided"] = out.get("over_odds", pd.Series(index=out.index)).notna() & out.get("under_odds", pd.Series(index=out.index)).notna()
    out["is_valid_american_odds"] = out["is_two_sided"] & out["is_valid_over_odds"] & out["is_valid_under_odds"]
    return out


def odds_quality_report(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "rows": 0,
            "rows_with_valid_american_odds": 0,
            "rows_with_invalid_american_odds": 0,
            "valid_american_odds_rate": 0.0,
        }
    checked = add_american_odds_quality(frame)
    valid = checked["is_valid_american_odds"]
    report = {
        "rows": int(len(checked)),
        "rows_with_valid_american_odds": int(valid.sum()),
        "rows_with_invalid_american_odds": int((~valid).sum()),
        "valid_american_odds_rate": float(valid.mean()),
        "rows_repaired": 0,
        "rows_dropped": int((~valid).sum()),
    }
    for col in ["over_odds", "under_odds"]:
        if col in checked.columns:
            values = pd.to_numeric(checked[col], errors="coerce")
            report[f"min_{col}"] = float(values.min()) if values.notna().any() else None
            report[f"max_{col}"] = float(values.max()) if values.notna().any() else None
    return report
