from __future__ import annotations

"""Frozen H-OVER eligibility rule.

Reconstructs exactly the rule used to pick H as the winning OVER target on
the SELECT block (see the session's three-way-split result: SELECT H-OVER
n=1432, hit=57.5%). The bias correction is learned from DERIVE_STAMPS only
and is itself frozen below as a literal constant, with a test
(`test_h_over_ranker.py::test_derive_bias_matches_frozen_constant`) that
recomputes it from raw data and asserts it still matches -- so silent drift
in the archived CSVs would fail loudly rather than quietly changing which
rows are "eligible".
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data_windows import DAILY_RUNS_ROOT, DERIVE_STAMPS, REPO_ROOT

TARGET = "H"


@lru_cache(maxsize=4)
def _cached_actual_lookup(processed_root_str: str) -> dict:
    """build_actual_lookup() scans thousands of per-player CSVs; cache it
    process-wide (tests and repeated eligible_rows_for_stamps() calls
    otherwise each pay that cost independently)."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
    import validate_historical_final_pools as vhfp

    return vhfp.build_actual_lookup(Path(processed_root_str))

# Frozen: mean(Prediction - Actual) for target=="H" rows across DERIVE_STAMPS
# only. See recompute_derive_bias() for the reproduction recipe; the
# consistency test asserts these stay equal.
FROZEN_H_BIAS = 0.0749739701851401


@dataclass(frozen=True)
class EligibleRow:
    date: str
    player: str
    game_id: str
    prediction: float
    corrected_prediction: float
    market_line: float
    corrected_edge: float
    raw_edge: float
    rmse: float
    mae: float
    history_rows: float
    market_books: float
    market_source: str
    market_line_std: float
    days_since_history: float | None
    win: int


def recompute_derive_bias(processed_root: Path | None = None) -> float:
    """Recompute the H-target bias from DERIVE_STAMPS raw data."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
    import validate_historical_final_pools as vhfp  # local import: script, not a package

    processed_root = processed_root or (REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB")
    actual_lookup = _cached_actual_lookup(str(processed_root))

    diffs = []
    for stamp in DERIVE_STAMPS:
        path = DAILY_RUNS_ROOT / stamp / f"daily_prediction_pool_{stamp}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        for _, row in frame.iterrows():
            if str(row.get("Target", "")).strip().upper() != TARGET:
                continue
            prediction = _to_float(row.get("Prediction"))
            game_date = str(row.get("Game_Date", ""))[:10]
            player_key = vhfp.normalize_player_key(row.get("Player_ID") or row.get("Player"))
            game_id = str(row.get("Game_ID", "") or "")
            if prediction is None or not game_date or not player_key or not game_id:
                continue
            actual = actual_lookup.get((game_date, player_key, TARGET, game_id))
            if actual is None:
                continue
            diffs.append(prediction - float(actual))
    if not diffs:
        raise RuntimeError("no DERIVE rows found to compute the H bias from")
    return float(np.mean(diffs))


def _to_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def eligible_rows_for_stamps(stamps: tuple[str, ...], processed_root: Path | None = None) -> pd.DataFrame:
    """Every H-OVER-eligible row (post frozen bias correction) across `stamps`.

    Pregame-only feature columns; the `win` column is the settled outcome,
    included for supervised fitting/evaluation, never as a ranking input.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
    import validate_historical_final_pools as vhfp

    processed_root = processed_root or (REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB")
    actual_lookup = _cached_actual_lookup(str(processed_root))

    rows: list[dict] = []
    for stamp in stamps:
        path = DAILY_RUNS_ROOT / stamp / f"daily_prediction_pool_{stamp}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        for _, row in frame.iterrows():
            if str(row.get("Target", "")).strip().upper() != TARGET:
                continue
            prediction = _to_float(row.get("Prediction"))
            market_line = _to_float(row.get("Market_Line"))
            if prediction is None or market_line is None:
                continue
            corrected_prediction = prediction - FROZEN_H_BIAS
            corrected_edge = corrected_prediction - market_line
            if corrected_edge <= 0:
                continue  # not H-OVER-eligible

            game_date = str(row.get("Game_Date", ""))[:10]
            player_key = vhfp.normalize_player_key(row.get("Player_ID") or row.get("Player"))
            game_id = str(row.get("Game_ID", "") or "")
            if not game_date or not player_key or not game_id:
                continue
            actual = actual_lookup.get((game_date, player_key, TARGET, game_id))
            if actual is None:
                continue
            result = vhfp.grade_result(float(actual), market_line, "OVER")
            if result not in ("win", "loss"):
                continue

            last_history_date = str(row.get("Last_History_Date", "") or "")
            days_since_history = None
            if last_history_date and game_date:
                try:
                    days_since_history = (
                        pd.Timestamp(game_date) - pd.Timestamp(last_history_date)
                    ).days
                except (ValueError, TypeError):
                    days_since_history = None

            rmse = _to_float(row.get("Model_Val_RMSE")) or 0.3
            rows.append(
                {
                    "date": stamp,
                    "player": str(row.get("Player", "")),
                    "player_key": player_key,
                    "game_id": game_id,
                    "prediction": prediction,
                    "corrected_prediction": corrected_prediction,
                    "market_line": market_line,
                    "corrected_edge": corrected_edge,
                    "raw_edge": prediction - market_line,
                    "rmse": rmse,
                    "mae": _to_float(row.get("Model_Val_MAE")) or rmse,
                    "history_rows": _to_float(row.get("History_Rows")) or 0.0,
                    "market_books": _to_float(row.get("Market_Books")) or 0.0,
                    "market_source": str(row.get("Market_Source", "") or "").strip().lower(),
                    "market_line_std": _to_float(row.get("Market_Line_Std")) or 0.0,
                    "days_since_history": days_since_history,
                    "win": 1 if result == "win" else 0,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["date", "player_key", "game_id"]).reset_index(drop=True)
