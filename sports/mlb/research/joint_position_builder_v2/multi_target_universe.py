from __future__ import annotations

"""Generalizes observation_universe.py beyond H to every target with real
DEVELOPMENT-window price coverage: R (57%), TB (58%), HR (53%). H, RBI, and
ERA are excluded here -- H and RBI have 0% real price coverage in
DEVELOPMENT_STAMPS, ERA likewise 0%; K and ER have only ~10% and are kept
out for now (see STATE.md). This directly tests the mission's "do not
assume H-OVER only" instruction rather than assuming it.

Reuses, unmodified: h_over_ranker.baselines.probability_score (the frozen
marginal model), the DERIVE/SELECT/TEST partition, and every downstream
primitive in pairs.py/risk_gate.py/calibration_check.py (all already
target-agnostic -- they operate on generic probability/price/support
columns, not on H specifically).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from sports.mlb.research.h_over_ranker.data_windows import DAILY_RUNS_ROOT, DERIVE_STAMPS, REPO_ROOT
from sports.mlb.research.h_over_ranker.baselines import probability_score
from sports.mlb.research.h_over_ranker.eligibility import _cached_actual_lookup, _to_float

MIN_HISTORY_ROWS_FOR_SUPPORT = 20
MAX_SANE_RMSE = 5.0

# Targets with real price coverage worth carrying (see STATE.md for the
# coverage table this was decided from). H is deliberately excluded here
# per the mission's own instruction to test rather than assume H-OVER-only.
PRICED_TARGETS = ("R", "TB", "HR")

# Half-integer lines (no push mass) confirmed per-target before relying on
# the "UNDER = 1 - OVER" complement shortcut; see STATE.md provenance.
HALF_INTEGER_TARGETS = {"R", "TB", "HR", "H"}


def _load_vhfp():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
    import validate_historical_final_pools as vhfp

    return vhfp


@lru_cache(maxsize=8)
def frozen_bias(target: str) -> float:
    """mean(Prediction - Actual) for `target` rows, DERIVE_STAMPS only.
    Cached so repeated calls (tests, multiple ablation runs) don't
    re-scan DERIVE every time."""
    vhfp = _load_vhfp()
    actual_lookup = _cached_actual_lookup(str(REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"))
    diffs = []
    for stamp in DERIVE_STAMPS:
        path = DAILY_RUNS_ROOT / stamp / f"daily_prediction_pool_{stamp}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        for _, row in frame.iterrows():
            if str(row.get("Target", "")).strip().upper() != target:
                continue
            prediction = _to_float(row.get("Prediction"))
            game_date = str(row.get("Game_Date", ""))[:10]
            player_key = vhfp.normalize_player_key(row.get("Player_ID") or row.get("Player"))
            game_id = str(row.get("Game_ID", "") or "")
            if prediction is None or not game_date or not player_key or not game_id:
                continue
            actual = actual_lookup.get((game_date, player_key, target, game_id))
            if actual is None:
                continue
            diffs.append(prediction - float(actual))
    if not diffs:
        raise RuntimeError(f"no DERIVE rows found to compute the {target} bias from")
    return float(np.mean(diffs))


def _decimal_price(american: float | None) -> float | None:
    if american is None or not np.isfinite(american) or abs(american) < 100.0:
        return None
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def build_multi_target_universe(
    stamps: tuple[str, ...], *, targets: tuple[str, ...] = PRICED_TARGETS, mode: str = "broad"
) -> pd.DataFrame:
    """mode: "narrow" (edge>0 only, i.e. today's OVER-only-style admission,
    generalized across targets) or "broad" (both directions, no edge-sign
    filter -- the mission's default position to test)."""
    if mode not in ("narrow", "broad"):
        raise ValueError("mode must be 'narrow' or 'broad'")
    vhfp = _load_vhfp()
    actual_lookup = _cached_actual_lookup(str(REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"))

    rows: list[dict] = []
    for target in targets:
        bias = frozen_bias(target)
        for stamp in stamps:
            path = DAILY_RUNS_ROOT / stamp / f"daily_prediction_pool_{stamp}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False)
            for _, row in frame.iterrows():
                if str(row.get("Target", "")).strip().upper() != target:
                    continue
                prediction = _to_float(row.get("Prediction"))
                market_line = _to_float(row.get("Market_Line"))
                if prediction is None or market_line is None:
                    continue
                corrected_prediction = prediction - bias
                corrected_edge = corrected_prediction - market_line
                if mode == "narrow" and corrected_edge <= 0:
                    continue
                direction = "OVER" if corrected_edge > 0 else ("UNDER" if corrected_edge < 0 else None)
                if direction is None:
                    continue

                game_date = str(row.get("Game_Date", ""))[:10]
                player_key = vhfp.normalize_player_key(row.get("Player_ID") or row.get("Player"))
                game_id = str(row.get("Game_ID", "") or "")
                if not game_date or not player_key or not game_id:
                    continue
                actual = actual_lookup.get((game_date, player_key, target, game_id))
                if actual is None:
                    continue
                result = vhfp.grade_result(float(actual), market_line, direction)
                if result not in ("win", "loss"):
                    continue  # pushes excluded

                rmse = _to_float(row.get("Model_Val_RMSE")) or 0.3
                history_rows = _to_float(row.get("History_Rows")) or 0.0
                over_prob = probability_score(corrected_prediction, market_line, rmse)
                marginal_probability = over_prob if direction == "OVER" else (1.0 - over_prob)

                american_price = row.get("Market_Over_Price") if direction == "OVER" else row.get("Market_Under_Price")
                decimal_price = _decimal_price(_to_float(american_price))
                marginal_ev = (marginal_probability * decimal_price - 1.0) if decimal_price is not None else None

                in_support = bool(
                    history_rows >= MIN_HISTORY_ROWS_FOR_SUPPORT and np.isfinite(rmse) and 0.0 < rmse < MAX_SANE_RMSE
                )

                rows.append(
                    {
                        "date": stamp,
                        "player": str(row.get("Player", "")),
                        "player_key": player_key,
                        "game_id": game_id,
                        "team": str(row.get("Team", "") or ""),
                        "target": target,
                        "direction": direction,
                        "market_line": market_line,
                        "corrected_prediction": corrected_prediction,
                        "corrected_edge": corrected_edge,
                        "marginal_probability": marginal_probability,
                        "decimal_price": decimal_price,
                        "marginal_ev": marginal_ev,
                        "rmse": rmse,
                        "history_rows": history_rows,
                        "market_source": str(row.get("Market_Source", "") or "").strip().lower(),
                        "market_line_std": _to_float(row.get("Market_Line_Std")) or 0.0,
                        "in_support": in_support,
                        "win": 1 if result == "win" else 0,
                    }
                )
    # market_line is part of the dedup identity -- two rows differing only
    # by line are different EVENTS (e.g. H OVER 0.5 vs H OVER 1.5), not
    # duplicates. Omitting it here would silently drop a real alternate
    # line. Not observed in this repo's current archived pools (checked:
    # zero player/target/game/direction groups carry >1 distinct
    # Market_Line across all 25 archived days), but the schema must not
    # rely on that absence -- see parlay_v2/candidate_adapter.py's
    # exact-event-identity guard and its regression test.
    return pd.DataFrame(rows).drop_duplicates(subset=["date", "player_key", "game_id", "target", "direction", "market_line"]).reset_index(drop=True)


def action_universe(universe: pd.DataFrame) -> pd.DataFrame:
    return universe[universe["in_support"] & universe["decimal_price"].notna()].reset_index(drop=True)
