from __future__ import annotations

"""OBSERVATION UNIVERSE and ACTION UNIVERSE construction.

OBSERVATION UNIVERSE: every H-target row with adequate support, regardless
of direction or individual EV -- may inform joint state even if never
wagered. No EV_i>0 requirement (that gate is removed here on purpose --
see manifest.py for why the removal is NOT unconditional).

ACTION UNIVERSE: the subset of the observation universe that is
wagerable/model-supported: in_support AND a real market decimal price
exists for that leg's own side. Still no EV_i>0 requirement.

"narrow" state == today's CONTROL eligibility exactly (H, OVER only,
corrected_edge>0) -- reuses h_over_ranker.eligibility unchanged.
"broad" state == every gradable H row, both directions, no edge-sign
filter -- this is "formerly filtered markets" for the ablation in
ablation.py. Uses the SAME frozen bias correction and the SAME frozen
marginal probability mechanism (h_over_ranker.baselines.probability_score)
in both modes; only which rows are admitted changes.

Marginal model reuse (required by the task): no new marginal model is fit
here. FROZEN_H_BIAS and probability_score come unmodified from the
already-frozen h_over_ranker package.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports.mlb.research.h_over_ranker.data_windows import DAILY_RUNS_ROOT, REPO_ROOT
from sports.mlb.research.h_over_ranker.eligibility import FROZEN_H_BIAS, TARGET, _cached_actual_lookup, _to_float
from sports.mlb.research.h_over_ranker.baselines import probability_score

MIN_HISTORY_ROWS_FOR_SUPPORT = 20  # matches this repo's existing convention
                                    # (FROZEN_SELECTOR_PROTOCOL.lookback_games,
                                    # BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates)
MAX_SANE_RMSE = 5.0  # generous upper bound; observed real RMSE is ~0.3-2.7


@dataclass(frozen=True)
class UniverseRow:
    date: str
    player: str
    player_key: str
    game_id: str
    team: str
    target: str
    direction: str
    market_line: float
    corrected_prediction: float
    corrected_edge: float
    marginal_probability: float
    decimal_price: float | None
    marginal_ev: float | None
    rmse: float
    history_rows: float
    market_source: str
    in_support: bool
    win: int


def _decimal_price(american: float | None) -> float | None:
    if american is None or not np.isfinite(american) or abs(american) < 100.0:
        return None
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def _load_vhfp():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
    import validate_historical_final_pools as vhfp

    return vhfp


def build_observation_universe(stamps: tuple[str, ...], *, mode: str) -> pd.DataFrame:
    """mode: "narrow" (H-OVER only, matches CONTROL eligibility) or "broad"
    (H, both directions, no edge-sign filter)."""
    if mode not in ("narrow", "broad"):
        raise ValueError("mode must be 'narrow' or 'broad'")
    vhfp = _load_vhfp()
    actual_lookup = _cached_actual_lookup(str(REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"))

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

            if mode == "narrow" and corrected_edge <= 0:
                continue  # CONTROL's exact eligibility gate

            direction = "OVER" if corrected_edge > 0 else ("UNDER" if corrected_edge < 0 else None)
            if direction is None:
                continue  # exactly on the line: no defined side

            game_date = str(row.get("Game_Date", ""))[:10]
            player_key = vhfp.normalize_player_key(row.get("Player_ID") or row.get("Player"))
            game_id = str(row.get("Game_ID", "") or "")
            if not game_date or not player_key or not game_id:
                continue
            actual = actual_lookup.get((game_date, player_key, TARGET, game_id))
            if actual is None:
                continue
            result = vhfp.grade_result(float(actual), market_line, direction)
            if result not in ("win", "loss"):
                continue  # pushes excluded; H lines are half-integers so this is rare/never

            rmse = _to_float(row.get("Model_Val_RMSE")) or 0.3
            history_rows = _to_float(row.get("History_Rows")) or 0.0
            over_prob = probability_score(corrected_prediction, market_line, rmse)
            marginal_probability = over_prob if direction == "OVER" else (1.0 - over_prob)

            american_price = row.get("Market_Over_Price") if direction == "OVER" else row.get("Market_Under_Price")
            decimal_price = _decimal_price(_to_float(american_price))
            marginal_ev = (marginal_probability * decimal_price - 1.0) if decimal_price is not None else None

            in_support = bool(
                history_rows >= MIN_HISTORY_ROWS_FOR_SUPPORT
                and np.isfinite(rmse)
                and 0.0 < rmse < MAX_SANE_RMSE
            )

            rows.append(
                {
                    "date": stamp,
                    "player": str(row.get("Player", "")),
                    "player_key": player_key,
                    "game_id": game_id,
                    "team": str(row.get("Team", "") or ""),
                    "target": TARGET,
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
                    "in_support": in_support,
                    "win": 1 if result == "win" else 0,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["date", "player_key", "game_id", "direction"]).reset_index(drop=True)


def action_universe(observation_universe: pd.DataFrame) -> pd.DataFrame:
    """Wagerable/model-supported subset: in_support AND a real leg price
    exists. Still no individual-EV requirement -- that is the point of V2."""
    return observation_universe[
        observation_universe["in_support"] & observation_universe["decimal_price"].notna()
    ].reset_index(drop=True)
