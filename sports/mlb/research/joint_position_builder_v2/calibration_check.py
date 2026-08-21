from __future__ import annotations

"""Price-independent joint-probability calibration check.

Real per-leg H-target market prices are entirely absent from
DEVELOPMENT_STAMPS (see REPORT.md) -- every marginal_ev/pair_class/joint_ev
in the real backtest is therefore None/"?" for lack of a real price, which
is correct behavior (mechanism-only), not a bug. This module runs the ONE
part of the mechanism that doesn't need a price at all: does the joint
probability model (independence over the frozen marginal model) correctly
predict the realized both-legs-win rate? It reuses build_world_distribution
exactly as pairs.py does, just without any EV/price gating.

Capped to the top-K rows per day by marginal_probability (not all
observation-universe rows) purely for combinatorial tractability --
enumerating every pair among ~100-200 daily rows is unnecessary for a
calibration check and would be slow; this is a diagnostic, not the action
pathway (pairs.py/ablation.py are unaffected by this cap).
"""

import numpy as np
import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import build_world_distribution, world_id_from_outcomes

TOP_K_PER_DAY = 15


def joint_probability_calibration(observation_universe: pd.DataFrame, *, top_k: int = TOP_K_PER_DAY) -> pd.DataFrame:
    from itertools import combinations

    records = []
    for date, day in observation_universe.groupby("date"):
        day = day[day["in_support"]].sort_values("marginal_probability", ascending=False).head(top_k).reset_index(drop=True)
        if len(day) < 2:
            continue
        for idx_i, idx_j in combinations(range(len(day)), 2):
            row_i, row_j = day.iloc[idx_i], day.iloc[idx_j]
            p_i, p_j = float(row_i["marginal_probability"]), float(row_j["marginal_probability"])
            clipped = np.clip([p_i, p_j], 1e-4, 1 - 1e-4)
            distribution = build_world_distribution(["i", "j"], clipped)
            p_joint = float(distribution.probabilities[world_id_from_outcomes([1, 1])])
            both_win = int(row_i["win"]) == 1 and int(row_j["win"]) == 1
            records.append({"date": date, "p_joint": p_joint, "both_win": int(both_win), "same_game": row_i["game_id"] == row_j["game_id"]})
    return pd.DataFrame(records)


def calibration_by_decile(calibration_rows: pd.DataFrame) -> pd.DataFrame:
    if calibration_rows.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_predicted_p_joint", "actual_both_win_rate"])
    rows = calibration_rows.copy()
    rows["bucket"] = pd.qcut(rows["p_joint"], q=10, duplicates="drop")
    grouped = rows.groupby("bucket", observed=True).agg(
        n=("both_win", "size"), mean_predicted_p_joint=("p_joint", "mean"), actual_both_win_rate=("both_win", "mean")
    )
    return grouped.reset_index()
