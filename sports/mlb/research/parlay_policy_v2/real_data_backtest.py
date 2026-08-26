"""Backtest parlay_policy_v2's gate mechanism against real, settled MLB legs.

This is the native MLB copy of the analysis that motivated porting the gate
here in the first place (originally run from the NBA side of the repo,
against this exact same MLB dataset, because NBA has no settled leg-level
data of its own -- see the NBA module's REPORT.md). Living here means MLB
owns its own real-data validation of this mechanism directly.

Data: sports/mlb/data/predictions/backtests/mlb_walk_forward_backtest_rows.csv,
policy source `published_real_market` (337 real, market_source="real" legs
across 11 dates, each with a model probability, a real settled result, and a
real American side price).

Two separate questions, both answered with real numbers:

  1. What does sports/parlay_analysis.py (CONTROL, unmodified, unimported by
     anything else in this package) actually score on these real, settled
     legs?
  2. What would parlay_policy_v2's gate (probability floor + real-quote EV,
     computed on the same real candidate pool CONTROL considers) have
     selected, and how did those real, already-settled parlays actually do?

CAVEAT, same as the NBA version: this dataset does not log per-candidate
joint_sigma, shared_failure_risk, compatible_state_score, shift_risk, or
lineup/role/injury/support state, so those gates are left at pass-through
defaults below and are NOT exercised here -- only the probability-floor and
actual-quote-EV-from-real-decimal-price mechanism is. Cross-game legs use
the product of each leg's own real decimal price as the actual quote --
CONTROL's own documented convention for a straight, non-same-game parlay
(see joint_position_builder_v2/REPORT.md), not a synthetic substitute for a
same-game SGP quote.

Run: python3 sports/mlb/research/parlay_policy_v2/real_data_backtest.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from sports.parlay_analysis import evaluate_historical_parlays, score_candidate_parlays
from sports.mlb.research.parlay_policy_v2.policy import (
    ParlayPolicy,
    american_to_decimal,
    brier_score,
    evaluate_candidate,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_CSV = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "backtests" / "mlb_walk_forward_backtest_rows.csv"
POLICY_SOURCE = "published_real_market"  # the only policy slice in this file that is 100% market_source="real"


def build_real_candidate_pairs(sub: pd.DataFrame) -> pd.DataFrame:
    """Every real 2-leg pair CONTROL's own gates (same-book, no forbidden
    same-player/same-game/same-market-bucket combos) would consider, graded
    against the real settled result of both legs."""
    hist = sub.rename(columns={"date": "market_date"}).copy()
    hist["selected_sportsbook_key"] = "real_aggregate"  # this file logs market_source real/synthetic, not a
    # per-book identity; treated as one shared real aggregate book so CONTROL's require_same_sportsbook gate
    # (which needs a matching key on both legs) can evaluate at all -- documented assumption, not fabricated data.

    rows_out = []
    for market_date, part in hist.groupby("market_date"):
        rows = part.to_dict(orient="records")
        for i, r in enumerate(rows):
            r["play_key"] = f"{market_date}-{i}"
        parlays = score_candidate_parlays(
            rows, sport="mlb", probability_field="probability", min_legs_per_parlay=2, max_legs_per_parlay=2
        )
        for p in parlays:
            idx = [int(x) for x in p["leg_indices"]]
            legs = [rows[i] for i in idx]
            results = [str(leg.get("result", "unresolved")) for leg in legs]
            if any(r == "push" for r in results):
                continue
            if not all(r in ("win", "loss") for r in results):
                continue
            won = 1 if all(r == "win" for r in results) else 0
            p1, p2 = float(legs[0]["probability"]), float(legs[1]["probability"])
            d1 = american_to_decimal(float(legs[0]["side_price"]))
            d2 = american_to_decimal(float(legs[1]["side_price"]))
            rows_out.append(
                {
                    "market_date": market_date,
                    "leg_a": f"{legs[0]['player']}|{legs[0]['target']}|{legs[0]['direction']}",
                    "leg_b": f"{legs[1]['player']}|{legs[1]['target']}|{legs[1]['direction']}",
                    "p1": p1,
                    "p2": p2,
                    "naive_joint": p1 * p2,
                    "actual_quote_decimal": d1 * d2,
                    "won": won,
                }
            )
    return pd.DataFrame(rows_out)


def apply_new_policy(pairs_df: pd.DataFrame, policy: ParlayPolicy) -> pd.DataFrame:
    selected = []
    for _, row in pairs_df.iterrows():
        candidate = {
            "leg_count": 2,
            "min_leg_probability": min(row["p1"], row["p2"]),
            "min_leg_sigma": 0.0,  # not logged in this real data -- LEG_PROBABILITY gate exercised, uncertainty penalty is not
            "joint_probability": row["naive_joint"],
            "joint_sigma": 0.0,  # not logged -- JOINT_UNCERTAINTY gate not exercised
            "joint_lcb": row["naive_joint"],  # no per-candidate variance logged -- no extra shrinkage applied here
            "dependency_penalty": 0.0,  # cross-game legs; independence is CONTROL's own modeling assumption too
            "actual_quote_decimal": row["actual_quote_decimal"],
            "shared_failure_risk": 0.0,  # not logged -- SHARED_FAILURE gate not exercised
            "compatible_state_score": 1.0,  # not logged -- STATE_COMPATIBILITY gate not exercised
            "shift_risk": 0.0,  # not logged -- SHIFT_RISK gate not exercised
            "lineup_confirmed": True,
            "role_stable": True,
            "material_injury_uncertainty": False,
            "all_legs_in_support": True,
            "joint_model_reliable": True,
        }
        result = evaluate_candidate(candidate, policy)
        selected.append({**row.to_dict(), **result})
    return pd.DataFrame(selected)


def main() -> dict:
    df = pd.read_csv(DATA_CSV)
    sub = df[df["policy"] == POLICY_SOURCE].copy()
    assert (sub["market_source"] == "real").all(), "expected every published_real_market row to be market_source=real"

    hist = sub.rename(columns={"date": "market_date"})
    hist["selected_sportsbook_key"] = "real_aggregate"  # see build_real_candidate_pairs docstring note
    control_summary = evaluate_historical_parlays(
        hist, sport="mlb", date_col="market_date", probability_col="probability",
        result_col="result", max_pairs_per_day=1, min_legs_per_parlay=2, max_legs_per_parlay=2,
    )

    pairs_df = build_real_candidate_pairs(sub)
    base_wins, base_n = int(pairs_df["won"].sum()), len(pairs_df)
    base_lo, base_hi = wilson_interval(base_wins, base_n)

    policy = ParlayPolicy(
        min_joint_probability=0.55, min_joint_lcb=0.50, min_leg_probability=0.68,
        uncertainty_lambda=0.0, max_shared_failure_risk=1.0, min_compatible_state_score=0.0,
        max_shift_risk=1.0, max_joint_uncertainty=1.0, min_actual_quote_ev=0.0,
    )
    sel_df = apply_new_policy(pairs_df, policy)
    elig = sel_df[sel_df["eligible"]]
    sel_wins, sel_n = int(elig["won"].sum()), len(elig)
    sel_lo, sel_hi = wilson_interval(sel_wins, sel_n)

    per_day_best = elig.sort_values(["actual_quote_ev", "market_date"], ascending=[False, True]).groupby("market_date").first()
    day_wins, day_n = int(per_day_best["won"].sum()), len(per_day_best)
    day_lo, day_hi = wilson_interval(day_wins, day_n)

    reasons = Counter()
    for r in sel_df["reasons"]:
        for x in r:
            reasons[x] += 1

    report = {
        "data_source": str(DATA_CSV.relative_to(REPO_ROOT)),
        "policy_source_filter": POLICY_SOURCE,
        "real_settled_legs": int(len(sub)),
        "real_dates": int(sub["date"].nunique()),
        "current_mlb_strategy_control": control_summary,
        "full_real_eligible_pair_pool": {
            "n": base_n, "wins": base_wins, "hit_rate": base_wins / base_n if base_n else None,
            "wilson95": [base_lo, base_hi],
        },
        "new_policy_v2_gate": {
            "eligible": sel_n, "of_pool": len(sel_df), "coverage": sel_n / len(sel_df) if len(sel_df) else None,
            "wins": sel_wins, "hit_rate": sel_wins / sel_n if sel_n else None, "wilson95": [sel_lo, sel_hi],
            "mean_actual_quote_ev": float(elig["actual_quote_ev"].mean()) if sel_n else None,
            "brier_full_pool": brier_score(pairs_df["won"], pairs_df["naive_joint"]),
            "brier_selected": brier_score(elig["won"], elig["naive_joint"]) if sel_n else None,
            "mean_predicted_p_selected": float(elig["naive_joint"].mean()) if sel_n else None,
            "rejection_reason_counts": dict(reasons),
        },
        "new_policy_v2_one_pick_per_day": {
            "n_days": day_n, "wins": day_wins, "hit_rate": day_wins / day_n if day_n else None,
            "wilson95": [day_lo, day_hi],
        },
        "gates_not_exercised_not_logged_in_this_data": [
            "JOINT_UNCERTAINTY (joint_sigma)", "SHARED_FAILURE (shared_failure_risk)",
            "STATE_COMPATIBILITY (compatible_state_score)", "SHIFT_RISK (shift_risk)",
            "LINEUP/ROLE/INJURY_UNCERTAINTY/OUT_OF_SUPPORT/JOINT_MODEL_UNRELIABLE (execution state)",
        ],
    }
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
