from __future__ import annotations

"""World-gate admission research (mission: "Resolve the remaining PARLAY_V2
APS / counterexample admission bottleneck with a falsification-driven
research and implementation pass").

PURE DEVELOPMENT-ONLY, OFFLINE RESEARCH. Never imported by production code
(run_parlay_v2.py, candidate_adapter.py, calibration/*, pair_ingest.py) --
this module exists only to determine, empirically and theoretically,
whether counterexample/APS information deserves to be a REQUIRED hard
gate, a BOUNDED-RISK gate, or an OBSERVE-ONLY diagnostic for a NEW
prospective policy version. It never touches PARLAY_POLICY_V2_PROSPECTIVE_002
(already frozen, immutable) and never reads TEST_STAMPS or any real
post-freeze prospective outcome.

Uses the SAME frozen DERIVE/SELECT chronological partition as
H_OVER_RANKER_V1 (h_over_ranker.data_windows) -- the same discipline this
repo already established for predictive-model development: DERIVE decides
mechanics/grid, SELECT is read exactly once to check them, TEST/prospective
stay untouched.

============================================================
ANALYTICAL RESULT (verified in the audit tests, not just asserted here)
============================================================
For a 2-leg pair with joint world distribution P over {LL, WL, LW, WW} (WW =
"both legs win", the only settlement-winning world), and APS retained set
C_t built at threshold T via outcome_worlds.build_binary_outcome_set:

    counterexample_mass(T) = sum of P(omega) for omega in C_t, omega != WW
    outside_mass(T)        = 1 - sum of P(omega) for omega in C_t
    rho(T) = counterexample_mass(T) + outside_mass(T)

Then, EXACTLY (not approximately):

    rho(T) = (1 - p_joint) + [P(WW) if WW is NOT in C_t else 0]

so rho(T) >= 1 - p_joint always, with equality whenever WW remains
retained. At T=1.0 (the current FROZEN_APS_THRESHOLD), C_t retains every
positive-probability world, so counterexample_mass(1.0) == 1 - p_joint
EXACTLY -- it is mathematically identical to BASELINE 2 in this research
("1 - predicted_joint_probability"), not merely correlated with it. At
T<1.0, raw counterexample_mass(T) can be made to look smaller purely by
excluding low-probability losing worlds into "outside_mass" -- WITHOUT any
real risk reduction -- while rho(T) cannot be gamed this way: it can only
ever equal or exceed the honest T=1.0 baseline. This is the concrete,
provable form of mission section 11's "world contraction should not
manufacture false certainty" warning, and it is why Variant B's primary
candidate quantity is evaluated against rho, not raw counterexample_mass,
in this research.

============================================================
SAMPLING (mission section 21 -- research pair observations must not use
"first N generated pairs")
============================================================
A real development day's cross-product of action-eligible legs reaches
tens of thousands of raw pairs (measured: up to ~106,000 on one real
DEVELOPMENT day). Processing literally all of them across 16 development
days is computationally excessive for a research pass and unnecessary for
valid inference. SAMPLE_CAP_PER_DAY bounds this via DETERMINISTIC,
quality/outcome-independent hash sampling: every raw pair's key is
SHA256("leg_a||leg_b") (legs sorted, so order-independent), pairs are
sorted by that hash, and the first SAMPLE_CAP_PER_DAY survive. This has NO
relationship to predicted probability, price, or realized outcome -- a day
with fewer than the cap is not sampled at all (sampling_rate=1.0).
sampling_method/version/inclusion count/rate are recorded on every row.
"""

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import (
    build_binary_outcome_set,
    build_world_distribution,
    world_id_from_outcomes,
)
from sports.mlb.research.h_over_ranker.data_windows import DERIVE_STAMPS, SELECT_STAMPS, verify_against_disk
from sports.mlb.research.joint_position_builder_v2.multi_target_universe import (
    PRICED_TARGETS,
    action_universe,
    build_multi_target_universe,
)

WORLD_GATE_RESEARCH_VERSION = "WORLD_GATE_RESEARCH_V1"

# Predeclared BEFORE reading any outcome (mission section 12): "Do not
# optimize a continuous threshold against outcomes." This exact tuple is
# frozen for the whole research pass -- DERIVE's sweep and SELECT's
# once-only check both use it unchanged.
APS_GRID: tuple[float, ...] = (1.0, 0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50)

# Deterministic, quality/outcome-independent research sampling cap -- see
# module docstring's SAMPLING section. Not a production gate.
SAMPLE_CAP_PER_DAY = 2000
SAMPLING_METHOD = "SHA256_PAIR_KEY_HASH_SORT_V1"


def _leg_id(row: pd.Series) -> str:
    target = row["target"] if "target" in row.index else "H"
    return f"{row['player']}|{target}|{row['direction']}|{row['market_line']}|{row['game_id']}"


def _pair_hash(leg_a: str, leg_b: str) -> str:
    a, b = sorted([leg_a, leg_b])
    return hashlib.sha256(f"{a}||{b}".encode("utf-8")).hexdigest()


def _diagnostics_at_threshold(distribution, ww_id: int, threshold: float) -> dict:
    """Builds a fresh BinaryOutcomeSet at `threshold` and returns the raw
    section-2 quantities. calibration_slates is irrelevant to which worlds
    are retained (verified in the audit tests) so it is passed as 0
    unconditionally here -- this never affects the result."""
    outcome_set = build_binary_outcome_set(distribution, aps_threshold=threshold, calibration_slates=0)
    probabilities = distribution.probabilities
    retained = outcome_set.world_ids
    retained_mass = float(probabilities[retained].sum()) if len(retained) else 0.0
    losing = retained[retained != ww_id]
    counterexample_mass = float(probabilities[losing].sum()) if len(losing) else 0.0
    outside_mass = 1.0 - retained_mass
    return {
        "retained_world_count": int(len(retained)),
        "retained_probability_mass": retained_mass,
        "counterexample_count": int(len(losing)),
        "counterexample_mass": counterexample_mass,
        "outside_mass": outside_mass,
        "rho": counterexample_mass + outside_mass,
        "nonvacuous_world_certificate": bool(len(retained) > 0 and len(losing) == 0),
    }


def _select_sampled_pairs(action: pd.DataFrame, *, cap: int) -> tuple[list[tuple[int, int]], float]:
    all_idx = list(combinations(range(len(action)), 2))
    leg_ids = [_leg_id(action.iloc[i]) for i in range(len(action))]
    hashed = sorted(all_idx, key=lambda ij: _pair_hash(leg_ids[ij[0]], leg_ids[ij[1]]))
    n_total = len(hashed)
    sampled = hashed[:cap]
    rate = (len(sampled) / n_total) if n_total else 0.0
    return sampled, rate


def build_pair_development_table(
    stamps: tuple[str, ...],
    *,
    sample_cap_per_day: int = SAMPLE_CAP_PER_DAY,
    aps_grid: tuple[float, ...] = APS_GRID,
    mode: str = "broad",
) -> pd.DataFrame:
    """One row per SAMPLED frozen pregame development candidate pair (mission
    section 6/21). World diagnostics are a pure function of PREGAME marginal
    probabilities -- computed identically regardless of win_i/win_j, which
    are attached only afterward from the same frozen day's already-settled
    archived outcomes (no leakage into the diagnostics themselves; see
    test_no_leakage_same_day_outcome_never_affects_world_diagnostics)."""
    rows: list[dict] = []
    ww_id = world_id_from_outcomes([1, 1])

    for stamp in stamps:
        universe = build_multi_target_universe((stamp,), targets=PRICED_TARGETS, mode=mode)
        if universe.empty:
            continue
        action = action_universe(universe).reset_index(drop=True)
        if len(action) < 2:
            continue

        sampled, sampling_rate = _select_sampled_pairs(action, cap=sample_cap_per_day)
        inclusion_hash = hashlib.sha256(f"{stamp}|{sample_cap_per_day}|{len(sampled)}".encode()).hexdigest()[:16]

        for i, j in sampled:
            row_i, row_j = action.iloc[i], action.iloc[j]
            leg_i, leg_j = _leg_id(row_i), _leg_id(row_j)
            same_game = bool(row_i["game_id"] == row_j["game_id"])
            same_team = bool(row_i.get("team") == row_j.get("team")) if "team" in action.columns else False

            p_i = float(np.clip(row_i["marginal_probability"], 1e-4, 1 - 1e-4))
            p_j = float(np.clip(row_j["marginal_probability"], 1e-4, 1 - 1e-4))
            distribution = build_world_distribution([leg_i, leg_j], [p_i, p_j])
            probabilities = distribution.probabilities
            p_joint = float(probabilities[ww_id])

            base = _diagnostics_at_threshold(distribution, ww_id, 1.0)
            grid = {T: _diagnostics_at_threshold(distribution, ww_id, T) for T in aps_grid}

            quoted_pair_price = None
            if not same_game and pd.notna(row_i["decimal_price"]) and pd.notna(row_j["decimal_price"]):
                quoted_pair_price = float(row_i["decimal_price"]) * float(row_j["decimal_price"])

            win_i, win_j = int(row_i["win"]), int(row_j["win"])
            both_win = bool(win_i and win_j)
            pair_loss = int(not both_win)
            actual_pair_return = None
            if quoted_pair_price is not None:
                actual_pair_return = (quoted_pair_price - 1.0) if both_win else -1.0

            row = {
                "date": stamp,
                "slate_id": stamp,
                "pair_id": _pair_hash(leg_i, leg_j),
                "leg_1_event_id": leg_i,
                "leg_2_event_id": leg_j,
                "same_game": same_game,
                "same_team": same_team,
                "market_pair_type": "|".join(sorted([str(row_i["target"]), str(row_j["target"])])),
                "line_pair_type": "__".join(sorted([
                    f"{row_i['target']}|{row_i['direction']}|{row_i['market_line']}",
                    f"{row_j['target']}|{row_j['direction']}|{row_j['market_line']}",
                ])),
                "state_pair_bucket": "H_OVER_RANKER_V1+MULTI_TARGET|" + mode,
                "quoted_pair_price": quoted_pair_price,
                "predicted_joint_probability": p_joint,
                "predicted_independence_probability": p_joint,  # no non-independence joint model exists (see pair_schema.py)
                "predicted_independence_failure": 1.0 - (p_i * p_j),  # BASELINE 1 -- note p_i*p_j == p_joint here (independence model)
                "retained_world_count": base["retained_world_count"],
                "retained_probability_mass": base["retained_probability_mass"],
                "outside_mass": base["outside_mass"],
                "counterexample_count": base["counterexample_count"],
                "counterexample_mass": base["counterexample_mass"],
                "counterexample_fraction": (base["counterexample_count"] / base["retained_world_count"]) if base["retained_world_count"] else None,
                "rho": base["rho"],
                "world_contraction_bits": float(np.log2(4 / base["retained_world_count"])) if base["retained_world_count"] else None,
                "nonvacuous_world_certificate": base["nonvacuous_world_certificate"],
                "both_win": both_win,
                "pair_loss": pair_loss,
                "actual_pair_return": actual_pair_return,
                "predictive_version": "H_OVER_RANKER_V1+MULTI_TARGET",
                "world_model_version": "MLB_BINARY_OUTCOME_SET_V1_SHADOW",
                "sampling_method": SAMPLING_METHOD,
                "sampling_version": WORLD_GATE_RESEARCH_VERSION,
                "sampling_rate_this_day": sampling_rate,
                "inclusion_hash": inclusion_hash,
            }
            for T in aps_grid:
                g = grid[T]
                row[f"cx_mass_T{T}"] = g["counterexample_mass"]
                row[f"outside_mass_T{T}"] = g["outside_mass"]
                row[f"rho_T{T}"] = g["rho"]
                row[f"nonvacuous_T{T}"] = g["nonvacuous_world_certificate"]
                row[f"retained_mass_T{T}"] = g["retained_probability_mass"]
            rows.append(row)

    return pd.DataFrame(rows)


def usable_stamps(stamps: tuple[str, ...], *, mode: str = "broad") -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Splits `stamps` into (usable, empty) -- some frozen DEVELOPMENT_STAMPS
    have zero action-eligible rows on disk (verified: 4 of 8 DERIVE_STAMPS).
    Reports both rather than silently dropping the empty ones."""
    usable: list[str] = []
    empty: list[str] = []
    for stamp in stamps:
        universe = build_multi_target_universe((stamp,), targets=PRICED_TARGETS, mode=mode)
        if universe.empty:
            empty.append(stamp)
            continue
        action = action_universe(universe)
        (usable if len(action) >= 2 else empty).append(stamp)
    return tuple(usable), tuple(empty)


if __name__ == "__main__":
    verify_against_disk()
    derive_usable, derive_empty = usable_stamps(DERIVE_STAMPS)
    select_usable, select_empty = usable_stamps(SELECT_STAMPS)
    print(f"DERIVE usable={derive_usable} empty={derive_empty}")
    print(f"SELECT usable={select_usable} empty={select_empty}")
