from __future__ import annotations

"""2-leg candidate pairs: classification, joint probability, conservative
lower bound, D_S, and the compatible-worlds counterexample certificate.

3/4-leg promotion is intentionally not implemented anywhere in this module
-- see ablation.py / manifest.py.

D_S convention (documented because it matters for the "never substitute
product odds for a real quote" rule): for DIFFERENT-game pairs, the product
of the two legs' own decimal odds IS the real, standard sportsbook payout
convention for a straight (non-SGP) parlay across independent games -- not
an estimate. This repo's own CONTROL system
(sports/parlay_analysis.py::score_candidate_parlays) prices its
cross-game combined_decimal_price the same way. For SAME-GAME pairs, no
real same-game-parlay quote exists anywhere in this repo's data, and no
correlation model is being fit here (out of scope per the task) -- D_S is
therefore None for same-game pairs, and joint_EV/joint_EV_LCB are also
None: probability/mechanism only, never a real-money EV claim built on a
guessed price.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import (
    BinaryOutcomeSet,
    WorldDistribution,
    aps_world_scores,
    build_binary_outcome_set,
    build_world_distribution,
    world_id_from_outcomes,
)

PROBABILITY_EPSILON = 1e-4
SHRINKAGE_K = 1.0  # mechanical conservatism constant -- NOT a fitted parameter; see manifest.py


def pair_class(ev_i: float | None, ev_j: float | None) -> str:
    """++ / +- / -- from DIAGNOSTIC marginal EVs. Unknown (missing price) legs
    are never silently treated as +EV; a pair with any unknown leg EV is
    reported as "?" and excluded from the class comparison entirely."""
    if ev_i is None or ev_j is None:
        return "?"
    pos_i, pos_j = ev_i > 0.0, ev_j > 0.0
    if pos_i and pos_j:
        return "++"
    if not pos_i and not pos_j:
        return "--"
    return "+-"


def _leg_uncertainty(rmse: float, history_rows: float) -> float:
    return float(rmse) / np.sqrt(max(float(history_rows), 1.0))


def conservative_joint_lower_bound(p_joint: float, unc_i: float, unc_j: float, k: float = SHRINKAGE_K) -> float:
    """Mechanical shrinkage toward 0, NOT a fitted/calibrated confidence
    interval -- widens with each leg's own rmse and narrows with more
    history_rows support. Documented explicitly as a heuristic in
    manifest.py; a real calibrated bound is future work."""
    combined = float(np.sqrt(unc_i**2 + unc_j**2))
    shrinkage_factor = float(np.clip(k * combined, 0.0, 1.0))
    return float(np.clip(p_joint * (1.0 - shrinkage_factor), 0.0, 1.0))


@dataclass(frozen=True)
class PairCertificate:
    """Compatible-worlds certificate for a FIXED pair S={i,j}, reusing
    outcome_worlds.py's joint binary-world machinery unchanged.

    B_S(C_t) = {retained worlds where at least one of i,j loses}. The
    certificate holds iff B_S(C_t) is empty -- i.e. every retained world is
    the "both win" world. This is relative to the represented/calibrated
    world set, not a claim of real-world certainty (see class docstring in
    outcome_worlds.PerfectParlayCertificate for the identical caveat this
    package inherits)."""

    retained_world_count: int
    counterexample_count: int
    counterexample_mass: float
    world_contraction_bits: float | None
    logical_certificate: bool


def build_pair_certificate(
    distribution: WorldDistribution, outcome_set: BinaryOutcomeSet
) -> PairCertificate:
    both_win_world = world_id_from_outcomes([1, 1])
    retained_ids = outcome_set.world_ids
    retained_count = int(len(retained_ids))
    counterexample_ids = retained_ids[retained_ids != both_win_world]
    counterexample_mass = float(distribution.probabilities[counterexample_ids].sum())
    contraction = float(np.log2(4 / retained_count)) if retained_count > 0 else None
    return PairCertificate(
        retained_world_count=retained_count,
        counterexample_count=int(len(counterexample_ids)),
        counterexample_mass=counterexample_mass,
        world_contraction_bits=contraction,
        logical_certificate=bool(retained_count > 0 and len(counterexample_ids) == 0),
    )


@dataclass(frozen=True)
class CandidatePair:
    date: str
    leg_i: str
    leg_j: str
    game_i: str
    game_j: str
    same_game: bool
    p_i: float
    p_j: float
    ev_i: float | None
    ev_j: float | None
    pair_class: str
    p_joint: float
    p_joint_l: float
    d_s: float | None
    joint_ev: float | None
    joint_ev_lcb: float | None
    certificate: PairCertificate
    win_i: int
    win_j: int
    aps_score_true_world: float
    support_min_history_rows: float
    support_max_rmse: float

    @property
    def both_win(self) -> bool:
        return bool(self.win_i == 1 and self.win_j == 1)

    @property
    def true_world_id(self) -> int:
        return world_id_from_outcomes([self.win_i, self.win_j])


def enumerate_candidate_pairs(
    day_action_rows: pd.DataFrame,
    *,
    aps_threshold: float,
    calibration_slates: int,
    interactions_by_pair: dict[tuple[str, str], np.ndarray] | None = None,
) -> list[CandidatePair]:
    """2-LEG PAIRS ONLY. interactions_by_pair lets unit tests inject a
    hand-constructed dependence structure (see test theorem #2); the real
    backtest never passes it (independence -- see pairs.py module
    docstring for why that's the correct default here, not a shortcut)."""
    pairs: list[CandidatePair] = []
    rows = day_action_rows.reset_index(drop=True)
    for idx_i, idx_j in combinations(range(len(rows)), 2):
        row_i, row_j = rows.iloc[idx_i], rows.iloc[idx_j]
        # target is part of the identity: a player can carry the same
        # direction/market_line under two different targets (e.g. R-OVER-0.5
        # and TB-OVER-0.5), which collided into one leg id before this
        # target-aware fix (caught when generalizing beyond H -- see
        # multi_target_universe.py). game_id is ALSO part of the identity:
        # on a doubleheader day the same player can carry an identical
        # target/direction/market_line in two different games (both games'
        # lines land on the same number), which is a second, distinct
        # collision found while running the real multi-target backtest --
        # those are genuinely two different legs, not the same leg twice.
        target_i = row_i["target"] if "target" in row_i.index else "H"
        target_j = row_j["target"] if "target" in row_j.index else "H"
        leg_i = f"{row_i['player']}|{target_i}|{row_i['direction']}|{row_i['market_line']}|{row_i['game_id']}"
        leg_j = f"{row_j['player']}|{target_j}|{row_j['direction']}|{row_j['market_line']}|{row_j['game_id']}"
        same_game = bool(row_i["game_id"] == row_j["game_id"])

        p_i, p_j = float(row_i["marginal_probability"]), float(row_j["marginal_probability"])
        interactions = None
        if interactions_by_pair is not None:
            interactions = interactions_by_pair.get((leg_i, leg_j)) or interactions_by_pair.get((leg_j, leg_i))
        clipped = np.clip([p_i, p_j], PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        distribution = build_world_distribution([leg_i, leg_j], clipped, interactions=interactions)
        p_joint = float(distribution.probabilities[world_id_from_outcomes([1, 1])])

        unc_i = _leg_uncertainty(row_i["rmse"], row_i["history_rows"])
        unc_j = _leg_uncertainty(row_j["rmse"], row_j["history_rows"])
        p_joint_l = conservative_joint_lower_bound(p_joint, unc_i, unc_j)

        d_s = None
        if not same_game and row_i["decimal_price"] is not None and row_j["decimal_price"] is not None:
            d_s = float(row_i["decimal_price"]) * float(row_j["decimal_price"])
        joint_ev = (p_joint * d_s - 1.0) if d_s is not None else None
        joint_ev_lcb = (p_joint_l * d_s - 1.0) if d_s is not None else None

        scores = aps_world_scores(distribution)
        outcome_set = build_binary_outcome_set(
            distribution, aps_threshold=aps_threshold, calibration_slates=calibration_slates
        )
        certificate = build_pair_certificate(distribution, outcome_set)

        pairs.append(
            CandidatePair(
                date=str(row_i["date"]),
                leg_i=leg_i,
                leg_j=leg_j,
                game_i=str(row_i["game_id"]),
                game_j=str(row_j["game_id"]),
                same_game=same_game,
                p_i=p_i,
                p_j=p_j,
                ev_i=row_i["marginal_ev"],
                ev_j=row_j["marginal_ev"],
                pair_class=pair_class(row_i["marginal_ev"], row_j["marginal_ev"]),
                p_joint=p_joint,
                p_joint_l=p_joint_l,
                d_s=d_s,
                joint_ev=joint_ev,
                joint_ev_lcb=joint_ev_lcb,
                certificate=certificate,
                win_i=int(row_i["win"]),
                win_j=int(row_j["win"]),
                aps_score_true_world=float(scores[world_id_from_outcomes([int(row_i["win"]), int(row_j["win"])])]),
                support_min_history_rows=float(min(row_i["history_rows"], row_j["history_rows"])),
                support_max_rmse=float(max(row_i["rmse"], row_j["rmse"])),
            )
        )
    return pairs
