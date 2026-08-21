from __future__ import annotations

"""2x2 chronological ablation: {narrow, broad} state x {++_only, all_classes} pairs.

    A: narrow state + ++ pairs only   (closest to CONTROL's own admission rule)
    B: broad  state + ++ pairs only   (isolates: information gained from formerly
                                        filtered markets, holding the pair-class
                                        filter fixed)
    C: narrow state + all pair classes (isolates: action value gained from
                                        allowing marginally -EV legs, holding
                                        the state input fixed)
    D: broad  state + all pair classes (both effects together)

DEVELOPMENT_STAMPS only (DERIVE+SELECT, the same frozen partition
h_over_ranker uses). TEST_STAMPS stays retired -- see
manifest.PROSPECTIVE_CONFIRMATION_PROTOCOL for why a "fresh TEST" isn't
attempted here (this repo does not yet have real days beyond
DEVELOPMENT_STAMPS/TEST_STAMPS to test on).
"""

from dataclasses import dataclass

import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import conformal_aps_threshold
from sports.mlb.research.h_over_ranker.data_windows import DEVELOPMENT_STAMPS, verify_against_disk

from .observation_universe import action_universe, build_observation_universe
from .pairs import CandidatePair, PairCertificate, enumerate_candidate_pairs
from .risk_gate import SelectiveRiskCertificate, build_selective_risk_certificate, gate_and_rank_day

MIN_CALIBRATION_PAIRS = 20  # matches this repo's existing convention (see h_over_ranker, conditional_chain)
TARGET_MISCOVERAGE = 0.10
JOINT_EV_LCB_MARGIN = 0.0
MIN_SUPPORT_HISTORY_ROWS = 20.0
RISK_TARGET = 0.30  # configurable failure-rate ceiling for the selective-risk gate

VARIANTS = {
    "A_narrow_pp": {"state": "narrow", "pair_filter": "pp_only"},
    "B_broad_pp": {"state": "broad", "pair_filter": "pp_only"},
    "C_narrow_all": {"state": "narrow", "pair_filter": "all_classes"},
    "D_broad_all": {"state": "broad", "pair_filter": "all_classes"},
}

_PAIR_RECORD_COLUMNS = (
    "date", "leg_i", "leg_j", "same_game", "pair_class", "p_i", "p_j", "ev_i", "ev_j",
    "p_joint", "p_joint_l", "d_s", "joint_ev", "joint_ev_lcb",
    "counterexample_mass", "counterexample_count", "retained_world_count",
    "world_contraction_bits", "logical_certificate", "win_i", "win_j", "both_win",
    "support_min_history_rows", "support_max_rmse",
    "calibration_pairs_prior", "calibration_days_prior", "evaluated",
)


def _filter_pairs(pairs: list[CandidatePair], pair_filter: str) -> list[CandidatePair]:
    if pair_filter == "pp_only":
        return [p for p in pairs if p.pair_class == "++"]
    return [p for p in pairs if p.pair_class in ("++", "+-", "--")]


def _pair_to_record(pair: CandidatePair, calibration_pairs_prior: int, calibration_days_prior: int) -> dict:
    return {
        "date": pair.date,
        "leg_i": pair.leg_i,
        "leg_j": pair.leg_j,
        "same_game": pair.same_game,
        "pair_class": pair.pair_class,
        "p_i": pair.p_i,
        "p_j": pair.p_j,
        "ev_i": pair.ev_i,
        "ev_j": pair.ev_j,
        "p_joint": pair.p_joint,
        "p_joint_l": pair.p_joint_l,
        "d_s": pair.d_s,
        "joint_ev": pair.joint_ev,
        "joint_ev_lcb": pair.joint_ev_lcb,
        "counterexample_mass": pair.certificate.counterexample_mass,
        "counterexample_count": pair.certificate.counterexample_count,
        "retained_world_count": pair.certificate.retained_world_count,
        "world_contraction_bits": pair.certificate.world_contraction_bits,
        "logical_certificate": pair.certificate.logical_certificate,
        "win_i": pair.win_i,
        "win_j": pair.win_j,
        "both_win": pair.both_win,
        "support_min_history_rows": pair.support_min_history_rows,
        "support_max_rmse": pair.support_max_rmse,
        "calibration_pairs_prior": calibration_pairs_prior,
        "calibration_days_prior": calibration_days_prior,
        "evaluated": calibration_pairs_prior >= MIN_CALIBRATION_PAIRS,
    }


@dataclass
class VariantResult:
    name: str
    state: str
    pair_filter: str
    all_pairs: pd.DataFrame
    action_decisions: pd.DataFrame
    risk_certificate: SelectiveRiskCertificate


def run_variant(name: str, state: str, pair_filter: str) -> VariantResult:
    verify_against_disk()
    universe = build_observation_universe(DEVELOPMENT_STAMPS, mode=state)
    action = action_universe(universe)
    dates = sorted(action["date"].unique())

    calibration_scores: list[float] = []
    calibration_days_seen: set[str] = set()
    pair_records: list[dict] = []

    for date in dates:
        day_rows = action[action["date"] == date].reset_index(drop=True)
        if len(day_rows) < 2:
            continue
        if len(calibration_scores) < MIN_CALIBRATION_PAIRS:
            threshold = 1.0  # not yet calibrated: retain everything (diagnostics only, never actioned -- see below)
        else:
            threshold = conformal_aps_threshold(calibration_scores, target_miscoverage=TARGET_MISCOVERAGE)

        day_pairs = enumerate_candidate_pairs(
            day_rows, aps_threshold=threshold, calibration_slates=len(calibration_scores)
        )
        filtered = _filter_pairs(day_pairs, pair_filter)
        for pair in filtered:
            pair_records.append(_pair_to_record(pair, len(calibration_scores), len(calibration_days_seen)))
            # walk-forward: append AFTER recording, so no pair's own outcome
            # ever informs the threshold used to score it or any pair before it.
            calibration_scores.append(pair.aps_score_true_world)
        calibration_days_seen.add(date)

    all_pairs = pd.DataFrame(pair_records, columns=list(_PAIR_RECORD_COLUMNS))
    evaluated_pairs = all_pairs[all_pairs["evaluated"]] if not all_pairs.empty else all_pairs
    if evaluated_pairs.empty:
        risk_certificate = SelectiveRiskCertificate("INSUFFICIENT_EVALUATED_PAIRS", RISK_TARGET, None, 0, 0, 0, None, None)
    else:
        risk_certificate = build_selective_risk_certificate(evaluated_pairs, risk_target=RISK_TARGET)

    action_rows = []
    if not evaluated_pairs.empty:
        for date, day_pairs_df in evaluated_pairs.groupby("date", sort=True):
            day_candidate_pairs = [_row_to_candidate_pair(r) for _, r in day_pairs_df.iterrows()]
            decision = gate_and_rank_day(
                day_candidate_pairs,
                joint_ev_lcb_margin=JOINT_EV_LCB_MARGIN,
                min_support_history_rows=MIN_SUPPORT_HISTORY_ROWS,
                risk_certificate=risk_certificate,
            )
            selected = day_candidate_pairs[decision.selected_pair_index] if decision.selected_pair_index is not None else None
            action_rows.append(
                {
                    "date": date,
                    "action": decision.action,
                    "selected_pair": f"{selected.leg_i} + {selected.leg_j}" if selected else None,
                    "selected_both_win": selected.both_win if selected else None,
                    "selected_joint_ev_lcb": selected.joint_ev_lcb if selected else None,
                    "reason": decision.reason,
                }
            )
    action_decisions = pd.DataFrame(action_rows)

    return VariantResult(name, state, pair_filter, all_pairs, action_decisions, risk_certificate)


def _row_to_candidate_pair(row: pd.Series) -> CandidatePair:
    return CandidatePair(
        date=row["date"],
        leg_i=row["leg_i"],
        leg_j=row["leg_j"],
        game_i="",
        game_j="",
        same_game=bool(row["same_game"]),
        p_i=row["p_i"],
        p_j=row["p_j"],
        ev_i=row["ev_i"],
        ev_j=row["ev_j"],
        pair_class=row["pair_class"],
        p_joint=row["p_joint"],
        p_joint_l=row["p_joint_l"],
        d_s=row["d_s"],
        joint_ev=row["joint_ev"],
        joint_ev_lcb=row["joint_ev_lcb"],
        certificate=PairCertificate(
            retained_world_count=row["retained_world_count"],
            counterexample_count=row["counterexample_count"],
            counterexample_mass=row["counterexample_mass"],
            world_contraction_bits=row["world_contraction_bits"],
            logical_certificate=row["logical_certificate"],
        ),
        win_i=int(row["win_i"]),
        win_j=int(row["win_j"]),
        aps_score_true_world=0.0,  # not needed for gating/ranking; not recomputed here
        support_min_history_rows=row["support_min_history_rows"],
        support_max_rmse=row["support_max_rmse"],
    )


def run_all_variants() -> dict[str, VariantResult]:
    return {name: run_variant(name, cfg["state"], cfg["pair_filter"]) for name, cfg in VARIANTS.items()}
