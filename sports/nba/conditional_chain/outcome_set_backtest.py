from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .outcome_worlds import (
    aps_world_scores,
    build_binary_outcome_set,
    build_world_distribution,
    certify_perfect_parlay,
    conformal_aps_threshold,
    guaranteed_winner_indices,
    search_parlay_proof_frontier,
    world_id_from_outcomes,
)
from .protocol import BINARY_OUTCOME_SET_PROTOCOL, BinaryOutcomeSetProtocol
from .proof_trajectory import (
    certificate_world_ceiling,
    minimum_support_contraction_bits,
)
from .survival_builder import score_recent_regime_candidates


EVIDENCE_STATUS = "REPEATEDLY_INSPECTED_SYNTHETIC_RESEARCH_EVIDENCE"


@dataclass(frozen=True)
class OutcomeSetReplay:
    decisions: pd.DataFrame
    calibration_scores: tuple[float, ...]
    report: dict[str, Any]


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_date",
        "player",
        "market",
        "side",
        "robust_score",
        "leg_result",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"outcome-set replay is missing columns: {missing}")
    prepared = frame.copy()
    prepared["event_date"] = pd.to_datetime(
        prepared["event_date"], errors="raise"
    ).dt.normalize()
    prepared["leg_result"] = pd.to_numeric(prepared["leg_result"], errors="coerce")
    prepared["binary_leg_win"] = prepared["leg_result"].eq(1.0).astype(int)
    return prepared


def _candidate_id(row: pd.Series) -> str:
    line = row.get("line", "")
    return "|".join(
        [
            str(row["player"]),
            str(row["market"]),
            str(row["side"]).upper(),
            str(line),
        ]
    )


def _replay_report(
    decisions: pd.DataFrame, protocol: BinaryOutcomeSetProtocol
) -> dict[str, Any]:
    evaluated = decisions.loc[decisions["evaluated"]].copy()
    realized_winner_counts = (
        evaluated["candidate_count"] - evaluated["non_win_settlements"]
        if len(evaluated)
        else pd.Series(dtype=float)
    )
    report: dict[str, Any] = {
        "outcome_set_version": protocol.version,
        "calibration_method": protocol.calibration_method,
        "binary_success_definition": protocol.binary_success_definition,
        "evidence_status": EVIDENCE_STATUS,
        "production_authorizable": False,
        "target_miscoverage": protocol.target_miscoverage,
        "target_marginal_coverage": 1.0 - protocol.target_miscoverage,
        "evaluated_slates": int(len(evaluated)),
        "total_slates": int(len(decisions)),
        "status_counts": {
            str(status): int(count)
            for status, count in decisions["status"].value_counts(dropna=False).items()
        },
        "empirical_outcome_set_coverage": (
            float(evaluated["realized_world_covered"].mean())
            if len(evaluated)
            else None
        ),
        "mean_retained_worlds": (
            float(evaluated["retained_world_count"].mean()) if len(evaluated) else None
        ),
        "retained_world_count_summary": {
            "minimum": (
                int(evaluated["retained_world_count"].min()) if len(evaluated) else None
            ),
            "median": (
                float(evaluated["retained_world_count"].median())
                if len(evaluated)
                else None
            ),
            "mean": (
                float(evaluated["retained_world_count"].mean())
                if len(evaluated)
                else None
            ),
            "maximum": (
                int(evaluated["retained_world_count"].max()) if len(evaluated) else None
            ),
        },
        "mean_retained_world_fraction": (
            float(evaluated["retained_world_fraction"].mean())
            if len(evaluated)
            else None
        ),
        "mean_guaranteed_winners": (
            float(evaluated["guaranteed_winner_count"].mean())
            if len(evaluated)
            else None
        ),
        "maximum_guaranteed_winners": (
            int(evaluated["guaranteed_winner_count"].max()) if len(evaluated) else 0
        ),
        "logical_certificates_by_leg_count": {},
        "structural_certificate_feasibility_by_leg_count": {},
        "ex_post_oracle_feasibility_by_leg_count": {},
        "shadow_proof_frontier_by_leg_count": {},
        "realized_reservoir_winner_count": {
            "minimum": (
                int(realized_winner_counts.min())
                if len(realized_winner_counts)
                else None
            ),
            "mean": (
                float(realized_winner_counts.mean())
                if len(realized_winner_counts)
                else None
            ),
            "maximum": (
                int(realized_winner_counts.max())
                if len(realized_winner_counts)
                else None
            ),
        },
        "interpretation": (
            "A perfect-path certificate exists only when every retained binary world assigns "
            "every selected leg a win. Marginal outcome-set coverage does not establish the "
            "conditional failure rate on selected action slates."
        ),
    }
    for leg_count in protocol.requested_leg_counts:
        column = f"logical_{leg_count}_leg_certificate"
        report["logical_certificates_by_leg_count"][str(leg_count)] = (
            int(evaluated[column].sum()) if len(evaluated) else 0
        )
        if len(evaluated):
            ceilings = (
                evaluated["candidate_count"]
                .astype(int)
                .map(
                    lambda candidate_count: certificate_world_ceiling(
                        candidate_count, leg_count
                    )
                )
            )
            retained = evaluated["retained_world_count"].astype(int)
            nonempty = retained.gt(0)
            cardinality_feasible = nonempty & retained.le(ceilings)
            excess = (retained - ceilings).clip(lower=0)
            contraction_bits = pd.Series(
                [
                    minimum_support_contraction_bits(count, ceiling)
                    for count, ceiling in zip(retained, ceilings)
                ],
                index=evaluated.index,
                dtype=float,
            )
            certified = evaluated[column].astype(bool)
        else:
            ceilings = pd.Series(dtype=int)
            cardinality_feasible = pd.Series(dtype=bool)
            excess = pd.Series(dtype=float)
            contraction_bits = pd.Series(dtype=float)
            certified = pd.Series(dtype=bool)
        report["structural_certificate_feasibility_by_leg_count"][str(leg_count)] = {
            "necessary_condition": "0 < |C| <= 2^(M-n)",
            "world_ceiling_at_maximum_reservoir": certificate_world_ceiling(
                protocol.maximum_candidates, leg_count
            ),
            "cardinality_feasible_slates": int(cardinality_feasible.sum()),
            "evaluated_slates": int(len(evaluated)),
            "cardinality_feasibility_rate": (
                float(cardinality_feasible.mean()) if len(evaluated) else None
            ),
            "logical_certificates_on_cardinality_feasible_slates": int(
                (certified & cardinality_feasible).sum()
            ),
            "mean_positive_world_excess_above_ceiling": (
                float(excess.mean()) if len(excess) else None
            ),
            "mean_signed_world_gap_to_ceiling": (
                float((evaluated["retained_world_count"] - ceilings).mean())
                if len(evaluated)
                else None
            ),
            "mean_minimum_support_cardinality_bits_required": (
                float(contraction_bits.mean()) if len(contraction_bits) else None
            ),
            "interpretation": (
                "Cardinality feasibility is necessary but not sufficient. Removed worlds "
                "must also align on the same winner coordinates. The reported bit value is "
                "a support-cardinality log ratio, not Shannon information gain."
            ),
        }
        oracle_feasible = realized_winner_counts >= leg_count
        report["ex_post_oracle_feasibility_by_leg_count"][str(leg_count)] = {
            "feasible_slates": int(oracle_feasible.sum()),
            "evaluated_slates": int(len(evaluated)),
            "feasibility_rate": (
                float(oracle_feasible.mean()) if len(oracle_feasible) else None
            ),
            "evidence_role": "EX_POST_UPPER_BOUND_NOT_SELECTABLE_INFORMATION",
        }
        frontier_hit_column = f"frontier_{leg_count}_leg_hit"
        control_hit_column = f"control_{leg_count}_leg_hit"
        frontier_actions = (
            evaluated.loc[evaluated[frontier_hit_column].notna()].copy()
            if len(evaluated)
            else evaluated
        )
        report["shadow_proof_frontier_by_leg_count"][str(leg_count)] = {
            "action_slates": int(len(frontier_actions)),
            "complete_wins": (
                int(frontier_actions[frontier_hit_column].sum())
                if len(frontier_actions)
                else 0
            ),
            "hit_rate": (
                float(frontier_actions[frontier_hit_column].mean())
                if len(frontier_actions)
                else None
            ),
            "ordinary_top_n_complete_wins": (
                int(frontier_actions[control_hit_column].sum())
                if len(frontier_actions)
                else 0
            ),
            "ordinary_top_n_hit_rate": (
                float(frontier_actions[control_hit_column].mean())
                if len(frontier_actions)
                else None
            ),
            "mean_counterexample_worlds": (
                float(
                    frontier_actions[
                        f"frontier_{leg_count}_counterexample_worlds"
                    ].mean()
                )
                if len(frontier_actions)
                else None
            ),
            "mean_counterexample_mass_within_set": (
                float(
                    frontier_actions[f"frontier_{leg_count}_counterexample_mass"].mean()
                )
                if len(frontier_actions)
                else None
            ),
        }
    return report


def chronological_outcome_set_replay(
    reservoir: pd.DataFrame,
    *,
    block_label: str,
    initial_history: pd.DataFrame | None = None,
    initial_calibration_scores: Iterable[float] | None = None,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> OutcomeSetReplay:
    """Calibrate joint label-powerset sets using only earlier settled slates."""

    frame = _prepare_frame(reservoir)
    history_parts = [frame]
    if initial_history is not None:
        initial = _prepare_frame(initial_history)
        if len(initial) and initial["event_date"].max() >= frame["event_date"].min():
            raise ValueError("initial_history must end before the replay block")
        history_parts.insert(0, initial)
    available_history = pd.concat(history_parts, ignore_index=True, sort=False)
    initial_scores = (
        list(initial_calibration_scores)
        if initial_calibration_scores is not None
        else []
    )
    calibration_scores = initial_scores.copy()
    decision_rows: list[dict[str, Any]] = []

    for event_date, slate in frame.groupby("event_date", sort=True):
        scored = score_recent_regime_candidates(
            slate,
            available_history,
            as_of_date=event_date,
        )
        scored = scored.sort_values(
            ["survival_probability", "robust_score", "player", "market"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).head(protocol.maximum_candidates)
        scored = scored.reset_index(drop=True)
        scored["candidate_id"] = scored.apply(_candidate_id, axis=1)
        resolved = scored["leg_result"].isin([0.0, 0.5, 1.0])
        if not bool(resolved.all()):
            decision_rows.append(
                {
                    "block": block_label,
                    "event_date": pd.Timestamp(event_date),
                    "evaluated": False,
                    "status": "UNRESOLVED_WORLD",
                    "calibration_slates": len(calibration_scores),
                }
            )
            continue

        distribution = build_world_distribution(
            scored["candidate_id"],
            scored["survival_probability"],
            protocol=protocol,
        )
        true_world_id = world_id_from_outcomes(scored["binary_leg_win"].astype(int))
        true_score = float(aps_world_scores(distribution)[true_world_id])
        evaluated = len(calibration_scores) >= protocol.minimum_calibration_slates
        row: dict[str, Any] = {
            "block": block_label,
            "event_date": pd.Timestamp(event_date),
            "evaluated": evaluated,
            "status": "CALIBRATION_WARMUP",
            "candidate_count": distribution.candidate_count,
            "total_world_count": distribution.world_count,
            "true_world_id": true_world_id,
            "true_world_aps_score": true_score,
            "calibration_slates": len(calibration_scores),
            "outcome_set_version": protocol.version,
            "non_win_settlements": int(scored["binary_leg_win"].eq(0).sum()),
            "push_settlements_mapped_to_non_win": int(
                scored["leg_result"].eq(0.5).sum()
            ),
        }
        for leg_count in protocol.requested_leg_counts:
            row[f"logical_{leg_count}_leg_certificate"] = False
            row[f"frontier_{leg_count}_leg_hit"] = np.nan
            row[f"control_{leg_count}_leg_hit"] = np.nan
            row[f"frontier_{leg_count}_candidate_ids"] = ""
            row[f"frontier_{leg_count}_counterexample_worlds"] = np.nan
            row[f"frontier_{leg_count}_counterexample_mass"] = np.nan
            row[f"frontier_{leg_count}_posterior_all_win_probability"] = np.nan

        if evaluated:
            threshold = conformal_aps_threshold(
                calibration_scores,
                target_miscoverage=protocol.target_miscoverage,
            )
            outcome_set = build_binary_outcome_set(
                distribution,
                aps_threshold=threshold,
                calibration_slates=len(calibration_scores),
                protocol=protocol,
            )
            guaranteed = guaranteed_winner_indices(outcome_set)
            row.update(
                {
                    "status": (
                        "JOINT_OUTCOME_SET_EVALUATED"
                        if outcome_set.world_count
                        else "EMPTY_OUTCOME_SET_ABSTAIN"
                    ),
                    "aps_threshold": threshold,
                    "retained_world_count": outcome_set.world_count,
                    "retained_world_fraction": (
                        outcome_set.world_count / distribution.world_count
                    ),
                    "realized_world_covered": bool(
                        true_world_id in set(outcome_set.world_ids.tolist())
                    ),
                    "guaranteed_winner_count": len(guaranteed),
                    "guaranteed_candidate_ids": "|".join(
                        distribution.candidate_ids[index] for index in guaranteed
                    ),
                }
            )
            for leg_count in protocol.requested_leg_counts:
                certificate = certify_perfect_parlay(
                    scored,
                    outcome_set,
                    requested_leg_count=leg_count,
                    path_certificate=None,
                    protocol=protocol,
                )
                row[f"logical_{leg_count}_leg_certificate"] = (
                    certificate.logical_implication_proven
                )
                row[f"certificate_{leg_count}_status"] = certificate.status
                if outcome_set.world_count == 0:
                    continue
                frontier = search_parlay_proof_frontier(
                    scored,
                    outcome_set,
                    requested_leg_count=leg_count,
                    protocol=protocol,
                )
                wins_by_id = dict(
                    zip(scored["candidate_id"], scored["binary_leg_win"].astype(bool))
                )
                row[f"frontier_{leg_count}_candidate_ids"] = "|".join(
                    frontier.selected_candidate_ids
                )
                row[f"frontier_{leg_count}_counterexample_worlds"] = (
                    frontier.counterexample_world_count
                )
                row[f"frontier_{leg_count}_counterexample_mass"] = (
                    frontier.counterexample_mass_within_set
                )
                row[f"frontier_{leg_count}_posterior_all_win_probability"] = (
                    frontier.posterior_all_win_probability
                )
                row[f"frontier_{leg_count}_leg_hit"] = all(
                    wins_by_id[candidate_id]
                    for candidate_id in frontier.selected_candidate_ids
                )
                row[f"control_{leg_count}_leg_hit"] = bool(
                    scored.head(leg_count)["binary_leg_win"].astype(bool).all()
                )
        else:
            row.update(
                {
                    "aps_threshold": np.nan,
                    "retained_world_count": np.nan,
                    "retained_world_fraction": np.nan,
                    "realized_world_covered": np.nan,
                    "guaranteed_winner_count": np.nan,
                    "guaranteed_candidate_ids": "",
                }
            )
        decision_rows.append(row)
        calibration_scores.append(true_score)

    decisions = pd.DataFrame(decision_rows)
    report = {
        "block": block_label,
        "starting_calibration_slates": int(len(initial_scores)),
        "ending_calibration_slates": int(len(calibration_scores)),
        **_replay_report(decisions, protocol),
    }
    return OutcomeSetReplay(
        decisions=decisions,
        calibration_scores=tuple(float(score) for score in calibration_scores),
        report=report,
    )


def combine_outcome_set_replays(
    replays: Iterable[OutcomeSetReplay],
    *,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> OutcomeSetReplay:
    replay_list = list(replays)
    decisions = pd.concat([item.decisions for item in replay_list], ignore_index=True)
    report = {
        **_replay_report(decisions, protocol),
        "blocks": [item.report for item in replay_list],
        "research_gate": {
            "status": (
                "PATH_INFORMATION_REQUIRED"
                if not bool(
                    decisions.loc[
                        decisions["evaluated"], "logical_2_leg_certificate"
                    ].any()
                )
                else "LOGICAL_INTERSECTION_OBSERVED_NOT_PROSPECTIVE"
            ),
            "path_incremental_value_required": True,
            "selective_risk_certificate_required": True,
        },
    }
    return OutcomeSetReplay(
        decisions=decisions,
        calibration_scores=replay_list[-1].calibration_scores,
        report=report,
    )
