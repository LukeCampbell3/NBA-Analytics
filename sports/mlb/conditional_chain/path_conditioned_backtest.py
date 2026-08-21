from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta

from .outcome_worlds import (
    BinaryOutcomeSet,
    WorldDistribution,
    aps_world_scores,
    build_binary_outcome_set,
    build_world_distribution,
    conformal_aps_threshold,
    search_parlay_proof_frontier,
    certify_perfect_parlay,
    update_world_distribution,
    world_id_from_outcomes,
)
from .path_world_evidence import (
    CHECKPOINT_LABELS,
    CHECKPOINT_MINUTES,
    build_candidate_evidence_path,
    candidate_posteriors_to_world_log_evidence,
    endpoint_posteriors,
    fit_path_evidence_bundle,
    merge_candidates_with_paths,
    path_posteriors,
)
from .proof_trajectory import build_proof_trajectory
from .protocol import BINARY_OUTCOME_SET_PROTOCOL, BinaryOutcomeSetProtocol


MECHANISM_STATUS = "REAL_MLB_PATH_MECHANISM_IMPLEMENTED_INCREMENTAL_VALUE_UNPROVEN"
PRIMARY_LEG_COUNT = 2

# Unlike sports/nba/conditional_chain, this package does not port a
# survival_builder/frozen_selector/chain_resolver layer. The reservoir is
# expected to already carry a day-of prior probability
# (``survival_probability``) and a stable historical base-rate feature
# (``robust_score``) produced by the existing MLB prediction-pool pipeline
# (see sports/mlb/scripts/generate_daily_prediction_pool.py and
# sports/mlb/scripts/pick_survival_model.py, and
# sports/mlb/conditional_chain/build_reservoir_from_history.py for the
# adapter that maps published pool output onto this contract). This module
# only adds checkpoint path evidence on top of that already-scored pool; it
# never re-derives the prior itself.


@dataclass(frozen=True)
class PathConditionedReplay:
    decisions: pd.DataFrame
    proof_trajectories: pd.DataFrame
    checkpoint_evidence: pd.DataFrame
    candidate_evidence: pd.DataFrame
    report: dict[str, Any]
    selective_risk_report: dict[str, Any]
    ablation_report: dict[str, Any]
    calibration_scores: dict[str, tuple[float, ...]]


def _prepare_reservoir(
    frame: pd.DataFrame, *, protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL
) -> pd.DataFrame:
    required = {
        "event_date",
        "player",
        "market",
        "side",
        "robust_score",
        "survival_probability",
        "leg_result",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"path-conditioned replay is missing columns: {missing}")
    result = frame.copy()
    result["event_date"] = pd.to_datetime(result["event_date"], errors="raise").dt.normalize()
    result["robust_score"] = pd.to_numeric(result["robust_score"], errors="coerce")
    result["survival_probability"] = np.clip(
        pd.to_numeric(result["survival_probability"], errors="coerce"),
        protocol.score_epsilon,
        1.0 - protocol.score_epsilon,
    )
    result["leg_result"] = pd.to_numeric(result["leg_result"], errors="coerce")
    result["binary_leg_win"] = result["leg_result"].eq(1.0).astype(int)
    return result


def _prepare_paths(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "event_date" not in result:
        raise ValueError("path features are missing event_date")
    result["event_date"] = pd.to_datetime(result["event_date"], errors="raise").dt.normalize()
    return result


def _candidate_id(row: pd.Series) -> str:
    line = row.get("line", "")
    return "|".join(
        [str(row["player"]), str(row["market"]), str(row["side"]).upper(), str(line)]
    )


def _fixed_metrics(
    outcome_set: BinaryOutcomeSet,
    candidate_ids: Iterable[str],
) -> tuple[int | None, float | None, bool]:
    target = tuple(str(value) for value in candidate_ids)
    if outcome_set.world_count == 0:
        return None, None, False
    distribution = outcome_set.distribution
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(distribution.candidate_ids)}
    indices = np.asarray([index_by_id[value] for value in target], dtype=int)
    retained_ids = outcome_set.world_ids
    retained = distribution.outcomes[retained_ids]
    counterexample = ~retained[:, indices].all(axis=1)
    counterexample_ids = retained_ids[counterexample]
    retained_mass = float(distribution.probabilities[retained_ids].sum())
    counterexample_mass = float(distribution.probabilities[counterexample_ids].sum())
    return (
        int(len(counterexample_ids)),
        counterexample_mass / retained_mass if retained_mass > 0.0 else None,
        len(counterexample_ids) == 0,
    )


def _threshold(scores: list[float], protocol: BinaryOutcomeSetProtocol) -> float | None:
    if len(scores) < protocol.minimum_calibration_slates:
        return None
    return conformal_aps_threshold(scores, target_miscoverage=protocol.target_miscoverage)


def _exact_failure_ucb(failures: int, actions: int, alpha: float = 0.05) -> float | None:
    if actions <= 0:
        return None
    if failures >= actions:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, actions - failures))


def _selective_risk_report(
    evaluated: pd.DataFrame,
    *,
    risk_target: float | None,
    minimum_development_actions: int = 10,
) -> dict[str, Any]:
    if evaluated.empty:
        return {
            "status": "INSUFFICIENT_EVALUATED_SLATES",
            "risk_target": risk_target,
            "development_slates": 0,
            "validation_slates": 0,
        }
    ordered = evaluated.sort_values("event_date", kind="mergesort").reset_index(drop=True)
    split = max(len(ordered) // 2, 1)
    development = ordered.iloc[:split].copy()
    validation = ordered.iloc[split:].copy()
    mass_column = f"real_path_{PRIMARY_LEG_COUNT}_leg_counterexample_mass"
    hit_column = f"fixed_{PRIMARY_LEG_COUNT}_leg_hit"
    if mass_column not in development.columns or hit_column not in development.columns:
        return {
            "status": "NO_FIXED_PRIMARY_TARGET",
            "risk_target": risk_target,
            "development_slates": int(len(development)),
            "validation_slates": int(len(validation)),
        }
    finite = development[mass_column].notna() & development[hit_column].notna()
    development = development.loc[finite]
    candidate_thresholds = sorted(set(float(value) for value in development[mass_column]))
    sweep: list[dict[str, Any]] = []
    for threshold in candidate_thresholds:
        actions = development.loc[development[mass_column] <= threshold]
        failures = int((~actions[hit_column].astype(bool)).sum())
        sweep.append(
            {
                "threshold": threshold,
                "actions": int(len(actions)),
                "coverage": float(len(actions) / len(development)) if len(development) else 0.0,
                "failures": failures,
                "failure_rate": float(failures / len(actions)) if len(actions) else None,
                "exact_one_sided_95_failure_ucb": _exact_failure_ucb(failures, len(actions)),
            }
        )
    report: dict[str, Any] = {
        "risk_target": risk_target,
        "bound": "Clopper-Pearson exact one-sided 95% binomial upper bound",
        "development_slates": int(len(development)),
        "validation_slates": int(len(validation)),
        "development_threshold_sweep": sweep,
        "frozen_threshold": None,
        "validation": None,
    }
    if risk_target is None:
        report["status"] = "NO_PREDECLARED_RISK_TARGET"
        return report
    if not 0.0 < risk_target < 1.0:
        raise ValueError("risk_target must lie strictly between zero and one")
    eligible = [
        row
        for row in sweep
        if row["actions"] >= minimum_development_actions
        and row["exact_one_sided_95_failure_ucb"] is not None
        and row["exact_one_sided_95_failure_ucb"] <= risk_target
    ]
    if not eligible:
        report["status"] = "NO_DEVELOPMENT_THRESHOLD_MEETS_RISK_BOUND"
        return report
    frozen = max(eligible, key=lambda row: (row["coverage"], row["threshold"]))
    threshold = float(frozen["threshold"])
    validation_actions = validation.loc[
        validation[mass_column].notna() & (validation[mass_column] <= threshold)
    ]
    failures = int((~validation_actions[hit_column].astype(bool)).sum())
    ucb = _exact_failure_ucb(failures, len(validation_actions))
    report["frozen_threshold"] = threshold
    report["validation"] = {
        "actions": int(len(validation_actions)),
        "coverage": float(len(validation_actions) / len(validation)) if len(validation) else 0.0,
        "failures": failures,
        "failure_rate": float(failures / len(validation_actions)) if len(validation_actions) else None,
        "exact_one_sided_95_failure_ucb": ucb,
    }
    report["status"] = (
        "SELECTIVE_RISK_BOUND_SUPPORTED_ON_VALIDATION"
        if ucb is not None and ucb <= risk_target
        else "SELECTIVE_RISK_BOUND_NOT_SUPPORTED_ON_VALIDATION"
    )
    return report


def _paired_bootstrap_lcb(values: np.ndarray, *, alpha: float = 0.05 / 3.0) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return None
    rng = np.random.default_rng(20260820)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, alpha))



def _one_sided_sign_flip_p(values: np.ndarray, *, samples: int = 50_000) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return None
    observed = float(values.mean())
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(20260821)
    exceedances = 0
    remaining = int(samples)
    while remaining > 0:
        chunk = min(5_000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(chunk, len(values)))
        permuted = (signs * values).mean(axis=1)
        exceedances += int(np.sum(permuted >= observed))
        remaining -= chunk
    return float((exceedances + 1) / (samples + 1))

def _ablation_report(evaluated: pd.DataFrame, path_certificate: dict[str, Any]) -> dict[str, Any]:
    hit_column = f"fixed_{PRIMARY_LEG_COUNT}_leg_hit"
    if evaluated.empty or hit_column not in evaluated.columns:
        return {
            "status": MECHANISM_STATUS,
            "primary_leg_count": PRIMARY_LEG_COUNT,
            "primary_loss": "Brier loss of fixed-parlay retained counterexample mass vs realized failure",
            "path_confirmation_valid": bool(
                path_certificate
                and path_certificate.get("status") == "PATH_INCREMENTAL_VALUE_SUPPORTED"
                and path_certificate.get("path_authorized", False)
            ),
            "variants": {},
            "paired_real_path_improvement_vs_controls": {},
            "interpretation": "No fixed primary-leg target was available for confirmatory ablation.",
        }
    actual_failure = (~evaluated[hit_column].astype(bool)).astype(float)
    variants = ["endpoint_only", "real_path", "shuffled_path", "inverted_path"]
    metrics: dict[str, Any] = {}
    losses: dict[str, np.ndarray] = {}
    for variant in variants:
        column = f"{variant}_{PRIMARY_LEG_COUNT}_leg_counterexample_mass"
        if column not in evaluated or not len(evaluated):
            metrics[variant] = {"slates": 0, "brier": None, "mean_counterexample_mass": None}
            losses[variant] = np.asarray([], dtype=float)
            continue
        values = pd.to_numeric(evaluated[column], errors="coerce")
        valid = values.notna() & actual_failure.notna()
        risk = values.loc[valid].to_numpy(dtype=float)
        failure = actual_failure.loc[valid].to_numpy(dtype=float)
        loss = np.square(risk - failure)
        losses[variant] = loss
        metrics[variant] = {
            "slates": int(len(loss)),
            "brier": float(loss.mean()) if len(loss) else None,
            "mean_counterexample_mass": float(risk.mean()) if len(risk) else None,
        }

    comparisons: dict[str, Any] = {}
    real = losses.get("real_path", np.asarray([], dtype=float))
    for control in ("endpoint_only", "shuffled_path", "inverted_path"):
        control_loss = losses.get(control, np.asarray([], dtype=float))
        if len(real) and len(control_loss) == len(real):
            improvement = control_loss - real
            comparisons[control] = {
                "mean_brier_improvement": float(improvement.mean()),
                "paired_bootstrap_one_sided_familywise_lcb": _paired_bootstrap_lcb(improvement),
                "one_sided_sign_flip_p": _one_sided_sign_flip_p(improvement),
            }
        else:
            comparisons[control] = {
                "mean_brier_improvement": None,
                "paired_bootstrap_one_sided_familywise_lcb": None,
                "one_sided_sign_flip_p": None,
            }
    path_valid = bool(
        path_certificate
        and path_certificate.get("status") == "PATH_INCREMENTAL_VALUE_SUPPORTED"
        and path_certificate.get("path_authorized", False)
    )
    lcbs = [comparisons[name]["paired_bootstrap_one_sided_familywise_lcb"] for name in comparisons]
    p_values = [comparisons[name]["one_sided_sign_flip_p"] for name in comparisons]
    enough = bool(
        lcbs
        and all(value is not None for value in lcbs)
        and all(value is not None for value in p_values)
    )
    checkpoint_alpha = 0.05 / 3.0
    if (
        path_valid
        and enough
        and all(float(value) > 0.0 for value in lcbs)
        and all(float(value) < checkpoint_alpha for value in p_values)
    ):
        status = "REAL_MLB_PATH_INCREMENTAL_VALUE_SUPPORTED"
    elif path_valid and enough:
        status = "REAL_MLB_PATH_INCREMENTAL_VALUE_NOT_SUPPORTED"
    else:
        status = MECHANISM_STATUS
    return {
        "status": status,
        "primary_leg_count": PRIMARY_LEG_COUNT,
        "primary_loss": "Brier loss of fixed-parlay retained counterexample mass vs realized failure",
        "path_confirmation_valid": path_valid,
        "variants": metrics,
        "familywise_alpha": 0.05,
        "per_control_alpha": 0.05 / 3.0,
        "minimum_paired_slates": 30,
        "paired_real_path_improvement_vs_controls": comparisons,
        "interpretation": (
            "Support requires the frozen real temporal path to improve fixed-parlay risk resolution "
            "beyond endpoint-only and endpoint-preserving destroyed-path controls."
        ),
    }


def chronological_path_conditioned_replay(
    reservoir: pd.DataFrame,
    path_features: pd.DataFrame,
    *,
    path_certificate: dict[str, Any],
    block_label: str,
    initial_history: pd.DataFrame | None = None,
    initial_path_history: pd.DataFrame | None = None,
    initial_calibration_scores: dict[str, Iterable[float]] | None = None,
    risk_target: float | None = None,
    protocol: BinaryOutcomeSetProtocol = BINARY_OUTCOME_SET_PROTOCOL,
) -> PathConditionedReplay:
    frame = _prepare_reservoir(reservoir)
    paths = _prepare_paths(path_features)
    history_parts = [frame]
    path_history_parts = [paths]
    if initial_history is not None:
        initial = _prepare_reservoir(initial_history)
        if len(initial) and initial["event_date"].max() >= frame["event_date"].min():
            raise ValueError("initial_history must end before replay block")
        history_parts.insert(0, initial)
    if initial_path_history is not None:
        initial_paths = _prepare_paths(initial_path_history)
        if len(initial_paths) and initial_paths["event_date"].max() >= paths["event_date"].min():
            raise ValueError("initial_path_history must end before replay block")
        path_history_parts.insert(0, initial_paths)
    available_history = pd.concat(history_parts, ignore_index=True, sort=False)
    available_paths = pd.concat(path_history_parts, ignore_index=True, sort=False)

    calibration: dict[str, list[float]] = {
        "prior": [],
        "endpoint_only": [],
        **{f"real::{label}": [] for label in CHECKPOINT_LABELS},
        **{f"shuffled::{label}": [] for label in CHECKPOINT_LABELS},
        **{f"inverted::{label}": [] for label in CHECKPOINT_LABELS},
    }
    if initial_calibration_scores:
        for key, values in initial_calibration_scores.items():
            if key in calibration:
                calibration[key].extend(float(value) for value in values)

    decision_rows: list[dict[str, Any]] = []
    trajectory_frames: list[pd.DataFrame] = []
    checkpoint_evidence_frames: list[pd.DataFrame] = []
    candidate_evidence_frames: list[pd.DataFrame] = []

    for event_date, slate in frame.groupby("event_date", sort=True):
        # The reservoir already carries a day-of survival_probability from
        # the existing MLB pool pipeline (see module docstring above); there
        # is no in-package recent-regime scorer to re-derive it from
        # available_history the way sports/nba/conditional_chain does.
        scored = slate.copy()
        current_paths = paths.loc[paths["event_date"].eq(pd.Timestamp(event_date))].copy()
        scored_with_paths = merge_candidates_with_paths(
            scored, current_paths, require_complete=False
        )
        if scored_with_paths.empty:
            decision_rows.append(
                {
                    "block": block_label,
                    "event_date": pd.Timestamp(event_date),
                    "evaluated": False,
                    "status": "PATH_DATA_UNAVAILABLE",
                }
            )
            continue
        scored_with_paths = scored_with_paths.sort_values(
            ["survival_probability", "robust_score", "player", "market"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).head(protocol.maximum_candidates).reset_index(drop=True)
        scored_with_paths["candidate_id"] = scored_with_paths.apply(_candidate_id, axis=1)
        if len(scored_with_paths) < min(protocol.requested_leg_counts):
            decision_rows.append(
                {
                    "block": block_label,
                    "event_date": pd.Timestamp(event_date),
                    "evaluated": False,
                    "status": "INSUFFICIENT_PATH_CANDIDATES",
                    "candidate_count": int(len(scored_with_paths)),
                }
            )
            continue
        if not bool(scored_with_paths["leg_result"].isin([0.0, 0.5, 1.0]).all()):
            decision_rows.append(
                {
                    "block": block_label,
                    "event_date": pd.Timestamp(event_date),
                    "evaluated": False,
                    "status": "UNRESOLVED_WORLD",
                }
            )
            continue

        prior = build_world_distribution(
            scored_with_paths["candidate_id"],
            scored_with_paths["survival_probability"],
            protocol=protocol,
        )
        true_world_id = world_id_from_outcomes(scored_with_paths["binary_leg_win"].astype(int))
        history_before = available_history.loc[available_history["event_date"] < pd.Timestamp(event_date)]
        paths_before = available_paths.loc[available_paths["event_date"] < pd.Timestamp(event_date)]

        bundles = {
            mode: fit_path_evidence_bundle(
                history_before,
                paths_before,
                as_of_date=event_date,
                mode=mode,
            )
            for mode in ("real", "shuffled", "inverted")
        }
        endpoint_probability = endpoint_posteriors(
            bundles["real"],
            scored_with_paths,
            scored_with_paths["survival_probability"],
        )
        endpoint_evidence = candidate_posteriors_to_world_log_evidence(prior, endpoint_probability)
        endpoint_distribution = update_world_distribution(prior, endpoint_evidence)

        path_objects = {}
        for mode in ("real", "shuffled", "inverted"):
            candidate_post = path_posteriors(
                bundles[mode],
                scored_with_paths,
                scored_with_paths["survival_probability"],
            )
            path_object = build_candidate_evidence_path(prior, candidate_post)
            path_objects[mode] = path_object
            diagnostics = path_object.world_path.diagnostics.copy()
            diagnostics["block"] = block_label
            diagnostics["event_date"] = pd.Timestamp(event_date)
            diagnostics["variant"] = mode
            checkpoint_evidence_frames.append(diagnostics)
            candidate_diagnostics = path_object.diagnostics.copy()
            candidate_diagnostics["block"] = block_label
            candidate_diagnostics["event_date"] = pd.Timestamp(event_date)
            candidate_diagnostics["variant"] = mode
            candidate_evidence_frames.append(candidate_diagnostics)

        score_map = {
            "prior": float(aps_world_scores(prior)[true_world_id]),
            "endpoint_only": float(aps_world_scores(endpoint_distribution)[true_world_id]),
        }
        for mode, path_object in path_objects.items():
            for label, distribution in zip(CHECKPOINT_LABELS, path_object.world_path.distributions[1:]):
                score_map[f"{mode}::{label}"] = float(aps_world_scores(distribution)[true_world_id])

        required_keys = ["prior", "endpoint_only", *[f"real::{label}" for label in CHECKPOINT_LABELS]]
        thresholds = {key: _threshold(calibration[key], protocol) for key in calibration}
        evaluated = all(thresholds[key] is not None for key in required_keys)
        row: dict[str, Any] = {
            "block": block_label,
            "event_date": pd.Timestamp(event_date),
            "evaluated": evaluated,
            "status": "CALIBRATION_WARMUP",
            "candidate_count": prior.candidate_count,
            "candidate_order": "||".join(prior.candidate_ids),
            "true_world_id": true_world_id,
            "evidence_training_rows": bundles["real"].training_rows,
            "evidence_history_end_exclusive": bundles["real"].history_end_exclusive,
            "path_certificate_status": path_certificate.get("status") if path_certificate else None,
            "path_certificate_authorized": bool(path_certificate and path_certificate.get("path_authorized", False)),
        }

        if evaluated:
            prior_set = build_binary_outcome_set(
                prior,
                aps_threshold=float(thresholds["prior"]),
                calibration_slates=len(calibration["prior"]),
                protocol=protocol,
            )
            endpoint_set = build_binary_outcome_set(
                endpoint_distribution,
                aps_threshold=float(thresholds["endpoint_only"]),
                calibration_slates=len(calibration["endpoint_only"]),
                protocol=protocol,
            )
            real_thresholds = [float(thresholds["prior"])] + [
                float(thresholds[f"real::{label}"]) for label in CHECKPOINT_LABELS
            ]
            real_path = path_objects["real"].world_path
            t240_set = build_binary_outcome_set(
                real_path.distributions[1],
                aps_threshold=float(thresholds[f"real::{CHECKPOINT_LABELS[0]}"]),
                calibration_slates=len(calibration[f"real::{CHECKPOINT_LABELS[0]}"]),
                protocol=protocol,
            )
            fixed_targets: dict[int, tuple[str, ...]] = {}
            if t240_set.world_count:
                for leg_count in protocol.requested_leg_counts:
                    if leg_count <= prior.candidate_count:
                        fixed_targets[leg_count] = search_parlay_proof_frontier(
                            scored_with_paths,
                            t240_set,
                            requested_leg_count=leg_count,
                            protocol=protocol,
                        ).selected_candidate_ids
            trajectory = build_proof_trajectory(
                scored_with_paths,
                real_path,
                aps_thresholds=real_thresholds,
                calibration_slates=len(calibration["prior"]),
                fixed_targets=fixed_targets,
                protocol=protocol,
            )
            trajectory_frame = trajectory.diagnostics.copy()
            trajectory_frame["block"] = block_label
            trajectory_frame["event_date"] = pd.Timestamp(event_date)
            trajectory_frames.append(trajectory_frame)

            final_real_set = build_binary_outcome_set(
                real_path.distributions[-1],
                aps_threshold=float(thresholds[f"real::{CHECKPOINT_LABELS[-1]}"]),
                calibration_slates=len(calibration[f"real::{CHECKPOINT_LABELS[-1]}"]),
                protocol=protocol,
            )
            comparison_sets: dict[str, BinaryOutcomeSet] = {"endpoint_only": endpoint_set, "real_path": final_real_set}
            for mode in ("shuffled", "inverted"):
                key = f"{mode}::{CHECKPOINT_LABELS[-1]}"
                mode_threshold = thresholds[key]
                if mode_threshold is not None:
                    comparison_sets[f"{mode}_path"] = build_binary_outcome_set(
                        path_objects[mode].world_path.distributions[-1],
                        aps_threshold=float(mode_threshold),
                        calibration_slates=len(calibration[key]),
                        protocol=protocol,
                    )
            wins_by_id = dict(
                zip(scored_with_paths["candidate_id"], scored_with_paths["binary_leg_win"].astype(bool))
            )
            row["status"] = "PATH_CONDITIONED_OUTCOME_SET_EVALUATED"
            row["prior_realized_world_covered"] = bool(true_world_id in set(prior_set.world_ids.tolist()))
            row["real_path_realized_world_covered"] = bool(true_world_id in set(final_real_set.world_ids.tolist()))
            for leg_count, target in fixed_targets.items():
                prefix = f"{leg_count}_leg"
                row[f"fixed_{prefix}_candidate_ids"] = "|".join(target)
                row[f"fixed_{prefix}_hit"] = bool(all(wins_by_id[value] for value in target))
                for variant, outcome_set in comparison_sets.items():
                    count, mass, logical = _fixed_metrics(outcome_set, target)
                    row[f"{variant}_{prefix}_counterexample_worlds"] = count
                    row[f"{variant}_{prefix}_counterexample_mass"] = mass
                    row[f"{variant}_{prefix}_logical_certificate"] = logical
                certificate = certify_perfect_parlay(
                    scored_with_paths,
                    final_real_set,
                    requested_leg_count=leg_count,
                    path_certificate=path_certificate,
                    protocol=protocol,
                )
                row[f"real_path_{prefix}_certificate_status"] = certificate.status
                row[f"real_path_{prefix}_path_certificate_valid"] = certificate.path_certificate_valid
        decision_rows.append(row)
        for key, score in score_map.items():
            calibration[key].append(score)

    decisions = pd.DataFrame(decision_rows)
    if len(decisions) and "evaluated" in decisions.columns:
        evaluated_frame = decisions.loc[decisions["evaluated"].fillna(False).astype(bool)].copy()
    else:
        evaluated_frame = decisions.iloc[0:0].copy()
    ablation = _ablation_report(evaluated_frame, path_certificate)
    selective = _selective_risk_report(evaluated_frame, risk_target=risk_target)
    report = {
        "status": ablation["status"],
        "block": block_label,
        "evaluated_slates": int(len(evaluated_frame)),
        "total_slates": int(len(decisions)),
        "candidate_evidence_contract": "candidate-conditioned likelihood evidence applied to exact joint binary worlds",
        "path_checkpoints": list(CHECKPOINT_LABELS),
        "target_freeze_checkpoint": CHECKPOINT_LABELS[0],
        "path_certificate": path_certificate,
        "production_authorized": False,
        "selective_risk_status": selective["status"],
        "ablation_status": ablation["status"],
    }
    return PathConditionedReplay(
        decisions=decisions,
        proof_trajectories=(
            pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()
        ),
        checkpoint_evidence=(
            pd.concat(checkpoint_evidence_frames, ignore_index=True) if checkpoint_evidence_frames else pd.DataFrame()
        ),
        candidate_evidence=(
            pd.concat(candidate_evidence_frames, ignore_index=True) if candidate_evidence_frames else pd.DataFrame()
        ),
        report=report,
        selective_risk_report=selective,
        ablation_report=ablation,
        calibration_scores={key: tuple(float(value) for value in values) for key, values in calibration.items()},
    )
