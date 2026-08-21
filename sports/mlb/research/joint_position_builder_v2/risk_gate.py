from __future__ import annotations

"""Selective empirical-risk gate and final JOINT POSITION action gating.

Statistical method mirrors sports/mlb/conditional_chain/path_conditioned_backtest.py
's _selective_risk_report (Clopper-Pearson exact one-sided UCB, a
development/validation temporal split, a threshold sweep on development
only) -- reimplemented standalone here rather than imported, because that
function's column-naming is hardcoded to its own module's convention; the
underlying method is identical on purpose.

Final gating happens at the JOINT POSITION level, never at the leg level
(per the task spec): a leg with EV_i<0 is never excluded on its own, only a
whole pair is accepted or the day is abstained.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

MINIMUM_DEVELOPMENT_ACTIONS = 10


def clopper_pearson_failure_ucb(failures: int, actions: int, alpha: float = 0.05) -> float | None:
    if actions <= 0:
        return None
    if failures >= actions:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, actions - failures))


@dataclass(frozen=True)
class SelectiveRiskCertificate:
    status: str
    risk_target: float | None
    frozen_threshold: float | None
    development_actions: int
    validation_actions: int
    validation_failures: int
    validation_failure_rate: float | None
    validation_failure_ucb: float | None


def build_selective_risk_certificate(
    pairs_df: pd.DataFrame, *, risk_target: float, minimum_development_actions: int = MINIMUM_DEVELOPMENT_ACTIONS
) -> SelectiveRiskCertificate:
    """pairs_df: one row per evaluated pair-day, chronologically ordered,
    with `counterexample_mass` and `both_win` columns. Splits in half by
    date (development / validation, no shuffling), sweeps candidate
    counterexample-mass thresholds on development only, freezes whichever
    threshold both meets `minimum_development_actions` and clears
    risk_target on its Clopper-Pearson UCB, then reports that frozen
    threshold's real performance on validation (touched only once)."""
    if pairs_df.empty:
        return SelectiveRiskCertificate("INSUFFICIENT_EVALUATED_PAIRS", risk_target, None, 0, 0, 0, None, None)
    ordered = pairs_df.sort_values("date", kind="mergesort").reset_index(drop=True)
    split = max(len(ordered) // 2, 1)
    development, validation = ordered.iloc[:split], ordered.iloc[split:]

    candidate_thresholds = sorted(set(float(v) for v in development["counterexample_mass"]))
    eligible = []
    for threshold in candidate_thresholds:
        actions = development[development["counterexample_mass"] <= threshold]
        if len(actions) < minimum_development_actions:
            continue
        failures = int((~actions["both_win"].astype(bool)).sum())
        ucb = clopper_pearson_failure_ucb(failures, len(actions))
        if ucb is not None and ucb <= risk_target:
            eligible.append((threshold, len(actions)))
    if not eligible:
        return SelectiveRiskCertificate(
            "NO_DEVELOPMENT_THRESHOLD_MEETS_RISK_BOUND", risk_target, None, len(development), len(validation), 0, None, None
        )
    frozen_threshold = max(eligible, key=lambda pair: (pair[1], -pair[0]))[0]
    validation_actions = validation[validation["counterexample_mass"] <= frozen_threshold]
    failures = int((~validation_actions["both_win"].astype(bool)).sum())
    ucb = clopper_pearson_failure_ucb(failures, len(validation_actions))
    status = (
        "SELECTIVE_RISK_BOUND_SUPPORTED_ON_VALIDATION"
        if ucb is not None and ucb <= risk_target
        else "SELECTIVE_RISK_BOUND_NOT_SUPPORTED_ON_VALIDATION"
    )
    return SelectiveRiskCertificate(
        status,
        risk_target,
        frozen_threshold,
        len(development),
        len(validation_actions),
        failures,
        float(failures / len(validation_actions)) if len(validation_actions) else None,
        ucb,
    )


@dataclass(frozen=True)
class ActionDecision:
    date: str
    action: str  # "ACT" or "ABSTAIN"
    selected_pair_index: int | None
    reason: str


def gate_and_rank_day(
    day_pairs: list[Any],  # list[pairs.CandidatePair]
    *,
    joint_ev_lcb_margin: float,
    min_support_history_rows: float,
    risk_certificate: SelectiveRiskCertificate | None,
) -> ActionDecision:
    """Prototype rule: joint_EV_LCB > margin AND adequate support AND
    in-support AND selective empirical-risk gate passes. One pair per day
    maximum; otherwise abstain. Ranking: lower failure-risk UCB, lower
    counterexample mass, higher joint_EV_LCB, stronger support."""
    date = day_pairs[0].date if day_pairs else ""
    if risk_certificate is None or risk_certificate.status not in (
        "SELECTIVE_RISK_BOUND_SUPPORTED_ON_VALIDATION",
    ):
        return ActionDecision(date, "ABSTAIN", None, "no supported selective-risk certificate yet (SHADOW_ONLY)")

    eligible_indices = [
        idx
        for idx, pair in enumerate(day_pairs)
        if pair.joint_ev_lcb is not None
        and pair.joint_ev_lcb > joint_ev_lcb_margin
        and pair.support_min_history_rows >= min_support_history_rows
        and pair.certificate.counterexample_mass <= (risk_certificate.frozen_threshold or 0.0)
    ]
    if not eligible_indices:
        return ActionDecision(date, "ABSTAIN", None, "no pair cleared joint_EV_LCB/support/risk thresholds")

    def rank_key(idx: int):
        pair = day_pairs[idx]
        # No per-pair failure-risk UCB is fit (that would need its own dev/
        # validation split per pair, infeasible at this sample size). The
        # frozen certificate's threshold sweep already assumes
        # counterexample_mass monotonically predicts empirical risk (lower
        # mass -> tighter threshold -> lower validated failure UCB), so mass
        # stands in as the ranking proxy for "lower empirical failure-risk
        # UCB" -- the spec's first two ranking criteria collapse to one
        # ordering key here, documented rather than faked as two.
        return (
            pair.certificate.counterexample_mass,
            -pair.joint_ev_lcb,
            -pair.support_min_history_rows,
        )

    best = min(eligible_indices, key=rank_key)
    return ActionDecision(date, "ACT", best, "cleared joint_EV_LCB, support, in-support, and risk gate")
