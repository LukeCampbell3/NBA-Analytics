from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta as beta_distribution

from .protocol import SURVIVAL_BUILDER_PROTOCOL, SurvivalBuilderProtocol


REQUIRED_CANDIDATE_COLUMNS = {
    "event_date",
    "player",
    "market",
    "side",
    "robust_score",
}
REQUIRED_HISTORY_COLUMNS = {
    "event_date",
    "market",
    "side",
    "leg_result",
}


@dataclass(frozen=True)
class SurvivalBuild:
    scored_reservoir: pd.DataFrame
    primary_parlay: pd.DataFrame
    alternatives: dict[int, pd.DataFrame]
    status: str
    publication_authorized: bool
    diagnostics: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _posterior_mean(
    wins: float,
    trials: float,
    protocol: SurvivalBuilderProtocol,
) -> float:
    return float(
        (wins + protocol.jeffreys_alpha)
        / (trials + protocol.jeffreys_alpha + protocol.jeffreys_beta)
    )


def _posterior_lcb(
    wins: float,
    trials: float,
    protocol: SurvivalBuilderProtocol,
) -> float:
    return float(
        beta_distribution.ppf(
            protocol.credible_lower_quantile,
            wins + protocol.jeffreys_alpha,
            trials - wins + protocol.jeffreys_beta,
        )
    )


def _clip_probability(value: float, protocol: SurvivalBuilderProtocol) -> float:
    return float(np.clip(value, protocol.score_epsilon, 1.0 - protocol.score_epsilon))


def score_recent_regime_candidates(
    reservoir: pd.DataFrame,
    resolved_history: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    protocol: SurvivalBuilderProtocol = SURVIVAL_BUILDER_PROTOCOL,
) -> pd.DataFrame:
    """Apply a date-safe empirical-Bayes correction for the current prop regime."""

    _require_columns(reservoir, REQUIRED_CANDIDATE_COLUMNS, "reservoir")
    _require_columns(resolved_history, REQUIRED_HISTORY_COLUMNS, "resolved_history")
    scored = reservoir.copy()
    scored["event_date"] = pd.to_datetime(
        scored["event_date"], errors="raise"
    ).dt.normalize()
    if as_of_date is None:
        dates = scored["event_date"].drop_duplicates()
        if len(dates) != 1:
            raise ValueError(
                "reservoir must contain one event date when as_of_date is omitted"
            )
        as_of = pd.Timestamp(dates.iloc[0]).normalize()
    else:
        as_of = pd.Timestamp(as_of_date).normalize()
    if bool(scored["event_date"].ne(as_of).any()):
        raise ValueError("all reservoir rows must match as_of_date")

    history = resolved_history.copy()
    history["event_date"] = pd.to_datetime(
        history["event_date"], errors="raise"
    ).dt.normalize()
    history["side"] = history["side"].astype(str).str.upper()
    history["leg_result"] = pd.to_numeric(history["leg_result"], errors="coerce")
    window_start = as_of - pd.Timedelta(days=protocol.lookback_days)
    history = history.loc[
        history["event_date"].ge(window_start)
        & history["event_date"].lt(as_of)
        & history["leg_result"].isin([0.0, 1.0])
    ].copy()

    overall_trials = float(len(history))
    overall_wins = float(history["leg_result"].sum())
    overall_mean = _posterior_mean(overall_wins, overall_trials, protocol)
    overall_lcb = _posterior_lcb(overall_wins, overall_trials, protocol)

    category_stats: dict[tuple[str, str], dict[str, float]] = {}
    for (market, side), group in history.groupby(["market", "side"], sort=False):
        trials = float(len(group))
        wins = float(group["leg_result"].sum())
        category_stats[(str(market), str(side))] = {
            "trials": trials,
            "mean": _posterior_mean(wins, trials, protocol),
            "lcb": _posterior_lcb(wins, trials, protocol),
            "reliability": float(trials / (trials + protocol.category_prior_strength)),
        }

    adjusted_probabilities: list[float] = []
    lower_bounds: list[float] = []
    category_trials: list[int] = []
    category_means: list[float] = []
    category_lcbs: list[float] = []
    reliabilities: list[float] = []
    adjustments: list[float] = []
    for _, row in scored.iterrows():
        side = str(row["side"]).upper()
        key = (str(row["market"]), side)
        stats = category_stats.get(
            key,
            {
                "trials": 0.0,
                "mean": overall_mean,
                "lcb": overall_lcb,
                "reliability": 0.0,
            },
        )
        base = _clip_probability(float(row["robust_score"]), protocol)
        reliability = float(stats["reliability"])
        log_odds_adjustment = reliability * (
            logit(_clip_probability(float(stats["mean"]), protocol))
            - logit(_clip_probability(overall_mean, protocol))
        )
        adjusted = _clip_probability(
            float(expit(logit(base) + log_odds_adjustment)), protocol
        )
        lower_adjustment = reliability * (
            logit(_clip_probability(float(stats["lcb"]), protocol))
            - logit(_clip_probability(overall_mean, protocol))
        )
        empirical_lower = _clip_probability(
            float(expit(logit(base) + lower_adjustment)), protocol
        )
        adjusted_probabilities.append(adjusted)
        lower_bounds.append(min(base, empirical_lower))
        category_trials.append(int(stats["trials"]))
        category_means.append(float(stats["mean"]))
        category_lcbs.append(float(stats["lcb"]))
        reliabilities.append(reliability)
        adjustments.append(float(log_odds_adjustment))

    scored["side"] = scored["side"].astype(str).str.upper()
    scored["regime_history_start"] = window_start
    scored["regime_history_end_exclusive"] = as_of
    scored["regime_overall_trials"] = int(overall_trials)
    scored["regime_overall_mean"] = overall_mean
    scored["regime_overall_lcb"] = overall_lcb
    scored["regime_category_trials"] = category_trials
    scored["regime_category_mean"] = category_means
    scored["regime_category_lcb"] = category_lcbs
    scored["regime_category_reliability"] = reliabilities
    scored["regime_log_odds_adjustment"] = adjustments
    scored["survival_probability"] = adjusted_probabilities
    scored["survival_marginal_lcb"] = lower_bounds
    scored["survival_policy_version"] = protocol.version
    return scored


def _combination_metrics(legs: pd.DataFrame) -> dict[str, float]:
    probabilities = legs["survival_probability"].to_numpy(dtype=float)
    lower_bounds = legs["survival_marginal_lcb"].to_numpy(dtype=float)
    return {
        "independence_reference": float(np.prod(probabilities)),
        "frechet_lower_reference": float(
            max(0.0, lower_bounds.sum() - (len(lower_bounds) - 1.0))
        ),
        "minimum_marginal_probability": float(probabilities.min()),
        "minimum_marginal_lcb": float(lower_bounds.min()),
        "expected_failures": float(np.sum(1.0 - probabilities)),
    }


def _best_combination(
    scored: pd.DataFrame, leg_count: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    best: tuple[tuple[Any, ...], pd.DataFrame, dict[str, float]] | None = None
    for indices in combinations(scored.index, leg_count):
        legs = scored.loc[list(indices)].copy()
        if legs["player"].astype(str).nunique() != leg_count:
            continue
        metrics = _combination_metrics(legs)
        players = tuple(sorted(legs["player"].astype(str)))
        key = (
            metrics["independence_reference"],
            metrics["frechet_lower_reference"],
            metrics["minimum_marginal_probability"],
            float(legs["robust_score"].sum()),
            tuple(reversed(players)),
        )
        if best is None or key > best[0]:
            best = (key, legs, metrics)
    if best is None:
        return scored.iloc[0:0].copy(), {}
    selected = (
        best[1]
        .sort_values(
            ["survival_probability", "robust_score", "player"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return selected, best[2]


def build_survival_parlays(
    reservoir: pd.DataFrame,
    resolved_history: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    protocol: SurvivalBuilderProtocol = SURVIVAL_BUILDER_PROTOCOL,
) -> SurvivalBuild:
    """Build a survival-first pair and a separately measured three-leg extension."""

    scored = score_recent_regime_candidates(
        reservoir,
        resolved_history,
        as_of_date=as_of_date,
        protocol=protocol,
    )
    alternatives: dict[int, pd.DataFrame] = {}
    metrics_by_count: dict[int, dict[str, float]] = {}
    for leg_count in protocol.allowed_leg_counts:
        selected, metrics = _best_combination(scored, leg_count)
        if len(selected) != leg_count:
            continue
        selected["survival_leg_count"] = leg_count
        selected["publication_authorized"] = False
        alternatives[leg_count] = selected
        metrics_by_count[leg_count] = metrics

    primary = alternatives.get(protocol.primary_leg_count, scored.iloc[0:0].copy())
    if len(primary) != protocol.primary_leg_count:
        status = "INSUFFICIENT_SURVIVAL_LEGS"
        selective_action = False
    else:
        primary_metrics = metrics_by_count[protocol.primary_leg_count]
        selective_action = bool(
            primary_metrics["frechet_lower_reference"]
            >= protocol.selective_frechet_floor
        )
        status = (
            "SURVIVAL_SELECTIVE_ACTION_SHADOW"
            if selective_action
            else "SURVIVAL_ABSTAIN_SHADOW"
        )
    return SurvivalBuild(
        scored_reservoir=scored,
        primary_parlay=primary,
        alternatives=alternatives,
        status=status,
        publication_authorized=False,
        diagnostics={
            "survival_policy_version": protocol.version,
            "recommended_leg_count": protocol.primary_leg_count,
            "allowed_leg_counts": list(protocol.allowed_leg_counts),
            "selective_action": selective_action,
            "selective_frechet_floor": protocol.selective_frechet_floor,
            "four_leg_status": "REJECTED_NO_CROSS_VERSION_IMPROVEMENT",
            "history_rows_in_window": (
                int(scored["regime_overall_trials"].iloc[0]) if len(scored) else 0
            ),
            "selection_metrics": metrics_by_count,
            "publication_mode": protocol.publication_mode,
        },
    )
