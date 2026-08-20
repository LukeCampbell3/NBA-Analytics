from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import binomtest

from .backtest import adapt_validation_pool_ledger
from .frozen_selector import select_frozen_board
from .protocol import (
    FROZEN_SELECTOR_PROTOCOL,
    SURVIVAL_BUILDER_PROTOCOL,
    FrozenSelectorProtocol,
    SurvivalBuilderProtocol,
)
from .survival_builder import build_survival_parlays


EVIDENCE_STATUS = "REPEATEDLY_INSPECTED_SYNTHETIC_RESEARCH_EVIDENCE"


@dataclass(frozen=True)
class SurvivalReplay:
    decisions: pd.DataFrame
    selected_legs: pd.DataFrame
    report: dict[str, Any]


def _leg_result(actual: float, line: float, side: str) -> float:
    if np.isclose(actual, line):
        return 0.5
    return float(actual > line) if side == "OVER" else float(actual < line)


def build_transfer_reservoir(
    validation_pool: pd.DataFrame,
    historical_actuals: pd.DataFrame,
    *,
    selector_protocol: FrozenSelectorProtocol = FROZEN_SELECTOR_PROTOCOL,
) -> pd.DataFrame:
    """Reconstruct the complete published top-K pool for a transfer window."""

    adapted = adapt_validation_pool_ledger(validation_pool)
    adapted = adapted.dropna(subset=["market", "line", "actual", "p_over"])
    rows: list[pd.DataFrame] = []
    for event_date in sorted(adapted["event_date"].unique()):
        slate = adapted.loc[adapted["event_date"].eq(event_date)]
        selection = select_frozen_board(
            slate,
            historical_actuals,
            protocol=selector_protocol,
        )
        if not selection.published:
            continue
        reservoir = selection.reservoir.copy().reset_index(drop=True)
        reservoir["event_date"] = pd.Timestamp(event_date).normalize()
        reservoir["rank"] = np.arange(1, len(reservoir) + 1)
        reservoir["leg_result"] = [
            _leg_result(float(row["actual"]), float(row["line"]), str(row["side"]))
            for _, row in reservoir.iterrows()
        ]
        reservoir["market_evidence"] = "SYNTHETIC_THRESHOLD_HISTORY"
        reservoir["production_authorizable"] = False
        rows.append(reservoir)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _exact_interval(wins: int, trials: int, alpha: float = 0.05) -> list[float] | None:
    if trials <= 0:
        return None
    lower = (
        0.0 if wins == 0 else beta_distribution.ppf(alpha / 2, wins, trials - wins + 1)
    )
    upper = (
        1.0
        if wins == trials
        else beta_distribution.ppf(1 - alpha / 2, wins + 1, trials - wins)
    )
    return [float(lower), float(upper)]


def _decision_metrics(decisions: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for leg_count, group in decisions.groupby("leg_count", sort=True):
        trials = int(len(group))
        policy_wins = int(group["policy_hit"].sum())
        control_wins = int(group["control_hit"].sum())
        policy_only = int((group["policy_hit"] & ~group["control_hit"]).sum())
        control_only = int((group["control_hit"] & ~group["policy_hit"]).sum())
        discordant = policy_only + control_only
        action = group.loc[group["policy_action"]]
        action_trials = int(len(action))
        action_policy_wins = int(action["policy_hit"].sum())
        action_control_wins = int(action["control_hit"].sum())
        action_policy_only = int((action["policy_hit"] & ~action["control_hit"]).sum())
        action_control_only = int((action["control_hit"] & ~action["policy_hit"]).sum())
        action_discordant = action_policy_only + action_control_only
        legacy = group["legacy_fixed_rank_core_hit"].dropna().astype(bool)
        metrics[str(int(leg_count))] = {
            "action_slates": trials,
            "policy_wins": policy_wins,
            "policy_hit_rate": float(policy_wins / trials) if trials else None,
            "policy_exact_95_ci": _exact_interval(policy_wins, trials),
            "control_wins": control_wins,
            "control_hit_rate": float(control_wins / trials) if trials else None,
            "absolute_hit_rate_change": (
                float((policy_wins - control_wins) / trials) if trials else None
            ),
            "policy_only_wins": policy_only,
            "control_only_wins": control_only,
            "paired_one_sided_p": (
                float(
                    binomtest(
                        policy_only,
                        discordant,
                        p=0.5,
                        alternative="greater",
                    ).pvalue
                )
                if discordant
                else 1.0
            ),
            "mean_independence_reference": float(
                group["policy_independence_reference"].mean()
            ),
            "mean_frechet_lower_reference": float(
                group["policy_frechet_lower_reference"].mean()
            ),
            "legacy_fixed_rank_core": {
                "action_slates": int(len(legacy)),
                "wins": int(legacy.sum()),
                "hit_rate": float(legacy.mean()) if len(legacy) else None,
                "status": "REJECTED_MODEL_VERSION_COUPLED_BASELINE",
            },
            "selective_policy": {
                "action_slates": action_trials,
                "abstained_slates": trials - action_trials,
                "slate_coverage": float(action_trials / trials) if trials else 0.0,
                "policy_wins": action_policy_wins,
                "policy_hit_rate": (
                    float(action_policy_wins / action_trials) if action_trials else None
                ),
                "policy_exact_95_ci": _exact_interval(
                    action_policy_wins, action_trials
                ),
                "control_wins_on_action_slates": action_control_wins,
                "control_hit_rate_on_action_slates": (
                    float(action_control_wins / action_trials)
                    if action_trials
                    else None
                ),
                "policy_only_wins": action_policy_only,
                "control_only_wins": action_control_only,
                "paired_one_sided_p": (
                    float(
                        binomtest(
                            action_policy_only,
                            action_discordant,
                            p=0.5,
                            alternative="greater",
                        ).pvalue
                    )
                    if action_discordant
                    else 1.0
                ),
            },
        }
    return metrics


def chronological_survival_replay(
    reservoir: pd.DataFrame,
    *,
    block_label: str,
    initial_history: pd.DataFrame | None = None,
    warmup_slates: int | None = None,
    protocol: SurvivalBuilderProtocol = SURVIVAL_BUILDER_PROTOCOL,
) -> SurvivalReplay:
    """Replay one policy decision per date with no same-day outcome access."""

    required = {
        "event_date",
        "player",
        "market",
        "side",
        "robust_score",
        "rank",
        "leg_result",
    }
    missing = sorted(required - set(reservoir.columns))
    if missing:
        raise ValueError(f"survival replay reservoir is missing columns: {missing}")
    frame = reservoir.copy()
    frame["event_date"] = pd.to_datetime(
        frame["event_date"], errors="raise"
    ).dt.normalize()
    dates = sorted(frame["event_date"].unique())
    if warmup_slates is None:
        warmup_slates = (
            0 if initial_history is not None else protocol.minimum_warmup_slates
        )
    if warmup_slates < 0 or warmup_slates >= len(dates):
        raise ValueError("warmup_slates must leave at least one evaluation slate")

    history_parts = [frame]
    if initial_history is not None:
        initial = initial_history.copy()
        initial["event_date"] = pd.to_datetime(
            initial["event_date"], errors="raise"
        ).dt.normalize()
        if len(initial) and initial["event_date"].max() >= min(dates):
            raise ValueError("initial_history must end before the evaluation reservoir")
        history_parts.insert(0, initial)
    available_history = pd.concat(history_parts, ignore_index=True, sort=False)

    decision_rows: list[dict[str, Any]] = []
    leg_rows: list[pd.DataFrame] = []
    for event_date in dates[warmup_slates:]:
        slate = frame.loc[frame["event_date"].eq(event_date)].copy()
        build = build_survival_parlays(
            slate,
            available_history,
            as_of_date=event_date,
            protocol=protocol,
        )
        for leg_count, policy_legs in build.alternatives.items():
            control = slate.sort_values("rank", kind="mergesort").head(leg_count)
            legacy_core = slate.loc[slate["rank"].isin([1, 4])]
            legacy_core_hit = (
                bool(legacy_core["leg_result"].eq(1.0).all())
                if leg_count == 2 and len(legacy_core) == 2
                else None
            )
            if len(control) != leg_count or len(policy_legs) != leg_count:
                continue
            selection_metrics = build.diagnostics["selection_metrics"][leg_count]
            policy_hit = bool(policy_legs["leg_result"].eq(1.0).all())
            control_hit = bool(control["leg_result"].eq(1.0).all())
            decision_rows.append(
                {
                    "block": block_label,
                    "event_date": pd.Timestamp(event_date),
                    "leg_count": int(leg_count),
                    "policy_hit": policy_hit,
                    "control_hit": control_hit,
                    "legacy_fixed_rank_core_hit": legacy_core_hit,
                    "policy_action": bool(
                        leg_count == protocol.primary_leg_count
                        and build.diagnostics["selective_action"]
                    ),
                    "policy_players": "|".join(policy_legs["player"].astype(str)),
                    "control_players": "|".join(control["player"].astype(str)),
                    "policy_independence_reference": selection_metrics[
                        "independence_reference"
                    ],
                    "policy_frechet_lower_reference": selection_metrics[
                        "frechet_lower_reference"
                    ],
                    "minimum_policy_marginal": selection_metrics[
                        "minimum_marginal_probability"
                    ],
                    "history_rows_in_window": build.diagnostics[
                        "history_rows_in_window"
                    ],
                    "survival_policy_version": protocol.version,
                }
            )
            selected = policy_legs.copy()
            selected["block"] = block_label
            selected["decision_date"] = pd.Timestamp(event_date)
            selected["policy_hit"] = policy_hit
            leg_rows.append(selected)

    decisions = pd.DataFrame(decision_rows)
    selected_legs = (
        pd.concat(leg_rows, ignore_index=True) if leg_rows else pd.DataFrame()
    )
    report = {
        "block": block_label,
        "survival_policy_version": protocol.version,
        "evidence_status": EVIDENCE_STATUS,
        "production_authorizable": False,
        "warmup_slates": int(warmup_slates),
        "evaluation_slates": (
            int(decisions["event_date"].nunique()) if len(decisions) else 0
        ),
        "metrics_by_leg_count": _decision_metrics(decisions) if len(decisions) else {},
    }
    return SurvivalReplay(
        decisions=decisions, selected_legs=selected_legs, report=report
    )


def combine_survival_replays(replays: Iterable[SurvivalReplay]) -> SurvivalReplay:
    replay_list = list(replays)
    decisions = pd.concat([item.decisions for item in replay_list], ignore_index=True)
    selected = pd.concat(
        [item.selected_legs for item in replay_list], ignore_index=True
    )
    combined_metrics = _decision_metrics(decisions)
    primary_metrics = combined_metrics[str(SURVIVAL_BUILDER_PROTOCOL.primary_leg_count)]
    selective = primary_metrics["selective_policy"]
    research_checks = {
        "minimum_action_slates": bool(
            selective["action_slates"]
            >= SURVIVAL_BUILDER_PROTOCOL.minimum_research_action_slates
        ),
        "minimum_slate_coverage": bool(
            selective["slate_coverage"]
            >= SURVIVAL_BUILDER_PROTOCOL.minimum_research_slate_coverage
        ),
        "positive_paired_direction": bool(
            selective["policy_only_wins"] > selective["control_only_wins"]
        ),
        "one_sided_significance": bool(
            selective["paired_one_sided_p"] < SURVIVAL_BUILDER_PROTOCOL.one_sided_alpha
        ),
    }
    research_gate_status = (
        "RESEARCH_GATE_PASSED_NOT_PROSPECTIVE"
        if all(research_checks.values())
        else "RESEARCH_GATE_NOT_PASSED"
    )
    report = {
        "survival_policy_version": SURVIVAL_BUILDER_PROTOCOL.version,
        "evidence_status": EVIDENCE_STATUS,
        "production_authorizable": False,
        "primary_leg_count": SURVIVAL_BUILDER_PROTOCOL.primary_leg_count,
        "allowed_leg_counts": list(SURVIVAL_BUILDER_PROTOCOL.allowed_leg_counts),
        "four_leg_status": "REJECTED_NO_CROSS_VERSION_IMPROVEMENT",
        "selective_frechet_floor": (SURVIVAL_BUILDER_PROTOCOL.selective_frechet_floor),
        "minimum_research_slate_coverage": (
            SURVIVAL_BUILDER_PROTOCOL.minimum_research_slate_coverage
        ),
        "minimum_research_action_slates": (
            SURVIVAL_BUILDER_PROTOCOL.minimum_research_action_slates
        ),
        "one_sided_alpha": SURVIVAL_BUILDER_PROTOCOL.one_sided_alpha,
        "blocks": [item.report for item in replay_list],
        "combined_metrics_by_leg_count": combined_metrics,
        "research_gate": {
            "status": research_gate_status,
            "checks": research_checks,
        },
        "interpretation": (
            "The policy is a cross-version research improvement, not a prospective return "
            "certificate. Primary publication remains disabled."
        ),
    }
    return SurvivalReplay(decisions=decisions, selected_legs=selected, report=report)
