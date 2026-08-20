from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.stats import beta as beta_distribution


CORE_POLICY_FAMILY = "NBA_RANK_RELIABILITY_CORE_V1"


@dataclass(frozen=True)
class FrozenCorePolicy:
    version: str
    source_policy_version: str
    source_model_version: str
    ranks: tuple[int, ...]
    training_cutoff: pd.Timestamp
    resolved_decisions_per_rank: dict[int, int]
    rank_hit_rate: dict[int, float]
    rank_lcb: dict[int, float]
    status: str
    production_authorized: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_policy_version": self.source_policy_version,
            "source_model_version": self.source_model_version,
            "ranks": list(self.ranks),
            "training_cutoff": str(self.training_cutoff),
            "resolved_decisions_per_rank": self.resolved_decisions_per_rank,
            "rank_hit_rate": self.rank_hit_rate,
            "rank_lcb": self.rank_lcb,
            "status": self.status,
            "production_authorized": self.production_authorized,
        }


def fit_rank_reliability_core(
    resolved_reservoir: pd.DataFrame,
    *,
    source_policy_version: str,
    source_model_version: str,
    training_cutoff: str | pd.Timestamp,
    candidate_ranks: tuple[int, ...] = (1, 2, 3, 4),
    core_legs: int = 2,
    minimum_resolved_decisions: int = 20,
    lower_quantile: float = 0.10,
) -> FrozenCorePolicy:
    """Freeze a small core using only prior same-version rank reliability."""

    required = {"event_date", "rank", "leg_result", "selector_version"}
    missing = sorted(required - set(resolved_reservoir.columns))
    if missing:
        raise ValueError(f"core-policy training rows are missing columns: {missing}")
    versions = set(resolved_reservoir["selector_version"].dropna().astype(str))
    if versions != {source_policy_version}:
        raise ValueError(
            f"core policy may use one exact source policy version; observed {sorted(versions)}"
        )
    cutoff = pd.Timestamp(training_cutoff).normalize()
    training = resolved_reservoir.copy()
    training["event_date"] = pd.to_datetime(training["event_date"]).dt.normalize()
    training = training.loc[
        training["event_date"].lt(cutoff)
        & training["rank"].isin(candidate_ranks)
        & training["leg_result"].isin([0.0, 1.0])
    ]

    counts: dict[int, int] = {}
    rates: dict[int, float] = {}
    lcbs: dict[int, float] = {}
    for rank in candidate_ranks:
        outcomes = training.loc[training["rank"].eq(rank), "leg_result"].astype(float)
        count = int(len(outcomes))
        wins = int(outcomes.sum())
        counts[rank] = count
        rates[rank] = float(wins / count) if count else 0.0
        lcbs[rank] = (
            float(beta_distribution.ppf(lower_quantile, 0.5 + wins, 0.5 + count - wins))
            if count
            else 0.0
        )

    enough_history = all(
        counts[rank] >= minimum_resolved_decisions for rank in candidate_ranks
    )
    selected = tuple(
        sorted(
            candidate_ranks,
            key=lambda rank: (lcbs[rank], rates[rank], -rank),
            reverse=True,
        )[:core_legs]
    )
    selected = tuple(sorted(selected))
    status = "FROZEN_SHADOW" if enough_history else "INSUFFICIENT_SAME_VERSION_HISTORY"
    version = (
        f"{CORE_POLICY_FAMILY}::{source_policy_version}::{source_model_version}::"
        f"{cutoff.strftime('%Y%m%d')}::R{'_'.join(str(rank) for rank in selected)}"
    )
    return FrozenCorePolicy(
        version=version,
        source_policy_version=source_policy_version,
        source_model_version=source_model_version,
        ranks=selected,
        training_cutoff=cutoff,
        resolved_decisions_per_rank=counts,
        rank_hit_rate=rates,
        rank_lcb=lcbs,
        status=status,
    )


def select_frozen_core(
    reservoir: pd.DataFrame,
    policy: FrozenCorePolicy,
) -> pd.DataFrame:
    if policy.status != "FROZEN_SHADOW":
        return reservoir.iloc[0:0].copy()
    required = {"rank", "selector_version", "model_version", "player"}
    missing = sorted(required - set(reservoir.columns))
    if missing:
        raise ValueError(f"core-policy candidate rows are missing columns: {missing}")
    versions = set(reservoir["selector_version"].dropna().astype(str))
    if versions != {policy.source_policy_version}:
        raise ValueError("core-policy source version does not match current candidates")
    model_versions = set(reservoir["model_version"].dropna().astype(str))
    if model_versions != {policy.source_model_version}:
        raise ValueError("core-policy model version does not match current candidates")
    selected = reservoir.loc[reservoir["rank"].isin(policy.ranks)].copy()
    if len(selected) != len(policy.ranks) or selected["player"].nunique() != len(
        selected
    ):
        return reservoir.iloc[0:0].copy()
    selected["core_policy_version"] = policy.version
    selected["core_policy_status"] = "SHADOW_ONLY"
    selected["publication_authorized"] = False
    return selected.sort_values("rank", kind="mergesort").reset_index(drop=True)
