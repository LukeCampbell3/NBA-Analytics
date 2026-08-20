from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .conditional_extension import ConditionalExtensionModel, extension_feature_row
from .protocol import FROZEN_SELECTOR_PROTOCOL, FrozenSelectorProtocol


CHAIN_POLICY_VERSION = "NBA_CONDITIONAL_CHAIN_V0_SHADOW"
PATH_REQUIRED_STATUS = "PATH_INCREMENTAL_VALUE_SUPPORTED"


@dataclass(frozen=True)
class ChainResolution:
    control_parlay: pd.DataFrame
    shadow_chain: pd.DataFrame
    status: str
    path_used: bool
    publication_authorized: bool
    diagnostics: dict[str, Any]


def _candidate_path_score(row: pd.Series) -> tuple[float, float, float]:
    side_sign = 1.0 if str(row["side"]).upper() == "OVER" else -1.0
    direction_support = float(np.tanh(side_sign * float(row["delta_share"]) / 0.01))
    efficiency = float(np.clip(row["player_path_efficiency"], 0.0, 1.0))
    reversal_penalty = min(float(row["direction_reversals"]) / 3.0, 1.0)
    path_support = direction_support * (0.5 + 0.5 * efficiency) - 0.15 * reversal_penalty
    return direction_support, path_support, reversal_penalty


def _score_chain(
    chain: pd.DataFrame,
    extension_model: ConditionalExtensionModel | None = None,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    diagnostics = chain.apply(_candidate_path_score, axis=1, result_type="expand")
    diagnostics.columns = ["direction_support", "path_support", "reversal_penalty"]
    chain = chain.copy()
    chain[diagnostics.columns] = diagnostics
    chain = chain.sort_values(
        ["robust_score", "selected_probability"], ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    marginal_log_score = float(
        np.log(np.clip(chain["robust_score"].astype(float), 1e-6, 1.0)).mean()
    )
    mean_support = float(diagnostics["path_support"].mean())
    minimum_support = float(diagnostics["path_support"].min())

    by_team_market = chain.assign(_direction=chain["side"].str.upper().map({"OVER": 1.0, "UNDER": -1.0}))
    coherence_values: list[float] = []
    for _, group in by_team_market.groupby(["team", "market"], dropna=False):
        if len(group) < 2:
            continue
        required = group["_direction"].to_numpy(dtype=float)
        movement = np.sign(group["delta_share"].to_numpy(dtype=float))
        coherence_values.extend((required * movement).tolist())
    allocation_coherence = float(np.mean(coherence_values)) if coherence_values else 0.0
    score = (
        marginal_log_score
        + 0.20 * mean_support
        + 0.10 * minimum_support
        + 0.05 * allocation_coherence
    )
    score_parts = {
        "marginal_log_score": marginal_log_score,
        "mean_path_support": mean_support,
        "minimum_path_support": minimum_support,
        "allocation_coherence": allocation_coherence,
    }
    if extension_model is not None and extension_model.fitted:
        extension_rows = pd.DataFrame(
            [extension_feature_row(chain.iloc[:index], chain.iloc[index]) for index in range(1, 4)]
        )
        extension_probabilities = extension_model.predict_survival(extension_rows)
        anchor_probability = float(np.clip(chain["selected_probability"].iloc[0], 1e-6, 1.0))
        score = float(
            np.log(anchor_probability)
            + np.log(np.clip(extension_probabilities, 1e-6, 1.0)).sum()
        )
        score_parts.update(
            {
                "anchor_probability": anchor_probability,
                "mean_conditional_extension_probability": float(extension_probabilities.mean()),
                "minimum_conditional_extension_probability": float(extension_probabilities.min()),
                "conditional_log_survival_score": score,
            }
        )
        chain["conditional_extension_probability"] = np.r_[
            anchor_probability, extension_probabilities
        ]
    return score, score_parts, chain


def resolve_conditional_chain(
    reservoir: pd.DataFrame,
    path_features: pd.DataFrame,
    path_certificate: dict[str, Any],
    *,
    extension_model: ConditionalExtensionModel | None = None,
    protocol: FrozenSelectorProtocol = FROZEN_SELECTOR_PROTOCOL,
) -> ChainResolution:
    """Build a developmental same-game chain without changing the frozen control."""

    control = reservoir.head(protocol.parlay_legs).copy().reset_index(drop=True)
    certificate_status = str(path_certificate.get("status", ""))
    path_authorized = bool(path_certificate.get("path_authorized", False))
    if certificate_status != PATH_REQUIRED_STATUS or not path_authorized:
        return ChainResolution(
            control_parlay=control,
            shadow_chain=reservoir.iloc[0:0].copy(),
            status="PATH_NOT_CERTIFIED",
            path_used=False,
            publication_authorized=False,
            diagnostics={
                "chain_policy_version": CHAIN_POLICY_VERSION,
                "certificate_status": certificate_status or "MISSING",
            },
        )

    required_candidates = {
        "event_id",
        "team",
        "player",
        "market",
        "side",
        "robust_score",
        "selected_probability",
    }
    required_path = {
        "event_id",
        "team",
        "player",
        "delta_share",
        "player_path_efficiency",
        "direction_reversals",
    }
    missing_candidates = sorted(required_candidates - set(reservoir.columns))
    missing_path = sorted(required_path - set(path_features.columns))
    if missing_candidates or missing_path:
        raise ValueError(
            f"chain inputs missing columns: candidates={missing_candidates}, path={missing_path}"
        )

    path_columns = list(required_path)
    merged = reservoir.merge(
        path_features[path_columns],
        on=["event_id", "team", "player"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.drop_duplicates(subset=["player"], keep="first")
    candidate_chains: list[tuple[float, dict[str, float], pd.DataFrame]] = []
    for _, event_pool in merged.groupby("event_id", sort=True):
        if len(event_pool) < protocol.parlay_legs:
            continue
        for indices in combinations(event_pool.index, protocol.parlay_legs):
            chain = event_pool.loc[list(indices)].copy()
            score, score_parts, chain = _score_chain(chain, extension_model)
            candidate_chains.append((score, score_parts, chain))

    if not candidate_chains:
        return ChainResolution(
            control_parlay=control,
            shadow_chain=merged.iloc[0:0].copy(),
            status="NO_FOUR_LEG_SHARED_GAME_CHAIN",
            path_used=True,
            publication_authorized=False,
            diagnostics={
                "chain_policy_version": CHAIN_POLICY_VERSION,
                "path_covered_candidates": int(len(merged)),
            },
        )

    candidate_chains.sort(key=lambda item: (item[0], sorted(item[2]["player"].tolist())), reverse=True)
    score, score_parts, selected = candidate_chains[0]
    selected = selected.sort_values(
        ["robust_score", "selected_probability"], ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    selected["chain_policy_version"] = CHAIN_POLICY_VERSION
    selected["chain_score"] = score
    model_used = bool(extension_model is not None and extension_model.fitted)
    return ChainResolution(
        control_parlay=control,
        shadow_chain=selected,
        status=(
            "CONDITIONAL_EXTENSION_MODEL_SHADOW"
            if model_used
            else "PATH_POLICY_DEVELOPMENT_SHADOW"
        ),
        path_used=True,
        publication_authorized=False,
        diagnostics={
            "chain_policy_version": CHAIN_POLICY_VERSION,
            "conditional_extension_model_used": model_used,
            "conditional_extension_training_rows": (
                int(extension_model.training_rows) if model_used else 0
            ),
            "evaluated_same_game_chains": int(len(candidate_chains)),
            "selected_event_id": str(selected["event_id"].iloc[0]),
            "chain_score": float(score),
            **score_parts,
        },
    )


def evaluate_one_chain_per_slate(selected_legs: pd.DataFrame) -> dict[str, Any]:
    """Evaluate only independent slate decisions, never all generated combinations."""

    required = {"slate_date", "decision_id", "player", "hit"}
    missing = sorted(required - set(selected_legs.columns))
    if missing:
        raise ValueError(f"selected chain ledger is missing required columns: {missing}")
    frame = selected_legs.copy()
    frame["hit"] = pd.to_numeric(frame["hit"], errors="coerce")
    if bool(frame["hit"].isna().any()):
        raise ValueError("selected chain ledger contains unresolved legs")

    decisions = []
    for (slate_date, decision_id), legs in frame.groupby(["slate_date", "decision_id"], sort=True):
        if len(legs) != FROZEN_SELECTOR_PROTOCOL.parlay_legs:
            raise ValueError(f"decision {decision_id} does not contain exactly four legs")
        decisions.append(
            {
                "slate_date": slate_date,
                "decision_id": decision_id,
                "parlay_hit": bool(legs["hit"].eq(1.0).all()),
                "leg_hits": int(legs["hit"].eq(1.0).sum()),
            }
        )
    decision_frame = pd.DataFrame(decisions)
    leg_wr = float(frame["hit"].mean()) if len(frame) else np.nan
    parlay_wr = float(decision_frame["parlay_hit"].mean()) if len(decision_frame) else np.nan
    return {
        "statistical_unit": "one_final_slate_decision",
        "decisions": int(len(decision_frame)),
        "legs": int(len(frame)),
        "individual_leg_wr": leg_wr,
        "four_leg_parlay_wr": parlay_wr,
        "retention_ratio": parlay_wr / leg_wr if leg_wr > 0 else None,
        "parlay_gap": leg_wr - parlay_wr if len(frame) else None,
    }
