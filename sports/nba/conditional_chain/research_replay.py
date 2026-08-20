from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

from .core_policy import fit_rank_reliability_core
from .frozen_selector import select_frozen_board
from .protocol import FROZEN_SELECTOR_PROTOCOL, FrozenSelectorProtocol


RESEARCH_MARKET_EVIDENCE = "SYNTHETIC_THRESHOLD_HISTORY"
CANDIDATE_UNIVERSE_EVIDENCE = "FULL_CANDIDATE_UNIVERSE"
DEFAULT_REPORTED_HOLDOUT_START = pd.Timestamp("2026-02-11")
MASTER_REQUIRED_COLUMNS = {
    "date",
    "player",
    "target",
    "direction",
    "market_line",
    "prediction",
    "actual",
    "source",
    "edge_kind",
}
MARKET_MAP = {
    "PTS": "player_points",
    "TRB": "player_rebounds",
    "AST": "player_assists",
}


@dataclass(frozen=True)
class ResearchReplay:
    reservoir: pd.DataFrame
    slate_decisions: pd.DataFrame
    report: dict[str, Any]


def adapt_master_research_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(MASTER_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"master research ledger is missing required columns: {missing}"
        )
    training = frame.loc[
        frame["source"].eq("training_ledger") & frame["edge_kind"].eq("probability")
    ].copy()
    if training.empty:
        raise ValueError("master research ledger contains no probability training rows")

    adapted = pd.DataFrame(index=training.index)
    adapted["event_date"] = pd.to_datetime(
        training["date"], errors="raise"
    ).dt.normalize()
    adapted["player"] = training["player"].astype(str)
    adapted["market"] = training["target"].astype(str).str.upper().map(MARKET_MAP)
    adapted["side"] = training["direction"].astype(str).str.upper()
    adapted["line"] = pd.to_numeric(training["market_line"], errors="coerce")
    adapted["p_over"] = pd.to_numeric(training["prediction"], errors="coerce")
    adapted["actual"] = pd.to_numeric(training["actual"], errors="coerce")
    adapted["selected_odds"] = pd.to_numeric(
        training.get("selected_odds", pd.Series(np.nan, index=training.index)),
        errors="coerce",
    )
    adapted = adapted.dropna(subset=["market", "line", "p_over", "actual"])

    if "selected_probability" in training:
        expected = np.where(
            adapted["side"].eq("OVER"), adapted["p_over"], 1.0 - adapted["p_over"]
        )
        supplied = pd.to_numeric(
            training.loc[adapted.index, "selected_probability"], errors="coerce"
        )
        comparable = supplied.notna()
        mismatched = ~np.isclose(
            supplied.loc[comparable].to_numpy(dtype=float),
            np.asarray(expected)[comparable.to_numpy()],
        )
        if bool(mismatched.any()):
            raise ValueError(
                "master ledger selected-side probability semantics are inconsistent"
            )
    return adapted.reset_index(drop=True)


def _leg_result(actual: float, line: float, side: str) -> float:
    if np.isclose(actual, line):
        return 0.5
    return float(actual > line) if side == "OVER" else float(actual < line)


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


def _decision_metrics(legs: pd.DataFrame, ranks: tuple[int, ...]) -> dict[str, Any]:
    selected = legs.loc[legs["rank"].isin(ranks)].copy()
    expected_legs = len(ranks)
    decisions = selected.groupby("event_date", sort=True)["leg_result"].agg(
        legs="size",
        parlay_hit=lambda values: bool(values.eq(1.0).all()),
    )
    decisions = decisions.loc[decisions["legs"].eq(expected_legs)]
    resolved = selected.loc[selected["leg_result"].isin([0.0, 1.0])]
    wins = int(decisions["parlay_hit"].sum())
    trials = int(len(decisions))
    leg_wr = float(resolved["leg_result"].mean()) if len(resolved) else None
    parlay_wr = float(wins / trials) if trials else None
    return {
        "ranks": list(ranks),
        "decisions": trials,
        "parlay_wins": wins,
        "parlay_wr": parlay_wr,
        "parlay_wr_exact_95_ci": _exact_interval(wins, trials),
        "resolved_legs": int(len(resolved)),
        "individual_leg_wr": leg_wr,
        "retention_ratio": (
            float(parlay_wr / leg_wr)
            if parlay_wr is not None and leg_wr is not None and leg_wr > 0
            else None
        ),
    }


def _conditional_extension_metrics(control: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for extension_rank in (2, 3, 4):
        prefixes = 0
        extension_hits = 0
        for _, decision in control.groupby("event_date", sort=True):
            ordered = decision.sort_values("rank", kind="mergesort")
            prefix = ordered.loc[ordered["rank"] < extension_rank, "leg_result"]
            extension = ordered.loc[ordered["rank"].eq(extension_rank), "leg_result"]
            if len(extension) != 1 or not bool(prefix.eq(1.0).all()):
                continue
            prefixes += 1
            extension_hits += int(float(extension.iloc[0]) == 1.0)
        rows.append(
            {
                "extension_rank": extension_rank,
                "surviving_prefixes": prefixes,
                "extension_hits": extension_hits,
                "conditional_survival_rate": (
                    float(extension_hits / prefixes) if prefixes else None
                ),
            }
        )
    return rows


def replay_master_research_ledger(
    master: pd.DataFrame,
    *,
    reported_holdout_start: str | pd.Timestamp = DEFAULT_REPORTED_HOLDOUT_START,
    protocol: FrozenSelectorProtocol = FROZEN_SELECTOR_PROTOCOL,
) -> ResearchReplay:
    """Reproduce the frozen control while keeping synthetic evidence non-actionable."""

    candidates = adapt_master_research_ledger(master)
    history = (
        candidates[["event_date", "player", "market", "actual"]]
        .drop_duplicates(["event_date", "player", "market"], keep="last")
        .sort_values("event_date", kind="mergesort")
    )
    reservoir_rows: list[pd.DataFrame] = []
    slate_rows: list[dict[str, Any]] = []
    for event_date in sorted(candidates["event_date"].unique()):
        slate = candidates.loc[candidates["event_date"].eq(event_date)]
        selection = select_frozen_board(slate, history, protocol=protocol)
        slate_row = {
            "event_date": pd.Timestamp(event_date),
            "candidate_rows": int(len(slate)),
            "published": bool(selection.published),
            "status": selection.status,
            "parlay_hit": False,
            "reservoir_winners": 0,
        }
        if selection.published:
            reservoir = selection.reservoir.copy()
            reservoir["event_date"] = pd.Timestamp(event_date)
            reservoir["rank"] = np.arange(1, len(reservoir) + 1)
            reservoir["leg_result"] = [
                _leg_result(float(row["actual"]), float(row["line"]), str(row["side"]))
                for _, row in reservoir.iterrows()
            ]
            reservoir["control_selected"] = reservoir["rank"].le(protocol.parlay_legs)
            reservoir["market_evidence"] = RESEARCH_MARKET_EVIDENCE
            reservoir["candidate_universe_evidence"] = CANDIDATE_UNIVERSE_EVIDENCE
            reservoir["production_authorizable"] = False
            control = reservoir.loc[reservoir["control_selected"]]
            slate_row["parlay_hit"] = bool(control["leg_result"].eq(1.0).all())
            slate_row["reservoir_winners"] = int(reservoir["leg_result"].eq(1.0).sum())
            reservoir_rows.append(reservoir)
        slate_rows.append(slate_row)

    reservoir = (
        pd.concat(reservoir_rows, ignore_index=True)
        if reservoir_rows
        else pd.DataFrame()
    )
    slates = pd.DataFrame(slate_rows)
    holdout_start = pd.Timestamp(reported_holdout_start).normalize()
    published = slates.loc[slates["published"]]
    all_control = reservoir.loc[reservoir["control_selected"]]
    reported_control = all_control.loc[all_control["event_date"].ge(holdout_start)]
    development_control = all_control.loc[all_control["event_date"].lt(holdout_start)]
    core_policy = fit_rank_reliability_core(
        all_control,
        source_policy_version=protocol.version,
        source_model_version="UNVERSIONED_MASTER_RESEARCH_MODEL",
        training_cutoff=holdout_start,
    )

    line_fraction = np.mod(candidates["line"].to_numpy(dtype=float), 1.0)
    conventional_grid = np.isclose(line_fraction, 0.0) | np.isclose(line_fraction, 0.5)
    report = {
        "selector_version": protocol.version,
        "candidate_universe_evidence": CANDIDATE_UNIVERSE_EVIDENCE,
        "market_evidence": RESEARCH_MARKET_EVIDENCE,
        "production_authorizable": False,
        "production_blockers": [
            "NO_BOOK_QUOTE_PROVENANCE",
            "NO_OBSERVED_PRICE_PROVENANCE",
            "NO_QUOTE_TIMESTAMP_OR_RAW_SOURCE_HASH",
            "SYNTHETIC_NONSTANDARD_LINE_GRID",
            "REPORTED_HOLDOUT_REPEATEDLY_INSPECTED",
        ],
        "candidate_rows": int(len(candidates)),
        "eligible_calendar_dates": int(candidates["event_date"].nunique()),
        "published_slates": int(len(published)),
        "slate_coverage": float(len(published) / candidates["event_date"].nunique()),
        "line_audit": {
            "conventional_integer_or_half_lines": int(conventional_grid.sum()),
            "conventional_integer_or_half_fraction": float(conventional_grid.mean()),
            "verified_book_quotes": 0,
        },
        "all_published_control": _decision_metrics(all_control, (1, 2, 3, 4)),
        "development_control": _decision_metrics(development_control, (1, 2, 3, 4)),
        "reported_27_slate_control": {
            **_decision_metrics(reported_control, (1, 2, 3, 4)),
            "period_start": str(holdout_start.date()),
            "evidence_reuse_status": "REPEATEDLY_INSPECTED_DEVELOPMENT_EVIDENCE",
        },
        "reported_27_slate_conditional_extensions": _conditional_extension_metrics(
            reported_control
        ),
        "reported_27_slate_research_core_ranks_1_4": {
            **_decision_metrics(reported_control, core_policy.ranks),
            "status": "SHADOW_ONLY_NOT_CROSS_VERSION_STABLE",
            "frozen_core_policy": core_policy.as_dict(),
        },
        "reservoir_diagnosis": {
            "reservoir_size": protocol.reservoir_size,
            "slates_with_at_least_four_winners": int(
                published["reservoir_winners"].ge(4).sum()
            ),
            "published_slates": int(len(published)),
            "minimum_winners_in_reservoir": int(published["reservoir_winners"].min()),
            "mean_winners_in_reservoir": float(published["reservoir_winners"].mean()),
            "conclusion": "CHAIN_RESOLUTION_IS_PRIMARY_BOTTLENECK",
        },
    }
    reported_metrics = report["reported_27_slate_control"]
    leg_wr = reported_metrics["individual_leg_wr"]
    report["reported_27_slate_control"]["independence_expected_four_leg_wr"] = (
        float(leg_wr**4) if leg_wr is not None else None
    )
    report["reported_27_slate_control"]["interpretation"] = (
        "Observed four-leg performance is close to the product implied by marginal leg accuracy; "
        "a materially higher hit rate requires validated positive conditional dependence."
    )
    return ResearchReplay(reservoir=reservoir, slate_decisions=slates, report=report)
