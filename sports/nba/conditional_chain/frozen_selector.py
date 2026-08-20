from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

from .protocol import FROZEN_SELECTOR_PROTOCOL, FrozenSelectorProtocol


REQUIRED_CANDIDATE_COLUMNS = {
    "event_date",
    "player",
    "market",
    "side",
    "line",
    "p_over",
}
REQUIRED_HISTORY_COLUMNS = {"event_date", "player", "market", "actual"}


@dataclass(frozen=True)
class FrozenSelection:
    scored_pool: pd.DataFrame
    reservoir: pd.DataFrame
    control_parlay: pd.DataFrame
    published: bool
    status: str
    selector_version: str


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def selected_probability(p_over: pd.Series, side: pd.Series) -> pd.Series:
    over = pd.to_numeric(p_over, errors="coerce")
    normalized_side = side.astype(str).str.upper()
    invalid_side = ~normalized_side.isin(["OVER", "UNDER"])
    if bool(invalid_side.any()):
        values = sorted(normalized_side.loc[invalid_side].unique().tolist())
        raise ValueError(f"unsupported side values: {values}")
    if bool(((over < 0.0) | (over > 1.0) | over.isna()).any()):
        raise ValueError("p_over must contain finite probabilities in [0, 1]")
    return pd.Series(
        np.where(normalized_side.eq("OVER"), over, 1.0 - over),
        index=p_over.index,
        dtype=float,
    )


def _state_value(actual: float, line: float, side: str) -> float:
    if np.isclose(actual, line):
        return 0.5
    over_hit = actual > line
    return float(over_hit if side == "OVER" else not over_hit)


def _candidate_state_lcb(
    candidate: pd.Series,
    history_lookup: Mapping[tuple[str, str], pd.DataFrame],
    protocol: FrozenSelectorProtocol,
) -> tuple[float, int, float]:
    event_date = pd.Timestamp(candidate["event_date"])
    key = (str(candidate["player"]), str(candidate["market"]).lower())
    player_history = history_lookup.get(key)
    if player_history is None:
        return np.nan, 0, np.nan
    prior = player_history.loc[player_history["_event_date"] < event_date]
    prior = prior.tail(protocol.lookback_games)
    if prior.empty:
        return np.nan, 0, np.nan

    side = str(candidate["side"]).upper()
    line = float(candidate["line"])
    samples = np.asarray(
        [_state_value(float(actual), line, side) for actual in prior["actual"]],
        dtype=float,
    )
    successes = float(samples.sum())
    failures = float(len(samples) - successes)
    lcb = beta_distribution.ppf(
        protocol.credible_lower_quantile,
        protocol.jeffreys_alpha + successes,
        protocol.jeffreys_beta + failures,
    )
    return float(lcb), int(len(samples)), float(samples.mean())


def score_frozen_selector(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    *,
    protocol: FrozenSelectorProtocol = FROZEN_SELECTOR_PROTOCOL,
) -> pd.DataFrame:
    """Apply the immutable Q25 selector semantics without same-day leakage."""

    _require_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "candidates")
    _require_columns(history, REQUIRED_HISTORY_COLUMNS, "history")

    scored = candidates.copy()
    scored["event_date"] = pd.to_datetime(scored["event_date"], errors="raise").dt.normalize()
    scored["side"] = scored["side"].astype(str).str.upper()
    scored["selected_probability"] = selected_probability(scored["p_over"], scored["side"])
    scored["corrected_edge"] = scored["selected_probability"] - protocol.break_even_probability

    historical = history.copy()
    historical["_event_date"] = pd.to_datetime(historical["event_date"], errors="raise").dt.normalize()
    historical["actual"] = pd.to_numeric(historical["actual"], errors="coerce")
    historical = historical.loc[historical["actual"].notna()].copy()
    historical["_player_key"] = historical["player"].astype(str)
    historical["_market_key"] = historical["market"].astype(str).str.lower()
    history_lookup = {
        (player, market): group.sort_values("_event_date")
        for (player, market), group in historical.groupby(
            ["_player_key", "_market_key"], sort=False
        )
    }

    state = [_candidate_state_lcb(row, history_lookup, protocol) for _, row in scored.iterrows()]
    scored["state_lcb"] = [value[0] for value in state]
    scored["state_history_n"] = [value[1] for value in state]
    scored["state_empirical_rate"] = [value[2] for value in state]
    scored["robust_score"] = np.minimum(
        scored["selected_probability"], scored["state_lcb"]
    ) + np.where(scored["side"].eq("OVER"), protocol.over_bonus, 0.0)
    scored["selector_version"] = protocol.version
    scored["eligible"] = (
        scored["corrected_edge"].gt(0.0)
        & scored["state_history_n"].gt(0)
        & scored["state_lcb"].notna()
    )
    return scored


def select_frozen_board(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    *,
    protocol: FrozenSelectorProtocol = FROZEN_SELECTOR_PROTOCOL,
) -> FrozenSelection:
    scored = score_frozen_selector(candidates, history, protocol=protocol)
    eligible = scored.loc[scored["eligible"]].copy()
    eligible = eligible.sort_values(
        ["robust_score", "selected_probability", "player", "market"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    eligible = eligible.drop_duplicates(subset=["player"], keep="first")
    reservoir = eligible.head(protocol.reservoir_size).copy().reset_index(drop=True)
    control = reservoir.head(protocol.parlay_legs).copy().reset_index(drop=True)
    published = bool(
        len(control) == protocol.parlay_legs
        and control["robust_score"].min() >= protocol.publication_floor
    )
    if len(control) < protocol.parlay_legs:
        status = "INSUFFICIENT_ELIGIBLE_LEGS"
    elif not published:
        status = "PUBLICATION_FLOOR_NOT_MET"
    else:
        status = "FROZEN_CONTROL_PUBLISHED"
    return FrozenSelection(
        scored_pool=scored,
        reservoir=reservoir,
        control_parlay=control,
        published=published,
        status=status,
        selector_version=protocol.version,
    )
