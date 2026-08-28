from __future__ import annotations

"""NEW PARLAY CANDIDATE ADAPTER (mission section 3) -- the ONLY bridge
between today's pregame MLB predictive universe and PARLAY_CERTIFICATION_V2.

CRITICAL SEPARATION OF PREDICTION AND CERTIFICATION (mission section 2):
this module NEVER performs certification and NEVER emits an authoritative
field (`certified`, `safe`, `supported`, `production_authorized`,
`risk_passed`, or anything that reads as one). Everything it produces is
descriptive/proposal-only -- a `PairCandidate` is the model saying "I
propose this pair," nothing more. Only
sports/mlb/research/parlay_certification_v2/policy.py (fed by
world_certificate.py) decides ACT/ABSTAIN, and only
parlay_certification_v2/state_machine.py (driven by anytime_monitor.py's
real prospective evidence) may ever say a policy is supported.

Reuses, unmodified: h_over_ranker.baselines.probability_score (the frozen
marginal model), multi_target_universe.build_multi_target_universe /
action_universe (the predictive observation/action universe -- itself
target-agnostic and already fixed for the alternate-line dedup hazard,
see multi_target_universe.py), and outcome_worlds.build_world_distribution
(independence joint model, unmodified).

EXACT-EVENT IDENTITY (mission section 5/14): every probability, price, and
support diagnostic in a candidate leg is looked up by the FULL event key
(player, game, target, side, line) via `exact_event_key`. There is no code
path in this module that reads a probability/price for one line and
attaches it to a leg carrying a different line -- `_row_to_leg` builds a
leg directly from a single source row, and `_event_key_of_leg` is used
only for dedup/collision detection, never for cross-line lookup. See
test_candidate_adapter.py::test_alternate_lines_never_share_probability_or_price
for the mandatory regression test this guards.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import (
    build_binary_outcome_set,
    build_world_distribution,
    world_id_from_outcomes,
)
from sports.mlb.research.h_over_ranker.baselines import probability_score
from sports.mlb.research.h_over_ranker.eligibility import _to_float
from sports.mlb.research.joint_position_builder_v2.multi_target_universe import (
    MAX_SANE_RMSE,
    MIN_HISTORY_ROWS_FOR_SUPPORT,
    PRICED_TARGETS,
    _decimal_price,
    action_universe,
    build_multi_target_universe,
    frozen_bias,
)
from sports.mlb.research.parlay_certification_v2.world_certificate import build_nonvacuous_world_certificate

ADAPTER_VERSION = "PARLAY_V2_CANDIDATE_ADAPTER_V1"
JOINT_PROBABILITY_METHOD = "independence_binary_world_model"

MIN_HISTORY_ROWS_FOR_LEG_SUPPORT = 20
MAX_SANE_RMSE_FOR_LEG_SUPPORT = 5.0
MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT = 20  # matches this repo's existing MIN_CALIBRATION_PAIRS/minimum_calibration_slates convention


def exact_event_key(player_key: str, game_id: str, target: str, side: str, line: float) -> tuple[str, str, str, str, float]:
    """The full identity of ONE market event. Two rows that differ only in
    `line` (e.g. H OVER 0.5 vs H OVER 1.5) are DIFFERENT events and must
    never share a probability, price, or support diagnostic. Always use
    this key (never a subset of it) for any per-event lookup."""
    return (str(player_key), str(game_id), str(target), str(side), float(line))


@dataclass(frozen=True)
class Leg:
    player: str
    player_id: str
    game_id: str
    target: str
    side: str
    line: float
    book: str | None
    decimal_price: float | None
    quote_timestamp: str | None
    # descriptive-only, not authoritative:
    model_probability_estimate: float
    in_support: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "player_id": self.player_id,
            "game_id": self.game_id,
            "target": self.target,
            "side": self.side,
            "line": self.line,
            "book": self.book,
            "decimal_price": self.decimal_price,
            "quote_timestamp": self.quote_timestamp,
            "model_probability_estimate": self.model_probability_estimate,
            "in_support": self.in_support,
        }


@dataclass(frozen=True)
class PairCandidate:
    """A PROPOSAL only -- see module docstring. No field on this object,
    or anywhere in `as_dict()`, is or may become an authorization flag."""

    slate_id: str
    candidate_id: str
    leg_1: Leg
    leg_2: Leg
    joint_probability_estimate: float
    joint_probability_method: str
    joint_score: float
    support: dict[str, bool]
    world_diagnostics: dict[str, Any]
    predictive_version: str
    state_version: str
    adapter_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "slate_id": self.slate_id,
            "candidate_id": self.candidate_id,
            "leg_1": self.leg_1.as_dict(),
            "leg_2": self.leg_2.as_dict(),
            "joint_probability_estimate": self.joint_probability_estimate,
            "joint_probability_method": self.joint_probability_method,
            "joint_score": self.joint_score,
            "support": dict(self.support),
            "world_diagnostics": dict(self.world_diagnostics),
            "predictive_version": self.predictive_version,
            "state_version": self.state_version,
            "adapter_version": self.adapter_version,
        }


def _row_to_leg(row: pd.Series) -> Leg:
    in_support = bool(
        float(row["history_rows"]) >= MIN_HISTORY_ROWS_FOR_LEG_SUPPORT
        and np.isfinite(row["rmse"])
        and 0.0 < float(row["rmse"]) < MAX_SANE_RMSE_FOR_LEG_SUPPORT
    )
    return Leg(
        player=str(row["player"]),
        player_id=str(row["player_key"]),
        game_id=str(row["game_id"]),
        target=str(row["target"]),
        side=str(row["direction"]),
        line=float(row["market_line"]),
        book=(str(row["market_source"]) if row.get("market_source") else None),
        decimal_price=(float(row["decimal_price"]) if pd.notna(row.get("decimal_price")) else None),
        quote_timestamp=str(row["date"]),  # daily-pool granularity; no finer quote timestamp exists upstream
        model_probability_estimate=float(row["marginal_probability"]),
        in_support=in_support,
    )


def build_candidates_for_day(
    day_rows: pd.DataFrame,
    *,
    slate_id: str,
    aps_threshold: float,
    calibration_slates: int,
    predictive_version: str,
    state_version: str,
) -> list[PairCandidate]:
    """day_rows: one day's action-eligible rows from
    multi_target_universe.action_universe (already exact-event-scoped: one
    row per (date, player_key, game_id, target, direction, market_line)).
    Builds every cross-game 2-leg pair (mission section 6: initially
    prefer cross-game; same-game pairs are still constructed as PROPOSALS
    -- descriptive world diagnostics only -- but never priced, since no
    real SGP quote/dependence model exists here, matching pairs.py's D_S
    convention)."""
    rows = day_rows.reset_index(drop=True)
    candidates: list[PairCandidate] = []
    state_support = bool(calibration_slates >= MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT)

    for idx_i, idx_j in combinations(range(len(rows)), 2):
        row_i, row_j = rows.iloc[idx_i], rows.iloc[idx_j]
        leg_i, leg_j = _row_to_leg(row_i), _row_to_leg(row_j)

        world_id_i = f"{leg_i.player}|{leg_i.target}|{leg_i.side}|{leg_i.line}|{leg_i.game_id}"
        world_id_j = f"{leg_j.player}|{leg_j.target}|{leg_j.side}|{leg_j.line}|{leg_j.game_id}"
        clipped = np.clip([leg_i.model_probability_estimate, leg_j.model_probability_estimate], 1e-4, 1 - 1e-4)
        distribution = build_world_distribution([world_id_i, world_id_j], clipped)  # independence -- see module docstring
        p_joint = float(distribution.probabilities[world_id_from_outcomes([1, 1])])

        outcome_set = build_binary_outcome_set(distribution, aps_threshold=aps_threshold, calibration_slates=calibration_slates)
        losing_world_ids = np.array([w for w in range(4) if w != world_id_from_outcomes([1, 1])])
        certificate = build_nonvacuous_world_certificate(outcome_set.world_ids, distribution.probabilities, losing_world_ids)

        support = {
            "leg_1_support": leg_i.in_support,
            "leg_2_support": leg_j.in_support,
            "state_support": state_support,
            "in_support": bool(leg_i.in_support and leg_j.in_support and state_support),
        }
        world_diagnostics = {
            "retained_world_count": certificate.retained_world_count,
            "retained_probability_mass": certificate.retained_probability_mass,
            "counterexample_count": certificate.counterexample_count,
            "counterexample_mass": certificate.counterexample_mass,
            "nonvacuous_world_certificate": certificate.certified,
        }
        candidate_id = f"{slate_id}|{world_id_i}||{world_id_j}"
        candidates.append(
            PairCandidate(
                slate_id=slate_id,
                candidate_id=candidate_id,
                leg_1=leg_i,
                leg_2=leg_j,
                joint_probability_estimate=p_joint,
                joint_probability_method=JOINT_PROBABILITY_METHOD,
                joint_score=certificate.retained_probability_mass,  # ranking diagnostic only -- see module docstring
                support=support,
                world_diagnostics=world_diagnostics,
                predictive_version=predictive_version,
                state_version=state_version,
                adapter_version=ADAPTER_VERSION,
            )
        )
    return candidates


def build_day_action_universe(stamps: tuple[str, ...], date: str, *, mode: str = "broad") -> pd.DataFrame:
    """BACKTEST/RESEARCH ONLY -- reuses multi_target_universe, which
    requires a KNOWN settlement outcome for every row (it grades win/loss
    against Player-Predictor/Data-Proc-MLB history). Do not call this for
    a live, pregame slate -- use build_pregame_action_rows for that."""
    universe = build_multi_target_universe(stamps, targets=PRICED_TARGETS, mode=mode)
    action = action_universe(universe)
    return action[action["date"].astype(str) == str(date)].reset_index(drop=True)


def build_pregame_action_rows(
    pool_csv: pd.DataFrame,
    *,
    stamp: str,
    targets: tuple[str, ...] = PRICED_TARGETS,
    mode: str = "broad",
) -> pd.DataFrame:
    """LIVE-SAFE builder: consumes one day's daily_prediction_pool_*.csv
    (already loaded as a DataFrame) and produces the SAME row schema as
    multi_target_universe.action_universe -- WITHOUT ever requiring a
    known settlement outcome (no `actual`/`win` field is read or
    computed). This is the correct source for a genuinely pregame
    'today's candidate universe': everything it needs (Prediction,
    Market_Line, Model_Val_RMSE, History_Rows, Market_Over_Price/
    Market_Under_Price, Market_Source) is available before first pitch.

    Reuses frozen_bias(target) and probability_score UNCHANGED -- the
    frozen marginal model is not touched by operating pregame instead of
    retrospectively; only the settlement-dependent grading step is
    skipped, because it cannot exist yet."""
    if mode not in ("narrow", "broad"):
        raise ValueError("mode must be 'narrow' or 'broad'")
    rows: list[dict] = []
    for target in targets:
        bias = frozen_bias(target)
        target_rows = pool_csv[pool_csv.get("Target", "").astype(str).str.strip().str.upper() == target]
        for _, row in target_rows.iterrows():
            raw_player_type = row.get("Player_Type", "hitter")
            player_type = "hitter" if pd.isna(raw_player_type) else str(raw_player_type or "hitter").strip().lower()
            opposing_pitcher = row.get("Opposing_Pitcher", "")
            starter_missing = pd.isna(opposing_pitcher) or not str(opposing_pitcher or "").strip()
            if player_type != "pitcher" and starter_missing:
                # Hitter probabilities are matchup-dependent. Keep the parlay
                # action universe aligned with the singles/publication gate so
                # an unresolved probable starter cannot re-enter as a leg.
                continue
            prediction = _to_float(row.get("Prediction"))
            market_line = _to_float(row.get("Market_Line"))
            if prediction is None or market_line is None:
                continue
            corrected_prediction = prediction - bias
            corrected_edge = corrected_prediction - market_line
            if mode == "narrow" and corrected_edge <= 0:
                continue
            direction = "OVER" if corrected_edge > 0 else ("UNDER" if corrected_edge < 0 else None)
            if direction is None:
                continue

            game_date = str(row.get("Game_Date", ""))[:10]
            player = str(row.get("Player", ""))
            game_id = str(row.get("Game_ID", "") or "")
            if not game_date or not player or not game_id:
                continue

            rmse = _to_float(row.get("Model_Val_RMSE")) or 0.3
            history_rows = _to_float(row.get("History_Rows")) or 0.0
            over_prob = probability_score(corrected_prediction, market_line, rmse)
            marginal_probability = over_prob if direction == "OVER" else (1.0 - over_prob)

            american_price = row.get("Market_Over_Price") if direction == "OVER" else row.get("Market_Under_Price")
            decimal_price = _decimal_price(_to_float(american_price))

            in_support = bool(
                history_rows >= MIN_HISTORY_ROWS_FOR_SUPPORT and np.isfinite(rmse) and 0.0 < rmse < MAX_SANE_RMSE
            )

            rows.append(
                {
                    "date": stamp,
                    "player": player,
                    "player_key": str(row.get("Player_ID") or player),
                    "game_id": game_id,
                    "target": target,
                    "direction": direction,
                    "market_line": market_line,
                    "marginal_probability": marginal_probability,
                    "decimal_price": decimal_price,
                    "rmse": rmse,
                    "history_rows": history_rows,
                    "market_source": str(row.get("Market_Source", "") or "").strip().lower(),
                    "in_support": in_support,
                }
            )
    columns = [
        "date", "player", "player_key", "game_id", "target", "direction", "market_line",
        "marginal_probability", "decimal_price", "rmse", "history_rows", "market_source", "in_support",
    ]
    # Always construct with explicit columns, even for rows=[] -- pd.DataFrame([])
    # has zero columns, which would crash the filter below with a KeyError
    # instead of legitimately returning an empty (but well-formed) frame.
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates(
        subset=["date", "player_key", "game_id", "target", "direction", "market_line"]
    ).reset_index(drop=True)
    return frame[frame["in_support"] & frame["decimal_price"].notna()].reset_index(drop=True)
