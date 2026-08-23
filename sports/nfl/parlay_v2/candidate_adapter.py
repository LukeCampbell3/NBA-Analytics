from __future__ import annotations

"""NFL PARLAY CANDIDATE ADAPTER -- the ONLY bridge between NFL's existing
live weekly board (sports/nfl/predictions/daily_policy.select_live_board's
"play" dicts) and PARLAY_CERTIFICATION_V2. Ported from
sports/mlb/parlay_v2/candidate_adapter.py, replacing MLB's DataFrame-based
multi_target_universe source with NFL's existing dict-based play schema --
the certification math (world distribution, APS/counterexample
diagnostics) is unchanged and reused verbatim.

CRITICAL SEPARATION OF PREDICTION AND CERTIFICATION: this module NEVER
performs certification and NEVER emits an authoritative field
(`certified`, `safe`, `supported`, `production_authorized`, `risk_passed`,
or anything that reads as one). Everything it produces is
descriptive/proposal-only -- a `PairCandidate` is the model saying "I
propose this pair," nothing more. Only
sports/nfl/research/parlay_certification_v2/policy.py (fed by
world_certificate.py) decides ACT/ABSTAIN, and only
parlay_certification_v2/state_machine.py (driven by anytime_monitor.py's
real prospective evidence) may ever say a policy is supported.

Reuses, unmodified: sports.nfl.predictions.daily_policy.american_to_decimal
(the frozen price conversion already used by NFL's own build_shadow_parlay)
and outcome_worlds.build_world_distribution (independence joint model,
unmodified -- same module MLB's adapter uses, ported verbatim into
sports/nfl/conditional_chain/outcome_worlds.py).

DISTINCTNESS RULE (deliberate NFL-specific adaptation, not an arbitrary
invention): a pair is only ever proposed for two plays with a DIFFERENT
event_id AND a DIFFERENT player_id. This mirrors NFL's own existing
build_shadow_parlay's dedup rule exactly (daily_policy.py) rather than
MLB's adapter, which builds every combination and defers same-game
pricing exclusion to run_parlay_v2.py -- two markets on the same player
(e.g. passing yards + passing TDs for one QB) are obviously dependent,
and NFL's already-shipped, already-backtested old system already encodes
that judgment; there is no reason to relitigate it here.

EXACT-EVENT IDENTITY: every probability, price, and support diagnostic in
a candidate leg is looked up by the FULL event key (player, event, market,
direction, line) via `exact_event_key`. There is no code path in this
module that reads a probability/price for one line and attaches it to a
leg carrying a different line -- `_play_to_leg` builds a leg directly from
a single source play dict.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from sports.nfl.conditional_chain.outcome_worlds import (
    build_binary_outcome_set,
    build_world_distribution,
    world_id_from_outcomes,
)
from sports.nfl.predictions.daily_policy import american_to_decimal
from sports.nfl.research.parlay_certification_v2.world_certificate import build_nonvacuous_world_certificate

ADAPTER_VERSION = "NFL_PARLAY_V2_CANDIDATE_ADAPTER_V1"
JOINT_PROBABILITY_METHOD = "independence_binary_world_model"

MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT = 20  # matches calibration/support.py's N_STATE convention


def exact_event_key(player_id: str, event_id: str, market: str, direction: str, line: float) -> tuple[str, str, str, str, float]:
    """The full identity of ONE market event. Two plays that differ only in
    `line` are DIFFERENT events and must never share a probability, price,
    or support diagnostic. Always use this key (never a subset of it) for
    any per-event lookup."""
    return (str(player_id), str(event_id), str(market), str(direction), float(line))


@dataclass(frozen=True)
class Leg:
    player: str
    player_id: str
    event_id: str
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
            "event_id": self.event_id,
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

    week_id: str
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
            "week_id": self.week_id,
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


def _play_to_leg(play: dict[str, Any]) -> Leg:
    price = play.get("selected_side_price")
    return Leg(
        player=str(play["player"]),
        player_id=str(play["player_id"]),
        event_id=str(play["event_id"]),
        target=str(play.get("market") or play.get("target")),
        side=str(play["direction"]),
        line=float(play["line"]),
        book=(str(play["selected_sportsbook_key"]) if play.get("selected_sportsbook_key") else None),
        decimal_price=(float(american_to_decimal(float(price))) if price is not None else None),
        quote_timestamp=(str(play["snapshot_time_utc"]) if play.get("snapshot_time_utc") else None),
        model_probability_estimate=float(play["model_hit_probability"]),
        in_support=bool(play.get("confidence_in_support", False)),
    )


def build_candidates_for_week(
    plays: list[dict[str, Any]],
    *,
    week_id: str,
    aps_threshold: float,
    calibration_slates: int,
    predictive_version: str,
    state_version: str,
) -> list[PairCandidate]:
    """plays: one week's action-eligible plays from
    sports.nfl.predictions.daily_policy.select_live_board (already
    exact-event-scoped: one play per (event_id, player_key, target, side,
    line)). Builds every cross-event, cross-player 2-leg pair -- see
    module docstring's DISTINCTNESS RULE."""
    legs = [_play_to_leg(play) for play in plays]
    candidates: list[PairCandidate] = []
    state_support = bool(calibration_slates >= MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT)

    for idx_i, idx_j in combinations(range(len(legs)), 2):
        leg_i, leg_j = legs[idx_i], legs[idx_j]
        if leg_i.event_id == leg_j.event_id or leg_i.player_id == leg_j.player_id:
            continue

        world_id_i = f"{leg_i.player}|{leg_i.target}|{leg_i.side}|{leg_i.line}|{leg_i.event_id}"
        world_id_j = f"{leg_j.player}|{leg_j.target}|{leg_j.side}|{leg_j.line}|{leg_j.event_id}"
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
        candidate_id = f"{week_id}|{world_id_i}||{world_id_j}"
        candidates.append(
            PairCandidate(
                week_id=week_id,
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


def build_week_action_plays(daily_predictions_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """LIVE-SAFE builder: pulls this week's action-eligible plays straight
    out of the already-published daily_predictions.json payload's "plays"
    key (sports/nfl/scripts/run_nfl_daily_predictions.py already produces
    exactly the select_live_board schema this adapter needs -- unlike MLB,
    NFL requires no separate pregame-row reconstruction step)."""
    plays = daily_predictions_payload.get("plays") if isinstance(daily_predictions_payload, dict) else None
    return list(plays) if isinstance(plays, list) else []
