from __future__ import annotations

"""Settled pair-observation ingestion. Ported from
sports/mlb/parlay_v2/calibration/pair_ingest.py, replacing MLB's
multi_target_universe-based grading with settlement_source.grade_play
(real nflverse play-by-play aggregation).

For ONE settled week's archived snapshot, regenerates the frozen
candidate pairing deterministically (pairing is a pure function of that
week's frozen archived plays -- re-deriving it post-hoc for settlement
produces identical pair_ids to what a live pregame run would have
produced) and admits a PairObservation for every pair among legs that
individually pass REQUIRED leg-level predictive support (calibration/
support.py, market_support + line_support + state_support) as of the
current ledger state.

Scope note: unlike MLB (up to ~300-450 legs/day, hence
MAX_ELIGIBLE_LEGS_PER_DAY=40), NFL's own MAXIMUM_WEEKLY_PICKS=6
(daily_policy.py) already bounds a week's play list tightly -- C(6,2)=15
pairs at most. MAX_ELIGIBLE_LEGS_PER_WEEK is kept anyway, at the same
value, for structural parity and as a harmless ceiling should that cap
ever grow; this is a volume bound, never a quality gate, matching MLB's
own documented rationale.

Same-event pairs are never constructed at all here (candidate_adapter.py's
DISTINCTNESS RULE already excludes them) -- unlike MLB, which still
includes same-game pairs with quoted_pair_price=None for research
counting purposes. NFL has no equivalent research need for same-event
pair counts (its own existing build_shadow_parlay already treats
same-event/same-player pairs as never valid), so n_same_game is always 0
here by construction, and this is disclosed rather than silently
reproducing MLB's inclusion.
"""

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from sports.nfl.conditional_chain.outcome_worlds import build_binary_outcome_set, build_world_distribution, world_id_from_outcomes
from sports.nfl.predictions.daily_policy import american_to_decimal
from sports.nfl.research.parlay_certification_v2 import manifest

from .pair_schema import build_pair_observation
from .pair_store import PairObservationStore
from .settlement_source import SETTLEMENT_SOURCE_VERSION, grade_play, load_season_actuals
from .snapshot import build_snapshot
from .store import CalibrationStore
from .support import evaluate_support

PAIR_INGEST_VERSION = "NFL_PAIR_INGEST_V1"
PREDICTIVE_VERSION = "NFL_PASSING_LOSS_AWARE_META_POLICY_V2"
FROZEN_APS_THRESHOLD = 1.0  # matches run_parlay_v2.FROZEN_APS_THRESHOLD
MAX_ELIGIBLE_LEGS_PER_WEEK = 40  # see module docstring


def _event_id(play: dict[str, Any]) -> str:
    return f"{play.get('player_id')}|{play.get('event_id')}|{play.get('target') or play.get('market')}|{play.get('direction')}|{play.get('line')}"


def _price_bucket(quoted_pair_price: float | None) -> str:
    if quoted_pair_price is None:
        return "no_quote"
    edges = [1.5, 2.0, 3.0, 5.0, 10.0]
    for edge in edges:
        if quoted_pair_price < edge:
            return f"<{edge}"
    return f">={edges[-1]}"


def _leg_in_support(play: dict[str, Any], *, calibration_rows: list[dict], independent_slate_count: int) -> bool:
    target = str(play.get("target") or play.get("market"))
    support = evaluate_support(
        calibration_rows,
        market_bucket=target,
        line_bucket=f"{target}|{play.get('direction')}|{play.get('line')}",
        state_bucket=f"{PREDICTIVE_VERSION}|{SETTLEMENT_SOURCE_VERSION}",
        independent_slate_count=independent_slate_count,
    )
    return support.in_support


def ingest_settled_pairs(
    snapshot_path: Path,
    *,
    season: int,
    week: int,
    pair_ledger_path: Path,
    calibration_store: CalibrationStore,
    actuals_cache_path: Path | None = None,
    policy_version: str | None = None,
) -> dict:
    # Resolved at CALL time (not a literal default) so this always follows
    # whichever prospective attempt is currently frozen.
    policy_version = policy_version or manifest.PROSPECTIVE_POLICY_ID
    week_id = f"{season}-W{week:02d}"

    snapshot_path = Path(snapshot_path)
    if not snapshot_path.is_file():
        return {"week_id": week_id, "action_eligible_legs": 0, "support_passing_legs": 0, "legs_used_for_pairing": 0, "pairs_admitted": 0, "pairs_already_present": 0, "pair_ledger_path": str(pair_ledger_path), "reason": "snapshot_not_found"}

    with open(snapshot_path, encoding="utf-8") as f:
        payload = json.load(f)
    plays_raw = payload.get("plays") if isinstance(payload, dict) else None
    plays = list(plays_raw) if isinstance(plays_raw, list) else []

    actuals = load_season_actuals(season, cache_path=actuals_cache_path)
    now = datetime.now(timezone.utc).isoformat()
    calibration_snapshot = build_snapshot(calibration_store, as_of=now)
    calibration_rows = calibration_store.observations_as_of(now)

    graded_plays = []
    for play in plays:
        win = grade_play(play, actuals, season=season, week=week)
        if win is None:
            continue
        graded_plays.append((play, win))

    eligible_idx = [
        i for i, (play, _win) in enumerate(graded_plays)
        if _leg_in_support(play, calibration_rows=calibration_rows, independent_slate_count=calibration_snapshot.independent_slate_count)
    ]

    # Deterministic, quality-blind truncation -- see MAX_ELIGIBLE_LEGS_PER_WEEK.
    # Sorted by event_id (a structural identity key), never by predicted
    # probability/price/etc.
    paired_idx = sorted(eligible_idx, key=lambda i: _event_id(graded_plays[i][0]))[:MAX_ELIGIBLE_LEGS_PER_WEEK]

    pair_store = PairObservationStore(pair_ledger_path)
    admitted = 0
    already_present = 0

    for i, j in combinations(paired_idx, 2):
        play_i, win_i = graded_plays[i]
        play_j, win_j = graded_plays[j]
        if play_i.get("event_id") == play_j.get("event_id") or play_i.get("player_id") == play_j.get("player_id"):
            continue  # distinctness rule -- see module docstring
        leg_1_event_id, leg_2_event_id = _event_id(play_i), _event_id(play_j)

        clipped = np.clip([float(play_i["model_hit_probability"]), float(play_j["model_hit_probability"])], 1e-4, 1 - 1e-4)
        distribution = build_world_distribution([leg_1_event_id, leg_2_event_id], clipped)
        p_joint = float(distribution.probabilities[world_id_from_outcomes([1, 1])])
        outcome_set = build_binary_outcome_set(distribution, aps_threshold=FROZEN_APS_THRESHOLD, calibration_slates=0)
        losing_ids = np.array([w for w in range(4) if w != world_id_from_outcomes([1, 1])])
        counterexample_ids = np.array([w for w in outcome_set.world_ids if w in set(losing_ids.tolist())])
        counterexample_mass = float(distribution.probabilities[counterexample_ids].sum()) if len(counterexample_ids) else 0.0

        price_i = play_i.get("selected_side_price")
        price_j = play_j.get("selected_side_price")
        quoted_pair_price = None
        if price_i is not None and price_j is not None:
            quoted_pair_price = float(american_to_decimal(float(price_i))) * float(american_to_decimal(float(price_j)))

        target_i = str(play_i.get("target") or play_i.get("market"))
        target_j = str(play_j.get("target") or play_j.get("market"))
        market_pair_type = "|".join(sorted([target_i, target_j]))
        line_pair_type = "__".join(sorted([
            f"{target_i}|{play_i.get('direction')}|{play_i.get('line')}",
            f"{target_j}|{play_j.get('direction')}|{play_j.get('line')}",
        ]))

        observation = build_pair_observation(
            slate_id=week_id,
            leg_1_event_id=leg_1_event_id,
            leg_2_event_id=leg_2_event_id,
            same_game=False,  # distinctness rule excludes same-event pairs by construction
            same_team=bool(play_i.get("team") and play_i.get("team") == play_j.get("team")),
            market_pair_type=market_pair_type,
            line_pair_type=line_pair_type,
            state_bucket_pair=f"{PREDICTIVE_VERSION}|{SETTLEMENT_SOURCE_VERSION}",
            price_bucket=_price_bucket(quoted_pair_price),
            quoted_pair_price=quoted_pair_price,
            predicted_joint_probability=p_joint,
            # No non-independence joint model exists in this repo yet --
            # both fields are the same value today, so joint_residual is
            # always 0.0 (see pair_schema.py's module docstring).
            predicted_independence_probability=p_joint,
            counterexample_count=int(len(counterexample_ids)),
            counterexample_mass=counterexample_mass,
            retained_world_count=int(outcome_set.world_count),
            retained_probability_mass=float(distribution.probabilities[outcome_set.world_ids].sum()) if outcome_set.world_count else 0.0,
            calibration_snapshot_id=calibration_snapshot.calibration_snapshot_id,
            predictive_version=PREDICTIVE_VERSION,
            policy_version=policy_version,
            decision_timestamp=str(play_i.get("snapshot_time_utc") or f"{week_id}T00:00:00Z"),  # best-effort, matches calibration/ingest.py's same limitation
            leg_1_result=int(win_i),
            leg_2_result=int(win_j),
            settlement_status="settled",
            settlement_timestamp=now,
        )
        if pair_store.admit(observation):
            admitted += 1
        else:
            already_present += 1

    return {
        "week_id": week_id,
        "action_eligible_legs": int(len(graded_plays)),
        "support_passing_legs": int(len(eligible_idx)),
        "legs_used_for_pairing": int(len(paired_idx)),
        "pairs_admitted": admitted,
        "pairs_already_present": already_present,
        "pair_ledger_path": str(pair_ledger_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit one settled NFL week's real candidate pairs into the pair-level calibration ledger (research stream, never a policy-action input).")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to the archived weekly plays snapshot JSON.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--pair-ledger", type=Path, required=True)
    parser.add_argument("--calibration-ledger", type=Path, required=True, help="The leg-level calibration ledger (calibration/store.py) -- support gating for which legs get paired is evaluated against this.")
    parser.add_argument("--actuals-cache", type=Path, default=None)
    args = parser.parse_args()
    calibration_store = CalibrationStore(args.calibration_ledger)
    summary = ingest_settled_pairs(
        args.snapshot, season=args.season, week=args.week, pair_ledger_path=args.pair_ledger,
        calibration_store=calibration_store, actuals_cache_path=args.actuals_cache,
    )
    print(summary)


if __name__ == "__main__":
    main()
