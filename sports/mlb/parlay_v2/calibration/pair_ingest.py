from __future__ import annotations

"""Settled pair-observation ingestion (mission section 9/10). For ONE
settled day, regenerates the frozen candidate pairing deterministically
(candidate generation is a pure function of that day's frozen archived
predictive/market data -- re-deriving it post-hoc for settlement produces
identical pair_ids to what a live pregame run would have produced) and
admits a PairObservation for every pair among legs that individually pass
REQUIRED leg-level predictive support (calibration/support.py,
market_support + line_support + state_support) as of the current ledger
state.

Scope note (deliberate, not an oversight): pairing is restricted to
support-passing legs, not the full raw leg universe (~300-450/day, which
would make C(L,2) tens of thousands of pairs/day -- an unbounded, ever-
growing git-committed ledger). This mirrors mission section 10's own
sizing example ("one slate: 80 frozen candidate pairs") much more closely
than the unrestricted cross-product would, and is NOT a support gate in
disguise: it bounds WHICH pairs get a calibration observation recorded,
it does not gate policy action (see run_parlay_v2.py, which evaluates
support independently per live slate) and it does not touch
joint_support's own OBSERVE_ONLY/UNESTABLISHED status.

Second scope note, added after a real smoke test against a matured
synthetic ledger: because market_support/line_support are bucketed by
MARKET TYPE (e.g. "R|OVER|0.5"), not by specific player/game, a single
day with a mature ledger can put most same-type legs simultaneously in
support -- 203 of 420 legs in the smoke test, i.e. C(203,2) = 20,503
candidate pairs for ONE day. That is the same unbounded-ledger-growth
risk this module already exists to avoid (see above), just triggered by
ledger maturity rather than raw leg count. MAX_ELIGIBLE_LEGS_PER_DAY
bounds this the same way: once support-passing legs exceed the cap, the
excess is dropped deterministically (sorted by leg event_id, a purely
structural key with no relationship to predicted probability, price, or
any other quality signal -- so this is a volume cap on the RESEARCH
ledger, never a quality-based selection gate). `support_passing_legs` in
the returned summary is always the honest PRE-cap count;
`legs_used_for_pairing` is the post-cap count actually paired, so this
truncation is never silently invisible to an auditor.

Same-game pairs ARE included (with quoted_pair_price=None, matching
pairs.py's D_S convention) -- mission section 11 explicitly wants
n_same_game/n_cross_game counted, which would be meaningless if same-game
pairs were excluded here.
"""

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import build_binary_outcome_set, build_world_distribution, world_id_from_outcomes
from sports.mlb.research.joint_position_builder_v2.multi_target_universe import (
    PRICED_TARGETS,
    action_universe,
    build_multi_target_universe,
)
from sports.mlb.research.parlay_certification_v2 import manifest

from .pair_schema import build_pair_observation
from .pair_store import PairObservationStore
from .snapshot import build_snapshot
from .store import CalibrationStore
from .support import evaluate_support

PAIR_INGEST_VERSION = "PAIR_INGEST_V1"
PREDICTIVE_VERSION = "H_OVER_RANKER_V1+MULTI_TARGET"
FROZEN_APS_THRESHOLD = 1.0  # matches run_parlay_v2.FROZEN_APS_THRESHOLD
MAX_ELIGIBLE_LEGS_PER_DAY = 40  # bounds C(n,2) <= 780 pairs/day -- see module docstring


def _event_id(row: pd.Series) -> str:
    return f"{row['player_key']}|{row['game_id']}|{row['target']}|{row['direction']}|{row['market_line']}"


def _price_bucket(quoted_pair_price: float | None) -> str:
    if quoted_pair_price is None:
        return "no_quote"
    edges = [1.5, 2.0, 3.0, 5.0, 10.0]
    for edge in edges:
        if quoted_pair_price < edge:
            return f"<{edge}"
    return f">={edges[-1]}"


def _leg_in_support(row: pd.Series, *, calibration_rows: list[dict], independent_slate_count: int) -> bool:
    support = evaluate_support(
        calibration_rows,
        market_bucket=str(row["target"]),
        line_bucket=f"{row['target']}|{row['direction']}|{row['market_line']}",
        state_bucket=f"{PREDICTIVE_VERSION}|broad",
        independent_slate_count=independent_slate_count,
    )
    return support.in_support


def ingest_settled_pairs(
    stamp: str,
    *,
    pair_ledger_path: Path,
    calibration_store: CalibrationStore,
    targets: tuple[str, ...] = PRICED_TARGETS,
    mode: str = "broad",
    policy_version: str | None = None,
) -> dict:
    # Resolved at CALL time (not a literal default) so this always follows
    # whichever prospective attempt is currently frozen -- manifest.py's
    # own "current" pointer, not a stale hardcoded id from whenever this
    # file was last edited.
    policy_version = policy_version or manifest.PROSPECTIVE_POLICY_ID
    universe = build_multi_target_universe((stamp,), targets=targets, mode=mode)
    action = action_universe(universe).reset_index(drop=True)

    now = datetime.now(timezone.utc).isoformat()
    calibration_snapshot = build_snapshot(calibration_store, as_of=now)
    calibration_rows = calibration_store.observations_as_of(now)

    eligible_idx = [
        i for i in range(len(action))
        if _leg_in_support(action.iloc[i], calibration_rows=calibration_rows, independent_slate_count=calibration_snapshot.independent_slate_count)
    ]

    # Deterministic, quality-blind truncation -- see MAX_ELIGIBLE_LEGS_PER_DAY
    # and the module docstring's second scope note. Sorted by event_id (a
    # structural identity key), never by predicted probability/price/etc.
    paired_idx = sorted(eligible_idx, key=lambda i: _event_id(action.iloc[i]))[:MAX_ELIGIBLE_LEGS_PER_DAY]

    pair_store = PairObservationStore(pair_ledger_path)
    admitted = 0
    already_present = 0

    for i, j in combinations(paired_idx, 2):
        row_i, row_j = action.iloc[i], action.iloc[j]
        leg_1_event_id, leg_2_event_id = _event_id(row_i), _event_id(row_j)
        same_game = bool(row_i["game_id"] == row_j["game_id"])
        same_team = bool(row_i.get("team") == row_j.get("team")) if "team" in action.columns else False

        clipped = np.clip([row_i["marginal_probability"], row_j["marginal_probability"]], 1e-4, 1 - 1e-4)
        distribution = build_world_distribution([leg_1_event_id, leg_2_event_id], clipped)
        p_joint = float(distribution.probabilities[world_id_from_outcomes([1, 1])])
        outcome_set = build_binary_outcome_set(distribution, aps_threshold=FROZEN_APS_THRESHOLD, calibration_slates=0)
        losing_ids = np.array([w for w in range(4) if w != world_id_from_outcomes([1, 1])])
        counterexample_ids = np.array([w for w in outcome_set.world_ids if w in set(losing_ids.tolist())])
        counterexample_mass = float(distribution.probabilities[counterexample_ids].sum()) if len(counterexample_ids) else 0.0

        quoted_pair_price = None
        if not same_game and pd.notna(row_i["decimal_price"]) and pd.notna(row_j["decimal_price"]):
            quoted_pair_price = float(row_i["decimal_price"]) * float(row_j["decimal_price"])

        market_pair_type = "|".join(sorted([str(row_i["target"]), str(row_j["target"])]))
        line_pair_type = "__".join(sorted([
            f"{row_i['target']}|{row_i['direction']}|{row_i['market_line']}",
            f"{row_j['target']}|{row_j['direction']}|{row_j['market_line']}",
        ]))

        observation = build_pair_observation(
            slate_id=stamp,
            leg_1_event_id=leg_1_event_id,
            leg_2_event_id=leg_2_event_id,
            same_game=same_game,
            same_team=same_team,
            market_pair_type=market_pair_type,
            line_pair_type=line_pair_type,
            state_bucket_pair=f"{PREDICTIVE_VERSION}|{mode}",
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
            decision_timestamp=f"{stamp}T17:00:00Z",  # best-effort, matches calibration/ingest.py's same limitation
            leg_1_result=int(row_i["win"]),
            leg_2_result=int(row_j["win"]),
            settlement_status="settled",
            settlement_timestamp=now,
        )
        if pair_store.admit(observation):
            admitted += 1
        else:
            already_present += 1

    return {
        "stamp": stamp,
        "action_eligible_legs": int(len(action)),
        "support_passing_legs": int(len(eligible_idx)),
        "legs_used_for_pairing": int(len(paired_idx)),
        "pairs_admitted": admitted,
        "pairs_already_present": already_present,
        "pair_ledger_path": str(pair_ledger_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit one settled MLB slate's real candidate pairs into the pair-level calibration ledger (research stream, never a policy-action input).")
    parser.add_argument("--stamp", required=True, help="Settled slate date stamp, e.g. 20260820")
    parser.add_argument("--pair-ledger", type=Path, required=True)
    parser.add_argument("--calibration-ledger", type=Path, required=True, help="The leg-level calibration ledger (calibration/store.py) -- support gating for which legs get paired is evaluated against this.")
    parser.add_argument("--mode", choices=["narrow", "broad"], default="broad")
    args = parser.parse_args()
    calibration_store = CalibrationStore(args.calibration_ledger)
    summary = ingest_settled_pairs(args.stamp, pair_ledger_path=args.pair_ledger, calibration_store=calibration_store, mode=args.mode)
    print(summary)


if __name__ == "__main__":
    main()
