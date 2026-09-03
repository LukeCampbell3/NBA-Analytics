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

PAIR_INGEST_VERSION = "PAIR_INGEST_V1_1"  # additive: per-leg marginal/decimal_price/no_vig capture
PREDICTIVE_VERSION = "H_OVER_RANKER_V1+MULTI_TARGET"
FROZEN_APS_THRESHOLD = 1.0  # matches run_parlay_v2.FROZEN_APS_THRESHOLD
MAX_ELIGIBLE_LEGS_PER_DAY = 40  # bounds C(n,2) <= 780 pairs/day -- see module docstring

# Where the source pool CSVs live -- same path multi_target_universe reads
# from, referenced here (only for no-vig capture) so pair_ingest never
# reaches inside multi_target_universe's private state. parents[4]: this
# file lives at sports/mlb/parlay_v2/calibration/pair_ingest.py, so four
# levels up is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
DAILY_RUNS_ROOT = _REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"


def _american_to_decimal(american) -> float | None:
    """Match multi_target_universe._decimal_price exactly (kept local so
    this module never depends on a private helper of another module)."""
    try:
        value = float(american)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or abs(value) < 100.0:
        return None
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def _no_vig_probability(chosen_side_decimal: float | None, other_side_decimal: float | None) -> float | None:
    """No-vig implied probability for the CHOSEN side, given both sides'
    real decimal quotes at decision time.

    Returns None whenever either side is missing / non-priced. This is
    the honest answer -- with only one side quoted, no juice can be
    stripped and the chosen-side 1/price includes the book's vig.
    """
    if chosen_side_decimal is None or other_side_decimal is None:
        return None
    if chosen_side_decimal <= 1.0 or other_side_decimal <= 1.0:
        return None
    p_chosen = 1.0 / chosen_side_decimal
    p_other = 1.0 / other_side_decimal
    total = p_chosen + p_other
    if not (1.0 < total < 2.0):  # sanity: overround should exist but not exceed 100%
        return None
    return p_chosen / total


def _pair_side_capture(stamp: str) -> dict:
    """Return a per-leg dict keyed by
    (player_key_lower, game_id, target, direction, market_line)
    with (chosen_side_decimal, other_side_decimal, no_vig_prob_chosen).

    Reads the same source CSV multi_target_universe reads, but only
    to recover the OPPOSITE-side price (the action universe drops it
    after choosing a side). No-vig capture is opt-in: a missing CSV
    or a row without both prices yields no entry, and the caller
    silently populates None for those legs.
    """
    path = DAILY_RUNS_ROOT / stamp / f"daily_prediction_pool_{stamp}.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    # Local player-key normalization (light, tolerant): lower-casing +
    # underscore-joining. multi_target_universe uses vhfp's normalizer;
    # we compare only lower-cased strings so both keys converge for
    # normal alphanumeric names.
    def _norm(x) -> str:
        return str(x or "").strip().lower().replace(" ", "_")

    lookup: dict[tuple, tuple[float | None, float | None, float | None]] = {}
    for _, row in frame.iterrows():
        try:
            target = str(row.get("Target") or "").strip().upper()
            game_id = str(row.get("Game_ID") or "").strip()
            player_key = _norm(row.get("Player_ID") or row.get("Player"))
            market_line = float(row["Market_Line"]) if pd.notna(row.get("Market_Line")) else None
        except (TypeError, ValueError, KeyError):
            continue
        if not (target and game_id and player_key and market_line is not None):
            continue

        over_dec = _american_to_decimal(row.get("Market_Over_Price"))
        under_dec = _american_to_decimal(row.get("Market_Under_Price"))
        # Emit an entry for each direction independently -- action rows
        # come in one direction at a time.
        for direction, chosen, other in (
            ("OVER", over_dec, under_dec),
            ("UNDER", under_dec, over_dec),
        ):
            lookup[(player_key, game_id, target, direction, market_line)] = (
                chosen, other, _no_vig_probability(chosen, other),
            )
    return lookup


def _lookup_side_capture(
    lookup: dict, action_row: pd.Series
) -> tuple[float | None, float | None]:
    """Extract (chosen_side_decimal, no_vig_probability) for one action
    row from the pre-built side-capture lookup. Both values are None
    when the pool row is missing or one side is unpriced.
    """
    key = (
        str(action_row.get("player_key") or "").strip().lower(),
        str(action_row.get("game_id") or ""),
        str(action_row.get("target") or "").strip().upper(),
        str(action_row.get("direction") or "").strip().upper(),
        float(action_row["market_line"]) if pd.notna(action_row.get("market_line")) else None,
    )
    entry = lookup.get(key)
    if entry is None:
        return None, None
    chosen, _other, no_vig = entry
    return chosen, no_vig


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

    # Side-capture lookup for no-vig computation. Built once per slate,
    # NOT inside the pair loop -- (~500 pair inserts/day, one CSV read
    # is enough). Empty when the source CSV is missing; in that case
    # every per-leg no-vig field falls back to None and the schema
    # is v1-shape by omission, exactly the backward-compat contract.
    side_capture = _pair_side_capture(stamp)

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

        leg_1_chosen_dec, leg_1_no_vig = _lookup_side_capture(side_capture, row_i)
        leg_2_chosen_dec, leg_2_no_vig = _lookup_side_capture(side_capture, row_j)

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
            leg_1_marginal_probability=float(row_i["marginal_probability"]),
            leg_2_marginal_probability=float(row_j["marginal_probability"]),
            leg_1_decimal_price=(float(leg_1_chosen_dec) if leg_1_chosen_dec is not None else None),
            leg_2_decimal_price=(float(leg_2_chosen_dec) if leg_2_chosen_dec is not None else None),
            leg_1_no_vig_market_probability=(float(leg_1_no_vig) if leg_1_no_vig is not None else None),
            leg_2_no_vig_market_probability=(float(leg_2_no_vig) if leg_2_no_vig is not None else None),
        )
        if pair_store.admit(observation):
            admitted += 1
        else:
            already_present += 1

    legs_with_no_vig = sum(
        1 for i in paired_idx
        if _lookup_side_capture(side_capture, action.iloc[i])[1] is not None
    )
    return {
        "stamp": stamp,
        "action_eligible_legs": int(len(action)),
        "support_passing_legs": int(len(eligible_idx)),
        "legs_used_for_pairing": int(len(paired_idx)),
        "legs_with_no_vig_capture": legs_with_no_vig,
        "side_capture_available": bool(side_capture),
        "pairs_admitted": admitted,
        "pairs_already_present": already_present,
        "pair_ledger_path": str(pair_ledger_path),
        "pair_ingest_version": PAIR_INGEST_VERSION,
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
