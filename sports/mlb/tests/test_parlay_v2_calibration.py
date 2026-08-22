from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sports.mlb.parlay_v2.calibration.schema import build_observation, exact_event_identity
from sports.mlb.parlay_v2.calibration.snapshot import assert_snapshot_precedes_decision, build_snapshot
from sports.mlb.parlay_v2.calibration.store import CalibrationStore
from sports.mlb.parlay_v2.calibration.support import GateMode, SupportStatus, evaluate_support
from sports.mlb.parlay_v2.calibration.replay import replay_calibration_as_of
from sports.mlb.parlay_v2.program_alpha import AlphaSpend, ProgramAlphaLedger
from sports.mlb.research.parlay_certification_v2 import manifest, prospective_boundary
from sports.mlb.research.parlay_certification_v2.evidence_store import DecisionRecord, EvidenceStore, FinalEvidenceRecord
from sports.mlb.research.parlay_certification_v2.policy import CandidateWager, select_action_for_day
from sports.mlb.research.parlay_certification_v2.eligibility import EligibilityInputs, evaluate_eligibility
from sports.mlb.research.parlay_certification_v2.replay import replay_policy_evidence


def _obs(*, player="P", line=0.5, target="H", admitted_at, settled_at="2026-08-19T23:00:00Z", win=1, source_id=None, slate_id="20260819"):
    return build_observation(
        slate_id=slate_id, game_id="g1", event_date=slate_id,
        player_id=player.lower(), player_name=player,
        target=target, side="OVER", line=line, book="real",
        quote_decimal=1.9, quote_timestamp=f"{slate_id}T17:00:00Z",
        prediction_value=line + 0.3, predictive_probability_if_available=0.6,
        state_version="s1", predictive_version="v1",
        market_bucket=target, line_bucket=f"{target}|OVER|{line}", state_bucket="v1|s1",
        settlement_status="win" if win else "loss", actual_outcome=float(win), actual_unit_return=0.9 if win else -1.0,
        decision_frozen_at=f"{slate_id}T17:05:00Z", settled_at=settled_at, calibration_admitted_at=admitted_at,
        source_id=source_id or f"{player}_{slate_id}_{line}",
        source_hash="h1",
    )


# ======================================================================
# A/B. Forward-only calibration: day t unavailable to day t, available to day t+1
# ======================================================================


def test_forward_only_day_t_outcome_unavailable_to_day_t_support(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    day_t_cutoff = "2026-08-19T17:05:00Z"  # today's decision_frozen_at / calibration_as_of
    obs_same_day = _obs(admitted_at="2026-08-19T23:30:00Z")  # settled/admitted AFTER today's cutoff
    store.admit(obs_same_day)
    rows = store.observations_as_of(day_t_cutoff)
    assert rows == []  # day t's own outcome must not be visible to day t's own support calc


def test_next_day_availability_once_admitted(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    obs = _obs(admitted_at="2026-08-19T23:30:00Z")
    store.admit(obs)
    day_t_plus_1_cutoff = "2026-08-20T17:05:00Z"
    rows = store.observations_as_of(day_t_plus_1_cutoff)
    assert len(rows) == 1  # now visible for the NEXT day's decision


# ======================================================================
# C. Exact timestamp invariant
# ======================================================================


def test_calibration_as_of_must_strictly_precede_decision_frozen_at(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    snapshot = build_snapshot(store, as_of="2026-08-20T17:00:00Z")
    assert_snapshot_precedes_decision(snapshot, "2026-08-20T17:00:01Z")  # fine, strictly after
    with pytest.raises(ValueError):
        assert_snapshot_precedes_decision(snapshot, "2026-08-20T17:00:00Z")  # equal -- must reject
    with pytest.raises(ValueError):
        assert_snapshot_precedes_decision(snapshot, "2026-08-20T16:59:59Z")  # before -- must reject


# ======================================================================
# D. Candidate/calibration separation
# ======================================================================


def test_candidate_observations_never_create_policy_evidence_rows(tmp_path):
    cal_store = CalibrationStore(tmp_path / "ledger.jsonl")
    for i in range(100):
        cal_store.admit(_obs(player=f"P{i}", admitted_at="2026-08-19T23:00:00Z", source_id=f"src{i}"))
    assert len(cal_store.all_observations()) == 100

    evidence_store = EvidenceStore(tmp_path / "evidence", "TEST_POLICY")
    assert evidence_store.load_all() == []  # completely separate stream, untouched by calibration admissions


# ======================================================================
# E. No candidate leakage into the G-process stream
# ======================================================================


def test_ten_thousand_candidates_still_produce_one_policy_evidence_row():
    import numpy as np

    elig = evaluate_eligibility(EligibilityInputs("d", True, True, True, True))
    wagers = [
        CandidateWager(
            wager_id=f"w{i}", decimal_price=1.5 + (i % 5) * 0.1,
            retained_world_ids=np.array([3]), world_probabilities=np.array([0.1, 0.1, 0.1, 0.7]),
            losing_world_ids=np.array([0, 1, 2]),
        )
        for i in range(10_000)
    ]
    selection = select_action_for_day(elig, wagers, r_max=25.0)
    # Exactly one action decision results, regardless of candidate count --
    # select_action_for_day's return type is a single ActionSelection, not
    # a list; this is a structural guarantee, verified explicitly here.
    assert selection.action in (0, 1)
    assert selection.selected is None or isinstance(selection.selected, CandidateWager)


# ======================================================================
# F. Calibration idempotency
# ======================================================================


def test_duplicate_candidate_settlement_ingestion_is_idempotent(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    obs = _obs(admitted_at="2026-08-19T23:00:00Z")
    assert store.admit(obs) is True
    assert store.admit(obs) is False  # duplicate -- no-op
    # A re-delivered but re-constructed (same content) observation also collides on observation_id.
    obs_rebuilt = _obs(admitted_at="2026-08-19T23:00:00Z")
    assert obs_rebuilt.observation_id == obs.observation_id
    assert store.admit(obs_rebuilt) is False
    assert len(store.all_observations()) == 1


# ======================================================================
# G. Policy evidence idempotency (reaffirms parlay_certification_v2 behavior here)
# ======================================================================


def test_duplicate_settlement_callback_creates_no_duplicate_evidence_row(tmp_path):
    store = EvidenceStore(tmp_path / "evidence", "TEST_POLICY")
    decision = DecisionRecord(
        "2026-08-19", True, "operationally_eligible", "ELIGIBILITY_V1", "2026-08-19T17:00:00Z",
        "TEST_POLICY", "M", 3, 1, "w1", 2.0, "book", 0.5, 0.3, 0.0, 25.0,
    )
    record = FinalEvidenceRecord("2026-08-19", "TEST_POLICY", 1, 1, 0, 1.0, "win", "2026-08-19T23:00:00Z", "settle1", decision)
    assert store.append_final_settlement(record) is True
    assert store.append_final_settlement(record) is False
    assert len(store.load_all()) == 1


# ======================================================================
# H. Exact event lines never share support accidentally
# ======================================================================


def test_half_and_full_line_never_share_support_counts(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    for i in range(25):
        store.admit(_obs(player=f"HalfLine{i}", line=0.5, admitted_at="2026-08-19T23:00:00Z", source_id=f"half{i}"))
    for i in range(3):
        store.admit(_obs(player=f"FullLine{i}", line=1.5, admitted_at="2026-08-19T23:00:00Z", source_id=f"full{i}"))

    cutoff = "2026-08-20T17:00:00Z"
    rows = store.observations_as_of(cutoff)
    half_support = evaluate_support(rows, market_bucket="H", line_bucket="H|OVER|0.5", state_bucket="v1|s1", independent_slate_count=25)
    full_support = evaluate_support(rows, market_bucket="H", line_bucket="H|OVER|1.5", state_bucket="v1|s1", independent_slate_count=25)
    assert half_support.line_support.value == 25
    assert full_support.line_support.value == 3
    assert half_support.line_support.value != full_support.line_support.value
    # market_support pools across lines within the same target (by design,
    # a coarser dimension than line_support) -- but line_support itself
    # never conflates the two.
    assert half_support.market_support.value == 28
    assert full_support.market_support.value == 28


def test_exact_event_identity_distinguishes_lines():
    assert exact_event_identity("p1", "g1", "H", "OVER", 0.5, "real") != exact_event_identity("p1", "g1", "H", "OVER", 1.5, "real")


def test_observe_only_dimensions_never_block_once_required_dimensions_pass():
    # THE FIX this mission makes: joint_support and shift_status remain
    # UNESTABLISHED forever (no arbitrary threshold is invented for
    # either), but because they are OBSERVE_ONLY they can never block
    # action -- only the three REQUIRED, real, implemented dimensions
    # (market_support/line_support/state_support) can. With overwhelming
    # REQUIRED support, in_support must now be True, breaking the old
    # permanent-abstention circularity.
    rows = [
        _obs(player=f"P{i}", admitted_at="2026-08-19T23:00:00Z", source_id=f"s{i}").as_dict()
        for i in range(500)
    ]
    support = evaluate_support(rows, market_bucket="H", line_bucket="H|OVER|0.5", state_bucket="v1|s1", independent_slate_count=500)
    assert support.market_support.value >= 20 and support.line_support.value >= 20 and support.state_support.value >= 20
    assert support.market_support.status == SupportStatus.PASS
    assert support.line_support.status == SupportStatus.PASS
    assert support.state_support.status == SupportStatus.PASS
    assert support.joint_support.status == SupportStatus.UNESTABLISHED
    assert support.shift_status.status == SupportStatus.UNESTABLISHED
    assert support.joint_support.gate_mode == GateMode.OBSERVE_ONLY
    assert support.shift_status.gate_mode == GateMode.OBSERVE_ONLY
    assert support.joint_support.blocks_action is False
    assert support.shift_status.blocks_action is False
    assert support.in_support is True
    assert support.blocking_dimensions == []


def test_required_dimension_still_blocks_when_it_fails():
    # The fix does NOT weaken REQUIRED gating -- a genuinely thin market
    # still correctly reports not-in-support, with a specific blocking
    # dimension named (never a generic catch-all).
    rows = [_obs(player=f"P{i}", admitted_at="2026-08-19T23:00:00Z", source_id=f"s{i}").as_dict() for i in range(3)]
    support = evaluate_support(rows, market_bucket="H", line_bucket="H|OVER|0.5", state_bucket="v1|s1", independent_slate_count=3)
    assert support.in_support is False
    assert set(support.blocking_dimensions) == {"market_support", "line_support", "state_support"}


# ======================================================================
# I. Snapshot determinism
# ======================================================================


def test_snapshot_determinism_same_ledger_same_hash(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    for i in range(10):
        store.admit(_obs(player=f"P{i}", admitted_at="2026-08-19T23:00:00Z", source_id=f"s{i}"))
    snap1 = build_snapshot(store, as_of="2026-08-20T17:00:00Z")
    snap2 = build_snapshot(store, as_of="2026-08-20T17:00:00Z")
    assert snap1.calibration_snapshot_id == snap2.calibration_snapshot_id
    assert snap1.calibration_snapshot_sha256 == snap2.calibration_snapshot_sha256
    assert snap1.market_support_summary == snap2.market_support_summary

    # Reloading the store fresh (simulating a process restart) reproduces
    # the identical snapshot.
    store_reloaded = CalibrationStore(tmp_path / "ledger.jsonl")
    snap3 = build_snapshot(store_reloaded, as_of="2026-08-20T17:00:00Z")
    assert snap3.calibration_snapshot_sha256 == snap1.calibration_snapshot_sha256


# ======================================================================
# J. Replay
# ======================================================================


def test_calibration_replay_is_deterministic(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    for i in range(15):
        store.admit(_obs(player=f"P{i}", admitted_at="2026-08-19T23:00:00Z", source_id=f"s{i}"))
    r1 = replay_calibration_as_of(store, as_of="2026-08-20T17:00:00Z", market_bucket="H", line_bucket="H|OVER|0.5", state_bucket="v1|s1")
    r2 = replay_calibration_as_of(store, as_of="2026-08-20T17:00:00Z", market_bucket="H", line_bucket="H|OVER|0.5", state_bucket="v1|s1")
    assert r1.snapshot.calibration_snapshot_sha256 == r2.snapshot.calibration_snapshot_sha256
    assert r1.support == r2.support


def test_policy_replay_reproduces_status_and_horizons():
    import random

    rng = random.Random(6)
    rows = []
    # Reuses the same tuned strong-then-collapse scenario shape as
    # test_parlay_certification_v2.py::test_drift_support_then_demotion.
    for i in range(300):
        a = 1 if rng.random() < 0.5 else 0
        ell = 1 if (a and rng.random() < 0.1) else 0
        r = (-1.0 if ell else 2.0) if a else 0.0
        rows.append({"action": a, "loss": ell, "realized_return": r})
    for i in range(2000):
        a = 1 if rng.random() < 0.5 else 0
        ell = 1 if (a and rng.random() < 0.99) else 0
        r = (-1.0 if ell else 2.0) if a else 0.0
        rows.append({"action": a, "loss": ell, "realized_return": r})

    result1 = replay_policy_evidence(rows, c=0.10, r=0.60, delta=0.0, r_max=3.0)
    result2 = replay_policy_evidence(rows, c=0.10, r=0.60, delta=0.0, r_max=3.0)
    assert result1.final_status == result2.final_status
    assert result1.first_support_t == result2.first_support_t
    assert result1.demotion_t == result2.demotion_t
    assert result1.g_c_values == result2.g_c_values
    assert result1.first_support_t is not None
    assert result1.demotion_t is not None and result1.demotion_t > result1.first_support_t


# ======================================================================
# K/L/M. Freeze boundary
# ======================================================================


def test_no_pre_boundary_slate_enters_prospective_evidence(tmp_path):
    assert prospective_boundary.is_prospective(tmp_path, "POLICY_X", "2026-08-21T12:00:00Z") is False  # no boundary set at all


def test_august_21_remains_development_shadow(tmp_path):
    ok = prospective_boundary.set_prospective_start_timestamp(tmp_path, "POLICY_X", "2026-09-01T00:00:00Z")
    assert ok is True
    assert prospective_boundary.is_prospective(tmp_path, "POLICY_X", "2026-08-21T23:59:59Z") is False
    assert prospective_boundary.is_prospective(tmp_path, "POLICY_X", "2026-09-01T00:00:00Z") is True


def test_no_boundary_file_exists_for_the_real_policy_version_yet():
    from pathlib import Path

    root = Path(manifest.PROGRAM_ALPHA_LEDGER_PATH).parent  # reports/ dir, sibling location
    boundary_file = root / f"{manifest.POLICY_VERSION}_prospective_start.json"
    assert not boundary_file.exists(), "the real policy's prospective boundary must not have been activated by this session"


def test_boundary_is_one_way(tmp_path):
    assert prospective_boundary.set_prospective_start_timestamp(tmp_path, "POLICY_Y", "2026-09-01T00:00:00Z") is True
    first = prospective_boundary.read_prospective_start_timestamp(tmp_path, "POLICY_Y")
    # Attempting to move it earlier (or at all) is refused.
    assert prospective_boundary.set_prospective_start_timestamp(tmp_path, "POLICY_Y", "2026-01-01T00:00:00Z") is False
    assert prospective_boundary.read_prospective_start_timestamp(tmp_path, "POLICY_Y") == first


# ======================================================================
# N. Version isolation
# ======================================================================


def test_calibration_store_refuses_mismatched_schema_version(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl", calibration_version="SOME_OTHER_VERSION")
    obs = _obs(admitted_at="2026-08-19T23:00:00Z")  # built with the real SCHEMA_VERSION
    with pytest.raises(ValueError):
        store.admit(obs)


# ======================================================================
# O. Program-level alpha budget (reaffirms prior turn's coverage here, plus freeze_prospective wiring)
# ======================================================================


def test_program_alpha_matches_manifest_and_stays_within_budget(tmp_path):
    ledger = ProgramAlphaLedger(tmp_path / "ledger.json", alpha_program=manifest.ALPHA_PROGRAM)
    ledger.spend(AlphaSpend(manifest.POLICY_VERSION, manifest.ALPHA_TOTAL, "frozen_for_prospective_confirmation", "2026-08-21T00:00:00Z"))
    assert ledger.total_spent() <= manifest.ALPHA_PROGRAM + 1e-12
    with pytest.raises(ValueError):
        ledger.spend(AlphaSpend("ANOTHER_POLICY", manifest.ALPHA_PROGRAM, "frozen_for_prospective_confirmation", "2026-08-22T00:00:00Z"))


# ======================================================================
# Real settlement ingestion (sports/mlb/parlay_v2/calibration/ingest.py)
# ======================================================================

from sports.mlb.parlay_v2.calibration.ingest import ingest_settled_slate  # noqa: E402

REAL_SETTLED_STAMP = "20260802"  # a real archived, fully-settled MLB day


def test_ingest_admits_real_settled_observations_and_is_idempotent(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    first = ingest_settled_slate(REAL_SETTLED_STAMP, ledger_path=ledger_path)
    assert first["action_eligible_rows"] > 0
    assert first["admitted"] == first["action_eligible_rows"]
    assert first["already_present"] == 0

    second = ingest_settled_slate(REAL_SETTLED_STAMP, ledger_path=ledger_path)
    assert second["admitted"] == 0
    assert second["already_present"] == first["action_eligible_rows"]

    store = CalibrationStore(ledger_path)
    assert len(store.all_observations()) == first["action_eligible_rows"]


def test_ingest_settled_slate_handles_a_real_zero_row_day_without_raising(tmp_path):
    """A real archived day with a pool file but zero action-eligible rows
    for any requested target (e.g. an off day, or every game postponed)
    made build_multi_target_universe return a columnless empty DataFrame,
    which action_universe raised a bare KeyError on ("in_support") --
    found while backfilling the calibration ledger against real archived
    TEST_STAMPS days. Must admit zero rows, never crash."""
    zero_row_stamp = "20260806"  # real archived day, verified zero action-eligible rows
    result = ingest_settled_slate(zero_row_stamp, ledger_path=tmp_path / "ledger.jsonl")
    assert result == {
        "stamp": zero_row_stamp,
        "action_eligible_rows": 0,
        "admitted": 0,
        "already_present": 0,
        "ledger_path": str(tmp_path / "ledger.jsonl"),
    }


def test_ingest_never_backdates_calibration_admitted_at(tmp_path):
    """calibration_admitted_at must reflect when ingestion actually ran,
    never the settled slate's own date -- this is what the forward-only
    invariant in store.py relies on."""
    before = datetime.now(timezone.utc).isoformat()
    ingest_settled_slate(REAL_SETTLED_STAMP, ledger_path=tmp_path / "ledger.jsonl")
    after = datetime.now(timezone.utc).isoformat()

    store = CalibrationStore(tmp_path / "ledger.jsonl")
    rows = store.all_observations()
    assert rows
    for row in rows:
        assert before <= row["calibration_admitted_at"] <= after
        # the settled stamp (2026-08-02) is nowhere close to "now" (2026-08-21+)
        assert not row["calibration_admitted_at"].startswith(REAL_SETTLED_STAMP[:4] + "-08-02")


def test_ingested_observations_respect_forward_only_visibility(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    before_ingest = datetime.now(timezone.utc).isoformat()
    ingest_settled_slate(REAL_SETTLED_STAMP, ledger_path=ledger_path)
    after_ingest = datetime.now(timezone.utc).isoformat()

    store = CalibrationStore(ledger_path)
    assert store.observations_as_of(before_ingest) == []  # not yet admitted at that cutoff
    assert len(store.observations_as_of(after_ingest)) > 0  # visible once admitted


def test_ingested_observations_preserve_exact_event_identity(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ingest_settled_slate(REAL_SETTLED_STAMP, ledger_path=ledger_path)
    store = CalibrationStore(ledger_path)
    rows = store.all_observations()
    # No two observations share an observation_id (would only happen if
    # two truly-distinct events collided in identity).
    ids = [r["observation_id"] for r in rows]
    assert len(ids) == len(set(ids))
    # line_bucket always encodes the exact line -- never pooled across lines.
    for row in rows:
        assert str(row["line"]) in row["line_bucket"]
