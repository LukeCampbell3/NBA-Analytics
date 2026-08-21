from __future__ import annotations

import itertools
import json
import random
from dataclasses import fields

import numpy as np
import pytest

from sports.mlb.conditional_chain.outcome_worlds import world_id_from_outcomes
from sports.mlb.research.parlay_certification_v2 import manifest
from sports.mlb.research.parlay_certification_v2.anytime_monitor import (
    AlphaAllocation,
    anytime_bound,
    default_equal_split,
    evaluate_simultaneous_certificate,
    g_c_bounds,
    g_c_value,
    g_l_bounds,
    g_l_value,
    g_v_bounds,
    g_v_value,
)
from sports.mlb.research.parlay_certification_v2.eligibility import (
    EligibilityDecision,
    EligibilityInputs,
    evaluate_eligibility,
)
from sports.mlb.research.parlay_certification_v2.evidence_store import (
    DecisionRecord,
    EvidenceStore,
    FinalEvidenceRecord,
)
from sports.mlb.research.parlay_certification_v2.policy import (
    CandidateWager,
    build_decision_record,
    select_action_for_day,
)
from sports.mlb.research.parlay_certification_v2.settlement import (
    SettlementInput,
    SettlementStatus,
    is_loss,
    reject_if_price_exceeds_bound,
    resolve_return,
)
from sports.mlb.research.parlay_certification_v2.state_machine import PolicyStatus, next_status
from sports.mlb.research.parlay_certification_v2.world_certificate import (
    build_nonvacuous_world_certificate,
    naive_vacuous_rule_certified,
    world_coverage_loss_bound,
)

# ======================================================================
# A. Three-process algebra -- >=10,000 randomized cases
# ======================================================================


def test_three_process_algebra_randomized_10000_cases():
    rng = random.Random(20260821)
    n_cases = 10_000
    checked_loss = 0
    checked_value = 0
    for _ in range(n_cases):
        n = rng.randint(5, 40)
        c = rng.uniform(0.05, 0.95)
        r = rng.uniform(0.05, 0.95)
        delta = rng.uniform(-0.5, 0.5)
        a = [rng.randint(0, 1) for _ in range(n)]
        ell = [rng.randint(0, 1) for _ in range(n)]
        rt = [rng.uniform(-1.0, 3.0) for _ in range(n)]

        g_c = [g_c_value(a[t], c) for t in range(n)]
        g_l = [g_l_value(a[t], ell[t], r) for t in range(n)]
        g_v = [g_v_value(a[t], rt[t], delta) for t in range(n)]

        mean_a = sum(a) / n
        # coverage >= c iff mean(G_C) >= 0
        assert (mean_a >= c) == (sum(g_c) / n >= -1e-12)

        n_actions = sum(a)
        if n_actions > 0:
            checked_loss += 1
            checked_value += 1
            loss_risk = sum(a[t] * ell[t] for t in range(n)) / n_actions
            # loss risk <= r iff mean[A(ell-r)] <= 0
            assert (loss_risk <= r + 1e-12) == (sum(g_l) <= 1e-9)
            value_per_action = sum(a[t] * rt[t] for t in range(n)) / n_actions
            # return/action >= delta iff mean[A(R-delta)] >= 0
            assert (value_per_action >= delta - 1e-12) == (sum(g_v) >= -1e-9)
    assert checked_loss > 5000 and checked_value > 5000  # sanity: most trials had at least one action


def test_g_process_bounds_contain_all_values():
    rng = random.Random(7)
    for _ in range(2000):
        c = rng.uniform(0.01, 0.99)
        r = rng.uniform(0.01, 0.99)
        delta = rng.uniform(-2.0, 2.0)
        r_max = rng.uniform(1.0, 30.0)
        a = rng.randint(0, 1)
        ell = rng.randint(0, 1)
        rt = rng.uniform(-1.0, r_max)
        gc, gl, gv = g_c_value(a, c), g_l_value(a, ell, r), g_v_value(a, rt, delta)
        bc, bl, bv = g_c_bounds(c), g_l_bounds(r), g_v_bounds(delta, r_max)
        assert bc.low - 1e-9 <= gc <= bc.high + 1e-9
        assert bl.low - 1e-9 <= gl <= bl.high + 1e-9
        assert bv.low - 1e-9 <= gv <= bv.high + 1e-9


# ======================================================================
# B. World theorem -- exhaustive small binary universes
# ======================================================================


def test_world_theorem_exhaustive_2leg_universe():
    """4 worlds (2 legs): 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1). Position S =
    'both legs win' loses in worlds {0,1,2}, wins only in world 3. For
    every one of the 16 possible retained subsets C of {0,1,2,3}: B_S(C)
    empty iff every retained world is nonlosing for S (i.e. C subset of
    {3})."""
    all_worlds = [world_id_from_outcomes([w0, w1]) for w0, w1 in itertools.product([0, 1], repeat=2)]
    assert sorted(all_worlds) == [0, 1, 2, 3]
    losing_worlds = np.array([0, 1, 2])  # every world except "both win" (id 3)
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])

    for size in range(0, 5):
        for subset in itertools.combinations(all_worlds, size):
            retained = np.array(subset, dtype=int)
            cert = build_nonvacuous_world_certificate(retained, uniform_probs, losing_worlds)
            expected_zero_counterexamples = all(w not in {0, 1, 2} for w in subset)
            assert cert.zero_loss_counterexamples == expected_zero_counterexamples
            # Full logical certificate additionally requires nonempty + positive mass.
            expected_certified = expected_zero_counterexamples and size > 0
            assert cert.certified == expected_certified


# ======================================================================
# C. Vacuous certificate -- C=empty / zero mass MUST NOT certify
# ======================================================================


def test_empty_world_set_does_not_certify():
    cert = build_nonvacuous_world_certificate(np.array([], dtype=int), np.array([0.25, 0.25, 0.25, 0.25]), np.array([0, 1, 2]))
    assert cert.nonempty is False
    assert cert.certified is False


def test_zero_probability_mass_retained_world_does_not_certify():
    # retained_count > 0 (world id 3 IS in the retained set) but its
    # probability mass is exactly zero -- must still not certify.
    probs = np.array([0.5, 0.5, 0.0, 0.0])  # world 3 (index 3) has zero mass
    cert = build_nonvacuous_world_certificate(np.array([3]), probs, np.array([0, 1, 2]))
    assert cert.nonempty is True
    assert cert.positive_mass is False
    assert cert.certified is False


def test_naive_rule_certifies_empty_set_but_v2_refuses():
    retained = np.array([], dtype=int)
    losing = np.array([0, 1, 2])
    # THE BUG: the naive rule only checks B_S(C)=empty, which is vacuously
    # true for an empty C.
    assert naive_vacuous_rule_certified(retained, losing) is True
    # V2 refuses.
    cert = build_nonvacuous_world_certificate(retained, np.array([0.25, 0.25, 0.25, 0.25]), losing)
    assert cert.certified is False


def test_nonempty_valid_certificate_still_works():
    probs = np.array([0.1, 0.1, 0.1, 0.7])
    cert = build_nonvacuous_world_certificate(np.array([3]), probs, np.array([0, 1, 2]))
    assert cert.certified is True


def test_world_coverage_bridge_theorem():
    # L <= min(1, alpha_world / c)
    assert world_coverage_loss_bound(alpha_world=0.05, c=0.5) == pytest.approx(0.10)
    assert world_coverage_loss_bound(alpha_world=0.9, c=0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        world_coverage_loss_bound(alpha_world=0.05, c=0.0)


# ======================================================================
# D. Eligibility attack -- 100 eligible days, 50 actions -> coverage stays .50
# ======================================================================


def test_eligibility_attack_coverage_cannot_be_redefined_by_abstentions():
    eligible_inputs = [
        EligibilityInputs(date=f"day{i}", required_feed_available=True, slate_has_mlb_games=True,
                           required_system_component_available=True, decision_cutoff_met=True)
        for i in range(100)
    ]
    decisions = [evaluate_eligibility(inp) for inp in eligible_inputs]
    assert all(d.eligible for d in decisions)

    # 50 of the 100 eligible days take an action; 50 abstain. Coverage is
    # computed purely from eligibility.eligible AND policy action, with no
    # path for an abstention to flip `eligible`.
    actions = [1] * 50 + [0] * 50
    coverage = sum(actions) / sum(1 for d in decisions if d.eligible)
    assert coverage == pytest.approx(0.50)

    # It is structurally impossible to construct an EligibilityDecision
    # whose `eligible` field depends on `actions`: evaluate_eligibility's
    # only argument type (EligibilityInputs) has no action/outcome field.
    input_field_names = {f.name for f in fields(EligibilityInputs)}
    assert input_field_names == {
        "date", "required_feed_available", "slate_has_mlb_games",
        "required_system_component_available", "decision_cutoff_met",
    }


def test_abstention_reasons_remain_eligible_true():
    """slate exists but no pair passes / prices unattractive / low
    confidence / all candidates fail thresholds -- these must all be E=1,
    A=0, never E=0. Modeled here via select_action_for_day returning
    action=0 while eligibility.eligible stays True throughout."""
    elig = evaluate_eligibility(EligibilityInputs("d", True, True, True, True))
    assert elig.eligible is True
    selection = select_action_for_day(elig, candidates=[], r_max=25.0)
    assert selection.action == 0
    assert elig.eligible is True  # untouched by the policy decision


# ======================================================================
# E. Settlement
# ======================================================================


def test_settlement_normal_win_and_loss():
    r_win = resolve_return(SettlementInput(SettlementStatus.WIN, accepted_decimal_price=2.5), r_max=25.0)
    assert r_win == pytest.approx(1.5)
    assert is_loss(r_win) is False
    r_loss = resolve_return(SettlementInput(SettlementStatus.LOSS), r_max=25.0)
    assert r_loss == pytest.approx(-1.0)
    assert is_loss(r_loss) is True


@pytest.mark.parametrize("status", [SettlementStatus.PUSH, SettlementStatus.VOID, SettlementStatus.CANCELED])
def test_settlement_zero_return_statuses_are_not_losses(status):
    r = resolve_return(SettlementInput(status), r_max=25.0)
    assert r == 0.0
    assert is_loss(r) is False  # push/void/refund with R=0 is NOT a loss


def test_settlement_repriced_leg_void():
    r = resolve_return(SettlementInput(SettlementStatus.REPRICED_VOID, repriced_decimal_price=1.8), r_max=25.0)
    assert r == pytest.approx(0.8)


def test_settlement_win_requires_price_and_repriced_void_requires_repriced_price():
    with pytest.raises(ValueError):
        resolve_return(SettlementInput(SettlementStatus.WIN), r_max=25.0)
    with pytest.raises(ValueError):
        resolve_return(SettlementInput(SettlementStatus.REPRICED_VOID), r_max=25.0)


def test_settlement_duplicate_and_out_of_order_callbacks_are_idempotent(tmp_path):
    store = EvidenceStore(tmp_path, "TEST_POLICY_V0")
    decision = DecisionRecord(
        date="2026-08-21", eligible=True, eligibility_reason="operationally_eligible", eligibility_version="ELIGIBILITY_V1",
        decision_timestamp_utc="2026-08-21T17:00:00Z", policy_version="TEST_POLICY_V0", predictive_model_version="TEST_MODEL",
        candidate_universe_size=3, action=1, selected_wager="legA+legB", accepted_decimal_price=2.1, accepted_book="test_book",
        c=0.5, r=0.3, delta=0.0, r_max=25.0, world_certificate_diagnostics=None,
    )
    record = FinalEvidenceRecord(
        date="2026-08-21", policy_version="TEST_POLICY_V0", eligible=1, action=1, loss=0, realized_return=1.1,
        settlement_status="win", settlement_timestamp_utc="2026-08-21T23:10:00Z", source_id="game_824990_final",
        decision_record=decision,
    )
    assert store.append_final_settlement(record) is True
    # Duplicate settlement callback for the SAME event -- must be a no-op.
    assert store.append_final_settlement(record) is False
    # Out-of-order re-delivery of the same event (different object, same source_id).
    late_duplicate = FinalEvidenceRecord(
        date="2026-08-21", policy_version="TEST_POLICY_V0", eligible=1, action=1, loss=0, realized_return=1.1,
        settlement_status="win", settlement_timestamp_utc="2026-08-21T23:59:59Z", source_id="game_824990_final",
        decision_record=decision,
    )
    assert store.append_final_settlement(late_duplicate) is False
    rows = store.load_all()
    assert len(rows) == 1  # exactly one final slate evidence record


def test_settlement_canceled_game_and_two_distinct_days_both_recorded(tmp_path):
    store = EvidenceStore(tmp_path, "TEST_POLICY_V0")
    decision = DecisionRecord(
        date="2026-08-22", eligible=True, eligibility_reason="operationally_eligible", eligibility_version="ELIGIBILITY_V1",
        decision_timestamp_utc="2026-08-22T17:00:00Z", policy_version="TEST_POLICY_V0", predictive_model_version="TEST_MODEL",
        candidate_universe_size=0, action=0, selected_wager=None, accepted_decimal_price=None, accepted_book=None,
        c=0.5, r=0.3, delta=0.0, r_max=25.0,
    )
    record = FinalEvidenceRecord(
        date="2026-08-22", policy_version="TEST_POLICY_V0", eligible=1, action=0, loss=0, realized_return=0.0,
        settlement_status="canceled", settlement_timestamp_utc="2026-08-22T18:00:00Z", source_id="game_999_canceled",
        decision_record=decision,
    )
    assert store.append_final_settlement(record) is True
    assert len(store.load_all()) == 1  # fresh tmp_path per test -- exactly this one record


# ======================================================================
# F. Price bound
# ======================================================================


def test_price_bound_rejects_actions_outside_r_max():
    with pytest.raises(ValueError):
        reject_if_price_exceeds_bound(30.0, r_max=25.0)  # implies R=29 > R_max=25
    reject_if_price_exceeds_bound(20.0, r_max=25.0)  # implies R=19 <= 25, no raise

    with pytest.raises(ValueError):
        resolve_return(SettlementInput(SettlementStatus.WIN, accepted_decimal_price=50.0), r_max=25.0)


def test_policy_excludes_over_bound_candidate_but_still_acts_on_others():
    elig = evaluate_eligibility(EligibilityInputs("d", True, True, True, True))
    worlds3 = np.array([3])
    losing = np.array([0, 1, 2])
    probs = np.array([0.1, 0.1, 0.1, 0.7])
    over_bound = CandidateWager("over_bound", decimal_price=50.0, retained_world_ids=worlds3, world_probabilities=probs, losing_world_ids=losing)
    valid = CandidateWager("valid_pick", decimal_price=2.0, retained_world_ids=worlds3, world_probabilities=probs, losing_world_ids=losing)
    selection = select_action_for_day(elig, [over_bound, valid], r_max=25.0)
    assert selection.action == 1
    assert selection.selected.wager_id == "valid_pick"


# ======================================================================
# G. Missing quote -- E=1, A=0, never E=0
# ======================================================================


def test_missing_quote_stays_eligible_and_abstains():
    elig = evaluate_eligibility(EligibilityInputs("d", True, True, True, True))
    no_price_candidate = CandidateWager("no_quote", decimal_price=None, retained_world_ids=np.array([3]), world_probabilities=np.array([0.1, 0.1, 0.1, 0.7]), losing_world_ids=np.array([0, 1, 2]))
    selection = select_action_for_day(elig, [no_price_candidate], r_max=25.0)
    assert elig.eligible is True
    assert selection.action == 0
    assert selection.reason == "no_certified_candidate"


# ======================================================================
# H. Simultaneous alpha
# ======================================================================


def test_alpha_allocation_must_not_exceed_total():
    default_equal_split(0.05)  # fine
    AlphaAllocation(alpha_total=0.05, alpha_c=0.02, alpha_l=0.02, alpha_v=0.01)  # fine, sums to 0.05
    with pytest.raises(ValueError):
        AlphaAllocation(alpha_total=0.05, alpha_c=0.02, alpha_l=0.02, alpha_v=0.02)  # sums to 0.06 > 0.05


def test_manifest_alpha_allocation_is_valid():
    alloc = AlphaAllocation(alpha_total=manifest.ALPHA_TOTAL, alpha_c=manifest.ALPHA_C, alpha_l=manifest.ALPHA_L, alpha_v=manifest.ALPHA_V)
    assert alloc.alpha_c + alloc.alpha_l + alloc.alpha_v <= alloc.alpha_total + 1e-12


# ======================================================================
# I. Anytime null stress -- exactly one target violated, no spurious full cert
# ======================================================================


def _run_days(n, a_seq, ell_seq, r_seq):
    return list(a_seq[:n]), list(ell_seq[:n]), list(r_seq[:n])


def test_null_stress_coverage_violated_only():
    n = 200
    c, r, delta, r_max = 0.6, 0.5, 0.0, 25.0
    rng = random.Random(1)
    # Action rate ~0.3, well below c=0.6 -- coverage should NOT be supported.
    a = [1 if rng.random() < 0.30 else 0 for _ in range(n)]
    ell = [0 for _ in range(n)]  # never lose when acting -> loss risk trivially fine
    rvals = [0.5 if a[t] else 0.0 for t in range(n)]  # decent positive return when acting -> value fine
    g_c = [g_c_value(a[t], c) for t in range(n)]
    g_l = [g_l_value(a[t], ell[t], r) for t in range(n)]
    g_v = [g_v_value(a[t], rvals[t], delta) for t in range(n)]
    cert = evaluate_simultaneous_certificate(g_c, g_l, g_v, c=c, r=r, delta=delta, r_max=r_max, alpha_allocation=default_equal_split(0.05))
    assert cert.coverage_supported is False
    assert cert.fully_supported is False


def test_null_stress_loss_risk_violated_only():
    n = 200
    c, r, delta, r_max = 0.05, 0.10, 0.0, 25.0
    rng = random.Random(2)
    a = [1 for _ in range(n)]  # always act -> coverage trivially fine (>=c)
    ell = [1 if rng.random() < 0.9 else 0 for _ in range(n)]  # loses ~90% of the time, way above r=0.10
    rvals = [1.0 if not ell[t] else -1.0 for t in range(n)]
    g_c = [g_c_value(a[t], c) for t in range(n)]
    g_l = [g_l_value(a[t], ell[t], r) for t in range(n)]
    g_v = [g_v_value(a[t], rvals[t], delta) for t in range(n)]
    cert = evaluate_simultaneous_certificate(g_c, g_l, g_v, c=c, r=r, delta=delta, r_max=r_max, alpha_allocation=default_equal_split(0.05))
    assert cert.loss_supported is False
    assert cert.fully_supported is False


def test_null_stress_return_violated_only():
    n = 200
    c, r, delta, r_max = 0.05, 0.95, 0.5, 25.0
    a = [1 for _ in range(n)]
    ell = [0 for _ in range(n)]
    rvals = [0.05 for _ in range(n)]  # always positive, but well below delta=0.5
    g_c = [g_c_value(a[t], c) for t in range(n)]
    g_l = [g_l_value(a[t], ell[t], r) for t in range(n)]
    g_v = [g_v_value(a[t], rvals[t], delta) for t in range(n)]
    cert = evaluate_simultaneous_certificate(g_c, g_l, g_v, c=c, r=r, delta=delta, r_max=r_max, alpha_allocation=default_equal_split(0.05))
    assert cert.value_supported is False
    assert cert.fully_supported is False


# ======================================================================
# J. Strong alternative -- clearly qualifying synthetic process reaches support
# ======================================================================


def test_strong_alternative_reaches_full_support():
    n = 500
    c, r, delta, r_max = 0.10, 0.60, 0.0, 3.0
    rng = random.Random(3)
    # Acts ~40% of days (>> c=0.10), loses only ~15% when acting (<< r=0.60),
    # nets a clearly positive average return when acting (>> delta=0.0).
    # r_max=3.0 (not an unrealistically generous 25.0) keeps G_V's bounded
    # range tight enough for the conservative reference radius to resolve
    # at a moderate n -- a very wide R_max legitimately demands more data,
    # which is itself a real property of this conservative construction,
    # not a bug (see test_null_stress_return_violated_only for the
    # under-margin case).
    a = [1 if rng.random() < 0.40 else 0 for _ in range(n)]
    ell = [1 if (a[t] and rng.random() < 0.15) else 0 for t in range(n)]
    rvals = [(-1.0 if ell[t] else 2.2) if a[t] else 0.0 for t in range(n)]
    g_c = [g_c_value(a[t], c) for t in range(n)]
    g_l = [g_l_value(a[t], ell[t], r) for t in range(n)]
    g_v = [g_v_value(a[t], rvals[t], delta) for t in range(n)]
    cert = evaluate_simultaneous_certificate(g_c, g_l, g_v, c=c, r=r, delta=delta, r_max=r_max, alpha_allocation=default_equal_split(0.05))
    assert cert.fully_supported is True


# ======================================================================
# K. Drift -- strong early period, then severe degradation -> demotion
# ======================================================================


def test_drift_support_then_demotion():
    # Seed/margins chosen (see development notes) so support is reached
    # and stays stable through the ENTIRE good period (no boundary
    # flicker), and demotion only occurs well after the collapse begins --
    # a clean demonstration of "later cumulative evidence" causing
    # demotion, not early noise near the certification threshold. Because
    # this monitor's mean is CUMULATIVE from t=1 (not a rolling window,
    # per the sequential/predictable-mean, non-stationary design), a
    # regime collapse takes a while to drag the running average down --
    # that slowness is itself a real, correctly-conservative property of
    # this construction, not a bug, which is why n_bad is large here.
    c, r, delta, r_max = 0.10, 0.60, 0.0, 3.0
    rng = random.Random(6)
    n_good, n_bad = 300, 2000
    a_good = [1 if rng.random() < 0.50 else 0 for _ in range(n_good)]
    ell_good = [1 if (a_good[t] and rng.random() < 0.10) else 0 for t in range(n_good)]
    r_good = [(-1.0 if ell_good[t] else 2.0) if a_good[t] else 0.0 for t in range(n_good)]

    a_bad = [1 if rng.random() < 0.50 else 0 for _ in range(n_bad)]
    ell_bad = [1 if (a_bad[t] and rng.random() < 0.99) else 0 for t in range(n_bad)]  # collapses
    r_bad = [(-1.0 if ell_bad[t] else 2.0) if a_bad[t] else 0.0 for t in range(n_bad)]

    a = a_good + a_bad
    ell = ell_good + ell_bad
    rvals = r_good + r_bad

    status = PolicyStatus.DEVELOPMENT
    status = next_status(status, fully_supported=False, t=0).next
    assert status == PolicyStatus.FROZEN_PROSPECTIVE_INCONCLUSIVE

    first_support_t = None
    demotion_t = None
    g_c_cum: list[float] = []
    g_l_cum: list[float] = []
    g_v_cum: list[float] = []
    alloc = default_equal_split(0.05)
    for t in range(1, len(a) + 1):
        g_c_cum.append(g_c_value(a[t - 1], c))
        g_l_cum.append(g_l_value(a[t - 1], ell[t - 1], r))
        g_v_cum.append(g_v_value(a[t - 1], rvals[t - 1], delta))
        cert = evaluate_simultaneous_certificate(g_c_cum, g_l_cum, g_v_cum, c=c, r=r, delta=delta, r_max=r_max, alpha_allocation=alloc)
        transition = next_status(status, fully_supported=cert.fully_supported, t=t)
        status = transition.next
        if status == PolicyStatus.FROZEN_POLICY_PROSPECTIVELY_SUPPORTED and first_support_t is None:
            first_support_t = t
            status = next_status(status, fully_supported=cert.fully_supported, t=t).next  # -> SUPPORTED_CURRENT
        if status == PolicyStatus.PRODUCTION_DEMOTED and demotion_t is None:
            demotion_t = t

    assert first_support_t is not None, "strong early period should reach support at some horizon"
    assert demotion_t is not None, "severe degradation should eventually demote"
    assert demotion_t > first_support_t
    assert demotion_t > n_good, "demotion should be attributable to the collapse period, not early boundary noise"
    assert status == PolicyStatus.PRODUCTION_DEMOTED
    # Never auto re-promotes.
    final = next_status(status, fully_supported=True, t=len(a) + 1)
    assert final.next == PolicyStatus.PRODUCTION_DEMOTED


# ======================================================================
# L. No leakage
# ======================================================================


def test_eligibility_inputs_schema_has_no_outcome_or_model_fields():
    names = {f.name for f in fields(EligibilityInputs)}
    forbidden_substrings = ("win", "loss", "prob", "score", "edge", "confidence", "result", "settle", "return", "pair", "candidate")
    for name in names:
        low = name.lower()
        assert not any(bad in low for bad in forbidden_substrings), f"EligibilityInputs.{name} looks outcome/model-dependent"


def test_candidate_wager_schema_has_no_settlement_outcome_field():
    names = {f.name for f in fields(CandidateWager)}
    forbidden_substrings = ("win", "loss", "settle", "actual", "result")
    for name in names:
        low = name.lower()
        assert not any(bad in low for bad in forbidden_substrings), f"CandidateWager.{name} looks settlement-dependent"


def test_decision_record_has_no_settlement_fields():
    """DecisionRecord is frozen at decision cutoff, BEFORE settlement --
    it must not carry any settlement/outcome field (those belong only to
    FinalEvidenceRecord, appended strictly after)."""
    names = {f.name for f in fields(DecisionRecord)}
    forbidden = {"loss", "realized_return", "settlement_status", "settlement_timestamp_utc"}
    assert names.isdisjoint(forbidden)


# ======================================================================
# M. Version isolation
# ======================================================================


def test_version_isolation_rejects_mismatched_policy_version(tmp_path):
    store = EvidenceStore(tmp_path, "POLICY_A")
    decision = DecisionRecord(
        date="2026-08-21", eligible=True, eligibility_reason="operationally_eligible", eligibility_version="ELIGIBILITY_V1",
        decision_timestamp_utc="2026-08-21T17:00:00Z", policy_version="POLICY_B", predictive_model_version="TEST_MODEL",
        candidate_universe_size=1, action=0, selected_wager=None, accepted_decimal_price=None, accepted_book=None,
        c=0.5, r=0.3, delta=0.0, r_max=25.0,
    )
    record = FinalEvidenceRecord(
        date="2026-08-21", policy_version="POLICY_B", eligible=1, action=0, loss=0, realized_return=0.0,
        settlement_status="void", settlement_timestamp_utc="2026-08-21T23:00:00Z", source_id="x1", decision_record=decision,
    )
    with pytest.raises(ValueError):
        store.append_final_settlement(record)  # POLICY_B record into a POLICY_A store


def test_version_isolation_separate_files_never_pooled(tmp_path):
    store_a = EvidenceStore(tmp_path, "POLICY_A")
    store_b = EvidenceStore(tmp_path, "POLICY_B")
    assert store_a.path != store_b.path
    decision_a = DecisionRecord("d", True, "operationally_eligible", "ELIGIBILITY_V1", "t", "POLICY_A", "M", 1, 0, None, None, None, 0.5, 0.3, 0.0, 25.0)
    store_a.append_final_settlement(FinalEvidenceRecord("d", "POLICY_A", 1, 0, 0, 0.0, "void", "t", "a1", decision_a))
    assert len(store_a.load_all()) == 1
    assert len(store_b.load_all()) == 0  # nothing leaked across the version boundary


# ======================================================================
# N. Restart / replay
# ======================================================================


def test_replay_is_deterministic(tmp_path):
    c, r, delta, r_max = 0.10, 0.60, 0.0, 25.0
    rng = random.Random(99)
    n = 60
    store = EvidenceStore(tmp_path, "POLICY_REPLAY")
    for t in range(n):
        a_t = 1 if rng.random() < 0.5 else 0
        ell_t = 1 if (a_t and rng.random() < 0.3) else 0
        r_t = (-1.0 if ell_t else 1.4) if a_t else 0.0
        decision = DecisionRecord(f"d{t}", True, "operationally_eligible", "ELIGIBILITY_V1", f"t{t}", "POLICY_REPLAY", "M", 5, a_t, "w" if a_t else None, 2.4 if a_t else None, "book" if a_t else None, c, r, delta, r_max)
        store.append_final_settlement(FinalEvidenceRecord(f"d{t}", "POLICY_REPLAY", 1, a_t, ell_t, r_t, "win" if (a_t and not ell_t) else ("loss" if ell_t else "void"), f"s{t}", f"src{t}", decision))

    def cumulative_g_and_bounds(rows):
        g_c_vals = [g_c_value(row["action"], c) for row in rows]
        g_l_vals = [g_l_value(row["action"], row["loss"], r) for row in rows]
        g_v_vals = [g_v_value(row["action"], row["realized_return"], delta) for row in rows]
        cert = evaluate_simultaneous_certificate(g_c_vals, g_l_vals, g_v_vals, c=c, r=r, delta=delta, r_max=r_max, alpha_allocation=default_equal_split(0.05))
        return g_c_vals, g_l_vals, g_v_vals, cert

    rows_first = store.load_all()
    g_c1, g_l1, g_v1, cert1 = cumulative_g_and_bounds(rows_first)

    # "Restart": fresh EvidenceStore instance over the same on-disk file.
    store_reloaded = EvidenceStore(tmp_path, "POLICY_REPLAY")
    rows_second = store_reloaded.load_all()
    g_c2, g_l2, g_v2, cert2 = cumulative_g_and_bounds(rows_second)

    assert rows_first == rows_second
    assert g_c1 == g_c2 and g_l1 == g_l2 and g_v1 == g_v2
    assert cert1.coverage_bound == cert2.coverage_bound
    assert cert1.loss_bound == cert2.loss_bound
    assert cert1.value_bound == cert2.value_bound
    assert cert1.fully_supported == cert2.fully_supported


# ======================================================================
# Manifest / production-authorization invariants
# ======================================================================


def test_production_authorized_is_false_and_never_settable_programmatically():
    assert manifest.PRODUCTION_AUTHORIZED is False


def test_manifest_status_reflects_a_deliberate_freeze_never_production():
    """STATUS was deliberately advanced to FROZEN_PROSPECTIVE_INCONCLUSIVE
    (the mission that fixed the circular support-gate bug -- see
    manifest.CONCLUSION_REASONING for the three-step freeze this
    accompanied). The invariant this test guards has not changed: STATUS
    must never silently reach a production-authorizing value on its own,
    and PRODUCTION_AUTHORIZED must stay False regardless of STATUS."""
    assert manifest.STATUS == "FROZEN_PROSPECTIVE_INCONCLUSIVE"
    assert manifest.STATUS not in ("SUPPORTED_CURRENT", "PRODUCTION_AUTHORIZED")
    assert manifest.PRODUCTION_AUTHORIZED is False


def test_no_ast_assignment_sets_production_authorized_true():
    """Static guard (mirrors the convention already used by
    joint_position_builder_v2/h_over_ranker): no source file in this
    package may contain the literal token sequence that would set
    PRODUCTION_AUTHORIZED to a truthy value."""
    import ast
    from pathlib import Path

    pkg_root = Path(__file__).resolve().parents[1] / "research" / "parlay_certification_v2"
    for py_file in pkg_root.glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "PRODUCTION_AUTHORIZED" in targets:
                    assert isinstance(node.value, ast.Constant) and node.value.value is False, (
                        f"{py_file}: PRODUCTION_AUTHORIZED must only ever be assigned the literal False"
                    )
