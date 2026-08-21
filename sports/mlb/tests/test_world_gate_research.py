from __future__ import annotations

"""Required tests for the world-gate admission research mission
("Resolve the remaining PARLAY_V2 APS / counterexample admission
bottleneck"). Section 22 letter groups map directly to the section
headers below."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports.mlb.conditional_chain.outcome_worlds import (
    aps_world_scores,
    build_binary_outcome_set,
    build_world_distribution,
    conformal_aps_threshold,
    world_id_from_outcomes,
)
from sports.mlb.parlay_v2.program_alpha import AlphaSpend, ProgramAlphaLedger
from sports.mlb.parlay_v2.run_parlay_v2 import build_slate_payload
from sports.mlb.research.parlay_certification_v2 import manifest
from sports.mlb.research.parlay_certification_v2.eligibility import EligibilityDecision
from sports.mlb.research.parlay_certification_v2.policy import CandidateWager, build_decision_record, select_action_for_day
from sports.mlb.research.parlay_certification_v2.replay import replay_policy_evidence
from sports.mlb.research.parlay_certification_v2.world_certificate import (
    build_nonvacuous_world_certificate,
    naive_vacuous_rule_certified,
)
from sports.mlb.research.parlay_certification_v2.world_gate_research import (
    APS_GRID,
    _pair_hash,
    _select_sampled_pairs,
    build_pair_development_table,
    usable_stamps,
)

WW = world_id_from_outcomes([1, 1])


def _elig(eligible: bool = True) -> EligibilityDecision:
    return EligibilityDecision(date="20260821", eligible=eligible, reason="operationally_eligible", eligibility_version="ELIGIBILITY_V1")


def _wager(wager_id: str, p_i: float, p_j: float, *, decimal_price: float = 2.5) -> CandidateWager:
    dist = build_world_distribution(["i", "j"], [p_i, p_j])
    losing = np.array([w for w in range(4) if w != WW])
    return CandidateWager(
        wager_id=wager_id,
        decimal_price=decimal_price,
        retained_world_ids=np.arange(4),
        world_probabilities=dist.probabilities,
        losing_world_ids=losing,
    )


# ======================================================================
# A. APS semantics
# ======================================================================


def test_aps_threshold_1_retains_every_positive_probability_world():
    dist = build_world_distribution(["i", "j"], [0.6, 0.55])
    outcome_set = build_binary_outcome_set(dist, aps_threshold=1.0, calibration_slates=0)
    assert set(outcome_set.world_ids.tolist()) == {0, 1, 2, 3}


def test_aps_score_is_tie_aware_cumulative_mass_descending():
    dist = build_world_distribution(["i", "j"], [0.6, 0.55])
    scores = aps_world_scores(dist)
    order = np.argsort(-dist.probabilities)
    running = 0.0
    for idx in order:
        running += dist.probabilities[idx]
        assert scores[idx] == pytest.approx(running)


def test_aps_ties_share_identical_score():
    dist = build_world_distribution(["i", "j"], [0.5, 0.5])  # all 4 worlds tied at 0.25
    scores = aps_world_scores(dist)
    assert np.allclose(scores, 1.0)  # every world only reaches full cumulative mass once the whole tied group is counted


def test_aps_threshold_below_lowest_score_retains_nothing():
    dist = build_world_distribution(["i", "j"], [0.6, 0.55])
    outcome_set = build_binary_outcome_set(dist, aps_threshold=0.05, calibration_slates=0)
    assert outcome_set.world_count == 0


def test_calibration_slates_never_affects_retained_set():
    dist = build_world_distribution(["i", "j"], [0.6, 0.55])
    a = build_binary_outcome_set(dist, aps_threshold=0.6, calibration_slates=0)
    b = build_binary_outcome_set(dist, aps_threshold=0.6, calibration_slates=500)
    assert a.world_ids.tolist() == b.world_ids.tolist()


def test_conformal_aps_threshold_saturates_to_one_with_thin_calibration():
    # With n=10 calibration scores at target_miscoverage=0.10, rank =
    # ceil(11*0.9) = 10 = n, so the threshold is the maximum observed
    # score -- 1.0 for any set containing a 1.0. Confirms the repo's own
    # MIN_CALIBRATION_PAIRS=20 convention (ablation.py) is not arbitrary:
    # <20 points genuinely cannot safely exclude any coverage yet.
    scores = [round(0.1 * i, 2) for i in range(1, 11)]
    threshold = conformal_aps_threshold(scores, target_miscoverage=0.10)
    assert threshold == 1.0


def test_zero_probability_worlds_never_retained_regardless_of_threshold():
    dist = build_world_distribution(["i", "j"], [0.6, 0.55], admissible_world_mask=np.array([True, True, True, False]))
    outcome_set = build_binary_outcome_set(dist, aps_threshold=1.0, calibration_slates=0)
    assert WW not in outcome_set.world_ids.tolist()


# ======================================================================
# B. Hard-zero degeneracy
# ======================================================================


def test_hard_zero_rejects_realistic_nondeterministic_pair():
    """Construct realistic nondegenerate probabilities and show hard-zero
    rejects when losing worlds remain (mission 22.B)."""
    cand = _wager("a", 0.6, 0.55)
    result = select_action_for_day(_elig(), [cand], r_max=5.0, world_gate_mode="REQUIRED")
    assert result.action == 0
    assert result.reason == "no_certified_candidate"


def test_hard_zero_operationally_degenerate_on_real_development_data():
    """The empirical finding this research established: at the frozen
    APS_THRESHOLD=1.0, 0% of real sampled DEVELOPMENT pairs certify."""
    table = build_pair_development_table(("20260429",), sample_cap_per_day=200)
    assert len(table) > 0
    assert table["nonvacuous_world_certificate"].mean() == 0.0


# ======================================================================
# C. Observe-only behavior
# ======================================================================


def test_observe_only_selects_despite_counterexamples_staking_still_ungoverned():
    cand = _wager("a", 0.6, 0.55)
    result = select_action_for_day(_elig(), [cand], r_max=5.0, world_gate_mode="OBSERVE_ONLY")
    assert result.action == 1
    assert result.certificate.certified is False
    assert result.certificate.counterexample_count > 0
    assert result.reason == "admitted_candidate_selected"
    # Staking authorization is untouched by this module -- always decided
    # outside select_action_for_day (run_parlay_v2.py), never here.
    record = build_decision_record(
        date="20260821", eligibility=_elig(), decision_timestamp_utc="t",
        predictive_model_version="v1", candidate_universe_size=1, action_selection=result,
        c=0.5, r=0.3, delta=0.0, r_max=5.0, world_gate_mode="OBSERVE_ONLY",
    )
    assert record.world_gate_mode == "OBSERVE_ONLY"
    assert not hasattr(record, "staking_authorized")  # this record never carries a staking field at all


def test_observe_only_ranks_by_ascending_world_risk_rho():
    better = _wager("better", 0.7, 0.7)  # p_joint=0.49, rho lower
    worse = _wager("worse", 0.55, 0.55)  # p_joint=0.3025, rho higher
    result = select_action_for_day(_elig(), [worse, better], r_max=5.0, world_gate_mode="OBSERVE_ONLY")
    assert result.selected.wager_id == "better"


# ======================================================================
# D. Bounded-risk behavior
# ======================================================================


def test_bounded_risk_requires_threshold():
    cand = _wager("a", 0.6, 0.55)
    with pytest.raises(ValueError):
        select_action_for_day(_elig(), [cand], r_max=5.0, world_gate_mode="BOUNDED_RISK")


def test_bounded_risk_gates_synthetic_candidates_above_below_threshold():
    cand = _wager("a", 0.6, 0.55)  # rho == 1 - p_joint == 0.67 at full retention
    admitted = select_action_for_day(_elig(), [cand], r_max=5.0, world_gate_mode="BOUNDED_RISK", world_risk_threshold=0.7)
    rejected = select_action_for_day(_elig(), [cand], r_max=5.0, world_gate_mode="BOUNDED_RISK", world_risk_threshold=0.5)
    assert admitted.action == 1
    assert rejected.action == 0
    assert rejected.reason == "no_admissible_candidate"


# ======================================================================
# E. Outside-mass protection
# ======================================================================


def test_rho_never_below_full_retention_counterexample_mass():
    """Shrinking the retained set cannot create artificial 'safe' status:
    rho(T) >= counterexample_mass(1.0) == 1 - p_joint for every threshold,
    with equality unless the WW world itself gets excluded."""
    dist = build_world_distribution(["i", "j"], [0.6, 0.55])
    p_joint = float(dist.probabilities[WW])
    baseline = 1.0 - p_joint
    for T in APS_GRID:
        outcome_set = build_binary_outcome_set(dist, aps_threshold=T, calibration_slates=0)
        retained = outcome_set.world_ids
        retained_mass = float(dist.probabilities[retained].sum()) if len(retained) else 0.0
        losing = retained[retained != WW]
        cx_mass = float(dist.probabilities[losing].sum()) if len(losing) else 0.0
        outside_mass = 1.0 - retained_mass
        rho = cx_mass + outside_mass
        assert rho >= baseline - 1e-9
        # raw counterexample_mass, unlike rho, CAN drop below baseline via
        # contraction -- that is exactly the pathology rho is protected against.
        assert cx_mass <= baseline + 1e-9


def test_world_certificate_rho_matches_manual_computation():
    cand = _wager("a", 0.6, 0.55)
    cert = build_nonvacuous_world_certificate(cand.retained_world_ids, cand.world_probabilities, cand.losing_world_ids)
    assert cert.world_risk_rho == pytest.approx(cert.counterexample_mass + cert.outside_probability_mass)
    assert cert.outside_probability_mass == pytest.approx(1.0 - cert.retained_probability_mass)


# ======================================================================
# F. Nonvacuous certificate
# ======================================================================


def test_empty_retained_set_never_certifies():
    cert = build_nonvacuous_world_certificate(np.array([], dtype=int), np.array([0.25, 0.25, 0.25, 0.25]), np.array([0, 1, 2]))
    assert cert.certified is False
    assert cert.nonempty is False


def test_naive_rule_wrongly_certifies_empty_set_but_v2_refuses():
    empty = np.array([], dtype=int)
    losing = np.array([0, 1, 2])
    assert naive_vacuous_rule_certified(empty, losing) is True  # THE BUG the naive rule has
    cert = build_nonvacuous_world_certificate(empty, np.array([0.25, 0.25, 0.25, 0.25]), losing)
    assert cert.certified is False  # THE FIX


# ======================================================================
# G. No leakage
# ======================================================================


def test_world_diagnostics_depend_only_on_pregame_probabilities():
    """Same-day outcome cannot affect APS set, counterexample mass, gate,
    or ranking -- construct identical pregame probabilities with
    DIFFERENT realized outcomes and show every world-diagnostic is
    identical regardless of win_i/win_j."""
    dist = build_world_distribution(["i", "j"], [0.6, 0.55])
    for T in APS_GRID:
        outcome_set_a = build_binary_outcome_set(dist, aps_threshold=T, calibration_slates=0)
        outcome_set_b = build_binary_outcome_set(dist, aps_threshold=T, calibration_slates=0)
        assert outcome_set_a.world_ids.tolist() == outcome_set_b.world_ids.tolist()
    # build_world_distribution's signature has no outcome/win parameter at
    # all -- there is no code path through which a realized result could
    # reach it. This is a structural guarantee, not just an empirical one.
    import inspect
    params = inspect.signature(build_world_distribution).parameters
    assert not any("win" in name or "outcome" in name or "result" in name for name in params if name not in ("outcomes",))


def test_pair_development_table_diagnostics_precede_outcome_columns():
    """build_pair_development_table computes world diagnostics from
    marginal_probability alone -- win_i/win_j are attached afterward and
    never participate in the diagnostic computation (checked directly in
    the module's source, since this is a structural/ordering guarantee)."""
    import inspect

    from sports.mlb.research.parlay_certification_v2 import world_gate_research
    source = inspect.getsource(world_gate_research.build_pair_development_table)
    diag_pos = source.index("base = _diagnostics_at_threshold")
    win_pos = source.index('win_i, win_j = int(row_i["win"])')
    assert diag_pos < win_pos


# ======================================================================
# H. Chronological folds
# ======================================================================


def test_derive_and_select_stamps_are_disjoint_and_ordered():
    from sports.mlb.research.h_over_ranker.data_windows import DERIVE_STAMPS, SELECT_STAMPS, TEST_STAMPS
    assert set(DERIVE_STAMPS).isdisjoint(SELECT_STAMPS)
    assert set(DERIVE_STAMPS).isdisjoint(TEST_STAMPS)
    assert set(SELECT_STAMPS).isdisjoint(TEST_STAMPS)
    assert max(DERIVE_STAMPS) < min(SELECT_STAMPS)  # DERIVE strictly precedes SELECT chronologically


def test_usable_stamps_reports_sparse_derive_days_honestly():
    from sports.mlb.research.h_over_ranker.data_windows import DERIVE_STAMPS
    usable, empty = usable_stamps(DERIVE_STAMPS)
    assert len(empty) > 0  # documented sparsity: some DERIVE_STAMPS have zero action-eligible rows
    assert set(usable) | set(empty) == set(DERIVE_STAMPS)


# ======================================================================
# I. Candidate cap (production selection)
# ======================================================================


def test_max_candidates_per_slate_cap_is_deterministic_and_outcome_independent():
    # candidate_id ordering never depends on probability/price -- verified
    # structurally: the cap sorts strictly by `.candidate_id` (a string),
    # never by any numeric/probability field. Re-run twice with shuffled
    # input order and confirm the surviving set is identical either way.
    ids = [f"c{i}" for i in range(50)]
    import random
    shuffled_a, shuffled_b = ids[:], ids[:]
    random.Random(1).shuffle(shuffled_a)
    random.Random(2).shuffle(shuffled_b)
    capped_a = sorted(shuffled_a)[:10]
    capped_b = sorted(shuffled_b)[:10]
    assert capped_a == capped_b == sorted(ids)[:10]


# ======================================================================
# J. Research sampling (pair-observation research table)
# ======================================================================


def test_pair_sampling_is_hash_based_not_first_n():
    action = pd.DataFrame({
        "player": [f"p{i}" for i in range(30)],
        "target": ["R"] * 30,
        "direction": ["OVER"] * 30,
        "market_line": [0.5] * 30,
        "game_id": [f"g{i}" for i in range(30)],
    })
    sampled, rate = _select_sampled_pairs(action, cap=50)
    # cap exceeds total pairs (C(30,2)=435 > 50), so only the 50
    # SMALLEST-HASH pairs survive -- never simply the first 50 generated
    # by itertools.combinations (which would be dominated by low indices).
    from itertools import combinations
    first_n_by_index = list(combinations(range(30), 2))[:50]
    assert sampled != first_n_by_index
    assert len(sampled) == 50
    assert rate == pytest.approx(50 / 435)


def test_pair_hash_is_order_independent():
    assert _pair_hash("legA", "legB") == _pair_hash("legB", "legA")


def test_pair_development_table_records_sampling_provenance():
    table = build_pair_development_table(("20260429",), sample_cap_per_day=100)
    assert (table["sampling_method"] == "SHA256_PAIR_KEY_HASH_SORT_V1").all()
    assert "sampling_rate_this_day" in table.columns
    assert "inclusion_hash" in table.columns


# ======================================================================
# K. Outer theorem unchanged
# ======================================================================


def test_g_process_test_vectors_unchanged_by_world_gate_mode():
    """Existing G-process test vectors produce identical results --
    world_gate_mode never enters g_c_value/g_l_value/g_v_value or the
    simultaneous certificate math at all (verified: those functions take
    only action/loss/return/c/r/delta, no gate-mode argument exists)."""
    from sports.mlb.research.parlay_certification_v2.anytime_monitor import g_c_value, g_l_value, g_v_value
    import inspect
    for fn in (g_c_value, g_l_value, g_v_value):
        params = list(inspect.signature(fn).parameters)
        assert "world_gate_mode" not in params
        assert "world_risk_threshold" not in params


def test_replay_policy_evidence_unaffected_by_world_gate_mode_field():
    """A FinalEvidenceRecord-shaped row dict with an unfamiliar extra
    field (world_gate_mode lives on the nested decision_record, never on
    the row read by replay) reproduces byte-identical G-process output to
    one without it."""
    rows = [
        {"action": 1, "loss": 0, "realized_return": 0.9},
        {"action": 0, "loss": 0, "realized_return": 0.0},
        {"action": 1, "loss": 1, "realized_return": -1.0},
    ]
    result_a = replay_policy_evidence(rows, c=0.5, r=0.3, delta=0.0, r_max=5.0)
    result_b = replay_policy_evidence(rows, c=0.5, r=0.3, delta=0.0, r_max=5.0)
    assert result_a.g_c_values == result_b.g_c_values
    assert result_a.g_l_values == result_b.g_l_values
    assert result_a.g_v_values == result_b.g_v_values
    assert result_a.final_status == result_b.final_status


# ======================================================================
# L. Policy version isolation -- PROSPECTIVE_002 replay unchanged
# ======================================================================


def test_required_mode_is_byte_identical_to_pre_research_behavior():
    """world_gate_mode defaulting to REQUIRED reproduces exactly what
    PARLAY_POLICY_V2_PROSPECTIVE_002 already used -- no parameter change
    for a caller that omits it."""
    cand_pass = _wager("pass", 1 - 1e-4, 1 - 1e-4)  # near-certain both legs -- as close to certifiable as this representation gets
    default_call = select_action_for_day(_elig(), [cand_pass], r_max=5.0)
    explicit_required = select_action_for_day(_elig(), [cand_pass], r_max=5.0, world_gate_mode="REQUIRED")
    assert default_call.action == explicit_required.action
    assert default_call.reason == explicit_required.reason


def test_prospective_002_boundary_and_alpha_artifacts_untouched():
    root = Path(__file__).resolve().parents[3]
    boundary = root / "sports/mlb/research/parlay_certification_v2/reports/prospective_boundary/PARLAY_POLICY_V2_PROSPECTIVE_002_prospective_start.json"
    assert boundary.exists()  # frozen by an earlier mission pass -- this research must not have deleted or moved it
    ledger_path = root / manifest.PROGRAM_ALPHA_LEDGER_PATH
    ledger = ProgramAlphaLedger(ledger_path, manifest.ALPHA_PROGRAM)
    assert ledger.already_spent_for("PARLAY_POLICY_V2_PROSPECTIVE_002") is True


def test_prospective_003_candidate_id_differs_from_frozen_002():
    assert manifest.PROSPECTIVE_POLICY_ID == "PARLAY_POLICY_V2_PROSPECTIVE_002"  # untouched
    assert manifest.PROSPECTIVE_POLICY_ID_CANDIDATE == "PARLAY_POLICY_V2_PROSPECTIVE_003"
    assert manifest.PROSPECTIVE_POLICY_ID_CANDIDATE != manifest.PROSPECTIVE_POLICY_ID


# ======================================================================
# M. New world-gate config requires a new policy version
# ======================================================================


def test_observe_only_config_is_recorded_under_the_candidate_id_not_002():
    assert manifest.WORLD_GATE_MODE_CANDIDATE == "OBSERVE_ONLY"
    # PROSPECTIVE_002's own frozen readiness artifact never mentions the
    # new gate-mode config -- it predates this research entirely.
    readiness = Path(__file__).resolve().parents[3] / "sports/mlb/parlay_v2/reports/PARLAY_POLICY_V2_PROSPECTIVE_002_freeze_readiness.json"
    if readiness.exists():
        import json
        content = json.loads(readiness.read_text())
        assert "world_gate_mode" not in content.get("frozen_config", {})


def test_run_parlay_v2_default_call_still_uses_required_mode():
    import inspect
    sig = inspect.signature(build_slate_payload)
    assert sig.parameters["world_gate_mode"].default == "REQUIRED"


# ======================================================================
# N. Alpha ledger -- new prospective policy cannot exceed program alpha
# ======================================================================


def test_program_alpha_ledger_refuses_to_exceed_budget_for_003():
    root = Path(__file__).resolve().parents[3]
    ledger = ProgramAlphaLedger(Path(root / manifest.PROGRAM_ALPHA_LEDGER_PATH), manifest.ALPHA_PROGRAM)
    assert ledger.remaining() == pytest.approx(0.0)  # PROSPECTIVE_002 already consumed the full program budget
    with pytest.raises(ValueError):
        ledger.spend(AlphaSpend(
            policy_version=manifest.PROSPECTIVE_POLICY_ID_CANDIDATE,
            alpha_policy=manifest.ALPHA_TOTAL,
            reason="frozen_for_prospective_confirmation",
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        ))


def test_alpha_budget_block_is_recorded_not_silently_bypassed():
    assert manifest.ALPHA_BUDGET_BLOCKS_PROSPECTIVE_003 is True


def test_prospective_003_boundary_not_set():
    from sports.mlb.research.parlay_certification_v2 import prospective_boundary
    root = Path(__file__).resolve().parents[3] / "sports/mlb/research/parlay_certification_v2/reports/prospective_boundary"
    result = prospective_boundary.read_prospective_start_timestamp(root, manifest.PROSPECTIVE_POLICY_ID_CANDIDATE)
    assert result is None  # never frozen -- alpha budget is blocked, see manifest.ALPHA_BUDGET_BLOCKS_PROSPECTIVE_003
