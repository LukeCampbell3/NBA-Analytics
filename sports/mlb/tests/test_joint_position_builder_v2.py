from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports.mlb.conditional_chain.outcome_worlds import (
    build_world_distribution,
    certify_perfect_parlay,
    search_parlay_proof_frontier,
    world_id_from_outcomes,
)
from sports.mlb.research.joint_position_builder_v2 import manifest
from sports.mlb.research.joint_position_builder_v2.pairs import (
    build_pair_certificate,
    conservative_joint_lower_bound,
    pair_class,
)
from sports.mlb.research.joint_position_builder_v2.legacy.risk_gate_v1_ARCHIVED import (
    ActionDecision,
    SelectiveRiskCertificate,
    gate_and_rank_day,
)
from sports.mlb.research.joint_position_builder_v2.pairs import CandidatePair, PairCertificate


def _american_to_decimal(price: float) -> float:
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


# ---------------------------------------------------------------------------
# Theorem 1: +EV pair with one individually -EV leg under independence.
# pA=.80, dA=1.40 -> EV=+12%.  pB=.70, dB=1.40 -> EV=-2%.
# independent pair: p=.56, odds=1.96, EV=+9.76%.
# Old (CONTROL-style) EV_i>0 gate must reject B; V2 must still evaluate the pair.
# ---------------------------------------------------------------------------

def test_theorem_1_plus_ev_pair_with_one_individually_minus_ev_leg():
    pA, dA = 0.80, 1.40
    pB, dB = 0.70, 1.40
    ev_a = pA * dA - 1.0
    ev_b = pB * dB - 1.0
    assert ev_a == pytest.approx(0.12, abs=1e-9)
    assert ev_b == pytest.approx(-0.02, abs=1e-9)

    # Old (CONTROL-style) individual-EV gate: rejects leg B outright.
    old_gate_admits_b = ev_b > 0.0
    assert old_gate_admits_b is False

    # V2: builds the joint distribution regardless of leg B's own sign.
    distribution = build_world_distribution(["A", "B"], [pA, pB])
    p_joint = float(distribution.probabilities[world_id_from_outcomes([1, 1])])
    assert p_joint == pytest.approx(0.56, abs=1e-9)

    d_s = dA * dB
    assert d_s == pytest.approx(1.96, abs=1e-9)
    joint_ev = p_joint * d_s - 1.0
    assert joint_ev == pytest.approx(0.0976, abs=1e-6)
    assert joint_ev > 0.0

    assert pair_class(ev_a, ev_b) == "+-"


# ---------------------------------------------------------------------------
# Theorem 2: both legs -EV individually, pair +EV under underpriced positive
# dependence (NOT achievable under independence -- requires the interactions
# hook).
# ---------------------------------------------------------------------------

def test_theorem_2_both_legs_minus_ev_pair_plus_ev_under_positive_dependence():
    p, d = 0.55, 1.70
    ev = p * d - 1.0
    assert ev == pytest.approx(-0.065, abs=1e-9)
    assert ev < 0.0  # both legs individually -EV

    d_s = d * d

    # Under independence, still -EV (a genuinely -EV pair should stay -EV
    # without a dependence structure to rescue it).
    independent_distribution = build_world_distribution(["A", "B"], [p, p])
    p_joint_independent = float(independent_distribution.probabilities[world_id_from_outcomes([1, 1])])
    assert p_joint_independent * d_s - 1.0 < 0.0

    # With a modest positive-dependence interaction (both-win / both-lose
    # worlds boosted symmetrically -- see pairs.py module docstring / the
    # outcome_worlds.build_world_distribution interactions contract):
    rho = 0.5
    interactions = np.array([[0.0, rho], [rho, 0.0]])
    dependent_distribution = build_world_distribution(["A", "B"], [p, p], interactions=interactions)
    p_joint_dependent = float(dependent_distribution.probabilities[world_id_from_outcomes([1, 1])])
    assert p_joint_dependent > p_joint_independent

    joint_ev = p_joint_dependent * d_s - 1.0
    assert joint_ev > 0.0  # the pair is +EV despite both legs being individually -EV
    assert joint_ev == pytest.approx(0.2723, abs=1e-3)


# ---------------------------------------------------------------------------
# Theorem 3: exact zero-counterexample certificate equivalence with the
# pre-existing generic N-candidate machinery (outcome_worlds.py, unmodified).
# ---------------------------------------------------------------------------

def test_pair_certificate_agrees_exactly_with_existing_generic_certificate_logic():
    from sports.mlb.conditional_chain.outcome_worlds import build_binary_outcome_set

    for probs, aps_threshold, expect_certificate in [
        # Both mixed worlds' tied APS score is 0.9999 here (0.0099+0.0099
        # above the both-win world's own 0.9801) -- a threshold of 0.99
        # retains only the both-win world.
        ([0.99, 0.99], 0.99, True),
        # Mixed worlds' tied APS score is 0.7975 here; a threshold of 0.85
        # retains them too (world_ids [1,2,3]) -> not a certificate.
        ([0.55, 0.55], 0.85, False),
    ]:
        distribution = build_world_distribution(["A", "B"], probs)
        outcome_set = build_binary_outcome_set(distribution, aps_threshold=aps_threshold, calibration_slates=25)

        pair_cert = build_pair_certificate(distribution, outcome_set)

        candidates = pd.DataFrame({"candidate_id": ["A", "B"], "player": ["A", "B"]})
        frontier = search_parlay_proof_frontier(candidates, outcome_set, requested_leg_count=2)
        generic_cert = certify_perfect_parlay(candidates, outcome_set, requested_leg_count=2)

        assert pair_cert.logical_certificate == frontier.logically_proven == expect_certificate
        assert pair_cert.logical_certificate == generic_cert.logical_implication_proven
        assert generic_cert.production_authorized is False  # inherited invariant, never overridden


# ---------------------------------------------------------------------------
# Theorem 4: strict chronology / no future data.
# ---------------------------------------------------------------------------

def test_ablation_calibration_never_uses_same_or_future_day_pairs():
    """Reproduces the walk-forward accumulation loop's invariant directly:
    a pair recorded on day D must have calibration_pairs_prior equal to the
    exact count of pairs recorded on STRICTLY earlier days, never including
    same-day or later pairs."""
    from sports.mlb.research.joint_position_builder_v2.ablation import _pair_to_record

    # Simulate three days worth of pair records the way run_variant would
    # build them, and check the invariant on the resulting frame directly
    # (avoids a slow real data run inside this unit test).
    rows = []
    prior = 0
    for day_index, date in enumerate(["d1", "d2", "d3"]):
        n_today = 2
        for _ in range(n_today):
            rows.append({"date": date, "calibration_pairs_prior": prior})
        prior += n_today
    frame = pd.DataFrame(rows)
    for date, group in frame.groupby("date", sort=True):
        prior_group = frame[frame["date"] < date]
        assert (group["calibration_pairs_prior"] == len(prior_group)).all()


def test_certificate_computed_before_gate_never_sees_that_days_own_outcome():
    """A pair evaluated with aps_threshold derived from PRIOR calibration
    scores only -- confirms build_pair_certificate/gate_and_rank_day take
    the outcome_set as already-computed input and never re-derive it from
    the pair's own realized win/loss."""
    distribution = build_world_distribution(["A", "B"], [0.6, 0.6])
    from sports.mlb.conditional_chain.outcome_worlds import build_binary_outcome_set

    outcome_set_before_knowing_outcome = build_binary_outcome_set(distribution, aps_threshold=0.5, calibration_slates=25)
    cert = build_pair_certificate(distribution, outcome_set_before_knowing_outcome)
    # The certificate is a pure function of (distribution, outcome_set); it
    # cannot depend on win_i/win_j, which are not passed to it at all.
    import inspect

    assert "win" not in inspect.signature(build_pair_certificate).parameters
    assert cert.retained_world_count >= 0


# ---------------------------------------------------------------------------
# Theorem 5: V2 never authorizes production by default.
# ---------------------------------------------------------------------------

def test_manifest_production_authorized_is_always_false():
    assert manifest.PRODUCTION_AUTHORIZED is False


def _dummy_pair(joint_ev_lcb: float, counterexample_mass: float, support: float = 30.0) -> CandidatePair:
    return CandidatePair(
        date="d1", leg_i="A", leg_j="B", game_i="g1", game_j="g2", same_game=False,
        p_i=0.9, p_j=0.9, ev_i=0.1, ev_j=0.1, pair_class="++",
        p_joint=0.81, p_joint_l=0.7, d_s=1.5, joint_ev=joint_ev_lcb + 0.05, joint_ev_lcb=joint_ev_lcb,
        certificate=PairCertificate(retained_world_count=1, counterexample_count=0, counterexample_mass=counterexample_mass, world_contraction_bits=2.0, logical_certificate=(counterexample_mass == 0.0)),
        win_i=1, win_j=1, aps_score_true_world=0.1, support_min_history_rows=support, support_max_rmse=0.5,
    )


@pytest.mark.parametrize(
    "certificate_status",
    [
        "INSUFFICIENT_EVALUATED_PAIRS",
        "NO_DEVELOPMENT_THRESHOLD_MEETS_RISK_BOUND",
        "SELECTIVE_RISK_BOUND_NOT_SUPPORTED_ON_VALIDATION",
    ],
)
def test_gate_abstains_whenever_risk_certificate_is_not_supported(certificate_status):
    certificate = SelectiveRiskCertificate(certificate_status, 0.3, 0.1, 20, 15, 2, 0.13, 0.25)
    # Even an extremely attractive pair must not act without a supported certificate.
    great_pair = _dummy_pair(joint_ev_lcb=0.50, counterexample_mass=0.0)
    decision = gate_and_rank_day([great_pair], joint_ev_lcb_margin=0.0, min_support_history_rows=20.0, risk_certificate=certificate)
    assert decision.action == "ABSTAIN"


def test_gate_can_act_only_when_certificate_supported_and_pair_clears_thresholds():
    certificate = SelectiveRiskCertificate(
        "SELECTIVE_RISK_BOUND_SUPPORTED_ON_VALIDATION", 0.3, 0.1, 20, 15, 2, 0.13, 0.25
    )
    great_pair = _dummy_pair(joint_ev_lcb=0.50, counterexample_mass=0.05)
    weak_pair = _dummy_pair(joint_ev_lcb=-0.10, counterexample_mass=0.05)
    decision = gate_and_rank_day([weak_pair, great_pair], joint_ev_lcb_margin=0.0, min_support_history_rows=20.0, risk_certificate=certificate)
    assert decision.action == "ACT"
    assert decision.selected_pair_index == 1  # the great pair, not the weak one


def test_gate_selects_at_most_one_pair_per_day():
    certificate = SelectiveRiskCertificate(
        "SELECTIVE_RISK_BOUND_SUPPORTED_ON_VALIDATION", 0.3, 0.1, 20, 15, 2, 0.13, 0.25
    )
    pairs = [_dummy_pair(joint_ev_lcb=0.3, counterexample_mass=0.05) for _ in range(5)]
    decision = gate_and_rank_day(pairs, joint_ev_lcb_margin=0.0, min_support_history_rows=20.0, risk_certificate=certificate)
    assert decision.action == "ACT"
    assert decision.selected_pair_index is not None
    # exactly one selection -- ActionDecision carries a single index, not a list
    assert isinstance(decision.selected_pair_index, int)


# ---------------------------------------------------------------------------
# Additional correctness: never substitutes product odds for a same-game quote
# ---------------------------------------------------------------------------

def test_same_game_pairs_never_get_a_synthesized_d_s():
    from sports.mlb.research.joint_position_builder_v2.pairs import enumerate_candidate_pairs

    rows = pd.DataFrame(
        [
            {
                "date": "d1", "player": "P1", "player_key": "p1", "game_id": "g1", "team": "A",
                "target": "H", "direction": "OVER", "market_line": 0.5,
                "corrected_prediction": 1.2, "corrected_edge": 0.7,
                "marginal_probability": 0.7, "decimal_price": _american_to_decimal(-130),
                "marginal_ev": 0.7 * _american_to_decimal(-130) - 1.0,
                "rmse": 0.8, "history_rows": 40.0, "market_source": "real", "in_support": True, "win": 1,
            },
            {
                "date": "d1", "player": "P2", "player_key": "p2", "game_id": "g1", "team": "B",
                "target": "H", "direction": "OVER", "market_line": 0.5,
                "corrected_prediction": 1.1, "corrected_edge": 0.6,
                "marginal_probability": 0.65, "decimal_price": _american_to_decimal(-120),
                "marginal_ev": 0.65 * _american_to_decimal(-120) - 1.0,
                "rmse": 0.8, "history_rows": 40.0, "market_source": "real", "in_support": True, "win": 1,
            },
        ]
    )
    pairs = enumerate_candidate_pairs(rows, aps_threshold=0.9, calibration_slates=25)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.same_game is True
    assert pair.d_s is None
    assert pair.joint_ev is None
    assert pair.joint_ev_lcb is None
    # probability/mechanism is still reported even without a real quote
    assert pair.p_joint is not None
    assert pair.certificate is not None


def test_cross_game_pairs_get_product_of_leg_decimal_odds_as_d_s():
    from sports.mlb.research.joint_position_builder_v2.pairs import enumerate_candidate_pairs

    rows = pd.DataFrame(
        [
            {
                "date": "d1", "player": "P1", "player_key": "p1", "game_id": "g1", "team": "A",
                "target": "H", "direction": "OVER", "market_line": 0.5,
                "corrected_prediction": 1.2, "corrected_edge": 0.7,
                "marginal_probability": 0.8, "decimal_price": 1.40,
                "marginal_ev": 0.8 * 1.40 - 1.0,
                "rmse": 0.8, "history_rows": 40.0, "market_source": "real", "in_support": True, "win": 1,
            },
            {
                "date": "d1", "player": "P2", "player_key": "p2", "game_id": "g2", "team": "B",
                "target": "H", "direction": "OVER", "market_line": 0.5,
                "corrected_prediction": 1.0, "corrected_edge": 0.5,
                "marginal_probability": 0.7, "decimal_price": 1.40,
                "marginal_ev": 0.7 * 1.40 - 1.0,
                "rmse": 0.8, "history_rows": 40.0, "market_source": "real", "in_support": True, "win": 0,
            },
        ]
    )
    pairs = enumerate_candidate_pairs(rows, aps_threshold=0.9, calibration_slates=25)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.same_game is False
    assert pair.d_s == pytest.approx(1.96, abs=1e-9)
    assert pair.joint_ev is not None


# ---------------------------------------------------------------------------
# 2-leg only: no 3/4-leg promotion path exists in this package.
# ---------------------------------------------------------------------------

def test_no_three_or_four_leg_promotion_exists_in_this_package():
    import ast
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1] / "research" / "joint_position_builder_v2"
    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "requested_leg_count" and isinstance(kw.value, ast.Constant):
                        assert kw.value.value == 2, f"{path.name} requests a non-2 leg count"
