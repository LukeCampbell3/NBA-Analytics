import numpy as np

from sports.mlb.parlay_v2.robust_parlay_search import (
    ROBUST_PAIR_CANDIDATE,
    UNSUPPORTED,
    JointDependencyEvidence,
    LegProbabilityEvidence,
    LPAParlaySearch,
    RobustParlayPolicy,
    value_two_leg_parlay,
)


def leg(leg_id, game, p, price, samples, cert=True, support=500):
    return LegProbabilityEvidence(
        leg_id=leg_id,
        game_id=game,
        mean_probability=p,
        decimal_price=price,
        effective_support=support,
        probability_samples=samples,
        pca_edge_certified=cert,
        sampling_source="PIPELINE_BOOTSTRAP",
    )


def policy(**kw):
    base = dict(
        bootstrap_draws=5000,
        min_leg_support=50,
        min_joint_support=500,
        min_joint_mean_probability=0.50,
        max_loss_probability=0.50,
        p_ev_positive_threshold=0.80,
        kelly_cap=0.05,
        kelly_multiplier=0.25,
        random_seed=7,
    )
    base.update(kw)
    return RobustParlayPolicy(**base)


def test_uncertainty_rejects_attractive_point_estimate():
    s1 = np.array([0.55] * 30 + [0.72] * 70)
    s2 = np.array([0.52] * 30 + [0.70] * 70)
    a = leg("A", "G1", 0.68, 1.55, s1)
    b = leg("B", "G2", 0.66, 1.55, s2)
    v = value_two_leg_parlay(
        "AB", a, b,
        JointDependencyEvidence("INDEPENDENT", support_count=5000),
        combined_decimal_price=2.60,
        policy=policy(),
    )
    assert v.ev_mean > 0
    assert v.ev_lcb < 0
    assert v.classification != ROBUST_PAIR_CANDIDATE
    assert "LOWER_BOUND_EV_NOT_POSITIVE" in v.classification_reasons


def test_strong_pair_can_pass_robust_gates_and_beat_singles():
    rng = np.random.default_rng(1)
    s1 = np.clip(rng.normal(0.82, 0.015, 2000), 0.75, 0.9)
    s2 = np.clip(rng.normal(0.80, 0.015, 2000), 0.73, 0.9)
    a = leg("A", "G1", 0.82, 1.40, s1)
    b = leg("B", "G2", 0.80, 1.40, s2)
    v = value_two_leg_parlay(
        "AB", a, b,
        JointDependencyEvidence("INDEPENDENT", support_count=5000),
        combined_decimal_price=2.25,
        policy=policy(),
    )
    assert v.p_joint > 0.60
    assert v.ev_lcb > 0
    assert v.p_ev_positive >= 0.80
    assert v.growth_advantage > 0
    assert v.classification == ROBUST_PAIR_CANDIDATE


def test_same_game_without_common_world_is_unsupported():
    s = np.linspace(0.74, 0.82, 1000)
    a = leg("A", "G1", 0.78, 1.50, s)
    b = leg("B", "G1", 0.78, 1.50, s)
    v = value_two_leg_parlay(
        "AB", a, b,
        JointDependencyEvidence("INDEPENDENT", support_count=5000),
        combined_decimal_price=2.4,
        policy=policy(),
    )
    assert v.classification == UNSUPPORTED
    assert "SAME_GAME_REQUIRES_COMMON_WORLD_OR_JOINT_SAMPLES" in v.classification_reasons


def test_common_world_dependence_changes_joint_probability():
    rng = np.random.default_rng(3)
    s1 = np.full(1000, 0.70)
    s2 = np.full(1000, 0.70)
    a = leg("A", "G1", 0.70, 1.55, s1)
    b = leg("B", "G1", 0.70, 1.55, s2)
    worlds = np.array([[1, 1]] * 600 + [[1, 0]] * 100 + [[0, 1]] * 100 + [[0, 0]] * 200)
    rng.shuffle(worlds)
    v = value_two_leg_parlay(
        "AB", a, b,
        JointDependencyEvidence("COMMON_WORLD", common_world_outcomes=worlds, support_count=1000),
        combined_decimal_price=2.0,
        policy=policy(max_dependency_burden=0.5),
    )
    assert abs(v.p_joint - 0.60) < 0.03
    assert v.dependency_burden > 0.15


def test_lpa_selects_best_robust_pair_and_repairs_after_update():
    rng = np.random.default_rng(4)
    s = np.clip(rng.normal(0.82, 0.01, 1500), 0.77, 0.88)
    a = leg("A", "G1", 0.82, 1.40, s)
    b = leg("B", "G2", 0.82, 1.40, s)
    c = leg("C", "G3", 0.82, 1.40, s)
    dep = JointDependencyEvidence("INDEPENDENT", support_count=5000)
    v1 = value_two_leg_parlay("AB", a, b, dep, combined_decimal_price=2.20, policy=policy())
    v2 = value_two_leg_parlay("AC", a, c, dep, combined_decimal_price=2.30, policy=policy())
    assert v1.classification == ROBUST_PAIR_CANDIDATE
    assert v2.classification == ROBUST_PAIR_CANDIDATE
    search = LPAParlaySearch([v1, v2], policy=policy(), mode="robust")
    r1 = search.result()
    assert r1.selected_pair_id == "AC"
    first_expansions = search.expansions

    v1b = value_two_leg_parlay("AB", a, b, dep, combined_decimal_price=2.45, policy=policy())
    search.update_valuation(v1b)
    before = search.expansions
    r2 = search.result()
    repair_expansions = search.expansions - before
    assert r2.selected_pair_id == "AB"
    assert repair_expansions <= first_expansions


def test_non_pca_certified_pair_cannot_pass_robust_gates():
    rng = np.random.default_rng(5)
    s = np.clip(rng.normal(0.82, 0.01, 1000), 0.78, 0.87)
    a = leg("A", "G1", 0.82, 1.40, s, cert=False)
    b = leg("B", "G2", 0.82, 1.40, s, cert=True)
    v = value_two_leg_parlay(
        "AB", a, b,
        JointDependencyEvidence("INDEPENDENT", support_count=5000),
        combined_decimal_price=2.3,
        policy=policy(),
    )
    assert v.classification != ROBUST_PAIR_CANDIDATE
    assert "PCA_EDGE_NOT_CERTIFIED" in v.classification_reasons
