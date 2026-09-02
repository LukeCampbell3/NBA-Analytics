from __future__ import annotations

import json

import pytest

from sports.mlb.unified.v2_1_challenger import (
    UncertaintyComponents, UnifiedPolicyV21, evaluate_challenger_candidate,
    hits_distribution, line_sensitivity,
    lower_bound_probability, maximum_acceptable_negative_price,
    pareto_frontier, select_challenger, total_bases_distribution,
)
from sports.mlb.unified.v2_evidence import (
    append_hash_linked_settlement, capture_policy_generation, read_ledger,
)
from sports.mlb.unified.v2_diagnostics import (
    boundary_diagnostic, rank_performance, top_k_performance, uncertainty_diagnostics,
)


def components(value=.01):
    return {name: value for name in UncertaintyComponents.__annotations__}


def candidate(identifier="a", **overrides):
    value = {
        "candidate_id": identifier, "slate_id": "MLB_20260901", "event_id": "1",
        "player_id": identifier, "capability": "batter_hits", "market_id": "m",
        "selection_id": "s", "line": .5, "sportsbook": "fanduel",
        "quoted_odds": -120, "quote_timestamp": "2026-09-01T12:00:00Z",
        "prediction_timestamp": "2026-09-01T12:05:00Z",
        "decision_timestamp": "2026-09-01T12:05:00Z", "lineup_status": "CONFIRMED",
        "player_status": "ACTIVE", "model_version": "m1", "calibrator_version": "c1",
        "raw_probability": .72, "calibrated_probability": .70, "usable_probability": .68,
        "market_implied_probability": 120/220, "no_vig_market_probability": .53,
        "uncertainty": None, "uncertainty_components": components(), "support_score": 200,
        "support_status": "IN_SUPPORT", "ood_status": "IN_SUPPORT",
        "identity_status": "CONFIRMED", "edge": .13, "raw_ev": .32,
        "calibrated_ev": .28, "conservative_ev": .24, "market_type": "batter_hits",
        "side": "over",
    }
    value.update(overrides)
    return value


def test_uncertainty_components_and_lower_bound_are_exact():
    item = UncertaintyComponents(**components(.02))
    assert item.total == pytest.approx((8 * .02**2) ** .5)
    assert lower_bound_probability(.70, .05) == pytest.approx(.65)
    assert maximum_acceptable_negative_price(.60) == pytest.approx(-150)


def test_challenger_fails_when_components_are_missing():
    result = evaluate_challenger_candidate(candidate(uncertainty_components=None))
    assert result["admissible"] is False
    assert "UNCERTAINTY_COMPONENTS_UNAVAILABLE" in result["rejection_reasons"]


def test_pareto_and_ranking_are_deterministic():
    rows = [
        {"candidate_id": "a", "probability_lcb": .62, "conservative_expected_value": .10},
        {"candidate_id": "b", "probability_lcb": .61, "conservative_expected_value": .09},
        {"candidate_id": "c", "probability_lcb": .60, "conservative_expected_value": .15},
    ]
    assert {row["candidate_id"] for row in pareto_frontier(rows)} == {"a", "c"}
    policy = UnifiedPolicyV21(top_k=1)
    first = select_challenger([candidate("a"), candidate("b", usable_probability=.66)], policy)
    second = select_challenger([candidate("b", usable_probability=.66), candidate("a")], policy)
    assert [row["candidate_id"] for row in first["selected"]] == [row["candidate_id"] for row in second["selected"]]
    assert len(first["selected"]) == 1


def test_price_movement_and_low_support_fail_closed():
    expensive = evaluate_challenger_candidate(candidate(quoted_odds=-300))
    assert expensive["admissible"] is False
    assert "EDGE_LCB_NOT_POSITIVE" in expensive["rejection_reasons"]
    weak = evaluate_challenger_candidate(candidate(support_score=4))
    assert weak["admissible"] is False
    assert "SUPPORT_INVALID" in weak["rejection_reasons"]


def test_opportunity_conditioned_hit_and_total_base_distributions_and_line_sensitivity():
    pa = {3: .25, 4: .75}
    hits = hits_distribution(pa, .25)
    assert sum(hits.values()) == pytest.approx(1)
    assert sum(value*probability for value, probability in hits.items()) == pytest.approx(3.75*.25)
    tb = total_bases_distribution(pa, {0: .70, 1: .20, 2: .05, 3: .01, 4: .04})
    assert sum(tb.values()) == pytest.approx(1)
    offers = line_sensitivity(tb, [{"line": .5, "odds": -180}, {"line": 1.5, "odds": -105}], .02)
    assert {row["line"] for row in offers} == {.5, 1.5}
    assert all("conservative_ev" in row for row in offers)


def test_prediction_evidence_is_immutable_and_settlement_hash_linked(tmp_path):
    path = tmp_path / "ledger.jsonl"
    evaluated = evaluate_challenger_candidate(candidate())
    evidence = {**candidate(), **evaluated, "ranking_position": 1,
                "final_selection_decision": True, "conservative_ev": evaluated["conservative_expected_value"]}
    kwargs = dict(generation_id="g1", generated_at_utc="2026-09-01T12:05:00Z",
                  run_date="2026-09-01", baseline_policy_hash="b"*64,
                  challenger_policy_hash="c"*64, baseline_candidates=[evidence],
                  challenger_candidates=[evidence], disagreements=[])
    assert capture_policy_generation(path, **kwargs)
    assert capture_policy_generation(path, **kwargs) is False
    changed = json.loads(json.dumps(kwargs))
    changed["challenger_candidates"][0]["usable_probability"] = .99
    with pytest.raises(ValueError, match="collision"):
        capture_policy_generation(path, **changed)
    assert append_hash_linked_settlement(
        path, generation_id="g1", candidate_id="a", official_outcome=1,
        settlement="won", realized_return=.833333, source_identity="MLB_STATSAPI_FINAL_FEED",
        source_response_sha256="d"*64, settled_at_utc="2026-09-02T03:00:00Z")
    rows = read_ledger(path)
    assert rows[1]["settlement_payload"]["prediction_payload_sha256"] == rows[0]["prediction_payload_sha256"]


def test_prospective_diagnostics_fail_closed_and_then_measure_full_population():
    assert uncertainty_diagnostics([])["state"] == "INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    rows = []
    for slate in range(20):
        for rank in range(1, 5):
            rows.append({"slate_id": str(slate), "ranking_position": rank,
                         "usable_probability": .75-rank*.03, "uncertainty": rank*.02,
                         "outcome": int(rank <= 2), "realized_return": .7 if rank <= 2 else -1,
                         "edge_lcb": .04-rank*.005})
    uncertainty = uncertainty_diagnostics(rows)
    assert uncertainty["state"] == "DIAGNOSTIC"
    assert uncertainty["monotonic_degradation"] is True
    assert rank_performance(rows)["1"]["hit_rate"] == 1
    assert top_k_performance(rows)["top_2"]["selections"] == 40
    boundary = boundary_diagnostic(rows, "edge_lcb", .03, .02)
    assert boundary["just_above"]["count"] > 0
