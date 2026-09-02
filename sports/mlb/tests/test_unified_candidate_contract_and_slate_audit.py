from sports.mlb.unified.candidate_contract import terminal_decision
from sports.mlb.unified.slate_audit import audit_candidates, authoritative_status


def row(**overrides):
    value = {
        "candidate_id": "a", "player_id": "1", "market_type": "batter_hits",
        "identity_status": "CONFIRMED", "lineup_status": "CONFIRMED",
        "run_date": "2026-09-01", "quote_timestamp": "2026-09-01T16:00:00Z",
        "rejection_reasons": [], "final_selection_decision": True,
        "conservative_expected_value": .08,
    }
    value.update(overrides)
    return value


def test_terminal_state_distinguishes_missing_data_from_no_edge():
    assert terminal_decision([]) == "NO_FULLY_EVALUABLE_CANDIDATES"
    assert terminal_decision([{"rejection_reasons": ["LINEUP_INVALID"]}]) == "DATA_CONTRACT_INCOMPLETE"
    assert terminal_decision([{"rejection_reasons": ["PROBABILITY_EDGE_BELOW_FLOOR"]}]) == "NO_RELIABLE_EDGE_FOUND"
    assert terminal_decision([{"rejection_reasons": [], "final_selection_decision": True}]) == "CHALLENGER_SELECTIONS_AVAILABLE"


def test_identity_lineup_and_freshness_are_hard_publication_invariants():
    report = audit_candidates([
        row(candidate_id="hamilton", identity_status="UNKNOWN", show_betslip_action=True),
        row(candidate_id="guzman", lineup_status="NOT_IN_POSTED_LINEUP"),
        row(candidate_id="old", run_date="2026-08-28"),
    ], run_date="2026-09-01")
    assert report["publication_integrity"] == "FAIL"
    assert {"IDENTITY_MISMATCH", "LINEUP_ROLE_INVALID", "STALE_ARTIFACT"} <= set(report["fatal_issues"])


def test_research_and_negative_ev_rows_cannot_expose_betslip_action():
    candidate = row(final_selection_decision=False, rejection_reasons=["NON_POSITIVE_CONSERVATIVE_EV"],
                    conservative_expected_value=-.01, sportsbook_deeplink="https://example.test")
    assert authoritative_status(candidate) == "REJECTED_VALUE"
    report = audit_candidates([candidate], run_date="2026-09-01")
    assert report["issue_counts"]["NON_ACTIONABLE_BETSLIP_CTA"] == 1
    assert report["issue_counts"]["NEGATIVE_EV_BETSLIP_CTA"] == 1


def test_identical_calibrated_probabilities_are_exposed_as_plateau_not_assumed_bug():
    candidates = [row(candidate_id=str(i), player_id=str(i), calibrated_probability=.6025157232704403)
                  for i in range(3)]
    report = audit_candidates(candidates, run_date="2026-09-01")
    assert report["probability_plateaus"][0]["issue"] == "CALIBRATION_PLATEAU_REVIEW"
