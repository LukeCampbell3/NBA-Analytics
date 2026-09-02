import pytest

from sports.mlb.unified.validation import GradedBet, ReplayEvidence, assert_point_in_time, bankroll_paths, evaluate


def record(result, evidence=ReplayEvidence.EXACT_POINT_IN_TIME):
    return GradedBet(.6, 2.0, result, "2026-08-01T12:00:00Z", "2026-08-01T18:00:00Z", evidence)


def test_validation_metrics_exclude_unavailable_evidence():
    report = evaluate([record("WIN"), record("LOSS"), record("WIN", ReplayEvidence.UNAVAILABLE)])
    assert report["bets"] == 2 and report["roi"] == 0
    assert report["evidence_counts"]["UNAVAILABLE"] == 1
    assert bankroll_paths([record("WIN")])["flat_1"] == 101


def test_post_start_predictions_fail_point_in_time_gate():
    bad = GradedBet(.6, 2, "WIN", "2026-08-01T19:00:00Z", "2026-08-01T18:00:00Z", ReplayEvidence.EXACT_POINT_IN_TIME)
    with pytest.raises(ValueError, match="POST_START"):
        assert_point_in_time(bad)
