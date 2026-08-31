import pytest

from sports.mlb.unified.decision import DecisionPolicy, decide
from sports.mlb.unified.schemas import BetCandidate, EvidenceState


def candidate(**overrides):
    values = dict(
        candidate_id="c1", game_id="g1", subject_type="player", subject_id="p1",
        team="SEA", opponent="TOR", market_type="batter_hits", period="game",
        event_identity="g1:p1:game", side="over", line=0.5, sportsbook="fanduel",
        sportsbook_market_id="m1", sportsbook_selection_id="s1", american_price=-150,
        decimal_price=None, structural_probability=.68, market_conditioned_probability=None,
        raw_probability=.68, calibrated_probability=.66, uncertainty=.03,
        usable_probability=None, support_status="SUPPORTED", lineup_status="CONFIRMED",
        role_status="CONFIRMED", identity_status="CONFIRMED",
        evidence_state=EvidenceState.PROSPECTIVE_SHADOW,
    )
    values.update(overrides)
    return BetCandidate(**values)


def test_universal_gate_uses_usable_probability_for_ev():
    result = decide(candidate(), DecisionPolicy())
    assert result.usable_probability == .63
    assert result.market_break_even_probability == pytest.approx(.6)
    assert result.conservative_expected_value > 0
    assert result.rejection_reasons == []


def test_short_price_trap_fails_even_with_high_raw_probability():
    result = decide(candidate(american_price=-4500, raw_probability=.97, calibrated_probability=.97, uncertainty=.011), DecisionPolicy())
    assert result.usable_probability == .959
    assert "PROBABILITY_EDGE_BELOW_FLOOR" in result.rejection_reasons
    assert "NON_POSITIVE_CONSERVATIVE_EV" in result.rejection_reasons


def test_missing_probability_price_or_identity_fails_closed():
    result = decide(candidate(calibrated_probability=None, raw_probability=None, american_price=None, identity_status="UNKNOWN"), DecisionPolicy())
    assert {"PROBABILITY_UNAVAILABLE", "PRICE_UNAVAILABLE", "IDENTITY_INVALID"} <= set(result.rejection_reasons)
