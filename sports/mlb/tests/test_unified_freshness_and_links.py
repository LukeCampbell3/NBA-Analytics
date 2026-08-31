from datetime import date, datetime, timezone

import pytest

from sports.mlb.unified.deeplink_validation import validate_fanduel_link
from sports.mlb.unified.freshness import validate_freshness


def test_stale_and_wrong_slate_artifacts_fail_closed():
    payload = {"run_date":"2026-08-31","generated_at_utc":"2026-08-31T12:00:00Z"}
    validate_freshness(payload, expected_slate_date=date(2026,8,31), now=datetime(2026,8,31,13,tzinfo=timezone.utc), maximum_age_hours=30)
    with pytest.raises(ValueError, match="STALE"):
        validate_freshness(payload, expected_slate_date=date(2026,8,31), now=datetime(2026,9,2,13,tzinfo=timezone.utc), maximum_age_hours=30)
    with pytest.raises(ValueError, match="SLATE_DATE"):
        validate_freshness(payload, expected_slate_date=date(2026,9,1), now=datetime(2026,8,31,13,tzinfo=timezone.utc), maximum_age_hours=30)


def test_fanduel_exact_link_requires_real_market_and_selection_ids():
    good = "https://sportsbook.fanduel.com/addToBetslip?marketId=10&selectionId=20"
    assert validate_fanduel_link(good, require_exact_selection=True) == (True, "VALID")
    assert validate_fanduel_link("https://sportsbook.fanduel.com/", require_exact_selection=True)[0] is False
    assert validate_fanduel_link("https://example.com/?marketId=1&selectionId=2", require_exact_selection=True)[0] is False
