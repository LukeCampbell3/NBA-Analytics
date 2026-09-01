from sports.mlb.scripts.run_mlb_pitcher_parlay_quality_daily import _store_legacy_max_hit_diagnostic


def test_legacy_max_hit_control_cannot_survive_as_frontend_fallback_field() -> None:
    payload = {"max_hit_control": {"candidate_authorized": False}}

    _store_legacy_max_hit_diagnostic(payload, None)

    assert "max_hit_control" not in payload
    assert payload["diagnostics"]["legacy_max_hit_control"] is None
    assert payload["diagnostics"]["legacy_max_hit_control_publication_authority"] is False
