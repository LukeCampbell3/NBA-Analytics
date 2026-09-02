import json

from sports.mlb.unified.pipeline import export_payload, run


def test_pipeline_fails_closed_and_does_not_copy_parlay_legs_to_singles(tmp_path):
    play = {"game_id":"g1", "player":"A", "player_id":"a", "target":"H", "direction":"OVER", "market_line":.5, "final_hit_probability":.7, "selected_side_price":-110, "historical_bucket_support":100, "lineup_status":"CONFIRMED", "selected_sportsbook_key":"fanduel"}
    (tmp_path / "daily_predictions.json").write_text(json.dumps({"plays":[play]}))
    combo = {"legs":[{"market":"game_total", "side":"over", "line":8.5, "model_probability":.7, "price_american":-110, "leg_authorized":True}]}
    (tmp_path / "same_game_predictions.json").write_text(json.dumps({"games":[{"game_id":"g1", "home_team":"A", "away_team":"B", "combo_candidates":[combo]}]}))
    result = run(tmp_path)
    assert [c.subject_id for c in result.singles] == ["a"]
    assert any(c.market_type == "game_total" and "UNCERTAINTY_UNAVAILABLE" in c.rejection_reasons for c in result.rejected)
    payload = export_payload(result)
    assert payload["evidence"]["publication_authority"] is False
    assert payload["schema_version"] == "unified_mlb_v1"
