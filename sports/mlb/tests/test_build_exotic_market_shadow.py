from sports.mlb.scripts.build_exotic_market_shadow import build_payload


def test_only_matching_aggregate_targets_are_scored():
    source = {"status": "ok", "games": [{"game_id": "1", "combo_candidates": [{
        "leg_a": {"market": "game_total", "side": "over", "line": 8.5, "model_probability": .60,
                  "price_american": -110, "price_confirmed": True},
        "leg_b": {"market": "moneyline", "side": "home", "model_probability": .70,
                  "price_american": -150, "price_confirmed": True},
    }]}]}
    payload = build_payload(source)
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["market"] == "game_total"
    assert payload["candidates"][0]["publication_authority"] is False


def test_granular_markets_are_declared_but_not_fabricated():
    payload = build_payload({"status": "ok", "games": []})
    readiness = {row["market"]: row["readiness"] for row in payload["market_registry"]}
    assert readiness["pitcher_strikeouts_inning"] == "EVENT_MODEL_REQUIRED"
    assert readiness["plate_appearance_pitch_count"] == "EVENT_MODEL_REQUIRED"
    assert payload["candidates"] == []
