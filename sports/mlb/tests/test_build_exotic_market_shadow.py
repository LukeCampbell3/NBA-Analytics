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


def test_opposite_sides_of_same_game_total_are_not_double_published():
    source = {
        "status": "ok",
        "run_date": "2026-08-31",
        "games": [{
            "game_id": "823982",
            "away_team": "NYY",
            "home_team": "LAA",
            "combo_candidates": [{
                "leg_a": {
                    "market": "game_total", "side": "over", "line": 7.5,
                    "model_probability": .562, "price_american": -120, "price_confirmed": True,
                },
                "leg_b": {
                    "market": "game_total", "side": "under", "line": 7.5,
                    "model_probability": .438, "price_american": -102, "price_confirmed": True,
                },
            }],
        }],
    }
    payload = build_payload(source)
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["side"] == "over"
    assert payload["candidates"][0]["expected_value_per_unit"] > 0
    assert payload["diagnostic_rejection_count"] == 1
    assert payload["diagnostic_rejections"][0]["side"] == "under"
    assert payload["diagnostic_rejections"][0]["rejection_reason"] == "DOMINATED_OPPOSITE_SIDE_OR_DUPLICATE_MARKET"


def test_all_negative_ev_sides_are_diagnostics_not_primary_picks():
    source = {"status": "ok", "games": [{"game_id": "2", "combo_candidates": [{
        "leg_a": {"market": "game_total", "side": "over", "line": 8.5, "model_probability": .47,
                  "price_american": 100, "price_confirmed": True},
        "leg_b": {"market": "game_total", "side": "under", "line": 8.5, "model_probability": .53,
                  "price_american": -125, "price_confirmed": True},
    }]}]}
    payload = build_payload(source)
    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []
    assert payload["diagnostic_rejection_count"] == 2
    assert {row["rejection_reason"] for row in payload["diagnostic_rejections"]} == {"NON_POSITIVE_MODEL_EV"}
