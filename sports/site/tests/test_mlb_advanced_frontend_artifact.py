from sports.site.pipeline import build_static_site


def test_game_conditioned_payload_is_public_prediction_data() -> None:
    assert "sequential_pa_hitter_predictions.json" in build_static_site.PREDICTION_DATA_FILES
