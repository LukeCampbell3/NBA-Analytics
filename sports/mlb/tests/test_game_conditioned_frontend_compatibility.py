from sports.mlb.scripts.run_sequential_pa_hitter_model import (
    FRONTEND_SCHEMA_VERSION,
    GAME_CONDITIONED_SCHEMA_EXTENSION,
)


def test_game_conditioned_frontend_keeps_worker_compatibility_envelope():
    assert FRONTEND_SCHEMA_VERSION == "mlb_sequential_pa_frontend_v1"
    assert GAME_CONDITIONED_SCHEMA_EXTENSION == "mlb_game_conditioned_hitter_frontend_v2"
