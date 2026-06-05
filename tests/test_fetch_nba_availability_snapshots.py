from pathlib import Path

import pandas as pd

import importlib.util


SCRIPT = Path("Player-Predictor/scripts/fetch_nba_availability_snapshots.py")
SPEC = importlib.util.spec_from_file_location("fetch_nba_availability_snapshots", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_availability_maps_status_to_probabilities():
    raw = pd.DataFrame(
        {
            "Player": ["Player One", "Player Two", "Player Three"],
            "Team": ["lal", "BOS", "NYK"],
            "Status": ["Questionable - knee", "Out", "Probable"],
        }
    )
    normalized = MODULE.normalize_availability(
        raw,
        snapshot_time="2026-01-03T18:00:00Z",
        game_start_time="2026-01-04T00:00:00Z",
        source="test",
    )
    probs = dict(zip(normalized["player"], normalized["out_probability"]))
    assert probs["Player_One"] == 0.45
    assert probs["Player_Two"] == 1.0
    assert probs["Player_Three"] == 0.15
    assert normalized["availability_confidence"].between(0, 1).all()
