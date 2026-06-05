import importlib.util
from pathlib import Path

import pandas as pd
import sys


SCRIPT = Path("Player-Predictor/scripts/build_market_snapshot_sequence_v9_6.py")
sys.path.insert(0, str(SCRIPT.parent.resolve()))
SPEC = importlib.util.spec_from_file_location("build_market_snapshot_sequence_v9_6", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sequence_builder_labels_open_prelock_close_and_filters_bad_odds():
    rows = pd.DataFrame(
        {
            "date": ["2026-01-01"] * 4,
            "book": ["Book"] * 4,
            "player": ["Player_One"] * 4,
            "market": ["PTS"] * 4,
            "line": [10.5, 11.5, 12.5, 13.5],
            "over_odds": [-110, -120, -11, -105],
            "under_odds": [-110, 100, -120, -115],
            "snapshot_time": [
                "2026-01-01T16:00:00Z",
                "2026-01-01T20:00:00Z",
                "2026-01-01T21:00:00Z",
                "2026-01-01T23:00:00Z",
            ],
            "game_start_time": ["2026-01-01T22:00:00Z"] * 4,
        }
    )
    normalized = MODULE._normalize(rows)
    valid = normalized[normalized["is_valid_american_odds"]].copy()
    sequenced = MODULE._label_snapshot_types(valid)
    assert len(valid) == 3
    assert set(sequenced["snapshot_type"]) == {"open", "prelock", "close"}
    attachable = MODULE._derive_current_close(sequenced)
    assert attachable.iloc[0]["current_line"] == 11.5
    assert attachable.iloc[0]["close_line"] == 13.5
    assert attachable.iloc[0]["close_status"] == "true_sequence_close"


def test_sequence_builder_marks_single_snapshot_as_not_clv():
    rows = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "book": ["Book"],
            "player": ["Player_One"],
            "market": ["PTS"],
            "line": [10.5],
            "over_odds": [-110],
            "under_odds": [-110],
            "snapshot_time": ["2026-01-01T16:00:00Z"],
        }
    )
    normalized = MODULE._normalize(rows)
    sequenced = MODULE._label_snapshot_types(normalized[normalized["is_valid_american_odds"]].copy())
    assert sequenced.iloc[0]["snapshot_type"] == "single_snapshot"
    attachable = MODULE._derive_current_close(sequenced)
    assert attachable.iloc[0]["close_status"] == "provisional_single_snapshot_not_clv"


def test_sequence_builder_blocks_clv_without_game_start_time():
    rows = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01"],
            "book": ["Book", "Book"],
            "player": ["Player_One", "Player_One"],
            "market": ["PTS", "PTS"],
            "line": [10.5, 11.5],
            "over_odds": [-110, -115],
            "under_odds": [-110, -105],
            "snapshot_time": ["2026-01-01T16:00:00Z", "2026-01-01T20:00:00Z"],
        }
    )
    normalized = MODULE._normalize(rows)
    sequenced = MODULE._label_snapshot_types(normalized[normalized["is_valid_american_odds"]].copy())
    assert set(sequenced["snapshot_type"]) == {"open", "close"}
    attachable = MODULE._derive_current_close(sequenced)
    assert attachable.iloc[0]["close_status"] == "sequence_close_game_start_missing"
