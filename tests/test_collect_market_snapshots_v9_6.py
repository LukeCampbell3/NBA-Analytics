import importlib.util
from pathlib import Path

import pandas as pd
import sys


SCRIPT = Path("Player-Predictor/scripts/collect_market_snapshots_v9_6.py")
sys.path.insert(0, str(SCRIPT.parent.resolve()))
SPEC = importlib.util.spec_from_file_location("collect_market_snapshots_v9_6", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_append_collection_deduplicates_snapshot_rows(tmp_path):
    path = tmp_path / "collected.csv"
    rows = pd.DataFrame(
        {
            "snapshot_time": ["2026-01-01T16:00:00Z", "2026-01-01T16:00:00Z"],
            "book": ["Book", "Book"],
            "game_id": ["1", "1"],
            "player_id": ["10", "10"],
            "player": ["Player_One", "Player_One"],
            "market": ["PTS", "PTS"],
            "line": [10.5, 10.5],
            "over_odds": [-110, -110],
            "under_odds": [-110, -110],
        }
    )
    combined, appended = MODULE.append_collection(path, rows)
    assert len(combined) == 1
    assert appended == 1

    combined, appended = MODULE.append_collection(path, rows)
    assert len(combined) == 1
    assert appended == 0
