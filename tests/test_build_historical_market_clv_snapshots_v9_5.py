import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path("Player-Predictor/scripts/build_historical_market_clv_snapshots_v9_5.py")
SPEC = importlib.util.spec_from_file_location("build_historical_market_clv_snapshots_v9_5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_snapshots_requires_two_sided_no_vig_and_marks_clv_status():
    rows = pd.DataFrame(
        {
            "date": ["2026-03-26", "2026-03-26", "2026-03-25"],
            "player": ["Player_One", "Player_One", "Player_Two"],
            "market": ["PTS", "PTS", "AST"],
            "line": [10.5, 11.5, 5.5],
            "over_odds": [-110, -120, -105],
            "under_odds": [-110, 100, -115],
            "snapshot_time": pd.to_datetime(
                ["2026-03-26T16:00:00Z", "2026-03-26T22:00:00Z", "2026-03-27T16:00:00Z"],
                utc=True,
            ),
        }
    )
    snapshots, report = MODULE.build_snapshots(rows)
    assert report["snapshot_rows"] == 2
    one = snapshots[snapshots["player"].eq("Player_One")].iloc[0]
    assert one["current_line"] == 10.5
    assert one["close_line"] == 11.5
    assert one["close_status"] == "same_day_latest_snapshot_proxy_commence_missing"
    two = snapshots[snapshots["player"].eq("Player_Two")].iloc[0]
    assert two["close_status"] == "archived_historical_market_not_clv"
    assert 0 < one["no_vig_over"] < 1
