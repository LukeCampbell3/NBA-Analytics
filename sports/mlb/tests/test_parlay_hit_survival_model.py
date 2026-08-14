from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from sports.mlb.scripts.parlay_hit_survival_model import build_training_rows


def test_training_rows_shift_rolling_features_but_keep_pregame_batting_order(tmp_path: Path) -> None:
    player_dir = tmp_path / "Test_Player"
    player_dir.mkdir()
    start = date(2026, 3, 1)
    rows = []
    for index in range(38):
        rows.append(
            {
                "Date": (start + timedelta(days=index)).isoformat(),
                "Player": "Test Player",
                "Game_ID": f"g{index}",
                "H": float(index % 3),
                "Market_H": 0.5,
                "H_market_gap": 0.7,
                "H_rolling_avg": 0.2 + (index / 100.0),
                "Batting_Order": 2 + (index % 4),
                "Is_Home": index % 2,
                "Did_Not_Play": 0,
            }
        )
    pd.DataFrame(rows).to_csv(player_dir / "2026_processed_processed.csv", index=False)

    frame, context = build_training_rows(tmp_path, before_date=date(2026, 5, 1))

    assert len(frame) == 3
    first = frame.iloc[0]
    assert first["history_rows"] == 35
    assert first["baseline"] == rows[34]["H_rolling_avg"]
    assert first["last_hits"] == rows[34]["H"]
    assert first["batting_order"] == rows[35]["Batting_Order"]
    assert context["testplayer"]["last_hits"] == rows[-1]["H"]
