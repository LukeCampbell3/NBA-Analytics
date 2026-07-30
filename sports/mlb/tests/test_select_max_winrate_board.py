from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import select_max_winrate_board as selector


def _row(player: str, under_price: object) -> dict[str, object]:
    return {
        "Prediction_Run_Date": "2026-07-29",
        "Game_Date": "2026-07-29",
        "Game_ID": f"game-{player}",
        "Player": player,
        "Team": "AAA",
        "Opponent": "BBB",
        "Target": "TB",
        "Prediction": 0.5,
        "Market_Line": 2.5,
        "Market_Source": "real",
        "Market_Books": 8,
        "Market_Line_Std": 0.1,
        "Market_Over_Price": 130,
        "Market_Under_Price": under_price,
        "Edge": -2.0,
        "History_Rows": 75,
        "Days_Since_History": 1,
    }


def test_selector_requires_valid_price_for_recommended_side(tmp_path: Path) -> None:
    pool_csv = tmp_path / "pool.csv"
    pd.DataFrame(
        [
            _row("Priced Under", -180),
            _row("Missing Under", float("nan")),
            _row("Invalid Under", -1.8),
        ]
    ).to_csv(pool_csv, index=False)

    board = selector.select_max_winrate_board(pool_csv)

    assert [candidate.player for candidate in board] == ["Priced Under"]
    assert board[0].selected_side_price == -180.0


def test_export_marks_selected_side_price_confirmed(tmp_path: Path) -> None:
    pool_csv = tmp_path / "pool.csv"
    out_csv = tmp_path / "board.csv"
    pd.DataFrame([_row("Priced Under", -180)]).to_csv(pool_csv, index=False)
    board = selector.select_max_winrate_board(pool_csv)

    selector.write_exporter_csv(out_csv, board)

    with out_csv.open(encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert exported[0]["Market_Under_Price"] == "-180.000000"
    assert exported[0]["Price_Confirmed"] == "1"
