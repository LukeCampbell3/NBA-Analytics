from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.build_price_defense_shadow_boards import build_price_defense_shadow_boards
from research.market_quality.compare_price_defense_boards import compare_price_defense_boards


def _base_candidate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "player": "Player A",
        "market_player_raw": "Player A",
        "player_name": "Player A",
        "market_date": "2026-05-26",
        "target": "PTS",
        "direction": "OVER",
        "side": "OVER",
        "market_type": "PTS_OVER",
        "line": 20.5,
        "market_line": 20.5,
        "market_side_price": -110.0,
        "over_price": -110.0,
        "under_price": 110.0,
        "price_source": "current_market_snapshot_pre_event",
        "price_source_type": "ARCHIVED_ENTRY",
        "odds_snapshot_time": "2026-05-26T18:00:00+00:00",
        "market_commence_time_utc": "2026-05-27T00:30:00+00:00",
        "expected_win_rate": 0.58,
        "edge": 0.08,
        "abs_edge": 0.08,
        "final_confidence": 0.05,
        "thompson_ev": 0.12,
        "ev_adjusted": 0.10,
        "market_books": 5,
        "history_rows": 40,
        "candidate_id": "candidate::player-a-pts-over",
    }
    row.update(overrides)
    return row


def test_build_price_defense_shadow_boards_creates_shadow_board_and_source_report(tmp_path: Path) -> None:
    candidate_pool_csv = tmp_path / "candidate_pool.csv"
    production_board_csv = tmp_path / "production_board.csv"
    output_dir = tmp_path / "price_defense_shadow"

    candidates = pd.DataFrame(
        [
            _base_candidate_row(),
            _base_candidate_row(
                player="Player B",
                market_player_raw="Player B",
                candidate_id="candidate::player-b-pts-over",
                edge=0.07,
                abs_edge=0.07,
                expected_win_rate=0.57,
                market_books=6,
                history_rows=50,
            ),
        ]
    )
    production_board = pd.DataFrame([_base_candidate_row()])

    candidates.to_csv(candidate_pool_csv, index=False)
    production_board.to_csv(production_board_csv, index=False)

    report = build_price_defense_shadow_boards(
        output_dir=output_dir,
        candidate_pool_csv=candidate_pool_csv,
        production_board_csv=production_board_csv,
        append_max_extra_plays=1,
    )

    assert output_dir.joinpath("price_defense_shadow_board.csv").exists()
    assert output_dir.joinpath("source_consistency_report.json").exists()
    assert output_dir.joinpath("price_defense_shadow_report.json").exists()
    assert report["total_candidate_rows"] == 2
    assert report["total_production_rows"] == 1
    assert report["shadow_board_rows"] >= 1
    assert report["production_behavior_changed"] is False

    source_report = json.loads(output_dir.joinpath("source_consistency_report.json").read_text(encoding="utf-8"))
    assert source_report["production_rows_missing_from_candidate_pool"] == 0


def test_compare_price_defense_boards_reports_differences(tmp_path: Path) -> None:
    production_board_csv = tmp_path / "production_board.csv"
    shadow_board_csv = tmp_path / "shadow_board.csv"
    output_dir = tmp_path / "comparison"

    production_board = pd.DataFrame(
        [
            _base_candidate_row(),
        ]
    )
    shadow_board = pd.DataFrame(
        [
            _base_candidate_row(),
            _base_candidate_row(
                player="Player B",
                market_player_raw="Player B",
                candidate_id="candidate::player-b-pts-over",
                edge=0.07,
                abs_edge=0.07,
                expected_win_rate=0.57,
                market_books=6,
                history_rows=50,
                append_shadow_added=True,
            ),
        ]
    )

    production_board.to_csv(production_board_csv, index=False)
    shadow_board.to_csv(shadow_board_csv, index=False)

    report = compare_price_defense_boards(
        output_dir=output_dir,
        production_board_csv=production_board_csv,
        shadow_board_csv=shadow_board_csv,
    )

    assert report["production_rows"] == 1
    assert report["shadow_rows"] == 2
    assert report["common_rows"] == 1
    assert report["production_only_rows"] == 0
    assert report["shadow_only_rows"] == 1
    assert report["shadow_added_rows"] == 1
    assert report["production_behavior_changed"] is False
