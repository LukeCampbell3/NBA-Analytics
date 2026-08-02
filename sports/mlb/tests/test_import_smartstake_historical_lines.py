from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import import_smartstake_historical_lines as importer


@pytest.mark.parametrize(
    ("decimal", "american"),
    [(1.5, -200), (1.9091, -110), (2.0, 100), (2.25, 125)],
)
def test_decimal_to_american(decimal: float, american: int) -> None:
    assert importer.decimal_to_american(decimal) == american


def test_compaction_query_enforces_pregame_closing_quote_and_supported_markets() -> None:
    query = importer.build_compaction_sql(
        sources=["example.parquet"],
        start_date="2026-05-01",
        end_date="2026-05-31",
        books=["fanduel", "draftkings"],
    )

    assert "ts < start_time" in query
    assert "ORDER BY ts DESC" in query
    assert "quote_rank = 1" in query
    assert "batter_total_bases" in query
    assert "pitcher_strikeouts" in query
    assert "per_offer_closing" in query
