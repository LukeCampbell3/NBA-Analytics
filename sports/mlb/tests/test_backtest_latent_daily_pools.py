from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sports.mlb.scripts import backtest_latent_daily_pools as replay


def test_no_vig_quotes_requires_both_sides_at_the_same_book() -> None:
    universe = pd.DataFrame(
        [
            {"book": "fanduel", "player_name": "Jose Ramirez", "market": "H", "side": "OVER", "line": 0.5, "price": -150},
            {"book": "fanduel", "player_name": "Jose Ramirez", "market": "H", "side": "UNDER", "line": 0.5, "price": 120},
            {"book": "draftkings", "player_name": "Jose Ramirez", "market": "H", "side": "OVER", "line": 0.5, "price": -145},
        ]
    )

    quotes = replay.no_vig_quotes(universe)

    assert len(quotes) == 1
    assert quotes.iloc[0]["book"] == "fanduel"
    expected = replay.implied_probability(-150) / (
        replay.implied_probability(-150) + replay.implied_probability(120)
    )
    assert quotes.iloc[0]["market_probability"] == pytest.approx(expected)


def test_latest_complete_snapshots_uses_latest_immutable_capture(tmp_path: Path) -> None:
    run_root = tmp_path / "20260806" / "governance" / "slates" / "slate-1"
    for snapshot, observed in (("early", "2026-08-06T13:00:00+00:00"), ("late", "2026-08-06T14:00:00+00:00")):
        directory = run_root / snapshot
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"snapshot_id": snapshot, "observed_at_utc": observed}),
            encoding="utf-8",
        )
        pd.DataFrame([{"value": 1}]).to_csv(directory / "feature_pool.csv.gz", index=False)
        pd.DataFrame([{"value": 1}]).to_csv(directory / "candidate_universe.csv.gz", index=False)

    snapshots = replay.latest_complete_snapshots(tmp_path, start_date=date(2026, 8, 6))

    assert len(snapshots) == 1
    assert snapshots[0]["snapshot_id"] == "late"


def test_market_ticket_is_same_book_and_uses_distinct_games() -> None:
    rows = []
    for player, game_id, probability in (
        ("A", "g1", 0.70),
        ("B", "g1", 0.69),
        ("C", "g2", 0.68),
    ):
        rows.append(
            {
                "run_date": "2026-08-06",
                "snapshot_id": "snapshot",
                "book": "fanduel",
                "player": player,
                "player_key": player.lower(),
                "game_id": game_id,
                "market_probability": probability,
                "hybrid_probability": probability,
                "latent_probability": probability,
                "ensemble_std": 0.01,
                "decimal_price": 1.60,
                "win": 1,
                "_numeric": {},
                "_categorical": {},
            }
        )

    ticket = replay.choose_ticket(rows, leg_count=2, strategy="market", bundle=None)

    assert ticket is not None
    assert ticket["book"] == "fanduel"
    assert ticket["players"] == ["A", "C"]
    assert len(set(ticket["games"])) == 2
