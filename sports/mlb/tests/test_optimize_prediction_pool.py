from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import optimize_prediction_pool as optimizer


def test_build_price_lookup_uses_real_single_book_price_not_averaged_consensus(tmp_path: Path) -> None:
    """build_price_lookup() must read the real, per-book, per-timestamp
    long-format archive and pick one real executable price (with a real
    book identity) -- never the book-blind averaged consensus price the
    old wide-file source produced."""
    rows = [
        {
            "event_date_et": "2026-07-29",
            "player_name_norm": "Example_Player",
            "market_key": "batter_runs_scored",
            "bookmaker_key": "draftkings",
            "line": 0.5,
            "over_price": 125,
            "under_price": -155,
            "fetched_at_utc": "2026-07-29T18:00:00Z",
        },
        {
            "event_date_et": "2026-07-29",
            "player_name_norm": "Example_Player",
            "market_key": "batter_runs_scored",
            "bookmaker_key": "fanduel",
            "line": 0.5,
            "over_price": 130,
            "under_price": -160,
            "fetched_at_utc": "2026-07-29T18:00:00Z",
        },
    ]
    long_path = tmp_path / "history_player_props_long.csv"
    pd.DataFrame(rows).to_csv(long_path, index=False)

    price_lookup = optimizer.build_price_lookup(long_path)
    price_row = price_lookup[("2026-07-29", "Example_Player")]

    assert price_row["Market_R_over_price"] == 130
    assert price_row["Market_R_over_book_key"] == "fanduel"
    assert price_row["Market_R_over_price_time"] == "2026-07-29T18:00:00Z"
    assert price_row["Market_R_under_price"] == -155
    assert price_row["Market_R_under_book_key"] == "draftkings"


def test_build_price_lookup_missing_file_is_empty(tmp_path: Path) -> None:
    assert optimizer.build_price_lookup(tmp_path / "nonexistent.csv") == {}


def _write_processed_player_file(root: Path, season: int) -> None:
    frame = pd.DataFrame(
        [
            {
                "Date": "2026-07-28",
                "Game_Index": 1,
                "Player_Type": "hitter",
                "Player": "Example_Player",
                "Team": "NYY",
                "Opponent": "BOS",
                "Is_Home": 1,
                "Commence_Time_UTC": "2026-07-28T23:00:00Z",
                "Game_ID": "1001",
                "R": 1.0,
                "Market_R": 0.5,
                "Market_Source_R": "real",
                "R_market_gap": 0.0,
                "R_rolling_avg": 0.5,
                "R_lag1": 0.5,
            },
            {
                "Date": "2026-07-29",
                "Game_Index": 2,
                "Player_Type": "hitter",
                "Player": "Example_Player",
                "Team": "NYY",
                "Opponent": "TB",
                "Is_Home": 0,
                "Commence_Time_UTC": "2026-07-29T23:00:00Z",
                "Game_ID": "1002",
                "R": 1.0,
                "Market_R": 0.5,
                "Market_Source_R": "real",
                "R_market_gap": 0.1,
                "R_rolling_avg": 0.5,
                "R_lag1": 1.0,
            },
        ]
    )
    player_dir = root / "Example_Player"
    player_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(player_dir / f"{season}_processed_processed.csv", index=False)


def test_build_historical_universe_carries_real_book_key_and_price_time(tmp_path: Path) -> None:
    """The rebuilt universe must carry the same real book identity and
    decision-time timestamp the live board publishes, not just a bare
    price -- this is what unblocks build_candidate()'s price_confirmed
    check for backtesting/optimization."""
    data_dir = tmp_path / "processed"
    _write_processed_player_file(data_dir, season=2026)

    long_rows = [
        {
            "event_date_et": "2026-07-29",
            "player_name_norm": "Example_Player",
            "market_key": "batter_runs_scored",
            "bookmaker_key": "draftkings",
            "line": 0.5,
            "over_price": 125,
            "under_price": -155,
            "fetched_at_utc": "2026-07-29T18:00:00Z",
        },
    ]
    long_path = tmp_path / "history_player_props_long.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    price_lookup = optimizer.build_price_lookup(long_path)

    universe = optimizer.build_historical_universe(
        season=2026,
        data_dir=data_dir,
        manifest=tmp_path / "no_manifest.json",
        sample_cache=tmp_path / "universe.csv",
        refresh_sample_cache=True,
        min_modeled_history_rows=1,
        price_lookup=price_lookup,
    )

    row = universe.loc[universe["Target"] == "R"].iloc[0]
    assert row["Market_Over_Price"] == 125
    assert row["Market_Over_Book_Key"] == "draftkings"
    assert row["Market_Over_Price_Time"] == "2026-07-29T18:00:00Z"
    assert row["Market_Under_Price"] == -155
    assert row["Market_Under_Book_Key"] == "draftkings"


def test_build_historical_universe_leaves_book_key_blank_when_unpriced(tmp_path: Path) -> None:
    """A day/player with no real price archive entry must stay explicitly
    unconfirmable (blank book key), never fall back to a guess."""
    data_dir = tmp_path / "processed"
    _write_processed_player_file(data_dir, season=2026)

    universe = optimizer.build_historical_universe(
        season=2026,
        data_dir=data_dir,
        manifest=tmp_path / "no_manifest.json",
        sample_cache=tmp_path / "universe.csv",
        refresh_sample_cache=True,
        min_modeled_history_rows=1,
        price_lookup={},
    )

    row = universe.loc[universe["Target"] == "R"].iloc[0]
    assert row["Market_Over_Book_Key"] == ""
    assert row["Market_Under_Book_Key"] == ""
