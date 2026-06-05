from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FETCH_PATH = REPO_ROOT / "Player-Predictor" / "scripts" / "fetch_nba_market_snapshots.py"
ATTACH_PATH = REPO_ROOT / "Player-Predictor" / "scripts" / "attach_market_snapshots_v9_3.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


FETCH = _load_module("fetch_nba_market_snapshots", FETCH_PATH)
ATTACH = _load_module("attach_market_snapshots_v9_3", ATTACH_PATH)


def test_extract_rotowire_nba_bundles_and_schema_rows() -> None:
    html = """
    <html>
      <script>
        const dayNBA = "2026-05-08";
        const prop = "pts";
        const settings = {
          container: propID,
          data: [
            {
              "gameID":"100",
              "playerID":"200",
              "name":"Test Player",
              "team":"MIN",
              "opp":"SAS",
              "draftkings_pts":"22.5",
              "draftkings_ptsOver":"-120",
              "draftkings_ptsUnder":"100"
            }
          ]
        };
      </script>
      <script>
        const dayNBA = "2026-05-08";
        const prop = "reb";
        const settings = {
          container: propID,
          data: [
            {
              "gameID":"100",
              "playerID":"200",
              "name":"Test Player",
              "team":"MIN",
              "opp":"SAS",
              "fanduel_reb":"7.5",
              "fanduel_rebOver":"105",
              "fanduel_rebUnder":"-125"
            }
          ]
        };
      </script>
      <script>
        const dayNBA = "2026-05-08";
        const prop = "ast";
        const settings = {
          container: propID,
          data: [
            {
              "gameID":"100",
              "playerID":"200",
              "name":"Test Player",
              "team":"MIN",
              "opp":"SAS",
              "draftkings_ast":"4.5",
              "draftkings_astOver":"140",
              "draftkings_astUnder":"-170"
            }
          ]
        };
      </script>
    </html>
    """

    market_date, bundles = FETCH.extract_rotowire_bundles(html)
    assert market_date == "2026-05-08"
    assert sorted(bundles) == ["ast", "pts", "reb"]

    book_rows = FETCH.build_book_snapshots(market_date, bundles, "2026-05-09T01:02:03+00:00")
    canonical = FETCH.build_canonical_snapshots(book_rows)

    assert set(canonical["market"]) == {"PTS", "TRB", "AST"}
    assert set(FETCH.SUPPORTED_MARKETS.values()).issuperset(canonical["market"])
    assert canonical["close_status"].eq("provisional_current_snapshot_not_closing").all()
    assert canonical[ATTACH.REQUIRED_FIELDS].notna().all().all()

    pts = canonical.loc[canonical["market"] == "PTS"].iloc[0]
    assert pts["player"] == "Test_Player"
    assert pts["book"] == "DraftKings"
    assert pts["line"] == 22.5
    assert 0.0 < pts["no_vig_over"] < 1.0
    assert round(float(pts["no_vig_over"] + pts["no_vig_under"]), 10) == 1.0


def test_attach_snapshots_blocks_low_match_rate_and_preserves_neutral_fallback() -> None:
    rows = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "game_id": "hist_game",
                "player_id": "hist_player",
                "player": "Historical_Player",
                "market": "PTS",
                "line": 20.5,
                "over_odds": -110,
                "under_odds": -110,
                "market_no_vig_over": 0.5,
                "market_no_vig_under": 0.5,
            }
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_time": "2026-05-09T01:02:03+00:00",
                "date": "2026-05-08",
                "book": "DraftKings",
                "game_id": "future_game",
                "player_id": "future_player",
                "player": "Future_Player",
                "market": "PTS",
                "line": 22.5,
                "over_odds": -120,
                "under_odds": 100,
                "no_vig_over": 0.5238095238,
                "no_vig_under": 0.4761904762,
                "open_line": 22.5,
                "current_line": 22.5,
                "close_line": 22.5,
                "close_over_odds": -120,
                "close_under_odds": 100,
            }
        ]
    )

    merged, attachment = ATTACH.attach_snapshots(rows, snapshots)

    assert attachment["schema_validation"]["status"] == "pass"
    assert attachment["match_rate"] == 0.0
    assert attachment["matched_rows"] == 0
    assert merged.loc[0, "market_no_vig_over"] == 0.5
    assert merged.loc[0, "over_odds"] == -110


def test_attach_snapshots_matches_and_replaces_market_probability() -> None:
    rows = pd.DataFrame(
        [
            {
                "date": "2026-05-08",
                "game_id": "100",
                "player_id": "200",
                "player": "Test_Player",
                "market": "PTS",
                "line": 20.5,
                "over_odds": -110,
                "under_odds": -110,
                "market_no_vig_over": 0.5,
                "market_no_vig_under": 0.5,
            }
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "snapshot_time": "2026-05-09T01:02:03+00:00",
                "date": "2026-05-08",
                "book": "DraftKings",
                "game_id": 100,
                "player_id": 200,
                "player": "Test_Player",
                "market": "PTS",
                "line": 22.5,
                "over_odds": -120,
                "under_odds": 100,
                "no_vig_over": 0.5238095238,
                "no_vig_under": 0.4761904762,
                "open_line": 22.5,
                "current_line": 22.5,
                "close_line": 22.5,
                "close_over_odds": -120,
                "close_under_odds": 100,
            }
        ]
    )

    merged, attachment = ATTACH.attach_snapshots(rows, snapshots)

    assert attachment["match_rate"] == 1.0
    assert attachment["join_keys"] == ["game_id", "player_id", "market"]
    assert merged.loc[0, "line"] == 22.5
    assert merged.loc[0, "over_odds"] == -120
    assert merged.loc[0, "book"] == "DraftKings"
    assert round(float(merged.loc[0, "market_no_vig_over"]), 10) == 0.5238095238
