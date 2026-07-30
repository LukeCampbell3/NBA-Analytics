from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from fetch_nba_market_props import build_rotowire_frames
from fetch_nba_market_props import extract_rotowire_page_payload


def _script(prop: str, rows: str) -> str:
    return f"""
    <script>
      document.addEventListener('rwjs:ready', function(){{
        const dayNBA = "2026-04-28";
        const prop = "{prop}";
        const settings = {{
          container: propID,
          data: {rows}
        }};
      }});
    </script>
    """


def test_extract_rotowire_nba_props_and_build_consensus_frames() -> None:
    player = """
      {
        "gameID": "game-1",
        "name": "Nikola Jokic",
        "team": "DEN",
        "opp": "@MIN",
        "draftkings_%s": "%s",
        "draftkings_%sOver": "-115",
        "draftkings_%sUnder": "-105",
        "fanduel_%s": "%s",
        "fanduel_%sOver": "-110",
        "fanduel_%sUnder": "-110"
      }
    """
    html = (
        "<html><body>"
        + _script("pts", "[" + player % ("pts", "28.5", "pts", "pts", "pts", "28.5", "pts", "pts") + "]")
        + _script("reb", "[" + player % ("reb", "12.5", "reb", "reb", "reb", "12.5", "reb", "reb") + "]")
        + _script("ast", "[" + player % ("ast", "9.5", "ast", "ast", "ast", "9.5", "ast", "ast") + "]")
        + "</body></html>"
    )

    page_date, bundles = extract_rotowire_page_payload(html)
    long_df, wide_df = build_rotowire_frames(
        market_date=page_date,
        bundles=bundles,
        fetched_at_utc="2026-04-28T14:00:00+00:00",
    )

    assert page_date == "2026-04-28"
    assert sorted(bundles) == ["ast", "pts", "reb"]
    assert len(long_df) == 6
    assert len(wide_df) == 1

    row = wide_df.iloc[0]
    assert row["Player"] == "Nikola_Jokic"
    assert row["Market_Home_Team"] == "MIN"
    assert row["Market_Away_Team"] == "DEN"
    assert row["Market_PTS"] == 28.5
    assert row["Market_TRB"] == 12.5
    assert row["Market_AST"] == 9.5
    assert row["Market_PTS_books"] == 2
    assert row["Market_PTS_over_price"] == pytest.approx(-112.5)
    assert row["Market_Provider"] == "rotowire"
    assert row["Market_Book"] == "rotowire_consensus"
    assert row["Market_Price_Source"] == "rotowire_embedded_multi_book"
    assert row["Market_Price_Source_Type"] == "LIVE_ENTRY"


def test_extract_rotowire_nba_props_rejects_inactive_page() -> None:
    with pytest.raises(RuntimeError, match="may not have an active slate"):
        extract_rotowire_page_payload("<html><body>There are no odds available right now.</body></html>")
