from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Player-Predictor" / "scripts" / "fetch_mlb_market_props.py"
SPEC = importlib.util.spec_from_file_location("fetch_mlb_market_props", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_rotowire_page_payload_and_build_frames() -> None:
    html = """
    <html>
      <body>
        <script>
          document.addEventListener('rwjs:ready', function(){
            const dayMLB = "2026-04-28";
            const settings = {
              container: 'moneyline-props',
              data: [
                {
                  "gameID":"11",
                  "name":"Aaron Judge",
                  "team":"NYY",
                  "opp":"@TEX",
                  "draftkings_onehit":"-225",
                  "fanduel_onehomerun":"310",
                  "mgm_onerbi":"150"
                }
              ]
            };
          });
        </script>
        <script>
          document.addEventListener('rwjs:ready', function(){
            const dayMLB = "2026-04-28";
            const prop = "strikeouts";
            const settings = {
              container: propID,
              data: [
                {
                  "gameID":"22",
                  "name":"Jacob deGrom",
                  "team":"TEX",
                  "opp":"NYY",
                  "draftkings_strikeouts":"6.5",
                  "draftkings_strikeoutsOver":"-150",
                  "draftkings_strikeoutsUnder":"120"
                }
              ]
            };
          });
        </script>
      </body>
    </html>
    """

    page_date, bundles = MODULE.extract_rotowire_page_payload(html)
    assert page_date == "2026-04-28"
    assert sorted(bundles) == ["moneyline", "strikeouts"]

    long_df, wide_df = MODULE.build_rotowire_frames(
        market_date=page_date,
        bundles=bundles,
        fetched_at_utc="2026-04-28T01:02:03+00:00",
    )

    assert not long_df.empty
    assert not wide_df.empty

    judge_row = wide_df.loc[wide_df["Player"] == "Aaron_Judge"].iloc[0]
    assert judge_row["Market_H"] == 0.5
    assert judge_row["Market_H_over_price"] == -225.0
    assert judge_row["Market_HR"] == 0.5
    assert judge_row["Market_HR_over_price"] == 310.0
    assert judge_row["Market_RBI"] == 0.5
    assert judge_row["Market_RBI_over_price"] == 150.0
    assert judge_row["Market_Home_Team"] == "TEX"
    assert judge_row["Market_Away_Team"] == "NYY"
    assert judge_row["Market_Player_Team"] == "NYY"
    assert judge_row["Market_Player_Opponent"] == "TEX"

    degrom_row = wide_df.loc[wide_df["Player"] == "Jacob_deGrom"].iloc[0]
    assert degrom_row["Market_K"] == 6.5
    assert degrom_row["Market_K_over_price"] == -150.0
    assert degrom_row["Market_K_under_price"] == 120.0
    assert degrom_row["Market_Home_Team"] == "TEX"
    assert degrom_row["Market_Away_Team"] == "NYY"
    assert degrom_row["Market_Player_Team"] == "TEX"
    assert degrom_row["Market_Player_Opponent"] == "NYY"


def test_consensus_american_price_averages_implied_probability() -> None:
    price = MODULE.consensus_american_price(pd.Series([-110, 110]))

    assert abs(price) >= 100.0
    assert abs(price) == 100.0


def test_provider_contract_derives_existing_selector_schema() -> None:
    base = {
        "source": "the_odds_api",
        "event_id": "event-1",
        "game_start_utc": "2026-08-04T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Texas Rangers",
        "sportsbook": "fanduel",
        "market_type": "batter_total_bases",
        "player_name": "Aaron Judge",
        "line": 1.5,
        "canonical_selected": True,
    }
    contract = pd.DataFrame(
        [
            {**base, "side": "over", "price_american": -115},
            {**base, "side": "under", "price_american": -105},
        ]
    )

    long_df, wide_df = MODULE.build_frames_from_contract(contract, "2026-08-04T15:00:00Z")

    assert len(long_df) == 1
    assert long_df.iloc[0]["over_price"] == -115
    assert long_df.iloc[0]["under_price"] == -105
    assert list(wide_df.columns) == MODULE.MARKET_WIDE_COLUMNS
    assert wide_df.iloc[0]["Market_TB"] == 1.5


def test_provider_contract_never_blends_prices_across_unlike_lines() -> None:
    rows = []
    for book, line, over_price in [
        ("fanduel", 1.5, -110),
        ("draftkings", 1.5, -120),
        ("bet365", 2.5, 250),
    ]:
        rows.append(
            {
                "source": "the_odds_api",
                "event_id": "event-1",
                "game_start_utc": "2026-08-04T23:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Texas Rangers",
                "sportsbook": book,
                "market_type": "batter_total_bases",
                "player_name": "Aaron Judge",
                "line": line,
                "side": "over",
                "price_american": over_price,
                "canonical_selected": True,
            }
        )

    _, wide_df = MODULE.build_frames_from_contract(pd.DataFrame(rows), "2026-08-04T15:00:00Z")

    expected_price = MODULE.consensus_american_price(pd.Series([-110, -120]))
    assert wide_df.iloc[0]["Market_TB"] == 1.5
    assert wide_df.iloc[0]["Market_TB_books"] == 2
    assert wide_df.iloc[0]["Market_TB_over_price"] == expected_price
