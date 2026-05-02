from __future__ import annotations

import importlib.util
from pathlib import Path


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

    degrom_row = wide_df.loc[wide_df["Player"] == "Jacob_deGrom"].iloc[0]
    assert degrom_row["Market_K"] == 6.5
    assert degrom_row["Market_K_over_price"] == -150.0
    assert degrom_row["Market_K_under_price"] == 120.0
    assert degrom_row["Market_Home_Team"] == "TEX"
    assert degrom_row["Market_Away_Team"] == "NYY"
