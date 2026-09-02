import pandas as pd

from sports.nfl.scripts.backtest_nfl_week_market_policy import build_report


POLICY_REPORT = {
    "model_version": "test_policy",
    "evidence_as_of_utc": "2025-12-31T00:00:00Z",
    "selected_policy": {
        "minimum_side_probability": 0.58,
        "minimum_no_vig_advantage": 0.10,
        "minimum_price": -130,
        "maximum_price": 130,
        "weekly_cap": 6,
    },
}


def _pool(target: str, wins: int, losses: int, season: int) -> pd.DataFrame:
    results = ["win"] * wins + ["loss"] * losses
    return pd.DataFrame({
        "season": [season] * len(results),
        "week": [(index % 18) + 1 for index in range(len(results))],
        "target": [target] * len(results),
        "estimated_side_probability": [0.65] * len(results),
        "probability_advantage": [0.12] * len(results),
        "selected_price": [100] * len(results),
        "player_display_name": [f"Player {index}" for index in range(len(results))],
        "result": results,
        "profit_units": [1.0 if value == "win" else -1.0 for value in results],
        "side": ["under"] * len(results),
    })


def test_only_consistently_positive_supported_capability_receives_authority() -> None:
    pools = {}
    for label, season in (("stress_2021", 2021), ("stress_2022", 2022), ("confirmation_2025", 2025)):
        passing = _pool("passing", 40, 20, season)
        rushing = _pool("rushing", 20, 40, season)
        receiving = _pool("receiving", 20, 40, season)
        pools[label] = pd.concat([passing, rushing, receiving], ignore_index=True)
    report = build_report(POLICY_REPORT, pools)
    assert report["capabilities"]["passing"]["selection_authority"] is True
    assert report["capabilities"]["rushing"]["state"] == "NO_RELIABLE_EDGE_FOUND"
    assert report["capabilities"]["receiving"]["selection_authority"] is False
