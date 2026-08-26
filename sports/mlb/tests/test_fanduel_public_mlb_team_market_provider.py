from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"))

from fanduel_public_mlb_team_market_provider import FanduelPublicMlbTeamMarketProvider  # noqa: E402

NOW = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)


def _content_payload(event_id: int = 35973141, hours_out: float = 5.0) -> dict:
    open_date = (NOW + timedelta(hours=hours_out)).isoformat().replace("+00:00", "Z")
    return {
        "attachments": {
            "events": {
                str(event_id): {
                    "eventId": event_id, "eventTypeId": 7511,
                    "name": "Chicago Cubs (K Gausman) @ Arizona Diamondbacks (M Kelly)",
                    "openDate": open_date, "inPlay": False,
                }
            }
        }
    }


def _runner(selection_id: int, *, status="ACTIVE", side=None, handicap=None, price=None) -> dict:
    runner = {"selectionId": selection_id, "runnerStatus": status}
    if side is not None:
        runner["result"] = {"type": side}
    if handicap is not None:
        runner["handicap"] = handicap
    if price is not None:
        runner["winRunnerOdds"] = {"americanDisplayOdds": {"americanOddsInt": price}}
    return runner


def _event_payload(event_id: int = 35973141, *, moneyline_status="OPEN", total_status="OPEN") -> dict:
    return {
        "attachments": {
            "events": {str(event_id): {"eventId": event_id, "inPlay": False, "name": "Chicago Cubs (K Gausman) @ Arizona Diamondbacks (M Kelly)"}},
            "markets": {
                "m1": {
                    "marketId": "m1", "eventId": event_id, "marketType": "MONEY_LINE", "marketStatus": moneyline_status, "inPlay": False,
                    "runners": [
                        _runner(1, side="AWAY", price=130),
                        _runner(2, side="HOME", price=-150),
                    ],
                },
                "m2": {
                    "marketId": "m2", "eventId": event_id, "marketType": "TOTAL_POINTS_(OVER/UNDER)", "marketStatus": total_status, "inPlay": False,
                    "runners": [
                        _runner(3, side="OVER", handicap=8.5, price=-110),
                        _runner(4, side="UNDER", handicap=8.5, price=-110),
                    ],
                },
                "m3": {
                    # A real per-inning market this provider must ignore.
                    "marketId": "m3", "eventId": event_id, "marketType": "5TH_INNING_TOTAL_RUNS", "marketStatus": "OPEN", "inPlay": False,
                    "runners": [_runner(5, side="OVER", handicap=1.5, price=100)],
                },
            },
        }
    }


def _provider(**kwargs) -> FanduelPublicMlbTeamMarketProvider:
    return FanduelPublicMlbTeamMarketProvider(
        content_payload=_content_payload(), event_payloads={"35973141": _event_payload()}, now=NOW, sleep_fn=lambda _: None, **kwargs
    )


def test_disabled_reports_disabled_status() -> None:
    import os
    os.environ["MLB_FANDUEL_PUBLIC_ENABLED"] = "0"
    try:
        provider = _provider()
        assert provider.collect_team_market_odds()["status"] == "disabled"
    finally:
        del os.environ["MLB_FANDUEL_PUBLIC_ENABLED"]


def test_collects_real_moneyline_and_game_total_rows() -> None:
    provider = _provider()
    result = provider.collect_team_market_odds()
    assert result["status"] == "success"
    targets = {row["target"] for row in result["odds"]}
    assert targets == {"moneyline", "game_total"}


def test_moneyline_row_has_real_home_and_away_prices() -> None:
    provider = _provider()
    result = provider.collect_team_market_odds()
    ml_row = next(row for row in result["odds"] if row["target"] == "moneyline")
    assert ml_row["home_moneyline"] == -150
    assert ml_row["away_moneyline"] == 130
    assert ml_row["home_team"] == "Arizona Diamondbacks"
    assert ml_row["away_team"] == "Chicago Cubs"
    assert ml_row["sportsbook"] == "fanduel"


def test_game_total_row_has_real_line_and_prices() -> None:
    provider = _provider()
    result = provider.collect_team_market_odds()
    total_row = next(row for row in result["odds"] if row["target"] == "game_total")
    assert total_row["line"] == 8.5
    assert total_row["over_price"] == -110
    assert total_row["under_price"] == -110


def test_moneyline_row_has_real_per_selection_deeplinks() -> None:
    provider = _provider()
    result = provider.collect_team_market_odds()
    ml_row = next(row for row in result["odds"] if row["target"] == "moneyline")
    assert ml_row["home_moneyline_deeplink"] == "https://sportsbook.fanduel.com/addToBetslip?marketId=m1&selectionId=2"
    assert ml_row["away_moneyline_deeplink"] == "https://sportsbook.fanduel.com/addToBetslip?marketId=m1&selectionId=1"


def test_game_total_row_has_real_per_selection_deeplinks() -> None:
    provider = _provider()
    result = provider.collect_team_market_odds()
    total_row = next(row for row in result["odds"] if row["target"] == "game_total")
    assert total_row["over_deeplink"] == "https://sportsbook.fanduel.com/addToBetslip?marketId=m2&selectionId=3"
    assert total_row["under_deeplink"] == "https://sportsbook.fanduel.com/addToBetslip?marketId=m2&selectionId=4"


def test_ignores_real_per_inning_markets_not_in_the_team_market_set() -> None:
    provider = _provider()
    result = provider.collect_team_market_odds()
    assert len(result["odds"]) == 2  # moneyline + game_total only, never the 5th-inning market


def test_skips_a_real_suspended_market() -> None:
    provider = FanduelPublicMlbTeamMarketProvider(
        content_payload=_content_payload(),
        event_payloads={"35973141": _event_payload(moneyline_status="SUSPENDED")},
        now=NOW, sleep_fn=lambda _: None,
    )
    result = provider.collect_team_market_odds()
    assert result["status"] == "success"
    targets = {row["target"] for row in result["odds"]}
    assert targets == {"game_total"}  # moneyline suspended -> dropped, not fabricated


def test_no_events_reports_no_props() -> None:
    provider = FanduelPublicMlbTeamMarketProvider(content_payload={"attachments": {"events": {}}}, now=NOW, sleep_fn=lambda _: None)
    result = provider.collect_team_market_odds()
    assert result["status"] == "no_props"


def test_filters_out_a_real_far_future_placeholder_event() -> None:
    """A real 'MLB Player Markets' futures placeholder (no ' @ ' in the
    name) must never be treated as a real scheduled game."""
    provider = FanduelPublicMlbTeamMarketProvider(
        content_payload={
            "attachments": {
                "events": {
                    "999": {"eventId": 999, "eventTypeId": 7511, "name": "MLB Player Markets", "openDate": "2099-01-01T00:00:00.000Z"}
                }
            }
        },
        now=NOW, sleep_fn=lambda _: None,
    )
    result = provider.collect_team_market_odds()
    assert result["status"] == "no_props"
