from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ODDS_DIR = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds"
PROVIDERS_DIR = ODDS_DIR / "providers"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mlb_odds" / "permitted_embedded_json.html"
sys.path.insert(0, str(ODDS_DIR))
sys.path.insert(0, str(PROVIDERS_DIR))

import provider_router
from odds_contract import CONTRACT_COLUMNS, ensure_contract, reconcile_observations, validate_contract
from permitted_scrape_mlb_provider import PermittedScrapeMlbProvider
from sportsgameodds_mlb_provider import SportsGameOddsMlbProvider
from the_odds_api_mlb_provider import TheOddsApiMlbProvider


NOW = datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc)


def observation(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "fixture",
        "source_market_id": "market-1",
        "sportsbook": "fanduel",
        "event_id": "event-1",
        "external_event_id": "external-event-1",
        "player_id": "aaronjudge",
        "external_player_id": "source-aaronjudge",
        "player_name": "Aaron Judge",
        "home_team": "New York Yankees",
        "away_team": "Texas Rangers",
        "game_start_utc": "2026-08-04T23:05:00Z",
        "league": "MLB",
        "market_type": "batter_total_bases",
        "side": "over",
        "line": 1.5,
        "price_american": -115,
        "observed_at_utc": "2026-08-04T14:55:00Z",
        "source_updated_at_utc": "2026-08-04T14:55:00Z",
        "source_url_or_endpoint": "https://example.test/mlb-odds",
        "acquisition_method": "fixture",
        "raw_record_hash": "a" * 64,
        "parser_version": "fixture-v1",
        "normalization_version": "mlb-odds-contract-v1",
        "validation_status": "UNVALIDATED",
    }
    row.update(overrides)
    return row


class HealthyScrape:
    def validate_config(self):
        return {"status": "ok"}

    def collect_player_props(self):
        return {"status": "success", "odds": [observation(source="scrape", acquisition_method="scrape")]}

    def normalize(self, rows):
        return ensure_contract(pd.DataFrame(rows), source="scrape", acquisition_method="scrape")


class HealthyApi:
    def validate_config(self):
        return {"status": "ok"}

    def collect_player_props(self):
        return {"status": "success", "odds": [observation(source="the_odds_api", acquisition_method="api")]}

    def normalize(self, rows):
        return ensure_contract(pd.DataFrame(rows), source="the_odds_api", acquisition_method="api")


class MissingCredentials:
    def validate_config(self):
        return {"status": "missing_credentials", "message": "key missing"}


class MissingSource:
    def validate_config(self):
        return {"status": "missing_source", "message": "source missing"}


class FailedPrimary:
    def validate_config(self):
        return {"status": "ok"}

    def collect_player_props(self):
        return {"status": "api_error", "message": "upstream unavailable"}


def isolated_router(monkeypatch, tmp_path, priority, classes):
    monkeypatch.setattr(provider_router, "SNAPSHOT_DIR", tmp_path / "empty-cache")
    return provider_router.MlbProviderRouter(
        provider_priority=priority,
        provider_classes=classes,
        now=NOW,
    )


def test_missing_sportsgameodds_key_does_not_block_healthy_scraper(monkeypatch, tmp_path) -> None:
    router = isolated_router(
        monkeypatch,
        tmp_path,
        ["scrape", "sportsgameodds"],
        {"scrape": HealthyScrape, "sportsgameodds": MissingCredentials},
    )
    frame, info = router.get_fresh_odds()
    assert frame is not None and not frame.empty
    assert info["terminal_status"] == "MLB_FRESH_ODDS_AVAILABLE"
    assert info["source_state"] == "MLB_ODDS_PRIMARY_HEALTHY"
    assert info["provider_results"][1]["provider_status"] == "missing_credentials"


def test_missing_scraper_source_does_not_block_healthy_api(monkeypatch, tmp_path) -> None:
    router = isolated_router(
        monkeypatch,
        tmp_path,
        ["scrape", "the_odds_api"],
        {"scrape": MissingSource, "the_odds_api": HealthyApi},
    )
    frame, info = router.get_fresh_odds()
    assert frame is not None and not frame.empty
    assert info["source_state"] == "MLB_ODDS_FALLBACK_ACTIVE"
    assert info["successful_provider"] == "the_odds_api"


def test_missing_credentials_for_only_enabled_api_are_reported(monkeypatch, tmp_path) -> None:
    router = isolated_router(monkeypatch, tmp_path, ["the_odds_api"], {"the_odds_api": MissingCredentials})
    frame, info = router.get_fresh_odds()
    assert frame is None
    assert info["terminal_status"] == "MLB_WAITING_FOR_FRESH_PROPS"
    assert info["provider_results"][0]["provider_status"] == "missing_credentials"


def test_primary_failure_activates_secondary(monkeypatch, tmp_path) -> None:
    router = isolated_router(
        monkeypatch,
        tmp_path,
        ["scrape", "the_odds_api"],
        {"scrape": FailedPrimary, "the_odds_api": HealthyApi},
    )
    _, info = router.get_fresh_odds()
    assert info["source_state"] == "MLB_ODDS_FALLBACK_ACTIVE"
    assert [row["provider_status"] for row in info["provider_results"][:2]] == ["api_error", "success"]


def test_loading_page_and_changed_html_trigger_schema_drift(tmp_path) -> None:
    for name, body in [("loading.html", "<html>Loading...</html>"), ("changed.html", "<html><div>markets</div></html>")]:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        result = PermittedScrapeMlbProvider(fixture_path=path).collect_player_props()
        assert result["status"] == "schema_drift"
        assert "MLB_ODDS_SOURCE_SCHEMA_DRIFT" in result["message"]


def test_scraper_rejects_conflicting_duplicate_side(tmp_path) -> None:
    first = observation(source="scrape", acquisition_method="scrape")
    second = {**first, "price_american": -105, "raw_record_hash": "b" * 64}
    payload = {"source_timestamp": "2026-08-04T14:55:00Z", "records": [first, second]}
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = PermittedScrapeMlbProvider(fixture_path=path).collect_player_props()

    assert result["status"] == "schema_drift"
    assert "conflicting prices" in result["message"]


def test_embedded_json_fixture_normalizes_and_replays_deterministically() -> None:
    provider = PermittedScrapeMlbProvider(fixture_path=FIXTURE)
    first = provider.collect_player_props()
    second = provider.collect_player_props()
    assert first["status"] == second["status"] == "success"
    assert first["odds"] == second["odds"]
    normalized = provider.normalize(first["odds"])
    assert list(normalized["side"]) == ["over", "under"]
    assert set(CONTRACT_COLUMNS).issubset(normalized.columns)
    assert first["evidence"]["records_accepted"] == 2


def test_the_odds_api_fixture_uses_same_contract() -> None:
    payload = {
        "id": "event-1",
        "commence_time": "2026-08-04T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Texas Rangers",
        "bookmakers": [
            {
                "key": "fanduel",
                "last_update": "2026-08-04T14:55:00Z",
                "markets": [
                    {
                        "key": "batter_total_bases",
                        "last_update": "2026-08-04T14:55:00Z",
                        "outcomes": [{"name": "Over", "description": "Aaron Judge", "price": -115, "point": 1.5}],
                    }
                ],
            }
        ],
    }
    provider = TheOddsApiMlbProvider(fixture_payloads=[payload])
    result = provider.collect_player_props()
    normalized = provider.normalize(result["odds"])
    assert result["status"] == "success"
    assert set(CONTRACT_COLUMNS).issubset(normalized.columns)
    assert normalized.iloc[0]["price_decimal"] > 1.0
    scrape_columns = set(PermittedScrapeMlbProvider(fixture_path=FIXTURE).normalize(
        PermittedScrapeMlbProvider(fixture_path=FIXTURE).collect_player_props()["odds"]
    ).columns)
    assert set(normalized.columns) == scrape_columns


def test_sportsgameodds_requests_and_retains_book_specific_alternate_lines(monkeypatch) -> None:
    event = {
        "eventID": "event-1",
        "status": {"startsAt": "2026-08-04T23:05:00Z", "live": False},
        "teams": {
            "home": {"teamID": "NYY", "names": {"short": "NYY"}},
            "away": {"teamID": "TEX", "names": {"short": "TEX"}},
        },
        "players": {
            "AARON_JUDGE_1_MLB": {"name": "Aaron Judge", "teamID": "NYY"},
        },
        "odds": {
            "batting_hits-AARON_JUDGE_1_MLB-game-ou-over": {
                "betTypeID": "ou",
                "sideID": "over",
                "statID": "batting_hits",
                "playerID": "AARON_JUDGE_1_MLB",
                "bookOverUnder": "0.5",
                "byBookmaker": {
                    "draftkings": {
                        "available": True,
                        "odds": "-210",
                        "overUnder": "0.5",
                        "deeplink": "https://sportsbook.draftkings.com/leagues/baseball/mlb",
                        "altLines": [
                            {
                                "available": True,
                                "odds": "+220",
                                "overUnder": "1.5",
                                "deeplink": "https://sportsbook.draftkings.com/leagues/baseball/mlb",
                            },
                            {"available": False, "odds": "+500", "overUnder": "2.5"},
                        ],
                    },
                    "fanduel": {
                        "available": True,
                        "odds": "-195",
                        "overUnder": "0.5",
                        "deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=42.100&selectionId=1001",
                        "altLines": [{
                            "available": True,
                            "odds": "+235",
                            "overUnder": "1.5",
                            "deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=42.101&selectionId=1002",
                        }],
                    },
                },
            },
        },
    }
    requested_params = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"success": True, "data": [event]}

    def fake_get(_url, *, headers, params, timeout):
        assert headers["x-api-key"] == "fixture-key"
        assert timeout == 30
        requested_params.update(params)
        return Response()

    monkeypatch.setattr("sportsgameodds_mlb_provider.requests.get", fake_get)
    monkeypatch.setenv("MLB_ENABLE_LIVE_API_TESTS", "1")
    provider = SportsGameOddsMlbProvider(api_key="fixture-key")
    result = provider.collect_player_props()
    normalized = provider.normalize(result["odds"])

    assert requested_params["includeAltLines"] == "true"
    assert result["diagnostic_odds"] == []
    assert len(normalized) == 4
    assert set(normalized["line"]) == {0.5, 1.5}
    assert set(normalized.loc[normalized["line"].eq(1.5), "price_american"]) == {220.0, 235.0}
    fanduel = normalized.loc[normalized["sportsbook"].eq("fanduel")].sort_values("line")
    assert list(fanduel["sportsbook_deeplink"]) == [
        "https://sportsbook.fanduel.com/addToBetslip?marketId=42.100&selectionId=1001",
        "https://sportsbook.fanduel.com/addToBetslip?marketId=42.101&selectionId=1002",
    ]
    assert 2.5 not in set(normalized["line"])
    assert provider.get_accounting()["alternate_book_rows"] == 2


def test_duplicate_records_are_deduplicated_without_losing_sources() -> None:
    rows = [
        observation(source="scrape", raw_record_hash="a" * 64),
        observation(source="scrape", raw_record_hash="a" * 64),
        observation(source="the_odds_api", raw_record_hash="b" * 64),
    ]
    reconciled = reconcile_observations(pd.DataFrame(rows))
    assert len(reconciled) == 2
    assert set(reconciled["source"]) == {"scrape", "the_odds_api"}
    assert reconciled["source_count"].eq(2).all()


def test_unlike_lines_are_not_merged_and_disagreement_is_recorded() -> None:
    rows = [
        observation(source="scrape", line=1.5, price_american=-115),
        observation(source="the_odds_api", line=2.5, price_american=105, raw_record_hash="b" * 64),
    ]
    reconciled = reconcile_observations(pd.DataFrame(rows))
    assert len(reconciled) == 2
    assert set(reconciled["line"]) == {1.5, 2.5}
    assert reconciled["line_disagreement"].all()
    assert reconciled["source_count"].eq(1).all()


def test_cross_source_price_disagreement_selects_freshest() -> None:
    rows = [
        observation(source="scrape", observed_at_utc="2026-08-04T14:50:00Z", source_updated_at_utc="2026-08-04T14:50:00Z"),
        observation(
            source="the_odds_api", price_american=-105, raw_record_hash="b" * 64,
            observed_at_utc="2026-08-04T14:58:00Z", source_updated_at_utc="2026-08-04T14:58:00Z",
        ),
    ]
    reconciled = reconcile_observations(pd.DataFrame(rows))
    assert reconciled["price_disagreement"].all()
    selected = reconciled.loc[reconciled["canonical_selected"]].iloc[0]
    assert selected["source"] == "the_odds_api"
    assert selected["price_american"] == -105


def test_stale_scraped_and_api_odds_are_rejected_identically() -> None:
    for source, method in [("scrape", "scrape"), ("the_odds_api", "api")]:
        row = observation(
            source=source,
            acquisition_method=method,
            observed_at_utc="2026-08-04T12:00:00Z",
            source_updated_at_utc="2026-08-04T12:00:00Z",
        )
        valid, report = validate_contract(pd.DataFrame([row]), max_age_seconds=3600, now=NOW)
        assert valid.empty
        assert report["rejection_reasons"]["STALE_ODDS"] == 1


def test_price_format_conflict_and_unsupported_market_fail_validation() -> None:
    row = observation(price_decimal=9.99, market_type="unsupported_market")
    valid, report = validate_contract(pd.DataFrame([row]), max_age_seconds=3600, now=NOW)
    assert valid.empty
    assert report["rejection_reasons"]["PRICE_FORMAT_CONFLICT"] == 1
    assert report["rejection_reasons"]["UNSUPPORTED_MARKET"] == 1


def test_started_event_is_rejected_and_staking_has_no_fresh_source(monkeypatch, tmp_path) -> None:
    class StartedApi(HealthyApi):
        def collect_player_props(self):
            return {"status": "success", "odds": [observation(game_start_utc="2026-08-04T14:00:00Z")]}

    router = isolated_router(monkeypatch, tmp_path, ["the_odds_api"], {"the_odds_api": StartedApi})
    frame, info = router.get_fresh_odds()
    assert frame is None
    assert info["no_fresh_odds_available"] is True
    assert info["terminal_status"] == "MLB_WAITING_FOR_FRESH_PROPS"
    assert info["provider_results"][0]["validation"]["rejection_reasons"]["EVENT_STARTED"] == 1
